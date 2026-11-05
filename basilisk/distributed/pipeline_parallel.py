# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, b = microbatch, L = sequence length, D = model dimension.

import torch
import torch.distributed as dist

from basilisk.distributed.tensor_parallel import _rank_and_world_size
from basilisk.distributed.utils import print_peak_memory
from basilisk.kernels.liger import cross_entropy


class PipelineParallel:
    """GPipe: all microbatch forwards, then all microbatch backwards."""

    def __init__(self, model: torch.nn.Module, n_microbatches: int = 2):
        if n_microbatches < 1:
            raise ValueError(f"n_microbatches must be at least 1, got {n_microbatches}")
        self.model = model
        self.n_microbatches = n_microbatches


    def forward_backward(self, tokens_BL: torch.Tensor) -> torch.Tensor:
        rank, world_size = _rank_and_world_size()
        batch, length = tokens_BL.shape
        if batch % self.n_microbatches != 0:
            raise ValueError(
                f"batch {batch} is not divisible by n_microbatches {self.n_microbatches}"
            )
        # Same padded length on every rank, so the recv buffer needs no extra message
        total_count = int((tokens_BL[:, 1:] != -100).sum().item())
        if total_count < 1:
            raise ValueError("batch has no target tokens")

        dim = self.model.dim
        dtype = next(self.model.parameters()).dtype
        device = tokens_BL.device
        last = rank == world_size - 1
        # Held until the matching backward. This is the GPipe activation memory
        inputs: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []

        for index, tokens_bL in enumerate(tokens_BL.chunk(self.n_microbatches, dim=0)):
            if rank == 0:
                incoming = tokens_bL.clamp(min=0)  # -100 is padding
            else:
                incoming_value = torch.empty(
                    tokens_bL.shape[0], length, dim, device=device, dtype=dtype
                )
                dist.recv(incoming_value, src=rank - 1, tag=index)
                incoming = incoming_value.requires_grad_(True)
                inputs.append(incoming)
            outgoing = self.model(incoming)
            if last:
                local_sum = cross_entropy(
                    outgoing[:, :-1].reshape(-1, outgoing.size(-1)),  # (b * (L - 1), vocab)
                    tokens_bL[:, 1:].reshape(-1),  # (b * (L - 1),)
                    ignore_index=-100,
                    reduction="sum",
                )
                losses.append(local_sum / total_count)
            else:
                expected = (tokens_bL.shape[0], length, dim)
                if tuple(outgoing.shape) != expected:
                    raise RuntimeError(f"stage output {tuple(outgoing.shape)} != {expected}")
                outputs.append(outgoing)
                dist.send(outgoing.detach().contiguous(), dst=rank + 1, tag=index)

        # The last microbatch's gradient is sent and received first
        for index in reversed(range(self.n_microbatches)):
            tag = self.n_microbatches + index
            print_peak_memory()
            if last:
                losses[index].backward()
            else:
                grad = torch.empty_like(outputs[index])
                dist.recv(grad, src=rank + 1, tag=tag)
                outputs[index].backward(grad)
            if rank > 0:
                grad_in = inputs[index].grad
                if grad_in is None:
                    raise RuntimeError("stage input received no gradient")
                dist.send(grad_in.contiguous(), dst=rank - 1, tag=tag)

        if not last:
            return torch.zeros((), device=device, dtype=dtype)
        return torch.stack([loss.detach() for loss in losses]).sum()
