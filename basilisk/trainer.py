# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension.

import torch
import torch.distributed as dist

from basilisk.distributed.context_parallel import sum_parameter_gradients, zigzag_positions
from basilisk.distributed.data_parallel import DataParallel
from basilisk.distributed.pipeline_parallel import PipelineParallel
from basilisk.distributed.tensor_parallel import _rank_and_world_size
from basilisk.optimizers.adamw import AdamW


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        parallel: DataParallel | None = None,
        pipeline: PipelineParallel | None = None,
    ):
        if parallel is not None and pipeline is not None:
            raise ValueError("pipeline parallel does not use DataParallel")
        if pipeline is not None and getattr(model, "context_parallel", False):
            raise ValueError("pipeline parallel does not combine with context parallel")
        self.device = get_device()
        self.model = model.to(self.device)
        self.parallel = parallel
        self.pipeline = pipeline
        if parallel is not None:
            parallel.broadcast_parameters()
        self.optimizer = AdamW(self.model)


    def step(self, tokens_BL: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.train()
        tokens_BL = tokens_BL.to(self.device)
        if self.pipeline is not None:
            self.optimizer.zero_grad()
            loss = self.pipeline.forward_backward(tokens_BL).detach()
            self.optimizer.step()
            return loss, loss
        # -100 is padding. The embedding table has no row for it
        inputs_BL = tokens_BL.clamp(min=0)
        logits = self.model(inputs_BL)  # (B, L, vocab)
        if self._context_parallel_active():
            loss, global_loss, local_loss = self._context_parallel_loss(tokens_BL, logits)
            self.optimizer.zero_grad()
            loss.backward()
            sum_parameter_gradients(self.model)
            self.optimizer.step()
            return global_loss, local_loss
        targets = tokens_BL[:, 1:]  # (B, L - 1)
        token_count = int((targets != -100).sum().item())
        if token_count < 1:
            raise ValueError("batch has no target tokens")
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),  # (B * (L - 1), vocab)
            targets.reshape(-1),  # (B * (L - 1),)
            ignore_index=-100,
        )
        self.optimizer.zero_grad()
        loss.backward()
        local_loss = loss.detach()
        if self.parallel is not None:
            loss = self.parallel.synchronize(loss, token_count)
        self.optimizer.step()
        return loss.detach(), local_loss


    def _context_parallel_active(self) -> bool:
        if not getattr(self.model, "context_parallel", False):
            return False
        if self.parallel is not None:
            raise ValueError("context parallel does not use DataParallel")
        _, world_size = _rank_and_world_size()
        return world_size > 1


    def _context_parallel_loss(
        self,
        tokens_BL: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Needed because half the tokens are on another rank, so cross entropy must be computed across ranks.
        rank, world_size = _rank_and_world_size()
        positions = zigzag_positions(tokens_BL.shape[1], rank, world_size)
        positions = positions.to(tokens_BL.device)
        if logits.shape[1] != positions.shape[0]:
            raise RuntimeError(
                f"expected logits length {positions.shape[0]}, got {logits.shape[1]}"
            )
        next_pos = positions + 1
        valid = next_pos < tokens_BL.shape[1]
        targets = tokens_BL.index_select(1, next_pos.clamp(max=tokens_BL.shape[1] - 1))
        targets = torch.where(valid.unsqueeze(0), targets, torch.full_like(targets, -100))
        local_count = (targets != -100).sum()
        total_count = local_count.to(logits.dtype)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
        if total_count.item() < 1:
            raise ValueError("batch has no target tokens")
        local_sum = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # (B * L, vocab)
            targets.reshape(-1),  # (B * L,)
            ignore_index=-100,
            reduction="sum",
        )
        loss = local_sum / total_count
        reported_sum = local_sum.detach()
        dist.all_reduce(reported_sum, op=dist.ReduceOp.SUM)
        global_loss = (reported_sum / total_count).detach()
        local_loss = local_sum.detach() / local_count.to(logits.dtype).clamp(min=1)
        return loss, global_loss, local_loss


    def train(self, batches) -> None:
        for tokens_BL in batches:
            self.step(tokens_BL)
