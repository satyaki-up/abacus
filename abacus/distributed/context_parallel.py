import torch
import torch.distributed as dist

from abacus.distributed.tensor_parallel import _rank_and_world_size


def zigzag_positions(length: int, rank: int, world_size: int) -> torch.Tensor:
    """Rank i keeps chunk i and chunk 2*world_size-1-i, so causal work is balanced."""
    chunks = 2 * world_size
    if length % chunks != 0:
        raise ValueError(f"sequence length {length} is not divisible by {chunks}")
    chunk = length // chunks
    first = rank * chunk
    second = (chunks - 1 - rank) * chunk
    return torch.cat((
        torch.arange(first, first + chunk),
        torch.arange(second, second + chunk),
    ))


def all_gather_positions(positions: torch.Tensor) -> torch.Tensor:
    _, world_size = _rank_and_world_size()
    if world_size == 1:
        return positions
    local = positions.contiguous()
    gathered = local.new_empty((local.shape[0] * world_size,))
    dist.all_gather_single(gathered, local)
    return gathered


class _AllGatherSequenceBlock(torch.autograd.Function):
    """All-gather K or V along the sequence. Backward sums grads onto the owner."""

    @staticmethod
    def forward(ctx, block_BHPK: torch.Tensor) -> torch.Tensor:
        _, world_size = _rank_and_world_size()
        ctx.world_size = world_size
        if world_size == 1:
            return block_BHPK.detach().clone()
        local_PBHK = block_BHPK.detach().permute(2, 0, 1, 3).contiguous()
        gathered_PBHK = local_PBHK.new_empty((local_PBHK.shape[0] * world_size, *local_PBHK.shape[1:]))
        dist.all_gather_single(gathered_PBHK, local_PBHK)
        return gathered_PBHK.permute(1, 2, 0, 3).contiguous()  # (B, H, P * ranks, K)


    @staticmethod
    def backward(ctx, grad_BHPK: torch.Tensor) -> torch.Tensor:
        if ctx.world_size == 1:
            return grad_BHPK
        grad_PBHK = grad_BHPK.permute(2, 0, 1, 3).contiguous()
        local_PBHK = grad_PBHK.new_empty((grad_PBHK.shape[0] // ctx.world_size, *grad_PBHK.shape[1:]))
        dist.reduce_scatter_single(local_PBHK, grad_PBHK, op=dist.ReduceOp.SUM)
        return local_PBHK.permute(1, 2, 0, 3).contiguous()  # (B, H, P, K)


def sum_parameter_gradients(model: torch.nn.Module) -> None:
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    if not grads:
        raise RuntimeError("backward produced no gradients")
    flat = torch.cat([grad.reshape(-1) for grad in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for grad in grads:
        numel = grad.numel()
        grad.copy_(flat[offset : offset + numel].view_as(grad))
        offset += numel
