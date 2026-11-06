import torch
import torch.distributed as dist


def _rank_and_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class _SumOutputs(torch.autograd.Function):

    @staticmethod
    def forward(ctx, partial_BLD: torch.Tensor) -> torch.Tensor:
        out_BLD = partial_BLD.detach().clone()
        dist.all_reduce(out_BLD, op=dist.ReduceOp.SUM)
        return out_BLD

    @staticmethod
    def backward(ctx, grad_BLD: torch.Tensor) -> torch.Tensor:
        return grad_BLD


def _sequence_first(x_BLD: torch.Tensor) -> torch.Tensor:
    return x_BLD.detach().transpose(0, 1).contiguous()  # (L, B, D)


def _batch_first(x_LBD: torch.Tensor) -> torch.Tensor:
    return x_LBD.detach().transpose(0, 1).contiguous()  # (B, L, D)


def _all_gather_sequence(x_BlD: torch.Tensor) -> torch.Tensor:
    _, world_size = _rank_and_world_size()
    local_LBD = _sequence_first(x_BlD)  # (l, B, D)
    gathered_LBD = local_LBD.new_empty((local_LBD.shape[0] * world_size, *local_LBD.shape[1:]))
    dist.all_gather_single(gathered_LBD, local_LBD)
    return _batch_first(gathered_LBD)  # (B, L, D)


def _reduce_scatter_sequence(partial_BLD: torch.Tensor) -> torch.Tensor:
    _, world_size = _rank_and_world_size()
    partial_LBD = _sequence_first(partial_BLD)  # (L, B, D)
    local_LBD = partial_LBD.new_empty((partial_LBD.shape[0] // world_size, *partial_LBD.shape[1:]))
    dist.reduce_scatter_single(local_LBD, partial_LBD, op=dist.ReduceOp.SUM)
    return _batch_first(local_LBD)  # (B, l, D)


def _slice_sequence(x_BLD: torch.Tensor) -> torch.Tensor:
    rank, world_size = _rank_and_world_size()
    shard_LBD = _sequence_first(x_BLD).chunk(world_size, dim=0)[rank]  # (l, B, D)
    return _batch_first(shard_LBD)  # (B, l, D)


class _GatherFromSequenceParallel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_BlD: torch.Tensor) -> torch.Tensor:
        return _all_gather_sequence(x_BlD)

    @staticmethod
    def backward(ctx, grad_BLD: torch.Tensor) -> torch.Tensor:
        return _reduce_scatter_sequence(grad_BLD)


class _GatherReplicated(torch.autograd.Function):
    """After the last post-norm."""

    @staticmethod
    def forward(ctx, x_BlD: torch.Tensor) -> torch.Tensor:
        ctx.rank, ctx.world_size = _rank_and_world_size()
        return _all_gather_sequence(x_BlD)

    @staticmethod
    def backward(ctx, grad_BLD: torch.Tensor) -> torch.Tensor:
        grad_LBD = _sequence_first(grad_BLD)  # (L, B, D)
        return _batch_first(grad_LBD.chunk(ctx.world_size, dim=0)[ctx.rank])  # (B, l, D)


class _ReduceScatterToSequenceParallel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, partial_BLD: torch.Tensor) -> torch.Tensor:
        return _reduce_scatter_sequence(partial_BLD)

    @staticmethod
    def backward(ctx, grad_BlD: torch.Tensor) -> torch.Tensor:
        return _all_gather_sequence(grad_BlD)


class _ScatterToSequenceParallel(torch.autograd.Function):
    """Before the norm before the first transformer block."""

    @staticmethod
    def forward(ctx, x_BLD: torch.Tensor) -> torch.Tensor:
        return _slice_sequence(x_BLD)

    @staticmethod
    def backward(ctx, grad_BlD: torch.Tensor) -> torch.Tensor:
        return _all_gather_sequence(grad_BlD)


class _AllReduceGrads(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        return weight

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        out = grad.detach().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out
