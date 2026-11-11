# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   D = model dimension,
#   S = tokens in this rank's shard.

import torch
import torch.distributed as dist


def _rank_and_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def shard_counts(tokens: int, world_size: int) -> list[int]:
    """Contiguous token shards"""
    base, extra = divmod(tokens, world_size)
    return [base + (1 if rank < extra else 0) for rank in range(world_size)]


def exchange_counts(input_splits: torch.Tensor) -> torch.Tensor:
    """One count per rank. output[i] is how many tokens rank i sends here."""
    output_splits = torch.empty_like(input_splits)
    dist.all_to_all_single(output_splits, input_splits.contiguous())
    return output_splits


class _AllToAll(torch.autograd.Function):
    """Variable-size token exchange. Backward swaps the split directions."""

    @staticmethod
    def forward(
        ctx,
        tokens: torch.Tensor,
        input_splits: torch.Tensor,
        output_splits: torch.Tensor,
    ) -> torch.Tensor:
        ctx.input_splits = tuple(int(size) for size in input_splits.tolist())
        ctx.output_splits = tuple(int(size) for size in output_splits.tolist())
        out = tokens.new_empty(sum(ctx.output_splits), *tokens.shape[1:])
        dist.all_to_all_single(
            out,
            tokens.detach().contiguous(),
            output_split_sizes=list(ctx.output_splits),
            input_split_sizes=list(ctx.input_splits),
        )
        return out


    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        out = grad.new_empty(sum(ctx.input_splits), *grad.shape[1:])
        dist.all_to_all_single(
            out,
            grad.detach().contiguous(),
            output_split_sizes=list(ctx.input_splits),
            input_split_sizes=list(ctx.output_splits),
        )
        return out, None, None


class _GatherTokenShards(torch.autograd.Function):

    @staticmethod
    def forward(ctx, local_SD: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        rank, world_size = _rank_and_world_size()
        ctx.rank = rank
        ctx.counts = tuple(int(size) for size in counts.tolist())
        ctx.offset = sum(ctx.counts[:rank])
        if world_size == 1:
            return local_SD
        local_count = ctx.counts[rank]
        repeated = local_SD.detach().repeat(world_size, *([1] * (local_SD.ndim - 1)))
        out = local_SD.new_empty(sum(ctx.counts), *local_SD.shape[1:])
        dist.all_to_all_single(
            out,
            repeated.contiguous(),
            output_split_sizes=list(ctx.counts),
            input_split_sizes=[local_count] * world_size,
        )
        return out


    @staticmethod
    def backward(ctx, grad_TD: torch.Tensor) -> tuple[torch.Tensor, None]:
        count = ctx.counts[ctx.rank]
        return grad_TD[ctx.offset : ctx.offset + count], None


class _SumReplicatedGrads(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        out = grad.detach().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out
