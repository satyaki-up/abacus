# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   M = MLP hidden dimension,
#   H = attention heads,
#   K = query/key head dimension, V = value head dimension.

import torch

from basilisk.distributed.context_parallel import zigzag_positions
from basilisk.distributed.tensor_parallel import (
    _GatherReplicated,
    _ScatterToSequenceParallel,
    _rank_and_world_size,
)
from basilisk.embedding import Embedding
from basilisk.norm import RMSNorm
from basilisk.rope import Rotary
from basilisk.transformer_block import TransformerBlock


class Transformer(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_blocks: int,
        vocab_size: int,
        dropout: float = 0.0,
        tensor_parallel: bool = False,
        sequence_parallel: bool = False,
        context_parallel: bool = False,
    ):
        super().__init__()
        if sequence_parallel and not tensor_parallel:
            raise ValueError("sequence_parallel requires tensor_parallel")
        if context_parallel and (tensor_parallel or sequence_parallel):
            raise ValueError("context_parallel does not combine with tensor or sequence parallel")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size
        self.sequence_parallel = sequence_parallel
        self.context_parallel = context_parallel
        _, self.world_size = _rank_and_world_size() if sequence_parallel else (0, 1)
        self.embedding = Embedding(vocab_size, dim)
        self.rotary = Rotary(head_dim)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    n_heads,
                    head_dim,
                    dropout,
                    tensor_parallel,
                    sequence_parallel,
                    context_parallel,
                )
                for _ in range(n_blocks)
            ]
        )
        self.norm = RMSNorm(dim, sequence_parallel)
        self.w_D_vocab = torch.nn.Parameter(torch.empty(dim, vocab_size))  # (D, vocab)
        torch.nn.init.xavier_uniform_(self.w_D_vocab)


    def forward(
        self,
        tokens_BL: torch.Tensor,
    ) -> torch.Tensor:
        if self.sequence_parallel and self.world_size > 1:
            length = tokens_BL.shape[1]
            if length % self.world_size != 0:
                raise ValueError(
                    f"sequence length {length} is not divisible by world size {self.world_size}"
                )
        if self.context_parallel:
            rank, world_size = _rank_and_world_size()
            if world_size > 1:
                positions = zigzag_positions(tokens_BL.shape[1], rank, world_size)
                tokens_BL = tokens_BL.index_select(1, positions.to(tokens_BL.device))
        x_BLD = self.embedding(tokens_BL)  # (B, L, D)
        if self.sequence_parallel and self.world_size > 1:
            # Norms, dropout, and the residual stay on this shard. GEMMs gather inside the block.
            x_BLD = _ScatterToSequenceParallel.apply(x_BLD)  # (B, L / ranks, D)
        for block in self.blocks:
            x_BLD = block(x_BLD, self.rotary)
        x_BLD = self.norm(x_BLD)
        if self.sequence_parallel and self.world_size > 1:
            # Logits are replicated. Backward slices this gradient instead of summing it.
            x_BLD = _GatherReplicated.apply(x_BLD)  # (B, L, D)
        return x_BLD @ self.w_D_vocab  # (B, L, vocab)


class PipelineStage(torch.nn.Module):
    """One pipeline rank: an even slice of blocks, plus the ends on the edge ranks.

    Rank 0 owns the embedding. The last rank owns the final norm and the logits.
    With 4 ranks and 4 blocks, rank i owns block i.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_blocks: int,
        vocab_size: int,
        rank: int,
        world_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if world_size < 1:
            raise ValueError(f"world_size must be at least 1, got {world_size}")
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be at least 1, got {n_blocks}")
        if n_blocks % world_size != 0:
            raise ValueError(
                f"n_blocks {n_blocks} is not divisible by world size {world_size}"
            )
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside world size {world_size}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size
        self.rank = rank
        self.world_size = world_size
        per_rank = n_blocks // world_size
        if rank == 0:
            self.embedding = Embedding(vocab_size, dim)
        self.rotary = Rotary(head_dim)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(dim, n_heads, head_dim, dropout)
                for _ in range(per_rank)
            ]
        )
        if rank == world_size - 1:
            self.norm = RMSNorm(dim)
            self.w_D_vocab = torch.nn.Parameter(torch.empty(dim, vocab_size))  # (D, vocab)
            torch.nn.init.xavier_uniform_(self.w_D_vocab)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            x = self.embedding(x)  # (b, L, D)
        for block in self.blocks:
            x = block(x, self.rotary)  # (b, L, D)
        if self.rank == self.world_size - 1:
            x = self.norm(x)  # (b, L, D)
            return x @ self.w_D_vocab  # (b, L, vocab)
        return x  # (b, L, D)
