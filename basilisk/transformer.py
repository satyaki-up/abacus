# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   M = MLP hidden dimension,
#   H = attention heads,
#   K = query/key head dimension, V = value head dimension.

import torch

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
    ):
        super().__init__()
        if sequence_parallel and not tensor_parallel:
            raise ValueError("sequence_parallel requires tensor_parallel")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size
        self.sequence_parallel = sequence_parallel
        _, self.world_size = _rank_and_world_size() if sequence_parallel else (0, 1)
        self.embedding = Embedding(vocab_size, dim)
        self.rotary = Rotary(head_dim)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    dim, n_heads, head_dim, dropout, tensor_parallel, sequence_parallel
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
