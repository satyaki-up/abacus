# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   M = MLP hidden dimension,
#   H = attention heads,
#   K = query/key head dimension, V = value head dimension.

import torch

from basilisk.attention import MultiHeadAttention
from basilisk.mlp import MLP, MLPTensorParallel
from basilisk.norm import RMSNorm


class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
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
        self.norm_attention = RMSNorm(dim, sequence_parallel)
        self.attention = MultiHeadAttention(
            dim, n_heads, head_dim, tensor_parallel, sequence_parallel, context_parallel
        )
        self.norm_mlp = RMSNorm(dim, sequence_parallel)
        if tensor_parallel:
            self.mlp = MLPTensorParallel(
                dim, hidden_dim=4 * dim, sequence_parallel=sequence_parallel
            )
        else:
            self.mlp = MLP(dim, hidden_dim=4 * dim)
        self.dropout = torch.nn.Dropout(dropout)


    def forward(
        self,
        x_BLD: torch.Tensor,
        rotary: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        normed_BLD = self.norm_attention(x_BLD)  # (B, L, D)
        attn_BLD = self.attention(normed_BLD, rotary)  # (B, L, D)
        x_BLD = x_BLD + self.dropout(attn_BLD)  # (B, L, D)
        mlp_BLD = self.mlp(self.norm_mlp(x_BLD))  # (B, L, D)
        return x_BLD + self.dropout(mlp_BLD)  # (B, L, D)
