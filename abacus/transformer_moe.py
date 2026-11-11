# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   M = MLP hidden dimension,
#   H = attention heads,
#   K = query/key head dimension, V = value head dimension.

import torch

from abacus.embedding import Embedding
from abacus.norm import RMSNorm
from abacus.rope import Rotary
from abacus.transformer_block import TransformerBlockMoe


class TransformerMoe(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_blocks: int,
        vocab_size: int,
        n_experts: int,
        top_k: int,
        dropout: float = 0.0,
        hidden_dim: int | None = None,
        bias_update_speed: float = 1e-3,
        route_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size
        self.n_experts = n_experts
        self.top_k = top_k
        self.embedding = Embedding(vocab_size, dim)
        self.rotary = Rotary(head_dim)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlockMoe(
                    dim,
                    n_heads,
                    head_dim,
                    n_experts,
                    top_k,
                    dropout,
                    hidden_dim=hidden_dim,
                    bias_update_speed=bias_update_speed,
                    route_scale=route_scale,
                )
                for _ in range(n_blocks)
            ]
        )
        self.norm = RMSNorm(dim)
        self.w_D_vocab = torch.nn.Parameter(torch.empty(dim, vocab_size))  # (D, vocab)
        torch.nn.init.xavier_uniform_(self.w_D_vocab)


    def forward(
        self,
        tokens_BL: torch.Tensor,
    ) -> torch.Tensor:
        x_BLD = self.embedding(tokens_BL)  # (B, L, D)
        for block in self.blocks:
            x_BLD = block(x_BLD, self.rotary)
        x_BLD = self.norm(x_BLD)
        return x_BLD @ self.w_D_vocab  # (B, L, vocab)
