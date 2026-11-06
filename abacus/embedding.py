# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension.

import torch


class Embedding(torch.nn.Module):

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.w_vocab_D = torch.nn.Parameter(torch.empty(vocab_size, dim))  # (vocab, D)
        torch.nn.init.normal_(self.w_vocab_D)


    def forward(
        self,
        tokens_BL: torch.Tensor,
    ) -> torch.Tensor:
        return self.w_vocab_D[tokens_BL]  # (B, L, D)
