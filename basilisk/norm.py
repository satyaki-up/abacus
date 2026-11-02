# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension.

import torch

from basilisk.distributed.tensor_parallel import _AllReduceGrads, _rank_and_world_size


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, sequence_parallel: bool = False):
        super().__init__()
        self.dim = dim
        self.sequence_parallel = sequence_parallel
        self.world_size = _rank_and_world_size()[1] if sequence_parallel else 1
        self.w_gamma_D = torch.nn.Parameter(torch.ones(self.dim))


    def forward(
        self,
        x_BLD: torch.Tensor,
    ) -> torch.Tensor:
        gamma_D = self.w_gamma_D
        if self.sequence_parallel and self.world_size > 1:
            gamma_D = _AllReduceGrads.apply(gamma_D)
        rms_BL = torch.sqrt(x_BLD.square().mean(dim=-1, keepdim=True) + 1e-5)  # (B, L, 1)
        return gamma_D * x_BLD / rms_BL  # (B, L, D)
