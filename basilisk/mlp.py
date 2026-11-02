# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   M = MLP hidden dimension.

import torch

from basilisk.distributed.tensor_parallel import (
    _GatherFromSequenceParallel,
    _ReduceScatterToSequenceParallel,
    _AllReduceGrads,
    _SumOutputs,
    _rank_and_world_size,
)


class MLP(torch.nn.Module):

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = dim if hidden_dim is None else hidden_dim
        self.w_gate_DM = torch.nn.Parameter(torch.empty(self.dim, self.hidden_dim))
        self.w_up_DM = torch.nn.Parameter(torch.empty(self.dim, self.hidden_dim))
        self.w_down_MD = torch.nn.Parameter(torch.empty(self.hidden_dim, self.dim))
        torch.nn.init.xavier_uniform_(self.w_gate_DM)
        torch.nn.init.xavier_uniform_(self.w_up_DM)
        torch.nn.init.xavier_uniform_(self.w_down_MD)


    def forward(
        self,
        x_BLD: torch.Tensor,
    ) -> torch.Tensor:
        gate_BLM = torch.nn.functional.silu(x_BLD @ self.w_gate_DM)  # (B, L, M)
        up_BLM = x_BLD @ self.w_up_DM  # (B, L, M)
        return (gate_BLM * up_BLM) @ self.w_down_MD  # (B, L, D)


class MLPTensorParallel(torch.nn.Module):

    def __init__(self, dim: int, hidden_dim: int | None = None, sequence_parallel: bool = False):
        super().__init__()
        self.dim = dim
        self.hidden_dim = dim if hidden_dim is None else hidden_dim
        self.sequence_parallel = sequence_parallel
        self.rank, self.world_size = _rank_and_world_size()
        if self.hidden_dim % self.world_size != 0:
            raise ValueError(
                f"MLP hidden dim {self.hidden_dim} is not divisible by world size {self.world_size}"
            )
        local_hidden = self.hidden_dim // self.world_size
        gate_DM = torch.empty(self.dim, self.hidden_dim)
        up_DM = torch.empty(self.dim, self.hidden_dim)
        down_MD = torch.empty(self.hidden_dim, self.dim)
        torch.nn.init.xavier_uniform_(gate_DM)
        torch.nn.init.xavier_uniform_(up_DM)
        torch.nn.init.xavier_uniform_(down_MD)
        start = self.rank * local_hidden
        self.w_gate_DM = torch.nn.Parameter(gate_DM[:, start : start + local_hidden].clone())
        self.w_up_DM = torch.nn.Parameter(up_DM[:, start : start + local_hidden].clone())
        self.w_down_MD = torch.nn.Parameter(down_MD[start : start + local_hidden].clone())


    def forward(
        self,
        x_BLD: torch.Tensor,
    ) -> torch.Tensor:
        if self.world_size > 1 and self.sequence_parallel:
            x_BLD = _GatherFromSequenceParallel.apply(x_BLD)  # (B, L, D) full sequence
        elif self.world_size > 1:
            x_BLD = _AllReduceGrads.apply(x_BLD)
        gate_BLM = torch.nn.functional.silu(x_BLD @ self.w_gate_DM)  # (B, L, M / ranks)
        up_BLM = x_BLD @ self.w_up_DM  # (B, L, M / ranks)
        partial_BLD = (gate_BLM * up_BLM) @ self.w_down_MD  # (B, L, D)
        if self.world_size > 1 and self.sequence_parallel:
            return _ReduceScatterToSequenceParallel.apply(partial_BLD)  # (B, L, D) this rank's shard
        if self.world_size > 1:
            return _SumOutputs.apply(partial_BLD)  # (B, L, D)
        return partial_BLD  # (B, L, D)
