# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   M = expert hidden dimension,
#   E = experts, K = experts selected per token,
#   T = B * L tokens, S = tokens dispatched to one expert.

import torch

from abacus.distributed.expert_parallel import (
    _AllToAll,
    _GatherTokenShards,
    _SumReplicatedGrads,
    _rank_and_world_size,
    exchange_counts,
    shard_counts,
)
from abacus.moe.router import Router


class MoE(torch.nn.Module):
    """Top-k mixture of SwiGLU experts."""

    def __init__(
        self,
        dim: int,
        n_experts: int,
        top_k: int,
        hidden_dim: int | None = None,
        bias_update_speed: float = 1e-3,
        route_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.hidden_dim = dim if hidden_dim is None else hidden_dim
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be at least 1, got {self.hidden_dim}")
        self.router = Router(
            dim,
            n_experts,
            top_k,
            bias_update_speed=bias_update_speed,
            route_scale=route_scale,
        )
        gate_EDM = torch.empty(n_experts, dim, self.hidden_dim)
        up_EDM = torch.empty(n_experts, dim, self.hidden_dim)
        down_EMD = torch.empty(n_experts, self.hidden_dim, dim)
        for expert in range(n_experts):
            torch.nn.init.xavier_uniform_(gate_EDM[expert])
            torch.nn.init.xavier_uniform_(up_EDM[expert])
            torch.nn.init.xavier_uniform_(down_EMD[expert])
        self.w_gate_EDM = torch.nn.Parameter(gate_EDM)  # (E, D, M)
        self.w_up_EDM = torch.nn.Parameter(up_EDM)  # (E, D, M)
        self.w_down_EMD = torch.nn.Parameter(down_EMD)  # (E, M, D)


    def forward(
        self,
        x_BLD: torch.Tensor,
    ) -> torch.Tensor:
        weights_BLK, indices_BLK = self.router(x_BLD)  # (B, L, K)
        batch, length, _ = x_BLD.shape
        x_TD = x_BLD.reshape(batch * length, self.dim)
        weights_TK = weights_BLK.reshape(batch * length, self.top_k)
        indices_TK = indices_BLK.reshape(batch * length, self.top_k)
        if x_TD.shape[0] == 0:
            return x_BLD
        parts = []
        for expert in range(self.n_experts):
            chosen_TK = indices_TK.eq(expert)
            selected_T = chosen_TK.any(dim=-1)
            if not bool(selected_T.any()):
                continue
            x_SD = x_TD[selected_T]
            gate_SM = torch.nn.functional.silu(x_SD @ self.w_gate_EDM[expert])  # (S, M)
            up_SM = x_SD @ self.w_up_EDM[expert]  # (S, M)
            y_SD = (gate_SM * up_SM) @ self.w_down_EMD[expert]  # (S, D)
            weight_S = (weights_TK * chosen_TK).sum(dim=-1)[selected_T]  # (S,)
            part_TD = x_TD.new_zeros(x_TD.shape)
            part_TD[selected_T] = y_SD * weight_S.unsqueeze(-1)
            parts.append(part_TD)
        out_TD = torch.stack(parts).sum(dim=0)
        return out_TD.reshape(batch, length, self.dim)  # (B, L, D)


class MoEExpertParallel(torch.nn.Module):
    """Top-k SwiGLU experts, one expert on each rank."""

    def __init__(
        self,
        dim: int,
        n_experts: int,
        top_k: int,
        hidden_dim: int | None = None,
        bias_update_speed: float = 1e-3,
        route_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.hidden_dim = dim if hidden_dim is None else hidden_dim
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be at least 1, got {self.hidden_dim}")
        self.rank, self.world_size = _rank_and_world_size()
        if self.world_size > 1 and self.n_experts != self.world_size:
            raise ValueError(
                f"expert parallel keeps 1 expert on each rank, "
                f"got {self.n_experts} experts and world size {self.world_size}"
            )
        self.router = Router(
            dim,
            n_experts,
            top_k,
            bias_update_speed=bias_update_speed,
            route_scale=route_scale,
            expert_parallel=self.world_size > 1,
        )
        gate_EDM = torch.empty(n_experts, dim, self.hidden_dim)
        up_EDM = torch.empty(n_experts, dim, self.hidden_dim)
        down_EMD = torch.empty(n_experts, self.hidden_dim, dim)
        for expert in range(n_experts):
            torch.nn.init.xavier_uniform_(gate_EDM[expert])
            torch.nn.init.xavier_uniform_(up_EDM[expert])
            torch.nn.init.xavier_uniform_(down_EMD[expert])
        local = 1 if self.world_size > 1 else n_experts
        start = self.rank if self.world_size > 1 else 0
        self.w_gate_EDM = torch.nn.Parameter(gate_EDM[start : start + local].clone())  # (local, D, M)
        self.w_up_EDM = torch.nn.Parameter(up_EDM[start : start + local].clone())  # (local, D, M)
        self.w_down_EMD = torch.nn.Parameter(down_EMD[start : start + local].clone())  # (local, M, D)


    def forward(
        self,
        x_BLD: torch.Tensor,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return MoE.forward(self, x_BLD)
        x_BLD = _SumReplicatedGrads.apply(x_BLD)
        weights_BLK, indices_BLK = self.router(x_BLD)  # (B, L, K)
        batch, length, _ = x_BLD.shape
        tokens = batch * length
        if tokens == 0:
            return x_BLD
        x_TD = x_BLD.reshape(tokens, self.dim)
        weights_TK = weights_BLK.reshape(tokens, self.top_k)
        indices_TK = indices_BLK.reshape(tokens, self.top_k)
        counts = shard_counts(tokens, self.world_size)
        offset = sum(counts[: self.rank])
        local_n = counts[self.rank]
        local_SD = self._dispatch(
            x_TD[offset : offset + local_n],
            weights_TK[offset : offset + local_n],
            indices_TK[offset : offset + local_n],
        )
        counts_E = torch.tensor(counts, device=x_BLD.device, dtype=torch.long)
        out_TD = _GatherTokenShards.apply(local_SD, counts_E)
        return out_TD.reshape(batch, length, self.dim)  # (B, L, D)


    def _dispatch(
        self,
        local_SD: torch.Tensor,
        weights_SK: torch.Tensor,
        indices_SK: torch.Tensor,
    ) -> torch.Tensor:
        # One row per selected expert. A token's K slots can go to different ranks.
        flat_ids = indices_SK.reshape(-1)
        flat_weights = weights_SK.reshape(-1)
        repeated_SD = local_SD.repeat_interleave(self.top_k, dim=0)
        # Group by destination rank. all_to_all sends one contiguous chunk per rank.
        order = torch.argsort(flat_ids, stable=True)
        input_splits = torch.bincount(flat_ids, minlength=self.world_size)
        output_splits = exchange_counts(input_splits)
        received_SD = _AllToAll.apply(repeated_SD[order], input_splits, output_splits)
        gate_SM = torch.nn.functional.silu(received_SD @ self.w_gate_EDM[0])  # (S, M)
        up_SM = received_SD @ self.w_up_EDM[0]  # (S, M)
        expert_SD = (gate_SM * up_SM) @ self.w_down_EMD[0]  # (S, D)
        # Send the results home. This rank now sends what it received, and receives what it sent.
        returned_SD = _AllToAll.apply(expert_SD, output_splits, input_splits)
        # inverse[order] is identity, so this undoes the argsort.
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.shape[0], device=order.device)
        restored_SD = returned_SD.index_select(0, inverse)
        weighted_SD = restored_SD * flat_weights.unsqueeze(-1)
        return weighted_SD.reshape(local_SD.shape[0], self.top_k, self.dim).sum(dim=1)
