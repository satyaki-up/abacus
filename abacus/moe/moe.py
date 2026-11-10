# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   M = expert hidden dimension,
#   E = experts, K = experts selected per token,
#   T = B * L tokens, S = tokens dispatched to one expert.

import torch

from abacus.moe.router import Router


class MoE(torch.nn.Module):
    """Top-k mixture of SwiGLU experts.

    ``forward`` takes ``(B, L, D)`` and returns ``(B, L, D)``, the same contract
    as ``MLP``, so a block can swap this in later. Each token is mixed from the
    experts the router picks; nothing is dropped. The Switch auxiliary loss is
    not applied.
    """

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
