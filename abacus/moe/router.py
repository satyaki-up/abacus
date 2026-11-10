# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   E = experts, K = experts selected per token.

import torch


class Router(torch.nn.Module):
    """Sigmoid top-k router with DeepSeek-V3 auxiliary-loss-free balancing."""

    def __init__(
        self,
        dim: int,
        n_experts: int,
        top_k: int,
        bias_update_speed: float = 1e-3,
        route_scale: float = 1.0,
    ):
        super().__init__()
        if n_experts < 1:
            raise ValueError(f"n_experts must be at least 1, got {n_experts}")
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k must be in [1, {n_experts}], got {top_k}")
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.bias_update_speed = bias_update_speed
        self.route_scale = route_scale
        self.w_DE = torch.nn.Parameter(torch.empty(dim, n_experts))  # (D, E)
        torch.nn.init.xavier_uniform_(self.w_DE)
        self.register_buffer("expert_bias_E", torch.zeros(n_experts, dtype=self.w_DE.dtype))


    def forward(
        self,
        x_BLD: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores_BLE = torch.sigmoid(x_BLD @ self.w_DE)  # (B, L, E)
        # Bias breaks ties for the choice only
        selection_BLE = scores_BLE.detach() + self.expert_bias_E
        indices_BLK = torch.topk(selection_BLE, self.top_k, dim=-1).indices  # (B, L, K)
        weights_BLK = scores_BLE.gather(dim=-1, index=indices_BLK)  # (B, L, K)
        weights_BLK = weights_BLK / weights_BLK.sum(dim=-1, keepdim=True)
        weights_BLK = weights_BLK * self.route_scale
        if self.training:
            self.update_bias(indices_BLK)
        return weights_BLK, indices_BLK


    @torch.no_grad()
    def update_bias(
        self,
        indices_BLK: torch.Tensor,
    ) -> None:
        # DeepSeek steps b_i by gamma * sign(average load - load_i) after the batch
        if self.bias_update_speed == 0:
            return
        counts_E = torch.bincount(indices_BLK.reshape(-1), minlength=self.n_experts)
        counts_E = counts_E.to(device=self.expert_bias_E.device)
        # Positive when an expert took fewer tokens than an even split
        violation_E = counts_E.sum() - counts_E * self.n_experts
        step_E = torch.sign(violation_E.to(dtype=self.expert_bias_E.dtype))
        self.expert_bias_E.add_(self.bias_update_speed * step_E)
