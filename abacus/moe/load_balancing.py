# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   T = tokens, E = experts, K = experts selected per token.

import torch


def load_balancing_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    coeff: float = 1e-2,
) -> torch.Tensor:
    """Switch Transformer auxiliary loss, ``coeff * N * sum_i f_i * P_i``.

    ``router_probs`` has shape ``(..., E)`` and is a distribution over experts.
    ``expert_indices`` selects the dispatched expert for each token, shape
    ``(...)`` or ``(..., K)``. ``f_i`` is the fraction of dispatch slots sent to
    expert ``i``, and ``P_i`` is that expert's mean router probability. A
    balanced router scores ``coeff``; Switch uses ``1e-2``. This is not applied
    by ``MoE``.
    """
    if router_probs.ndim < 1:
        raise ValueError("router_probs must include an expert dimension")
    n_experts = router_probs.shape[-1]
    if n_experts < 1:
        raise ValueError(f"router_probs has no experts, got shape {tuple(router_probs.shape)}")
    batch = router_probs.shape[:-1]
    if expert_indices.shape[: len(batch)] != batch:
        raise ValueError(
            f"expert_indices shape {tuple(expert_indices.shape)} does not match "
            f"router_probs batch {tuple(batch)}"
        )
    extra = expert_indices.ndim - len(batch)
    if extra > 1:
        raise ValueError(
            f"expert_indices has {extra} expert-choice dimensions, expected 0 or 1"
        )
    probs_TE = router_probs.reshape(-1, n_experts).float()
    tokens = probs_TE.shape[0]
    if tokens == 0:
        return router_probs.new_zeros(())
    slots_TK = expert_indices.reshape(tokens, -1).to(dtype=torch.long)
    # A token counts once per expert even if a slot list repeated it.
    dispatch_TE = torch.nn.functional.one_hot(slots_TK, n_experts).amax(dim=1).float()
    k = slots_TK.shape[-1]
    f_E = dispatch_TE.sum(dim=0) / (tokens * k)  # (E,)
    p_E = probs_TE.sum(dim=0) / tokens  # (E,)
    loss = torch.sum(f_E * p_E) * (coeff * n_experts)
    return loss.to(dtype=router_probs.dtype)
