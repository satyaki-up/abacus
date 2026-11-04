import torch


class AdamW:

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        owned: dict[int, tuple[int, int]] | None = None,
        grad_shards: dict[int, torch.Tensor] | None = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        self.params = list(model.parameters())
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        # ZeRO: id(param) -> (begin, end) of the slice this rank updates.
        self.owned = owned
        # ZeRO-2/3 gradients live here. param.grad has to match the parameter's shape.
        self.grad_shards = grad_shards
        # Keyed by id(param), because a Parameter cannot be a dict key.
        # Each entry holds step (int) plus the moment buffers exp_avg and exp_avg_sq.
        self.state: dict[int, dict[str, torch.Tensor | int]] = {}
        if owned is not None:
            for param in self.params:
                bounds = owned.get(id(param))
                if bounds is None:
                    continue
                begin, end = bounds
                self.state[id(param)] = {
                    "step": 0,
                    "exp_avg": torch.zeros(end - begin, dtype=param.dtype, device=param.device),
                    "exp_avg_sq": torch.zeros(end - begin, dtype=param.dtype, device=param.device),
                }


    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None
        # Stages 2 and 3 keep the owned gradient beside param.grad. Drop it with the full gradient.
        if self.grad_shards is not None:
            self.grad_shards.clear()


    def _weight_and_grad(
        self,
        param: torch.nn.Parameter,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Stage 0: every rank holds the whole parameter.
        if self.owned is None:
            return param, param.grad
        bounds = self.owned.get(id(param))
        # This rank does not own the parameter, so it must not decay or step it.
        if bounds is None:
            return param, None
        begin, end = bounds
        flat = param.data.reshape(-1)
        # Stage 3 stores only the owned elements. Stages 1 and 2 still hold the full parameter.
        weight = flat if flat.numel() == end - begin else flat[begin:end]
        # Stages 2 and 3 already reduced this slice. Stage 1 still has the full gradient.
        if self.grad_shards is not None and id(param) in self.grad_shards:
            return weight, self.grad_shards[id(param)]
        if param.grad is None:
            return weight, None
        grad = param.grad.reshape(-1)
        if grad.numel() != end - begin:
            grad = grad[begin:end]
        return weight, grad


    @torch.no_grad()
    def step(self) -> None:
        for param in self.params:
            weight, grad = self._weight_and_grad(param)
            if grad is None:
                # Skipping an owned parameter would leave its step count behind the other ranks.
                if self.owned is not None and id(param) in self.owned:
                    raise RuntimeError("sharded parameter is missing a gradient")
                continue
            if grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")

            state = self.state.get(id(param))
            if state is None:
                state = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(weight),
                    "exp_avg_sq": torch.zeros_like(weight),
                }
                self.state[id(param)] = state

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if self.weight_decay != 0:
                weight.mul_(1 - self.lr * self.weight_decay)

            exp_avg.lerp_(grad, 1 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            bias_correction1 = 1 - self.beta1 ** state["step"]
            bias_correction2 = 1 - self.beta2 ** state["step"]
            step_size = self.lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(self.eps)
            weight.addcdiv_(exp_avg, denom, value=-step_size)
