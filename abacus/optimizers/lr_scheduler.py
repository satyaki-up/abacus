# Warmup first 2% of steps, decay last 10%.


class WSD:

    def __init__(self, total_steps: int, lr: float = 5e-4):
        if total_steps < 1:
            raise ValueError(f"total_steps must be at least 1, got {total_steps}")
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.total_steps = total_steps
        self.lr = lr
        self.warmup_steps = total_steps * 2 // 100
        self.decay_steps = total_steps * 10 // 100
        self._step = 0


    def learning_rate(self, step: int) -> float:
        if step < 1 or step > self.total_steps:
            raise ValueError(f"step must be in 1..{self.total_steps}, got {step}")
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.lr * step / self.warmup_steps
        if self.decay_steps > 0 and step > self.total_steps - self.decay_steps:
            progress = (step - (self.total_steps - self.decay_steps)) / self.decay_steps
            return self.lr * (1 - progress)
        return self.lr


    def step(self) -> float:
        self._step += 1
        return self.learning_rate(self._step)
