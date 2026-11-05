# uv run python scripts/benchmark_liger.py
import math

import torch
import torch.nn.functional as F

RUNS = 100
WARMUP = 10
N_TOKENS = (4096, 2048, 1024)
VOCAB = 32000


def percentile(samples: list[float], percent: float) -> float:
    ordered = sorted(samples)
    rank = math.ceil(percent / 100 * len(ordered))
    return ordered[rank - 1]


def time_ms(fn, logits: torch.Tensor, targets: torch.Tensor) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn(logits, targets)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def benchmark(name: str, n_tokens: int, fn, logits: torch.Tensor, targets: torch.Tensor) -> None:
    for _ in range(WARMUP):
        fn(logits, targets)
    torch.cuda.synchronize()
    samples = [time_ms(fn, logits, targets) for _ in range(RUNS)]
    print(
        f"{name} tokens {n_tokens} "
        f"p50 {percentile(samples, 50):.3f} ms "
        f"p90 {percentile(samples, 90):.3f} ms "
        f"p99 {percentile(samples, 99):.3f} ms",
        flush=True,
    )


def pytorch_cross_entropy(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        input,
        target,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
    )


def liger_cross_entropy(kernel, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return kernel(
        input,
        target,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Liger cross entropy needs a CUDA tensor")
    try:
        from liger_kernel.transformers.functional import liger_cross_entropy as kernel
    except ImportError as error:
        raise ImportError("Install liger-kernel to train with --liger") from error
    torch.manual_seed(0)

    def liger(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return liger_cross_entropy(kernel, input, target)

    for n_tokens in N_TOKENS:
        logits = torch.randn(n_tokens, VOCAB, device="cuda")
        targets = torch.randint(0, VOCAB, (n_tokens,), device="cuda")
        benchmark("pytorch", n_tokens, pytorch_cross_entropy, logits, targets)
        benchmark("liger", n_tokens, liger, logits, targets)


if __name__ == "__main__":
    main()
