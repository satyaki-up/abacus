import torch

# Training scripts set this from --liger. Every kernel in this module reads it.
_enabled = False


def set_use_liger(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if not _enabled:
        return torch.nn.functional.cross_entropy(
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    if input.device.type != "cuda":
        raise RuntimeError("Liger cross entropy needs a CUDA tensor")
    try:
        from liger_kernel.transformers.functional import liger_cross_entropy
    except ImportError as error:
        raise ImportError("Install liger-kernel to train with --liger") from error
    loss = liger_cross_entropy(
        input,
        target,
        weight=weight,
        size_average=size_average,
        ignore_index=ignore_index,
        reduce=reduce,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    peak = torch.cuda.max_memory_allocated(input.device)
    # print(f"max CUDA memory {peak / 1024**2:.1f} MB", flush=True)
    return loss
