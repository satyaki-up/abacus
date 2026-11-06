# uv run python scripts/memory_estimator.py
import argparse
import math

from basilisk.model_config import DIM, HEAD_DIM, N_BLOCKS, N_HEADS
from basilisk.transformer import Transformer


# Llama 2 vocabulary written by scripts/prep_data.py.
VOCAB_SIZE = 32000
SEQ_LEN = 2048
BATCH_SIZE = 1
FP32_BYTES = 4
GB = 1024**3


def activation_elements(model: Transformer, batch_size: int, seq_len: int) -> int:
    # One tensor per value the dense forward writes. Views are not counted again.
    batch, length = batch_size, seq_len
    dim = model.dim
    n_heads = model.n_heads
    head_dim = model.head_dim
    elements = math.prod((batch, length, dim))  # embedding
    for block in model.blocks:
        hidden = block.mlp.hidden_dim
        elements += math.prod((batch, length, dim))  # attention norm
        elements += 3 * math.prod((batch, n_heads, length, head_dim))  # q, k, v
        elements += math.prod((batch, n_heads, length, length))  # scores
        elements += math.prod((batch, n_heads, length, head_dim))  # attention context
        elements += math.prod((batch, length, dim))  # attention out
        elements += math.prod((batch, length, dim))  # attention residual
        elements += math.prod((batch, length, dim))  # mlp norm
        elements += 2 * math.prod((batch, length, hidden))  # gate, up
        elements += math.prod((batch, length, hidden))  # gate * up
        elements += math.prod((batch, length, dim))  # mlp out
        elements += math.prod((batch, length, dim))  # mlp residual
    elements += math.prod((batch, length, dim))  # final norm
    elements += math.prod((batch, length, model.vocab_size))  # logits
    return elements


def main() -> None:
    parser = argparse.ArgumentParser(description="Count Transformer parameters and FP32 memory.")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = parser.parse_args()
    if args.vocab_size < 1:
        raise ValueError(f"--vocab-size must be at least 1, got {args.vocab_size}")
    if args.seq_len < 1:
        raise ValueError(f"--seq-len must be at least 1, got {args.seq_len}")
    model = Transformer(
        dim=DIM,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        n_blocks=N_BLOCKS,
        vocab_size=args.vocab_size,
    )
    n_params = sum(parameter.numel() for parameter in model.parameters())
    activation = activation_elements(model, BATCH_SIZE, args.seq_len)
    # exp_avg and exp_avg_sq, each zeros_like(parameter). The step counter is one int.
    optimizer_elements = 2 * n_params
    print(
        f"params {n_params}\n"
        f"weights memory {n_params * FP32_BYTES / GB:.4f} GB\n"
        f"activation memory {activation * FP32_BYTES / GB:.4f} GB\n"
        f"optimizer memory {optimizer_elements * FP32_BYTES / GB:.4f} GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
