# Trains a Transformer on the documents written by scripts/prep_data.py.
# All-gather context parallel similar to Llama 3
# The sequence is cut into 2 * ranks chunks. Rank i keeps chunk i and the chunk from the other end.

# One process: uv run python examples/train_gpt_cp.py --limit 100
# Two ranks on one CPU:
# uv run torchrun --standalone --local-addr 127.0.0.1 --nproc_per_node=2 examples/train_gpt_cp.py --limit 32
import argparse
from pathlib import Path
import time

import torch

from basilisk.distributed.utils import init_process_group, dist_env_vars, rank, shutdown, world_size
from basilisk.kernels.liger import set_use_liger
from basilisk.trainer import Trainer, get_device
from basilisk.transformer import Transformer


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pile-10k.pt"
DIM = 64
N_HEADS = 4
HEAD_DIM = 16
N_BLOCKS = 2
BATCH_SIZE = 2


def load_documents(path: Path) -> tuple[list[torch.Tensor], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run: uv run python scripts/prep_data.py")
    saved = torch.load(path, weights_only=True)
    documents = saved["tokens"]
    if not documents:
        raise ValueError(f"{path} contains no documents")
    return documents, int(saved["vocab_size"])


def select_documents(
    documents: list[torch.Tensor],
    limit: int | None,
) -> list[torch.Tensor]:
    if limit is not None:
        documents = documents[:limit]
    kept: list[torch.Tensor] = []
    for document in documents:
        if document.ndim != 1:
            raise ValueError(f"Expected a document of shape (L,), got {tuple(document.shape)}")
        if document.numel() < 2:
            continue
        kept.append(document)
    return kept


def padded_batches(documents: list[torch.Tensor], batch_size: int, multiple_of: int = 1):
    usable = (len(documents) // batch_size) * batch_size
    if usable == 0:
        raise ValueError(
            f"Need at least {batch_size} documents with 2 or more tokens, found {len(documents)}"
        )
    for start in range(0, usable, batch_size):
        group = documents[start : start + batch_size]
        length = max(document.numel() for document in group)
        if multiple_of > 1 and length % multiple_of != 0:
            length += multiple_of - (length % multiple_of)
        batch = torch.full((batch_size, length), -100, dtype=torch.long)
        for row, document in enumerate(group):
            batch[row, : document.numel()] = document
        yield batch


def main() -> None:
    device = get_device()
    backend = "nccl" if device.type == "cuda" else "gloo"
    parser = argparse.ArgumentParser(description="Train a Transformer on tokenized pile-10k documents.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--liger", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be at least 1, got {args.limit}")
    set_use_liger(args.liger)

    started = time.perf_counter()
    distributed = dist_env_vars()
    if distributed:
        init_process_group(backend, device)
    try:
        documents, vocab_size = load_documents(args.data)
        documents = select_documents(documents, args.limit)
        process_rank = rank() if distributed else 0
        processes = world_size() if distributed else 1
        torch.manual_seed(0)
        model = Transformer(
            dim=DIM,
            n_heads=N_HEADS,
            head_dim=HEAD_DIM,
            n_blocks=N_BLOCKS,
            vocab_size=vocab_size,
            context_parallel=True,
        ).to(device)
        trainer = Trainer(model)
        total_steps = len(documents) // BATCH_SIZE
        # Two chunks per rank
        sequence_multiple = 2 * processes
        batches = padded_batches(documents, BATCH_SIZE, multiple_of=sequence_multiple)
        for step, tokens_BL in enumerate(batches, start=1):
            tokens_BL = tokens_BL.to(device)
            loss, _ = trainer.step(tokens_BL)
            if process_rank == 0:
                print(
                    f"step {step}/{total_steps} loss {loss.item():.4f} "
                    f"batch {tokens_BL.shape[0]} length {tokens_BL.shape[1]}\n",
                    flush=True,
                )
        if process_rank == 0:
            print(f"Execution time {time.perf_counter() - started:.1f}", flush=True)
    finally:
        if distributed:
            shutdown()


if __name__ == "__main__":
    main()
