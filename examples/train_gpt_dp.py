# Trains a Transformer on the documents written by scripts/prep_data.py.

# One process: uv run python examples/train_gpt.py --limit 100
# Four ranks on one CPU:
# uv run torchrun --standalone --local-addr 127.0.0.1 --nproc_per_node=4 examples/train_gpt_dp.py --limit 32
# Same run with ZeRO stage 3:
# uv run torchrun --standalone --local-addr 127.0.0.1 --nproc_per_node=4 examples/train_gpt_dp.py --zero 3 --limit 32
import argparse
from pathlib import Path
import time

import torch
import torch.distributed as dist

from basilisk.distributed.data_parallel import DataParallel
from basilisk.distributed.utils import init_process_group, dist_env_vars, rank, shutdown, world_size
from basilisk.trainer import Trainer, get_device
from basilisk.transformer import Transformer


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pile-10k.pt"
DIM = 64
N_HEADS = 4
HEAD_DIM = 16
N_BLOCKS = 2
PER_RANK_BATCH_SIZE = 2


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
    rank: int,
    world_size: int,
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
    if world_size == 1:
        return kept
    # Drop the tail so every rank runs the same number of steps.
    usable = (len(kept) // world_size) * world_size
    if usable == 0:
        raise ValueError(
            f"Need at least {world_size} documents with 2 or more tokens, found {len(kept)}"
        )
    return kept[:usable][rank::world_size]


def padded_batches(documents: list[torch.Tensor], batch_size: int):
    usable = (len(documents) // batch_size) * batch_size
    if usable == 0:
        raise ValueError(
            f"Need at least {batch_size} documents with 2 or more tokens, found {len(documents)}"
        )
    for start in range(0, usable, batch_size):
        group = documents[start : start + batch_size]
        length = max(document.numel() for document in group)
        batch = torch.full((batch_size, length), -100, dtype=torch.long)
        for row, document in enumerate(group):
            batch[row, : document.numel()] = document
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Transformer on tokenized pile-10k documents.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--zero", type=int, choices=(0, 1, 2, 3), default=0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be at least 1, got {args.limit}")

    started = time.perf_counter()
    distributed = dist_env_vars()
    if args.zero > 0 and not distributed:
        raise ValueError("--zero above 0 needs more than one process")
    if distributed:
        init_process_group()
    try:
        documents, vocab_size = load_documents(args.data)
        process_rank = rank() if distributed else 0
        processes = world_size() if distributed else 1
        documents = select_documents(documents, args.limit, process_rank, processes)
        device = get_device()
        torch.manual_seed(0)
        model = Transformer(
            dim=DIM,
            n_heads=N_HEADS,
            head_dim=HEAD_DIM,
            n_blocks=N_BLOCKS,
            vocab_size=vocab_size,
        ).to(device)
        parallel = DataParallel(model, stage=args.zero) if distributed else None
        trainer = Trainer(model, parallel)
        examples_per_rank = (len(documents) // PER_RANK_BATCH_SIZE) * PER_RANK_BATCH_SIZE
        total_examples = examples_per_rank * processes
        total_steps = examples_per_rank // PER_RANK_BATCH_SIZE
        for step, tokens_BL in enumerate(padded_batches(documents, PER_RANK_BATCH_SIZE), start=1):
            tokens_BL = tokens_BL.to(device)
            loss, local_loss = trainer.step(tokens_BL)
            if distributed:
                print(
                    f"step {step} rank {process_rank} loss {local_loss.item():.4f} "
                    f"batch {tokens_BL.shape[0]} length {tokens_BL.shape[1]}\n",
                    flush=True,
                )
                dist.barrier()
                if process_rank == 0:
                    done = step * processes * tokens_BL.shape[0]
                    print(
                        f"step {step} cumulative loss {loss.item():.4f} {done}/{total_examples}\n",
                        flush=True,
                    )
            elif process_rank == 0:
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
