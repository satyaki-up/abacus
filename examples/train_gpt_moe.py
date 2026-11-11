# Trains a mixture-of-experts Transformer on the documents written by scripts/prep_data.py.
# --expert-parallel puts one expert on each rank. Every rank reads the same documents.

# One process: uv run python examples/train_gpt_moe.py --limit 100
# Four ranks on one CPU:
# uv run torchrun --standalone --local-addr 127.0.0.1 --nproc_per_node=4 examples/train_gpt_moe.py --expert-parallel --limit 256
# Multi-GPU: uv run torchrun --nproc_per_node=4 examples/train_gpt_moe.py --expert-parallel --wandb
import argparse
from pathlib import Path
import time

import torch

from abacus.distributed.utils import init_process_group, dist_env_vars, rank, shutdown, world_size
from abacus.kernels.liger import set_use_liger
from abacus.model_config import DIM, HEAD_DIM, N_BLOCKS, N_HEADS
from abacus.trainer import Trainer, get_device
from abacus.transformer_moe import TransformerMoe


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pile-10k.pt"
N_EXPERTS = 4
TOP_K = 1
BATCH_SIZE = 4


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
    device = get_device()
    backend = "nccl" if device.type == "cuda" else "gloo"
    parser = argparse.ArgumentParser(description="Train a mixture-of-experts Transformer on tokenized pile-10k documents.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expert-parallel", action="store_true")
    parser.add_argument("--liger", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log cumulative loss to Weights & Biases.")
    parser.add_argument("--wandb-project", default="abacus")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be at least 1, got {args.limit}")
    set_use_liger(args.liger)

    started = time.perf_counter()
    distributed = dist_env_vars()
    if distributed:
        init_process_group(backend, device)
    run = None
    try:
        if args.expert_parallel and distributed and world_size() != N_EXPERTS:
            raise ValueError(
                f"expert parallel keeps 1 expert on each rank, "
                f"need {N_EXPERTS} ranks, found {world_size()}"
            )
        if distributed and world_size() > 1 and not args.expert_parallel:
            raise ValueError("more than one process needs --expert-parallel")
        documents, vocab_size = load_documents(args.data)
        process_rank = rank() if distributed else 0
        documents = select_documents(documents, args.limit)
        total_steps = len(documents) // BATCH_SIZE
        torch.manual_seed(0)
        model = TransformerMoe(
            dim=DIM,
            n_heads=N_HEADS,
            head_dim=HEAD_DIM,
            n_blocks=N_BLOCKS,
            vocab_size=vocab_size,
            n_experts=N_EXPERTS,
            top_k=TOP_K,
            expert_parallel=args.expert_parallel,
        ).to(device)
        trainer = Trainer(model, total_steps=total_steps)
        if args.wandb and process_rank == 0:
            try:
                import wandb
            except ImportError as error:
                raise ImportError("Install wandb to train with --wandb") from error
            run = wandb.init(
                project=args.wandb_project,
                config={
                    "data": str(args.data),
                    "limit": args.limit,
                    "expert_parallel": args.expert_parallel,
                    "liger": args.liger,
                    "wandb": args.wandb,
                },
            )
        for step, tokens_BL in enumerate(padded_batches(documents, BATCH_SIZE), start=1):
            tokens_BL = tokens_BL.to(device)
            loss, _ = trainer.step(tokens_BL)
            if process_rank == 0:
                print(
                    f"step {step}/{total_steps} loss {loss.item():.4f} "
                    f"batch {tokens_BL.shape[0]} length {tokens_BL.shape[1]}\n",
                    flush=True,
                )
            if run is not None:
                run.log({"cumulative_loss": loss.item()}, step=step)
        if process_rank == 0:
            print(f"Execution time {time.perf_counter() - started:.1f}", flush=True)
    finally:
        if run is not None:
            run.finish()
        if distributed:
            shutdown()


if __name__ == "__main__":
    main()
