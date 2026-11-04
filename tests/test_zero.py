import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from basilisk.distributed.data_parallel import DataParallel
from basilisk.distributed.utils import init_process_group
from basilisk.optimizers.adamw import AdamW
from basilisk.trainer import Trainer
from basilisk.transformer import Transformer


def _local_grads(model: torch.nn.Module, tokens: torch.Tensor):
    # Same loss as Trainer.step, so the reference gradient is the mean over this batch's tokens.
    model.train()
    model.zero_grad()
    logits = model(tokens.clamp(min=0))
    targets = tokens[:, 1:]
    token_count = int((targets != -100).sum().item())
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )
    loss.backward()
    grads = [param.grad.detach().clone() for param in model.parameters()]
    return loss.detach(), token_count, grads


def _check(stage: int) -> None:
    # Ranks start from the same weights but different amounts of padding. Dividing by the
    # world size would not match synchronize, which weights each rank's gradient by its token count.
    torch.manual_seed(0)
    model = Transformer(dim=32, n_heads=4, head_dim=8, n_blocks=2, vocab_size=32)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    batches = [
        torch.tensor([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ]),
        torch.tensor([
            [1, 2, 3, 4, -100, -100],
            [7, 8, -100, -100, -100, -100],
        ]),
    ]
    reference = Transformer(dim=32, n_heads=4, head_dim=8, n_blocks=2, vocab_size=32)
    reference.load_state_dict(initial)
    reference_optimizer = AdamW(reference)
    parallel = DataParallel(model, stage=stage)
    trainer = Trainer(model, parallel)
    for _ in range(2):
        # The second step reads the moments from the first, so grads are recomputed at the updated weights.
        scratch = Transformer(dim=32, n_heads=4, head_dim=8, n_blocks=2, vocab_size=32)
        scratch.load_state_dict({name: value.detach().clone() for name, value in reference.state_dict().items()})
        loss_0, count_0, grads_0 = _local_grads(scratch, batches[0])
        scratch.load_state_dict({name: value.detach().clone() for name, value in reference.state_dict().items()})
        loss_1, count_1, grads_1 = _local_grads(scratch, batches[1])
        total = count_0 + count_1
        for param, grad_0, grad_1 in zip(reference.parameters(), grads_0, grads_1, strict=True):
            param.grad = (grad_0 * count_0 + grad_1 * count_1) / total
        reference_optimizer.step()
        loss, _local = trainer.step(batches[dist.get_rank()])
        expected_loss = (loss_0 * count_0 + loss_1 * count_1) / total
        torch.testing.assert_close(loss, expected_loss)
        for param, reference_param in zip(parallel.full_parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(param, reference_param)


def _worker(rank: int, world_size: int) -> None:
    # One process group for every stage. Each stage builds a new model, so stage 3 hooks are not reused.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29531"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    init_process_group()
    try:
        for stage in (0, 1, 2, 3):
            _check(stage)
    finally:
        dist.destroy_process_group()


def test_zero_stages_match_token_weighted_adam():
    # Two CPU ranks. After each step, every stage's parameters must match one AdamW
    # update on the token-weighted gradient.
    world_size = 2
    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
