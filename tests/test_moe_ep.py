import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from abacus.distributed.utils import init_process_group
from abacus.moe.moe import MoE, MoEExpertParallel
from abacus.transformer_moe import TransformerMoe


def test_one_process_matches_moe():
    torch.manual_seed(0)
    reference = MoE(dim=4, n_experts=4, top_k=2, hidden_dim=8, bias_update_speed=0.1)
    torch.manual_seed(0)
    parallel = MoEExpertParallel(dim=4, n_experts=4, top_k=2, hidden_dim=8, bias_update_speed=0.1)
    reference.train()
    parallel.train()
    x_BLD = torch.randn(2, 3, 4)
    torch.testing.assert_close(parallel(x_BLD), reference(x_BLD))
    torch.testing.assert_close(parallel.router.expert_bias_E, reference.router.expert_bias_E)


def _assert_close_grad(actual: torch.Tensor | None, expected: torch.Tensor | None, name: str) -> None:
    if actual is None or expected is None:
        assert actual is None and expected is None, name
        return
    torch.testing.assert_close(actual, expected)


def _compare_parameters(reference: torch.nn.Module, parallel: torch.nn.Module, rank: int) -> None:
    ref = dict(reference.named_parameters())
    par = dict(parallel.named_parameters())
    assert ref.keys() == par.keys()
    for name, param in par.items():
        ref_param = ref[name]
        if param.shape == ref_param.shape:
            torch.testing.assert_close(param, ref_param)
            _assert_close_grad(param.grad, ref_param.grad, name)
            continue
        assert param.shape[0] == 1
        torch.testing.assert_close(param, ref_param[rank : rank + 1])
        ref_grad = None if ref_param.grad is None else ref_param.grad[rank : rank + 1]
        _assert_close_grad(param.grad, ref_grad, name)


def _run_pair(
    reference: torch.nn.Module,
    parallel: torch.nn.Module,
    x: torch.Tensor,
    rank: int,
) -> None:
    reference.train()
    parallel.train()
    _compare_parameters(reference, parallel, rank)
    x_ref = x.detach().clone().requires_grad_(True)
    x_ep = x.detach().clone().requires_grad_(True)
    for _ in range(2):
        y_ref = reference(x_ref)
        y_ep = parallel(x_ep)
        y_ref.sum().backward()
        y_ep.sum().backward()
        torch.testing.assert_close(y_ep, y_ref)
        torch.testing.assert_close(x_ep.grad, x_ref.grad)
        _compare_parameters(reference, parallel, rank)
        reference.zero_grad()
        parallel.zero_grad()
        x_ref.grad = None
        x_ep.grad = None


def _check(rank: int) -> None:
    world = dist.get_world_size()
    with pytest.raises(ValueError, match="1 expert on each rank"):
        MoEExpertParallel(dim=4, n_experts=world + 1, top_k=1)
    torch.manual_seed(0)
    reference = MoE(dim=4, n_experts=world, top_k=1, hidden_dim=8, bias_update_speed=0.25)
    torch.manual_seed(0)
    parallel = MoEExpertParallel(dim=4, n_experts=world, top_k=1, hidden_dim=8, bias_update_speed=0.25)
    # 6 tokens across 4 ranks is 2, 2, 1, 1. One token leaves three ranks with an empty shard.
    _run_pair(reference, parallel, torch.randn(2, 3, 4), rank)
    _run_pair(reference, parallel, torch.randn(1, 1, 4), rank)

    torch.manual_seed(1)
    reference = MoE(dim=4, n_experts=world, top_k=2, hidden_dim=8, bias_update_speed=0.25, route_scale=1.5)
    torch.manual_seed(1)
    parallel = MoEExpertParallel(
        dim=4, n_experts=world, top_k=2, hidden_dim=8, bias_update_speed=0.25, route_scale=1.5
    )
    _run_pair(reference, parallel, torch.randn(2, 3, 4), rank)

    torch.manual_seed(2)
    reference = MoE(dim=4, n_experts=world, top_k=1, hidden_dim=8, bias_update_speed=0.0)
    torch.manual_seed(2)
    parallel = MoEExpertParallel(dim=4, n_experts=world, top_k=1, hidden_dim=8, bias_update_speed=0.0)
    with torch.no_grad():
        reference.router.w_DE.fill_(-8)
        reference.router.w_DE[:, 0] = 8
        parallel.router.w_DE.copy_(reference.router.w_DE)
    _run_pair(reference, parallel, torch.randn(2, 3, 4), rank)

    torch.manual_seed(3)
    reference_model = TransformerMoe(
        dim=8, n_heads=2, head_dim=4, n_blocks=1, vocab_size=16, n_experts=world, top_k=1,
        bias_update_speed=0.25,
    )
    torch.manual_seed(3)
    parallel_model = TransformerMoe(
        dim=8, n_heads=2, head_dim=4, n_blocks=1, vocab_size=16, n_experts=world, top_k=1,
        bias_update_speed=0.25, expert_parallel=True,
    )
    tokens = torch.randint(0, 16, (2, 5))
    reference_model.train()
    parallel_model.train()
    for _ in range(2):
        logits_ref = reference_model(tokens)
        logits_ep = parallel_model(tokens)
        logits_ref.sum().backward()
        logits_ep.sum().backward()
        torch.testing.assert_close(logits_ep, logits_ref)
        _compare_parameters(reference_model, parallel_model, rank)
        reference_model.zero_grad()
        parallel_model.zero_grad()


def _worker(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29547"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    init_process_group()
    try:
        _check(rank)
    finally:
        dist.destroy_process_group()


def test_expert_parallel_matches_single_process():
    world_size = 4
    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
