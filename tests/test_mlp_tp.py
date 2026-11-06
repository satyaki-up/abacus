import os
import sys

import pytest
import torch
import torch.distributed as dist

from abacus.distributed.tensor_parallel import _AllReduceGrads, _SumOutputs


@pytest.fixture(scope="module")
def single_rank_group():
    if dist.is_initialized():
        yield
        return
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29611"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


pytestmark = pytest.mark.usefixtures("single_rank_group")


def test_sum_outputs_gradcheck():
    partial_BLD = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(_SumOutputs.apply, (partial_BLD,))


def test_all_reduce_grads_gradcheck():
    x_BLD = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(_AllReduceGrads.apply, (x_BLD,))
