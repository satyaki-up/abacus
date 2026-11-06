import torch

from abacus.mlp import MLP


def test_mlp_matches_torch():
    torch.manual_seed(0)
    dim = 4
    hidden_dim = 8
    x_BLD = torch.randn(2, 3, dim)

    mlp = MLP(dim, hidden_dim)
    gate = torch.nn.Linear(dim, hidden_dim, bias=False)
    up = torch.nn.Linear(dim, hidden_dim, bias=False)
    down = torch.nn.Linear(hidden_dim, dim, bias=False)
    with torch.no_grad():
        gate.weight.copy_(mlp.w_gate_DM.T)
        up.weight.copy_(mlp.w_up_DM.T)
        down.weight.copy_(mlp.w_down_MD.T)

    expected = down(torch.nn.functional.silu(gate(x_BLD)) * up(x_BLD))
    torch.testing.assert_close(mlp(x_BLD), expected)
