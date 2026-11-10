import torch

from abacus.mlp import MLP
from abacus.moe.moe import MoE
from abacus.moe.router import Router



def test_expert_bias():
    router = Router(dim=1, n_experts=4, top_k=2, bias_update_speed=10.0, route_scale=2.5)
    with torch.no_grad():
        router.w_DE.copy_(torch.tensor([[2.0, 0.0, -2.0, 1.0]]))
        router.expert_bias_E.copy_(torch.tensor([0.0, 0.0, 10.0, 0.0]))
    x_BLD = torch.ones(1, 4, 1)
    scores_BLE = torch.sigmoid(x_BLD @ router.w_DE)
    router.train()

    weights_BLK, indices_BLK = router(x_BLD)
    # Expert 2 wins a slot only because of the bias
    starting_bias_E = torch.tensor([0.0, 0.0, 10.0, 0.0])
    expected_indices = torch.topk(scores_BLE + starting_bias_E, k=2, dim=-1).indices
    selected_BLK = scores_BLE.gather(dim=-1, index=expected_indices)
    expected_weights = selected_BLK / selected_BLK.sum(dim=-1, keepdim=True) * router.route_scale
    torch.testing.assert_close(indices_BLK, expected_indices)
    torch.testing.assert_close(weights_BLK, expected_weights)
    assert torch.all(indices_BLK[..., 0] == 2)

    # Experts 0 and 2 took every slot, so their bias drops and the idle experts rise.
    stepped_bias_E = torch.tensor([-10.0, 10.0, 0.0, 10.0])
    torch.testing.assert_close(router.expert_bias_E, stepped_bias_E)

    weights_next, indices_next = router(x_BLD)
    expected_next = torch.topk(scores_BLE + stepped_bias_E, k=2, dim=-1).indices
    selected_next = scores_BLE.gather(dim=-1, index=expected_next)
    expected_weights_next = selected_next / selected_next.sum(dim=-1, keepdim=True) * router.route_scale
    torch.testing.assert_close(indices_next, expected_next)
    torch.testing.assert_close(weights_next, expected_weights_next)
    assert torch.all(indices_next[..., 0] == 3)
    assert torch.all(indices_next[..., 1] == 1)


def test_one_expert_matches_mlp():
    torch.manual_seed(0)
    dim = 4
    hidden_dim = 8
    moe = MoE(dim, n_experts=1, top_k=1, hidden_dim=hidden_dim)
    mlp = MLP(dim, hidden_dim)
    with torch.no_grad():
        mlp.w_gate_DM.copy_(moe.w_gate_EDM[0])
        mlp.w_up_DM.copy_(moe.w_up_EDM[0])
        mlp.w_down_MD.copy_(moe.w_down_EMD[0])
    x_BLD = torch.randn(2, 3, dim)
    torch.testing.assert_close(moe(x_BLD), mlp(x_BLD))

