import torch

from abacus.optimizers.adamw import AdamW


def test_adamw_matches_torch():
    for weight_decay in (0.0, 1e-2):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 3)
        reference = torch.nn.Linear(4, 3)
        reference.load_state_dict(model.state_dict())

        optimizer = AdamW(model, weight_decay=weight_decay)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), weight_decay=weight_decay)

        for step in range(10):
            torch.manual_seed(step)
            for param, reference_param in zip(model.parameters(), reference.parameters(), strict=True):
                grad = torch.randn_like(param)
                param.grad = grad
                reference_param.grad = grad.clone()
            optimizer.step()
            reference_optimizer.step()

        for param, reference_param in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(param, reference_param)
            state = optimizer.state[id(param)]
            reference_state = reference_optimizer.state[reference_param]
            torch.testing.assert_close(state["exp_avg"], reference_state["exp_avg"])
            torch.testing.assert_close(state["exp_avg_sq"], reference_state["exp_avg_sq"])


def test_adamw_owned_slice_matches_full_step():
    # A ZeRO rank steps only its flat slice. That slice must match a full AdamW
    # step, and the elements it does not own must stay at their initial values.
    torch.manual_seed(0)
    model = torch.nn.Linear(6, 4)
    full = torch.nn.Linear(6, 4)
    full.load_state_dict(model.state_dict())
    initial = [param.detach().clone() for param in model.parameters()]
    # Weight is (4, 6) and bias is (4,), so [1:4] sits inside both.
    owned = {id(param): (1, 4) for param in model.parameters()}
    optimizer = AdamW(model, owned=owned)
    full_optimizer = AdamW(full)

    for step in range(4):
        torch.manual_seed(step)
        for param, full_param in zip(model.parameters(), full.parameters(), strict=True):
            grad = torch.randn_like(param)
            param.grad = grad
            full_param.grad = grad.clone()
        optimizer.step()
        full_optimizer.step()

    for param, full_param, start in zip(model.parameters(), full.parameters(), initial, strict=True):
        torch.testing.assert_close(param.view(-1)[1:4], full_param.view(-1)[1:4])
        torch.testing.assert_close(param.view(-1)[:1], start.view(-1)[:1])
        torch.testing.assert_close(param.view(-1)[4:], start.view(-1)[4:])
        state = optimizer.state[id(param)]
        full_state = full_optimizer.state[id(full_param)]
        torch.testing.assert_close(state["exp_avg"], full_state["exp_avg"].view(-1)[1:4])
        torch.testing.assert_close(state["exp_avg_sq"], full_state["exp_avg_sq"].view(-1)[1:4])
