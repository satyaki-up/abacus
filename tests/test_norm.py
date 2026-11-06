import torch

from abacus.norm import RMSNorm


def test_rms_norm_matches_torch():
    torch.manual_seed(0)
    dim = 4
    x_BLD = torch.randn(2, 3, dim)

    norm = RMSNorm(dim)
    with torch.no_grad():
        norm.w_gamma_D.copy_(torch.randn(dim))
    reference = torch.nn.RMSNorm(dim, eps=1e-5)
    with torch.no_grad():
        reference.weight.copy_(norm.w_gamma_D)

    torch.testing.assert_close(norm(x_BLD), reference(x_BLD))
