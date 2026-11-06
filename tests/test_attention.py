from einops import rearrange
import torch
import torch.nn.functional as F

from basilisk.attention import MultiHeadAttention, ScaledDotProductAttention
from basilisk.rope import Rotary


def test_scaled_dot_product_attention_matches_torch():
    torch.manual_seed(0)
    dim = 4
    q_BLD = torch.randn(2, 3, dim)
    k_BLD = torch.randn(2, 3, dim)
    v_BLD = torch.randn(2, 3, dim)

    attention = ScaledDotProductAttention(dim)
    actual = attention(q_BLD, k_BLD, v_BLD)
    expected = F.scaled_dot_product_attention(
        q_BLD @ attention.w_q_DD,
        k_BLD @ attention.w_k_DD,
        v_BLD @ attention.w_v_DD,
        is_causal=True,
    )

    torch.testing.assert_close(actual, expected)


def _reference_attention(attention, x_BLD, rotary=None, positions=None, causal=False):
    q_BHLK = attention.q_norm(attention._split_heads(x_BLD, attention.w_q_DHK))
    k_BHLK = attention.k_norm(attention._split_heads(x_BLD, attention.w_k_DHK))
    v_BHLK = attention._split_heads(x_BLD, attention.w_v_DHK)
    if rotary is not None:
        q_BHLK, k_BHLK = attention._apply_rope(q_BHLK, k_BHLK, rotary, positions)
    out_BHLK = F.scaled_dot_product_attention(q_BHLK, k_BHLK, v_BHLK, is_causal=causal)
    joined_BLH = rearrange(out_BHLK, "b h l k -> b l (h k)")
    return joined_BLH @ rearrange(attention.w_o_HKD, "h k d -> (h k) d")


def test_multi_head_attention_normalizes_q_and_k_before_rope():
    torch.manual_seed(0)
    dim = 4
    n_heads = 2
    head_dim = dim // n_heads
    x_BLD = torch.randn(2, 3, dim)
    attention = MultiHeadAttention(dim, n_heads, head_dim)
    rotary = Rotary(head_dim)
    with torch.no_grad():
        attention.q_norm.w_gamma_D.copy_(torch.randn(head_dim))
        attention.k_norm.w_gamma_D.copy_(torch.randn(head_dim))

    actual = attention(x_BLD, rotary)
    expected = _reference_attention(attention, x_BLD, rotary, causal=True)

    torch.testing.assert_close(actual, expected)


def test_context_parallel_attention_normalizes_q_and_k_before_rope():
    torch.manual_seed(0)
    dim = 4
    n_heads = 2
    head_dim = dim // n_heads
    x_BLD = torch.randn(2, 4, dim)
    attention = MultiHeadAttention(dim, n_heads, head_dim, context_parallel=True)
    rotary = Rotary(head_dim)
    with torch.no_grad():
        attention.q_norm.w_gamma_D.copy_(torch.randn(head_dim))
        attention.k_norm.w_gamma_D.copy_(torch.randn(head_dim))

    actual = attention(x_BLD, rotary)
    expected = _reference_attention(attention, x_BLD, rotary, causal=True)

    torch.testing.assert_close(actual, expected)
