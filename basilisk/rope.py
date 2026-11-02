# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   H = attention heads,
#   K = query/key head dimension.

import torch


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (K / 2,)
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=2, positions=None):
        # Context parallel passes the global positions of this shard
        if positions is not None:
            return self._angles(x, positions)
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            positions = torch.arange(seq_len, device=x.device)
            self.cos_cached, self.sin_cached = self._angles(x, positions)
        return self.cos_cached, self.sin_cached  # (1, 1, L, K)


    def _angles(self, x, positions):
        t = positions.to(device=x.device).type_as(self.inv_freq)  # (L,)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (L, K / 2)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)  # (L, K)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]  # (1, 1, L, K)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]  # (B, H, L, K / 2)
    return torch.cat(
        (-x2, x1), dim=-1
    )  # (B, H, L, K)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)  # (B, H, L, K)