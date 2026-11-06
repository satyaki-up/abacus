# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   H = attention heads,
#   K = query/key head dimension, V = value head dimension.

from dataclasses import dataclass
from einops import rearrange
import torch

from basilisk.distributed.context_parallel import (
    _AllGatherSequenceBlock,
    all_gather_positions,
    zigzag_positions,
)
from basilisk.distributed.tensor_parallel import (
    _GatherFromSequenceParallel,
    _ReduceScatterToSequenceParallel,
    _AllReduceGrads,
    _SumOutputs,
    _rank_and_world_size,
)
from basilisk.norm import RMSNorm
from basilisk.rope import Rotary, apply_rotary_pos_emb


def _later_than_query(query_positions: torch.Tensor, key_positions: torch.Tensor) -> torch.Tensor:
    # True where a key is strictly after its query, so that score is masked out.
    return query_positions[:, None] < key_positions[None, :]


class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w_q_DD = torch.nn.Parameter(torch.empty(self.dim, self.dim))
        self.w_k_DD = torch.nn.Parameter(torch.empty(self.dim, self.dim))
        self.w_v_DD = torch.nn.Parameter(torch.empty(self.dim, self.dim))
        torch.nn.init.xavier_uniform_(self.w_q_DD)
        torch.nn.init.xavier_uniform_(self.w_k_DD)
        torch.nn.init.xavier_uniform_(self.w_v_DD)


    def forward(
        self,
        q_BLD: torch.Tensor,
        k_BLD: torch.Tensor,
        v_BLD: torch.Tensor,
    ) -> torch.Tensor:
        q_BLD = q_BLD @ self.w_q_DD
        k_BLD = k_BLD @ self.w_k_DD
        v_BLD = v_BLD @ self.w_v_DD

        attn_BLL = torch.matmul(q_BLD, k_BLD.transpose(-2, -1))
        attn_BLL = attn_BLL / (self.dim ** 0.5)
        query_positions = torch.arange(q_BLD.shape[1], device=q_BLD.device)
        key_positions = torch.arange(k_BLD.shape[1], device=k_BLD.device)
        attn_BLL = attn_BLL.masked_fill(
            _later_than_query(query_positions, key_positions), float("-inf")
        )
        attn_BLL = torch.softmax(attn_BLL, dim=-1)
        return torch.matmul(attn_BLL, v_BLD)


class MultiHeadAttention(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        tensor_parallel: bool = False,
        sequence_parallel: bool = False,
        context_parallel: bool = False,
    ):
        super().__init__()
        if sequence_parallel and not tensor_parallel:
            raise ValueError("sequence_parallel requires tensor_parallel")
        if context_parallel and (tensor_parallel or sequence_parallel):
            raise ValueError("context_parallel does not combine with tensor or sequence parallel")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.tensor_parallel = tensor_parallel
        self.sequence_parallel = sequence_parallel
        self.context_parallel = context_parallel
        if tensor_parallel:
            self.rank, self.world_size = _rank_and_world_size()
            if n_heads % self.world_size != 0:
                raise ValueError(
                    f"n_heads {n_heads} is not divisible by world size {self.world_size}"
                )
        else:
            self.rank, self.world_size = 0, 1
        self.local_heads = n_heads // self.world_size
        # Q, K, V are column-parallel across heads. O is row-parallel.
        q_DHK = torch.empty(dim, n_heads, head_dim)
        k_DHK = torch.empty(dim, n_heads, head_dim)
        v_DHK = torch.empty(dim, n_heads, head_dim)
        o_HKD = torch.empty(n_heads, head_dim, dim)
        torch.nn.init.xavier_uniform_(rearrange(q_DHK, "d h k -> d (h k)"))
        torch.nn.init.xavier_uniform_(rearrange(k_DHK, "d h k -> d (h k)"))
        torch.nn.init.xavier_uniform_(rearrange(v_DHK, "d h k -> d (h k)"))
        torch.nn.init.xavier_uniform_(rearrange(o_HKD, "h k d -> (h k) d"))
        start = self.rank * self.local_heads
        self.w_q_DHK = torch.nn.Parameter(q_DHK[:, start : start + self.local_heads].clone())
        self.w_k_DHK = torch.nn.Parameter(k_DHK[:, start : start + self.local_heads].clone())
        self.w_v_DHK = torch.nn.Parameter(v_DHK[:, start : start + self.local_heads].clone())
        self.w_o_HKD = torch.nn.Parameter(o_HKD[start : start + self.local_heads].clone())
        # Shared across heads. Sum the scale's grad when heads or tokens are sharded.
        sync_qk = tensor_parallel or context_parallel
        self.q_norm = RMSNorm(head_dim, sync_grads=sync_qk)
        self.k_norm = RMSNorm(head_dim, sync_grads=sync_qk)


    def forward(
        self,
        x_BLD: torch.Tensor,
        rotary: Rotary | None = None,
    ) -> torch.Tensor:
        if self.context_parallel:
            return self._forward_context_parallel(x_BLD, rotary)
        if self.world_size > 1 and self.sequence_parallel:
            x_BLD = _GatherFromSequenceParallel.apply(x_BLD)  # (B, L, D) full sequence
        elif self.world_size > 1:
            x_BLD = _AllReduceGrads.apply(x_BLD)
        q_BHLK = self._split_heads(x_BLD, self.w_q_DHK)
        k_BHLK = self._split_heads(x_BLD, self.w_k_DHK)
        v_BHLK = self._split_heads(x_BLD, self.w_v_DHK)
        q_BHLK, k_BHLK = self._normalize_qk(q_BHLK, k_BHLK)
        if rotary is not None:
            q_BHLK, k_BHLK = self._apply_rope(q_BHLK, k_BHLK, rotary)

        attn_BHLL = torch.einsum("b h l k, b h s k -> b h l s", q_BHLK, k_BHLK)  # (B, H, L, L)
        attn_BHLL = attn_BHLL / (self.head_dim ** 0.5)
        positions = torch.arange(q_BHLK.shape[2], device=q_BHLK.device)
        attn_BHLL = attn_BHLL.masked_fill(_later_than_query(positions, positions), float("-inf"))
        attn_BHLL = torch.softmax(attn_BHLL, dim=-1)
        out_BHLK = torch.einsum("b h l s, b h s k -> b h l k", attn_BHLL, v_BHLK)  # (B, H, L, K)

        out_BLHK = rearrange(out_BHLK, "b h l k -> b l h k")  # (B, L, H, K)
        joined_BLH = rearrange(out_BLHK, "b l h k -> b l (h k)")  # (B, L, H * K)
        partial_BLD = joined_BLH @ rearrange(self.w_o_HKD, "h k d -> (h k) d")  # (B, L, D)
        if self.world_size > 1 and self.sequence_parallel:
            return _ReduceScatterToSequenceParallel.apply(partial_BLD)  # (B, L, D) this rank's shard
        if self.world_size > 1:
            return _SumOutputs.apply(partial_BLD)  # (B, L, D)
        return partial_BLD  # (B, L, D)


    def _forward_context_parallel(
        self,
        x_BLD: torch.Tensor,
        rotary: Rotary | None,
    ) -> torch.Tensor:
        # x is this rank's zigzag shard. Heads stay whole; only the sequence is split.
        rank, world_size = _rank_and_world_size()
        q_BHLK = self._split_heads(x_BLD, self.w_q_DHK)
        k_BHLK = self._split_heads(x_BLD, self.w_k_DHK)
        v_BHLK = self._split_heads(x_BLD, self.w_v_DHK)
        q_BHLK, k_BHLK = self._normalize_qk(q_BHLK, k_BHLK)
        positions = zigzag_positions(x_BLD.shape[1] * world_size, rank, world_size)
        positions = positions.to(device=x_BLD.device)
        if rotary is not None:
            q_BHLK, k_BHLK = self._apply_rope(q_BHLK, k_BHLK, rotary, positions)

        out_BHLK = self._attend_blocks(q_BHLK, k_BHLK, v_BHLK, positions)
        out_BLHK = rearrange(out_BHLK, "b h l k -> b l h k")  # (B, L, H, K)
        joined_BLH = rearrange(out_BLHK, "b l h k -> b l (h k)")  # (B, L, H * K)
        return joined_BLH @ rearrange(self.w_o_HKD, "h k d -> (h k) d")  # (B, L, D)


    def _attend_blocks(
        self,
        q_BHLK: torch.Tensor,
        k_BHLK: torch.Tensor,
        v_BHLK: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # One all-gather of this rank's whole K and V. Q stays the local shard.
        k_BHLK = _AllGatherSequenceBlock.apply(k_BHLK)  # (B, H, L, K)
        v_BHLK = _AllGatherSequenceBlock.apply(v_BHLK)
        key_positions = all_gather_positions(positions)
        scores_BHLS = torch.einsum("b h l k, b h s k -> b h l s", q_BHLK, k_BHLK)
        scores_BHLS = scores_BHLS / (self.head_dim ** 0.5)
        # Gathered keys are in rank order, so the mask uses global positions
        scores_BHLS = scores_BHLS.masked_fill(
            _later_than_query(positions, key_positions), float("-inf")
        )
        weights_BHLS = torch.softmax(scores_BHLS, dim=-1)
        return torch.einsum("b h l s, b h s k -> b h l k", weights_BHLS, v_BHLK)  # (B, H, L, K)


    def _normalize_qk(
        self,
        q_BHLK: torch.Tensor,
        k_BHLK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Over the head dimension, before RoPE
        return self.q_norm(q_BHLK), self.k_norm(k_BHLK)


    def _apply_rope(
        self,
        q_BHLK: torch.Tensor,
        k_BHLK: torch.Tensor,
        rotary: Rotary,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = rotary(q_BHLK, seq_dim=2, positions=positions)  # (1, 1, L, K)
        return apply_rotary_pos_emb(q_BHLK, k_BHLK, cos, sin)


    def _split_heads(self, x_BLD: torch.Tensor, w_DHK: torch.Tensor) -> torch.Tensor:
        projected_BLH = x_BLD @ rearrange(w_DHK, "d h k -> d (h k)")  # (B, L, H * K)
        return rearrange(projected_BLH, "b l (h k) -> b h l k", h=self.local_heads)  # (B, H, L, K)
