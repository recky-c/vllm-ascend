from typing import NamedTuple

import torch
import torch_npu

from vllm_ascend.attention.sfa.constants import (
    DEFAULT_KV_RMSNORM_EPSILON,
    NPU_QUANT_DST_TYPE_INT8,
)
from vllm_ascend.attention.utils import SFA_QSFA_TILE_SIZE


class SFAKVCacheLayout(NamedTuple):
    """Index layout of the composed ``(main..., indexer_k[, indexer_scale])`` tuple.

    Non-C8: ``(k_cache, v_cache, indexer_k_cache)`` — indexer at index 2.
    Sparse C8: ``(packed_kv, indexer_k, indexer_scale)`` — indexer at 1, scale at 2.
    Non-C8 scale index 3 is only valid when a 4-tensor legacy layout is present.
    """

    indexer_k_idx: int
    indexer_scale_idx: int

    @classmethod
    def from_flags(cls, use_sparse_c8_sfa: bool) -> "SFAKVCacheLayout":
        if use_sparse_c8_sfa:
            return cls(indexer_k_idx=1, indexer_scale_idx=2)
        return cls(indexer_k_idx=2, indexer_scale_idx=3)


class KVRmsnormRopeResult(NamedTuple):
    """Uniform return type for ``exec_kv`` / ``custom_kv_rmsnorm_rope``."""

    k_pe: torch.Tensor | None
    k_nope: torch.Tensor | None
    knope_scale: torch.Tensor | None = None


def custom_kv_rmsnorm_rope(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    *,
    epsilon: float = DEFAULT_KV_RMSNORM_EPSILON,
    dst_type: torch.dtype | int = torch.float8_e4m3fn,
    tile_size: int = SFA_QSFA_TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rms_in, rope_in = kv.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    k_nope, _ = torch_npu.npu_rms_norm(rms_in, gamma, epsilon=epsilon)
    k_rope = torch_npu.npu_interleave_rope(rope_in, cos, sin)

    prefix_shape = k_nope.shape[:-1]
    k_nope, knope_scale = torch_npu.npu_dynamic_block_quant(
        k_nope.contiguous().view(-1, 1, kv_lora_rank),
        dst_type=dst_type,
        row_block_size=1,
        col_block_size=tile_size,
    )
    if dst_type == NPU_QUANT_DST_TYPE_INT8 or dst_type == torch.int8:
        # Return byte views so the caller can concatenate all three components.
        return (
            k_rope.contiguous().view(torch.int8),
            k_nope.view(*prefix_shape, kv_lora_rank),
            knope_scale.to(torch.float32).view(*prefix_shape, -1).contiguous().view(torch.int8),
        )

    # A5 transports the BF16 rope and scale bytes through FP8-typed tensors.
    return (
        k_rope.view(torch.float8_e4m3fn),
        k_nope,
        knope_scale.view(knope_scale.shape[0], -1).view(torch.float8_e4m3fn),
    )
