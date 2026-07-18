import torch
import torch_npu

from vllm_ascend.attention.utils import SFA_QSFA_TILE_SIZE


def custom_kv_rmsnorm_rope(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    *,
    epsilon: float = 1e-5,
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
    if dst_type == 1 or dst_type == torch.int8:
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
