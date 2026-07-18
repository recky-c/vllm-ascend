from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.attention.mla_v1 import MLAPO_MAX_SUPPORTED_TOKENS
from vllm_ascend.attention.sfa.constants import HADAMARD_DIM, TRANSDATA_BLOCK_SIZE
from vllm_ascend.attention.utils import trans_rope_weight, transdata
from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8LinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_NZ,
    AscendDeviceType,
    dispose_layer,
    get_ascend_device_type,
)

if TYPE_CHECKING:
    from vllm_ascend.attention.sfa.impl import AscendSFAImpl

_hadamard_matrix_cache: torch.Tensor | None = None


@dataclass(frozen=True)
class SFAFeatureFlags:
    enable_sfa_prolog_v3: bool
    enable_mlapo: bool
    use_sparse_c8_indexer: bool
    use_sparse_c8_sfa: bool
    mlapo_is_quantized: bool | None = None
    prolog_v3_disable_reasons: tuple[str, ...] = ()
    mlapo_disable_reasons: tuple[str, ...] = ()


def is_w8a8_dynamic_linear(layer: torch.nn.Module | None) -> bool:
    quant_method = getattr(getattr(layer, "quant_method", None), "quant_method", None)
    return isinstance(quant_method, AscendW8A8DynamicLinearMethod)


def get_hadamard_matrix(dim: int = HADAMARD_DIM) -> torch.Tensor:
    global _hadamard_matrix_cache
    if _hadamard_matrix_cache is None:
        import scipy.linalg

        _hadamard_matrix_cache = torch.tensor(
            scipy.linalg.hadamard(dim),
            dtype=torch.bfloat16,
            device="npu",
        ) / (dim**0.5)
    return _hadamard_matrix_cache


def ensure_c8_hadamard(host: AscendSFAImpl) -> None:
    if not (host.has_indexer and host.use_sparse_c8_indexer):
        return

    from vllm_ascend.attention.sfa.impl import AscendSFAImpl

    if AscendSFAImpl.q_hadamard is None or AscendSFAImpl.k_hadamard is None:
        hadamard = get_hadamard_matrix()
        host.q_hadamard = hadamard
        host.k_hadamard = hadamard
        AscendSFAImpl.q_hadamard = hadamard
        AscendSFAImpl.k_hadamard = hadamard


def get_sfa_prolog_v3_unsupported_reasons(host: AscendSFAImpl) -> tuple[str, ...]:
    reasons: list[str] = []
    for name, layer in (
        ("fused_qkv_a_proj", host.fused_qkv_a_proj),
        ("q_proj", host.q_proj),
    ):
        if not is_w8a8_dynamic_linear(layer):
            reasons.append(f"Currently SFA mla_prolog_v3 only supports W8A8 dynamic quantization for {name}.")
    if host.kv_a_layernorm is None or host.q_a_layernorm is None:
        reasons.append("SFA mla_prolog_v3 requires q_a_layernorm and kv_a_layernorm.")
    if getattr(host.q_proj, "_chunk_size", 0):
        reasons.append("SFA mla_prolog_v3 does not support chunked q_proj weights yet.")
    if host.enable_dsa_cp:
        reasons.append("SFA mla_prolog_v3 does not support DSA-CP; DSA-CP takes precedence.")
    if host.is_kv_producer:
        reasons.append("SFA mla_prolog_v3 is disabled on KV producer workers.")
    return tuple(reasons)


def resolve_feature_flags_after_loading(host: AscendSFAImpl) -> SFAFeatureFlags:
    enable_sfa_prolog_v3 = host.enable_sfa_prolog_v3
    enable_mlapo = host.enable_mlapo
    use_sparse_c8_indexer = host.use_sparse_c8_indexer
    use_sparse_c8_sfa = host.use_sparse_c8_sfa
    mlapo_is_quantized: bool | None = None
    prolog_v3_disable_reasons: tuple[str, ...] = ()
    mlapo_disable_reasons: tuple[str, ...] = ()

    if enable_sfa_prolog_v3:
        reasons = get_sfa_prolog_v3_unsupported_reasons(host)
        if reasons:
            enable_sfa_prolog_v3 = False
            enable_mlapo = False
            prolog_v3_disable_reasons = reasons
            for msg in reasons:
                logger.warning_once(
                    f"{msg} Disable SFA mla_prolog_v3 for layer {host.layer_name}; fallback to native preprocessing."
                )
    elif enable_mlapo:
        quant_method = getattr(
            getattr(host.fused_qkv_a_proj, "quant_method", None),
            "quant_method",
            None,
        )
        reasons: list[str] = []
        is_quantized = isinstance(
            quant_method,
            (AscendW8A8LinearMethod, AscendW8A8MXFP8DynamicLinearMethod),
        )
        if host.fused_qkv_a_proj is None:
            reasons.append("fused_qkv_a_proj is None, mlapo is disabled.")
        if not is_quantized and get_ascend_device_type() != AscendDeviceType.A5:
            reasons.append(
                "Currently mlapo only supports W8A8 quantization in SFA scenario on non-A5 devices."
                "Some layers in your model are not quantized with W8A8,"
                "thus mlapo is disabled for these layers."
            )
        if host.enable_dsa_cp:
            reasons.append("Currently mlapo does not support SFA with CP,thus mlapo is disabled for these layers.")
        if reasons:
            enable_mlapo = False
            mlapo_disable_reasons = tuple(reasons)
            for msg in reasons:
                logger.warning_once(msg)
        else:
            mlapo_is_quantized = is_quantized

    return SFAFeatureFlags(
        enable_sfa_prolog_v3=enable_sfa_prolog_v3,
        enable_mlapo=enable_mlapo,
        use_sparse_c8_indexer=use_sparse_c8_indexer,
        use_sparse_c8_sfa=use_sparse_c8_sfa,
        mlapo_is_quantized=mlapo_is_quantized,
        prolog_v3_disable_reasons=prolog_v3_disable_reasons,
        mlapo_disable_reasons=mlapo_disable_reasons,
    )


def _trans_rope_vec(
    tensor: torch.Tensor,
    rope_dim: int,
    reshape_shape: tuple[int, ...],
    view_shape: tuple[int, ...],
) -> torch.Tensor:
    tensor = tensor.reshape(reshape_shape).contiguous()
    tensor = trans_rope_weight(tensor, rope_dim)
    return tensor.view(view_shape).contiguous()


def process_weights_for_fused_prolog_v3(host: AscendSFAImpl) -> None:
    assert host.fused_qkv_a_proj is not None
    assert host.q_proj is not None

    fused_weight = host.fused_qkv_a_proj.weight.data
    weight_dq = fused_weight[..., : host.q_lora_rank].contiguous()
    weight_dkv_kr = fused_weight[..., host.q_lora_rank :].contiguous()
    weight_uq_qr = host.q_proj.weight.data.contiguous()
    host.weight_dq = torch_npu.npu_format_cast(weight_dq, ACL_FORMAT_FRACTAL_NZ)
    host.weight_dkv_kr = torch_npu.npu_format_cast(weight_dkv_kr, ACL_FORMAT_FRACTAL_NZ)
    host.weight_uq_qr = torch_npu.npu_format_cast(weight_uq_qr, ACL_FORMAT_FRACTAL_NZ)

    q_a_proj_deq_scl = host.fused_qkv_a_proj.weight_scale[: host.q_lora_rank].contiguous()
    kv_a_proj_deq_scl = host.fused_qkv_a_proj.weight_scale[host.q_lora_rank :].contiguous()
    host.dequant_scale_w_dq = q_a_proj_deq_scl.view(1, -1).to(torch.float)
    host.dequant_scale_w_dkv_kr = kv_a_proj_deq_scl.view(1, -1).to(torch.float)
    host.dequant_scale_w_uq_qr = host.q_proj.weight_scale.data.view(1, -1).to(torch.float)
    if host.use_sparse_c8_sfa:
        host.sfa_qsfa_k_nope_clip_alpha = torch.ones(
            1,
            dtype=torch.float32,
            device=host.weight_dq.device,
        )
        if host.sfa_qsfa_kr_cache_dummy is None:
            # ckvkr_repo_mode=1 stores rope in the packed KV cache, but the
            # operator still requires kr_cache. Keep a stable, non-aliased
            # dummy so first-run tiling/graph capture cannot alias kv_cache.
            host.sfa_qsfa_kr_cache_dummy = torch.empty(
                0,
                dtype=torch.bfloat16,
                device=host.weight_dq.device,
            )
    if host.is_kv_consumer:
        # Decode-only workers only execute Prolog. Drop the native Linear
        # weights after their Prolog layouts and scales have been copied.
        dispose_layer(host.fused_qkv_a_proj)
        dispose_layer(host.q_proj)
        torch.npu.empty_cache()


def process_weights_for_fused_mlapo(host: AscendSFAImpl, act_dtype: torch.dtype) -> None:
    assert host.kv_a_proj_with_mqa is None
    assert host.fused_qkv_a_proj is not None

    kv_a_proj_wt = host.fused_qkv_a_proj.weight.data[..., host.q_lora_rank :].contiguous()
    q_a_proj_wt = host.fused_qkv_a_proj.weight.data[..., : host.q_lora_rank].contiguous()

    kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
    kv_a_proj_wt = trans_rope_weight(kv_a_proj_wt, host.qk_rope_head_dim)
    kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
    wd_qkv = torch.cat((kv_a_proj_wt, q_a_proj_wt), dim=-1)
    wd_qkv = wd_qkv.t().contiguous()
    wd_qkv = transdata(wd_qkv, block_size=TRANSDATA_BLOCK_SIZE).unsqueeze(0).contiguous()
    host.wd_qkv = torch_npu.npu_format_cast(wd_qkv, ACL_FORMAT_FRACTAL_NZ)

    kv_a_proj_deq_scl = host.fused_qkv_a_proj.deq_scale[host.q_lora_rank :].contiguous()
    q_a_proj_deq_scl = host.fused_qkv_a_proj.deq_scale[: host.q_lora_rank].contiguous()
    kv_rope_shape = (host.kv_lora_rank + host.qk_rope_head_dim, -1)
    kv_rope_flat_shape = (host.kv_lora_rank + host.qk_rope_head_dim,)
    kv_a_proj_deq_scl = _trans_rope_vec(
        kv_a_proj_deq_scl,
        host.qk_rope_head_dim,
        kv_rope_shape,
        kv_rope_flat_shape,
    )
    host.deq_scale_qkv = torch.cat((kv_a_proj_deq_scl, q_a_proj_deq_scl), dim=-1).contiguous()

    kv_a_proj_qt_bias = host.fused_qkv_a_proj.quant_bias[host.q_lora_rank :].contiguous()
    q_a_proj_qt_bias = host.fused_qkv_a_proj.quant_bias[: host.q_lora_rank].contiguous()
    kv_a_proj_qt_bias = _trans_rope_vec(
        kv_a_proj_qt_bias,
        host.qk_rope_head_dim,
        kv_rope_shape,
        kv_rope_flat_shape,
    )
    host.quant_bias_qkv = torch.cat((kv_a_proj_qt_bias, q_a_proj_qt_bias), dim=-1).contiguous()

    wu_q = host.q_proj.weight.data
    wu_q = wu_q.t().reshape(host.num_heads, host.qk_nope_head_dim + host.qk_rope_head_dim, -1)
    wu_q = trans_rope_weight(wu_q, host.qk_rope_head_dim)
    wu_q = wu_q.reshape(host.num_heads * (host.qk_nope_head_dim + host.qk_rope_head_dim), -1)
    wu_q = transdata(wu_q, block_size=TRANSDATA_BLOCK_SIZE).unsqueeze(0).contiguous()
    host.wu_q = torch_npu.npu_format_cast(wu_q, ACL_FORMAT_FRACTAL_NZ)

    qb_deq_scl = host.q_proj.deq_scale.data
    qb_deq_scl = qb_deq_scl.reshape(host.num_heads, host.qk_nope_head_dim + host.qk_rope_head_dim, -1)
    qb_deq_scl = trans_rope_weight(qb_deq_scl, host.qk_rope_head_dim)
    host.qb_deq_scl = qb_deq_scl.reshape(host.num_heads * (host.qk_nope_head_dim + host.qk_rope_head_dim))

    qb_qt_bias = host.q_proj.quant_bias.data
    qb_qt_bias = qb_qt_bias.reshape(host.num_heads, host.qk_nope_head_dim + host.qk_rope_head_dim, -1)
    qb_qt_bias = trans_rope_weight(qb_qt_bias, host.qk_rope_head_dim)
    host.qb_qt_bias = qb_qt_bias.reshape(host.num_heads * (host.qk_nope_head_dim + host.qk_rope_head_dim))

    device = host.q_proj.weight.device
    host.gamma1 = host.q_a_layernorm.weight.data  # type: ignore[union-attr]
    host.beta1 = host.q_a_layernorm.bias.data  # type: ignore[union-attr]
    host.gamma2 = host.kv_a_layernorm.weight.data  # type: ignore[union-attr]
    host.quant_scale0 = host.fused_qkv_a_proj.input_scale.data
    host.quant_offset0 = host.fused_qkv_a_proj.input_offset.data
    host.quant_scale1 = host.q_proj.input_scale.data
    host.quant_offset1 = host.q_proj.input_offset.data
    host.ctkv_scale = torch.tensor([1], dtype=act_dtype, device=device)
    host.q_nope_scale = torch.tensor([1], dtype=act_dtype, device=device)

    # On KV consumers (decode-only) MLAPO uses the transformed weights built above;
    # the original fused_qkv_a_proj/q_proj weights and quant params are no longer
    # referenced, so drop them to save memory.
    if (
        host.vllm_config.kv_transfer_config is not None
        and host.vllm_config.kv_transfer_config.is_kv_consumer
        and host.vllm_config.scheduler_config.max_num_batched_tokens <= MLAPO_MAX_SUPPORTED_TOKENS
    ):
        host.fused_qkv_a_proj.weight = None
        host.fused_qkv_a_proj.deq_scale = None
        host.fused_qkv_a_proj.quant_bias = None
        host.q_proj.weight = None
        host.q_proj.deq_scale = None
        host.q_proj.quant_bias = None
        torch.npu.empty_cache()


def process_weights_for_fused_mlapo_a5(host: AscendSFAImpl, act_dtype: torch.dtype) -> None:
    assert host.fused_qkv_a_proj is not None
    assert host.q_proj is not None
    weight_dq = host.fused_qkv_a_proj.weight.data[..., : host.q_lora_rank].contiguous()
    host.weight_dq = torch_npu.npu_format_cast(weight_dq, ACL_FORMAT_FRACTAL_NZ)

    weight_uq_qr = host.q_proj.weight.data.contiguous()
    host.weight_uq_qr_scale = host.q_proj.weight_scale.data.transpose(0, 1)
    host.weight_uq_qr_scale = host.weight_uq_qr_scale.reshape(
        -1,
        host.weight_uq_qr_scale.shape[1] * host.weight_uq_qr_scale.shape[2],
    )
    host.weight_uq_qr = torch_npu.npu_format_cast(weight_uq_qr, ACL_FORMAT_FRACTAL_NZ)

    weight_dkv_kr = host.fused_qkv_a_proj.weight.data[..., host.q_lora_rank :].contiguous()
    host.weight_dkv_kr = torch_npu.npu_format_cast(weight_dkv_kr, ACL_FORMAT_FRACTAL_NZ)

    weight_scale = host.fused_qkv_a_proj.weight_scale
    weight_scale = weight_scale.transpose(0, 1)
    weight_scale = weight_scale.reshape(-1, weight_scale.shape[1] * weight_scale.shape[2])
    host.weight_dq_scale = weight_scale[: host.q_lora_rank, ...]
    host.weight_dkv_kr_scale = weight_scale[host.q_lora_rank :, ...]


def process_weights_for_fused_mlapo_a5_float(host: AscendSFAImpl, act_dtype: torch.dtype) -> None:
    assert host.fused_qkv_a_proj is not None
    assert host.q_proj is not None
    host.fused_qkv_a_proj.weight.data = host.fused_qkv_a_proj.weight.data.T
    weight_dq = host.fused_qkv_a_proj.weight.data[..., : host.q_lora_rank].contiguous()
    host.weight_dq_cpu = weight_dq.cpu()
    host.weight_dq = torch_npu.npu_format_cast(weight_dq, ACL_FORMAT_FRACTAL_NZ)

    weight_uq_qr = host.q_proj.weight.data.T
    weight_uq_qr = weight_uq_qr.contiguous()
    host.weight_uq_qr_cpu = weight_uq_qr.cpu()
    host.weight_uq_qr = torch_npu.npu_format_cast(weight_uq_qr, ACL_FORMAT_FRACTAL_NZ)

    weight_dkv_kr = host.fused_qkv_a_proj.weight.data[..., host.q_lora_rank :].contiguous()
    host.weight_dkv_kr_cpu = weight_dkv_kr.cpu()
    host.weight_dkv_kr = torch_npu.npu_format_cast(weight_dkv_kr, ACL_FORMAT_FRACTAL_NZ)
