from typing import Any, NamedTuple

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import MLAAttentionImpl

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import MLAPO_MAX_SUPPORTED_TOKENS
from vllm_ascend.attention.sfa.constants import (
    BMM_TRANS_MAX_SUPPORTED_TOKENS,
    NPU_QUANT_DST_TYPE_INT8,
)
from vllm_ascend.attention.sfa.dsa_cp import finish_dsa_cp_kv_gather, start_dsa_cp_kv_gather
from vllm_ascend.attention.sfa.kv_quant import (
    KVRmsnormRopeResult,
    SFAKVCacheLayout,
    custom_kv_rmsnorm_rope,
)
from vllm_ascend.attention.sfa.metadata import M
from vllm_ascend.attention.sfa.o_proj_tp import OProjTPGather
from vllm_ascend.attention.sfa.weight_prep import (
    ensure_c8_hadamard,
    is_w8a8_dynamic_linear,
    process_weights_for_fused_mlapo,
    process_weights_for_fused_mlapo_a5,
    process_weights_for_fused_mlapo_a5_float,
    process_weights_for_fused_prolog_v3,
    resolve_feature_flags_after_loading,
)
from vllm_ascend.attention.utils import (
    SFA_QSFA_TILE_SIZE,
    get_sfa_qsfa_packed_head_dim,
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.mxfp_compat import FLOAT8_E8M0FNU_DTYPE
from vllm_ascend.memcache_comm_fence import (
    record_attention_compute_start,
)
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_ND,
    AscendDeviceType,
    dispose_layer,
    enable_dsa_cp,
    enable_dsa_cp_with_o_proj_tp,
    enable_sp,
    get_ascend_device_type,
    maybe_trans_nz,
)


def _get_indexer_types(configs: tuple[Any, ...]) -> Any | None:
    for config in configs:
        if config is None:
            continue
        indexer_types = getattr(config, "indexer_types", None)
        if indexer_types is not None:
            return indexer_types
    return None


def _has_shared_indexer_layers(configs: tuple[Any, ...]) -> bool:
    indexer_types = _get_indexer_types(configs)
    if indexer_types is None:
        return False
    return any(isinstance(indexer_type, str) and indexer_type.lower() == "shared" for indexer_type in indexer_types)


def _get_config_bool(configs: tuple[Any, ...], attr: str) -> bool:
    for config in configs:
        if config is not None and hasattr(config, attr):
            return bool(getattr(config, attr))
    return False


class SFAPreprocessResult(NamedTuple):
    """Unified output of the three SFA preprocess paths (prolog / mlapo / native)."""

    hidden_states: torch.Tensor
    ql_nope: torch.Tensor
    q_pe: torch.Tensor
    q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
    k_li: torch.Tensor | None
    k_li_scale: torch.Tensor | None
    o_proj_full_handle: torch.distributed.Work | None = None
    o_proj_full_param_handles: list[torch.distributed.Work | None] | None = None


class AscendSFAImpl(MLAAttentionImpl):
    """
    Ascend SFA (Sparse Flash Attention) implementation.

    Feature variants (DSA-CP, DCP hooks, Sparse C8, MLAPO/prolog_v3, o_proj TP)
    live in sibling modules under ``vllm_ascend.attention.sfa``; this class
    owns the mainline forward orchestration.

    DCP subclass hooks (overridden in ``sfa_cp.AscendSFADCPImpl``):
    - ``_get_full_kv``
    - ``_record_dcp_query_gather_context``
    """

    # Alias to OProjTPGather.pools for UT compatibility (test_sfa_o_proj_tp).
    o_proj_full_pools = OProjTPGather.pools

    # Shared Hadamard matrices for Sparse C8 indexer (set by ensure_c8_hadamard).
    q_hadamard: torch.Tensor | None = None
    k_hadamard: torch.Tensor | None = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # Required MLA / SFA kwargs (explicit for a clearer contract).
        assert "q_lora_rank" in kwargs, "q_lora_rank is required"
        assert "kv_lora_rank" in kwargs, "kv_lora_rank is required"
        assert "qk_nope_head_dim" in kwargs, "qk_nope_head_dim is required"
        assert "qk_rope_head_dim" in kwargs, "qk_rope_head_dim is required"
        assert "qk_head_dim" in kwargs, "qk_head_dim is required"
        assert "v_head_dim" in kwargs, "v_head_dim is required"
        assert "rotary_emb" in kwargs, "rotary_emb is required"
        assert "kv_b_proj" in kwargs, "kv_b_proj is required"
        assert "o_proj" in kwargs, "o_proj is required"
        assert "q_b_proj" in kwargs, "q_b_proj is required"
        assert "indexer" in kwargs, "indexer key is required (may be None with skip_topk)"

        self.q_lora_rank = kwargs["q_lora_rank"]
        self.kv_lora_rank = kwargs["kv_lora_rank"]
        self.qk_nope_head_dim = kwargs["qk_nope_head_dim"]
        self.qk_rope_head_dim = kwargs["qk_rope_head_dim"]
        self.qk_head_dim = kwargs["qk_head_dim"]
        self.v_head_dim = kwargs["v_head_dim"]
        self.rotary_emb = kwargs["rotary_emb"]
        self.q_proj = kwargs["q_proj"] if self.q_lora_rank is None else kwargs["q_b_proj"]
        self.fused_qkv_a_proj = kwargs.get("fused_qkv_a_proj")
        self.kv_b_proj = kwargs["kv_b_proj"]
        self.o_proj = kwargs["o_proj"]
        self.indexer = kwargs["indexer"]
        self.kv_a_proj_with_mqa = kwargs.get("kv_a_proj_with_mqa")
        self.kv_a_layernorm = kwargs.get("kv_a_layernorm")
        self.q_a_layernorm = kwargs.get("q_a_layernorm")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group
        self.q_b_proj = kwargs["q_b_proj"]
        self.skip_topk = kwargs.get("skip_topk", False)
        self.topk_indices_buffer = kwargs.get("topk_indices_buffer")
        self.layer_name = kwargs.get("layer_name")

        ascend_config = get_ascend_config()
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.vllm_config = get_current_vllm_config()
        kv_transfer_config = self.vllm_config.kv_transfer_config
        self.is_kv_producer = kv_transfer_config is not None and kv_transfer_config.is_kv_producer
        self.is_kv_consumer = kv_transfer_config is not None and kv_transfer_config.is_kv_consumer

        self.sfa_qsfa_tile_size = SFA_QSFA_TILE_SIZE
        self.sfa_qsfa_packed_kv_head_dim = 0
        self.sfa_qsfa_k_nope_clip_alpha: torch.Tensor | None = None
        self.sfa_qsfa_kr_cache_dummy: torch.Tensor | None = None

        self.local_num_heads = self.num_heads
        hf_config = self.vllm_config.model_config.hf_config
        hf_text_config = getattr(self.vllm_config.model_config, "hf_text_config", None)
        config_candidates = (hf_config, hf_text_config)
        self.index_cache_enabled = _get_config_bool(
            config_candidates,
            "use_index_cache",
        ) or _has_shared_indexer_layers(config_candidates)
        self.use_index_cache = self.skip_topk or self.index_cache_enabled
        self.has_indexer = self.indexer is not None
        if not self.has_indexer and not self.skip_topk:
            raise ValueError(
                "Indexer is required for DSA unless skip_topk is enabled. "
                f"Got indexer=None, skip_topk={self.skip_topk}, "
                f"layer_name={self.layer_name}."
            )
        if not self.has_indexer and self.topk_indices_buffer is None:
            raise ValueError(
                "topk_indices_buffer is required when indexer is None and "
                f"skip_topk is enabled. layer_name={self.layer_name}."
            )
        if self.has_indexer:
            self.n_head: int = self.indexer.n_head  # 64
            self.head_dim: int = self.indexer.head_dim  # 128
            self.wq_b = self.indexer.wq_b
            self.wk_weights_proj = self.indexer.wk_weights_proj
            self.k_norm = self.indexer.k_norm
        else:
            self.n_head = getattr(hf_config, "index_n_heads", 0)
            self.head_dim = getattr(hf_config, "index_head_dim", 0)
            self.wq_b = None
            self.wk_weights_proj = None
            self.k_norm = None
        self.cp_size = 1
        self.is_rope_neox_style = True
        self.use_torch_npu_lightning_indexer = False
        if self.vllm_config.model_config.hf_config.model_type in ["glm_moe_dsa"]:
            self.is_rope_neox_style = False
            self.use_torch_npu_lightning_indexer = True

        # Sparse C8 has two independent meanings in SFA:
        # - SFA packed KV cache for npu_kv_quant_sparse_flash_attention.
        # - C8 indexer cache for lightning indexer.
        # GLM5.2 can skip creating indexer on some layers, but these layers
        # still need the packed KV cache when sparse C8 is enabled.
        self.use_sparse_c8_indexer = self.has_indexer and ascend_config.is_sparse_c8_layer(self.indexer.k_cache.prefix)
        self.use_sparse_c8_sfa = self.use_sparse_c8_indexer or (
            ascend_config.enable_sparse_c8 and not self.has_indexer and self.skip_topk
        )
        if self.use_sparse_c8_sfa:
            if get_ascend_device_type() == AscendDeviceType.A5:
                self.c8_k_cache_dtype = torch.float8_e4m3fn
                self.c8_k_scale_cache_dtype = torch.float32
            else:
                self.c8_k_cache_dtype = torch.int8
                self.c8_k_scale_cache_dtype = torch.float16

        if self.use_sparse_c8_sfa:
            self.sfa_qsfa_packed_kv_head_dim = get_sfa_qsfa_packed_head_dim(
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self.sfa_qsfa_tile_size,
            )
        # PD decode consumers with sparse C8 use mla_prolog_v3 to write the packed KV cache.
        self.enable_sfa_prolog_v3 = (
            self.is_kv_consumer and self.use_sparse_c8_sfa and get_ascend_device_type() != AscendDeviceType.A5
        )
        self.enable_mlapo = ascend_config.enable_mlapo and not (
            self.enable_sfa_prolog_v3 or (self.use_sparse_c8_sfa and get_ascend_device_type() != AscendDeviceType.A5)
        )

        # Effective in SFA when FlashComm is enabled.
        self.enable_dsa_cp = enable_dsa_cp()
        self.enable_sp = enable_sp()

        # SFA DSA-CP mixed deployments keep o_proj in the existing TP layout.
        # Decode can use the TP-sharded o_proj directly after an activation
        # all-to-all, while prefill/mixed batches temporarily gather the TP
        # shards into a full-weight buffer because their SFA output is not
        # TP-sharded. This is part of the DSA-CP mixed-mode data path rather
        # than an independent user-facing feature switch.
        self.enable_dsa_cp_with_o_proj_tp = enable_dsa_cp_with_o_proj_tp()

        if self.enable_dsa_cp:
            self.local_num_heads = self.num_heads * self.tp_size

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config=None,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        # sfa does not need to update graph params
        pass

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # NOTE: We currently do not support quant kv_b_proj.
        assert isinstance(self.kv_b_proj.quant_method, UnquantizedLinearMethod)
        # NOTE: Weight will be reshaped next, we need to revert and transpose it.
        kv_b_proj_weight = torch_npu.npu_format_cast(self.kv_b_proj.weight.data, ACL_FORMAT_FRACTAL_ND).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.local_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.local_num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # NOTE: When we make a incontiguous weight contiguous, a new address will be allocated for the weight,
        # in graph + RL scenario, we only capture the graph once, and the weight address is expected to be the same
        # across iterations, so we need to copy the weight to the original address after making it contiguous.
        if not hasattr(self, "W_UV"):
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1).contiguous()
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()
        else:
            self.W_UV.copy_(W_UV.transpose(0, 1).contiguous())
            self.W_UK_T.copy_(W_UK.permute(1, 2, 0).contiguous())

        # TODO(zzzzwwjj): Currently, torch.ops._C_ascend.batch_matmul_transpose cannot support weight nz
        # self.W_UV = maybe_trans_nz(self.W_UV)

        # Dispose kv_b_proj since it is replaced by W_UV and W_UK_T to save memory
        dispose_layer(self.kv_b_proj)
        if self.enable_dsa_cp and self.enable_dsa_cp_with_o_proj_tp:
            self._init_o_proj_tp_full_params()

        flags = resolve_feature_flags_after_loading(self)
        self.enable_sfa_prolog_v3 = flags.enable_sfa_prolog_v3
        self.enable_mlapo = flags.enable_mlapo

        if self.enable_sfa_prolog_v3:
            process_weights_for_fused_prolog_v3(self)
        elif self.enable_mlapo:
            assert flags.mlapo_is_quantized is not None
            self.mlapo_is_quantized = flags.mlapo_is_quantized
            if get_ascend_device_type() == AscendDeviceType.A5:
                if self.mlapo_is_quantized:
                    process_weights_for_fused_mlapo_a5(self, act_dtype)
                else:
                    process_weights_for_fused_mlapo_a5_float(self, act_dtype)
            else:
                process_weights_for_fused_mlapo(self, act_dtype)

        if self.use_sparse_c8_indexer and get_ascend_device_type() == AscendDeviceType.A5:
            if hasattr(self, "mlapo_is_quantized") and not self.mlapo_is_quantized:
                self.c8_k_cache_dtype = act_dtype
                self.c8_k_scale_cache_dtype = act_dtype

        if not self.enable_mlapo and not self.enable_sfa_prolog_v3:
            # if mlapo, W_UK_T can't trans nz
            self.W_UK_T = maybe_trans_nz(self.W_UK_T)

        ensure_c8_hadamard(self)

    # Thin wrappers kept for UT / subclass compatibility.
    @staticmethod
    def _is_w8a8_dynamic_linear(layer: torch.nn.Module | None) -> bool:
        return is_w8a8_dynamic_linear(layer)

    def _get_sfa_prolog_v3_unsupported_reasons(self) -> list[str]:
        from vllm_ascend.attention.sfa.weight_prep import get_sfa_prolog_v3_unsupported_reasons

        return list(get_sfa_prolog_v3_unsupported_reasons(self))

    def _process_weights_for_fused_prolog_v3(self) -> None:
        process_weights_for_fused_prolog_v3(self)

    def _process_weights_for_fused_mlapo(self, act_dtype: torch.dtype):
        process_weights_for_fused_mlapo(self, act_dtype)

    def _process_weights_for_fused_mlapo_a5(self, act_dtype: torch.dtype):
        process_weights_for_fused_mlapo_a5(self, act_dtype)

    def _process_weights_for_fused_mlapo_a5_float(self, act_dtype: torch.dtype):
        process_weights_for_fused_mlapo_a5_float(self, act_dtype)

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        raise NotImplementedError("forward_mha is not supported for SFA attention. Use forward() instead.")

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("forward_mqa is not supported for SFA attention. Use forward() instead.")

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        S = 1
        x = x.view(B, N, S, D)
        x = torch_npu.npu_interleave_rope(x, cos, sin)
        return x.view(B, N, D)

    def _init_o_proj_tp_full_params(self):
        OProjTPGather.init_params(self)

    def _iter_o_proj_input_sharded_quant_params(self):
        yield from OProjTPGather.iter_input_sharded_quant_params(self)

    def _switch_o_proj_params(self, params: dict[str, torch.Tensor]):
        OProjTPGather.switch_params(self, params)

    def _get_o_proj_linear_method(self):
        return OProjTPGather.get_linear_method(self)

    def _is_o_proj_unquantized(self) -> bool:
        return OProjTPGather._is_o_proj_unquantized(self)

    def _apply_o_proj_full_weight(self, attn_output: torch.Tensor) -> torch.Tensor:
        return OProjTPGather.apply_full_weight(self, attn_output)

    def _handle_o_proj_weight_switch_and_forward(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
        o_proj_full_handle: torch.distributed.Work | None,
        o_proj_full_param_handles: list[torch.distributed.Work | None] | None,
        should_shard_weight: bool,
    ) -> tuple[torch.Tensor, bool]:
        return OProjTPGather.handle_weight_switch_and_forward(
            self,
            attn_output,
            output,
            o_proj_full_handle,
            o_proj_full_param_handles,
            should_shard_weight,
        )

    def _get_full_kv(self, k, attn_metadata):
        # Hook for DCP/PCP subclasses (see sfa_cp.AscendSFACPImpl).
        return k

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
    ) -> KVRmsnormRopeResult:
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA"

        use_custom_kv = self.use_sparse_c8_sfa and (
            get_ascend_device_type() != AscendDeviceType.A5 or self.enable_dsa_cp or not self.has_indexer
        )
        if use_custom_kv:
            assert self.kv_a_layernorm is not None
            k_pe, k_nope, knope_scale = custom_kv_rmsnorm_rope(
                kv_no_split,
                self.kv_a_layernorm.weight,
                cos,
                sin,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                dst_type=(
                    torch.float8_e4m3fn if get_ascend_device_type() == AscendDeviceType.A5 else NPU_QUANT_DST_TYPE_INT8
                ),
                tile_size=self.sfa_qsfa_tile_size,
            )
            return KVRmsnormRopeResult(k_pe=k_pe, k_nope=k_nope, knope_scale=knope_scale)

        if self.enable_dsa_cp:
            _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv_no_split,
                self.kv_a_layernorm.weight,  # type: ignore[union-attr]
                cos,
                sin,
                slots.to(torch.int64),
                kv_cache[1],
                kv_cache[0],
                epsilon=self.kv_a_layernorm.variance_epsilon,  # type: ignore[union-attr]
                cache_mode=cache_mode,
                is_output_kv=True,
            )
            return KVRmsnormRopeResult(k_pe=k_pe, k_nope=k_nope, knope_scale=None)

        torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,  # type: ignore[union-attr]
            cos,
            sin,
            slots.to(torch.int64),
            kv_cache[1],
            kv_cache[0],
            epsilon=self.kv_a_layernorm.variance_epsilon,  # type: ignore[union-attr]
            cache_mode=cache_mode,
        )
        return KVRmsnormRopeResult(k_pe=None, k_nope=None, knope_scale=None)

    # Return `ql_nope`, `q_pe`
    # TODO: Deduplicate with mla_v1.AscendMLAImpl._q_proj_and_k_up_proj in a follow-up.
    def _q_proj_and_k_up_proj(self, x):
        q_nope, q_pe = (
            self.q_proj(x)[0]
            .view(-1, self.local_num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    # TODO: Deduplicate with mla_v1.AscendMLAImpl._v_up_proj in a follow-up.
    def _v_up_proj(self, x):
        num_input_tokens, _, _ = x.shape
        if (
            x.dtype in [torch.float16, torch.bfloat16]
            and hasattr(torch.ops._C_ascend, "batch_matmul_transpose")
            and num_input_tokens <= BMM_TRANS_MAX_SUPPORTED_TOKENS
        ):
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank)
            res = torch.empty((num_input_tokens, self.local_num_heads, self.v_head_dim), dtype=x.dtype, device=x.device)
            torch.ops._C_ascend.batch_matmul_transpose(x, self.W_UV, res)
            x = res.reshape(-1, self.local_num_heads * self.v_head_dim)
        elif hasattr(torch_npu, "npu_transpose_batchmatmul"):
            # Convert from (N, B, L)/(N, B, 1, L) to (N, B, L)
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank)
            # Multiply (N, B, L) x (N, L, V) -> (B, N, V)
            x = torch_npu.npu_transpose_batchmatmul(x, self.W_UV, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
            # Convert from (N, B, V) to (B, N * V)
            x = x.reshape(-1, self.local_num_heads * self.v_head_dim)
        else:
            # Convert from (B, N, L) to (N, B, L)
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank).transpose(0, 1)
            # # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            x = torch.bmm(x, self.W_UV)
            # # Convert from (N, B, V) to (B, N * V)
            x = x.transpose(0, 1).reshape(-1, self.local_num_heads * self.v_head_dim)
        return x

    def _sfa_preprocess_with_mlapo(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return DeviceOperator.sfa_preprocess_with_mlapo(
            self,
            hidden_states,
            kv_cache,
            cos,
            sin,
            slot_mapping,
            num_input_tokens,
        )

    def _sfa_preprocess_with_prolog_v3(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        cache_mode: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        ql_nope, q_pe, _, q_c, q_c_scale = DeviceOperator.execute_sfa_mla_prolog_v3(
            self,
            hidden_states=hidden_states,
            rope_sin=sin,
            rope_cos=cos,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            cache_mode=cache_mode,
        )
        ql_nope = ql_nope.view(-1, self.local_num_heads, self.kv_lora_rank)
        q_pe = q_pe.view(-1, self.local_num_heads, self.qk_rope_head_dim)
        if self.has_indexer:
            if q_c is None:
                raise RuntimeError("npu_mla_prolog_v3 did not return query_norm for SFA indexer.")
            q_c = q_c.view(-1, self.q_lora_rank)
            if q_c_scale is not None and self.wq_b is not None and self._is_w8a8_dynamic_linear(self.wq_b):
                q_c = (q_c, q_c_scale.view(-1))
        else:
            q_c = None

        k_nope = kv_cache[0] if cache_mode == "TND" else None
        k_pe = kv_cache[1] if cache_mode == "TND" and not self.use_sparse_c8_sfa else None
        return hidden_states, ql_nope, q_pe, q_c, k_nope, k_pe

    def _apply_indexer_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        reshape_cos_sin_for_npu: bool,
    ) -> torch.Tensor:
        """Apply RoPE to indexer q/k activations (Triton path or npu_rotary_mul)."""
        if HAS_TRITON:
            if reshape_cos_sin_for_npu:
                cos = cos.view(-1, self.qk_rope_head_dim)
                sin = sin.view(-1, self.qk_rope_head_dim)
            return rope_forward_triton_siso(
                x, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )

        pe, nope = torch.split(x, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        if reshape_cos_sin_for_npu:
            cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
            sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)
        pe = pe.unsqueeze(2)
        pe = torch_npu.npu_rotary_mul(pe, cos, sin)
        pe = pe.squeeze(2)
        return torch.cat([pe, nope], dim=-1)

    def _quantize_for_c8_indexer(
        self,
        x: torch.Tensor,
        hadamard: torch.Tensor,
        *,
        unsqueeze_scale: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hadamard rotate + dynamic quant used by Sparse C8 lightning indexer."""
        x = x @ hadamard
        x, scale = torch_npu.npu_dynamic_quant(x.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
        scale = scale.to(self.c8_k_scale_cache_dtype)
        if unsqueeze_scale:
            scale = scale.unsqueeze(-1)
        return x, scale

    def indexer_select_pre_process(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        if not self.has_indexer:
            raise RuntimeError(
                f"indexer_select_pre_process should not be called when indexer is None. layer_name={self.layer_name}."
            )

        assert self.wk_weights_proj is not None
        assert self.k_norm is not None

        kw, _ = self.wk_weights_proj(x)
        k_li = kw[:, : self.head_dim]
        k_li = self.k_norm(k_li).unsqueeze(1)
        k_li = k_li.view(-1, 1, self.head_dim)

        k_li = self._apply_indexer_rope(k_li, cos, sin, reshape_cos_sin_for_npu=True)

        if self.use_sparse_c8_indexer:
            assert AscendSFAImpl.k_hadamard is not None
            k_li, k_li_scale = self._quantize_for_c8_indexer(k_li, AscendSFAImpl.k_hadamard, unsqueeze_scale=True)
        else:
            k_li_scale = None

        return k_li, k_li_scale

    def indexer_select_post_process(
        self,
        x: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ):
        if not self.has_indexer:
            raise RuntimeError(
                f"indexer_select_post_process should not be called when indexer is None. layer_name={self.layer_name}."
            )

        assert self.wk_weights_proj is not None
        assert self.wq_b is not None

        kw, _ = self.wk_weights_proj(x)
        weights = kw[:, self.head_dim :]
        if isinstance(q_c, tuple):
            q_c_tensor, q_c_scale = q_c
            q_c_tensor = q_c_tensor.view(-1, q_c_tensor.shape[-1])
            quant_matmul_kwargs = dict(
                bias=None,
                output_dtype=x.dtype,
            )
            if q_c_tensor.dtype == torch.float8_e4m3fn:
                if q_c_scale.dim() == 2:
                    q_c_scale = q_c_scale.view(q_c_scale.shape[0], -1, 2)
                quant_matmul_kwargs.update(
                    scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                    pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                    group_sizes=[1, 1, getattr(self.wq_b.quant_method.quant_method, "group_size", 32)],
                )
            elif q_c_scale.dim() > 1 and q_c_scale.shape[-1] == 1:
                q_c_scale = q_c_scale.squeeze(dim=-1)
            q_li = torch_npu.npu_quant_matmul(
                q_c_tensor,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=q_c_scale,
                **quant_matmul_kwargs,
            )
        else:
            q_li, _ = self.wq_b(q_c)
        q_li = q_li.view(-1, self.n_head, self.head_dim)
        # post_process receives cos/sin already in the caller's layout; do not reshape.
        q_li = self._apply_indexer_rope(q_li, cos, sin, reshape_cos_sin_for_npu=False)

        q_li_scale = None
        q_li_shape_ori = None
        if self.use_sparse_c8_indexer:
            q_li_shape_ori = q_li.shape
            assert AscendSFAImpl.q_hadamard is not None
            q_li, q_li_scale = self._quantize_for_c8_indexer(q_li, AscendSFAImpl.q_hadamard, unsqueeze_scale=False)

        record_attention_compute_start()
        return DeviceOperator.indexer_select_post_process(
            self,
            q_li,
            q_li_scale,
            q_li_shape_ori,
            weights,
            kv_cache,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            self.use_sparse_c8_indexer,
            self.use_torch_npu_lightning_indexer,
        )

    def _get_indexcache_topk_indices(self, num_tokens: int) -> torch.Tensor:
        if self.topk_indices_buffer is None:
            raise RuntimeError("IndexCache requires topk_indices_buffer when skip_topk is enabled.")
        topk_indices = self.topk_indices_buffer[:num_tokens]
        if topk_indices.dim() == 2:
            topk_indices = topk_indices.unsqueeze(1)
        return topk_indices

    def _update_indexcache_topk_indices(self, topk_indices: torch.Tensor) -> None:
        if self.topk_indices_buffer is None:
            return
        num_tokens = topk_indices.shape[0]
        topk_tokens = topk_indices.shape[-1]
        topk_indices_to_cache = topk_indices
        topk_indices_buffer = self.topk_indices_buffer[:num_tokens, :topk_tokens]
        if topk_indices_to_cache.dim() == 3 and topk_indices_buffer.dim() == 2:
            assert topk_indices_to_cache.shape[1] == 1
            topk_indices_to_cache = topk_indices_to_cache.squeeze(1)
        topk_indices_buffer.copy_(topk_indices_to_cache)

    def _execute_sparse_flash_attention_process(
        self, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
    ):
        return DeviceOperator.execute_sparse_flash_attention_process(
            self,
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        )

    def _record_dcp_query_gather_context(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        attn_metadata: M,
    ) -> None:
        # Hook for DCP subclasses (see sfa_cp.AscendSFADCPImpl).
        return

    def _compose_sfa_kv_cache(self, kv_cache) -> tuple[torch.Tensor, ...] | None:
        """Compose split cache handles into the tuple expected by SFA kernels.

        ``kv_cache`` contains only the main MLA cache owned by the attention
        layer, while ``self.indexer.k_cache.kv_cache`` contains the cache owned
        by the indexer layer. Their possible layouts are:

        - non-C8:
          main ``(k_cache, v_cache)`` + indexer ``(indexer_k_cache,)``
          -> ``(k_cache, v_cache, indexer_k_cache)``
        - Sparse C8:
          main ``(packed_kv_cache,)`` +
          indexer ``(indexer_k_cache, indexer_scale_cache)``
          -> ``(packed_kv_cache, indexer_k_cache, indexer_scale_cache)``

        Layers that reuse another layer's top-k indices have no local indexer;
        for those layers, the main cache tuple is returned unchanged.
        """
        # TODO: Remove this recomposition once SFA kernels accept split
        # main/indexer cache handles directly. The allocator now owns them as
        # separate cache specs, while the current kernel path still expects the
        # legacy combined tuple layout.
        main_cache = kv_cache
        if main_cache is None or not self.has_indexer:
            return main_cache

        indexer_cache = self.indexer.k_cache.kv_cache
        if indexer_cache is None:
            raise RuntimeError(f"SFA indexer cache is not initialized or bound. layer_name={self.layer_name}.")

        if self.use_sparse_c8_indexer:
            if len(indexer_cache) != 2:
                raise RuntimeError(
                    "Sparse C8 SFA indexer cache expects (k_cache, scale_cache), "
                    f"got {len(indexer_cache)} tensors for layer_name={self.layer_name}."
                )
            if len(main_cache) != 1:
                raise RuntimeError(
                    "Sparse C8 SFA main cache expects one packed KV tensor, "
                    f"got {len(main_cache)} tensors for layer_name={self.layer_name}."
                )
            return (main_cache[0], indexer_cache[0], indexer_cache[1])

        if len(indexer_cache) != 1:
            raise RuntimeError(
                "SFA indexer cache expects one k_cache tensor, "
                f"got {len(indexer_cache)} tensors for layer_name={self.layer_name}."
            )
        if len(main_cache) != 2:
            raise RuntimeError(
                "SFA main cache expects (k_cache, v_cache), "
                f"got {len(main_cache)} tensors for layer_name={self.layer_name}."
            )
        return (main_cache[0], main_cache[1], indexer_cache[0])

    def _maybe_indexer_pre_process(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.has_indexer:
            return self.indexer_select_pre_process(x=hidden_states, cos=cos, sin=sin)
        return None, None

    def _preprocess_with_prolog_v3(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        need_gather_q_kv: bool,
        layer_name: str,
    ) -> SFAPreprocessResult:
        if self.enable_sp:
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                hidden_states.contiguous(), need_gather_q_kv
            )
        assert slot_mapping.numel() == hidden_states.shape[0], (
            "SFA Prolog V3 requires one cache index per input token, "
            f"got token_x={hidden_states.shape[0]} and cache_index={slot_mapping.numel()}."
        )
        k_li, k_li_scale = self._maybe_indexer_pre_process(hidden_states, cos, sin)

        # Prolog updates the paged KV cache in place. Wait for the prompt
        # blocks before writing the first Decode token into their tail block.
        wait_for_kv_layer_from_connector(layer_name)
        hidden_states, ql_nope, q_pe, q_c, _, _ = self._sfa_preprocess_with_prolog_v3(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            cos=cos,
            sin=sin,
            slot_mapping=slot_mapping,
            cache_mode="PA_BSND",
        )
        return SFAPreprocessResult(
            hidden_states=hidden_states,
            ql_nope=ql_nope,
            q_pe=q_pe,
            q_c=q_c,
            k_li=k_li,
            k_li_scale=k_li_scale,
        )

    def _preprocess_with_mlapo(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
        need_gather_q_kv: bool,
        layer_name: str,
    ) -> SFAPreprocessResult:
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states.contiguous(), need_gather_q_kv)
        hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_with_mlapo(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            cos=cos,
            sin=sin,
            slot_mapping=slot_mapping,
            num_input_tokens=num_input_tokens,
        )
        k_li, k_li_scale = self._maybe_indexer_pre_process(hidden_states, cos, sin)
        wait_for_kv_layer_from_connector(layer_name)
        return SFAPreprocessResult(
            hidden_states=hidden_states,
            ql_nope=ql_nope,
            q_pe=q_pe,
            q_c=q_c,
            k_li=k_li,
            k_li_scale=k_li_scale,
        )

    def _preprocess_native(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping_cp: torch.Tensor | None,
        slot_mapping_sfa: torch.Tensor,
        attn_metadata: M,
        need_gather_q_kv: bool,
        layer_name: str,
        full_gather_o_proj_enabled: bool,
    ) -> SFAPreprocessResult:
        assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
        if self.enable_sp and not self.enable_dsa_cp:
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                hidden_states.contiguous(), need_gather_q_kv
            )
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_no_split = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized"
        q_c = self.q_a_layernorm(q_c)

        k_li, k_li_scale = self._maybe_indexer_pre_process(hidden_states, cos, sin)
        wait_for_kv_layer_from_connector(layer_name)

        if self.enable_dsa_cp:
            assert slot_mapping_cp is not None
            kv_slots = slot_mapping_cp
        else:
            kv_slots = slot_mapping_sfa
        kv_result = self.exec_kv(kv_no_split, cos, sin, kv_cache, kv_slots, attn_metadata)
        k_pe, k_nope, knope_scale = kv_result.k_pe, kv_result.k_nope, kv_result.knope_scale

        if (
            self.use_sparse_c8_sfa
            and not self.enable_dsa_cp
            and (get_ascend_device_type() != AscendDeviceType.A5 or not self.has_indexer)
        ):
            assert k_pe is not None
            assert k_nope is not None
            assert knope_scale is not None
            packed_kv = torch.cat([k_nope, k_pe, knope_scale], dim=-1)
            packed_head_dim = self.sfa_qsfa_packed_kv_head_dim
            assert packed_kv.shape[-1] == packed_head_dim
            torch_npu.npu_scatter_nd_update_(
                kv_cache[0].view(-1, packed_head_dim),
                slot_mapping_sfa.view(-1, 1),
                packed_kv.view(-1, packed_head_dim),
            )

        o_proj_full_handle = None
        o_proj_full_param_handles = None
        if self.enable_dsa_cp:
            assert k_pe is not None
            assert k_nope is not None
            gather_handle = start_dsa_cp_kv_gather(
                self,
                k_pe=k_pe,
                k_nope=k_nope,
                knope_scale=knope_scale,
                k_li=k_li,
                k_li_scale=k_li_scale,
                async_op=full_gather_o_proj_enabled,
            )

        ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
        q_pe = self.rope_single(q_pe, cos, sin)
        self._record_dcp_query_gather_context(ql_nope, q_pe, attn_metadata)

        if self.enable_dsa_cp:
            finish = finish_dsa_cp_kv_gather(
                self,
                gather_handle=gather_handle,
                kv_cache=kv_cache,
                slot_mapping_sfa=slot_mapping_sfa,
                attn_metadata=attn_metadata,
                full_gather_o_proj_enabled=full_gather_o_proj_enabled,
            )
            k_li = finish.k_li
            k_li_scale = finish.k_li_scale
            o_proj_full_handle = finish.o_proj_full_handle
            o_proj_full_param_handles = finish.o_proj_full_param_handles

        if self.has_indexer:
            assert k_li is not None
            k_li = self._get_full_kv(k_li, attn_metadata)

        return SFAPreprocessResult(
            hidden_states=hidden_states,
            ql_nope=ql_nope,
            q_pe=q_pe,
            q_c=q_c,
            k_li=k_li,
            k_li_scale=k_li_scale,
            o_proj_full_handle=o_proj_full_handle,
            o_proj_full_param_handles=o_proj_full_param_handles,
        )

    def _write_indexer_kv_cache(
        self,
        k_li: torch.Tensor,
        k_li_scale: torch.Tensor | None,
        kv_cache: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
        attn_metadata: M,
    ) -> None:
        layout = SFAKVCacheLayout.from_flags(self.use_sparse_c8_sfa)
        use_reshape_optim = get_ascend_config().c8_enable_reshape_optim

        def _store(src: torch.Tensor, cache_idx: int) -> None:
            if use_reshape_optim:
                torch.ops._C_ascend.store_kv_block(
                    src,
                    kv_cache[cache_idx],
                    attn_metadata.group_len,
                    attn_metadata.group_key_idx,
                    attn_metadata.group_key_cache_idx,
                    attn_metadata.block_size,
                )
            else:
                torch_npu.npu_scatter_nd_update_(
                    kv_cache[cache_idx].view(-1, src.shape[-1]),
                    slot_mapping.view(-1, 1),
                    src.view(-1, src.shape[-1]),
                )

        _store(k_li, layout.indexer_k_idx)
        if self.use_sparse_c8_indexer:
            assert len(kv_cache) == (3 if self.use_sparse_c8_sfa else 4)
            if k_li_scale is not None:
                _store(k_li_scale, layout.indexer_scale_idx)
        notify_kv_cache_written(self.layer_name or "")

    def _select_topk_indices(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        topk_num_tokens: int,
    ) -> torch.Tensor:
        if self.skip_topk:
            return self._get_indexcache_topk_indices(topk_num_tokens)
        if not self.has_indexer:
            raise RuntimeError(f"skip_topk is False but indexer is None. layer_name={self.layer_name}.")
        assert q_c is not None
        topk_indices = self.indexer_select_post_process(
            x=hidden_states,
            q_c=q_c,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            cos=cos,
            sin=sin,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
        )
        if self.use_index_cache:
            self._update_indexcache_topk_indices(topk_indices)
        return topk_indices

    def _apply_output_proj(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
        o_proj_full_handle: torch.distributed.Work | None,
        o_proj_full_param_handles: list[torch.distributed.Work | None] | None,
        full_gather_o_proj_enabled: bool,
    ) -> torch.Tensor | None:
        """Run o_proj (with optional DSA-CP TP gather). Returns early output or None."""
        if self.enable_dsa_cp_with_o_proj_tp:
            # SFA DSA-CP mixed mode keeps o_proj weight sharded in the TP domain:
            # 1. prefill/mixed: gather TP shards into a temporary full weight.
            # 2. decode-only: all-to-all hidden states, then run TP o_proj.
            result, require_o_proj_forward = self._handle_o_proj_weight_switch_and_forward(
                attn_output=attn_output,
                output=output,
                o_proj_full_handle=o_proj_full_handle,
                o_proj_full_param_handles=o_proj_full_param_handles,
                should_shard_weight=full_gather_o_proj_enabled,
            )
            if not require_o_proj_forward:
                return result
            attn_output = result

        output[...] = self.o_proj(attn_output)[0]
        return None

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        composed_kv_cache = self._compose_sfa_kv_cache(kv_cache)
        assert composed_kv_cache is not None
        kv_cache = composed_kv_cache

        cos = attn_metadata.cos
        sin = attn_metadata.sin
        slot_mapping = attn_metadata.slot_mapping
        slot_mapping_cp = None
        if self.enable_dsa_cp:
            assert attn_metadata.dsa_cp_context is not None
            slot_mapping_cp = attn_metadata.dsa_cp_context.slot_mapping_cp
            actual_seq_lengths_query = attn_metadata.dsa_cp_context.actual_seq_lengths_query
            actual_seq_lengths_key = attn_metadata.dsa_cp_context.actual_seq_lengths_key
        else:
            actual_seq_lengths_query = attn_metadata.cum_query_lens
            actual_seq_lengths_key = attn_metadata.seq_lens
        # DCP replicated indexer stores LI cache with the full/no-CP metadata, while
        # SFA KV remains stored with the DCP-sharded slot mapping.
        slot_mapping_sfa = (
            attn_metadata.dcp_context.slot_mapping
            if attn_metadata.dcp_context is not None
            else attn_metadata.slot_mapping
        )

        num_input_tokens = attn_metadata.num_input_tokens
        # Prefill/mixed DSA-CP computes o_proj with a temporary full weight.
        # Decode keeps the original TP path and only exchanges activations.
        full_gather_o_proj_enabled = self.enable_dsa_cp_with_o_proj_tp and attn_metadata.attn_state not in {
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        }

        if self.enable_sfa_prolog_v3 and attn_metadata.attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        ):
            prep = self._preprocess_with_prolog_v3(
                hidden_states, kv_cache, cos, sin, slot_mapping, need_gather_q_kv, layer_name
            )
        elif self.enable_mlapo and (
            get_ascend_device_type() == AscendDeviceType.A5 or num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS
        ):
            prep = self._preprocess_with_mlapo(
                hidden_states,
                kv_cache,
                cos,
                sin,
                slot_mapping,
                num_input_tokens,
                need_gather_q_kv,
                layer_name,
            )
        else:
            prep = self._preprocess_native(
                hidden_states,
                kv_cache,
                cos,
                sin,
                slot_mapping_cp,
                slot_mapping_sfa,
                attn_metadata,
                need_gather_q_kv,
                layer_name,
                full_gather_o_proj_enabled,
            )

        hidden_states = prep.hidden_states
        ql_nope = prep.ql_nope
        q_pe = prep.q_pe
        q_c = prep.q_c
        k_li = prep.k_li
        k_li_scale = prep.k_li_scale
        o_proj_full_handle = prep.o_proj_full_handle
        o_proj_full_param_handles = prep.o_proj_full_param_handles

        if kv_cache is not None and self.is_kv_producer:
            attn_metadata.reshape_cache_event = torch.npu.Event()

        if kv_cache is not None and self.has_indexer:
            assert k_li is not None
            self._write_indexer_kv_cache(k_li, k_li_scale, kv_cache, slot_mapping, attn_metadata)

        if self.enable_dsa_cp and attn_metadata.dsa_cp_context is not None:
            topk_num_tokens = attn_metadata.dsa_cp_context.local_end_with_pad - attn_metadata.dsa_cp_context.local_start
        else:
            topk_num_tokens = num_input_tokens or hidden_states.shape[0]

        topk_indices = self._select_topk_indices(
            hidden_states,
            q_c,
            kv_cache,
            attn_metadata,
            cos,
            sin,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            topk_num_tokens,
        )

        attn_output = self._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        )
        attn_output = self._v_up_proj(attn_output)

        early_output = self._apply_output_proj(
            attn_output,
            output,
            o_proj_full_handle,
            o_proj_full_param_handles,
            full_gather_o_proj_enabled,
        )
        if early_output is not None:
            return early_output

        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
        return output
