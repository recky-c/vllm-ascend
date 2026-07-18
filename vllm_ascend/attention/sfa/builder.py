from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa.backend import AscendSFABackend
from vllm_ascend.attention.sfa.constants import TND_LAYOUT_MAX_DECODE_THRESHOLD
from vllm_ascend.attention.sfa.dsa_cp import build_dsa_cp_context
from vllm_ascend.attention.sfa.metadata import AscendSFAMetadata
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    ascend_chunked_prefill_workspace_size,
)
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.utils import enable_dsa_cp
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class AscendSFAMetadataBuilder(MLACommonMetadataBuilder[AscendSFAMetadata]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls if metadata_cls is not None else AscendSFAMetadata,
            supports_dcp_with_varlen,
        )

        self.block_size = vllm_config.cache_config.block_size
        # Match the logical block size selected for BlockTable.
        self.kernel_block_size = select_common_block_size(kv_cache_spec.block_size, [AscendSFABackend])
        self.max_blocks = (vllm_config.model_config.max_model_len + self.block_size - 1) // self.block_size

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.actual_seq_lengths_query = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        self.actual_seq_lengths_key = torch.empty_like(self.actual_seq_lengths_query)
        self.spec_actual_seq_lengths_query: list[torch.Tensor] | None = None
        self.spec_actual_seq_lengths_key: list[torch.Tensor] | None = None
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= TND_LAYOUT_MAX_DECODE_THRESHOLD, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of {TND_LAYOUT_MAX_DECODE_THRESHOLD}, \
                got {self.decode_threshold}"
            )
            self.spec_actual_seq_lengths_query = [
                torch.zeros(max_num_reqs * (spec_token_num + 1) + 1, dtype=torch.int32, device=device)
                for _ in range(spec_token_num)
            ]
            self.spec_actual_seq_lengths_key = [
                torch.zeros(max_num_reqs * (spec_token_num + 1) + 1, dtype=torch.int32, device=device)
                for _ in range(spec_token_num)
            ]

        self.reorder_batch_threshold = self.decode_threshold
        self.attn_mask_builder = AttentionMaskBuilder(self.device)
        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
        self.enable_dsa_cp = enable_dsa_cp()

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        return ascend_chunked_prefill_workspace_size(vllm_config)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendSFAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch", scheduler_output: "SchedulerOutput") -> bool:
        # No need to reorder for Ascend SFA
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendSFAMetadata:
        # common_prefix_len / fast_build are unused; kept for API compatibility.
        return self._build(common_attn_metadata, draft_index=None)

    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
        **kwargs,
    ) -> AscendSFAMetadata:
        return self._build(common_attn_metadata, draft_index=draft_index)

    def _build(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int | None = None,
    ) -> AscendSFAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_input_tokens = common_attn_metadata.num_input_tokens

        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()

        block_size = self.kernel_block_size

        cum_query_lens = common_attn_metadata.query_start_loc[1 : num_reqs + 1]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]

        # Prefer _seq_lens_cpu (always available, updated during draft
        # iterations) over seq_lens_cpu (None in async spec decode mode).
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens_cpu = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        cos, sin = get_cos_and_sin_mla(input_positions, use_cache=(draft_index is None))

        dsa_cp_context = None
        if self.enable_dsa_cp:
            dsa_cp_context, cos, sin, slot_mapping = build_dsa_cp_context(
                num_input_tokens=num_input_tokens,
                num_actual_tokens=num_actual_tokens,
                num_reqs=num_reqs,
                cos=cos,
                sin=sin,
                slot_mapping=slot_mapping,
                cum_query_lens=cum_query_lens,
                seq_lens=seq_lens,
                query_start_loc=common_attn_metadata.query_start_loc,
                draft_index=draft_index,
                actual_seq_lengths_query=self.actual_seq_lengths_query,
                actual_seq_lengths_key=self.actual_seq_lengths_key,
                spec_actual_seq_lengths_query=self.spec_actual_seq_lengths_query,
                spec_actual_seq_lengths_key=self.spec_actual_seq_lengths_key,
            )

        if get_ascend_config().c8_enable_reshape_optim:
            torch.ops._C_ascend.store_kv_block_metadata(
                slot_mapping,
                common_attn_metadata.group_len,
                common_attn_metadata.group_key_idx,
                common_attn_metadata.group_key_cache_idx,
                block_size,
            )

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            cum_query_lens=cum_query_lens,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            attn_mask=self.attn_mask_builder.get_attention_mask(common_attn_metadata.causal, self.model_config),
            attn_state=common_attn_metadata.attn_state,
            block_table=block_table,
            sin=sin[:num_input_tokens],
            cos=cos[:num_input_tokens],
            dsa_cp_context=dsa_cp_context,
            block_size=block_size,
            group_len=common_attn_metadata.group_len,
            group_key_idx=common_attn_metadata.group_key_idx,
            group_key_cache_idx=common_attn_metadata.group_key_cache_idx,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError("Currently we only support building dummy metadata for DecodeOnly state")

        attn_metadata.attn_state = attn_state
        return attn_metadata
