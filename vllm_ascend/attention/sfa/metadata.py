from dataclasses import dataclass
from typing import NamedTuple, TypeVar

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import AscendPCPMetadata


class DCPQueryGatherContext(NamedTuple):
    """State needed to finish the async fused DCP query all-gather."""

    # The gathered fused query tensor: cat([ql_nope, q_pe], dim=-1).
    gathered: torch.Tensor
    # Async all-gather work handle. None means the gather completed synchronously.
    handle: torch.distributed.Work | None
    # Permutation that restores the original dimension order after dim>0 gather.
    restore_perm: tuple[int, ...] | None
    # Last-dimension sizes used to split the fused query back into ql_nope/q_pe.
    ql_nope_dim: int
    q_pe_dim: int


@dataclass
class DCPContext:
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_gather_context: DCPQueryGatherContext | None = None


@dataclass
class DSACPContext:
    num_tokens: int
    num_tokens_pad: int
    local_start: int
    local_end: int
    local_end_with_pad: int
    slot_mapping_cp: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_key: torch.Tensor


@dataclass
class AscendSFAMetadata:
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|
    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    cum_query_lens: torch.Tensor
    block_table: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.
    # The dimension of the attention heads
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    dcp_context: DCPContext | None = None
    dsa_cp_context: DSACPContext | None = None
    reshape_cache_event: torch.npu.Event = None
    sfa_cp_metadata: AscendPCPMetadata | None = None
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    block_size: int = 0
    group_len: torch.Tensor | None = None
    group_key_idx: torch.Tensor | None = None
    group_key_cache_idx: torch.Tensor | None = None


M = TypeVar("M", bound=AscendSFAMetadata)
