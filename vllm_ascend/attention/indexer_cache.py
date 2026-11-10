# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ordinary GLM indexer runtime views and page-strided reads."""

from importlib import import_module
from math import prod

import torch
import torch.nn.functional as F
from vllm.utils.torch_utils import get_dtype_size

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec, get_kv_cache_storage_head_size
from vllm_ascend.models.glm5next.cache_views import view_cache_component


def view_indexer_pages(
    backing: torch.Tensor,
    spec: AscendSFAIndexerCacheSpec,
    num_blocks: int,
    kernel_block_size: int,
) -> tuple[torch.Tensor, ...]:
    """Each manager block contains kernel pages (including DCP replicas).

    Every kernel page contains K followed by scale. Replication changes the
    number of physical pages, not slot numbering or the component ordering.
    The allocation boundary supplies the exact manager block count.
    """
    if kernel_block_size <= 0 or spec.block_size % kernel_block_size:
        raise ValueError("Indexer manager blocks must contain whole kernel pages")
    pieces = spec.block_size // kernel_block_size * spec.sfa_dcp_replicated_indexer_size
    page_stride = spec.page_size_bytes // pieces
    shape = (kernel_block_size, spec.num_kv_heads, get_kv_cache_storage_head_size(spec))
    key_bytes = prod(shape) * get_dtype_size(spec.dtype)
    keys = view_cache_component(
        backing, dtype=spec.dtype, num_blocks=num_blocks * pieces,
        shape=shape, offset=0, block_stride=page_stride,
    )
    if not spec.scale_dim:
        return (keys,)
    scale_shape = (kernel_block_size, spec.num_kv_heads, spec.scale_dim)
    if spec.cache_sparse_li_c4:
        scale_shape = (*scale_shape[:-1], spec.scale_dim // 2, 2)
    if key_bytes + prod(scale_shape) * get_dtype_size(spec.scale_dtype) != page_stride:
        raise ValueError("Indexer components must fill the physical kernel page")
    scales = view_cache_component(
        backing, dtype=spec.scale_dtype, num_blocks=num_blocks * pieces,
        shape=scale_shape, offset=key_bytes, block_stride=page_stride,
    )
    return keys, scales


def select_paged_indexer_topk(query, query_scale, query_shape, weights, caches, metadata, use_c4):
    """Use QLI V2's explicit key/scale strides; never densify the cache."""
    keys, scales = caches
    cu_seqlens_q = F.pad(metadata.actual_seq_lengths_query, (1, 0), value=0)
    seqused_k = metadata.actual_seq_lengths_key
    quant_mode = 2
    if use_c4:
        quant_mode = 5
    elif keys.dtype == torch.float8_e4m3fn:
        quant_mode = 1
    qli_metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=query_shape[1], num_heads_k=keys.shape[2], head_dim=query_shape[-1],
        topk=2048, quant_mode=quant_mode, cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k, batch_size=seqused_k.shape[0],
        max_seqlen_q=-1, max_seqlen_k=-1, layout_q="TND", layout_k="PA_BBND",
        mask_mode=3, cmp_ratio=1,
    )
    if use_c4:
        # The CANN wrapper reinterprets packed FP4/E8M0 ACL dtypes and passes
        # both strides explicitly. Mode 3 disables candidate filtering.
        operation = import_module("cann_ops_transformer.ops").quant_lightning_indexer_candidate
        query = query.view(*query_shape[:-1], query_shape[-1] // 2)
        weights = weights.float()
    else:
        operation = torch.ops._C_ascend.npu_quant_lightning_indexer_v2
        query = query.view(query_shape)
        query_scale = query_scale.view(query_shape[:-1])
        scales = scales.squeeze(-1)
        weights = weights.to(torch.float16 if quant_mode == 2 else torch.float32)
    outputs = operation(
        query, keys, weights, query_scale, scales, topk=2048, quant_mode=quant_mode,
        cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k, block_table=metadata.block_table,
        metadata=qli_metadata, layout_q="TND", layout_k="PA_BBND", mask_mode=3, cmp_ratio=1,
    )
    return outputs[0]
