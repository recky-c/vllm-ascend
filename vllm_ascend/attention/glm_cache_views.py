# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Experimental GLM page components over the planner's shared allocation."""

from math import prod

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    get_kv_cache_compression_ratio,
)


class GLMCacheComponents(tuple):
    """Kernel-ready components; the tag prevents upstream Tensor remapping.

    Manager and kernel block sizes are equal. This tuple owns no extra storage.
    """


def component_layout(spec):
    """Per-token shapes/dtypes preserving the existing page ABI."""
    if spec.num_kv_heads != 1 or get_kv_cache_compression_ratio(spec) != 1:
        raise ValueError("GLM views require one latent head without compression.")
    if isinstance(spec, AscendSFAIndexerCacheSpec):
        if spec.sfa_dcp_replicated_indexer_size != 1:
            raise ValueError("Replicated indexer pages are unsupported.")
        components = [((1, spec.head_size), spec.dtype)]
        if spec.scale_dim:
            shape = (1, spec.scale_dim)
            if spec.cache_sparse_li_c4:
                if spec.scale_dim % 2:
                    raise ValueError("C4 scale width must be even.")
                shape = (1, spec.scale_dim // 2, 2)
            components.append((shape, spec.scale_dtype))
        return components
    if not isinstance(spec, AscendMLAAttentionSpec) or spec.model_version is not None:
        raise ValueError("Only ordinary GLM main/indexer specs are supported.")
    if spec.uses_packed_sfa_main_cache:
        if spec.scale_dim or spec.dtype != torch.int8:
            raise ValueError("Packed main must include scales in its INT8 head width.")
        return [((1, spec.head_size), spec.dtype)]
    if spec.kv_lora_rank is None or spec.qk_rope_head_dim is None or spec.scale_dim:
        raise ValueError("Main spec must describe NoPE/RoPE explicitly.")
    if spec.kv_lora_rank + spec.qk_rope_head_dim != spec.head_size:
        raise ValueError("Main component widths do not match the page.")
    return [((1, spec.kv_lora_rank), spec.dtype), ((1, spec.qk_rope_head_dim), spec.dtype)]


def allocate_glm_cache_views(config, kernel_block_sizes, allocate):
    """Validate the whole config, allocate once, expose page component views.

    Page format: [all NoPE][all RoPE] or [all K][all scale]. Packed C8/TQ
    main retains per-token records. Descriptor placement fields are bytes.
    """
    if not config.kv_cache_tensors:
        return {}
    sizes = {descriptor.size for descriptor in config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("GLM descriptors must describe one shared backing.")
    specs = {}
    for group_id, group in enumerate(config.kv_cache_groups):
        if group.host_resident:
            raise ValueError("Host cache groups are unsupported.")
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if kernel_block_sizes[group_id] != spec.block_size:
                raise ValueError("GLM views do not support kernel block splitting.")
            if getattr(spec, "store_on_host", False):
                raise ValueError("Host cache pages are unsupported.")
            specs[name] = spec
    size = sizes.pop()
    views = []
    seen = set()
    for descriptor in config.kv_cache_tensors:
        if descriptor.host_resident or descriptor.layer_stride <= 0:
            raise ValueError("GLM views require device descriptors with positive layer stride.")
        blocks = config.num_blocks_of(descriptor)
        if blocks <= 0:
            raise ValueError("GLM cache requires at least one block.")
        for layer_index, name in enumerate(descriptor.layers):
            if name in seen or name not in specs:
                raise ValueError(f"Duplicate or unknown GLM owner: {name}")
            seen.add(name)
            spec = specs[name]
            components = component_layout(spec)
            page_bytes = sum(spec.block_size * prod(shape) * get_dtype_size(dtype) for shape, dtype in components)
            if page_bytes != spec.real_page_size_bytes or descriptor.block_stride < page_bytes:
                raise ValueError(f"GLM page geometry mismatch for {name}.")
            offset = descriptor.offset + layer_index * descriptor.layer_stride
            if offset < 0 or offset + (blocks - 1) * descriptor.block_stride + page_bytes > size:
                raise ValueError(f"GLM page exceeds backing for {name}.")
            owner_views = []
            for token_shape, dtype in components:
                item_size = get_dtype_size(dtype)
                if offset % item_size or descriptor.block_stride % item_size:
                    raise ValueError(f"Unaligned GLM component for {name}.")
                shape = (blocks, spec.block_size, *token_shape)
                strides = (descriptor.block_stride // item_size, *(prod(shape[i + 1 :]) for i in range(1, len(shape))))
                owner_views.append((shape, strides, offset, dtype))
                offset += spec.block_size * prod(token_shape) * item_size
            views.append((name, owner_views))
    if seen != set(specs):
        raise ValueError("GLM descriptors must cover every owner.")
    raw = allocate(size)
    if raw.dtype != torch.int8 or raw.ndim != 1 or raw.numel() != size or not raw.is_contiguous():
        raise ValueError("GLM allocator must return a contiguous INT8 backing.")
    result = {}
    for name, owner_views in views:
        components = []
        for shape, strides, offset, dtype in owner_views:
            item_size = get_dtype_size(dtype)
            # Typed anchor avoids requiring the entire backing size to be
            # divisible by each dtype. as_strided addresses the shared storage.
            anchor = raw[offset : offset + item_size].view(dtype)
            components.append(anchor.as_strided(shape, strides))
        result[name] = GLMCacheComponents(components)
    return result


def map_glm_cache_components(kv_caches, attn_groups, original_mapper):
    """Bridge until upstream mapping accepts backend component tuples."""
    groups = list(attn_groups)
    ready = {name: cache for name, cache in kv_caches.items() if isinstance(cache, GLMCacheComponents)}
    for group in groups:
        for name in group.layer_names:
            if name in ready and group.kernel_block_size not in (None, ready[name][0].shape[1]):
                raise ValueError("GLM components cannot be split into kernel blocks.")
    mapped = original_mapper({name: cache for name, cache in kv_caches.items() if name not in ready}, groups)
    mapped.update(ready)
    return mapped


def store_paged_components(components, updates, slots, scatter):
    """Copy current-token bytes through PA scatter, retaining cache strides.

    Norm requires equal K/V dtypes. Byte views preserve quantized K and scale
    bits without numeric conversion. Only current updates may be materialized.
    """
    if len(components) != len(updates) or len(components) not in (1, 2):
        raise ValueError("Paged store expects one or two matching components.")
    count = updates[0].shape[0]
    if slots.numel() != count:
        raise ValueError("Paged store requires one slot per update.")
    if count == 0:
        return
    destinations, sources = [], []
    for cache, update in zip(components, updates):
        if cache.dtype != update.dtype or update.shape[0] != count:
            raise ValueError("Paged store dtype/token mismatch.")
        width = prod(cache.shape[2:]) * cache.element_size()
        if prod(update.shape[1:]) * update.element_size() != width:
            raise ValueError("Paged store row width mismatch.")
        destination = cache.view(torch.int8).view(cache.shape[0], cache.shape[1], 1, width)
        if destination.stride(1) != width or destination.stride(-1) != 1:
            raise ValueError("Paged store requires dense rows inside each page.")
        destinations.append(destination)
        sources.append(update.contiguous().view(torch.int8).view(count, 1, width))
    # Existing PCP uses identical K/V updates and destinations for one cache.
    scatter(sources[0], sources[-1], destinations[0], destinations[-1], slots.reshape(-1))
