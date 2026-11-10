# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Standard shared-backing allocation with the Ascend byte allocator callback."""

from inspect import signature

from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs, create_kv_cache_views
from vllm.v1.worker.utils import allocate_kv_cache as upstream_allocate_kv_cache


def allocate_descriptor_cache(config, device, layout, allocate):
    """Return backing and standard manager views, allocating the budget once.

    Older supported upstream versions have no allocation callback. The small
    compatibility branch still delegates all shape/stride work to upstream.
    """
    if not config.kv_cache_tensors:
        return None, {}
    sizes = {descriptor.size for descriptor in config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("Standard descriptors must describe one shared backing")
    backing = None

    def allocate_once(size):
        nonlocal backing
        if backing is not None:
            raise ValueError("Standard descriptor allocation requested more than one backing")
        backing = allocate(size)
        return backing

    if "allocate" in signature(upstream_allocate_kv_cache).parameters and config.num_blocks:
        views = upstream_allocate_kv_cache(config, device, layout, allocate=allocate_once)
        return backing, views
    raw = allocate_once(sizes.pop())
    specs = {
        name: group.kv_cache_spec.kv_cache_specs[name]
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) else group.kv_cache_spec
        for group in config.kv_cache_groups for name in group.layer_names
    }
    views = {}
    for descriptor in config.kv_cache_tensors:
        if not descriptor.layers:
            continue
        spec = specs[descriptor.layers[0]]
        if any(specs[name] != spec for name in descriptor.layers):
            raise ValueError("Each standard descriptor must contain homogeneous owner specs")
        if not spec.has_layer_views:
            views.update((name, raw) for name in descriptor.layers)
        elif config.num_blocks:
            manager_views = create_kv_cache_views(raw, spec, config.num_blocks, layout, descriptor)
            views.update(zip(descriptor.layers, manager_views))
    return raw, views
