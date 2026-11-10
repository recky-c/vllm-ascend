# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared Ascend page planning and cache-config metadata helpers."""

from dataclasses import replace

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec


def _align_hybrid_attention_page_sizes(kv_cache_specs: dict[str, KVCacheSpec]) -> None:
    """Align dense K/V segments before padding a shared Attention/Mamba pool."""
    # MLA and compressed-cache subclasses manage their own storage layouts.
    attention_specs = {
        name: spec for name, spec in kv_cache_specs.items() if type(spec) in (FullAttentionSpec, SlidingWindowSpec)
    }
    if not attention_specs:
        return
    reference = max(attention_specs.values(), key=lambda spec: spec.real_page_size_bytes)
    page_size = reference.real_page_size_bytes
    for layer_name, spec in attention_specs.items():
        # Ascend packs each layer as [all K blocks][all V blocks]. Equal
        # padded pages alone do not isolate block IDs: a smaller SWA K segment
        # can overlap another request's full-attention V block. Grow scheduler
        # blocks before padding; the backend still splits them into kernel blocks.
        if (
            page_size % spec.real_page_size_bytes
            or spec.head_size * reference.head_size_v != reference.head_size * spec.head_size_v
        ):
            raise ValueError(
                f"Cannot align hybrid attention K/V pages for {layer_name}: "
                "the shared contiguous cache requires matching K/V size ratios "
                "and divisible unpadded page sizes."
            )
        ratio = page_size // spec.real_page_size_bytes
        if ratio > 1:
            kv_cache_specs[layer_name] = replace(
                spec,
                block_size=spec.block_size * ratio,
                page_size_padded=max(spec.page_size_padded, page_size) if spec.page_size_padded is not None else None,
            )


def align_hybrid_kv_cache_specs(
    kv_cache_specs: dict[str, KVCacheSpec],
    *,
    attention_layer_names: set[str] | None = None,
    descriptor_native: bool = False,
) -> dict[str, KVCacheSpec]:
    """Plan shared pages without changing the cache owners' input specs.

    Model planners select the owners participating in their shared page class.
    Descriptor-native layouts pad attention to the state page, while legacy
    contiguous K/V layouts first align dense K/V segments and then pad both
    attention and state pages to the largest existing page.
    """
    mamba_specs = {name: spec for name, spec in kv_cache_specs.items() if isinstance(spec, MambaSpec)}
    if not mamba_specs:
        return kv_cache_specs

    specs = dict(kv_cache_specs)
    if attention_layer_names is None:
        attention_layer_names = {
            name
            for name, spec in specs.items()
            if isinstance(spec, AttentionSpec) and not isinstance(spec, AscendSFAIndexerCacheSpec)
        }
    if descriptor_native:
        common_page_size = max(spec.page_size_bytes for spec in mamba_specs.values())
    else:
        _align_hybrid_attention_page_sizes(specs)
        common_page_size = max(spec.page_size_bytes for spec in specs.values())

    for name in attention_layer_names:
        spec = specs[name]
        if spec.page_size_bytes < common_page_size:
            specs[name] = replace(spec, page_size_padded=common_page_size)
    if not descriptor_native:
        for name, spec in mamba_specs.items():
            if spec.page_size_bytes < common_page_size:
                specs[name] = replace(spec, page_size_padded=common_page_size)
    return specs


def attach_kv_cache_config_metadata(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVCacheConfig:
    """Preserve PD and DSpark ownership through scheduler-side config copies."""
    kv_cache_config.kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None and speculative_config.method == "dspark":
        from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import resident_mla_context_group_ids

        # Upstream scheduler conversion keeps only the first spec in a uniform
        # group, losing draft residency when target and draft share block IDs.
        # Preserve the worker-derived ownership through that deepcopy. Remove
        # this bridge once upstream KVCacheConfig carries per-layer ownership.
        kv_cache_config.dspark_context_group_ids = resident_mla_context_group_ids(kv_cache_config.kv_cache_groups)
        draft_only_group_ids = []
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            specs = (
                tuple(group_spec.kv_cache_specs.values())
                if isinstance(group_spec, UniformTypeKVCacheSpecs)
                else (group_spec,)
            )
            # Keep mixed target/draft groups in the ordinary target lookup.
            # Empty PP-local layer names still carry the global group spec.
            if specs and all(
                isinstance(spec, MLAAttentionSpec) and spec.non_causal_multi_token_decode for spec in specs
            ):
                draft_only_group_ids.append(group_id)
        kv_cache_config.dspark_draft_only_group_ids = tuple(draft_only_group_ids)
    return kv_cache_config
