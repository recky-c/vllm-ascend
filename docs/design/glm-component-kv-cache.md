# GLM component KV cache draft

This stacked draft builds on `codex/kv-page-planning` (recky-c/vllm-ascend#12),
which builds on vllm-project/vllm-ascend#21370. It does not depend on #21083.

## Enable for development

Pass `--additional-config '{"glm_component_kv_cache": true}'` to opt in.
Default allocation remains unchanged until hardware correctness is established.
This is an experimental implementation, not a claim of validated model support.

The initial route accepts ordinary `glm_moe_dsa` SFA main/indexer backends,
resident caches, no CP/PD/KVPP/offload/sharing, and equal manager/kernel block
sizes. Flash, mixed backends, compressed or replicated caches, host pages and
zero-layer-stride descriptors are rejected when this option is enabled.

## Allocation and binding

`init_kv_cache -> allocate_kv_cache_main -> allocate_glm_cache_views`
validates the full cache configuration and allocates one byte backing. Each
component's address is computed from descriptor offset, layer stride and block
stride; it does not infer block count from total allocation bytes.

Ordinary main pages retain `[all token NoPE][all token RoPE]`. Indexer pages
retain `[all token K][all token scales]`. Packed C8/TurboQuant main pages retain
their existing packed token records. All component inner dimensions are dense;
only the first dimension carries the physical page stride.

The returned `GLMCacheComponents` is a tuple tag without additional storage.
`map_glm_cache_components` passes these already kernel-ready views through and
delegates all other caches to the original upstream mapping function. Existing
Ascend `bind_kv_cache_to_layers` assigns the tuple directly to each owner.
Main and indexer remain separate owners.

## Writes

`AscendSFAIndexerBackend.forward -> write_cache -> store_paged_components`
disables the flattened fused Triton store only for the tagged views. The normal
quantization path converts scales to their cache dtype before writing.

`AscendSFAImpl._store_packed_kv_cache -> store_paged_components` handles packed
main records. Ordinary main continues using its existing native preprocessing
operator with NoPE/RoPE component views.

`store_paged_components -> DeviceOperator.reshape_and_cache ->
npu_scatter_pa_kv_cache(cache_mode="Norm")` reinterprets both components as
INT8 bytes because the Norm K/V interface requires matching dtypes. Historical
cache is never copied or flattened across blocks. Only current-token updates
may be made contiguous. The single-component case passes identical K/V inputs
and destinations, following the existing PCP convention.

## Review and validation status

Tests were added for descriptor placement, shared storage, component formats,
padding preservation, negative-slot reference behavior, mapping, and invalid
configuration. These tests have not been executed. Local Ruff checks/formatting
were performed during development; verification was stopped at the author's
request before publication.

Before enabling by default, validate real A3/A5 PA scatter behavior for INT8
byte views, narrow FP16 scale rows, aliased single-component destinations,
negative slots, and entire-backing writeback. CPU reference tests cannot prove
native kernel support. GLM 5.2/5.3 non-quantized, C8/C4 indexer and packed C8/TQ
main combinations still need end-to-end accuracy, graph replay and performance
checks. No new Ascend C kernel is introduced in this draft.
