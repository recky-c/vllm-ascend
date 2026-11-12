# GLM component KV cache

This draft builds on recky-c/vllm-ascend#12 and vllm-project/vllm-ascend#21370.
It follows the allocation/view structure of #21083 at
`5597e6566547e204c9697c79286e0f0b5de8b401`, without merging that branch.

## Allocation and views

`init_kv_cache -> allocate_kv_cache_main -> unified_allocate_kv_cache`
allocates the descriptor-backed byte pool once. Layer regions use descriptor
`offset + layer_index * layer_stride`, with `block_stride` between pages.

`create_kv_cache_runtime_views -> backend.get_runtime_shapes/get_runtime_dtypes
-> create_cache_component_views` builds ordinary tuples of component tensors.
These functions live in the existing worker allocation module, following
#21083. Shapes and dtypes belong to the main/indexer backends, not a separate
GLM format dispatcher. Alignment and storage bounds remain checked.

Main pages retain `[all token NoPE][all token RoPE]`; packed C8/TurboQuant main
retains its packed token records. Indexer pages retain `[all K][all scale]`.
C4 keeps its scale grouping axes. Only the block dimension is strided.

Existing Ascend binding assigns these tuples directly. A small AttentionGroup
mapping adapter accepts SFA component tuples when manager/kernel block sizes
match; upstream Tensor mapping remains in place for Tensor caches. Remove the
adapter when upstream supports backend component views.

There is no GLMCacheComponents class, glm_cache_views.py module or new option.
Ordinary resident GLM SFA configurations use the new allocator automatically.
The entry-point scope predicate leaves Flash/mixed backends, CP, PD, KVPP,
offload, host cache and block-splitting configurations on their existing path.

## Cache writes

Indexer and packed main call `DeviceOperator.scatter_paged_cache`. Dense
caches retain the existing scatter path. Strided pages use byte views with
`reshape_and_cache -> npu_scatter_pa_kv_cache`, preserving the destination's
block stride. The single-component K/V alias convention follows the existing
PCP path; native correctness still requires hardware verification.

The existing fused INT8 indexer store remains enabled. Its addressing changes
from `slot * width` to `block * block_stride + token * width`, independently
for K and FP16 scales. Quantization and scale conversion stay unchanged.

## Validation status

The implementation and tests were rewritten for this structure. Per the
request to publish before validation, no tests, NPU runs, benchmarks or lint
checks were run for this revision. Previous draft checks do not validate this
rewrite. Before merge, validate A3/A5 narrow-scale byte writes, alias writes,
negative slots, full-backing isolation, graph replay, and model correctness
and performance for non-quantized, C8/C4 indexer and C8/TQ main combinations.
