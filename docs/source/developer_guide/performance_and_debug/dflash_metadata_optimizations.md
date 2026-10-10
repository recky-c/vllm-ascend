# DFlash metadata preparation optimizations

This branch ports DFlash metadata changes onto vLLM Ascend commit
`27c0024a9ccd1afa727f4cd16affae69bf4b4761`, paired with vLLM commit
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`.

## Included changes

- Reuse request metadata within one MRV2 target or draft build. Cache scopes end
  with that build; physical block tables, slot mappings, and GDN state indices
  remain specific to each cache group.
- Pass the prepared DFlash metadata through FULL graph execution without an
  additional build, adapting
  [bfa47193](https://github.com/Tame21/vllm-ascend/commit/bfa47193a623085ee4b171402e72ac941b706ac1).
- Reuse compatible dense/GDN metadata and compute aligned Mamba state indices
  across groups, adapting
  [vLLM Ascend PR #18139](https://github.com/vllm-project/vllm-ascend/pull/18139).
  Captured buffers preserve their addresses and refresh physical group mappings.
- Offer an experimental fixed maximum KV group width. Full attention, sliding
  window attention, and Mamba specifications remain in separate semantic buckets.
  The existing allocator determines storage offsets and state strides.

The port preserves the base revision's parallel attention, SFA offload variants,
TurboQuant, and host GDN paths. DCP/PCP builders and the host GDN builder retain
their original preparation contracts and do not opt into the new cross-group
reuse interfaces. Reuse therefore depends on the selected backend.

## Configuration

Both variables are defined centrally in `vllm_ascend/envs.py`, default to `0`,
and contain no sensitive information.

| Variable | Values | Behavior |
| --- | --- | --- |
| `VLLM_ASCEND_REUSE_BATCH_METADATA` | `0` or `1` | Enable the additional request-only FIA/GDN cache within a build. |
| `VLLM_ASCEND_KV_CACHE_GROUP_SIZE` | Nonnegative integer | `0` uses the existing grouping heuristic; a positive value sets a maximum group width in the generic uniform-page hybrid planner. |

Specialized model planners keep their existing behavior. Invalid variable values
raise `ValueError`. The reuse variable controls the additional request cache;
it does not disable the FULL handoff or the compatible metadata reuse adapted
from the changes above.

For an explicit width-five experiment, set these before starting the worker:

```bash
export VLLM_ASCEND_REUSE_BATCH_METADATA=1
export VLLM_ASCEND_KV_CACHE_GROUP_SIZE=5
```

Set `VLLM_ASCEND_KV_CACHE_GROUP_SIZE=0` and restart the worker to restore the
original group planner. Existing processes are unaffected by environment changes.

### Experimental draft FIA CPU length bounds

The additional configuration key `enable_dflash_draft_kv_optimistic_bound`
defaults to `false`. It adapts the intentionally optimistic GLM DSpark approach
in [PR #17571](https://github.com/vllm-project/vllm-ascend/pull/17571) to the
ordinary MRV2 Qwen3.5 DFlash draft query build:

```bash
--additional-config '{"enable_dflash_draft_kv_optimistic_bound": true}'
```

Merge this key into any existing additional configuration. Setting it to `false`
restores the original length source after restarting the worker.

For each request, upstream supplies
`U = min(target_cpu_upper_bound + num_query_per_req, max_model_len)`.
The draft FIA host `seq_lens_list` uses this per-request bound instead of
converting the rejection-corrected device length `E` with `.tolist()`. It never
replaces individual bounds with a batch maximum. Dummy graph rows use host
length one. The device `seq_lens` and `seq_lens_gpu`, KV write locations, RoPE,
sampling, target verification, and GDN state metadata retain their existing
accurate device inputs. GLM DSpark retains its existing behavior.

Enabling the option requires a real pure-decode proposal with dense Qwen3.5
target configuration (`qwen3_5` or `qwen3_5_text`), ordinary DFlash, no adaptive
verification, PP/PCP/DCP sizes one, and ordinary sink-free FIA builders. Only
active draft KV groups participate in the bound check; target GDN state pages
are not token-addressed draft KV. Every bound must fit the filled host block
allocation ledger for its request and draft group. Table width is not evidence
that blocks are allocated. Missing or unsupported CPU/allocation metadata falls
back to accurate device lengths without changing them.

The shared hybrid pool can recycle FP32 GDN state bytes into BF16 draft KV;
finite FP32 bytes may decode as BF16 NaNs. Enabling optimistic lengths therefore
also requires complete MRV2 new-block zeroing coverage. Initialization retains
the upstream Tensor-cache zeroer and adds an Ascend tuple K/V zeroer for target
and draft Full/SWA views, including virtual subblocks and physical page strides.
Aliased segments retain the widest payload without clearing page padding.
Only scheduler-provided newly allocated global block IDs are cleared, before
forward writes and prefix copies. Previously valid blocks are preserved.
Missing coverage or unsupported geometry keeps the original zeroer and exact
FIA lengths. Disabling the option preserves the original zeroing behavior.
The additional zeroing work is included in the unmeasured performance tradeoff.

Capture, dummy/profiling runs, prefill batches, standalone graph updates outside
a live proposal, and unsupported configurations keep the original exact path.
During a live FULL graph proposal, the existing metadata handoff and graph
parameter provider carry the same optimistic host list into replay. A fresh
build caused by a shape mismatch uses the same scoped policy. Scope is isolated
per execution context and reset after exceptions.

This option changes draft attention: rejected or stale KV positions in `[E,U)`
may become visible, especially in noncausal Full attention, and causal/SWA
alignment can also shift. It deliberately trades some draft acceptance for
less metadata synchronization; there is no exact device mask for that interval.
Target verification remains exact. Full-model output equivalence, acceptance,
and net throughput for this optimistic mode have not been measured.
The option does not remove the runner's earlier corrected-count D2H wait or
the target FIA readback, and has no guaranteed performance gain.

The adaptation passed 537 related CPU unit tests, including a real mixed-group
DFlash producer, FULL handoff, MRV2 zeroer initialization, and unsupported
zeroing-geometry fallback. Eight native FIA/zeroing component tests passed,
including noncausal
Full attention and causal sliding-window attention on an isolated NPU. They
exercise heterogeneous lengths, padding, page boundaries, model-length limits,
an 8K SWA context with recycled prefix pages, and the actual UpdatableGraph
parameter provider. Additional poisoned-recycle tests clear finite-FP32 bytes
that decode as BF16 NaNs, retain old valid blocks and shared-page padding, and
check finite Full/SWA graph output after clearing only new blocks.
Replay output is checked against eager attention using
the same upper bounds, not against exact attention. These component results
do not establish full-model accuracy, draft acceptance, or throughput.

## Validation and limitations

The port passed 608 CPU unit tests, including width-zero/width-five planner output
fed into the actual GDN metadata builder, changing batch sizes and physical block
IDs, graph buffer lifetime checks, and DCP runtime/capture/context fallback.
That planner integration test uses `mamba_cache_mode="none"`; it does not execute
the aligned-state Triton kernel.

A further 150 NPU metadata tests passed on the new base: aligned-state Triton
calculations, physical group mappings, changing inputs, and graph capture/replay.
These tests cover metadata preparation, not a complete model forward pass.

Whole-model checks of the later exact-metadata increment are described below.
They used this environment's existing runtime resources and process-local import
bootstrap. These runs establish compatibility for the exercised model paths in
that environment, rather than every new-base operator or native-build/ABI
combination. Other environments need matching native resources.

Earlier experiments on the old base are separate evidence, not measurements of
this combined port. In a Qwen3.5-9B DFlash TP2 width-five experiment, 38 KV groups
became 9, with 7 padding layer slots. Compared with the original group heuristic,
measured throughput increased by 49.16% at batch size 1 and 41.12% at batch size 4,
while KV capacity decreased by 36.88%. Only 8 of 10 greedy requests matched token
for token, so the strict accuracy gate failed. Width five remains an explicit
experimental option and must be evaluated for each workload.

## External runtime prerequisites

The earlier NPU tests used CANN 9.2 and an external FLA package with 64-bit Mamba
state address multiplication. In that package's `recurrent_gated_delta_rule.h`,
the state offset calculation must widen before multiplying, for example:

```cpp
static_cast<uint64_t>(stateStride0_) * ssmStateIndicesGm_.GetValue(seq_i)
```

This avoids overflowing a 32-bit intermediate for large physical state indices.
The external header, rebuilt FLA binaries, and CANN packages are not included in
this branch. Metadata reuse and fixed-width grouping do not replace that runtime
fix. Machine-specific launch scripts and the separate target-CPU-length
experiment are also excluded.

## PR18139 exact metadata and graph preparation increment

`enable_dflash_exact_metadata_optimizations` defaults to `true`. For ordinary
MRV2 DFlash FIA with PP/PCP/DCP equal to one, live pure-decode proposals copy the
current rejection-corrected device lengths **E** to a persistent pinned CPU
buffer once. The copy is queued on a dedicated stream after the final global
draft KV group writes the shared length buffer, before context KV work. Host
metadata waits for the copy event, then shares the exact mirror across compatible
FIA builders. Device length tensors and target attention retain their exact
contracts. Inactive padded requests use a benign FIA host length of one.

This retains one device-to-host copy and its event synchronization. It does not
remove the target runner's corrected-count readback. The earlier implementation
already shares FIA lists within each build, so this increment must not be
interpreted as reducing one synchronization per layer to one per proposal.

Supported FULL layouts reuse a template across proposals. The template key
includes live and padded shape, query width, step, group causality, builder/KV
specification and input layout. Every proposal refreshes exact host lists, device
length views, global-group physical block tables and slot mappings; mutable
per-step scratch and reshape events are cleared. Shrinking or growing the live
batch and changes in layout force a fresh build. Capture, dummy/profile runs,
prefill, adaptive verification, specialized/sink backends and context parallel
retain their original metadata path.

The separate `enable_dflash_deferred_metadata` control defaults to `false`.
The default prepares host metadata before replay and allocates no neutral KV
resources. When explicitly enabled, a compatible `UpdatableGraph` may queue its
independent query prefix before host metadata preparation. The update stream's
input dependency is recorded **before**
replay is queued, and every FIA task remains behind its external update event.
Only FIA-only graphs with preallocated independent neutral K/V and block-table
parameters may use this order. The first build and layout changes use the normal
metadata-first order. If host preparation fails after replay, all waiting FIA
tasks receive legal neutral parameters, the graph drains, caches are invalidated
and the original exception is raised. Neutral resources remain owned by the
graph; they add one zero KV page per captured FIA task and a separate block table.
Neutral KV supports only ND format, dense inner dimensions and a positive outer
page stride at least as large as the payload. Independent zero backing preserves
the live strides without changing live K/V or physical block tables. Packed,
overlapping or unknown geometry and other task kinds retain the normal order.

The deferred option remains experimental. Three historical native comparisons
observed intermittent noncausal Full graph-versus-first-functional differences
in BF16, including a failure after all task updates were submitted before any
external event was released. The responsible output and root cause remain
unclassified. Repeated passing runs and exact model tokens do not resolve this
component gate. Separating deferred execution from the default is a release
scope decision, not a numerical fix.

The previous `enable_dflash_draft_kv_optimistic_bound=true` option takes priority
and disables the new exact mirror, template and graph-overlap path together.
Its optimistic U experiment and new-block zeroer retain their prior behavior.
To use the original exact metadata path, set:

```bash
--additional-config '{"enable_dflash_exact_metadata_optimizations": false}'
```

`enable_gdn_graph_state_batching` is independently enabled by default. It batches
compatible MRV2 GDN graph-state copy, fill and reset operations; it does not
require aligned prefix caching and is not disabled by the optimistic flag.
Unsupported states/builders keep the per-group path. To disable both increments:

```bash
--additional-config '{"enable_dflash_exact_metadata_optimizations": false, "enable_gdn_graph_state_batching": false}'
```

The DFlash component test runs the real rejection/input producer, last-group
snapshot callback, metadata producer/builders, graph manager and external events.
It compares four rounds with different exact lengths and physical tables,
including shrink/growth, against eager attention with the same E for noncausal
Full and causal sparse-mode-4 SWA with window 4096. It also checks finite neutral
exception recovery, unchanged live KV, and recovery of the next proposal.
The small upstream graph-dispatch entry is replaced by `graph.replay` in this
component fixture; no model weights are loaded. Whole-model accuracy, draft
acceptance and throughput are separate validations and are not established by
these component comparisons.

The final default separation passed 734 related CPU tests and 12 native tests on
an isolated NPU, including the previous optimistic-bound and new-block-zeroer
regressions. Five fixed independent processes also passed the default
metadata-first four-round comparison without a diagnostic or resolver observer.
Explicit deferred fixtures retain prefix/neutral rescue coverage; they do not
dismiss the historical numerical failures. A fresh prefix sentinel, read on an
independent stream before FIA
parameter updates, proves current replay progress; an event completed during
capture alone is insufficient evidence. This test-only probe adds no polling to
the implementation. The exact mirror and overlap follow the fixed PR18139
[`ed96262`](https://github.com/vllm-project/vllm-ascend/commit/ed96262a41b4e44aade00a8888eac30abd50806c)
and [`f2c9582`](https://github.com/vllm-project/vllm-ascend/commit/f2c95825917da6f08f52b4f0adbe1cdd07609e6c)
increments, adapted to the current upstream uniform metadata producer.

## Whole-model validation of the default configuration

An independent Qwen3.5-9B DFlash TP1, K3, FULL-graph check compared published
`78349326e` with the increment's default configuration. Nine target and nine
draft graph sizes captured successfully. Across nine phases, all 27 requests and
4116 generated tokens matched exactly, as did each phase's speculative acceptance
counter deltas. Batch growth/shrink and long contexts were included. Two extra
batch-two phases with context lengths 8186 then 4094 brought the checked run to
31 requests and 4628 tokens.

One idle-only counter flush observed 519 exact snapshots and 490 template
refreshes, with zero early replays and no neutral resources owned by the three
used graph descriptors. The target's 24 GDN builders were
`AscendGDNHostMetadataBuilder`, with block-table updates unsupported and no
graph-state updater, so GDN batching fell back. These results verify the default
DFlash mirror/template path; they establish no whole-model GDN batching gain.
The six draft groups were five causal SWA4096 groups and one noncausal Full
group, using BF16 ND strided K/V with physical blocks 640 and kernel blocks 128.

Separate uninstrumented increment-OFF and increment-ON service launches matched
all output tokens and per-phase acceptance deltas for 29 main requests and the
four-request 8K-to-4K turnover workload. Deferred execution stayed disabled.
Five steady batch-four samples each generated 732 completion tokens:

| Metric | Increment OFF mean | Increment ON mean | Change |
| --- | --- | --- | --- |
| Batch wall time | 3.29432 s | 3.22185 s | -2.20% |
| Framework mean request TPOT | 14.4667 ms | 14.1489 ms | -2.20% |

Wall-time standard deviations were 0.02153 s and 0.02789 s; TPOT standard
deviations were 0.08730 ms and 0.12128 ms. This is an observation from five
synthetic samples on one device. Wall time includes prefill, client and API
work; TPOT is the framework's per-request mean. It is not a production benchmark,
a formal model-accuracy suite or confirmation of upstream performance claims.
Earlier timings from a wrapper that wrote per-proposal counters to a network
filesystem are excluded. The optimistic U mode remains unmeasured.

These model launches used CANN 9.2beta2, the existing corrected 64-bit FLA runtime
and process-local native-resource bootstrap with the installed
PyTorch/torch_npu/vLLM. No production package was replaced or native extension
rebuilt. This is a real model check in that environment, not a general native
build or ABI guarantee. The default results do not resolve the experimental
deferred path's historical numerical discrepancies.
