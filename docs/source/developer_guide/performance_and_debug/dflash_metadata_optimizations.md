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
and net throughput for this combined branch require separate measurement.
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

The combined branch on the new base has no full-model accuracy or performance
result. Python import checks using existing native extensions do not establish
compatibility with all operators in the new base, which includes renamed native
symbols. Build matching native extensions before deploying this revision.

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
