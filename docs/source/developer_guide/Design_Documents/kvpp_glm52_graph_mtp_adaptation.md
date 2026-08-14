# KVPP with GLM-5.2 Graph and MTP Adaptation Analysis

## Decision

The current development baseline is:

- vLLM-Ascend `acbd2bb28` from `main`.
- vLLM `58d3918e3`, as selected by
  `.github/vllm-main-verified.commit` on that vLLM-Ascend baseline.
- GLM-5.2 SFA graph support from upstream commit `b7bdfd59f` (PR #13958),
  already included in the baseline.
- GLM-5.2 SFA MTP support from commit `de26a0fba` (PR #14105), applied on
  top because it has not yet landed in `main`.

SP and DSA-CP are deferred. DSA-CP depends on the GLM SP/FlashComm path and
cannot be accepted as usable while that prerequisite is not functionally
stable. The KVPP branch therefore keeps DCP rejected, makes no DSA-CP support
claim, and removes the local MC2/SP behavior change. The generic SFA
Main/Indexer bundle staging fix remains: it prevents two cache tensors from
using overlapping MemFabric staging addresses and is required independently
of DSA-CP.

Upstream graph and MTP support means that GLM-5.2 can use each feature without
KVPP. The KVPP integration supports fixed-step MTP in eager mode. Graph mode
remains outside the KVPP support boundary.

## Current KVPP boundary

KVPP has three responsibilities:

1. vLLM assigns persistent cache ownership and two alternating scratch
   buffers.
2. the model runner builds one batch/forward transfer session;
3. SFA/MLA attention calls `enter_layer`, `wait_for_layer`, and `leave_layer`
   so a host-side scheduler can overlap MemFabric MTE transfer with projection
   and layer compute.

This boundary works for one eager target-model traversal. MTP draft forwards
remain outside the KVPP lifecycle by keeping their caches replicated locally.
Graph replay still removes the Python layer callbacks and remains a separate
integration gap.

## MTP adaptation

### Upstream capability now present

PR #14105 teaches the Ascend autoregressive speculator to recognize SFA and
SFA Indexer backends, lets SFA own its draft-step metadata, and adds an eager
GLM-5.2 MTP end-to-end case. It is a prerequisite, not the KVPP integration.

### Implemented design

Replicate the small MTP draft cache on every KVPP rank and layer-split only the
target model cache. MTP cache layers are identified by their model-layer range:
they begin at `num_hidden_layers` and extend for
`num_nextn_predict_layers`. This works whether Target and MTP caches share one
logical cache group or use separate groups; `is_eagle_group` is not required.

This design is preferable to transferring the draft cache on every speculative
step:

- the target model still executes one ordered transformer-layer traversal, so
  the existing two-scratch prefetch schedule remains valid;
- draft generation can perform multiple local forwards without re-entering the
  target KVPP scheduler;
- one small replicated draft cache costs much less memory than replicating the
  target cache and avoids communication in the latency-sensitive draft loop.

### Integration rules

1. Exclude MTP layer caches from KVPP owner partitioning and scratch aliasing.
   Allocate those layers normally on every rank.
2. Store owners only for layer-split target cache layers. Do not require an
   owner for replicated draft layers.
3. In the Ascend model runner, inject the KVPP hook only into target SFA/MLA
   implementations. The speculator's draft implementation must not share the
   target scheduler.
4. Use the one cache group containing KVPP-managed Target layers to select the
   block table and block size. Other groups may contain only replicated MTP
   caches.
5. Build active pages from the post-rollback computed length plus all currently
   scheduled Target verification tokens, including block-boundary crossings.
6. Allow only fixed-step `method="mtp"`. Keep other speculative methods and
   dynamic speculative decoding rejected initially.
7. Validate rejection/rollback semantics: rejected KV entries may remain in a
   page, but the next accepted write must overwrite the same logical slots and
   every owner transfer must expose an identical history to non-owners.

### Test matrix

- eager GLM-5.2, MTP off/on, KVPP off/on;
- one and three speculative tokens;
- sequence lengths immediately before and across a KV block boundary;
- all-accepted, partially accepted, and all-rejected draft batches;
- mixed prefill/decode batches and chunked prefill;
- exact greedy token comparison plus draft/acceptance metrics;
- cache-placement unit tests proving draft groups are replicated and excluded
  from the target execution plan.

Estimated implementation effort: about 2 to 3 person-days, plus 1 day of
multi-card correctness and performance validation.

## Graph-mode adaptation

### Upstream capability now present

PR #13958 adds graph metadata buffers for SFA, graph capture support for the
SFA and Indexer backends, and selects an executable backend when updating full
graph parameters. This makes ordinary GLM-5.2 SFA graph execution possible.

### Why KVPP cannot simply remove the eager gate

The current KVPP schedule is driven by Python callbacks inside each attention
forward. During ACL graph capture those callbacks run only while capturing;
during replay the host code does not run again. MemFabric ready/done messages,
the background transfer future, and scratch-buffer rotation would therefore
not be issued per replay.

Preloading all non-owned layers before replay is also incompatible with the
current memory design: two scratch buffers cannot simultaneously retain all
non-owned layers for a full-model graph.

### Recommended design

Introduce explicit graph boundaries around the layerwise KVPP rendezvous. The
host scheduler must remain outside replay, while graph-captured compute runs in
segments that preserve these boundaries:

1. host starts or completes transfer for layer N;
2. replay the graph segment that consumes layer N and computes the overlap
   window;
3. rotate scratch and start layer N+1;
4. continue with the next segment.

This requires either supported per-layer/breakable ACL graph segments or a
graph-safe transport operator whose inter-rank synchronization is replayable.
The segmented approach is the safer first implementation because the current
MemFabric control plane is host-driven.

### Required changes

1. Separate KVPP lifecycle control from attention `forward` into an explicit
   execution-session API usable by eager execution and graph-segment replay.
2. Add capture state: capture must never send ready/done messages or perform
   a real remote transfer. It should only establish fixed tensor addresses and
   event dependencies.
3. Establish stable scratch tensor addresses for every captured segment and
   prove that alternating buffers cannot be overwritten before the previous
   segment finishes consuming them.
4. Teach the graph manager to invoke the KVPP session between replay segments,
   including abort/error handling across all ranks.
5. Keep dynamic batch metadata outside the captured graph. Update active pages,
   block tables, slot mappings, and valid masks before every replay without a
   device-to-host synchronization.
6. Relax the eager-only platform gate only for the exact graph mode that has a
   segmented implementation. Continue rejecting unsupported full-model graph
   modes explicitly.

### Test matrix

- capture must perform zero MemFabric network transfers;
- repeated replay must transfer once per target layer per batch;
- eager versus graph greedy-token equality;
- capture sizes at minimum, maximum, and padded batch shapes;
- alternating scratch reuse over at least four consecutive layers;
- mixed prefill/decode, chunked prefill, and failure/abort recovery;
- TTFT and inter-token latency compared with eager KVPP and graph without KVPP.

Estimated implementation effort: 4 to 6 person-days if usable segmented ACL
graph hooks already exist; otherwise 7 to 10 person-days because a graph-safe
execution boundary must first be added.

## Recommended order

1. Stabilize the updated baseline with eager KVPP and no speculative decoding.
2. Implement MTP by replicating draft cache groups and validate eager mode.
3. Refactor KVPP into an explicit multi-forward execution-session API.
4. Implement graph segmentation and graph-aware capture/replay lifecycle.
5. Validate MTP plus graph only after both combinations pass independently.
6. Revisit SP and DSA-CP after the upstream GLM SP path is independently
   functional; do not couple that work to MTP or graph enablement.
