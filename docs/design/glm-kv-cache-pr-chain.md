# GLM KV cache refactoring PR chain

Scope: MRV2, GLM and GLM-5.3-Flash. Hardware validation is deferred at the request of the author.

| PR | Dependency | Review boundary |
| --- | --- | --- |
| upstream #21370 | main | Owners produce MLA/indexer specs; collection stops rebuilding specs. |
| fork #11 | #21370 | Physical state bytes, including indexer scales and Flash tail state. |
| fork #12 | #11 | Page alignment belongs to planning. |
| fork #13 | #12 | Flash circular owner, sparse roles and Core block-count materialization; retain dedicated grouping. |
| fork #15 | #13 | Descriptor-aware owner views and page-strided writes for the existing Flash two-pool layout; includes former #14. Generic grouping and packed allocation are withdrawn. |
| fork #16 | #15 | Ordinary GLM indexer kernel-page K/scale layout, owner views, strided writes/reads and transfer metadata. |
| fork #17 (WIP) | #16 | Complete ordinary GLM main MLA token-fused migration, including allocation and all affected consumers. Only view/store primitives are implemented so far. |

Eight PRs were originally created; seven remain open drafts. #14 is superseded
by #15. The original commits are retained: #15 appends a withdrawal of its packed
planning/allocation commit, rather than rewriting the earlier changes.

## Current scope decision: defer GLM Flash grouping unification

`get_glm5_next_kv_cache_groups` keeps its dedicated model-specific path. Unifying
it with upstream generic grouping is deferred and is not a prerequisite for the
remaining Spec, planning and allocation work.

- #12 retains page alignment in planning, including necessary dispatch to model-specific planners.
- #13 retains explicit owner roles and materialization using Core's resolved block count. It does not replace Flash grouping.
- #15 retains owner views that consume descriptor offsets, strides and explicit block counts from the existing two-pool plan.
- #15 withdraws the block-outermost opt-in to upstream grouping, packed byte-cost accounting and one-shared-backing allocation together. The allocator compatibility module introduced for that path is also withdrawn.
- #16 continues ordinary GLM indexer migration; it uses the paged-store primitive retained in #15, not Flash packed planning.
- #17 remains the WIP ordinary GLM main MLA migration. Flash grouping unification must not be introduced as its prerequisite.

The dedicated Flash grouping and existing two-pool allocation can remain in
place. Default Flash layout switching, removal of the dedicated grouping path,
and its coupled packed allocation are deferred work, not missing requirements
for this active chain.

#17 must stay draft until its allocation and affected consumers are complete.
The seven open review boundaries do not mean that all remaining ordinary GLM
migration work has been completed.

## Indexer contract (#16)

Each ordinary GLM indexer manager block contains consecutive kernel pages, including DCP replicas. Each kernel page stores K followed by scales. The owner creates aliasing views with equal byte strides and distinct offsets. The allocator supplies an explicit block count and one byte region per layer; it does not split K and scales. KVPP uses one physical indexer component. Layerwise shared slots can retain different capacities without inferring logical blocks from backing size.

Stores address pages explicitly, including the fused native INT8 store. Quantized reads use QLI V2 with explicit K and scale strides: INT8 mode 2, FP8 mode 1, and C4 mode 5. C4 requires the CANN candidate entry point with candidate filtering disabled, which carries both strides and packed dtype handling. Existing installs must include the matching operator build. The nonquantized layout remains dense.

MooncakePullConnector and MooncakePushConnector retain separate K and scale transfer records. Each record publishes its own content length, true stride and physical-to-manager block scale; they are not collapsed into a misleading single K-shaped record. DCP block ID expansion is unchanged.

## Token-fused MLA contract (#17, work in progress)

The nonquantized main MLA descriptor view follows upstream's per-token layout:
`[token 0 NoPE | RoPE][token 1 NoPE | RoPE]...` within each block.
The Ascend owner exposes two aliases of the same record. Their token stride is
`NoPE width + RoPE width`; slicing must not turn either alias into a dense pool.
Descriptor offsets and physical block strides remain authoritative. Dense manager
pages can split into kernel pages; splitting across interleaved owners remains rejected.
Packed quantized cache formats retain their existing ABI.

The generic paged store accepts these token-strided component aliases and checks
that each component is contiguous within a token. It addresses block and token
strides separately, preserving the other component and descriptor padding.

The current implementation only prepares the view/store primitives. Existing Flash NoPE-only records have
zero RoPE width, so their byte ordering is unchanged. Ordinary GLM main MLA is
not routed into these descriptor views yet. Before enabling it, fused MLAPO and
PROLOG writers, attention readers, PCP/DCP, host offload and transfer consumers
must agree on this layout within #17 before that PR is ready to merge. A partial native-write fallback
is insufficient because fused weight preparation can discard the original weights.
Flash grouping unification and the coupled packed allocator are explicitly deferred outside this PR.

This decision supersedes the earlier requirement to preserve component-major
main MLA bytes. It is recorded in all three local analysis documents on 2026-11-10.
The upstream reference is vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58`,
whose MLA cache kernel writes NoPE then RoPE within each token record.

## Compatibility boundaries and follow-up validation

This is a draft code chain, not a claim that every target in the analysis documents is enabled or verified:

- Flash retains its dedicated grouping and two-pool allocation. This chain no longer introduces the block-outermost packed opt-in or its additional block-size/KVPP/offload restrictions.
- Ordinary indexer page layout is selected for GLM with no connector or a modern Mooncake Pull/Push connector and no sparse host offload. Other connectors and host offload retain their existing pool ABI. Their wire/copy contracts require a separate migration before enabling page layout.
- Ordinary GLM main MLA allocation has not been converted to the Flash shared-backing path. This PR does not claim that the entire generic allocation migration is complete.
- Tests were added but not run. Before merging, validate byte/alias boundaries, graph replay stores, INT8/FP8/C4 QLI reads, DCP/PCP, KVPP, Mooncake cross-rank transfers and GLM model outputs on the matching Ascend runtimes.
- Flash grouping unification and coupled default-layout changes are deferred by scope decision, independently of validation. They are not prerequisites for the active chain.
