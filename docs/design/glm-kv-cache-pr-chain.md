# GLM KV cache refactoring PR chain

Scope: MRV2, GLM and GLM-5.3-Flash. Hardware validation is deferred at the request of the author.

| PR | Dependency | Review boundary |
| --- | --- | --- |
| upstream #21370 | main | Owners produce MLA/indexer specs; collection stops rebuilding specs. |
| fork #11 | #21370 | Physical state bytes, including indexer scales and Flash tail state. |
| fork #12 | #11 | Page alignment belongs to planning. |
| fork #13 | #12 | Flash circular owner, sparse roles, physical config builder. |
| fork #14 | #13 | Flash descriptor-aware owner runtime views and page-strided writes. |
| fork #15 | #14 | Standard Flash packed planning and one shared backing when block-outermost layout is selected. |
| fork #16 | #15 | Ordinary GLM indexer kernel-page K/scale layout, owner views, strided writes/reads and transfer metadata. |
| current PR | #16 | Upstream token-fused MLA descriptor views and token-strided component stores; ordinary MLA activation remains a separate atomic migration. |

Eight PRs have been prepared in this chain. This count describes the current delivery, not completion of the three analysis documents.

## Indexer contract (#16)

Each ordinary GLM indexer manager block contains consecutive kernel pages, including DCP replicas. Each kernel page stores K followed by scales. The owner creates aliasing views with equal byte strides and distinct offsets. The allocator supplies an explicit block count and one byte region per layer; it does not split K and scales. KVPP uses one physical indexer component. Layerwise shared slots can retain different capacities without inferring logical blocks from backing size.

Stores address pages explicitly, including the fused native INT8 store. Quantized reads use QLI V2 with explicit K and scale strides: INT8 mode 2, FP8 mode 1, and C4 mode 5. C4 requires the CANN candidate entry point with candidate filtering disabled, which carries both strides and packed dtype handling. Existing installs must include the matching operator build. The nonquantized layout remains dense.

MooncakePullConnector and MooncakePushConnector retain separate K and scale transfer records. Each record publishes its own content length, true stride and physical-to-manager block scale; they are not collapsed into a misleading single K-shaped record. DCP block ID expansion is unchanged.

## Token-fused MLA contract (current PR)

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

This PR prepares the view/store primitives. Existing Flash NoPE-only records have
zero RoPE width, so their byte ordering is unchanged. Ordinary GLM main MLA is
not routed into these descriptor views yet. Before enabling it, fused MLAPO and
PROLOG writers, attention readers, PCP/DCP, host offload and transfer consumers
must agree on this layout in the same enabling PR. A partial native-write fallback
is insufficient because fused weight preparation can discard the original weights.
The unfinished default Flash planner replacement is also excluded from this PR.

This decision supersedes the earlier requirement to preserve component-major
main MLA bytes. It is recorded in all three local analysis documents on 2026-11-10.
The upstream reference is vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58`,
whose MLA cache kernel writes NoPE then RoPE within each token record.

## Compatibility boundaries and follow-up validation

This is a draft code chain, not a claim that every target in the analysis documents is enabled or verified:

- Flash packed planning is selected by the existing block-outermost layout setting. Its legacy two-pool path remains. Packed Flash currently requires 128-token blocks and does not support KVPP or host offload.
- Ordinary indexer page layout is selected for GLM with no connector or a modern Mooncake Pull/Push connector and no sparse host offload. Other connectors and host offload retain their existing pool ABI. Their wire/copy contracts require a separate migration before enabling page layout.
- Ordinary GLM main MLA allocation has not been converted to the Flash shared-backing path. This PR does not claim that the entire generic allocation migration is complete.
- Tests were added but not run. Before merging, validate byte/alias boundaries, graph replay stores, INT8/FP8/C4 QLI reads, DCP/PCP, KVPP, Mooncake cross-rank transfers and GLM model outputs on the matching Ascend runtimes.
- Switching the default Flash layout and deleting compatibility branches should follow that validation. These follow-ups are not hidden inside the PR count above.
