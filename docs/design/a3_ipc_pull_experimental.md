# Experimental A3 IPC pull for KVPP

This development backend copies owner persistent KV directly into the consumer's
final local scratch. Attention continues to use local KV. It has no payload
staging allocation and does not permit arbitrary PyTorch allocations to be
exported: V1 cache allocation must use the explicit P2P pool.

The default backend remains MemFabric. This implementation is an experimental
correctness baseline, not a production performance recommendation.

On the tested 157 deployment, cumulative `HUGE_ONLY_P2P` allocation succeeded at
8 GiB and failed on the next 512 MiB allocation, although the generic P2P-HBM
query still reported free memory. Ordinary huge allocation reached the test's
16 GiB cap. This is a deployment observation, not an A3-wide hardware limit.
The IPC pool therefore defaults to a conservative 7 GiB software budget,
configured by `kvpp_ipc_pool_budget_bytes` (positive, at most the verified 8 GiB).
The worker limits automatic cache planning to this budget minus raw-allocation
alignment reserve. Explicit larger cache-memory requests are rejected; the pool
also rejects excessive block overrides before calling the driver allocator.

## Configuration

Use the existing `enable_kvpp` option together with:

```json
{
  "enable_kvpp": true,
  "kvpp_transport": "ipc_pull",
  "kvpp_ipc_kernel_library": "/absolute/path/to/libkv_copy.so",
  "kvpp_ipc_cores": 8
}
```

The library is built from the accompanying A3 validation project using the target
CANN AscendC toolchain. Its entry point is `kv_copy_launch`; its source and binary
hashes must be recorded together. The checked environment is Ascend910_9362,
CANN 9.1.0 and torch_npu 2.10.0.post4. The allocation pool uses private torch_npu
storage constructors, so other versions require explicit revalidation.

Use a supported eager MLA model and HCCS-connected devices on one machine.
Keep Expert Parallelism enabled for MoE validation and keep asynchronous
scheduling enabled. The backend rejects speculative decoding and connectors;
the V2 factory rejects IPC if no supported allocation pool is supplied.

## Ordering and lifetime

An engine epoch, forward number, layer/bundle, page-plan identity and allocation
epochs identify a transfer. Physical page positions stay unchanged. Slots are
the actual underlying allocation identities, including shared indexer/scale
views, rather than layer-number parity.

Each consumer publishes local readiness only after copying the whole bundle.
The owner can write its current-token KV only after every consumer has completed
the remote read. This matters when historical and new tokens share a tail page.
The owner read-lease release alone does not authorize allocator page recycling:
all other local readers and owners must also have finished.

At the next layer hook, the scheduler records the preceding layer's last cache
use on the compute stream. This includes submitted indexer/cache-load readers.
The next copy waits for both that slot's last-use event and forward metadata
readiness. Source mappings, allocation objects and descriptor tensors remain
referenced until drain completes device use. An early forward return drains the
prefetched layer before clearing the plan. Close first drains, then closes remote
imports, then exports, then permits the allocation pool to be freed.

## Deliberate limitations

The first implementation uses host collectives and device synchronization for
generation/ACK messages, one CPU page-plan validation per forward, and a drain at
forward completion. It retains default asynchronous scheduling but does not
claim those control operations are asynchronous or free of latency.

This version requires all cache readers to follow the model compute-stream
contract. Extra reader streams, resizing/rebinding caches, graph capture,
connector ownership, MTP rollback and cross-server use require further design
and tests. A peer failure relies on the configured process-group timeout; do not
free live allocations after a failed drain. A submission or ACK exception marks
the generation failed, retains in-flight descriptors and mappings, and refuses
further prefetch or manual reclamation until process-group teardown.

## Validation

Device-independent tests are under `tests/ut/distributed/kv_transfer/`.
The hardware regression is `tests/e2e/kvpp/test_ipc_pull_lifecycle.py`; run it with
`torchrun` on an ascending list of HCCS-connected visible devices, from the
validation project directory containing `build-release/lib/libkv_copy.so`.
It covers multiple layouts, actual scratch aliases, main/indexer/scale bundles,
tail-page writes, a delayed reader, a non-default compute stream, cross-forward
page reuse and an early-return drain. It is not a complete-model accuracy or
throughput test. The accompanying A3 report records measured results and failures.
