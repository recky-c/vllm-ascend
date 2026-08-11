# SPDX-License-Identifier: Apache-2.0
"""KVPP cache allocation hook.

Registers a ``KVCacheAllocationHook`` with vLLM when ``kvpp_size > 1``.
The hook is invoked once per worker inside ``get_kv_cache_configs`` and
returns, for each worker:

* ``allocation_groups`` — the projected KV cache groups containing only
  this worker's owned layers (persistent) plus a dual-scratch pair per
  distinct physical spec bucket (non-owned layers).
* ``scratch_aliases`` — mapping from each scratch layer name to the logical
  layer names that alias it, so vLLM expands ``KVCacheTensor.shared_by``.
* ``kvpp_layer_owners`` — layer name -> owner rank, written into
  ``KVCacheConfig.kvpp_layer_owners`` for the runtime.

The hook contains no transport, runtime, or attention logic. It is pure
CPU static planning that runs in the engine-core process.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.kv_cache_utils import (
    KVCacheAllocationResult,
    KVCacheGroupSpec,
    _project_kv_cache_groups_to_worker,
    register_kv_cache_allocation_hook,
)
from vllm.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_REGISTERED = False

#: Special owner marking a layer as replicated on every KVPP rank.
#: Such layers use persistent KV caches on all ranks and skip transport
#: entirely (no push/receive, no wait). Layer 0 is always replicated so
#: that the first attention layer has no transfer to wait for — its
#: ``begin_layer`` is a no-op and ``prepare_for_attention`` can start
#: layer 1's transfer overlapped by layer 0's SFA kernel.
KVPP_REPLICATED_OWNER = -1


def register_kvpp_allocation_hook() -> None:
    """Register the KVPP allocation hook if not already registered.

    Called by the platform during config validation. Safe to call multiple
    times; only registers once. When ``kvpp_size <= 1`` the hook is not
    registered and vLLM uses its default per-worker projection.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    register_kv_cache_allocation_hook(_kvpp_allocation_hook)
    _REGISTERED = True


def _get_kvpp_layer_owners(
    kv_cache_specs: dict[str, KVCacheSpec], kvpp_size: int
) -> dict[str, int]:
    """Assign each cache layer to a KVPP owner rank by transformer layer index.

    Layers sharing the same transformer layer index (e.g. GLM SFA main KV +
    indexer K) are always owned by the same rank. Layer index 0 is marked
    ``KVPP_REPLICATED_OWNER`` so every rank holds a persistent copy and no
    transfer is needed for it (it has no predecessor to overlap its
    transfer). The remaining layer indices are split contiguously across
    ranks; the first ``remainder`` ranks each get one extra layer.
    """
    layers_by_index: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_cache_specs:
        layers_by_index[extract_layer_index(layer_name)].append(layer_name)

    layer_indices = sorted(layers_by_index)
    if len(layer_indices) < kvpp_size + 1:
        raise ValueError(
            f"KVPP size ({kvpp_size}) plus the replicated layer 0 exceeds the "
            f"number of KV cache layer bundles ({len(layer_indices)})."
        )

    replicated_indices = {layer_indices[0]}
    partition_indices = layer_indices[1:]
    base, remainder = divmod(len(partition_indices), kvpp_size)
    owners: dict[str, int] = {}
    for layer_index in replicated_indices:
        for layer_name in layers_by_index[layer_index]:
            owners[layer_name] = KVPP_REPLICATED_OWNER
    offset = 0
    for owner_rank in range(kvpp_size):
        partition_size = base + int(owner_rank < remainder)
        for layer_index in partition_indices[offset : offset + partition_size]:
            for layer_name in layers_by_index[layer_index]:
                owners[layer_name] = owner_rank
        offset += partition_size
    return owners


def _get_kvpp_allocation_groups(
    logical_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
    owners: dict[str, int],
    kvpp_rank: int,
) -> tuple[list[KVCacheGroupSpec], dict[str, list[str]]]:
    """Build this worker's allocation: owned persistent + dual scratch.

    For each KV cache group:
      * Owned layers (owner == kvpp_rank) enter the allocation directly.
      * Non-owned layers are bucketed by spec equality; each bucket gets
        two scratch layers that the bucket's layers alias alternately.
    """
    allocation_spec: dict[str, KVCacheSpec] = {}
    scratch_aliases: dict[str, list[str]] = {}

    for group in logical_groups:
        local_names = [name for name in group.layer_names if name in worker_spec]
        owned_names = [
            name
            for name in local_names
            if owners[name] == kvpp_rank
            or owners[name] == KVPP_REPLICATED_OWNER
        ]
        non_owned_names = [
            name
            for name in local_names
            if owners[name] != kvpp_rank
            and owners[name] != KVPP_REPLICATED_OWNER
        ]
        allocation_names = list(owned_names)

        # Bucket non-owned layers by physical spec equality so that layers
        # with the same page layout share one dual-scratch pair.
        scratch_buckets: list[list[str]] = []
        for name in non_owned_names:
            for bucket in scratch_buckets:
                if worker_spec[name] == worker_spec[bucket[0]]:
                    bucket.append(name)
                    break
            else:
                scratch_buckets.append([name])

        for bucket in scratch_buckets:
            # Two scratch layers, each the full size of one persistent layer.
            # They alternate across the bucket so that layer N attention reads
            # scratch_0 while layer N+1 transfer fills scratch_1.
            scratch_names = bucket[:2]
            allocation_names.extend(scratch_names)
            for scratch_index, scratch_name in enumerate(scratch_names):
                scratch_aliases[scratch_name] = bucket[
                    scratch_index :: len(scratch_names)
                ]

        for layer_name in allocation_names:
            allocation_spec[layer_name] = worker_spec[layer_name]

    return (
        _project_kv_cache_groups_to_worker(logical_groups, allocation_spec),
        scratch_aliases,
    )


def _kvpp_allocation_hook(
    vllm_config: "VllmConfig",
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
    worker_index: int,
) -> KVCacheAllocationResult:
    """Per-worker allocation override for KVPP."""
    kvpp_size = vllm_config.parallel_config.kvpp_size
    if kvpp_size <= 1:
        return KVCacheAllocationResult(
            allocation_groups=_project_kv_cache_groups_to_worker(
                global_kv_cache_groups, worker_spec
            )
        )

    # Owner assignment is deterministic; recompute per worker is cheap and
    # avoids cross-worker state.
    owners = _get_kvpp_layer_owners(worker_spec, kvpp_size)
    kvpp_rank = worker_index % kvpp_size
    allocation_groups, scratch_aliases = _get_kvpp_allocation_groups(
        global_kv_cache_groups, worker_spec, owners, kvpp_rank
    )
    return KVCacheAllocationResult(
        allocation_groups=allocation_groups,
        scratch_aliases=scratch_aliases,
        kvpp_layer_owners=owners,
    )
