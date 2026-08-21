# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Iterable

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.kvpp_config import KVPPConfig


def project_kv_cache_groups_to_worker(
    global_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Project global logical groups onto one worker's PP-local layers.

    The projected list keeps one entry per global group so logical group
    indices stay aligned with the global topology; groups without local
    layers become empty entries instead of being compacted away.
    """
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        group_spec = group.kv_cache_spec
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(
            KVCacheGroupSpec(
                worker_layer_names,
                group_spec,
                is_eagle_group=group.is_eagle_group and bool(worker_layer_names),
            )
        )
    return projected_groups


def _get_replicated_mtp_layers(
    vllm_config: VllmConfig, local_layer_names: Iterable[str]
) -> set[str]:
    """Find MTP KV-cache layers within the current PP stage's local layers.

    This helper only selects MTP names from ``local_layer_names``; it does not
    decide whether the current worker should own MTP (non-last PP stages
    legitimately have no MTP cache).
    """
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.method != "mtp":
        return set()

    hf_config = vllm_config.model_config.hf_config
    mtp_start = getattr(hf_config, "num_hidden_layers", None)
    num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", None)
    if mtp_start is None or not num_mtp_layers:
        raise ValueError(
            "KVPP with MTP requires num_hidden_layers and a positive num_nextn_predict_layers in the model config."
        )
    mtp_end = mtp_start + num_mtp_layers
    return {
        layer_name
        for layer_name in local_layer_names
        if mtp_start <= extract_layer_index(layer_name) < mtp_end
    }


def get_kvpp_layer_owners(
    vllm_config: VllmConfig, local_layer_names: Iterable[str]
) -> dict[str, int]:
    """Partition PP-local Target KV layers across KVPP ranks.

    ``local_layer_names`` must already be PP-local (typically the keys of the
    current worker's cache spec). Replicated MTP layers are excluded from the
    owner partition and are absent from the returned mapping.
    """
    kvpp_size = KVPPConfig.from_vllm_config(vllm_config).size
    # Workers are separate Python processes and may receive layer names from
    # sets or differently ordered dictionaries. Keep both owner insertion
    # order and per-layer cache-bundle order identical on every rank.
    local_layer_names = tuple(
        sorted(local_layer_names, key=lambda name: (extract_layer_index(name), name))
    )
    replicated_layers = _get_replicated_mtp_layers(vllm_config, local_layer_names)
    layers_by_index: dict[int, list[str]] = defaultdict(list)
    for layer_name in local_layer_names:
        if layer_name not in replicated_layers:
            layers_by_index[extract_layer_index(layer_name)].append(layer_name)

    layer_indices = sorted(layers_by_index)
    if len(layer_indices) < kvpp_size:
        raise ValueError(f"KVPP size ({kvpp_size}) exceeds the number of KV cache layer bundles ({len(layer_indices)}).")

    base, remainder = divmod(len(layer_indices), kvpp_size)
    owners: dict[str, int] = {}
    offset = 0
    for owner_rank in range(kvpp_size):
        partition_size = base + int(owner_rank < remainder)
        for layer_index in layer_indices[offset : offset + partition_size]:
            for layer_name in layers_by_index[layer_index]:
                owners[layer_name] = owner_rank
        offset += partition_size
    return owners


def _get_worker_kvpp_rank(
    vllm_config: VllmConfig,
    worker_index: int,
) -> int:
    """Config-only KVPP group local rank for a worker.

    The global worker index is first reduced to the TP rank inside its PP
    stage, then to the KVPP group local rank. This avoids depending on the
    distributed KVPP group being initialized during cache placement.
    """
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    kvpp_size = KVPPConfig.from_vllm_config(vllm_config).size
    if tp_size % kvpp_size != 0:
        raise ValueError(
            f"tensor_parallel_size ({tp_size}) must be divisible by kvpp_size ({kvpp_size})."
        )
    tp_rank = worker_index % tp_size
    return tp_rank % kvpp_size


def _get_allocation_groups(
    logical_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
    owners: dict[str, int],
    kvpp_rank: int,
) -> tuple[list[KVCacheGroupSpec], dict[str, list[str]]]:
    """Per-KVPP-rank allocation view over PP-local logical groups.

    Target layers owned by this rank stay persistent; other owners' layers map
    onto two alternating scratch caches per layout. Layers absent from
    ``owners`` (replicated MTP) are allocated in full on every KVPP rank.
    """
    foreign_names = [
        name
        for group in logical_groups
        for name in group.layer_names
        if name not in worker_spec
    ]
    if foreign_names:
        raise ValueError(
            "KVPP placement received cache layers outside the current PP "
            f"stage: {sorted(foreign_names)}. PP projection must happen "
            "before KVPP allocation."
        )

    allocation_spec: dict[str, KVCacheSpec] = {}
    scratch_aliases: dict[str, list[str]] = {}

    for group in logical_groups:
        local_names = list(group.layer_names)
        managed_names = [name for name in local_names if name in owners]
        allocation_names = [
            name
            for name in local_names
            if name not in owners or owners[name] == kvpp_rank
        ]
        scratch_layout_groups: list[list[str]] = []
        for name in managed_names:
            if owners[name] == kvpp_rank:
                continue
            for layout_names in scratch_layout_groups:
                if worker_spec[name] == worker_spec[layout_names[0]]:
                    layout_names.append(name)
                    break
            else:
                scratch_layout_groups.append([name])
        for layout_names in scratch_layout_groups:
            scratch_names = layout_names[:2]
            allocation_names.extend(scratch_names)
            for scratch_index, scratch_name in enumerate(scratch_names):
                scratch_aliases[scratch_name] = layout_names[
                    scratch_index :: len(scratch_names)
                ]
        for layer_name in allocation_names:
            allocation_spec[layer_name] = worker_spec[layer_name]

    return (
        project_kv_cache_groups_to_worker(logical_groups, allocation_spec),
        scratch_aliases,
    )


def get_kv_cache_groups_for_worker(
    vllm_config: VllmConfig,
    global_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
    worker_index: int,
) -> list[KVCacheGroupSpec] | None:
    """Custom KV cache placement for a KVPP worker.

    The global logical groups are first projected to this worker's PP-local
    layers, then KVPP owner partitioning and allocation run entirely on the
    PP-local view. Group indices keep their global positions.
    """
    kvpp_size = KVPPConfig.from_vllm_config(vllm_config).size
    if kvpp_size <= 1:
        return None
    pp_local_groups = project_kv_cache_groups_to_worker(global_groups, worker_spec)
    owners = get_kvpp_layer_owners(vllm_config, worker_spec)
    kvpp_rank = _get_worker_kvpp_rank(vllm_config, worker_index)
    allocation_groups, _ = _get_allocation_groups(
        pp_local_groups, worker_spec, owners, kvpp_rank
    )
    return allocation_groups


def finalize_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    global_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
    worker_index: int,
) -> None:
    """Finalize per-worker KV cache config with the PP-local KVPP view.

    Uses the same PP-local projection as the allocation stage so the finalized
    logical cache topology and scratch aliases are consistent with what was
    allocated.
    """
    kvpp_size = KVPPConfig.from_vllm_config(vllm_config).size
    if kvpp_size <= 1:
        return
    pp_local_groups = project_kv_cache_groups_to_worker(global_groups, worker_spec)
    owners = get_kvpp_layer_owners(vllm_config, worker_spec)
    kvpp_rank = _get_worker_kvpp_rank(vllm_config, worker_index)
    _, scratch_aliases = _get_allocation_groups(
        pp_local_groups, worker_spec, owners, kvpp_rank
    )
    for tensor in kv_cache_config.kv_cache_tensors:
        expanded_names: list[str] = []
        for layer_name in tensor.shared_by:
            expanded_names.extend(scratch_aliases.get(layer_name, [layer_name]))
        tensor.shared_by = list(dict.fromkeys(expanded_names))
    kv_cache_config.kv_cache_groups = pp_local_groups
