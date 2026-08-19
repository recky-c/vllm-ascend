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
    vllm_config: VllmConfig, layer_names: Iterable[str]
) -> set[str]:
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
    replicated_layers = {
        layer_name
        for layer_name in layer_names
        if mtp_start <= extract_layer_index(layer_name) < mtp_end
    }
    if not replicated_layers:
        raise ValueError("KVPP could not identify any MTP KV-cache layers to replicate.")
    return replicated_layers


def get_kvpp_layer_owners(
    vllm_config: VllmConfig, layer_names: Iterable[str]
) -> dict[str, int]:
    kvpp_size = KVPPConfig.from_vllm_config(vllm_config).size
    # Workers are separate Python processes and may receive layer names from
    # sets or differently ordered dictionaries. Keep both owner insertion
    # order and per-layer cache-bundle order identical on every rank.
    layer_names = tuple(
        sorted(layer_names, key=lambda name: (extract_layer_index(name), name))
    )
    replicated_layers = _get_replicated_mtp_layers(vllm_config, layer_names)
    layers_by_index: dict[int, list[str]] = defaultdict(list)
    for layer_name in layer_names:
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


def _get_allocation_groups(
    logical_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
    owners: dict[str, int],
    kvpp_rank: int,
) -> tuple[list[KVCacheGroupSpec], dict[str, list[str]]]:
    allocation_spec: dict[str, KVCacheSpec] = {}
    scratch_aliases: dict[str, list[str]] = {}

    for group in logical_groups:
        local_names = [name for name in group.layer_names if name in worker_spec]
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
    kvpp_size = KVPPConfig.from_vllm_config(vllm_config).size
    if kvpp_size <= 1:
        return None
    owners = get_kvpp_layer_owners(vllm_config, worker_spec)
    allocation_groups, _ = _get_allocation_groups(
        global_groups, worker_spec, owners, worker_index % kvpp_size
    )
    return allocation_groups


def finalize_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    global_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
    worker_index: int,
) -> None:
    kvpp_size = KVPPConfig.from_vllm_config(vllm_config).size
    if kvpp_size <= 1:
        return
    owners = get_kvpp_layer_owners(vllm_config, worker_spec)
    _, scratch_aliases = _get_allocation_groups(
        global_groups, worker_spec, owners, worker_index % kvpp_size
    )
    for tensor in kv_cache_config.kv_cache_tensors:
        expanded_names: list[str] = []
        for layer_name in tensor.shared_by:
            expanded_names.extend(scratch_aliases.get(layer_name, [layer_name]))
        tensor.shared_by = list(dict.fromkeys(expanded_names))
    kv_cache_config.kv_cache_groups = project_kv_cache_groups_to_worker(
        global_groups, worker_spec
    )
