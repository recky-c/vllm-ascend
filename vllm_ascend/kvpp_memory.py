# SPDX-License-Identifier: Apache-2.0
"""Software budgeting for the experimentally verified A3 IPC allocation pool."""

IPC_ALLOCATION_ALIGNMENT = 2 * 1024 * 1024
# A software default, not a universal hardware limit. The tested deployment
# rejects cumulative P2P allocations above 8 GiB while HBM queries remain free.
DEFAULT_IPC_POOL_BUDGET_BYTES = 7 * 1024**3
VERIFIED_IPC_POOL_LIMIT_BYTES = 8 * 1024**3


def ipc_cache_payload_budget(available_bytes: int, pool_budget_bytes: int, physical_cache_entries: int) -> int:
    if physical_cache_entries <= 0 or pool_budget_bytes <= 0:
        raise ValueError("IPC cache budgeting requires a non-empty physical allocation plan")
    # Supported MLA V1 entries have at most two raw allocations (nope/rope,
    # or indexer/scale). Each may round up by less than one P2P alignment.
    padding = physical_cache_entries * 2 * IPC_ALLOCATION_ALIGNMENT
    if pool_budget_bytes <= padding:
        raise ValueError("IPC pool budget cannot cover physical allocation alignment")
    return min(available_bytes, pool_budget_bytes - padding)
