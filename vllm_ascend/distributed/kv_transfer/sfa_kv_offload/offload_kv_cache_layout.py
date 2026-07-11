# SPDX-License-Identifier: Apache-2.0
"""kv_cache tuple slot indices for SFA sparse KV offload.

Non-C8 offload:
  [0] main_k, [1] main_v, [2] indexer_k, [3] resident_k, [4] resident_v

C8 offload follows upstream SFA C8 packed KV layout:
  [0] packed_main_kv, [1] indexer_k, [2] indexer_scale,
  [3] resident_packed_main_kv
"""

OFFLOAD_MAIN_K = 0
OFFLOAD_MAIN_V = 1
OFFLOAD_INDEXER_K = 2
OFFLOAD_RESIDENT_K = 3
OFFLOAD_RESIDENT_V = 4

OFFLOAD_C8_MAIN_KV = 0
OFFLOAD_C8_INDEXER_K = 1
OFFLOAD_C8_INDEXER_S = 2
OFFLOAD_C8_RESIDENT_KV = 3

OFFLOAD_TUPLE_LEN = 5
OFFLOAD_C8_TUPLE_LEN = 4


def is_offload_c8_kv_cache(kv_cache: tuple) -> bool:
    return len(kv_cache) == OFFLOAD_C8_TUPLE_LEN
