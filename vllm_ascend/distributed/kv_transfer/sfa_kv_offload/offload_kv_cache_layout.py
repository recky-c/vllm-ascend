# SPDX-License-Identifier: Apache-2.0
"""kv_cache tuple slot indices for SFA sparse KV offload (+ LIC8 six-tuple).

Non-C8 offload (five-tuple):
  [0] main_k  [1] main_v  [2] indexer_k  [3] resident_k  [4] resident_v

C8 + offload (six-tuple): append scale at the tail; [0]–[4] unchanged.
  [5] indexer_s
"""

OFFLOAD_MAIN_K = 0
OFFLOAD_MAIN_V = 1
OFFLOAD_INDEXER_K = 2
OFFLOAD_RESIDENT_K = 3
OFFLOAD_RESIDENT_V = 4
OFFLOAD_INDEXER_S = 5

OFFLOAD_TUPLE_LEN = 5
OFFLOAD_C8_TUPLE_LEN = 6


def is_offload_c8_kv_cache(kv_cache: tuple) -> bool:
    return len(kv_cache) == OFFLOAD_C8_TUPLE_LEN
