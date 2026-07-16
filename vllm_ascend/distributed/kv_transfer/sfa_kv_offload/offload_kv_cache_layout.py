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

# 学习对照（D 侧 5/6-tuple 落点）:
#   MAIN_K/V     → decode 满块卸到 pinned CPU 池；PD pull 时满块进 CPU
#   INDEXER_K(/S)→ 常驻 HBM（每步 top-K）
#   RESIDENT_K/V → LRU 工作集 HBM（sparse_copy 回灌）

_OFFLOAD_SLOT_NAMES = {
    OFFLOAD_MAIN_K: "MAIN_K→CPU满块",
    OFFLOAD_MAIN_V: "MAIN_V→CPU满块",
    OFFLOAD_INDEXER_K: "INDEXER_K→HBM",
    OFFLOAD_RESIDENT_K: "RESIDENT_K→LRU/HBM",
    OFFLOAD_RESIDENT_V: "RESIDENT_V→LRU/HBM",
    OFFLOAD_INDEXER_S: "INDEXER_S→HBM(LIC8)",
}


def describe_offload_tuple_layout(tuple_len: int) -> str:
    """Human-readable 5/6-tuple slot map (for learning prints)."""
    parts = [f"[{i}]{_OFFLOAD_SLOT_NAMES[i]}" for i in range(tuple_len) if i in _OFFLOAD_SLOT_NAMES]
    return f"tuple_len={tuple_len}: " + ", ".join(parts)


def is_offload_c8_kv_cache(kv_cache: tuple) -> bool:
    return len(kv_cache) == OFFLOAD_C8_TUPLE_LEN
