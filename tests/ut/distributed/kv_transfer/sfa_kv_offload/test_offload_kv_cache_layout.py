"""Unit tests for the SFA sparse KV offload tuple-slot layout helpers.

Covers the six-tuple (LIC8) contract: [0]main_k [1]main_v [2]indexer_k
[3]resident_k [4]resident_v [5]indexer_s, and the five-tuple (non-C8) contract
where [5] is absent and [0]-[4] are unchanged.
"""

import torch

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.offload_kv_cache_layout import (
    OFFLOAD_C8_TUPLE_LEN,
    OFFLOAD_INDEXER_K,
    OFFLOAD_INDEXER_S,
    OFFLOAD_MAIN_K,
    OFFLOAD_MAIN_V,
    OFFLOAD_RESIDENT_K,
    OFFLOAD_RESIDENT_V,
    OFFLOAD_TUPLE_LEN,
    is_offload_c8_kv_cache,
)


def test_tuple_len_constants_match_slot_indices():
    # The tail-append contract: [0]-[4] identical for five/six-tuple, [5] only C8.
    assert OFFLOAD_TUPLE_LEN == 5
    assert OFFLOAD_C8_TUPLE_LEN == 6
    assert OFFLOAD_MAIN_K == 0
    assert OFFLOAD_MAIN_V == 1
    assert OFFLOAD_INDEXER_K == 2
    assert OFFLOAD_RESIDENT_K == 3
    assert OFFLOAD_RESIDENT_V == 4
    assert OFFLOAD_INDEXER_S == 5


def test_is_offload_c8_kv_cache_detects_six_tuple():
    six = tuple(torch.zeros(1) for _ in range(6))
    five = tuple(torch.zeros(1) for _ in range(5))
    assert is_offload_c8_kv_cache(six) is True
    assert is_offload_c8_kv_cache(five) is False


def test_is_offload_c8_kv_cache_rejects_other_lengths():
    for n in (0, 1, 2, 3, 4, 7, 8):
        tup = tuple(torch.zeros(1) for _ in range(n))
        assert is_offload_c8_kv_cache(tup) is False
