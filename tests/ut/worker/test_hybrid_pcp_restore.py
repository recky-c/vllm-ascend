# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.

import numpy as np
import torch

from vllm_ascend.worker.v2.hybrid_pcp import (
    build_linear_hidden_restore_idx,
    get_linear_rank_segments,
)


def test_build_linear_hidden_restore_idx_matches_full_ag_layout():
    """Mixed decode+prefill: idx maps into AG(full local padded), not prefill-only."""
    pcp = 2
    # req0 decode: 2 tokens; req1 prefill: 5 tokens → rank0 gets 3, rank1 gets 2.
    num_scheduled = np.array([2, 5], dtype=np.int32)
    is_prefilling = np.array([False, True])
    query_start_loc = np.array([0, 2, 7], dtype=np.int32)
    segments_by_rank = [
        get_linear_rank_segments(pcp, rank, num_scheduled, is_prefilling, query_start_loc)
        for rank in range(pcp)
    ]
    # Local layouts (decode-first, pad to max=5):
    #   rank0: [d0 d1 p0 p1 p2]
    #   rank1: [d0 d1 p3 p4 pad]
    padded = 5
    restore = build_linear_hidden_restore_idx(
        pcp_world_size=pcp,
        device=torch.device("cpu"),
        global_num_tokens=7,
        linear_num_tokens_padded=padded,
        segments_by_rank=segments_by_rank,
        is_prefilling=is_prefilling,
    ).numpy()

    # AG layout: [r0(5) | r1(5)]
    expected = np.array(
        [
            0,  # d0 ← rank0 local 0
            1,  # d1 ← rank0 local 1
            2,  # p0 ← rank0 local 2
            3,  # p1 ← rank0 local 3
            4,  # p2 ← rank0 local 4
            5 + 2,  # p3 ← rank1 local 2
            5 + 3,  # p4 ← rank1 local 3
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(restore, expected)


def test_build_linear_hidden_restore_idx_pure_decode_uses_rank0():
    pcp = 2
    num_scheduled = np.array([3], dtype=np.int32)
    is_prefilling = np.array([False])
    query_start_loc = np.array([0, 3], dtype=np.int32)
    segments_by_rank = [
        get_linear_rank_segments(pcp, rank, num_scheduled, is_prefilling, query_start_loc)
        for rank in range(pcp)
    ]
    restore = build_linear_hidden_restore_idx(
        pcp_world_size=pcp,
        device=torch.device("cpu"),
        global_num_tokens=3,
        linear_num_tokens_padded=3,
        segments_by_rank=segments_by_rank,
        is_prefilling=is_prefilling,
    ).numpy()
    np.testing.assert_array_equal(restore, np.array([0, 1, 2], dtype=np.int64))
