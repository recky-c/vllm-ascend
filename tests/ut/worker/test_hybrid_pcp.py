from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.attention.context_parallel.hybrid_pcp import (
    CacheInputDistribution,
    PCPInputLayout,
    contiguous_state_pcp_capability,
    dual_chunk_pcp_capability,
)
from vllm_ascend.attention.context_parallel.hybrid_pcp.bridge import (
    enter_hybrid_fa,
    exit_hybrid_fa,
)
from vllm_ascend.worker.v2.pcp.cache_plan import (
    build_dummy_fa_cache_write_plan,
    build_fa_cache_write_plan,
)
from vllm_ascend.worker.v2.pcp.contracts import (
    CacheWritePlan,
    CacheWriteSegment,
    CacheWriteSegmentKind,
)
from vllm_ascend.worker.v2.pcp.layout import (
    build_hybrid_pcp_layout,
    build_linear_prefill_lengths,
)


def _maps():
    linear = (
        np.array([0, 1, 2, 3]),
        np.array([4, 5, 6, -1]),
    )
    fa = (
        np.array([0, 1, 6, -1]),
        np.array([2, 3, 4, 5]),
    )
    return linear, fa


def test_hybrid_layout_builds_all_three_conversion_indices():
    linear, fa = _maps()

    layout = build_hybrid_pcp_layout(
        linear_prefill_by_rank=linear,
        fa_prefill_by_rank=fa,
        pcp_rank=0,
        num_decode_tokens=1,
        global_num_tokens=8,
        linear_num_tokens=5,
        fa_num_tokens=4,
        device=torch.device("cpu"),
    )

    assert layout.hybrid_linear_ag_restore_idx.tolist() == list(range(7))
    assert layout.hybrid_global_to_fa_idx.tolist() == [0, 1, 6, 0]
    assert layout.hybrid_fa_to_linear_idx.tolist() == [0, 1, 4, 5]
    assert layout.linear_valid_mask.tolist() == [True] * 5
    assert layout.has_prefill


def test_linear_prefill_lengths_keep_decode_replicated():
    lengths = build_linear_prefill_lengths(
        np.array([1, 7, 10], dtype=np.int32),
        np.array([False, True, True]),
        pcp_world_size=2,
        cp_interleave=2,
    )

    np.testing.assert_array_equal(
        lengths,
        np.array(
            [
                [1, 1],
                [4, 3],
                [6, 4],
            ],
            dtype=np.int32,
        ),
    )


def test_hybrid_layout_marks_linear_padding_invalid():
    linear, fa = _maps()

    layout = build_hybrid_pcp_layout(
        linear_prefill_by_rank=linear,
        fa_prefill_by_rank=fa,
        pcp_rank=1,
        num_decode_tokens=1,
        global_num_tokens=8,
        linear_num_tokens=4,
        fa_num_tokens=5,
        device=torch.device("cpu"),
    )

    assert layout.hybrid_fa_to_linear_idx.tolist() == [6, 7, 2, 0]
    assert layout.linear_valid_mask.tolist() == [
        True,
        True,
        True,
        True,
        False,
    ]


def test_hybrid_layout_rejects_missing_or_duplicate_global_tokens():
    linear, fa = _maps()
    invalid = (linear[0], np.array([4, 5, 5, -1]))

    with pytest.raises(ValueError, match="permutation"):
        build_hybrid_pcp_layout(
            linear_prefill_by_rank=invalid,
            fa_prefill_by_rank=fa,
            pcp_rank=0,
            num_decode_tokens=1,
            global_num_tokens=8,
            linear_num_tokens=5,
            fa_num_tokens=4,
            device=torch.device("cpu"),
        )


def test_pure_decode_layout_uses_empty_bridge_indices():
    empty = (np.empty(0, dtype=np.int64),) * 2

    layout = build_hybrid_pcp_layout(
        linear_prefill_by_rank=empty,
        fa_prefill_by_rank=empty,
        pcp_rank=0,
        num_decode_tokens=3,
        global_num_tokens=3,
        linear_num_tokens=3,
        fa_num_tokens=3,
        device=torch.device("cpu"),
    )

    assert not layout.has_prefill
    assert layout.hybrid_linear_ag_restore_idx.numel() == 0
    assert layout.hybrid_global_to_fa_idx.numel() == 0
    assert layout.hybrid_fa_to_linear_idx.numel() == 0
    assert layout.linear_valid_mask.tolist() == [True, True, True]


def test_cache_write_plan_masks_invalid_slots():
    slots = torch.tensor([4, 5, 6], dtype=torch.int64)
    valid_mask = torch.tensor([True, False, True])
    segment = CacheWriteSegment(
        kind=CacheWriteSegmentKind.PREFILL,
        start=0,
        stop=3,
        distribution=CacheInputDistribution.ALREADY_GLOBAL,
        slot_mapping=slots,
        valid_mask=valid_mask,
    )
    plan = CacheWritePlan(group_id=0, segments=(segment,))

    assert plan.segments[0].input_range == slice(0, 3)
    assert segment.effective_slot_mapping().tolist() == [4, -1, 6]


def test_cache_write_plan_requires_contiguous_segments():
    segment = CacheWriteSegment(
        kind=CacheWriteSegmentKind.PREFILL,
        start=1,
        stop=2,
        distribution=CacheInputDistribution.ALREADY_GLOBAL,
        slot_mapping=torch.tensor([0]),
    )

    with pytest.raises(ValueError, match="contiguous"):
        CacheWritePlan(group_id=0, segments=(segment,))


def test_cache_plan_builder_separates_decode_and_global_prefill():
    linear, fa = _maps()
    layout = build_hybrid_pcp_layout(
        linear_prefill_by_rank=linear,
        fa_prefill_by_rank=fa,
        pcp_rank=0,
        num_decode_tokens=1,
        global_num_tokens=8,
        linear_num_tokens=5,
        fa_num_tokens=4,
        device=torch.device("cpu"),
    )
    slots = torch.arange(8, dtype=torch.int64)

    plan = build_fa_cache_write_plan(
        group_id=2,
        capability=dual_chunk_pcp_capability(),
        layout=layout,
        global_slot_mapping=slots,
    )

    assert plan.group_id == 2
    assert [segment.kind for segment in plan.segments] == [
        CacheWriteSegmentKind.DECODE,
        CacheWriteSegmentKind.PREFILL,
    ]
    assert [segment.distribution for segment in plan.segments] == [
        CacheInputDistribution.LOCAL_REPLICATED,
        CacheInputDistribution.ALREADY_GLOBAL,
    ]
    assert plan.segments[0].slot_mapping.tolist() == [0]
    assert plan.segments[1].slot_mapping.tolist() == list(range(1, 8))


def test_dummy_cache_plan_masks_all_cache_writes():
    slots = torch.arange(3, dtype=torch.int64)

    plan = build_dummy_fa_cache_write_plan(
        group_id=0,
        slot_mapping=slots,
        device=torch.device("cpu"),
    )

    assert plan.segments[0].effective_slot_mapping().tolist() == [-1, -1, -1]


def test_group_capabilities_are_backend_neutral():
    fa = dual_chunk_pcp_capability()
    state = contiguous_state_pcp_capability()

    assert fa.input_layout == PCPInputLayout.DUAL_CHUNK_VIRTUAL
    assert fa.accepts(CacheInputDistribution.ALREADY_GLOBAL)
    assert fa.supports_piecewise
    assert state.input_layout == PCPInputLayout.CONTIGUOUS_CAUSAL_STATE
    assert not state.cache_input_distributions


def test_bridge_enters_and_exits_dual_chunk_with_one_collective_each():
    linear, fa = _maps()
    layout = build_hybrid_pcp_layout(
        linear_prefill_by_rank=linear,
        fa_prefill_by_rank=fa,
        pcp_rank=0,
        num_decode_tokens=1,
        global_num_tokens=8,
        linear_num_tokens=5,
        fa_num_tokens=4,
        device=torch.device("cpu"),
    )
    bridge_view = type("BridgeView", (), {"layout": layout})()

    class FakeGroup:
        def __init__(self, gathered):
            self.gathered = gathered
            self.calls = 0

        def all_gather(self, _tensor, dim=0):
            assert dim == 0
            self.calls += 1
            return self.gathered

    entry_group = FakeGroup(torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [999.0]]))
    operand = torch.tensor([[100.0], [0.0], [1.0], [2.0], [3.0]])
    with patch(
        "vllm_ascend.attention.context_parallel.hybrid_pcp.bridge.get_pcp_group",
        return_value=entry_group,
    ):
        entered = enter_hybrid_fa((operand,), bridge_view)

    assert entry_group.calls == 1
    assert entered.fa_operands[0].flatten().tolist() == [
        100.0,
        0.0,
        1.0,
        6.0,
        0.0,
    ]
    assert entered.global_prefill_operands[0].flatten().tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]

    exit_group = FakeGroup(
        torch.tensor(
            [
                [10.0],
                [11.0],
                [16.0],
                [999.0],
                [12.0],
                [13.0],
                [14.0],
                [15.0],
            ]
        )
    )
    fa_output = torch.tensor([[100.0], [10.0], [11.0], [16.0], [999.0]])
    with patch(
        "vllm_ascend.attention.context_parallel.hybrid_pcp.bridge.get_pcp_group",
        return_value=exit_group,
    ):
        restored = exit_hybrid_fa(fa_output, bridge_view)

    assert exit_group.calls == 1
    assert restored.flatten().tolist() == [
        100.0,
        10.0,
        11.0,
        12.0,
        13.0,
    ]
