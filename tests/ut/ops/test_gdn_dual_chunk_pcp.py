# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm_ascend.ops.gdn_pcp_conv as conv_module
import vllm_ascend.ops.gdn_pcp_ssm as ssm_module
from vllm_ascend.ops.gdn_pcp_conv import (
    _fold_dual_chunk_histories,
    prepare_dual_chunk_pcp_conv,
)
from vllm_ascend.ops.gdn_pcp_ssm import (
    _scan_dual_chunk_states,
    correct_dual_chunk_pcp_ssm_state,
)


class _PresetGatherGroup:
    world_size = 2
    rank_in_group = 0

    def __init__(self, remote_payloads):
        self.remote_payloads = iter(remote_payloads)

    def all_gather(self, tensor, dim):
        remote = next(self.remote_payloads).to(dtype=tensor.dtype)
        return torch.cat((tensor, remote), dim=dim)


def test_fold_dual_chunk_conv_histories_uses_global_chunk_order():
    # One request, PCP=2 => four canonical chunks. The physical owners are
    # rank0.head, rank1.head, rank1.tail, rank0.tail.
    tails = torch.tensor([[[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]], [[6.0, 7.0, 8.0]], [[9.0, 10.0, 11.0]]]])
    counts = torch.tensor([[3, 3, 3, 3]])
    valid = torch.ones((1, 4), dtype=torch.bool)
    initial = torch.tensor([[[-3.0, -2.0, -1.0]]])

    previous, final = _fold_dual_chunk_histories(tails, counts, valid, initial)

    assert torch.equal(previous[0, 0], initial[0])
    assert torch.equal(previous[0, 1], tails[0, 0])
    assert torch.equal(previous[0, 2], tails[0, 1])
    assert torch.equal(previous[0, 3], tails[0, 2])
    assert torch.equal(final[0], tails[0, 3])


def test_fold_dual_chunk_conv_histories_skips_missing_tail_chunk():
    tails = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]], [[0.0, 0.0]], [[7.0, 8.0]]]])
    counts = torch.tensor([[2, 2, 0, 2]])
    valid = counts > 0
    initial = torch.zeros((1, 1, 2))

    previous, final = _fold_dual_chunk_histories(tails, counts, valid, initial)

    assert torch.equal(previous[0, 2], tails[0, 1])
    assert torch.equal(previous[0, 3], tails[0, 1])
    assert torch.equal(final[0], tails[0, 3])


def test_prepare_conv_uses_distinct_temp_rows_for_shared_cache_slot(monkeypatch):
    group = _PresetGatherGroup(
        [
            torch.tensor([[[[2.0, 3.0]], [[4.0, 5.0]]]]),
            torch.tensor([[[1, 2], [2, 2]]]),
        ]
    )
    monkeypatch.setattr(conv_module, "get_pcp_group", lambda: group)
    persistent_state = torch.zeros((8, 2, 1))

    plan = prepare_dual_chunk_pcp_conv(
        conv_state=persistent_state,
        mixed_qkv=torch.tensor([[6.0], [7.0], [0.0], [1.0]]),
        query_start_loc=torch.tensor([0, 2, 4]),
        state_indices=torch.tensor([5, 5]),
        # rank0 local row order is tail then head.
        segment_ids=torch.tensor([3, 0]),
        segment_capacity=2,
        state_len=2,
    )

    assert torch.equal(plan.cache_indices, torch.tensor([0, 1]))
    assert torch.equal(plan.conv_state[:, :, 0], torch.tensor([[4.0, 5.0], [0.0, 0.0]]))
    plan.write_back_final_state()
    assert torch.equal(persistent_state[5, :, 0], torch.tensor([6.0, 7.0]))


def test_scan_dual_chunk_ssm_states_follows_head_then_reverse_tail():
    # Scalar affine transforms make the expected recurrence explicit:
    # s <- p_i + phi_i * s, in canonical chunk order 0, 1, 2, 3.
    local_final = torch.tensor([[[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]])
    transition = torch.tensor([[[[[2.0]]], [[[3.0]]], [[[4.0]]], [[[5.0]]]]])
    valid = torch.ones((1, 4), dtype=torch.bool)
    initial = torch.zeros((1, 1, 1, 1))

    previous, final = _scan_dual_chunk_states(
        local_final,
        transition,
        valid,
        initial,
    )

    assert torch.equal(previous.flatten(), torch.tensor([0.0, 1.0, 5.0, 23.0]))
    assert torch.equal(final.flatten(), torch.tensor([119.0]))


def test_scan_dual_chunk_ssm_states_treats_missing_chunk_as_identity():
    local_final = torch.tensor([[[[[1.0]]], [[[2.0]]], [[[99.0]]], [[[4.0]]]]])
    transition = torch.tensor([[[[[2.0]]], [[[3.0]]], [[[99.0]]], [[[5.0]]]]])
    valid = torch.tensor([[True, True, False, True]])
    initial = torch.zeros((1, 1, 1, 1))

    previous, final = _scan_dual_chunk_states(
        local_final,
        transition,
        valid,
        initial,
    )

    assert torch.equal(previous.flatten(), torch.tensor([0.0, 1.0, 5.0, 5.0]))
    assert torch.equal(final.flatten(), torch.tensor([29.0]))


def test_correct_ssm_maps_tail_first_local_rows_back_to_causal_order(monkeypatch):
    group = _PresetGatherGroup(
        [
            torch.tensor([[[[[2.0]]], [[[3.0]]]]]),
            torch.tensor([[[[[3.0]]], [[[4.0]]]]]),
            torch.tensor([[1, 2]]),
        ]
    )
    monkeypatch.setattr(ssm_module, "get_pcp_group", lambda: group)
    monkeypatch.setattr(
        ssm_module,
        "_compute_local_transition",
        lambda **kwargs: torch.tensor([[[[5.0]]], [[[2.0]]]]),
    )
    captured = {}

    def fake_recompute(**kwargs):
        captured["initial_state"] = kwargs["initial_state"].clone()
        return torch.tensor(0.0), torch.tensor(1.0)

    monkeypatch.setattr(ssm_module, "_recompute_local_h", fake_recompute)
    _h, _v_new, final = correct_dual_chunk_pcp_ssm_state(
        local_final_state=torch.tensor([[[[4.0]]], [[[1.0]]]]),
        k=torch.empty(0),
        w=torch.empty(0),
        u=torch.empty(0),
        g=torch.empty(0),
        cu_seqlens=torch.tensor([0, 1, 2]),
        # rank0 local row order is tail then head.
        segment_ids=torch.tensor([3, 0]),
        segment_capacity=2,
    )

    assert torch.equal(captured["initial_state"].flatten(), torch.tensor([23.0, 0.0]))
    assert torch.equal(final.flatten(), torch.tensor([119.0, 119.0]))
