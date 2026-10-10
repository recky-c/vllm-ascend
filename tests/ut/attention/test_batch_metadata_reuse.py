# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import is_dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from tests.ut.ops.test_gdn_attn_builder import BatchSpec, _make_builder, create_common_attn_metadata
from vllm_ascend import envs
from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder, AscendAttentionState
from vllm_ascend.attention.metadata_reuse import tensor_view_key


def assert_metadata_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif is_dataclass(actual):
        assert vars(actual).keys() == vars(expected).keys()
        for name, value in vars(actual).items():
            assert_metadata_equal(value, getattr(expected, name))
    else:
        assert actual == expected


def make_gdn_common(device, query_lens, seq_lens, draft_tokens):
    common = create_common_attn_metadata(BatchSpec(seq_lens, query_lens), 16, device)
    common.is_prefilling = torch.tensor([draft < 0 and q > 1 for q, draft in zip(query_lens, draft_tokens)])
    common.block_table_tensor = torch.arange(len(query_lens) * 4, dtype=torch.int32, device=device).reshape(-1, 4)
    return common


@pytest.mark.parametrize("graph", [False, True])
def test_gdn_reuse_matches_independent_groups_and_changed_inputs(graph):
    device = torch.device("cpu")
    mode = CUDAGraphMode.FULL_DECODE_ONLY if graph else CUDAGraphMode.NONE
    builders = [
        _make_builder(device=device, num_heads=32, num_speculative_tokens=3, cudagraph_mode=mode) for _ in range(4)
    ]
    previous_ptrs = None
    for accepted_values, offset, active in [([2, 4, 1], 10, 2), ([4, 2, 1], 100, 2), ([1, 1, 1], 200, 0)]:
        drafts = [3, 3, -1] if active else [-1, -1, -1]
        common = make_gdn_common(
            device, [4, 4, 4] if active else [0, 0, 0], [40, 48, 0] if active else [8, 8, 8], drafts
        )
        common.num_actual_tokens = active * 4
        accepted = torch.tensor(accepted_values, dtype=torch.int32)
        drafts = torch.tensor(drafts, dtype=torch.int32)
        shared = {}
        observed = []
        for group in range(2):
            local = common.replace(block_table_tensor=common.block_table_tensor + offset + 20 * group)
            expected = builders[group].build(0, local, accepted, drafts, num_actual_reqs=active if active else None)
            actual = builders[group + 2].build(
                0, local, accepted, drafts, num_actual_reqs=active if active else None, batch_metadata_cache=shared
            )
            assert_metadata_equal(actual, expected)
            observed.append(actual)
        assert len(shared) == 1
        if active:
            assert not torch.equal(observed[0].spec_state_indices_tensor, observed[1].spec_state_indices_tensor)
            if graph:
                ptrs = [entry.spec_query_start_loc.data_ptr() for entry in observed]
                assert ptrs[0] != ptrs[1]
                if previous_ptrs is not None:
                    assert ptrs == previous_ptrs
                previous_ptrs = ptrs
        elif graph:
            for builder in builders[2:]:
                assert torch.count_nonzero(builder.spec_query_start_loc[:4]) == 0
                assert torch.count_nonzero(builder.num_accepted_tokens[:3]) == 0


def test_gdn_reuse_mixed_prefill_keeps_physical_states_independent():
    device = torch.device("cpu")
    builders = [_make_builder(device=device, num_heads=32, num_speculative_tokens=3) for _ in range(4)]
    drafts = torch.tensor([3, -1, -1], dtype=torch.int32)
    accepted = torch.tensor([2, 1, 1], dtype=torch.int32)
    common = make_gdn_common(device, [4, 7, 1], [15, 20, 35], drafts.tolist())
    shared = {}
    for group in range(2):
        local = common.replace(block_table_tensor=common.block_table_tensor + group * 20)
        expected = builders[group].build(0, local, accepted, drafts)
        actual = builders[group + 2].build(0, local, accepted, drafts, batch_metadata_cache=shared)
        assert_metadata_equal(actual, expected)
    assert len(shared) == 1


def make_fia_builder(device, *, sliding=False):
    # Use the real build path without requiring a model or distributed group.
    builder = AscendAttentionMetadataBuilder.__new__(AscendAttentionMetadataBuilder)
    builder.device = device
    builder.pcp_enabled = False
    builder.decode_threshold = 4
    builder.model_config = SimpleNamespace(runner_type="generate")
    builder.vllm_config = SimpleNamespace(model_config=builder.model_config)
    builder.speculative_config = SimpleNamespace(parallel_drafting=True, use_dspark=lambda: False)
    builder.kv_cache_spec = (
        SlidingWindowSpec(block_size=640, num_kv_heads=4, head_size=128, dtype=torch.bfloat16, sliding_window=4096)
        if sliding
        else FullAttentionSpec(block_size=640, num_kv_heads=2, head_size=256, dtype=torch.bfloat16)
    )
    builder.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda causal, config: None)
    return builder


def test_fia_reuses_exact_lengths_but_not_physical_maps(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    builders = [make_fia_builder(torch.device("cpu"), sliding=bool(i)) for i in range(2)]
    common = make_gdn_common(torch.device("cpu"), [4, 4], [20, 30], [3, 3])
    common.attn_state = AscendAttentionState.SpecDecoding
    # Actual device lengths must win over the CPU upper bound in DFlash.
    common._seq_lens_cpu = torch.tensor([100, 100], dtype=torch.int32)
    common.seq_lens_cpu = common._seq_lens_cpu
    common.seq_lens_cpu_upper_bound = common._seq_lens_cpu
    for values in ([20, 30], [28, 35]):
        common.seq_lens.copy_(torch.tensor(values))
        shared, result = {}, []
        for group, builder in enumerate(builders):
            local = common.replace(
                block_table_tensor=common.block_table_tensor + 100 * group,
                slot_mapping=common.slot_mapping + 1000 * group,
                causal=bool(group),
            )
            result.append(builder.build(0, local, batch_metadata_cache=shared))
        assert len(shared) == 1
        assert result[0].seq_lens_list == result[1].seq_lens_list == list(values)
        assert result[0].query_start_loc is result[1].query_start_loc
        assert result[0].seq_lens_list is result[1].seq_lens_list
        assert not torch.equal(result[0].block_tables, result[1].block_tables)
        assert not torch.equal(result[0].slot_mapping, result[1].slot_mapping)
        assert result[0].causal is False and result[1].causal is True


def test_tensor_view_key_distinguishes_shape_stride_and_offset():
    values = torch.arange(16).reshape(4, 4)
    assert tensor_view_key(values) != tensor_view_key(values.T)
    assert tensor_view_key(values) != tensor_view_key(values.reshape(2, 8))
    assert tensor_view_key(values[0]) != tensor_view_key(values[1])
    assert tensor_view_key(values[:]) == tensor_view_key(values)


def test_metadata_reuse_env_is_default_off_and_strict(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_REUSE_BATCH_METADATA", raising=False)
    assert envs.VLLM_ASCEND_REUSE_BATCH_METADATA is False
    monkeypatch.setenv("VLLM_ASCEND_REUSE_BATCH_METADATA", "1")
    assert envs.VLLM_ASCEND_REUSE_BATCH_METADATA is True
    monkeypatch.setenv("VLLM_ASCEND_REUSE_BATCH_METADATA", "yes")
    with pytest.raises(ValueError):
        _ = envs.VLLM_ASCEND_REUSE_BATCH_METADATA


def test_gdn_shared_request_helper_runs_once_for_24_groups():
    builders = [_make_builder(device=torch.device("cpu"), num_heads=32, num_speculative_tokens=3) for _ in range(24)]
    common = make_gdn_common(torch.device("cpu"), [4, 4], [20, 30], [3, 3])
    accepted = torch.tensor([2, 4], dtype=torch.int32)
    drafts = torch.tensor([3, 3], dtype=torch.int32)
    original = type(builders[0])._build_request_metadata
    calls = []

    def counted(self, *args):
        calls.append(self)
        return original(self, *args)

    with patch.object(type(builders[0]), "_build_request_metadata", counted):
        for shared in (None, {}, {}):
            before = len(calls)
            for index, builder in enumerate(builders):
                builder.build(
                    0,
                    common.replace(block_table_tensor=common.block_table_tensor + index * 10),
                    accepted,
                    drafts,
                    batch_metadata_cache=shared,
                )
            assert len(calls) - before == (24 if shared is None else 1)
