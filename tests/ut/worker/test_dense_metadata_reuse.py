# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration checks for invocation-local dense metadata reuse."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_ascend.attention import attention_v1
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.worker.v2 import attn_utils


def _config(*, parallel_drafting=True, use_v2=True):
    return SimpleNamespace(
        use_v2_model_runner=use_v2,
        model_config=SimpleNamespace(runner_type="generate", max_model_len=4096),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            parallel_drafting=parallel_drafting,
            use_dspark=lambda: False,
        ),
    )


@pytest.fixture(autouse=True)
def _cpu_attention_dependencies(monkeypatch):
    # Keep real constructors/build/update methods, replacing only CPU harness
    # dependencies. Equivalent groups receive the same singleton-style mask.
    mask = torch.triu(torch.ones((4, 4), dtype=torch.bool), diagonal=1)

    def make_mask_builder(_device):
        return SimpleNamespace(
            get_attention_mask=MagicMock(side_effect=lambda causal, _model: mask if causal else None)
        )

    monkeypatch.setattr(attention_v1, "AttentionMaskBuilder", make_mask_builder)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor, *_args, **_kwargs: tensor)


def _make_inputs(monkeypatch, *, num_groups=24, with_padding=False, spec_type=FullAttentionSpec, configs=None):
    if configs is None:
        configs = [_config()] * num_groups
    spec = spec_type(
        block_size=128,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
        **({"sliding_window": 4096} if spec_type is SlidingWindowSpec else {}),
    )
    builders = []
    groups = []
    for i, config in enumerate(configs):
        builder = AscendAttentionMetadataBuilder(spec, [f"dense.{i}"], config, torch.device("cpu"))
        for method in ("build", "build_for_cudagraph_capture", "update_block_table", "_build_batch_metadata"):
            monkeypatch.setattr(builder, method, MagicMock(wraps=getattr(builder, method)))
        builders.append(builder)
        groups.append(
            SimpleNamespace(
                kv_cache_spec=spec,
                layer_names=[f"dense.{i}"],
                get_metadata_builder=lambda _, builder=builder: builder,
            )
        )
    offsets = [0, 4, 8, 12] if with_padding else [0, 4, 8]
    query_cpu = torch.tensor(offsets, dtype=torch.int32)
    kwargs = dict(
        attn_groups=[[group] for group in groups],
        num_reqs=len(offsets) - 1,
        num_actual_reqs=2,
        num_tokens=offsets[-1],
        num_actual_tokens=8,
        query_start_loc_gpu=query_cpu.clone(),
        query_start_loc_cpu=query_cpu,
        max_query_len=4,
        seq_lens=torch.tensor([20, 30], dtype=torch.int32),
        max_seq_len=300,
        block_tables=tuple(torch.arange(4, dtype=torch.int32).reshape(2, 2) + i * 100 for i in range(num_groups)),
        slot_mappings=tuple(torch.arange(offsets[-1], dtype=torch.int64) + i * 128 for i in range(num_groups)),
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        seq_lens_np=np.array([200, 300], dtype=np.int32),
        attn_state=AscendAttentionState.SpecDecoding,
    )
    return builders, kwargs


def _assert_physical_mappings(result, kwargs, *, padded):
    metadata = list(result.values())
    assert all(isinstance(item, AscendMetadata) for item in metadata)
    assert len({id(item) for item in metadata}) == len(metadata)
    assert len({item.block_tables.data_ptr() for item in metadata}) == len(metadata)
    assert len({item.slot_mapping.data_ptr() for item in metadata}) == len(metadata)
    for i, item in enumerate(metadata):
        torch.testing.assert_close(item.block_tables[:2], kwargs["block_tables"][i])
        if padded:
            assert item.block_tables.shape == (3, 2)
            torch.testing.assert_close(item.block_tables[2], torch.zeros(2, dtype=torch.int32))
        else:
            assert item.block_tables is kwargs["block_tables"][i]
        assert item.slot_mapping.numel() == 8
        assert item.slot_mapping.data_ptr() == kwargs["slot_mappings"][i].data_ptr()
        torch.testing.assert_close(item.slot_mapping, kwargs["slot_mappings"][i][:8])


@pytest.mark.parametrize("spec_type", [FullAttentionSpec, SlidingWindowSpec])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("legacy_cache", [False, True])
def test_dense_24_groups_share_real_request_metadata(monkeypatch, spec_type, padded, legacy_cache):
    monkeypatch.setenv("VLLM_ASCEND_REUSE_BATCH_METADATA", str(int(legacy_cache)))
    builders, kwargs = _make_inputs(monkeypatch, with_padding=padded, spec_type=spec_type)
    assert all(builder.supports_update_block_table for builder in builders)

    result = attn_utils.build_attn_metadata(**kwargs)

    assert builders[0].build.call_count == 1
    assert builders[0].update_block_table.call_count == 0
    assert all(builder.build.call_count == 0 and builder.update_block_table.call_count == 1 for builder in builders[1:])
    assert sum(builder._build_batch_metadata.call_count for builder in builders) == 1
    assert all("common_fia_metadata" in call.kwargs for call in builders[0].build.call_args_list)
    assert ("batch_metadata_cache" in builders[0].build.call_args.kwargs) is legacy_cache
    _assert_physical_mappings(result, kwargs, padded=padded)
    metadata = list(result.values())
    for attribute in (
        "query_start_loc",
        "seq_lens",
        "seq_lens_cpu",
        "seq_lens_list",
        "actual_seq_lengths_q",
        "attn_mask",
    ):
        assert len({id(getattr(item, attribute)) for item in metadata}) == 1
    for item in metadata:
        assert item.seq_lens_list == ([20, 30, 1] if padded else [20, 30])
        assert item.actual_seq_lengths_q == ([4, 8, 12] if padded else [4, 8])
        assert item.num_actual_tokens == 8
        assert item.reshape_cache_event is None
    assert builders[0].attn_mask_builder.get_attention_mask.call_count == 1
    assert all(builder.attn_mask_builder.get_attention_mask.call_count == 0 for builder in builders[1:])


def test_dense_reuse_is_scoped_to_each_invocation(monkeypatch):
    builders, kwargs = _make_inputs(monkeypatch, with_padding=True)
    first = attn_utils.build_attn_metadata(**kwargs)
    kwargs["query_start_loc_cpu"][1] = 3
    kwargs["query_start_loc_gpu"].copy_(kwargs["query_start_loc_cpu"])
    kwargs["seq_lens"].copy_(torch.tensor([21, 34], dtype=torch.int32))
    kwargs["max_query_len"] = 5
    second = attn_utils.build_attn_metadata(**kwargs)

    assert builders[0].build.call_count == 2
    assert all(builder.update_block_table.call_count == 2 for builder in builders[1:])
    _assert_physical_mappings(second, kwargs, padded=True)
    for i in range(24):
        old, new = first[f"dense.{i}"], second[f"dense.{i}"]
        assert old.seq_lens_list == [20, 30, 1]
        assert old.actual_seq_lengths_q == [4, 8, 12]
        assert new.seq_lens_list == [21, 34, 1]
        assert new.actual_seq_lengths_q == [3, 8, 12]
        assert (new.num_decodes, new.num_prefills, new.num_decode_tokens) == (1, 2, 3)
        assert new.seq_lens_list is not old.seq_lens_list
        assert new.actual_seq_lengths_q is not old.actual_seq_lengths_q


@pytest.mark.parametrize("legacy_cache", [False, True])
def test_dense_causality_partitions_templates_without_losing_fia_sharing(monkeypatch, legacy_cache):
    monkeypatch.setenv("VLLM_ASCEND_REUSE_BATCH_METADATA", str(int(legacy_cache)))
    builders, kwargs = _make_inputs(monkeypatch)
    kwargs["causal"] = {i: bool(i % 2) for i in range(24)}
    result = attn_utils.build_attn_metadata(**kwargs)

    assert [builder.build.call_count for builder in builders] == [1, 1] + [0] * 22
    assert sum(builder.update_block_table.call_count for builder in builders) == 22
    # Each causal mask needs its own full template, but common FIA conversions
    # still use the prior cache independently of the new template cache.
    assert sum(builder._build_batch_metadata.call_count for builder in builders) == 1
    assert len({id(item.seq_lens_list) for item in result.values()}) == 1
    for i, item in enumerate(result.values()):
        assert item.causal is bool(i % 2)
        assert (item.attn_mask is not None) is bool(i % 2)
    _assert_physical_mappings(result, kwargs, padded=False)


@pytest.mark.parametrize("parallel_drafting", [False, True])
def test_dense_separate_config_identities_require_separate_templates(monkeypatch, parallel_drafting):
    configs = [_config(parallel_drafting=parallel_drafting) for _ in range(2)]
    builders, kwargs = _make_inputs(monkeypatch, num_groups=2, configs=configs)
    result = attn_utils.build_attn_metadata(**kwargs)
    assert all(builder.build.call_count == 1 and builder.update_block_table.call_count == 0 for builder in builders)
    assert all(builder._build_batch_metadata.call_count == 1 for builder in builders)
    assert result["dense.0"].seq_lens_list is not result["dense.1"].seq_lens_list
    expected = [20, 30] if parallel_drafting else [200, 300]
    assert all(item.seq_lens_list == expected for item in result.values())


@pytest.mark.parametrize("legacy_cache", [False, True])
def test_dense_config_length_sources_do_not_reuse_upper_bounds(monkeypatch, legacy_cache):
    monkeypatch.setenv("VLLM_ASCEND_REUSE_BATCH_METADATA", str(int(legacy_cache)))
    configs = [_config(parallel_drafting=False), _config(parallel_drafting=True)]
    builders, kwargs = _make_inputs(monkeypatch, num_groups=2, configs=configs)
    result = attn_utils.build_attn_metadata(**kwargs)
    assert all(builder.build.call_count == 1 and builder.update_block_table.call_count == 0 for builder in builders)
    assert result["dense.0"].seq_lens_list == [200, 300]
    assert result["dense.1"].seq_lens_list == [20, 30]
    torch.testing.assert_close(result["dense.0"].seq_lens, torch.tensor([200, 300], dtype=torch.int32))
    torch.testing.assert_close(result["dense.1"].seq_lens, kwargs["seq_lens"])
    _assert_physical_mappings(result, kwargs, padded=False)


@pytest.mark.parametrize("spec_type", [FullAttentionSpec, SlidingWindowSpec])
@pytest.mark.parametrize("padded", [False, True])
def test_dense_capture_builds_each_group_and_retains_common_fia_cache(monkeypatch, spec_type, padded):
    monkeypatch.setenv("VLLM_ASCEND_REUSE_BATCH_METADATA", "1")
    builders, kwargs = _make_inputs(monkeypatch, with_padding=padded, spec_type=spec_type)
    result = attn_utils.build_attn_metadata(**kwargs, for_cudagraph_capture=True)
    assert all(builder.build_for_cudagraph_capture.call_count == 1 for builder in builders)
    assert all(builder.build.call_count == 1 and builder.update_block_table.call_count == 0 for builder in builders)
    assert sum(builder._build_batch_metadata.call_count for builder in builders) == 1
    assert all("batch_metadata_cache" not in builder.build.call_args.kwargs for builder in builders)
    assert len({id(item.actual_seq_lengths_q) for item in result.values()}) == 1
    assert all(builder.attn_mask_builder.get_attention_mask.call_count == 1 for builder in builders)
    _assert_physical_mappings(result, kwargs, padded=padded)


@pytest.mark.parametrize("missing_v2_flag", [False, True])
def test_dense_v1_or_old_configs_keep_individual_builds(monkeypatch, missing_v2_flag):
    config = _config(use_v2=False)
    if missing_v2_flag:
        del config.use_v2_model_runner
    builders, kwargs = _make_inputs(monkeypatch, configs=[config] * 24)
    assert not any(builder.supports_update_block_table for builder in builders)
    result = attn_utils.build_attn_metadata(**kwargs)
    assert all(builder.build.call_count == 1 and builder.update_block_table.call_count == 0 for builder in builders)
    assert sum(builder._build_batch_metadata.call_count for builder in builders) == 1
    _assert_physical_mappings(result, kwargs, padded=False)


def test_dense_update_clears_group_event_and_leaves_template_unchanged(monkeypatch):
    builders, kwargs = _make_inputs(monkeypatch, num_groups=2, with_padding=True)
    result = attn_utils.build_attn_metadata(**kwargs)
    template = result["dense.0"]
    event = object()
    template.reshape_cache_event = event
    updated = builders[1].update_block_table(template, kwargs["block_tables"][1], kwargs["slot_mappings"][1])
    assert updated is not template
    assert updated.reshape_cache_event is None
    assert template.reshape_cache_event is event
    assert template.block_tables.data_ptr() != updated.block_tables.data_ptr()
    assert template.slot_mapping.data_ptr() != updated.slot_mapping.data_ptr()
    torch.testing.assert_close(template.block_tables[:2], kwargs["block_tables"][0])
    torch.testing.assert_close(updated.block_tables[:2], kwargs["block_tables"][1])
    assert updated.seq_lens_list is template.seq_lens_list
