# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder, FIAParamProvider
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph, register_task
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.utils import AscendKVBlockZeroer
from vllm_ascend.worker.v2.attn_utils import _get_dflash_draft_fia_seq_lens_cpu
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator


@pytest.fixture(scope="module", autouse=True)
def init_zeroer_device_properties():
    init_device_properties_triton()


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("poison_recycled", [False, True])
def test_optimistic_fia_full_graph_matches_eager_bound(sliding, poison_recycled, monkeypatch):
    """Exercise native FIA and the production graph parameter provider.

    This checks the intentional upper-bound attention, not accuracy equivalence
    to exact attention or an end-to-end draft acceptance/performance result.
    """
    torch.npu.set_device(0)
    torch.manual_seed(7)
    device = torch.device("npu")
    block_size, num_heads, num_kv_heads, head_size = 128, 4, 2, 64
    query = torch.randn(12, num_heads, head_size, device=device, dtype=torch.bfloat16)
    key = torch.randn(67, block_size, num_kv_heads * head_size, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    key[0].zero_()
    value[0].zero_()
    if poison_recycled:
        # Finite GDN FP32 state bytes have a BF16 NaN in their lower half.
        # These two physical blocks are newly allocated; old valid blocks
        # must survive clearing all aliased Full/SWA K/V payloads.
        previous_key, previous_value = key.clone(), value.clone()
        new_ids = [2, 66]
        for cache in (key, value):
            for block_id in new_ids:
                cache[block_id].view(torch.int32).fill_(0x3F807FC1)
        assert torch.isfinite(key[2].view(torch.float32)).all()
        assert not torch.isfinite(key[2]).all()
        specs = [
            FullAttentionSpec(
                block_size=block_size, num_kv_heads=num_kv_heads, head_size=head_size, dtype=torch.bfloat16
            ),
            SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=torch.bfloat16,
                sliding_window=4096,
            ),
        ]
        groups = [
            SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=index, layer_names=[str(index)])
            for index, spec in enumerate(specs)
        ]
        zeroer = AscendKVBlockZeroer(device, pin_memory=False)
        zeroer.init_meta(
            groups,
            [block_size, block_size],
            "auto",
            set(),
            {str(index): SimpleNamespace(kv_cache=(key, value)) for index in range(2)},
            include_sliding_window=True,
            num_blocks=67,
        )
        zeroer.zero_block_ids(new_ids)
        torch.npu.synchronize()
        for cache, previous in ((key, previous_key), (value, previous_value)):
            for block_id in new_ids:
                assert torch.count_nonzero(cache[block_id]) == 0
            for block_id in (0, 1, 3, 65):
                torch.testing.assert_close(cache[block_id], previous[block_id], rtol=0, atol=0)
            # Forward fills only the valid prefix of the new long-request page;
            # its unwritten optimistic tail must retain finite zeros.
            cache[66, :125].copy_(previous[66, :125])
    # Different per-request lengths, a page-boundary crossing, model limit,
    # and a dummy request. SWA includes already-recycled prefix pages.
    block_table_cpu = torch.zeros(3, 64, dtype=torch.int32)
    block_table_cpu[0, :2] = torch.tensor([1, 2])
    block_table_cpu[1] = torch.arange(3, 67)
    if sliding:
        block_table_cpu[1, :31] = 0
    block_table = block_table_cpu.to(device)
    exact = torch.tensor([127, 8189, 8], dtype=torch.int32, device=device)
    query_start_loc_cpu = torch.tensor([0, 4, 8, 8], dtype=torch.int32)
    common = AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu.to(device),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=exact,
        _seq_lens_cpu=torch.tensor([130, 8192, 0], dtype=torch.int32),
        num_reqs=3,
        num_actual_tokens=12,
        num_input_tokens=12,
        max_query_len=4,
        max_seq_len=8192,
        block_table_tensor=block_table,
        slot_mapping=torch.full((12,), -1, dtype=torch.int32, device=device),
        causal=sliding,
    )
    # Use the real metadata build without initializing a distributed model.
    builder = object.__new__(AscendAttentionMetadataBuilder)
    builder.device, builder.pcp_enabled, builder.decode_threshold = device, False, 4
    builder.model_config = SimpleNamespace(runner_type="generate")
    builder.vllm_config = SimpleNamespace(model_config=builder.model_config)
    builder.speculative_config = SimpleNamespace(parallel_drafting=True, use_dspark=lambda: False)
    builder.kv_cache_spec = (
        SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
            sliding_window=4096,
        )
        if sliding
        else FullAttentionSpec(
            block_size=block_size, num_kv_heads=num_kv_heads, head_size=head_size, dtype=torch.bfloat16
        )
    )
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device=device), diagonal=1) if sliding else None
    builder.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda causal, config: mask)
    speculator = object.__new__(AscendDFlashSpeculator)
    speculator.num_query_per_req = 4
    capture_metadata = builder.build_for_cudagraph_capture(common)
    speculator._update_draft_attn_metadata({"draft": capture_metadata}, 3)
    assert capture_metadata.seq_lens_list == [127, 8189, 8]
    provider = FIAParamProvider("draft", 4096 if sliding else None, is_draft_model=True)
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        atten_mask=mask,
        block_table=block_table,
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=[4, 8, 12],
        actual_seq_lengths_kv=[127, 8189, 8],
        num_key_value_heads=num_kv_heads,
        num_heads=num_heads,
        pre_tokens=4096 if sliding else 2147483647,
        next_tokens=0 if sliding else 2147483647,
        scale=head_size**-0.5,
        sparse_mode=4 if sliding else 0,
    )
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**kwargs)
    output = torch.empty_like(query)
    lse = torch.empty(1, dtype=query.dtype, device=device)
    kwargs.update(workspace=workspace, out=[output, lse])
    stream = torch.npu.Stream()
    update_stream = torch.npu.Stream()
    graph = UpdatableGraph()
    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        for _ in range(2):
            torch_npu.npu_fused_infer_attention_score.out(**kwargs)
        stream.synchronize()
        with torch.npu.graph(graph, stream=stream):
            register_task(torch_npu.npu_fused_infer_attention_score.out, kwargs, provider)
    torch.npu.synchronize()
    tolist = torch.Tensor.tolist

    def no_exact_readback(tensor):
        assert tensor.data_ptr() != exact.data_ptr(), "draft performed an exact device length readback"
        return tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", no_exact_readback)
    for values, bounds in [([127, 8189, 8], [130, 8192, 0]), ([128, 8190, 8], [131, 8192, 0])]:
        exact.copy_(torch.tensor(values, dtype=torch.int32, device=device))
        common.dflash_draft_seq_lens_cpu_upper_bound = _get_dflash_draft_fia_seq_lens_cpu(
            torch.tensor(bounds, dtype=torch.int32), query_start_loc_cpu, 3, 8192
        )
        metadata = builder.build(0, common)
        speculator._update_draft_attn_metadata({"draft": metadata}, 3)
        assert metadata.seq_lens_gpu is exact and metadata.seq_lens is exact
        assert metadata.seq_lens_list == [bounds[0], bounds[1], 1]
        tasks = graph.resolve_tasks(ContextSource({"draft": metadata}))
        assert tasks[0].kwargs["actual_seq_lengths_kv"] == metadata.seq_lens_list
        graph.update(update_stream, tasks)
        with torch.npu.stream(stream):
            graph.replay()
        torch.npu.synchronize()
        actual = output.clone()
        assert torch.isfinite(actual).all()
        torch_npu.npu_fused_infer_attention_score.out(**{**kwargs, **provider.resolve({"draft": metadata})})
        torch.npu.synchronize()
        torch.testing.assert_close(actual, output, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("sliding", [False, True])
def test_poisoned_recycled_pages_zero_only_requested_payloads(sliding):
    """Keep old logical blocks and per-subblock shared-page padding intact."""
    device = torch.device("npu")
    raw_buffers = [torch.full((8, 16), 0x3F807FC1, dtype=torch.int32, device=device) for _ in range(2)]
    caches = [torch.as_strided(raw.view(torch.bfloat16), (8, 4, 1, 2), (32, 2, 2, 1)) for raw in raw_buffers]
    for raw, cache in zip(raw_buffers, caches):
        assert torch.isfinite(raw.view(torch.float32)).all()
        assert not torch.isfinite(cache).all()
    spec_kwargs = dict(block_size=8, num_kv_heads=1, head_size=2, dtype=torch.bfloat16)
    spec = SlidingWindowSpec(**spec_kwargs, sliding_window=16) if sliding else FullAttentionSpec(**spec_kwargs)
    group = SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0, layer_names=["draft"])
    zeroer = AscendKVBlockZeroer(device, pin_memory=False)
    zeroer.init_meta(
        [group],
        [4],
        "auto",
        set(),
        {"draft": SimpleNamespace(kv_cache=tuple(caches))},
        include_sliding_window=True,
        num_blocks=4,
    )
    snapshots = [raw.clone() for raw in raw_buffers]
    # Scheduler logical IDs 1 and 3 expand to kernel rows [2,3] and [6,7].
    zeroer.zero_block_ids([1, 3])
    torch.npu.synchronize()
    for raw, cache, previous in zip(raw_buffers, caches, snapshots):
        for row in (2, 3, 6, 7):
            assert torch.count_nonzero(cache[row]) == 0
            torch.testing.assert_close(raw[row, 4:], previous[row, 4:], rtol=0, atol=0)
        for row in (0, 1, 4, 5):
            torch.testing.assert_close(raw[row], previous[row], rtol=0, atol=0)
