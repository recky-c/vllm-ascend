# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_npu  # noqa: F401
from vllm.config.compilation import CUDAGraphMode

from tests.ut.attention.test_batch_metadata_reuse import assert_metadata_equal, make_gdn_common
from tests.ut.ops.test_gdn_attn_builder import _make_builder


@torch.inference_mode()
def test_capture_entry_then_shared_runtime_keeps_group_buffers():
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    builders = [
        _make_builder(
            device=device, num_heads=32, num_speculative_tokens=3, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        )
        for _ in range(4)
    ]
    common = make_gdn_common(device, [4, 4, 4], [40, 48, 56], [3, 3, 3])
    locals_ = [common.replace(block_table_tensor=common.block_table_tensor + 100 * g) for g in range(2)]
    metadata = [b.build_for_cudagraph_capture(m) for b, m in zip(builders[:2], locals_)]
    fields = ("spec_state_indices_tensor", "spec_sequence_masks", "spec_query_start_loc", "num_accepted_tokens")
    pointers = [[getattr(m, f).data_ptr() for f in fields] for m in metadata]
    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        consumer = [[getattr(m, f).clone() for f in fields] for m in metadata]
    drafts = torch.tensor([3, 3, -1], dtype=torch.int32)
    accepted = torch.tensor([2, 4, 1], dtype=torch.int32, device=device)
    common.seq_lens_cpu_upper_bound[-1] = 0
    common.seq_lens[-1] = 0
    for step in (1, 2):
        accepted.copy_(torch.tensor([step + 1, 5 - step, 1], dtype=torch.int32, device=device))
        cache = {}
        current = []
        for group, local in enumerate(locals_):
            local.num_actual_tokens = 8
            local.block_table_tensor.add_(1000)
            actual = builders[group].build(0, local, accepted, drafts, num_actual_reqs=2, batch_metadata_cache=cache)
            expected = builders[group + 2].build(0, local, accepted, drafts, num_actual_reqs=2)
            assert_metadata_equal(actual, expected)
            current.append(actual)
        assert len(cache) == 1
        assert pointers == [[getattr(m, f).data_ptr() for f in fields] for m in current]
        graph.replay()
        torch.npu.synchronize()
        for group, m in enumerate(current):
            for f, output in zip(fields, consumer[group]):
                torch.testing.assert_close(output, getattr(m, f), rtol=0, atol=0)
