# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream token-fused MLA byte layout, including descriptor gaps."""

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheTensor

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.models.glm5next.cache_views import view_glm_main_descriptor


@pytest.mark.parametrize("rope_dim", [0, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mla_components_alias_upstream_token_records(rope_dim, dtype):
    nope_dim, tokens, blocks = 4, 128, 3
    width = nope_dim + rope_dim
    spec = AscendMLAAttentionSpec(
        block_size=tokens, num_kv_heads=1, head_size=width, dtype=dtype,
    )
    offset, padding = 32, 64
    stride = spec.page_size_bytes + padding
    raw = torch.full((offset + blocks * stride,), 19, dtype=torch.uint8)
    descriptor = KVCacheTensor(
        size=raw.numel(), layers=["main"], offset=offset,
        layer_stride=spec.page_size_bytes, block_stride=stride,
    )
    nope, rope = view_glm_main_descriptor(raw, spec, descriptor, blocks, tokens, nope_dim, rope_dim)
    assert nope.shape == (blocks, tokens, 1, nope_dim)
    assert rope.shape == (blocks, tokens, 1, rope_dim)
    assert nope.stride(1) == rope.stride(1) == width
    assert nope.stride(0) * nope.element_size() == stride
    assert rope.untyped_storage().data_ptr() == nope.untyped_storage().data_ptr()
    nope[1, 7].fill_(3)
    rope[1, 7].fill_(5)
    address = offset + stride + 7 * width * 2
    record = raw[address:address + width * 2].view(dtype)
    assert torch.all(record[:nope_dim] == 3)
    assert torch.all(record[nope_dim:] == 5)
    assert torch.all(raw[:offset] == 19)
    assert torch.all(raw[offset + spec.page_size_bytes:offset + stride] == 19)
    assert torch.all(raw[offset + 2 * stride:] == 19)


def test_mla_dense_manager_page_splits_into_kernel_token_records():
    spec = AscendMLAAttentionSpec(
        block_size=256, num_kv_heads=1, head_size=6, dtype=torch.float16,
    )
    raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.uint8)
    descriptor = KVCacheTensor(
        size=raw.numel(), layers=["main"], offset=0,
        layer_stride=raw.numel(), block_stride=spec.page_size_bytes,
    )
    nope, rope = view_glm_main_descriptor(raw, spec, descriptor, 2, 128, 4, 2)
    assert nope.shape == (4, 128, 1, 4)
    nope[1, 0].fill_(2)
    rope[1, 0].fill_(7)
    assert torch.equal(raw[128 * 12:129 * 12].view(torch.float16), torch.tensor([2, 2, 2, 2, 7, 7]).half())
