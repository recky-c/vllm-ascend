# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from vllm.v1.core.kv_cache_utils import is_kv_cache_spec_uniform
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec


@pytest.mark.parametrize('indexer_first', [False, True])
def test_mixed_glm_cache_kinds_are_nonuniform_without_attribute_error(indexer_first):
    main = AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=656,
                                 dtype=torch.int8, cache_sparse_sfa_c8=True)
    indexer = AscendSFAIndexerCacheSpec(block_size=128, num_kv_heads=1, head_size=128,
                                      dtype=torch.int8, scale_dim=1, scale_dtype=torch.float16,
                                      cache_sparse_li_c8=True)
    items = [('main', main), ('indexer', indexer)]
    if indexer_first:
        items.reverse()
    assert not is_kv_cache_spec_uniform(dict(items))
    assert is_kv_cache_spec_uniform({'a': main, 'b': main})
    assert is_kv_cache_spec_uniform({'a': indexer, 'b': indexer})
