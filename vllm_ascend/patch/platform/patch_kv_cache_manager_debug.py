#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project (debug/kv-cache-memory-inspect).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrap KVCacheManager.allocate_slots / free with [KV_DEBUG] usage logs."""

from __future__ import annotations

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_debug import kv_block_ids_summary

from vllm_ascend.kv_usage_debug import log_alloc, log_free

_original_allocate_slots = KVCacheManager.allocate_slots
_original_free = KVCacheManager.free


def _patched_allocate_slots(self, request, num_new_tokens, *args, **kwargs):
    free_before = self.block_pool.get_num_free_blocks()
    num_new_computed = kwargs.get("num_new_computed_tokens", 0)
    if args:
        # positional: num_new_computed_tokens is first optional after num_new_tokens
        num_new_computed = args[0] if len(args) >= 1 else num_new_computed

    result = _original_allocate_slots(self, request, num_new_tokens, *args, **kwargs)
    free_after = self.block_pool.get_num_free_blocks()
    log_alloc(
        req_id=request.request_id,
        num_new_tokens=num_new_tokens,
        num_new_computed=num_new_computed,
        free_before=free_before,
        free_after=free_after if result is not None else None,
        new_block_ids=kv_block_ids_summary(result) if result is not None else None,
        ok=result is not None,
    )
    return result


def _patched_free(self, request) -> None:
    free_before = self.block_pool.get_num_free_blocks()
    _original_free(self, request)
    free_after = self.block_pool.get_num_free_blocks()
    log_free(request.request_id, free_before, free_after)


KVCacheManager.allocate_slots = _patched_allocate_slots  # type: ignore[method-assign]
KVCacheManager.free = _patched_free  # type: ignore[method-assign]
