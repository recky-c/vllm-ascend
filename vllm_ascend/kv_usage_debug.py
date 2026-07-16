# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Part of the debug/kv-cache-memory-inspect branch.
#
# Usage-phase KV debug helpers. Always-on on this debug branch (no env gate).
# Init-phase logs under [KV_DEBUG] are also always-on.
#
# IMPORTANT: use vllm.logger (same as model_runner init [KV_DEBUG] lines).
# A bare logging.getLogger("vllm_ascend.*") often does not show in EngineCore
# tee'd serve logs even when VLLM_ASCEND_KV_USAGE_DEBUG=1 is set.

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vllm.logger import logger

KV_DEBUG_TAG = "[KV_DEBUG]"

_step_count = 0
_attn_write_logs = 0
_attn_read_logs = 0

# Cap hot-path spam: first N steps fully, then sample.
_DETAIL_FIRST_STEPS = 8
_DETAIL_EVERY = 32
_MAX_ATTN_LOGS = 24


def kv_usage_debug_enabled() -> bool:
    """Always enabled on debug/kv-cache-memory-inspect."""
    return True


def kv_block_ids_summary(blocks: Any, max_items: int = 16) -> Any:
    """Summarize block ids for logs (vLLM 0.23.0 lacks vllm.v1.kv_debug)."""
    if blocks is None:
        return None
    if hasattr(blocks, "get_block_ids"):
        try:
            return blocks.get_block_ids(allow_none=True)
        except TypeError:
            return blocks.get_block_ids()
    if isinstance(blocks, Iterable) and not isinstance(blocks, (str, bytes, dict)):
        ids = []
        for idx, block in enumerate(blocks):
            if idx >= max_items:
                ids.append("...")
                break
            ids.append(getattr(block, "block_id", block))
        return ids
    return blocks


def begin_forward_step() -> int:
    """Call once per model-runner execute step. Returns the step id."""
    global _step_count
    _step_count += 1
    return _step_count


def current_step() -> int:
    return _step_count


def should_log_detail() -> bool:
    step = _step_count
    if step <= _DETAIL_FIRST_STEPS:
        return True
    return step % _DETAIL_EVERY == 0


def should_log_attn_write() -> bool:
    global _attn_write_logs
    if not should_log_detail():
        return False
    if _attn_write_logs >= _MAX_ATTN_LOGS:
        return False
    _attn_write_logs += 1
    return True


def should_log_attn_read() -> bool:
    global _attn_read_logs
    if not should_log_detail():
        return False
    if _attn_read_logs >= _MAX_ATTN_LOGS:
        return False
    _attn_read_logs += 1
    return True


def _preview_1d(tensor: Any, limit: int = 16) -> list[int]:
    if tensor is None:
        return []
    try:
        flat = tensor.detach().flatten()[:limit]
        if flat.is_cuda or str(flat.device).startswith("npu"):
            flat = flat.cpu()
        return [int(x) for x in flat.tolist()]
    except Exception:
        return []


def _preview_block_table(block_table: Any, max_reqs: int = 2, max_blocks: int = 8) -> list[list[int]]:
    if block_table is None:
        return []
    try:
        bt = block_table.detach()
        if bt.is_cuda or str(bt.device).startswith("npu"):
            bt = bt.cpu()
        rows = []
        for i in range(min(max_reqs, bt.shape[0])):
            row = bt[i, : min(max_blocks, bt.shape[1])]
            # trim trailing zeros for readability
            vals = [int(x) for x in row.tolist()]
            while vals and vals[-1] == 0:
                vals.pop()
            rows.append(vals)
        return rows
    except Exception:
        return []


def log_alloc(
    req_id: str,
    num_new_tokens: int,
    num_new_computed: int,
    free_before: int,
    free_after: int | None,
    new_block_ids: Any,
    ok: bool,
) -> None:
    logger.info(
        "%s: usage.allocate req_id=%s ok=%s num_new_tokens=%s "
        "num_new_computed=%s free_blocks %s->%s new_blocks=%s",
        KV_DEBUG_TAG,
        req_id,
        ok,
        num_new_tokens,
        num_new_computed,
        free_before,
        free_after if free_after is not None else "n/a",
        new_block_ids,
    )


def log_free(req_id: str, free_before: int, free_after: int) -> None:
    logger.info(
        "%s: usage.free req_id=%s free_blocks %s->%s",
        KV_DEBUG_TAG,
        req_id,
        free_before,
        free_after,
    )


def log_slot_mapping(
    step: int,
    kv_cache_gid: int,
    num_reqs: int,
    num_tokens: int,
    block_table: Any,
    slot_mapping: Any,
    block_size: int | None = None,
) -> None:
    if not should_log_detail():
        return
    slots = _preview_1d(slot_mapping, limit=24)
    bt_preview = _preview_block_table(block_table)
    logger.info(
        "%s: usage.slot_mapping step=%s gid=%s num_reqs=%s num_tokens=%s "
        "block_size=%s formula='slot=block_id*block_size+offset' "
        "block_table_preview=%s slot_mapping_preview=%s",
        KV_DEBUG_TAG,
        step,
        kv_cache_gid,
        num_reqs,
        num_tokens,
        block_size,
        bt_preview,
        slots,
    )


def log_kv_write(
    layer_name: str,
    attn_state: Any,
    num_tokens: int,
    slot_mapping: Any,
    key_cache_shape: Any,
    value_cache_shape: Any,
    backend: str = "?",
) -> None:
    if not should_log_attn_write():
        return
    logger.info(
        "%s: usage.kv_write backend=%s layer=%s attn_state=%s num_tokens=%s "
        "slots=%s key_cache_shape=%s value_cache_shape=%s "
        "note='writes K/V into kv_cache[slot]'",
        KV_DEBUG_TAG,
        backend,
        layer_name,
        attn_state,
        num_tokens,
        _preview_1d(slot_mapping, limit=16),
        list(key_cache_shape) if key_cache_shape is not None else None,
        list(value_cache_shape) if value_cache_shape is not None else None,
    )


def log_kv_read(
    layer_name: str,
    attn_state: Any,
    num_tokens: int,
    block_table: Any,
    key_cache_shape: Any,
    backend: str = "?",
) -> None:
    if not should_log_attn_read():
        return
    logger.info(
        "%s: usage.kv_read backend=%s layer=%s attn_state=%s num_tokens=%s "
        "block_table_preview=%s key_cache_shape=%s "
        "note='attention gathers history K/V via block_table (+topk for SFA)'",
        KV_DEBUG_TAG,
        backend,
        layer_name,
        attn_state,
        num_tokens,
        _preview_block_table(block_table),
        list(key_cache_shape) if key_cache_shape is not None else None,
    )
