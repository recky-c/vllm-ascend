# Temporary debug helpers for PCP + FlashComm1 (remove after root cause is fixed).

from __future__ import annotations

from typing import Any

PCP_FC1_DEBUG_MAX_LAYER = 8


def sp_pad_size_for_tp(num_tokens: int, tp_world_size: int) -> int:
    """Rows to append so num_tokens is divisible by tp_world_size (RS input)."""
    return (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size


def pcp_fc1_debug_enabled(extra_ctx: Any) -> bool:
    try:
        return bool(
            extra_ctx.flash_comm_v1_enabled and extra_ctx.max_tokens_across_pcp > 0
        )
    except (AssertionError, AttributeError):
        return False


def _resolve_layer_idx(extra_ctx: Any, layer_idx: int | None) -> int | None:
    if layer_idx is not None:
        return layer_idx
    return getattr(extra_ctx, "layer_idx", None)


def should_log_pcp_fc1_layer(
    extra_ctx: Any,
    layer_idx: int | None = None,
    *,
    always: bool = False,
) -> bool:
    if always:
        return True
    resolved = _resolve_layer_idx(extra_ctx, layer_idx)
    if resolved is None:
        return True
    return resolved <= PCP_FC1_DEBUG_MAX_LAYER


def log_pcp_fc1(
    tag: str,
    extra_ctx: Any,
    *,
    layer_idx: int | None = None,
    always: bool = False,
    **fields: Any,
) -> None:
    if not pcp_fc1_debug_enabled(extra_ctx):
        return
    if not should_log_pcp_fc1_layer(extra_ctx, layer_idx, always=always):
        return

    parts = [f"[PCP-FC1-DEBUG][{tag}]"]
    try:
        from vllm.distributed import get_pcp_group, get_tensor_model_parallel_rank

        parts.append(f"pcp_rank={get_pcp_group().rank_in_group}")
        parts.append(f"tp_rank={get_tensor_model_parallel_rank()}")
    except Exception:
        pass

    resolved_layer = _resolve_layer_idx(extra_ctx, layer_idx)
    if resolved_layer is not None:
        parts.append(f"layer_idx={resolved_layer}")

    try:
        parts.append(f"pad_size={extra_ctx.pad_size}")
    except Exception:
        pass

    for key, value in fields.items():
        parts.append(f"{key}={value}")

    print(" ".join(parts), flush=True)
