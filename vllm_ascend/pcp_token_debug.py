# Temporary debug helpers for PCP + FlashComm1 RS padding (remove after fix is validated).

from __future__ import annotations

from typing import Any

from vllm_ascend.utils import sp_pad_size_for_tp

PCP_TOKEN_DEBUG_MAX_LAYER = 8


def pcp_token_debug_enabled(
    *,
    flash_comm_v1: bool = False,
    pcp_size: int = 1,
    max_tokens_across_pcp: int = 0,
) -> bool:
    return bool(flash_comm_v1 and (pcp_size > 1 or max_tokens_across_pcp > 0))


def _resolve_layer_idx(extra_ctx: Any, layer_idx: int | None) -> int | None:
    if layer_idx is not None:
        return layer_idx
    return getattr(extra_ctx, "layer_idx", None)


def should_log_pcp_token_layer(
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
    return resolved <= PCP_TOKEN_DEBUG_MAX_LAYER


def log_pcp_token(
    tag: str,
    extra_ctx: Any | None = None,
    *,
    layer_idx: int | None = None,
    always: bool = False,
    **fields: Any,
) -> None:
    if extra_ctx is not None:
        try:
            if not pcp_token_debug_enabled(
                flash_comm_v1=extra_ctx.flash_comm_v1_enabled,
                max_tokens_across_pcp=getattr(extra_ctx, "max_tokens_across_pcp", 0),
            ):
                return
        except (AssertionError, AttributeError):
            return
        if not should_log_pcp_token_layer(extra_ctx, layer_idx, always=always):
            return

    parts = [f"[PCP-TOKEN-DEBUG][{tag}]"]
    try:
        from vllm.distributed import get_pcp_group, get_tensor_model_parallel_rank

        parts.append(f"pcp_rank={get_pcp_group().rank_in_group}")
        parts.append(f"tp_rank={get_tensor_model_parallel_rank()}")
    except Exception:
        pass

    if extra_ctx is not None:
        resolved_layer = _resolve_layer_idx(extra_ctx, layer_idx)
        if resolved_layer is not None:
            parts.append(f"layer_idx={resolved_layer}")
        try:
            parts.append(f"context_pad_size={extra_ctx.pad_size}")
            parts.append(f"context_num_tokens={extra_ctx.num_tokens}")
        except Exception:
            pass

    for key, value in fields.items():
        parts.append(f"{key}={value}")

    print(" ".join(parts), flush=True)
