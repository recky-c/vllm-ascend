# SPDX-License-Identifier: Apache-2.0
"""Debug helpers for Eagle3 + DCP multi-step draft slot_mapping investigation."""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.logger import logger

_ENABLED: bool | None = None
_LOG_COUNT = 0
_MAX_LOGS: int | None = None


def is_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.getenv("VLLM_ASCEND_DEBUG_EAGLE_DCP", "0") not in ("", "0", "false", "False")
    return _ENABLED


def _max_logs() -> int:
    global _MAX_LOGS
    if _MAX_LOGS is None:
        _MAX_LOGS = int(os.getenv("VLLM_ASCEND_DEBUG_EAGLE_DCP_MAX_LOGS", "64"))
    return _MAX_LOGS


def should_log() -> bool:
    if not is_enabled():
        return False
    global _LOG_COUNT
    if _LOG_COUNT >= _max_logs():
        return False
    _LOG_COUNT += 1
    return True


def reset_log_budget() -> None:
    global _LOG_COUNT
    _LOG_COUNT = 0


def log(tag: str, **fields: Any) -> None:
    if not should_log():
        return
    parts = [f"[EagleDcpDebug][{tag}]"]
    for key, value in fields.items():
        parts.append(f"{key}={_format_value(value)}")
    logger.warning(" ".join(parts))


def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, torch.Tensor):
        return _format_tensor(value)
    if isinstance(value, (list, tuple)):
        return str(value[:16])
    return str(value)


def _format_tensor(tensor: torch.Tensor, limit: int = 8) -> str:
    if tensor.numel() == 0:
        return f"Tensor{list(tensor.shape)}[]"
    flat = tensor.detach().flatten()
    n = min(limit, flat.numel())
    sample = flat[:n].cpu().tolist()
    return f"Tensor{list(tensor.shape)}{sample}"


def log_cp_topology(pcp_size: int, dcp_size: int, pcp_rank: int, dcp_rank: int) -> None:
    log(
        "topology",
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        pcp_rank=pcp_rank,
        dcp_rank=dcp_rank,
        cp_size=pcp_size * dcp_size,
    )
