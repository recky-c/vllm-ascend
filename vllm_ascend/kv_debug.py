import os
from collections.abc import Iterable
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def kv_debug_enabled() -> bool:
    return os.environ.get("VLLM_KV_DEBUG", "").lower() in _TRUE_VALUES


def kv_debug_log(logger: Any, msg: str, *args: Any) -> None:
    if kv_debug_enabled():
        logger.info("[KV_DEBUG] " + msg, *args)


def kv_ids_summary(ids: Any, max_items: int = 32) -> Any:
    if ids is None:
        return None
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, Iterable) and not isinstance(ids, (str, bytes, dict)):
        out = []
        for idx, value in enumerate(ids):
            if idx >= max_items:
                out.append("...")
                break
            out.append(getattr(value, "block_id", value))
        return out
    return ids


def kv_tensor_summary(tensor: Any) -> str:
    if tensor is None:
        return "None"
    shape = tuple(getattr(tensor, "shape", ()))
    dtype = getattr(tensor, "dtype", None)
    device = getattr(tensor, "device", None)
    stride = None
    try:
        stride = tuple(tensor.stride())
    except Exception:
        pass
    return f"shape={shape}, dtype={dtype}, device={device}, stride={stride}"
