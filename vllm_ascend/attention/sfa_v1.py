"""Compatibility shim for the SFA attention backend.

The implementation now lives in the ``vllm_ascend.attention.sfa`` package:

- ``sfa.backend``: ``AscendSFABackend``
- ``sfa.builder``: ``AscendSFAMetadataBuilder``
- ``sfa.impl``: ``AscendSFAImpl`` (mainline forward orchestration)
- ``sfa.metadata``: ``AscendSFAMetadata`` and per-feature context dataclasses
- ``sfa.kv_quant``: ``custom_kv_rmsnorm_rope`` / cache layout helpers
- ``sfa.constants``: shared named constants
- ``sfa.weight_prep``: MLAPO / prolog_v3 weight transforms and feature flags
- ``sfa.dsa_cp``: DSA-CP metadata build and KV gather helpers
- ``sfa.o_proj_tp``: DSA-CP mixed-mode o_proj TP gather

This module only re-exports the public names so existing import sites
(e.g. ``vllm_ascend.platform`` backend path strings) keep working.
"""

from vllm_ascend.attention.sfa import (
    BMM_TRANS_MAX_SUPPORTED_TOKENS,
    O_PROJ_ACLNN_INPUT_PARAMS,
    AscendSFABackend,
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    DCPContext,
    DCPQueryGatherContext,
    DSACPContext,
    custom_kv_rmsnorm_rope,
)
from vllm_ascend.attention.sfa.metadata import M

__all__ = [
    "AscendSFABackend",
    "AscendSFAImpl",
    "AscendSFAMetadata",
    "AscendSFAMetadataBuilder",
    "BMM_TRANS_MAX_SUPPORTED_TOKENS",
    "DCPContext",
    "DCPQueryGatherContext",
    "DSACPContext",
    "M",
    "O_PROJ_ACLNN_INPUT_PARAMS",
    "custom_kv_rmsnorm_rope",
]
