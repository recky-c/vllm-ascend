"""Unit tests for SFAKVOffloadWorker layer registration.

Covers:
- offload layer selection by tuple length (five/six-tuple in, others out)
- the mixed LIC8 / non-LIC8 guard: under offload, C8 must be uniform across
  sparse layers (the attention path gates the quant indexer on a GLOBAL flag),
  so coexisting five- and six-tuple layers must raise.

The worker module JIT-builds a C++ extension and imports memfabric_hybrid at
module load time, neither of which is available in the UT sandbox; both are
stubbed before the import below.
"""

from unittest.mock import MagicMock

# Stub heavy module-level dependencies BEFORE importing the worker.
# 1. cpu_sparse_attn cpp extension JIT build (torch.utils.cpp_extension.load).
import torch.utils.cpp_extension as _cpp_extension  # noqa: E402

_cpp_extension.load = MagicMock(return_value=MagicMock())  # noqa: E402

# 2. memfabric_hybrid.offload is not exported in the sandbox install.
import memfabric_hybrid  # noqa: E402

if not hasattr(memfabric_hybrid, "offload"):  # noqa: E402
    memfabric_hybrid.offload = MagicMock()  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_worker import (  # noqa: E402
    SFAKVOffloadWorker,
)


def _make_worker_without_init() -> SFAKVOffloadWorker:
    """Bypass __init__ (heavy); set only the attrs _register_offload_layers reads."""
    w = SFAKVOffloadWorker.__new__(SFAKVOffloadWorker)
    w.num_target_layers = 0
    w.tp_rank = 0
    w.pending_save_layer_ids = set()
    w.submitted_save_layer_ids = set()
    return w


def _tuple(n: int) -> tuple:
    return tuple(torch.zeros(1) for _ in range(n))


def test_register_selects_offload_tuples_and_skips_others():
    # Five- and six-tuple layers are offload candidates; single tensors and
    # other lengths are skipped. (5- and 6-tuple cannot coexist — see the
    # mixed guard test below — so exercise them in separate dicts.)
    for offload_len in (5, 6):
        w = _make_worker_without_init()
        kv_caches = {
            "layer.0": _tuple(offload_len),
            "layer.1": _tuple(offload_len),
            "indexer.layer.0": torch.zeros(1),  # single tensor, not an offload tuple
            "layer.2": _tuple(3),  # neither five- nor six-tuple
        }
        w._register_offload_layers(kv_caches)
        assert w.offload_layer_names == ["layer.0", "layer.1"]
        assert w.num_offload_layers == 2


def test_register_raises_when_no_offload_layers():
    w = _make_worker_without_init()
    with pytest.raises(ValueError, match="did not find SFA KV cache layers"):
        w._register_offload_layers({"layer.0": _tuple(3)})


def test_register_all_five_tuple_passes():
    w = _make_worker_without_init()
    w._register_offload_layers({"layer.0": _tuple(5), "layer.1": _tuple(5)})
    assert w.num_offload_layers == 2


def test_register_all_six_tuple_passes():
    w = _make_worker_without_init()
    w._register_offload_layers({"layer.0": _tuple(6), "layer.1": _tuple(6)})
    assert w.num_offload_layers == 2


def test_register_rejects_mixed_five_and_six_tuple():
    """Mixed LIC8 / non-LIC8 layers under offload would route a non-C8 layer
    through the quant indexer (global flag) — must raise at registration."""
    w = _make_worker_without_init()
    kv_caches = {"layer.0": _tuple(5), "layer.1": _tuple(6)}
    with pytest.raises(ValueError, match="mixed LIC8 / non-LIC8"):
        w._register_offload_layers(kv_caches)
