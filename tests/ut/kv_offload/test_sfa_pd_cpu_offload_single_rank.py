"""Regression tests for the TP-shared SFA PD CPU pool."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

import torch.utils.cpp_extension as _cpp_extension  # noqa: E402

_cpp_extension.load = MagicMock(return_value=MagicMock())

memfabric_hybrid = pytest.importorskip("memfabric_hybrid")
if not hasattr(memfabric_hybrid, "offload"):
    memfabric_hybrid.offload = MagicMock()

from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload import worker as worker_module  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.worker import (  # noqa: E402
    MembPullReadThread,
    SFAPDCpuOffloadConsumerWorker,
)


def _make_read_thread(partial_hbm_bid: int | None = 9) -> MembPullReadThread:
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._worker = SimpleNamespace(
        _dest_blocks_by_req={"req-0": ([7], [3, 4], 2, partial_hbm_bid)},
    )
    return thread


def test_non_owner_still_registers_memfabric_pull():
    consumer = SFAPDCpuOffloadConsumerWorker.__new__(SFAPDCpuOffloadConsumerWorker)
    consumer.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"transfer_backend": "memfabric"},
        ),
    )
    consumer.use_layerwise = True
    consumer.kv_cache_config = MagicMock()
    consumer._register_memfabric_pull = MagicMock()
    sfa_worker = SimpleNamespace(register_kv_caches=MagicMock())
    kv_caches = {"model.layers.0.self_attn.attn": tuple(MagicMock() for _ in range(5))}

    with patch.object(worker_module, "SFAKVOffloadWorker", return_value=sfa_worker):
        consumer.register_kv_caches(kv_caches)

    consumer._register_memfabric_pull.assert_called_once_with(kv_caches, None, None)


def _make_layer(
    k_cpu_ptr: int | None,
    v_cpu_ptr: int | None,
    with_scale: bool = False,
) -> dict:
    return {
        "layer_name": "model.layers.0.self_attn.attn",
        "pool_idx": 0,
        "offload_id": 0,
        "p_k_base": 1000,
        "p_v_base": 2000,
        "p_k_len": 10,
        "p_v_len": 20,
        "k_cpu_ptr": k_cpu_ptr,
        "v_cpu_ptr": v_cpu_ptr,
        "k_hbm_ptr": 5000,
        "v_hbm_ptr": 6000,
        "indexer": {
            "p_dsa_base": 7000,
            "p_dsa_len": 5,
            "d_base": 8000,
            "d_dsa_len": 10,
            "scale": 2,
            "shape": (16, 2, 1, 128),
        },
        "scale": (
            {
                "p_scale_base": 9000,
                "p_scale_len": 4,
                "d_scale_base": 10000,
                "d_scale_len": 8,
                "scale_factor": 2,
            }
            if with_scale
            else None
        ),
    }


def test_cpu_pool_owner_reads_full_main_partial_and_indexer():
    thread = _make_read_thread()

    local, _, _, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000),
        "req-0",
        [1, 2, 3],
        want_info=True,
    )

    assert info is not None
    assert info["n_main"] == 3
    assert info["n_indexer"] == 2
    assert 3030 in local
    assert 4060 in local
    assert 5090 in local
    assert 6180 in local
    assert 8070 in local


def test_non_owner_reads_only_partial_and_indexer_hbm():
    thread = _make_read_thread()

    local, _, _, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=None, v_cpu_ptr=None),
        "req-0",
        [1, 2, 3],
        want_info=True,
    )

    assert info is not None
    assert info["n_main"] == 1
    assert info["n_indexer"] == 2
    assert local == [5090, 6180, 8070]


def test_non_owner_reads_indexer_scale_without_partial_or_cpu_pool():
    thread = _make_read_thread(partial_hbm_bid=None)

    local, _, _, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=None, v_cpu_ptr=None, with_scale=True),
        "req-0",
        [1, 2, 3],
        want_info=True,
    )

    assert info is not None
    assert info["n_main"] == 0
    assert info["n_indexer"] == 2
    assert local == [8070, 10056]
