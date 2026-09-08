# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ipc_allocation import IpcAllocationPool, P2P_ALIGNMENT
from vllm_ascend.distributed.kv_transfer.kv_pool.ipc_pull_transport import IpcPullKVPPTransport


def test_pool_rejects_over_budget_before_calling_driver():
    pool = IpcAllocationPool.__new__(IpcAllocationPool)
    pool.closed = False
    pool.budget_bytes = 2 * P2P_ALIGNMENT
    pool.allocations = [SimpleNamespace(length=P2P_ALIGNMENT)]
    pool.runtime = Mock()
    with pytest.raises(MemoryError, match="before aclrtMalloc"):
        pool.allocate(P2P_ALIGNMENT + 1)
    pool.runtime.call.assert_not_called()


def test_submission_failure_keeps_resources_and_refuses_reclamation():
    transport = IpcPullKVPPTransport.__new__(IpcPullKVPPTransport)
    transport._failed_generation = None
    resources = [object(), object()]
    transport._pending_resources = resources
    transport.engine_epoch = "engine"
    transport.forward = 9
    transport.closed = False
    transport._prefetch_impl = Mock(side_effect=RuntimeError("submission failure"))
    with pytest.raises(RuntimeError, match="submission failure"):
        transport.prefetch("layer", ("main",), None, None, None, None)
    assert transport._failed_generation == ("engine", 9, "layer")
    with pytest.raises(RuntimeError, match="Cannot reclaim"):
        transport.close()
    assert transport._pending_resources is resources
