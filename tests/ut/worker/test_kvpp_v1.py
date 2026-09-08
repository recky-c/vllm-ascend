# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

from vllm_ascend.worker.kvpp_v1 import KVPPV1Runtime


def test_disabled_runtime_is_noop():
    runtime = KVPPV1Runtime()
    runtime.prepare_forward(True)
    runtime.complete_forward()
    runtime.close()


def test_adapter_forwards_history_and_lifecycle():
    shared = MagicMock()
    runtime = KVPPV1Runtime(shared)
    runtime.prepare_forward(True)
    shared.prepare_forward.assert_called_once_with(True)
    runtime.complete_forward()
    shared.complete_forward.assert_called_once_with()
    runtime.close()
    shared.close.assert_called_once_with()
