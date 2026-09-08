# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import KVPPConfig


def config(**additional):
    return SimpleNamespace(additional_config=additional,
                           parallel_config=SimpleNamespace(tensor_parallel_size=2),
                           kv_transfer_config=None, speculative_config=None)


def test_default_transport_is_preserved():
    value = KVPPConfig.from_vllm_config(config(enable_kvpp=True))
    assert value.transport == "memfabric" and value.size == 2


@pytest.mark.parametrize("backend", ["ipc_pull", "ipc_broadcast"])
@pytest.mark.parametrize("extra", [
    {"enable_kvpp": False},
    {"kvpp_ipc_kernel_library": None},
    {"kvpp_ipc_cores": 0},
    {"kvpp_ipc_cores": 41},
    {"kvpp_transport": "typo"},
    {"kvpp_ipc_pool_budget_bytes": 0},
    {"kvpp_ipc_pool_budget_bytes": 9 * 1024**3},
])
def test_invalid_ipc_config_rejected(extra, backend):
    values = dict(enable_kvpp=True, kvpp_transport=backend, kvpp_ipc_kernel_library="/test/copy.so")
    values.update(extra)
    with pytest.raises(ValueError):
        KVPPConfig.from_vllm_config(config(**values))


@pytest.mark.parametrize("backend", ["ipc_pull", "ipc_broadcast"])
@pytest.mark.parametrize("field", ["kv_transfer_config", "speculative_config"])
def test_unvalidated_lifecycles_rejected(field, backend):
    value = config(enable_kvpp=True, kvpp_transport=backend, kvpp_ipc_kernel_library="/test/copy.so")
    setattr(value, field, SimpleNamespace())
    with pytest.raises(ValueError, match="not yet validated"):
        KVPPConfig.from_vllm_config(value)


@pytest.mark.parametrize("backend", ["ipc_pull", "ipc_broadcast"])
def test_explicit_backends_are_supported(backend):
    value = config(enable_kvpp=True, kvpp_transport=backend, kvpp_ipc_kernel_library="/test/copy.so")
    assert KVPPConfig.from_vllm_config(value).transport == backend
