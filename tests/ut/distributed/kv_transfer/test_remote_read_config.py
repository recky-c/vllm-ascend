from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import KVPPConfig


@pytest.mark.parametrize('backend,enabled,flag', [('ipc_broadcast', True, True),
                                                ('ipc_pull', False, True),
                                                ('ipc_pull', True, 'true')])
def test_remote_read_rejects_invalid_selection(backend, enabled, flag):
    config = SimpleNamespace(additional_config=dict(enable_kvpp=enabled, kvpp_transport=backend,
                            kvpp_remote_read=flag, kvpp_ipc_kernel_library='/test.so'),
                            parallel_config=SimpleNamespace(tensor_parallel_size=2),
                            kv_transfer_config=None, speculative_config=None)
    with pytest.raises(ValueError):
        KVPPConfig.from_vllm_config(config)
