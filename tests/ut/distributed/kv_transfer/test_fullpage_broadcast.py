from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.distributed.kv_transfer.kv_pool.ipc_fullpage_broadcast_transport import (
    IpcFullPageBroadcastKVPPTransport,
)


@pytest.mark.parametrize('region,pages', [({'length': 16, 'stride': 32}, 4),
                                        ({'length': 16, 'stride': 8}, 4),
                                        ({'length': 0, 'stride': 0}, 4),
                                        ({'length': 16, 'stride': 16}, 0)])
def test_strided_or_empty_region_rejected(region, pages):
    with pytest.raises(ValueError, match='contiguous'):
        IpcFullPageBroadcastKVPPTransport.region_extent(region, pages)


def test_extent_excludes_allocation_padding():
    assert IpcFullPageBroadcastKVPPTransport.region_extent({'length': 129, 'stride': 129}, 17) == 2193


@pytest.mark.parametrize('backend,enabled,full', [('ipc_pull', True, True),
                                                ('ipc_broadcast', False, True),
                                                ('ipc_broadcast', True, 'true')])
def test_fullpage_config_rejects_wrong_backend_or_type(backend, enabled, full):
    config = SimpleNamespace(additional_config=dict(enable_kvpp=enabled, kvpp_transport=backend,
                            kvpp_broadcast_full_pages=full, kvpp_ipc_kernel_library='/test.so'),
                            parallel_config=SimpleNamespace(tensor_parallel_size=2),
                            kv_transfer_config=None, speculative_config=None)
    with pytest.raises(ValueError):
        KVPPConfig.from_vllm_config(config)
