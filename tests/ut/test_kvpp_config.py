# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.kvpp_config import KVPPConfig


def _make_vllm_config(
    connector: str | None = None,
    role: str | None = None,
) -> SimpleNamespace:
    kv_transfer_config = None
    if connector is not None:
        kv_transfer_config = SimpleNamespace(
            kv_connector=connector,
            kv_role=role,
        )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            tensor_parallel_size=8,
        ),
        kv_transfer_config=kv_transfer_config,
        model_config=SimpleNamespace(use_mla=True, is_hybrid=False),
        speculative_config=None,
    )


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
def test_validate_allows_mooncake_v2_on_kv_producer(connector: str) -> None:
    KVPPConfig(size=8).validate(_make_vllm_config(connector, "kv_producer"))


@pytest.mark.parametrize(
    ("connector", "role"),
    [
        ("MooncakeConnectorV2", "kv_consumer"),
        ("MooncakeConnectorV1", "kv_producer"),
        ("OffloadingConnector", "kv_producer"),
    ],
)
def test_validate_rejects_unadapted_kv_transfer(connector: str, role: str) -> None:
    with pytest.raises(ValueError, match="KVPP supports KV transfer only"):
        KVPPConfig(size=8).validate(_make_vllm_config(connector, role))
