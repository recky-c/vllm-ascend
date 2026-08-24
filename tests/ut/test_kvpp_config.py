# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.kvpp_config import KVPPConfig


def _make_vllm_config(
    connector: str | None = None,
    role: str | None = None,
    additional_config: dict | None = None,
    tensor_parallel_size: int = 8,
) -> SimpleNamespace:
    kv_transfer_config = None
    if connector is not None:
        kv_transfer_config = SimpleNamespace(
            kv_connector=connector,
            kv_role=role,
        )
    return SimpleNamespace(
        use_v2_model_runner=True,
        additional_config=additional_config or {},
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            tensor_parallel_size=tensor_parallel_size,
        ),
        kv_transfer_config=kv_transfer_config,
        model_config=SimpleNamespace(use_mla=True, is_hybrid=False),
        speculative_config=None,
    )


def test_from_vllm_config_disables_kvpp_by_default() -> None:
    config = KVPPConfig.from_vllm_config(_make_vllm_config())

    assert config.size == 1


def test_from_vllm_config_uses_tp_size_when_enabled() -> None:
    config = KVPPConfig.from_vllm_config(
        _make_vllm_config(
            additional_config={"enable_kvpp": True},
            tensor_parallel_size=16,
        )
    )

    assert config.size == 16


@pytest.mark.parametrize("value", [0, 1, "true", [], {}])
def test_from_vllm_config_rejects_non_boolean_enable_kvpp(value: object) -> None:
    with pytest.raises(ValueError, match="enable_kvpp must be a boolean"):
        KVPPConfig.from_vllm_config(
            _make_vllm_config(additional_config={"enable_kvpp": value})
        )


def test_from_vllm_config_rejects_legacy_kvpp_size() -> None:
    with pytest.raises(ValueError, match="kvpp_size is no longer supported"):
        KVPPConfig.from_vllm_config(
            _make_vllm_config(additional_config={"kvpp_size": 8})
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
