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


def _offload_config(**extra):
    config = _make_vllm_config("AscendStoreConnector", "kv_producer")
    config.use_v2_model_runner = False
    config.model_config.enforce_eager = True
    config.parallel_config.pipeline_parallel_size = 1
    config.kv_transfer_config.kv_connector_extra_config = {
        "backend": "memcache",
        "use_layerwise": True,
        **extra,
    }
    return config


def test_kvpp_accepts_layerwise_memcache_producer():
    KVPPConfig(size=8).validate(_offload_config())


@pytest.mark.parametrize("field", ["layerwise_num_shared_buffers", "layerwise_prefetch_layers"])
@pytest.mark.parametrize("value", [1, 2, True, "3"])
def test_kvpp_offload_rejects_insufficient_or_invalid_pipeline_window(field, value):
    with pytest.raises(ValueError, match=field):
        KVPPConfig(size=8).validate(_offload_config(**{field: value}))


@pytest.mark.parametrize("change", ["v2", "graph", "pp", "mtp", "consumer", "mooncake", "non_layerwise"])
def test_kvpp_offload_rejects_unadapted_combinations(change):
    config = _offload_config()
    if change == "v2":
        config.use_v2_model_runner = True
    elif change == "graph":
        config.model_config.enforce_eager = False
    elif change == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    elif change == "mtp":
        config.speculative_config = SimpleNamespace(method="mtp")
    elif change == "consumer":
        config.kv_transfer_config.kv_role = "kv_consumer"
    elif change == "mooncake":
        config.kv_transfer_config.kv_connector_extra_config["backend"] = "mooncake"
    else:
        config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = False
    with pytest.raises(ValueError):
        KVPPConfig(size=8).validate(config)
