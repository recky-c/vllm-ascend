from types import SimpleNamespace

from vllm_ascend.worker.kvpp_v1 import KVPPV1Runtime, validate_v1_mtp_layers


def _layer(index: int) -> str:
    return f"model.layers.{index}.self_attn.attn"


def _config(*, layers: int = 61):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=layers,
                num_nextn_predict_layers=1,
            )
        ),
    )


def test_validate_v1_mtp_layers_rejects_owned_draft_cache():
    layers = (_layer(0), _layer(1), _layer(61))
    owners = {_layer(0): 0, _layer(61): 0}

    validate_v1_mtp_layers(_config(), layers, {_layer(0): 0})
    try:
        validate_v1_mtp_layers(_config(), layers, owners)
        raise AssertionError("expected owned MTP cache to fail")
    except RuntimeError as exc:
        assert "replicated outside KVPP" in str(exc)


def test_kvpp_runtime_v1_disabled_begin_is_noop():
    KVPPV1Runtime().begin_forward(SimpleNamespace(), 1, [1])
