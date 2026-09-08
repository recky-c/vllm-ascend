# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import KVPPConfig


def config(**changes):
    values = dict(
        additional_config={"enable_kvpp": True},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        model_config=SimpleNamespace(enforce_eager=True, use_mla=True, is_hybrid=False),
        kv_transfer_config=None,
        speculative_config=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("granularity", ["tensor", "layer"])
def test_broadcast_granularity(granularity):
    result = KVPPConfig.from_vllm_config(
        config(additional_config={"enable_kvpp": True, "kvpp_broadcast_granularity": granularity})
    )
    assert result.size == 8
    assert result.broadcast_granularity == granularity


@pytest.mark.parametrize("tokens", [1, 3, 5])
def test_fixed_mtp_token_counts_are_supported(tokens):
    value = config(speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=tokens))
    KVPPConfig.from_vllm_config(value).validate(value)


def test_connector_combination_rejected():
    value = config(kv_transfer_config=object())
    with pytest.raises(ValueError, match="connectors"):
        KVPPConfig.from_vllm_config(value).validate(value)


def test_invalid_granularity_rejected_at_initialization():
    with pytest.raises(ValueError):
        KVPPConfig.from_vllm_config(config(additional_config={"kvpp_broadcast_granularity": "packed"}))
