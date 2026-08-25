from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import (
    MooncakeBaseConnectorWorker,
)


def test_mc_01_publishes_local_persistent_and_replicated_mtp_only():
    local_target = "model.layers.2.self_attn.attn"
    foreign_scratch_aliases = {
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    }
    mtp = "model.layers.4.mtp_block.self_attn.attn"
    all_layers = {local_target, *foreign_scratch_aliases, mtp}
    owners = {
        "model.layers.0.self_attn.attn": 0,
        "model.layers.1.self_attn.attn": 0,
        local_target: 1,
    }
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.ascend_config = SimpleNamespace(
        kvpp_config=SimpleNamespace(size=2)
    )
    worker.layer_name_to_group_index = dict.fromkeys(all_layers, 0)
    worker.vllm_config = SimpleNamespace()
    worker.tp_rank = 1

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker.get_kvpp_layer_owners",
        return_value=owners,
    ) as get_owners:
        published = worker._get_published_layer_names()

    get_owners.assert_called_once_with(worker.vllm_config, all_layers)
    assert published == {local_target, mtp}
    assert published.isdisjoint(foreign_scratch_aliases)
