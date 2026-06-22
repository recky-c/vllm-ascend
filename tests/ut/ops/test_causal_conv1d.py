from types import SimpleNamespace

import torch

from vllm_ascend.ops.triton.mamba import causal_conv1d


class _FakePCPGroup:
    world_size = 1
    rank_in_group = 0

    def all_gather(self, tensor, dim):
        return tensor


def test_causal_conv1d_fn_skips_zero_length_segments(monkeypatch):
    monkeypatch.setattr(
        causal_conv1d,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=SimpleNamespace(num_decodes=0),
        ),
    )
    monkeypatch.setattr(
        causal_conv1d,
        "get_pcp_group",
        lambda: _FakePCPGroup(),
    )

    dim = 4
    conv_width = 4
    x = torch.randn(dim, 3)
    weight = torch.randn(dim, conv_width)
    bias = torch.randn(dim)

    conv_states = torch.randn(2, dim, conv_width - 1)
    skipped_state = conv_states[0].clone()
    expected_conv_states = conv_states[1:].clone()

    output = causal_conv1d.causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        activation=None,
        conv_states=conv_states,
        cache_indices=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 0, 3], dtype=torch.int32),
    )
    expected_output = causal_conv1d.causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        activation=None,
        conv_states=expected_conv_states,
        cache_indices=torch.tensor([0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
    )

    assert torch.allclose(output, expected_output)
    assert torch.equal(conv_states[0], skipped_state)
