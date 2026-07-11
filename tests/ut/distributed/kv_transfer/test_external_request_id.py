# SPDX-License-Identifier: Apache-2.0
"""Tests for get_external_request_id (vLLM EngineCore 9-char suffix stripping).

Regression coverage for the fix that guards short / malformed ids: the old
``request_id[:-9]`` returned empty / garbage for ids shorter than 9 chars,
which corrupted the external_req_id -> internal_req_id map that get_finished
relies on.
"""

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    get_external_request_id as mooncake_get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    get_external_request_id as pd_get_external_request_id,
)

IMPLS = [pd_get_external_request_id, mooncake_get_external_request_id]


@pytest.mark.parametrize("impl", IMPLS, ids=["pd_protocol", "mooncake"])
class TestGetExternalRequestId:
    def test_strips_nine_char_suffix(self, impl):
        # external "req-abc" + 9-char EngineCore suffix -> "req-abc"
        request_id = "req-abc" + "012345678"
        assert len(request_id) == 16
        assert impl(request_id) == "req-abc"

    def test_exactly_nine_chars_returned_unchanged(self, impl):
        # len == 9 does not satisfy len > 9, so nothing is stripped.
        request_id = "012345678"
        assert impl(request_id) == "012345678"

    def test_short_id_returned_unchanged(self, impl):
        # Short / malformed ids must round-trip to themselves, not garbage.
        assert impl("short") == "short"
        assert impl("") == ""

    def test_short_id_does_not_corrupt_map(self, impl):
        # The map built from a short id must stay stable on re-lookup.
        request_id = "abc"
        external = impl(request_id)
        assert external == "abc"
        lookup = {external: request_id}
        assert lookup[impl(request_id)] == request_id


def test_both_definitions_agree():
    # The two copies must not drift.
    samples = [
        "req-abc" + "012345678",
        "012345678",
        "x",
        "",
        "a" * 100,
    ]
    for s in samples:
        assert pd_get_external_request_id(s) == mooncake_get_external_request_id(s)
