# SPDX-License-Identifier: Apache-2.0
"""Device-independent regressions for KVPP read leases and resource retention."""
import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock


MODULE_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/kv_transfer/kv_pool/ipc_lifecycle.py"
SPEC = importlib.util.spec_from_file_location("tested_ipc_lifecycle", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestTransferTicket(unittest.TestCase):
    def setUp(self):
        self.generation = MODULE.TransferGeneration("engine", 5, "layer", ("main", "scale"), 8, ((7, 3),))
        self.resources = (object(), object())
        self.ticket = MODULE.TransferTicket(self.generation, ((91, 2),), 0, frozenset({1, 2}), self.resources)

    def test_owner_waits_for_every_reader(self):
        self.ticket.local_cache_ready = Mock()
        self.ticket.acknowledge_reader(1, self.generation)
        with self.assertRaisesRegex(RuntimeError, "still has readers"):
            self.ticket.wait_owner_source_reusable()
        self.ticket.acknowledge_reader(2, self.generation)
        self.ticket.wait_owner_source_reusable()

    def test_generation_components_are_checked(self):
        for field, value in (("engine_epoch", "other"), ("forward", 6), ("layer", "another"),
                             ("bundle", ("main",)), ("page_plan", 9), ("allocation_epochs", ((7, 4),))):
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "Stale"):
                self.ticket.acknowledge_reader(1, replace(self.generation, **{field: value}))
        self.assertFalse(self.ticket.owner_source_reusable)

    def test_duplicate_and_unknown_reader(self):
        self.ticket.acknowledge_reader(1, self.generation)
        for rank in (1, 0, 3):
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                self.ticket.acknowledge_reader(rank, self.generation)

    def test_local_readiness_is_independent_of_owner_lease(self):
        stream = Mock()
        with self.assertRaisesRegex(RuntimeError, "not been established"):
            self.ticket.wait_local_cache_ready(stream)
        ready = self.ticket.local_cache_ready = Mock()
        self.ticket.wait_local_cache_ready(stream)
        stream.wait_event.assert_called_once_with(ready)
        self.assertFalse(self.ticket.owner_source_reusable)

    def test_no_release_before_last_reader_or_ack(self):
        with self.assertRaisesRegex(RuntimeError, "before receive/source completion"):
            self.ticket.release_after_last_cache_use(Mock())
        with self.assertRaisesRegex(RuntimeError, "last cache reader"):
            self.ticket.drain()
        self.assertEqual(self.ticket.retained_resources, self.resources)

    def test_strong_references_survive_until_device_last_use(self):
        ready = self.ticket.local_cache_ready = Mock()
        for rank in (2, 1):
            self.ticket.acknowledge_reader(rank, self.generation)
        last_use = Mock()
        last_use.synchronize.side_effect = lambda: self.assertEqual(self.ticket.retained_resources, self.resources)
        self.ticket.release_after_last_cache_use(last_use)
        self.assertEqual(self.ticket.retained_resources, self.resources)
        self.ticket.drain()
        ready.synchronize.assert_called_once()
        last_use.synchronize.assert_called_once()
        self.assertEqual(self.ticket.retained_resources, ())

    def test_duplicate_release_rejected(self):
        self.ticket.local_cache_ready = Mock()
        for rank in (1, 2):
            self.ticket.acknowledge_reader(rank, self.generation)
        self.ticket.release_after_last_cache_use(Mock())
        with self.assertRaisesRegex(RuntimeError, "already recorded"):
            self.ticket.release_after_last_cache_use(Mock())


if __name__ == "__main__":
    unittest.main()
