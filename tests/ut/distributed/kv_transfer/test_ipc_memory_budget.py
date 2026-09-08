# SPDX-License-Identifier: Apache-2.0
import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/kvpp_memory.py"
SPEC = importlib.util.spec_from_file_location("tested_ipc_memory", PATH)
MEMORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMORY)


class TestIpcMemoryBudget(unittest.TestCase):
    def test_generic_hbm_free_is_not_the_p2p_budget(self):
        gib = 1024**3
        budget = MEMORY.ipc_cache_payload_budget(40 * gib, 7 * gib, 16)
        self.assertEqual(budget, 7 * gib - 64 * 1024**2)

    def test_smaller_requested_budget_is_preserved(self):
        self.assertEqual(MEMORY.ipc_cache_payload_budget(123456, 1024**3, 16), 123456)

    def test_alignment_reserve_covers_all_raw_allocations(self):
        entries = 31
        total = MEMORY.DEFAULT_IPC_POOL_BUDGET_BYTES
        payload = MEMORY.ipc_cache_payload_budget(50 * 1024**3, total, entries)
        # Split unevenly across two raw allocations per entry, deliberately
        # maximizing unaligned tails instead of reproducing the budget formula.
        count = entries * 2
        lengths = [payload // count + (index < payload % count) for index in range(count)]
        alignment = MEMORY.IPC_ALLOCATION_ALIGNMENT
        actual = sum((length + alignment - 1) // alignment * alignment for length in lengths)
        self.assertLessEqual(actual, total)

    def test_invalid_or_too_small_physical_plan_rejected(self):
        for entries, budget in [(0, 1024**3), (-1, 1024**3), (16, 0), (16, 1024)]:
            with self.subTest(entries=entries, budget=budget), self.assertRaises(ValueError):
                MEMORY.ipc_cache_payload_budget(10 * 1024**3, budget, entries)


if __name__ == "__main__":
    unittest.main()
