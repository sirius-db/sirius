# Copyright 2026, Sirius Contributors. Licensed under the Apache License, Version 2.0.
"""CPU-only coverage of throughput refresh-state validation."""

import unittest
from unittest.mock import patch

import duckdb

import tpch_power_throughput as throughput


class ThroughputValidationTests(unittest.TestCase):
    def setUp(self):
        self.snapshots = {
            "baseline": {1: [(10,)], 6: [(20,)]},
            "after_rf1_set2": {1: [(11,)], 6: [(21,)]},
            "after_rf2_set2": {1: [(12,)], 6: [(22,)]},
        }

    @patch.object(throughput, "log")
    def test_queries_can_observe_different_committed_states(self, _log):
        rows = {
            1: {1: [(10,)], 6: [(21,)]},
            2: {1: [(12,)], 6: [(20,)]},
        }
        self.assertEqual(throughput.validate_throughput(rows, self.snapshots), {})

    @patch.object(throughput, "log")
    def test_mismatch_reports_every_candidate_snapshot(self, _log):
        failures = throughput.validate_throughput(
            {1: {1: [(999,)], 6: [(22,)]}}, self.snapshots
        )
        self.assertEqual(set(failures), {"stream1_q1"})
        self.assertEqual(set(failures["stream1_q1"]), set(self.snapshots))
        self.assertTrue(all(failures["stream1_q1"].values()))

    def test_cpu_snapshots_follow_commits_and_skip_invariant_queries(self):
        con = duckdb.connect()
        try:
            con.execute("CREATE TABLE sample AS SELECT 10 AS value")
            plan = [(1, ["SELECT value FROM sample"])] + [
                (q, ["SELECT * FROM must_not_run"])
                for q in throughput.REFRESH_INVARIANT_QUERIES
            ]
            before = throughput.cpu_pass(con, plan, 0)
            con.execute("UPDATE sample SET value = 11")
            after = throughput.cpu_pass(con, plan, 0)
            self.assertEqual(before, {1: [(10,)]})
            self.assertEqual(after, {1: [(11,)]})
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
