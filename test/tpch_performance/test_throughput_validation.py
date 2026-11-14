# Copyright 2026, Sirius Contributors. Licensed under the Apache License, Version 2.0.
"""CPU-only coverage of throughput refresh-state validation."""

import unittest
from unittest.mock import patch

import duckdb
from queries import queries_for_scale_factor

import tpch_power_throughput as throughput


class ThroughputValidationTests(unittest.TestCase):
    @patch.object(throughput, "log")
    def test_q15_measured_cte_rows_are_compared_with_cpu_reference(self, _log):
        con = duckdb.connect()
        self.addCleanup(con.close)
        con.execute(
            "CREATE TABLE supplier AS SELECT 1 s_suppkey, 'name' s_name, 'address' s_address, 'phone' s_phone"
        )
        con.execute(
            "CREATE TABLE lineitem AS SELECT 1 l_suppkey, 100::DECIMAL(12,2) l_extendedprice, 0.1::DECIMAL(12,2) l_discount, DATE '1993-06-01' l_shipdate"
        )
        statements = [queries_for_scale_factor(1)["q15"]]
        _, measured = throughput.timed_query(con, statements, 0)
        reference = throughput.cpu_pass(con, [(15, statements)], 0)
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0][0], 1)
        self.assertEqual(measured[0][-1], 90)
        self.assertIsNone(throughput.compare_rows(reference[15], measured))
        self.assertEqual(
            throughput.validate_throughput(
                {1: {15: measured}}, {"baseline": reference}
            ),
            {},
        )
        changed = [(*measured[0][:-1], 999)]
        self.assertIn(
            "stream1_q15",
            throughput.validate_throughput({1: {15: changed}}, {"baseline": reference}),
        )

    def test_legacy_q15_view_sequence_retains_select_rows(self):
        con = duckdb.connect()
        self.addCleanup(con.close)
        _, rows = throughput.timed_query(
            con,
            [
                "CREATE VIEW revenue AS SELECT 90 AS value",
                "SELECT * FROM revenue",
                "DROP VIEW revenue",
            ],
            0,
        )
        self.assertEqual(rows, [(90,)])
        self.assertFalse(
            throughput._is_select(
                "WITH values_cte AS (SELECT 1) INSERT INTO sample SELECT * FROM values_cte"
            )
        )

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
