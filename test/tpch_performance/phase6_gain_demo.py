# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

"""Gain-demonstration cell for the Top-N dynamic filters: a top-N whose ORDER BY key is a
grouping key that reaches a parquet scan with no joins in between -- the shape TPC-H's own
LIMIT queries structurally lack (their keys always sit below joins, over the admission cap,
or on aggregate outputs).

    SELECT l_orderkey, sum(l_quantity) AS total_qty
    FROM lineitem GROUP BY l_orderkey ORDER BY l_orderkey LIMIT 100

The group-key producer arms, its trace reaches the lineitem scan (SCAN_BIND), and lineitem
parquet is naturally ordered by l_orderkey, so the published inclusive boundary lets the
reader's row-group statistics prune nearly everything after the first revision. Interleaved
flag-off/flag-on pairs, alternating lead, byte-identity per pair, counters per execution.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from performance_test import (  # noqa: E402
    is_resolved,
    median_of,
    open_connection,
    paired_ratio_interval,
)

QUERY = (
    "SELECT l_orderkey, sum(l_quantity) AS total_qty FROM lineitem "
    "GROUP BY l_orderkey ORDER BY l_orderkey LIMIT 100"
)

COUNTERS_OF_INTEREST = [
    "top_n_group_producers_eligible",
    "top_n_group_offers",
    "top_n_group_witness_set_full",
    "top_n_group_prefilter_rows_in",
    "top_n_group_prefilter_rows_out",
    "top_n_revisions_published",
    "top_n_first_key_scan_targets",
    "reader_gate_row_groups_considered",
    "reader_gate_row_groups_pruned",
    "reader_gate_merges_skipped",
]


def read_counters(con) -> dict:
    return dict(
        con.execute("SELECT name, value FROM sirius_dynamic_filter_stats()").fetchall()
    )


def run_arm(con, arm: str) -> tuple:
    con.execute(
        f"SET enable_top_n_dynamic_filter = {'true' if arm == 'on' else 'false'}"
    )
    before = read_counters(con)
    start = time.perf_counter()
    rows = con.execute(QUERY).fetchall()
    elapsed = time.perf_counter() - start
    after = read_counters(con)
    deltas = {k: after[k] - before.get(k, 0) for k in after}
    return elapsed, repr(rows), deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="TPC-H parquet directory (lineitem/)"
    )
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()

    con = open_connection(args.input, gpu_execution=True, data_source="parquet")

    ratios, off_times, on_times = [], [], []
    on_deltas_sum: dict = {}
    for pair in range(args.warmups + args.pairs):
        lead = "off" if pair % 2 == 0 else "on"
        order = ("off", "on") if lead == "off" else ("on", "off")
        results = {}
        for arm in order:
            results[arm] = run_arm(con, arm)
        off_t, off_rows, _ = results["off"]
        on_t, on_rows, on_d = results["on"]
        if off_rows != on_rows:
            raise SystemExit(
                f"pair {pair}: RESULT MISMATCH between arms -- correctness bug"
            )
        phase = "warmup" if pair < args.warmups else "sample"
        print(
            f"pair {pair:2d} [{phase}] lead={lead} off={off_t:.3f}s on={on_t:.3f}s "
            f"ratio={on_t / off_t:.4f}"
        )
        if phase == "sample":
            # A demo over an unarmed producer would show a flat ratio and prove nothing --
            # fail loudly instead of demonstrating silence.
            if on_d.get("top_n_group_producers_eligible", 0) == 0:
                raise SystemExit(
                    f"pair {pair}: flag-on arm did not arm the group producer -- "
                    "the demo query no longer exercises the feature"
                )
            ratios.append(on_t / off_t)
            off_times.append(off_t)
            on_times.append(on_t)
            for k, v in on_d.items():
                on_deltas_sum[k] = on_deltas_sum.get(k, 0) + v

    low, point, high = paired_ratio_interval(ratios)
    print(
        f"\noff median {median_of(off_times):.3f}s  on median {median_of(on_times):.3f}s"
    )
    print(f"geomean ratio {point:.4f}  CI [{low:.4f}, {high:.4f}]  pairs {len(ratios)}")
    print(f"resolved at +/-2%: {is_resolved(ratios, 0.02)}")
    print("\nflag-on counter deltas (summed over sample executions):")
    for k in COUNTERS_OF_INTEREST:
        if k in on_deltas_sum:
            print(f"  {k}: {on_deltas_sum[k]:,}")


if __name__ == "__main__":
    main()
