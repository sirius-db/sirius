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

"""Calibrate the frozen arming-expectations JSON consumed by performance_test.py --mode ab.

Runs every requested TPC-H query a few times per arm (flag off / flag on, toggled via
`SET enable_top_n_dynamic_filter`) over one Sirius session, capturing per-execution
dynamic-filter counter deltas through sirius_dynamic_filter_stats(). SF1 suffices: the counters
being frozen are plan-time facts (producer eligibility, scan-target and endpoint-site placement),
which depend on plan shape, not data volume.

Freezing rules, from the dynamic_filter_stats header contract:
  - Plan-time counters are exact per execution; they are frozen as `exact` deltas when stable
    across repetitions and degraded to `nonzero` (with a warning) when not.
  - Delivery-time counters (offers, prefilter rows, revisions, pushes, post-decode applies,
    reader-gate movement) race batch arrival by design; ones that moved are listed as
    `direction_only` (documentation, not per-execution assertions), and a small allowlist of
    must-eventually-move fields that moved on every repetition is frozen as `cell_nonzero`.
  - The flag-off arm must never move a top_n_* counter, frozen as `zero_prefix: ["top_n_"]`.

The script also checks the observed arming against the recorded forecast (Q18 group producer;
Q2 row producer; Q3/Q10/Q21 self-consumption only; non-LIMIT queries untouched) and reports any
mismatch loudly -- an off-forecast arming profile changes the SF1000 expectations and must be
explained, not papered over.

Usage:
  pixi run python test/tpch_performance/phase5_calibrate_arming.py \\
      --input test_datasets/tpch_parquet_sf1 \\
      --output test/tpch_performance/phase5_arming_expectations.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from performance_test import (  # noqa: E402
    LIMIT_QUERIES,
    get_git_info,
    log,
    open_connection,
    parse_query_spec,
    read_dynamic_filter_counters,
)
from queries import QUERIES  # noqa: E402

AB_OPTION = "enable_top_n_dynamic_filter"
TOP_N_PREFIX = "top_n_"

# Plan-time facts per the dynamic_filter_stats header: constructed with the physical plan (twice
# per query on the transparent path), deterministic per execution, hence exact-assertable.
PLAN_TIME_FIELDS = (
    "top_n_producers_eligible",
    "top_n_producers_rejected",
    "top_n_producers_first_key_only",
    "top_n_first_key_scan_targets",
    "top_n_lex_scan_targets",
    "top_n_first_key_endpoint_sites_placed",
    "top_n_lex_endpoint_sites_placed",
    "top_n_sites_skipped_no_work_saved",
    "top_n_first_key_subsumed_by_lex",
    "top_n_group_producers_eligible",
    "top_n_group_producers_rejected",
)

# Delivery-time fields that, when they moved on every calibration repetition, are frozen as
# cell-level direction assertions (must move at least once across a query's whole AB cell).
CELL_NONZERO_CANDIDATES = (
    "top_n_offers",
    "top_n_group_offers",
    "top_n_group_witness_set_full",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate phase5 arming expectations from per-execution counter deltas"
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="TPC-H parquet directory (SF1 recommended; plan-time counters are size-independent)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "phase5_arming_expectations.json",
        ),
        help="Where to write the frozen expectations JSON",
    )
    p.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Comma-separated query list, e.g. '1,3,6-10' (default: all 22)",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Executions per (query, arm); exactness requires all reps to agree (default: 3)",
    )
    p.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Recorded in the JSON metadata (default: 1)",
    )
    return p.parse_args()


def run_arm(con, qnum: int, arm: str, reps: int) -> List[Dict[str, int]]:
    """Execute one query `reps` times under one arm, returning per-execution counter deltas."""
    con.execute(f"SET {AB_OPTION} = {'true' if arm == 'on' else 'false'};")
    delta_runs = []
    for rep in range(reps):
        before = read_dynamic_filter_counters(con)
        rows = con.execute(QUERIES[f"q{qnum}"]).fetchall()
        after = read_dynamic_filter_counters(con)
        deltas = {name: after[name] - before[name] for name in after}
        delta_runs.append(deltas)
        log(f"  q{qnum} {arm} rep{rep}: {len(rows)} rows")
    return delta_runs


def freeze_on_arm(
    qnum: int, delta_runs: List[Dict[str, int]], warnings: List[str]
) -> dict:
    """Freeze the flag-on arm's predicates from observed per-execution deltas."""
    spec: dict = {}
    exact: Dict[str, int] = {}
    nonzero: List[str] = []
    for name in PLAN_TIME_FIELDS:
        values = {run[name] for run in delta_runs}
        if len(values) == 1:
            exact[name] = values.pop()
        elif all(run[name] > 0 for run in delta_runs):
            nonzero.append(name)
            warnings.append(
                f"q{qnum}: plan-time counter {name} unstable across reps "
                f"({sorted(run[name] for run in delta_runs)}); degraded to nonzero"
            )
        else:
            warnings.append(
                f"q{qnum}: plan-time counter {name} unstable across reps "
                f"({sorted(run[name] for run in delta_runs)}); left unasserted"
            )
    spec["exact"] = exact
    if nonzero:
        spec["nonzero"] = nonzero

    moved_somewhere = sorted(
        name
        for name in delta_runs[0]
        if name.startswith(TOP_N_PREFIX)
        and name not in PLAN_TIME_FIELDS
        and any(run[name] > 0 for run in delta_runs)
    )
    if moved_somewhere:
        spec["direction_only"] = moved_somewhere
    cell_nonzero = [
        name
        for name in CELL_NONZERO_CANDIDATES
        if all(run[name] > 0 for run in delta_runs)
    ]
    if cell_nonzero:
        spec["cell_nonzero"] = cell_nonzero
    # When no top_n counter moved at all, pin the whole prefix at zero so a later mis-wiring
    # cannot arm an untouched query unnoticed.
    if (
        not moved_somewhere
        and all(value == 0 for value in exact.values())
        and not nonzero
    ):
        spec["zero_prefix"] = [TOP_N_PREFIX]
    return spec


def check_off_arm(
    qnum: int, delta_runs: List[Dict[str, int]], errors: List[str]
) -> None:
    for rep, deltas in enumerate(delta_runs):
        moved = {
            name: delta
            for name, delta in deltas.items()
            if name.startswith(TOP_N_PREFIX) and delta != 0
        }
        if moved:
            errors.append(f"q{qnum} flag-off rep{rep} moved top_n counters: {moved}")


def forecast_verdict(qnum: int, on_spec: dict) -> str:
    """Compare observed flag-on arming against the recorded per-query forecast."""
    exact = on_spec.get("exact", {})
    nonzero = set(on_spec.get("nonzero", []))

    def positive(name: str) -> bool:
        return exact.get(name, 0) > 0 or name in nonzero

    row_armed = positive("top_n_producers_eligible")
    group_armed = positive("top_n_group_producers_eligible")
    external = sum(
        exact.get(name, 0)
        for name in (
            "top_n_first_key_scan_targets",
            "top_n_lex_scan_targets",
            "top_n_first_key_endpoint_sites_placed",
            "top_n_lex_endpoint_sites_placed",
        )
    )
    if qnum == 18:
        expected = "group producer armed"
        ok = group_armed
    elif qnum == 2:
        expected = "row producer eligible"
        ok = row_armed
    elif qnum in (3, 10, 21):
        expected = (
            "row producer eligible, self-consumption only (no external targets/sites)"
        )
        ok = row_armed and external == 0
    elif qnum in LIMIT_QUERIES:
        expected = "some producer eligible"
        ok = row_armed or group_armed
    else:
        expected = "untouched (no producers eligible)"
        ok = not row_armed and not group_armed
    observed = (
        f"row_eligible={exact.get('top_n_producers_eligible', 'nonzero' if 'top_n_producers_eligible' in nonzero else 0)}, "
        f"group_eligible={exact.get('top_n_group_producers_eligible', 'nonzero' if 'top_n_group_producers_eligible' in nonzero else 0)}, "
        f"external_targets_sites={external}"
    )
    return f"{'MATCH' if ok else 'MISMATCH'}: expected {expected}; observed {observed}"


def main() -> None:
    args = parse_args()
    queries = parse_query_spec(args.queries)
    if not os.path.isdir(args.input):
        raise SystemExit(
            f"--input must be a TPC-H parquet directory; got {args.input!r}"
        )

    con = open_connection(args.input, gpu_execution=True, data_source="parquet")
    con.execute("SET gpu_execution = true;")

    expectations: dict = {}
    warnings: List[str] = []
    errors: List[str] = []
    verdicts: Dict[str, str] = {}
    try:
        for qnum in queries:
            log(f"--- calibrating q{qnum} ---")
            off_runs = run_arm(con, qnum, "off", args.reps)
            on_runs = run_arm(con, qnum, "on", args.reps)
            check_off_arm(qnum, off_runs, errors)
            on_spec = freeze_on_arm(qnum, on_runs, warnings)
            verdicts[f"q{qnum}"] = forecast_verdict(qnum, on_spec)
            log(f"  q{qnum} forecast check -- {verdicts[f'q{qnum}']}")
            expectations[f"q{qnum}"] = {
                "off": {"zero_prefix": [TOP_N_PREFIX]},
                "on": on_spec,
            }
    finally:
        con.close()

    if errors:
        for message in errors:
            log(f"ERROR: {message}")
        raise SystemExit(
            "flag-off arm moved top_n counters; the flag is not gating -- fix before freezing"
        )

    commit, _ = get_git_info()
    document = {
        "schema_version": 1,
        "option": AB_OPTION,
        "scale_factor": args.scale_factor,
        "calibrated": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "reps": args.reps,
        "input": os.path.realpath(args.input),
        "forecast_check": verdicts,
        "warnings": warnings,
        "queries": expectations,
    }
    with open(args.output, "w") as f:
        json.dump(document, f, indent=2)
        f.write("\n")
    log(f"Frozen arming expectations written to {args.output}")
    for message in warnings:
        log(f"WARNING: {message}")
    mismatches = [
        q for q, verdict in verdicts.items() if verdict.startswith("MISMATCH")
    ]
    if mismatches:
        log(
            f"FORECAST MISMATCH on {', '.join(mismatches)} -- report before running the cells"
        )
    else:
        log("Observed arming matches the forecast on every calibrated query")


if __name__ == "__main__":
    main()
