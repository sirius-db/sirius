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

"""Tests for the Phase-5 AB mode in performance_test.py: the statistics ported from
test/cpp/utils/measurement_harness.hpp (checked against independently recomputed constants) and a
dry run of the pair schedule over a stub executor (lead alternation, warmup handling, the
resolution-driven stop, foreign-occupancy discard-and-retry, the max-pairs cap, the pilot stop,
and the arming / result-identity aborts).

pytest-compatible; also runnable directly (`pixi run python test/tpch_performance/
test_ab_stats.py`) because pytest is not part of the pixi environments today.
"""

import json
import math
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from performance_test import (  # noqa: E402
    AbCellAbort,
    AbQueryPlan,
    AbRunAbort,
    ExecutionOutcome,
    GpuOccupancy,
    PairRecord,
    achieved_resolution,
    evaluate_arming,
    evaluate_cell_arming,
    is_resolved,
    log_ratio_stddev,
    median_of,
    paired_ratio_interval,
    pairs_needed_for_target,
    run_ab_query_cell,
    suite_geomean_interval,
)

# Fixed sample vector; the expected values below were recomputed independently of the module under
# test (mean/stddev of logs, exp back, 1.96/sqrt(n) half-width) and are asserted as literals.
RATIOS = [1.02, 0.98, 1.05, 1.01, 0.99, 1.03]


def assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    assert math.isclose(
        actual, expected, rel_tol=0.0, abs_tol=tol
    ), f"{actual} != {expected}"


# --- statistics -----------------------------------------------------------------------------------


def test_median_of_matches_harness_semantics() -> None:
    assert median_of([]) == 0.0
    assert median_of([3.0]) == 3.0
    assert median_of([5.0, 1.0, 3.0]) == 3.0
    # Even count: mean of the two middle elements (sample_series::median_of).
    assert median_of([4.0, 1.0, 3.0, 2.0]) == 2.5


def test_paired_ratio_interval_against_independent_constants() -> None:
    low, point, high = paired_ratio_interval(RATIOS)
    assert_close(point, 1.013059351684856)
    assert_close(low, 0.9926193165232784)
    assert_close(high, 1.033920288425168)
    assert_close(log_ratio_stddev(RATIOS), 0.02547327993415192)
    assert_close(achieved_resolution(RATIOS), 0.020650485950944764)
    assert pairs_needed_for_target(RATIOS, 0.02) == 7


def test_interval_degenerates_to_unity_below_two_pairs() -> None:
    assert paired_ratio_interval([]) == (1.0, 1.0, 1.0)
    assert paired_ratio_interval([1.07]) == (1.0, 1.0, 1.0)
    assert log_ratio_stddev([1.07]) == 0.0
    assert pairs_needed_for_target([1.07], 0.02) == 0
    assert not is_resolved([1.07], 0.02)


def test_resolved_thresholds() -> None:
    # achieved_resolution(RATIOS) is ~0.02065: unresolved at 0.02, resolved at 0.021.
    assert not is_resolved(RATIOS, 0.02)
    assert is_resolved(RATIOS, 0.021)
    assert not is_resolved(RATIOS, 0.0)  # a zero target can never be met


def test_zero_spread_needs_only_current_pairs() -> None:
    flat = [1.0] * 5
    assert log_ratio_stddev(flat) == 0.0
    assert pairs_needed_for_target(flat, 0.02) == len(flat)
    assert is_resolved(flat, 0.001)


def test_suite_geomean_interval_against_independent_constants() -> None:
    per_query = {
        "qa": [1.01, 1.02, 1.00, 1.03],
        "qb": [0.97, 0.99, 0.98, 1.00],
    }
    suite = suite_geomean_interval(per_query)
    assert suite is not None
    assert_close(suite[1], 0.9998249540392277)
    assert_close(suite[0], 0.9909165871741309)
    assert_close(suite[2], 1.0088134073628925)
    # Incomplete coverage (a query below two pairs) must not produce a suite number.
    assert suite_geomean_interval({"qa": per_query["qa"], "qb": [1.0]}) is None
    assert suite_geomean_interval({}) is None


# --- arming evaluation ----------------------------------------------------------------------------


def test_evaluate_arming_predicates() -> None:
    deltas = {"top_n_producers_eligible": 2, "top_n_offers": 5, "producers_enabled": 4}
    assert evaluate_arming(deltas, {"exact": {"top_n_producers_eligible": 2}}) == []
    assert evaluate_arming(deltas, {"exact": {"top_n_producers_eligible": 4}}) != []
    assert evaluate_arming(deltas, {"nonzero": ["top_n_offers"]}) == []
    assert evaluate_arming(deltas, {"nonzero": ["top_n_prefilter_rows_in"]}) != []
    assert evaluate_arming(deltas, {"zero": ["top_n_prefilter_rows_in"]}) == []
    assert evaluate_arming(deltas, {"zero": ["top_n_offers"]}) != []
    # zero_prefix sweeps every matching field; the non-matching producers_enabled may move.
    assert (
        evaluate_arming(
            {"producers_enabled": 4, "top_n_offers": 0}, {"zero_prefix": ["top_n_"]}
        )
        == []
    )
    assert evaluate_arming(deltas, {"zero_prefix": ["top_n_"]}) != []


def test_evaluate_cell_arming() -> None:
    totals = {"top_n_group_witness_set_full": 3, "top_n_group_offers": 0}
    assert (
        evaluate_cell_arming(totals, {"cell_nonzero": ["top_n_group_witness_set_full"]})
        == []
    )
    assert evaluate_cell_arming(totals, {"cell_nonzero": ["top_n_group_offers"]}) != []


# --- schedule dry runs over a stub executor -------------------------------------------------------


class StubExecutor:
    """Deterministic arm executor: fixed per-arm elapsed times (per-pair overridable), matching
    result texts unless told otherwise, and empty counter deltas unless supplied."""

    def __init__(
        self,
        off_s: float = 1.0,
        on_s: float = 1.0,
        per_pair_on_s: Optional[Dict[int, float]] = None,
        mismatch_pairs: Tuple[int, ...] = (),
        deltas: Optional[Dict[str, Dict[str, int]]] = None,
        fail: Optional[Tuple[str, int]] = None,
    ) -> None:
        self.off_s = off_s
        self.on_s = on_s
        self.per_pair_on_s = per_pair_on_s or {}
        self.mismatch_pairs = mismatch_pairs
        self.deltas = deltas or {"off": {}, "on": {}}
        self.fail = fail
        self.calls: List[Tuple[str, int, str]] = []

    def __call__(self, arm: str, pair_idx: int, phase: str) -> ExecutionOutcome:
        self.calls.append((arm, pair_idx, phase))
        if self.fail == (arm, pair_idx):
            return ExecutionOutcome(ok=False, error="stub failure")
        elapsed = (
            self.off_s if arm == "off" else self.per_pair_on_s.get(pair_idx, self.on_s)
        )
        text = "row\n"
        if arm == "on" and phase == "sample" and pair_idx in self.mismatch_pairs:
            text = "different row\n"
        return ExecutionOutcome(
            ok=True, elapsed_s=elapsed, result_text=text, deltas=dict(self.deltas[arm])
        )


def quiet_occupancy() -> GpuOccupancy:
    return GpuOccupancy(available=True, self_attributed=True, self_bytes=1 << 30)


def foreign_occupancy() -> GpuOccupancy:
    return GpuOccupancy(
        available=True, foreign_bytes=200 << 20, foreign_process_count=1
    )


def no_sleep(_: float) -> None:
    pass


def make_clock(step_s: float = 1.0):
    state = {"now": 0.0}

    def now() -> float:
        state["now"] += step_s
        return state["now"]

    return now


def run(executor, plan, arming_spec=None, occupancy_fn=quiet_occupancy, pair_log=None):
    return run_ab_query_cell(
        "q_test",
        executor,
        plan,
        arming_spec,
        occupancy_fn=occupancy_fn,
        sleep_fn=no_sleep,
        now_fn=make_clock(),
        pair_log=pair_log,
    )


def test_lead_arm_alternates_and_warmups_are_discarded() -> None:
    executor = StubExecutor()
    records: List[PairRecord] = []
    plan = AbQueryPlan(
        target_resolution=0.02, min_pairs=4, max_pairs=10, warmup_pairs=2
    )
    result = run(executor, plan, pair_log=records.append)

    warmup_calls = [c for c in executor.calls if c[2] == "warmup"]
    assert [c[0] for c in warmup_calls] == [
        "off",
        "on",
        "off",
        "on",
    ]  # warmups: fixed off lead
    sample_calls = [c for c in executor.calls if c[2] == "sample"]
    # Pair 0 leads off, pair 1 leads on, alternating (harness: lead = pair % 2).
    assert [c[0] for c in sample_calls[:4]] == ["off", "on", "on", "off"]

    # A perfectly flat stub resolves as soon as min_pairs is reached; warmups are not samples.
    assert result.attempted_pairs == plan.min_pairs
    assert len(result.paired_ratios) == plan.min_pairs
    assert all(r.phase == "warmup" and not r.kept for r in records[: plan.warmup_pairs])


def test_foreign_bracket_discards_and_retries() -> None:
    observations = iter(
        [quiet_occupancy(), foreign_occupancy()] + [quiet_occupancy()] * 50
    )
    executor = StubExecutor()
    plan = AbQueryPlan(
        target_resolution=0.02, min_pairs=3, max_pairs=10, warmup_pairs=0
    )
    result = run(executor, plan, occupancy_fn=lambda: next(observations))

    # Bracket order: cell-before, then one per pair. Pair 0 sees the foreign bracket and is
    # discarded; the loop retries with fresh pairs. min_pairs floors KEPT ratios (deliberately
    # stricter than the harness's attempted-pair floor): a discard consumes an attempt but never
    # counts toward the floor, so a discard-heavy cell cannot quote a resolution over a handful
    # of survivors.
    assert result.pairs_with_foreign == 1
    assert len(result.discard_notes) == 1
    assert "foreign GPU tenant" in result.discard_notes[0]
    assert result.attempted_pairs == plan.min_pairs + 1
    assert len(result.paired_ratios) == plan.min_pairs


def test_execution_error_discards_the_pair() -> None:
    executor = StubExecutor(fail=("on", 1))
    plan = AbQueryPlan(
        target_resolution=0.02, min_pairs=3, max_pairs=10, warmup_pairs=0
    )
    result = run(executor, plan)
    assert result.discard_notes == ["pair 1: on arm failed: stub failure"]
    # The kept floor: the failed pair consumes an attempt but a replacement pair must still run.
    assert result.attempted_pairs == plan.min_pairs + 1
    assert len(result.paired_ratios) == plan.min_pairs


def test_max_pairs_caps_an_unresolvable_query() -> None:
    # Alternating 10% swings: far too noisy for a 0.1% target within 8 pairs.
    executor = StubExecutor(
        per_pair_on_s={i: 1.1 if i % 2 == 0 else 0.9 for i in range(20)}
    )
    plan = AbQueryPlan(
        target_resolution=0.001, min_pairs=2, max_pairs=8, warmup_pairs=0
    )
    result = run(executor, plan)
    assert result.attempted_pairs == plan.max_pairs
    assert not is_resolved(result.paired_ratios, plan.target_resolution)
    assert (
        pairs_needed_for_target(result.paired_ratios, plan.target_resolution)
        > plan.max_pairs
    )


def test_pilot_stops_at_kept_pair_count() -> None:
    executor = StubExecutor(per_pair_on_s={i: 1.0 + 0.01 * (i % 3) for i in range(20)})
    plan = AbQueryPlan(
        target_resolution=0.5, min_pairs=2, max_pairs=50, warmup_pairs=1, pilot_pairs=4
    )
    result = run(executor, plan)
    assert len(result.paired_ratios) == 4  # exactly the pilot count, resolution ignored


def test_result_mismatch_aborts_the_run() -> None:
    executor = StubExecutor(mismatch_pairs=(1,))
    plan = AbQueryPlan(
        target_resolution=0.001, min_pairs=5, max_pairs=10, warmup_pairs=0
    )
    try:
        run(executor, plan)
    except AbRunAbort as e:
        assert "differs" in str(e)
    else:
        raise AssertionError("expected AbRunAbort")


def test_arming_violation_aborts_the_cell() -> None:
    executor = StubExecutor(
        deltas={
            "off": {"top_n_producers_eligible": 0},
            "on": {"top_n_producers_eligible": 0},
        }
    )
    plan = AbQueryPlan(target_resolution=0.02, min_pairs=2, max_pairs=5, warmup_pairs=1)
    spec = {"on": {"exact": {"top_n_producers_eligible": 2}}}
    try:
        run(executor, plan, arming_spec=spec)
    except AbCellAbort as e:
        assert "arming violation" in str(e)
    else:
        raise AssertionError("expected AbCellAbort")


def test_cell_level_arming_violation_aborts_after_sampling() -> None:
    executor = StubExecutor(deltas={"off": {}, "on": {"top_n_group_offers": 0}})
    plan = AbQueryPlan(target_resolution=0.5, min_pairs=2, max_pairs=5, warmup_pairs=0)
    spec = {"on": {"cell_nonzero": ["top_n_group_offers"]}}
    try:
        run(executor, plan, arming_spec=spec)
    except AbCellAbort as e:
        assert "cell-level" in str(e)
    else:
        raise AssertionError("expected AbCellAbort")


def test_persistent_foreign_tenant_parks_until_quiet() -> None:
    # Foreign brackets long enough to cross the park threshold, then a quiet host. The park loop
    # itself needs two consecutive quiet observations before resuming.
    observations = iter([foreign_occupancy()] * 6 + [quiet_occupancy()] * 50)
    executor = StubExecutor()
    plan = AbQueryPlan(
        target_resolution=0.02,
        min_pairs=2,
        max_pairs=20,
        warmup_pairs=0,
        park_threshold_s=1.0,
        park_poll_s=1.0,
    )
    result = run_ab_query_cell(
        "q_test",
        executor,
        plan,
        None,
        occupancy_fn=lambda: next(observations),
        sleep_fn=no_sleep,
        now_fn=make_clock(step_s=2.0),
    )
    assert result.parked_s > 0.0
    assert result.pairs_with_foreign >= 2
    assert len(result.paired_ratios) == plan.min_pairs


def test_require_cell_nonzero_merges_into_expectations() -> None:
    from performance_test import load_arming_expectations

    doc = {"queries": {"q18": {"on": {"cell_nonzero": ["top_n_group_offers"]}}}}
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f)
    try:
        merged = load_arming_expectations(
            path, [18], {"q18": ["top_n_group_witness_set_full", "top_n_group_offers"]}
        )
        # Appended after the frozen entries, deduplicated, frozen order preserved.
        assert merged["queries"]["q18"]["on"]["cell_nonzero"] == [
            "top_n_group_offers",
            "top_n_group_witness_set_full",
        ]
        try:
            load_arming_expectations(path, [18], {"q99": ["anything"]})
            raise AssertionError(
                "expected SystemExit for a query with no expectations entry"
            )
        except SystemExit:
            pass
    finally:
        os.unlink(path)


def test_setup_ab_dirs_refuses_overwriting_a_completed_cell() -> None:
    from performance_test import setup_ab_dirs

    with tempfile.TemporaryDirectory() as root:
        _, cell_dir, _ = setup_ab_dirs(root, "host_pinned")
        # Rerunning before the report lands is fine (the abort/resume case).
        setup_ab_dirs(root, "host_pinned")
        with open(os.path.join(cell_dir, "cell_report.json"), "w") as f:
            f.write("{}")
        # A completed cell must not be overwritten in place: this destroyed the full HOST-pinned
        # acceptance report once (subset reruns share the date+commit directory key).
        try:
            setup_ab_dirs(root, "host_pinned")
            raise AssertionError("expected SystemExit for a completed cell directory")
        except SystemExit as e:
            assert "cell_report.json" in str(e)
        # Sibling cells in the same run directory stay unaffected...
        setup_ab_dirs(root, "disk_hot")
        # ...and an explicit --name is the sanctioned rerun path.
        _, named_cell, _ = setup_ab_dirs(root, "host_pinned", name="rerun_q18")
        assert os.path.isdir(named_cell)
        assert named_cell != cell_dir


def test_verify_pinned_cache_hits_verdict_map() -> None:
    from performance_test import verify_pinned_cache_hits

    marker = "scan served from pinned cache"
    with tempfile.TemporaryDirectory() as cell_dir:
        os.makedirs(os.path.join(cell_dir, "q1"))
        with open(os.path.join(cell_dir, "q1", "sirius.log"), "w") as f:
            f.write(f"[info] [sirius_scan_manager] {marker} for entry lineitem\n")
        os.makedirs(os.path.join(cell_dir, "q2"))
        with open(os.path.join(cell_dir, "q2", "sirius.log"), "w") as f:
            f.write("[info] a log with no serve marker\n")
        # q3 has no log at all.
        verdicts = verify_pinned_cache_hits(cell_dir, [1, 2, 3])
        assert verdicts == {"q1": "ok", "q2": "no-cache-hit", "q3": "missing-log"}


def test_pinned_serve_marker_matches_the_emitter() -> None:
    # Two-sided string contract: the verifier greps for a literal the engine must emit. A rename
    # on either side already voided one live SF1000 pilot cell as 'no-cache-hit'; this pins the
    # matcher's literal against the scan manager's emitter so the drift fails here instead.
    emitter = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "src",
        "scan_manager",
        "sirius_scan_manager.cpp",
    )
    with open(emitter, errors="replace") as f:
        assert "scan served from pinned cache" in f.read()


def main() -> None:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("all tests passed")


if __name__ == "__main__":
    main()
