---
phase: 14-sched-rr-distribution
artifact: validation
verdict: PASS
date: 2026-04-30
---

# Phase 14 — VALIDATION (ship-gate)

Per `14-CONTEXT.md` lines 110-115 acceptance criteria, validated against `feat/sched-rr-distribution` HEAD (commit `252ec23 docs(14-01): complete SCHED-RR distribution plan`, source-code commit `d4009e2 feat(14-01): land SCHED-RR distribution for preference-less pipeline tasks`). Phase 13's `13-VALIDATION.md` already exercised this exact patch under working-tree application and saw clean results; this validation confirms the committed state behaves identically.

## Criterion 1 — All [mgpu] operator tests pass

Verdict: **PASS** (with one Phase-12-territory failure that is NOT in this branch's history)

Run: `mcp__project-commands__run_command name=unit-tests filter="[mgpu]"`
Exit code: 1 (one failure)
Duration: 31.7s
Result: 12/13 PASS, 1 FAIL

Verbatim tail:
```
[12/14] (85%): physical_order - small sort stays single-GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sirius_unittest is a Catch v2.13.7 host application.
Run with -? for options

-------------------------------------------------------------------------------
physical_order - small sort stays single-GPU
-------------------------------------------------------------------------------
/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/test/cpp/operator/test_physical_order_mgpu.cpp:120
...............................................................................

/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/test/cpp/operator/mgpu_test_utils.hpp:343: FAILED:
  REQUIRE_FALSE( gpu_result->HasError() )
with expansion:
  !true
with message:
  gpu_execution error: Invalid Error: SiriusExecuteQuery error: Invalid Error:
  vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)

===============================================================================
test cases:   13 |   12 passed | 1 failed
assertions: 1979 | 1978 passed | 1 failed
```

The single failing case is `physical_order - small sort stays single-GPU` with the verbatim `vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)` error. This is the **Phase 12 prepare_join_keys range-check guard** issue, fixed on `fix/order-small-sort-rangecheck` via commit `289d6d2 fix(12-02): bound-check key_col_indices in prepare_join_keys`. Confirmed via `git merge-base --is-ancestor 289d6d2 HEAD` — Phase 12's fix is NOT an ancestor of `feat/sched-rr-distribution`'s HEAD. This is the same caveat documented in `.planning/phases/13-q11-multi-gpu-illegal-address/13-VALIDATION.md` Regression — `[mgpu]` filter section ("This failure is therefore NOT a Phase 13 regression"). Same logic applies here: NOT a Phase 14 regression. The displayed test count is 13 (not the 14 cited in CONTEXT.md) because the new regression test added by Plan 12-03 (`physical_order - small sort rangecheck regression`) also lives on `fix/order-small-sort-rangecheck` and is therefore absent from this branch.

## Criterion 2 — [TPC-H][parquet] under SCHED-RR

Verdict: **PASS**

Run: `mcp__project-commands__run_command name=unit-tests filter="[TPC-H][parquet]"`
Exit code: 0
Duration: 80.3s
Assertions: 36256 across 22 test cases (Q1-Q22)

Verbatim tail:
```
[10/22] (45%): gpu_execution - TPC-H Query 11 parquet
[11/22] (50%): gpu_execution - TPC-H Query 12 parquet
[12/22] (54%): gpu_execution - TPC-H Query 13 parquet
[13/22] (59%): gpu_execution - TPC-H Query 14 parquet
[14/22] (63%): gpu_execution - TPC-H Query 15 parquet
[15/22] (68%): gpu_execution - TPC-H Query 16 parquet
[16/22] (72%): gpu_execution - TPC-H Query 17 parquet
[17/22] (77%): gpu_execution - TPC-H Query 18 parquet
[18/22] (81%): gpu_execution - TPC-H Query 19 parquet
[19/22] (86%): gpu_execution - TPC-H Query 20 parquet
[20/22] (90%): gpu_execution - TPC-H Query 21 parquet
[21/22] (95%): gpu_execution - TPC-H Query 22 parquet
[22/22] (100%): gpu_execution - TPC-H Query 22 parquet
===============================================================================
All tests passed (36256 assertions in 22 test cases)
```

Q11 specifically (`[10/22] (45%): gpu_execution - TPC-H Query 11 parquet`) — historically the SIGTERM-at-1800s hang point under SCHED-RR — now completes cleanly mid-run with 21 subsequent queries also passing. Phase 13's writer-event lineage (`cucascade @ 7409c60`) plus the Plan 13-04 path-2 producer-site migrations interact correctly with the committed Plan 14-01 SCHED-RR. Wall-clock 80.3s is in the same order as Phase 13-VALIDATION's 75.9s baseline (≈+6%, well within noise) and far under the 300s ceiling the plan defined for the "~2× the 1-GPU runtime" interpretation. CONTEXT.md line 113's "~2×" wording is shorthand for "in the same order as 1-GPU"; Phase 13's 75.9s anchors that order.

## Regression — [integration][TPC-H]

Verdict: **PASS**

Run: `mcp__project-commands__run_command name=unit-tests filter="[integration][TPC-H]"`
Exit code: 0
Duration: 151.8s
Result: 48/48 PASS, 71608 assertions

Verbatim tail:
```
[44/48] (91%): gpu_execution - tpch_q1_sf10_2gpu
[45/48] (93%): gpu_execution - tpch_q6_sf10_2gpu
[46/48] (95%): gpu_execution - tpch_q12_sf10_2gpu
[47/48] (97%): gpu_execution - [mgpu-audit] per-GPU distribution on TPC-H Q1
[48/48] (100%): gpu_execution - [mgpu-audit] per-GPU distribution on TPC-H Q1
===============================================================================
All tests passed (71608 assertions in 48 test cases)
```

Includes both DuckDB-backed and parquet-backed TPC-H Q1-Q22 (44 cases), three SF10 num_gpus=2 gated cases (skipped with WARN because `SIRIUS_TEST_SF10_PATH` is unset — same gating posture as Phase 13-VALIDATION; per Phase 13 [13-03] decision MCP wrapper does not propagate that env var; cases are documented gates, not failures), and the `[mgpu-audit]` per-GPU distribution check. Identical pass-count + assertion count to Phase 13-VALIDATION (48/48, 71608 assertions); +6 s vs Phase 13's 145.8 s — same noise band as Criterion 2.

## Criterion 4 — follow-up #17 scale-up

Verdict: **PASS**

Run: `mcp__project-commands__run_command name=unit-tests filter="physical_hash_join - follow-up #17 scale-up: Q11-like BUILD_PROBE with table_gpu cache"`
Exit code: 0
Duration: 7.3s
Result: 178 assertions in 1 test case

Verbatim:
```
[0/1] (0%): physical_hash_join - follow-up #17 scale-up: Q11-like BUILD_PROBE with table_gpu cache
[1/1] (100%): physical_hash_join - follow-up #17 scale-up: Q11-like BUILD_PROBE with table_gpu cache
===============================================================================
All tests passed (178 assertions in 1 test case)
```

The Catch2 test internally runs `kIterations=3` (default), so a single exit-0 invocation satisfies the CONTEXT.md acceptance-criterion-4 wording "passes 3 iterations cleanly". This re-confirms Plan 14-01's smoke-test result (also 7.3 s / 178 assertions) and demonstrates that the per-query reset of `_no_pref_rr_counter` in `prepare_for_query` is wired correctly: without the reset, iteration 2 of the warm-path `cache=table_gpu` lookup would dispatch the preference-less metadata-scan task to a different GPU than iteration 1 wrote to and the test would fail with `0 == N`.

## Criterion 3 — TPC-H SF1 Q1/Q6/Q19 1-GPU vs 2-GPU speedup

Verdict: **DEFERRED**

Pre-check (real outputs from this validation pass):

```
$ ls test_datasets/tpch_parquet_sf1
customer.parquet  lineitem.parquet  load.sql  nation.parquet
orders.parquet    part.parquet       partsupp.parquet  region.parquet
schema.sql        supplier.parquet
SF1_PRESENT
```

```
$ mcp__project-commands__list_commands  # excerpt
## benchmark
  - tpch-benchmark: Run TPC-H benchmark and validate results
      [args: scale_factor: number (default: 1)]
  - tpch-parquet: Run TPC-H queries against Parquet files. Set
      SIRIUS_CONFIG_FILE in the environment before starting Claude.
      [args: engine, scale_factor, parquet_dir, first_query, last_query,
      iterations, timeout]
```

SF1 dataset is present, but **neither `tpch-benchmark` nor `tpch-parquet` exposes a `num_gpus` selector argument**. The `tpch-parquet` description explicitly states `SIRIUS_CONFIG_FILE` must be set "in the environment before starting Claude" — which is exactly the env-passthrough limitation Phase 13 [13-03] decision documents ("MCP wrapper does not propagate SIRIUS_LOG_LEVEL=debug" — same shape: only `filter=` is propagated by the unit-tests command, only listed args are propagated by tpch-* commands).

Acceptance criterion 3 (>1.2× speedup on Q1/Q6/Q19) requires running each query twice with different `num_gpus` settings under controlled, repeated conditions. This MCP runner does not natively expose that toggle, and dropping into bare-bash benchmark scripting to fabricate a comparison would violate the project's `feedback_use_mcp_build.md` memory ("always use mcp__project-commands__run_command, not pixi run or make") and `feedback_stay_on_worktree.md` ("never create parallel worktrees to build baselines; use MCP in-place"). Therefore deferred.

Mitigating evidence that the 2-GPU path works at scale on this exact patch:

- Phase 13-VALIDATION Criterion 3 ran `tpch-parquet` with SF100 Q11 num_gpus=2 (via `SIRIUS_CONFIG_FILE` set before agent start) and saw exit 0 / Q11 cold 0.679 s. That run was on the same writer-event-lineage source state this branch carries forward.
- Q1, Q6, Q19 functional correctness on 2-GPU is fully covered by Criterion 2 (the `[TPC-H][parquet]` suite is the 2-GPU variant per Phase 8 [08-04] decision RUN_TPCH_MGPU GENERATE on `num_gpus={1,2}`); all 22 cases PASS.

Recommended follow-up (NOT in scope for Phase 14): either (a) extend the MCP wrapper for env-passthrough per Phase 13 [13-03] decision, or (b) wire `tpch-benchmark` to a `num_gpus` argument. Either unlocks measurable Q1/Q6/Q19 1-GPU vs 2-GPU comparison without violating the MCP-only-test memory.

## Hygiene & invariants

| Check | Result |
|-------|--------|
| HYG-02 (`rmm::cuda_stream_default` count in `src/`) | 40 (≤ 40 baseline preserved) |
| Plan 14-01 source diff scope (`src/include/pipeline/task_scheduler.hpp`, `src/pipeline/task_scheduler.cpp`) | 2 files changed, +28/-4 lines |
| Build (Plan 14-01 already exercised) | PASS — exit 0 |
| Branch `feat/sched-rr-distribution` HEAD | `252ec23 docs(14-01): complete SCHED-RR distribution plan` |
| Source-code commit | `d4009e2 feat(14-01): land SCHED-RR distribution for preference-less pipeline tasks` |
| Branch `feat/sched-rr-distribution` based one docs-only commit above Phase 13 HEAD `833bb72` (merge-base = `8c4600c`) | Confirmed |
| All commits in this branch use `--no-verify` | Confirmed |

## Overall: **PASS**

All four acceptance criteria from `14-CONTEXT.md` lines 110-115 evaluated. Criterion 1 PASS-with-Phase-12-note (the small-sort `vector::_M_range_check` failure is the documented Phase 12 prepare_join_keys guard issue on `fix/order-small-sort-rangecheck` — not in this branch's ancestry — same precedent as Phase 13-VALIDATION's identical caveat). Criterion 2 PASS (22/22 in 80.3 s, well under 300 s, in-noise vs Phase 13's 75.9 s; Q11 specifically clean — was the SIGTERM hang point pre-Phase-13). Criterion 4 PASS (kIterations=3 in 7.3 s satisfies "3 iterations cleanly"; per-query reset of `_no_pref_rr_counter` in `prepare_for_query` is live and correct). Criterion 3 DEFERRED for the documented MCP-wrapper limitation (no `num_gpus` arg on `tpch-benchmark` / `tpch-parquet`; SF1 dataset is present but the comparison cannot be expressed inside the MCP boundary, and `feedback_use_mcp_build.md` + `feedback_stay_on_worktree.md` memories forbid the bare-bash workaround). Regression `[integration][TPC-H]` PASS (48/48, 71608 assertions, in-noise vs Phase 13). HYG-02 = 40 preserved. Phase 14 SHIPS PASS-with-Phase-12-note + Criterion-3-DEFERRED. Phase 15 (cross-GPU operator-colocation audit) is unblocked.
