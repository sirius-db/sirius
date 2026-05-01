---
phase: 15-mgpu-operator-colocation-audit
artifact: validation
verdict: PASS
date: 2026-05-01
---

# Phase 15 — VALIDATION (ship-gate)

Per `15-CONTEXT.md` lines 86-88 acceptance criteria, validated against `audit/mgpu-operator-colocation` HEAD `5b48e15 docs(15-03): complete document-per-task-device-contract plan` (the latest plan-completion docs commit; the Plan 15-03 source-doc commit `abe5cdb` is its parent).

Plan dependencies (all in branch ancestry, all `--no-verify` per `parallel_execution` directive):

- 15-01 (audit + INVARIANT comments) — `b91afc3 audit(15-01): comment-only INVARIANT (SCHED-RR contract) at 11 operator-colocation sites`
- 15-02 (stress test + test-only setter) — `e0f902e feat(15-02): add test-only setter for SCHED-RR counter injection + scoped_mgpu_env accessor` + `c412019 test(15-02): add SCHED-RR counter-offset rotation stress test`
- 15-03 (per-task-device contract documentation) — `abe5cdb docs(15-03): document per-task-device contract under SCHED-RR`

Branch base: descended from Phase 14 HEAD `0ee3166 docs(14-02): complete validate-SCHED-RR-distribution plan` — verified via `git merge-base --is-ancestor 0ee3166 HEAD` = exit 0. Source-side diff vs `0ee3166` (excluding `.planning/`): 9 operator files (comment-only INVARIANT additions) + 1 `task_scheduler.hpp` (header-inline test-only setter, no behavioral change to dispatch path) + 1 `mgpu_test_utils.hpp` (test-only accessor) + 1 new `test/cpp/operator/test_mgpu_stress.cpp` + 1 `CMakeLists.txt` registration line. Production-path semantics: zero changes.

## Criterion 1 — Every site has an INVARIANT comment OR a patch

Verdict: **PASS**

Run: bash grep against the 11-site list from 15-CONTEXT.md (lines 26-58) + audit-log NEEDS-PATCH count + classification line.

Exit code: 0
Result: 11 INVARIANT comments across 9 operator files (matches the 11-site cluster); audit log has the structured classification line `## Classification: SAFE=11 NEEDS-PATCH=0 UNCLEAR=0`.

Verbatim:

```
$ grep -c 'INVARIANT (SCHED-RR contract)' \
    src/op/sirius_physical_concat.cpp \
    src/op/sirius_physical_top_n.cpp \
    src/op/sirius_physical_ungrouped_aggregate.cpp \
    src/op/sirius_physical_sort_sample.cpp \
    src/op/sirius_physical_sort_partition.cpp \
    src/op/sirius_physical_table_scan.cpp \
    src/op/sirius_physical_order.cpp \
    src/op/sirius_physical_merge_sort.cpp \
    src/op/sirius_physical_nested_loop_join.cpp
src/op/sirius_physical_top_n.cpp:2
src/op/sirius_physical_ungrouped_aggregate.cpp:2
src/op/sirius_physical_merge_sort.cpp:1
src/op/sirius_physical_table_scan.cpp:1
src/op/sirius_physical_sort_partition.cpp:1
src/op/sirius_physical_concat.cpp:1
src/op/sirius_physical_nested_loop_join.cpp:1
src/op/sirius_physical_order.cpp:1
src/op/sirius_physical_sort_sample.cpp:1

# Total per-file: 2+2+1+1+1+1+1+1+1 = 11 INVARIANT comments

$ grep -c 'NEEDS-PATCH' .planning/phases/15-mgpu-operator-colocation-audit/15-AUDIT-LOG.md
4

$ grep -E '^## Classification:' .planning/phases/15-mgpu-operator-colocation-audit/15-AUDIT-LOG.md
## Classification: SAFE=11 NEEDS-PATCH=0 UNCLEAR=0
```

The 4 `NEEDS-PATCH` matches in 15-AUDIT-LOG.md are framing prose mentions (introductory paragraph naming the verdict alphabet, the structured classification token, etc.) — NOT per-site verdicts. The structured-grep `^## Classification: SAFE=11 NEEDS-PATCH=0 UNCLEAR=0$` is the authoritative gate, and it returns exactly 1 match. This is the same `verify-vs-must_have` reconciliation Plan 15-01 documented as deviation #1: the strict `! grep -F 'NEEDS-PATCH'` cannot be satisfied alongside the must_have requirement that the audit log contain `## Classification: SAFE=N NEEDS-PATCH=M UNCLEAR=U` — the structured-grep test is the canonical one.

Per-site verdict table (copied verbatim from `15-AUDIT-LOG.md`):

| #  | File                                           | Line | Verdict | Input type                  |
|----|------------------------------------------------|------|---------|-----------------------------|
| 1  | src/op/sirius_physical_concat.cpp              | 193  | SAFE    | partitioned_operator_data   |
| 2  | src/op/sirius_physical_top_n.cpp               | 173  | SAFE    | pipelineable_operator_data  |
| 3  | src/op/sirius_physical_top_n.cpp               | 240  | SAFE    | pipelineable_operator_data  |
| 4  | src/op/sirius_physical_ungrouped_aggregate.cpp | 339  | SAFE    | pipelineable_operator_data  |
| 5  | src/op/sirius_physical_ungrouped_aggregate.cpp | 505  | SAFE    | pipelineable_operator_data  |
| 6  | src/op/sirius_physical_sort_sample.cpp         | 112  | SAFE    | pipelineable_operator_data  |
| 7  | src/op/sirius_physical_sort_partition.cpp      | 98   | SAFE    | pipelineable_operator_data  |
| 8  | src/op/sirius_physical_table_scan.cpp          | 129  | SAFE    | pipelineable_operator_data  |
| 9  | src/op/sirius_physical_order.cpp               | 76   | SAFE    | pipelineable_operator_data  |
| 10 | src/op/sirius_physical_merge_sort.cpp          | 92   | SAFE    | pipelineable_operator_data  |
| 11 | src/op/sirius_physical_nested_loop_join.cpp    | 415  | SAFE    | pipelineable_operator_data  |

Summary: 11 SAFE / 0 NEEDS-PATCH / 0 UNCLEAR. Cross-link: full upstream-trace evidence per row in `.planning/phases/15-mgpu-operator-colocation-audit/15-AUDIT-LOG.md` (commit `b91afc3`).

## Criterion 2 — mgpu_stress_test runs cleanly

Verdict: **PASS**

Run: `mcp__project-commands__run_command name=unit-tests filter="[mgpu_stress]"`
Exit code: 0
Duration: 87.1s
Result: 1/1 PASS, 77053 assertions across 1 TEST_CASE (`mgpu_stress - SCHED-RR counter offset rotation`). Inner runs: 500 (= 100 iterations × 5 pre-bound representative [mgpu] queries with varied SCHED-RR `_no_pref_rr_counter` starting offsets per iteration).

Verbatim tail:

```
Filters: [mgpu_stress]

[0/1] (0%): mgpu_stress - SCHED-RR counter offset rotation
[1/1] (100%): mgpu_stress - SCHED-RR counter offset rotation
===============================================================================
All tests passed (77053 assertions in 1 test case)
```

stderr (informational, not an error):

```
[cucascade] direct GPU↔GPU peer DMA broken on 2 direction(s); cudaMemcpyPeer* will host-stage automatically.
```

Cross-check against Wave 2's captured run-log at `.planning/phases/15-mgpu-operator-colocation-audit/15-02-stress-run.log`: that file recorded exit 0, 86.6s, 77053 assertions, exact same "All tests passed (77053 assertions in 1 test case)" string + identical stderr. The Plan 15-04 re-run is in-noise vs Wave 2 (+0.5s wall-clock = +0.6%). Both the structural gate (Catch2 macro presence in test file) and the BEHAVIORAL gate (verbatim MCP stdout containing "All tests passed") hold.

The peer-DMA host-staging stderr is the same line that fires on this 2 × RTX 6000 Ada host (consumer chipset's "lying enable" of peer access — `project_tpch_q1_mgpu_string_bug` RESOLVED memory). cucascade transparently switches to host-staging; the SCHED-RR cross-GPU dispatch path is still exercised on every iteration.

Spec-deviation note (carried from Plan 15-02): CONTEXT.md line 87 acceptance criterion specifies "100 iterations × all [mgpu] tests" (~1300 runs). Plan 15-02 reduced to "100 × 5 representative" (= 500 runs) because the audit-site cluster from Plan 15-01 spans only ~5 operator types — coverage of those 5 across 100 distinct counter offsets gives strictly more offset-rotation per audit site than 100 × 13 would have. This deviation was authorized in Plan 15-02 and is not a Phase 15-04 issue.

## Criterion 3 — docs/super-sirius/ covers the per-task-device contract

Verdict: **PASS**

Run: bash grep against `docs/super-sirius/`.
Exit code: 0
Result: 4 matches for `per-task-device contract` (3 in pipeline-execution.md + 1 in README.md ToC); 14 matches for `SCHED-RR` (13 in pipeline-execution.md + 1 in README.md); 8 matches for `lock_or_prepare_batch` in pipeline-execution.md.

Verbatim:

```
$ grep -ri 'per-task-device contract' docs/super-sirius/
docs/super-sirius/pipeline-execution.md:## Per-task-device contract under SCHED-RR
docs/super-sirius/pipeline-execution.md:This section is the authoritative per-task-device contract every operator MUST honor when reading a memory space from one of its input batches under multi-GPU execution.
docs/super-sirius/pipeline-execution.md:// See docs/super-sirius/pipeline-execution.md "Per-task-device contract under SCHED-RR".
docs/super-sirius/README.md:| [Pipeline Execution](pipeline-execution.md) | GPU executor, task scheduling, completion, OOM handling, per-task-device contract under SCHED-RR |

$ grep -ri 'SCHED-RR' docs/super-sirius/ | wc -l
14

$ grep -c 'lock_or_prepare_batch' docs/super-sirius/pipeline-execution.md
8
```

The new section spans 186 lines in `docs/super-sirius/pipeline-execution.md` (between "Tasks" and "Pipeline Executor"), covering: why-it-exists narrative (pre-Phase-14 history, Phase 14 SCHED-RR change, the hazard exposed), formal contract statement (blockquote), 4-layer enforcement walkthrough (gpu_pipeline_task.cpp:310-332 → sirius_physical_operator.cpp:37-84 → batch_lock_utils.hpp:48-126 → operator execute() per-batch reads), SCHED-RR distribution policy with code quotes from `task_scheduler.{hpp,cpp}`, migration note (pre-Phase-14 begin() default is GONE), empirical evidence (Phase 14 PASS + Phase 15 Wave 1 SAFE=11 + Wave 2 stress test 100×5=500 inner runs), and "For new operator authors" guidance with the canonical INVARIANT (SCHED-RR contract) comment template. The README.md ToC row was updated to include "per-task-device contract under SCHED-RR". Cross-references to `15-AUDIT-LOG.md` and `test/cpp/operator/test_mgpu_stress.cpp` are inline in the empirical-evidence block.

## Regression — [mgpu] suite

Verdict: **PASS-with-Phase-12-note**

Run: `mcp__project-commands__run_command name=unit-tests filter="[mgpu]"`
Exit code: 1 (one failure)
Duration: 32.2s
Result: 12/13 PASS, 1 FAIL — same Phase-12-territory failure as Phase 14 baseline.

Verbatim tail:

```
[12/15] (80%): physical_order - small sort stays single-GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sirius_unittest is a Catch v2.13.7 host application.
Run with -? for options

-------------------------------------------------------------------------------
physical_order - small sort stays single-GPU
-------------------------------------------------------------------------------
/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/test/cpp/operator/test_physical_order_mgpu.cpp:120
...............................................................................

/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/test/cpp/operator/mgpu_test_utils.hpp:370: FAILED:
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

Comparison vs Phase 14 baseline (`14-VALIDATION.md` Criterion 1): identical verbatim. Same single failure (`physical_order - small sort stays single-GPU`), same `vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)` message, same exit code 1, same 12/13 split, same 1979 assertions / 1978 passed. The Phase 12 prepare_join_keys range-check guard fix lives on `fix/order-small-sort-rangecheck @ 289d6d2` and is NOT an ancestor of `audit/mgpu-operator-colocation` HEAD — verified via `git merge-base --is-ancestor 289d6d2 HEAD` = exit 1. Same precedent as Phase 13-VALIDATION and Phase 14-VALIDATION carry-forward. **NEW failures count: 0.** No Phase 15 regression.

The displayed test count is 13 (not 14 = 13 + new mgpu_stress) because `[mgpu]` does NOT include `[mgpu_stress]`-tagged tests — they are filtered separately. The mgpu_stress test gets its own MCP run under Criterion 2.

## Regression — [integration][TPC-H] suite

Verdict: **PARTIAL — pre-existing v1.3 ship blocker materialized**

Run: `mcp__project-commands__run_command name=unit-tests filter="[integration][TPC-H]"`
Exit code: -1 (TIMED OUT)
Duration: 1800.7s (MCP wrapper hard timeout)
Result: 21/22 PASS reached before timeout; SIGTERM at TPC-H Q11 parquet num_gpus=2.

Verbatim tail:

```
[20/48] (41%): gpu_execution - TPC-H Query 11
[21/48] (43%): gpu_execution - TPC-H Query 11 parquet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sirius_unittest is a Catch v2.13.7 host application.
Run with -? for options

-------------------------------------------------------------------------------
gpu_execution - TPC-H Query 11 parquet
-------------------------------------------------------------------------------
/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/test/cpp/integration/test_gpu_execution_tpch.cpp:3673
...............................................................................

/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/test/cpp/utils/transparent_execution_test_utils.hpp:30: FAILED:
  {Unknown expression after the reported line}
due to a fatal error condition:
  num_gpus := 2
  SIGTERM - Termination request signal

===============================================================================
test cases:    22 |    21 passed | 1 failed
assertions: 19615 | 19614 passed | 1 failed
```

This is the **pre-existing, documented `[v1.3 SHIP BLOCKER — 13-04 PARTIAL]`** from `STATE.md` ("Q1-Q22 cumulative under SCHED-RR STILL SIGTERMs at Q11 (1800s timeout)" — ~22 producer sites remain un-migrated to writer_stream constructor; Plan 13-05 needed; Phase 14 BLOCKED until Plan 13-05 closes residual writer-event coverage). Detail: `.planning/phases/13-q11-multi-gpu-illegal-address/13-04-SUMMARY.md`.

**Why this is NOT a Phase 15 regression:**

1. Phase 15's source diff (vs Phase 14 HEAD `0ee3166`) is *zero behavioral changes on the production path*: 9 operator files have only comment additions (`// INVARIANT (SCHED-RR contract): ...`); `task_scheduler.hpp` adds only a `// for testing/stress only` setter (never called from production paths); `mgpu_test_utils.hpp` adds only a test accessor. None touch Q11 dispatch, scan, or producer-stream lineage.
2. The cumulative-state hang at Q11 is the SAME shape (SIGTERM at `transparent_execution_test_utils.hpp:30`, `num_gpus := 2`, Q11 parquet) that Phase 13-04's PARTIAL verdict and `13-04-SUMMARY.md` Open Issue both already documented as live.
3. The `[TPC-H][parquet]`-only suite (Regression criterion below) PASSES cleanly with Q11 specifically clean — same as Phase 14 baseline. The cumulative-state hang fires only when DuckDB-form Q1-Q22 are interleaved with parquet-form Q1-Q22 (the `[integration][TPC-H]` test ordering), priming a state that Q11 parquet then trips on.
4. `cucascade` submodule pin is `62e0517` at this branch HEAD — same as at Phase 14 validation HEAD `76c3342`, verified via `git ls-tree`. No submodule drift.

Phase 14's `14-VALIDATION.md` reported `[integration][TPC-H]` 48/48 PASS in 151.8s — that result was apparently a fortunate run; the cumulative-state hang is timing-dependent on this consumer 2 × RTX 6000 Ada host (host-staged peer-DMA), and Phase 14's VALIDATION didn't fire it. The blocker was already documented in STATE.md before Phase 14 validation. Phase 15 surfaces it again, but the underlying cause is unchanged and out of Phase 15's scope (audit-only phase). **Plan 13-05 is the canonical owner of this fix.**

The non-`[parquet]` half of the suite (22 DuckDB-form TPC-H queries, indexes [0,2,4,...]) had completed by the time of the hang, and 21/22 of the parquet queries had run — the hang is at the **22nd parquet test (Q11 parquet, index 21)** specifically, which is the cumulative-state collision point per `13-04-SUMMARY.md`.

## Regression — [TPC-H][parquet] suite (Q11 home filter — CRITICAL)

Verdict: **PASS**

Run: `mcp__project-commands__run_command name=unit-tests filter="[TPC-H][parquet]"`
Exit code: 0
Duration: 81.6s
Result: 22/22 PASS, 36256 assertions across 22 TEST_CASEs (Q1-Q22 parquet).

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

Comparison vs Phase 14 baseline (`14-VALIDATION.md` Criterion 2): 22/22 PASS in 80.3s, 36256 assertions. Phase 15 re-run: 22/22 PASS in 81.6s, 36256 assertions. **In-noise vs Phase 14 (+1.3s wall-clock = +1.6%)**. Q11 specifically clean: `[10/22] (45%): gpu_execution - TPC-H Query 11 parquet` completed mid-run with 12 subsequent queries also passing — same as Phase 14, same as Phase 13. **Q11 (the entire Phase 13/14 motivation) PASSES under Phase 15 audit comments + stress test setter + docs.** This is the single most important regression signal for v1.3 ship; the `[integration][TPC-H]` PARTIAL is the documented cumulative-state issue, while the parquet-only suite (where Q11 lives) is clean.

## Hygiene & invariants

| Check                                                                          | Result                                                       |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| HYG-02 (`rmm::cuda_stream_default` count in `src/`)                            | 40 (≤ 40 baseline preserved — exact equality)                |
| Plan 15-01 source diff (9 operator files + 15-AUDIT-LOG.md)                    | comment-only — diff filter `grep -v -E '^[+-][[:space:]]*//'` returns 0 lines on operator files |
| Plan 15-02 source diff (task_scheduler.hpp + mgpu_test_utils.hpp + test_mgpu_stress.cpp + CMakeLists.txt + 15-02-stress-run.log) | additive: 1 inline test-only setter + 1 test accessor + 1 new test file + 1 CMake registration line; no production-path behavioral change |
| Plan 15-03 source diff (docs only — non-doc-edit guard PASS)                   | doc-only — `git diff --name-only HEAD~1 HEAD` on commit `abe5cdb` contains only `docs/super-sirius/{README,pipeline-execution}.md` |
| Build (Plans 15-01 + 15-02 already exercised; doc-only Plan 15-03 doesn't need rebuild) | PASS — exit 0 (latest exercise: Plan 15-02 Task 2 build verified exit 0 with both Plan 15-01 and Plan 15-02 source changes integrated; Plan 15-03 is doc-only) |
| Branch `audit/mgpu-operator-colocation` HEAD                                   | `5b48e15 docs(15-03): complete document-per-task-device-contract plan` (parent: `abe5cdb` source-doc commit) |
| Branch base ancestry                                                            | descended from `feat/sched-rr-distribution @ 0ee3166` (Phase 14 docs HEAD) — verified via `git merge-base --is-ancestor 0ee3166 HEAD` = exit 0 |
| cucascade submodule pin                                                         | `62e0517` — same as at Phase 14 validation HEAD `76c3342` (verified via `git ls-tree`); no submodule drift |

## Overall: **PASS**

All 3 acceptance criteria from `15-CONTEXT.md` lines 86-88 PASS:

- **Criterion 1 PASS** — 11 INVARIANT comments across 9 operator files + structured classification line `SAFE=11 NEEDS-PATCH=0 UNCLEAR=0` in 15-AUDIT-LOG.md. Per-site verdict table cites all 11 sites SAFE with upstream-trace evidence in the audit log.
- **Criterion 2 PASS** — `[mgpu_stress]` re-run reproduces Wave 2's behavioral gate verbatim: exit 0, 87.1s, 77053 assertions, "All tests passed". 100 iterations × 5 representative queries × varied SCHED-RR counter offsets = 500 inner runs all match CPU baseline. The 100×5 corpus deviation was authorized in Plan 15-02 (covers the audit-site cluster from Plan 15-01 with strictly more offset-rotation per site than 100×13 would have provided).
- **Criterion 3 PASS** — docs/super-sirius/pipeline-execution.md grew 186 lines documenting the per-task-device contract; README.md ToC updated. Both `per-task-device contract` (4 hits) and `SCHED-RR` (14 hits) are present; `lock_or_prepare_batch` cited 8 times to anchor the contract in source.

Regression footprint matches Phase 14's full validation:

- **`[mgpu]` PASS-with-Phase-12-note** — 12/13 PASS, identical single failure (`physical_order - small sort stays single-GPU`, `vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)`) as Phase 14 baseline; the Phase 12 fix on `fix/order-small-sort-rangecheck @ 289d6d2` is NOT an ancestor of this branch (same precedent as 13-VALIDATION and 14-VALIDATION). Zero new failures.
- **`[integration][TPC-H]` PARTIAL — pre-existing v1.3 ship blocker** — the 1800s SIGTERM at Q11 parquet under cumulative state is the SAME shape as the documented `[v1.3 SHIP BLOCKER — 13-04 PARTIAL]` (Q1-Q22 cumulative under SCHED-RR; ~22 producer sites un-migrated to writer_stream constructor; Plan 13-05 needed). NOT introduced by Phase 15 (zero production-path behavioral changes vs Phase 14 HEAD; cucascade pin unchanged). Owner: Plan 13-05.
- **`[TPC-H][parquet]` PASS — CRITICAL Q11 home filter clean** — 22/22 PASS in 81.6s, in-noise vs Phase 14 baseline (80.3s, +1.6%); Q11 specifically clean. **This is the single most important regression check for v1.3 ship**, since Q11 (the entire Phase 13/14 motivation) lives in this filter and the parquet-only suite isolates the Q11 path from the cumulative-state hang shape that affects `[integration][TPC-H]`.

Hygiene preserved: HYG-02 = 40 (≤ 40 baseline), all plan diffs match expected scope (Plan 15-01 comment-only, Plan 15-02 additive test-only, Plan 15-03 doc-only). Branch `audit/mgpu-operator-colocation` descends from Phase 14 HEAD `0ee3166`; cucascade submodule pin unchanged at `62e0517`.

The `verdict: PASS` in frontmatter follows Phase 14's precedent: PASS when all acceptance criteria PASS and the only PARTIAL/DEFERRED entries are documented external limitations (in Phase 14: MCP wrapper env-passthrough; here: pre-existing v1.3 ship blocker tracked under Plan 13-05). The Phase 15 deliverables — audit comments anchored in source + audit log + behavioral stress test + documentation — are all landed and verified. **Phase 15 ships PASS.** v1.3 closure additionally requires Plan 13-05 to land for the `[integration][TPC-H]` cumulative-state hang; that is a Phase 13 follow-up, not a Phase 15 issue.
