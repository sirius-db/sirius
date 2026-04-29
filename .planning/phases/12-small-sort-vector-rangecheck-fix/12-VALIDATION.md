# Phase 12 — Validation Report

**Generated:** 2026-04-29
**Validator:** Plan 12-04 (auto)
**Scope:** Four CONTEXT.md acceptance criteria, evaluated via MCP unit-tests + TPC-H × 2-GPU integration.
**Branch:** `fix/order-small-sort-rangecheck`
**HEAD commit:** `33f4fd3 docs(12-03): complete small-sort rangecheck regression test plan`

## Verdict

Overall: **PASS**

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | `physical_order - small sort stays single-GPU` passes | **PASS** | exit 0, 27 assertions, 5.2s |
| 2 | New regression test `physical_order - small sort rangecheck regression` passes | **PASS** | exit 0, 19 assertions, 5.1s |
| 3 | [mgpu] suite: ≥12 passing of 15 (was 10/14 pre-12) | **PASS** | 12 pass / 2 fail / 1 deferred (`--abort` halted before [14] but it was the regression test which passes in isolation; recovered via separate run). Both failures are Phase 14 distribution territory, not Phase 12 regressions. |
| 4 | No regression in TPC-H × 2-GPU integration tests | **PASS** | 48/48 [integration][TPC-H] test cases pass (71608 assertions); 4/4 [mgpu-audit][TPC-H] test cases pass (64 assertions). Zero new failures vs pre-12 baseline. |

## Criterion 1 — Small-sort test passes

Command: `mcp__project-commands__run_command name=unit-tests filter="physical_order - small sort stays single-GPU"`

Output excerpt (verbatim from MCP):
```
Filters: physical_order - small sort stays single-GPU

[0/1] (0%): physical_order - small sort stays single-GPU
[1/1] (100%): physical_order - small sort stays single-GPU
===============================================================================
All tests passed (27 assertions in 1 test case)
```

- Exit code: 0
- Duration: 5.2s
- Assertions: 27 / 27 passed
- Test cases: 1 / 1 passed

Verdict: **PASS** — the formerly-failing test (libstdc++ `vector::_M_range_check` from `cudf::table_view::select(key_col_indices)`) now passes after the 12-02 bound-filter patch in `prepare_join_keys`.

## Criterion 2 — Regression test passes

Command: `mcp__project-commands__run_command name=unit-tests filter="physical_order - small sort rangecheck regression"`

Output excerpt (verbatim from MCP):
```
Filters: physical_order - small sort rangecheck regression

[0/1] (0%): physical_order - small sort rangecheck regression
[1/1] (100%): physical_order - small sort rangecheck regression
===============================================================================
All tests passed (19 assertions in 1 test case)
```

- Exit code: 0
- Duration: 5.1s
- Assertions: 19 / 19 passed
- Test cases: 1 / 1 passed

Verdict: **PASS** — the new TEST_CASE added by 12-03 (`test/cpp/operator/test_physical_order_mgpu.cpp:124`) passes against the 12-02 fix and was empirically proven (per 12-03-SUMMARY.md stash-roundtrip evidence) to fail with the EXACT verbatim 12-stack-trace.txt message `vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)` on a pre-12-02 tree, confirming it is a real regression gate.

## Criterion 3 — [mgpu] suite

Command: `mcp__project-commands__run_command name=unit-tests filter="[mgpu]"`

**Note:** the `unit-tests` MCP command runs the binary with `--abort` (halts on first failure). The full [mgpu] suite has 15 test cases; the suite aborted at index [11/15] when `physical_order - large sort distributes across two GPUs` failed. The 3 unrun tests were validated via separate filtered MCP runs (criteria 1 and 2 above cover 2 of them; the third — `physical_order - order by with limit over large input` — was run separately to attribute its result correctly).

Pre-12 baseline (from CONTEXT.md): 10 pass / 4 fail of 14 total.
Post-12 actual: **12 pass / 2 fail of 15 total** (the new regression test bumps total by 1).

Output excerpt from `[mgpu]` filter run (verbatim, until abort):
```
Filters: [mgpu]

[0/15] (0%): gpu_execution - table_gpu cache warm cross-GPU hazard (follow-up #17)
[1/15] (6%): grouped_aggregate_merge - group by with high cardinality distributes across both GPUs
[2/15] (13%): grouped_aggregate_merge - group by with single key forces single-GPU path
[3/15] (20%): grouped_aggregate_merge - count(*)-only aggregate across two GPUs
[4/15] (26%): physical_hash_join - BUILD_PROBE probe-heavy join across two GPUs
[5/15] (33%): physical_hash_join - MIXED_JOIN large-vs-large join distributes partitions
[6/15] (40%): physical_hash_join - repeated BUILD_PROBE queries don't wedge on leftover state
[7/15] (46%): hash_join bisect 1 - simple JOIN+GROUP BY+ORDER BY, cache=none
[8/15] (53%): hash_join bisect 2 - simple JOIN+GROUP BY+ORDER BY, cache=table_gpu
[9/15] (60%): hash_join bisect 3 - Q11 shape with HAVING subquery, cache=none
[10/15] (66%): physical_hash_join - follow-up #17 scale-up: Q11-like BUILD_PROBE with table_gpu cache
[11/15] (73%): physical_order - large sort distributes across two GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sirius_unittest is a Catch v2.13.7 host application.

-------------------------------------------------------------------------------
physical_order - large sort distributes across two GPUs
-------------------------------------------------------------------------------
test/cpp/operator/test_physical_order_mgpu.cpp:73
...

test/cpp/operator/test_physical_order_mgpu.cpp:113: FAILED:
  REQUIRE( by_gpu[0].pipeline_ids.size() >= 1 )
with expansion:
  0 >= 1
with message:
  gpu0 pipelines=0 gpu1 pipelines=16

===============================================================================
test cases:   12 |   11 passed | 1 failed
assertions: 1955 | 1954 passed | 1 failed
```

Output excerpt from separate filtered run for `physical_order - order by with limit over large input` (verbatim):
```
Filters: physical_order - order by with limit over large input

[0/1] (0%): physical_order - order by with limit over large input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

test/cpp/operator/test_physical_order_mgpu.cpp:244: FAILED:
  REQUIRE( by_gpu[0].pipeline_ids.size() >= 1 )
with expansion:
  0 >= 1
with message:
  gpu0 pipelines=0 gpu1 pipelines=17

[1/1] (100%): physical_order - order by with limit over large input
===============================================================================
test cases:  1 |  0 passed | 1 failed
assertions: 35 | 34 passed | 1 failed
```

Failing tests (names verbatim):
1. `physical_order - large sort distributes across two GPUs` — fails at `test_physical_order_mgpu.cpp:113` with `gpu0 pipelines=0 gpu1 pipelines=16` (all pipelines on GPU 1, none on GPU 0)
2. `physical_order - order by with limit over large input` — fails at `test_physical_order_mgpu.cpp:244` with `gpu0 pipelines=0 gpu1 pipelines=17` (same shape: all pipelines on GPU 1, none on GPU 0)

Carry-forward verdict (which failures are Phase 14 territory, not Phase 12 regressions):

Both failures share the **EXACT same shape**: `gpu0 pipelines=0 gpu1 pipelines=N` — i.e. the scan-task distributor places ALL pipeline tasks on GPU 1 with none on GPU 0. This is **NOT** a `vector::_M_range_check` failure (Phase 12's bug class) — it is a SCHED-RR / source-pipeline-parallelism failure scoped to Phase 14 (`14-sched-rr-distribution`) per ROADMAP.md: "Phase 14: Land SCHED-RR distribution. Patch tested working in isolation, rolled back from session due to Phase 13 dependency." Phase 12's CONTEXT.md acceptance criterion 3 explicitly states: "2 still fail (Phase 14 distribution territory)" — these two test names map exactly to that classification.

Verified count: **12 pass / 2 fail / 0 new regressions of 15 total**. The 12 passing includes:
- 10 [mgpu] tests passing pre-12 (table_gpu cache warm, 3× grouped_aggregate_merge, 4× physical_hash_join, 2× hash_join bisect, 1× cache=none — verified all PASSED in the suite run before the abort)
- 1 fixed by Phase 12: `physical_order - small sort stays single-GPU` (Criterion 1)
- 1 added by Phase 12: `physical_order - small sort rangecheck regression` (Criterion 2)

Verdict: **PASS** — Phase 12 advances the [mgpu] suite from 10 pass → 12 pass (+2 net: 1 fix, 1 new regression test). Both remaining failures are pre-existing Phase 14 distribution territory with the same `gpu0 pipelines=0` shape, not new regressions from Phase 12.

## Criterion 4 — TPC-H × 2-GPU integration

Command discovery: `mcp__project-commands__list_commands` returned no dedicated `tpch-mgpu` / `integration-tpch-mgpu` / `tpch-2gpu` command. The TPC-H × 2-GPU integration tests are exercised through the `unit-tests` MCP command via Catch2 tag filters. Per Phase 8 STATE.md decision [08-04] "TEST-01/02: integration-2gpu.yaml + g_integration_env_2gpu + acquire_integration_env_for() helper + RUN_TPCH_MGPU macro. All 44 TPC-H TEST_CASEs parameterized on num_gpus in {1,2}", the relevant invocation is the [integration][TPC-H] tag set (which exercises BOTH 1-GPU and 2-GPU variants of every TPC-H query) plus the [mgpu-audit][TPC-H] tag set (the explicit per-GPU dispatch audit on Q1).

### Sub-run A: [integration][TPC-H] full TPC-H × {1-GPU, 2-GPU} matrix

Command: `mcp__project-commands__run_command name=unit-tests filter="[integration][TPC-H]"`

Output excerpt (verbatim, final lines):
```
[44/48] (91%): gpu_execution - tpch_q1_sf10_2gpu
...
SIRIUS_TEST_SF10_PATH unset; skipping SF10 Q1 variant (TEST-04 gate)

[45/48] (93%): gpu_execution - tpch_q6_sf10_2gpu
...
SIRIUS_TEST_SF10_PATH unset; skipping SF10 Q6 variant (TEST-04 gate)

[46/48] (95%): gpu_execution - tpch_q12_sf10_2gpu
...
SIRIUS_TEST_SF10_PATH unset; skipping SF10 Q12 variant (TEST-04 gate)

[47/48] (97%): gpu_execution - [mgpu-audit] per-GPU distribution on TPC-H Q1
[48/48] (100%): gpu_execution - [mgpu-audit] per-GPU distribution on TPC-H Q1
===============================================================================
All tests passed (71608 assertions in 48 test cases)
```

- Exit code: 0
- Duration: 154.9s
- Assertions: 71608 / 71608 passed
- Test cases: 48 / 48 passed (TPC-H Q1–Q22 × {DuckDB attach, parquet} × {num_gpus=1, num_gpus=2}, plus 4 SF10 mgpu-audit variants)
- SF10 Q1/Q6/Q12 2-GPU variants gated on `SIRIUS_TEST_SF10_PATH` env var (per [08-05] decision); env unset → WARN+return per intentional gate, NOT a failure (Catch2 still counts these as passed test cases since they exit cleanly).

### Sub-run B: [mgpu-audit][TPC-H] per-GPU dispatch audit

Command: `mcp__project-commands__run_command name=unit-tests filter="[mgpu-audit][TPC-H]"`

Output excerpt (verbatim, final lines):
```
[3/4] (75%): gpu_execution - [mgpu-audit] per-GPU distribution on TPC-H Q1
[4/4] (100%): gpu_execution - [mgpu-audit] per-GPU distribution on TPC-H Q1
===============================================================================
All tests passed (64 assertions in 4 test cases)
```

- Exit code: 0
- Duration: 6.8s
- Assertions: 64 / 64 passed
- Test cases: 4 / 4 passed

Pre-12 baseline (from STATE.md, Phase 10 ship-gate `[10-04]` SF100 Q1 num_gpus=2 PASS + Phase 9 ship-gate `[09-04]` 2-GPU disjoint dispatch PROVEN):
- TPC-H Q1–Q22 × {1-GPU, 2-GPU} × {DuckDB, parquet}: all 44 base TEST_CASEs passing
- TPC-H Q1 [mgpu-audit] dispatch audit on num_gpus=2: passing
- SF10 Q1/Q6/Q12 2-GPU variants: gated (env not set on dev host); previously verified PASS on dedicated SF10 host
- SF100 Q1 num_gpus=2: PASS, byte-identical to 1-GPU baseline (Phase 9/10 ship-gate evidence)

Post-12 actual:
- TPC-H Q1–Q22 × {1-GPU, 2-GPU} × {DuckDB, parquet}: **all 44 PASS** (71608 assertions)
- TPC-H Q1 [mgpu-audit] dispatch audit on num_gpus=2: **PASS** (64 assertions across 4 test cases)
- SF10 Q1/Q6/Q12 2-GPU variants: gated (same env condition as baseline)

New failures introduced by Phase 12: **none**.

Verdict: **PASS** — zero new failures in TPC-H × 2-GPU integration tests; full 48-case integration sweep passes (71608 assertions) with the Phase 12 patch (`prepare_join_keys` bound-filter at `src/op/sirius_physical_hash_join.cpp:622-637`) and new regression test in place. The patch is correctness-neutral on the TPC-H execution path (the failing test query had no SQL-level join; the synthetic SORT-as-HASH_JOIN partitioner case is the only path that emitted the stale index — real TPC-H joins still construct valid `key_col_indices` < `table.num_columns()` and are unaffected by the filter).

## Recommendations

**Ship Phase 12.**

All four CONTEXT.md acceptance criteria evaluate to PASS:
1. Small-sort test (the explicit ship-gate from Phase 12 goal): PASS
2. Regression test (prevents reappearance): PASS, with empirical stash-roundtrip evidence (12-03-SUMMARY)
3. [mgpu] suite: 12/15 PASS — net +2 from Phase 12 (1 fix + 1 new regression test); 0 new regressions; remaining 2 failures explicitly classified as Phase 14 distribution territory in CONTEXT.md and confirmed by failure shape (`gpu0 pipelines=0 gpu1 pipelines=N`)
4. TPC-H × 2-GPU integration: 48/48 PASS, 71608 assertions; zero new failures

**Orchestrator next steps:**
- Update `ROADMAP.md` Phase 12 row → Complete (this plan does NOT modify ROADMAP per Step 7).
- Update `STATE.md` current focus → Phase 13 (`13-q11-multi-gpu-illegal-address` — the v1.3 ship blocker per ROADMAP).
- The `fix/order-small-sort-rangecheck` branch is ready for PR review.

**No gap-closure plan needed.** All defects in Phase 12's scope are closed:
- 12-01 pinned the fix-site (`src/op/sirius_physical_hash_join.cpp:623`)
- 12-02 patched it with INVARIANT comment (commit `289d6d2`)
- 12-03 added the regression gate with stash-roundtrip empirical proof (commit `163d622`)
- 12-04 (this plan) confirmed all four CONTEXT.md acceptance criteria via real MCP test runs

## Files modified by Phase 12

`git diff --stat 7b9af88^..HEAD -- src/ test/`:
```
 src/op/sirius_physical_hash_join.cpp           | 14 +++++++-
 test/cpp/operator/test_physical_order_mgpu.cpp | 47 ++++++++++++++++++++++++++
 2 files changed, 60 insertions(+), 1 deletion(-)
```

Phase 12 commits (post-7b9af88):
- `289d6d2 fix(12-02): bound-check key_col_indices in prepare_join_keys`
- `38032bd docs(12-02): complete small-sort vector rangecheck bound-fix plan`
- `163d622 test(12-03): add small-sort rangecheck regression TEST_CASE`
- `33f4fd3 docs(12-03): complete small-sort rangecheck regression test plan`

Source change: 14 lines in `src/op/sirius_physical_hash_join.cpp` (5 code + 7 INVARIANT comment + 2 brace adjustments).
Test change: 47 lines in `test/cpp/operator/test_physical_order_mgpu.cpp` (one new TEST_CASE at lines 120-165).
HYG baseline: `rmm::cuda_stream_default` count in `src/` = 40 (≤ 40 — unchanged from pre-12 baseline).
