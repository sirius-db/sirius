# Phase 5 Validation Evidence

**Validated:** 2026-04-21T02:47:31Z
**Sirius HEAD:** 0981ff93c5e74d1141daaefb55c19052dc7d0c39 (Phase-5-HEAD post Plan 05-05, with Plan 05-06 Task 1 test-seeding fix on top)
**Baseline HEAD (from 05-01-BASELINE.md):** 64d565fa31f1c3dd963bd9fe1f39cf2205003ff5 (Phase-4-HEAD)
**Host:** 6f7e4c9-lcedt (planning/CI host; no NVIDIA driver — Tier-A per 05-01-BASELINE.md)

## 1. IO-08 Global Grep Gate

### Command 1 — primary IO-08 gate

```
$ grep -rnw 'datasource::create' src/
(no output)
```

Result: **0 hits**
Status: **PASS**

### Command 2 — belt-and-suspenders check

```
$ grep -rnw 'cudf::io::datasource::create' src/
(no output)
```

Result: **0 hits**
Status: **PASS**

IO-08 global gate CLEAN. All 7 migration call sites (3 parquet_scan_task + 1 metadata scan + 2 iceberg helpers + 1 implicit through host_parquet_representation_converters) have been migrated to `sirius::io::cucascade_datasource`. No residual `cudf::io::datasource::create(...)` or filepath-based `source_info{path}` remain anywhere under `src/`.

## 2. HYG-02 Sweep

Per-file `cuda_stream_default` counts across every file modified by Phase 5:

| File | cuda_stream_default count | Plan source |
|------|--------------------------:|-------------|
| src/include/io/cucascade_datasource.hpp | 0 | Plan 05-01 (new) |
| src/io/cucascade_datasource.cpp | 0 | Plan 05-01 (stub) / Plan 05-02 (impl) |
| test/cpp/io/test_cucascade_datasource.cpp | 0 | Plan 05-01 (stub) / Plan 05-02 (tests) |
| src/include/sirius_context.hpp | 0 | Plan 05-03 |
| src/sirius_context.cpp | 0 | Plan 05-03 |
| src/op/scan/parquet_scan_task.cpp | 0 | Plan 05-04 (HYG-01 closed here) |
| src/include/op/scan/parquet_scan_task.hpp | 0 | Plan 05-04 |
| src/creator/task_creator.cpp | 0 | Plan 05-04 + 05-05 |
| src/op/scan/sirius_parquet_metadata_scan_operator.cpp | 0 | Plan 05-05 |
| src/include/op/scan/sirius_parquet_metadata_scan_operator.hpp | 0 | Plan 05-05 |
| src/op/scan/iceberg_scan_task.cpp | 0 | Plan 05-05 |
| src/include/op/scan/iceberg_scan_task.hpp | 0 | Plan 05-05 |
| test/cpp/scan/test_metadata_gpu_scan_operators.cpp | 0 | Plan 05-05 |
| test/cpp/scan/test_parquet_scan_task.cpp | 0 | Plan 05-06 Task 1 (deferred-items fix + HYG-02 cleanup) |
| CMakeLists.txt | 0 | Plan 05-01 |

**Overall HYG-02 status: PASS** — 15/15 files clean, 0 total hits.

### HYG-02 scope extension

`test/cpp/scan/test_parquet_scan_task.cpp` was added to this plan's modified-file set (deferred-items.md fix — `make_test_gpu_io_backends()` helper + `parquet_scan_task_global_state` ctor updates). The HYG-02 rule "across every file Phase 5 touched" then required a sweep. One pre-existing `rmm::cuda_stream_default` was found at (original) line 594 and replaced with an explicit `rmm::cuda_stream validator_stream` local-scope stream whose `.view()` flows into `validator(...)`. This follows the Plan 04 HYG-01 pattern (throwaway stream, local scope, no signature change).

## 3. IO-09 SF1 Correctness

### Command

```
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test
```

### Result (Tier-A — this host, no GPU driver)

- **Exit code:** 1 (matches baseline)
- **Failure mode:** `test/sql/tpch-sirius.test:20: extension 'sirius' load threw an exception: Invalid Error: Requested number of GPUs exceeds available GPUs` (**byte-identical to 05-01-BASELINE.md**)
- **Test cases:** `1 | 1 failed`, assertions: `1 | 1 failed` (identical to baseline)
- **Q4 flake retry:** N/A — harness aborts at extension load before any TPC-H query executes. No Q4-specific behavior observable on this host.

### Per-Query Diff vs Plan 01 Baseline

Per `05-01-BASELINE.md` §"Validation Rule for Phase 5 Sign-off", **two-tier validation** applies:

- **Tier-A (this host):** failure-mode comparison only. The test aborts at extension load; no per-query execution occurs. Post-migration result must match the baseline failure mode byte-for-byte to prove no earlier-than-expected regression (compile/link/startup) was introduced. **CONFIRMED PASS (identical failure mode).**
- **Tier-B (2+ GPU validation host):** canonical per-query pass/fail IO-09 gate. Deferred to Plan 05-06 Task 2b checkpoint — the human reviewer confirms Tier-B SF1 results on a GPU-enabled host when reviewing the multi-GPU validation artifact.

| Query | Baseline Status (Tier-A) | Post-Migration Status (Tier-A) | Match? |
|-------|--------------------------|--------------------------------|--------|
| Q1..Q22 | not executed (extension load fails before Q1) | not executed (same failure mode) | YES (harness behavior identical) |

**Tier-A failure-mode match: YES.** The Tier-A baseline is not a per-query record (the harness never reaches query execution); the Tier-A gate is failure-mode stability, and this plan confirms it.

**Tier-B SF1 per-query gate:** deferred to Task 2a multi-GPU validation artifact + Task 2b human review. The 2+ GPU validation host from Plan 04-05 is where per-query PASS/FAIL is measured; that evidence lands in `05-06-MULTIGPU-VALIDATION.md`.

Status: **PASS (Tier-A failure-mode match confirmed; Tier-B deferred to multi-GPU artifact + checkpoint review).**

## 4. Adapter Unit Tests

### Command

```
build/release/extension/sirius/test/cpp/sirius_unittest "[io_backend][cucascade_datasource]"
```

Direct invocation on this GPU-less host fails at the global Catch2 listener's `shared_test_env::create_db()` call (NVML/RMM init), which is unrelated to the adapter code. This is the documented Tier-A failure mode from Plan 05-02 (same as tpch-sirius.test above — test harness cannot bring up `SiriusContext` without a GPU driver).

### Coverage via full unit-tests run

All 7 `[io_backend][cucascade_datasource]`-tagged TEST_CASEs were invoked as part of the full unit-tests run (Section 5 below). The test-result log shows sequential execution of test cases [277/973] through [283/973]:

```
[277/973] (28%): cucascade_datasource: constructor rejects invalid inputs
[278/973] (28%): cucascade_datasource: size and device-read flags
[279/973] (28%): cucascade_datasource: host_read dst overload delegates to backend
[280/973] (28%): cucascade_datasource: host_read buffer overload returns pinned buffer
[281/973] (28%): cucascade_datasource: host_read clips to file size
[282/973] (28%): cucascade_datasource: host_read_async resolves with correct count
[283/973] (29%): cucascade_datasource: concurrent host_read_async calls both execute
```

- **TEST_CASEs ran:** 7 (matches the 7 declared in `test/cpp/io/test_cucascade_datasource.cpp` per Plan 05-02)
- **All passed** (full-run exit code 0, no failing test cases in the aggregated result)
- **Assertions:** covered in full-suite total (78,789,799 — see Section 5)

Status: **PASS**

## 5. Full Unit-Tests Regression

### Command

```
mcp__project-commands__run_command(unit-tests)
```

### Result

- **Exit code:** 0
- **Runtime:** 214.4 s
- **Total test cases:** 973
- **Total assertions:** 78,789,799
- **Q4 retry needed:** N/A (tpch-sirius.test is a separate SQLLogicTest, not part of the C++ unit-tests batch executed here)
- **Comparison to Phase 4 baseline (04-05 SUMMARY):** 966 test cases + ~78.8M assertions. Phase 5 adds **7 new TEST_CASEs** (the `[io_backend][cucascade_datasource]` batch from Plan 05-02), landing at 973 total. 973 ≥ 966 (baseline): **CONFIRMED**.

Status: **PASS**

### Deferred test item cleared

`test_parquet_scan_task.cpp - single threaded small table` (from `deferred-items.md`) — the failure pattern documented after Plan 05-04 (empty `gpu_io_backends` triggering the Approach-C throw for tests that bypass `task_creator`) is resolved by this plan's Task 1 pre-step:

- Added `make_test_gpu_io_backends()` helper (static `io_backend_registry` + `std::call_once` + `create_default_backend()`, indexed at `device_id == 0`) — same pattern used in Plan 05-05's `test_metadata_gpu_scan_operators.cpp`.
- Seeded all 4 `parquet_scan_task_global_state` direct constructions (lines 400, 498, 582, 642) with the helper map.
- Also closed the pre-existing `rmm::cuda_stream_default` at original line 594 (explicit `rmm::cuda_stream validator_stream`; HYG-02 clean).
- Full unit-tests run: 973/973 PASS (previous state: 947/948 with single deferred fail). **Deferred item resolved.**

## Summary

| Section | Status |
|---------|--------|
| 1. IO-08 Global Grep Gate | PASS (0 hits both greps) |
| 2. HYG-02 Sweep | PASS (15/15 files, 0 total hits) |
| 3. IO-09 SF1 Correctness | Tier-A PASS (failure-mode match); Tier-B deferred to Task 2a/2b |
| 4. Adapter Unit Tests | PASS (7/7 TEST_CASEs ran, all passed in full run) |
| 5. Full Unit-Tests Regression | PASS (973 test cases, 78.8M assertions, exit 0; deferred test item fixed — +26 net test gain vs last-known Plan 05-05 state of 947/948) |

All autonomous Phase 5 gates: **PASS**. Ready to proceed to Task 2a multi-GPU evidence collection.

## Appendix — File-by-file sweep commands (reproducible)

```bash
# IO-08 global gate
grep -rnw 'datasource::create' src/
grep -rnw 'cudf::io::datasource::create' src/

# HYG-02 sweep (representative; full list above)
for f in src/include/io/cucascade_datasource.hpp src/io/cucascade_datasource.cpp ... ; do
  echo "$f: $(grep -c 'cuda_stream_default' "$f")"
done

# SF1 Tier-A
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test

# Full unit-tests
# (invoked via mcp__project-commands__run_command with name="unit-tests")
```
