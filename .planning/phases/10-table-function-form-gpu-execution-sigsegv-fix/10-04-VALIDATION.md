---
phase: 10-table-function-form-gpu-execution-sigsegv-fix
plan: 10-04
type: validation
recorded: 2026-04-27T20:35:48Z
host: 6f7e4c9-lcedt (GPU 0: NVIDIA RTX 6000 Ada Generation; GPU 1: NVIDIA RTX 6000 Ada Generation; 2 × 49 GB; driver 595.58.03; CUDA 13.2)
branch: feature/single-node-multi-gpu2
head_commit: b168ce6 docs(10-03): complete fix plan — stream use-after-destroy SIGSEGV closed
verdict: PARTIAL
ship_gate: closed
---

# Phase 10 Plan 10-04 — Autonomous Ship-Gate Validation on 2-GPU Hardware

Autonomous validation run (per `feedback_mcp_tests_scope.md`). The agent executed the build, unit-tests, SF100 ship-gate, and VALIDATION authoring without user delegation.

**Net outcome:** SF100 Q1 num_gpus=2 ship-gate PASSES with byte-identical result vs 1-GPU baseline, cross-GPU batch_id intersection=0 across 71 scan batches (GPU0=42, GPU1=29), 5.70s wall-clock, zero fallbacks, zero cudaErrorInvalidValue, zero SIGSEGV. Both CRIT-2 target tests that failed in Phase 9 (`gpu_execution - filter equality parquet`, `gpu_execution - tpch_q1_sf10_2gpu`) are GREEN after the Phase 10-03 stream-use-after-destroy fix. The only remaining unit-test failure is the pre-existing `[mgpu-audit]` SIGSEGV at `test_gpu_execution_tpch_mgpu_audit.cpp:200`, which was confirmed pre-existing on the base branch before the fix (documented in 10-03-FIX.md). This one failure prevents `All tests passed` and results in a PARTIAL verdict per plan logic. The CRIT-2 Phase 10 fix objective (close the two previously-failing tests) is COMPLETE.

## ROADMAP Criterion-by-Criterion Verdict Summary

| # | Criterion | Verdict |
| - | --------- | ------- |
| 1 | SF100 TPC-H Q1 num_gpus=2 correct vs num_gpus=1 baseline, no SIGSEGV/cudaErrorInvalidValue/fallback | **PASS** |
| 2 | MCP unit-tests exits 0 with 88 SF1 variants + SF10 Q1/Q6/Q12 green — INCLUDING the previously-failing TABLE_FUNCTION tests | **PARTIAL** (both target tests GREEN; suite has 1 pre-existing [mgpu-audit] SIGSEGV — not a Phase 10 regression) |
| 4 | AUDIT disjointedness REQUIRE (Plan 09-03 carryover) | **PASS_CARRYOVER** (set_intersection REQUIRE passes in tpch_q1_sf10_2gpu; GPU0=4, GPU1=3 unique batch_ids, intersect=0) |
| 6 | SF100 [mgpu-audit] distributes scan_batches across both GPUs + wall-clock captured | **PASS** (71 scan batches distributed as GPU0=42, GPU1=29, disjoint; wall-clock 5.70s; 219 [mgpu-audit] entries) |

**Phase-level verdict:** PARTIAL — CRIT-1, CRIT-6, and the Phase 10 CRIT-2 target tests all PASS. The Phase 10 fix (stream use-after-destroy) is demonstrably closed: `filter equality parquet` and `tpch_q1_sf10_2gpu` both exit 0. One pre-existing `[mgpu-audit]` SIGSEGV prevents the suite from reaching `All tests passed`; this failure is explicitly documented as pre-existing in `10-03-FIX.md` and is not attributable to any Phase 10 change.

## Commands Run

```
# Task 1 — Pre-flight
git branch --show-current  → feature/single-node-multi-gpu2
git diff --quiet HEAD -- test/cpp/integration/integration.yaml     → exit 0 (CLEAN)
git diff --quiet HEAD -- test/cpp/integration/integration-2gpu.yaml → exit 0 (CLEAN)
grep '    num_gpus:' integration.yaml   → num_gpus: 1
grep '    num_gpus:' integration-2gpu.yaml → num_gpus: 2
nvidia-smi -L  → GPU 0 + GPU 1 both NVIDIA RTX 6000 Ada 49GB
ls /datasets/tpch_parquet_sf100/lineitem.parquet  → 22791208139 bytes

# Task 2 — MCP build + unit-tests
mcp__project-commands__run_command(name="build")          # exit=0 (49 targets rebuilt)
mcp__project-commands__run_command(name="unit-tests", filter="")  # exit=-1 (666 tests, 1 SIGSEGV: [mgpu-audit])

# Direct binary runs (Rule 3 auto-fix: MCP wrapper doesn't pass SIRIUS_TEST_SF10_PATH):
SIRIUS_TEST_SF10_PATH=/datasets/tpch_parquet_sf10 \
  ./build/release/extension/sirius/test/cpp/sirius_unittest \
  'gpu_execution - filter equality parquet'
# exit=0, All tests passed (31 assertions in 1 test case)

SIRIUS_TEST_SF10_PATH=/datasets/tpch_parquet_sf10 \
  SIRIUS_LOG_DIR=/tmp/claude/sirius-ph10/sf10-q1-run \
  SIRIUS_LOG_LEVEL=info \
  ./build/release/extension/sirius/test/cpp/sirius_unittest \
  'gpu_execution - tpch_q1_sf10_2gpu'
# exit=0, All tests passed (99 assertions in 1 test case)

SIRIUS_TEST_SF10_PATH=/datasets/tpch_parquet_sf10 \
  SIRIUS_LOG_DIR=/tmp/claude/sirius-ph10/audit-run \
  ./build/release/extension/sirius/test/cpp/sirius_unittest \
  '[mgpu-audit]'
# exit=139 (SIGSEGV in [mgpu-audit] per-GPU distribution test, pre-existing)
# 3/4 [mgpu-audit] tests PASS; 1 FAIL = same pre-existing failure

# Task 3 — SF100 CLI runs
git show HEAD:test/cpp/integration/integration.yaml > /tmp/claude/sirius-ph10/sirius-ph10-1gpu.yaml
# → num_gpus: 1 confirmed

# 2-GPU (ship-gate, via committed integration-2gpu.yaml):
SIRIUS_CONFIG_FILE=test/cpp/integration/integration-2gpu.yaml \
  SIRIUS_LOG_DIR=/tmp/claude/sirius-ph10/sf100-2gpu-logs \
  SIRIUS_LOG_LEVEL=info \
  /usr/bin/time -v -o /tmp/claude/sirius-ph10/sf100-2gpu.time \
  build/release/duckdb < sf100-q1-noinit.sql \
  > /tmp/claude/sirius-ph10/sf100-2gpu.csv 2>/tmp/claude/sirius-ph10/sf100-2gpu.stderr
# exit=0, wall-clock 0:05.70

# 1-GPU baseline:
SIRIUS_CONFIG_FILE=/tmp/claude/sirius-ph10/sirius-ph10-1gpu.yaml \
  SIRIUS_LOG_DIR=/tmp/claude/sirius-ph10/sf100-1gpu-logs \
  SIRIUS_LOG_LEVEL=info \
  /usr/bin/time -v -o /tmp/claude/sirius-ph10/sf100-1gpu.time \
  build/release/duckdb < sf100-q1-noinit.sql \
  > /tmp/claude/sirius-ph10/sf100-1gpu.csv 2>/tmp/claude/sirius-ph10/sf100-1gpu.stderr
# exit=0, wall-clock 0:05.45

diff /tmp/claude/sirius-ph10/sf100-1gpu.csv /tmp/claude/sirius-ph10/sf100-2gpu.csv
# diff_exit=0 (byte-identical)
```

## Transcript Excerpts

### MCP build (Task 2)

```
cd duckdb && cmake --build --preset release
ninja: Jobserver mode detected: -j24 --jobserver-auth=fifo:/tmp/GMfifo3771768
[1/49] Building CXX object ... sirius_extension ... task_creator.cpp.o
[5/49] Building CXX object ... sirius_physical_parquet_scan.cpp.o
[43/49] Linking CXX shared library extension/sirius/sirius.duckdb_extension
[46/49] Linking CXX executable extension/sirius/test/cpp/sirius_unittest
[48/49] Linking CXX executable duckdb
[49/49] Linking CXX shared library src/libduckdb.so
cd duckdb && cmake --build --preset release --target unittest
ninja: no work to do.

Exit code: 0
```

### Unit-tests (Task 2)

Two approaches were used because the MCP wrapper does not pass the agent's shell env to the child process (same Rule 3 auto-fix as Phase 9).

```
# RUN 1 — MCP full suite (no SIRIUS_TEST_SF10_PATH in child env):
test cases:      666 |      665 passed | 1 failed
assertions: 69186370 | 69186369 passed | 1 failed
Failing test: "gpu_execution - [mgpu-audit] per-GPU distribution on TPC-H Q1"
  File: test_gpu_execution_tpch_mgpu_audit.cpp:200
  Error: SIGSEGV - Segmentation violation signal
  (SF10 smoke tests skipped via WARN+return: "SIRIUS_TEST_SF10_PATH unset; skipping")

# RUN 2 — Target test 1 (direct binary, SIRIUS_TEST_SF10_PATH set):
./sirius_unittest 'gpu_execution - filter equality parquet'
All tests passed (31 assertions in 1 test case)
Exit code: 0  ← PREVIOUSLY SIGSEGV at assertion 19/31; NOW GREEN

# RUN 3 — Target test 2 (direct binary, SIRIUS_TEST_SF10_PATH set):
./sirius_unittest 'gpu_execution - tpch_q1_sf10_2gpu'
All tests passed (99 assertions in 1 test case)
Exit code: 0  ← PREVIOUSLY skipping SF10 gate; NOW runs full SF10 scale (99 assertions)

# RUN 4 — [mgpu-audit] tag isolated:
./sirius_unittest '[mgpu-audit]'
test cases: 4 | 3 passed | 1 failed
assertions: 233 | 232 passed | 1 failed
Failing: "gpu_execution - [mgpu-audit] per-GPU distribution on TPC-H Q1"
  SIGSEGV at test_gpu_execution_tpch_mgpu_audit.cpp:200
  Classification: PRE-EXISTING — confirmed on base branch before fix (10-03-FIX.md)
```

#### Q4 parquet retry (v1.1 precedent flake policy)

Not triggered — no Q4 parquet failure in any run. `Q4_FLAKE_RECOVERED`=false, `Q4_FLAKE_UNRECOVERED`=false.

### AUDIT TEST_CASE evidence (Task 2)

From `tpch_q1_sf10_2gpu` run (the SF10 AUDIT-exercising test):

```
# Per-GPU batch_id distribution from tpch_q1_sf10_2gpu run logs:
GPU 0: 4 unique batch_ids
GPU 1: 3 unique batch_ids
Cross-GPU intersection: 0
Total [mgpu-audit] entries: 108

# Sample [mgpu-audit] entries:
[2026-04-27 14:54:13.741] [mgpu-audit] scan_batch assigned to GPU 0 batch_id=0
[2026-04-27 14:54:13.741] [mgpu-audit] scan_batch assigned to GPU 0 batch_id=1
[2026-04-27 14:54:13.761] [mgpu-audit] scan_batch assigned to GPU 1 batch_id=2
[2026-04-27 14:54:13.761] [mgpu-audit] pipeline_task dispatched to GPU 0 task_id=7
[2026-04-27 14:54:13.763] [mgpu-audit] scan_batch assigned to GPU 1 batch_id=3
[2026-04-27 14:54:13.763] [mgpu-audit] pipeline_task dispatched to GPU 1 task_id=8
[2026-04-27 14:54:13.775] [mgpu-audit] scan_batch assigned to GPU 1 batch_id=4
[2026-04-27 14:54:13.775] [mgpu-audit] pipeline_task dispatched to GPU 0 task_id=9
[2026-04-27 14:54:13.782] [mgpu-audit] scan_batch assigned to GPU 0 batch_id=5
[2026-04-27 14:54:13.782] [mgpu-audit] pipeline_task dispatched to GPU 1 task_id=10
... (108 total entries; GPU0 and GPU1 coverage confirmed)
```

Plan 09-03 AUDIT disjointedness REQUIRE passes: `set_intersection(GPU0_scan_ids, GPU1_scan_ids) = ∅`.

### Runtime probe — preferred_device_id plumbing (Task 2)

```
compute_task entry with preferred_device_id=-1: 0
     16 preferred_device_id=0
     12 preferred_device_id=1
distinct positive preferred_device_id values: 2
```

**Plan 09-01 status: LIVE** — zero `-1` sentinels at `parquet_scan_task::compute_task` entry; both `preferred_device_id=0` AND `preferred_device_id=1` observed at SF10 scale.

### SF100 Q1 num_gpus=2 (Task 3)

- Wall-clock: 0:05.70
- Exit code: 0
- Row count (data rows): 4
- Stderr: empty (no errors)

CSV stdout:

```
l_returnflag,l_linestatus,sum_qty,sum_base_price,sum_disc_price,sum_charge,avg_qty,avg_price,avg_disc,count_order
A,F,3775127758.00,5660776097194.45,5377736398183.9374,5592847429515.927026,25.499370423275426,38236.1169843049,0.050002243530929025,148047881
N,F,98553062.00,147771098385.98,140384965965.0348,145999793032.775829,25.501556956882876,38237.19938880451,0.04998528433805397,3864590
N,O,7436302976.00,11150725681373.59,10593195308234.8523,11016932248183.655467,25.500009404374193,38237.227646360945,0.04999791831562552,291619617
R,F,3775724970.00,5661603032745.34,5378513563915.4097,5593662252666.916161,25.50006628406532,38236.697258452965,0.05000130433965412,148067261
```

### SF100 Q1 num_gpus=1 baseline (Task 3)

- Wall-clock: 0:05.45
- Exit code: 0
- Stderr: empty (no errors)

CSV stdout:

```
l_returnflag,l_linestatus,sum_qty,sum_base_price,sum_disc_price,sum_charge,avg_qty,avg_price,avg_disc,count_order
A,F,3775127758.00,5660776097194.45,5377736398183.9374,5592847429515.927026,25.499370423275426,38236.1169843049,0.050002243530929025,148047881
N,F,98553062.00,147771098385.98,140384965965.0348,145999793032.775829,25.501556956882876,38237.19938880451,0.04998528433805397,3864590
N,O,7436302976.00,11150725681373.59,10593195308234.8523,11016932248183.655467,25.500009404374193,38237.227646360945,0.04999791831562552,291619617
R,F,3775724970.00,5661603032745.34,5378513563915.4097,5593662252666.916161,25.50006628406532,38236.697258452965,0.05000130433965412,148067261
```

### SF100 CSV result-diff (Task 3)

```
diff_exit=0
```

(Output of `diff` is empty; byte-identical.)

### SF100 [mgpu-audit] excerpt (Task 3)

```
[2026-04-27 ...] [mgpu-audit] scan_batch assigned to GPU 0 batch_id=0 (available: 20346961408 bytes)
[2026-04-27 ...] [mgpu-audit] scan_batch assigned to GPU 0 batch_id=1 (available: 20346961408 bytes)
[2026-04-27 ...] [mgpu-audit] scan_batch assigned to GPU 1 batch_id=2 (available: 20346961408 bytes)
[2026-04-27 ...] [mgpu-audit] pipeline_task dispatched to GPU 0 task_id=71
[2026-04-27 ...] [mgpu-audit] pipeline_task dispatched to GPU 1 task_id=72
[2026-04-27 ...] [mgpu-audit] scan_batch assigned to GPU 1 batch_id=3 (available: 20045302047 bytes)
[2026-04-27 ...] [mgpu-audit] scan_batch assigned to GPU 1 batch_id=4 (available: 20045302047 bytes)
[2026-04-27 ...] [mgpu-audit] pipeline_task dispatched to GPU 0 task_id=73
[2026-04-27 ...] [mgpu-audit] scan_batch assigned to GPU 0 batch_id=5 (available: 19741035475 bytes)
[2026-04-27 ...] [mgpu-audit] pipeline_task dispatched to GPU 1 task_id=74
... (219 total entries; per-GPU breakdown in next section)
```

#### Per-GPU unique batch_id distribution (SF100 Q1 num_gpus=2)

| GPU | Unique batch_ids | Cross-GPU intersection |
|-----|-----------------|------------------------|
| 0   | 42              | 0                      |
| 1   | 29              | 0                      |

**Total batches: 71 (42 + 29); intersection 0; zero overlap.** Plan 09-02 batch→GPU affinity map is live at SF100 scale post Phase 10 fix.

## Per-Criterion Closure

| ROADMAP Criterion | Evidence | Status |
|-------------------|----------|--------|
| CRIT-1 — SF100 Q1 num_gpus=2 correct, no cudaErrorInvalidValue, no SIGSEGV, no fallback | wall-clock 2-GPU=0:05.70, 1-GPU=0:05.45, CSV diff empty=1, SF100_CUDA_ERR=0, SF100_SIGSEGV=0, SF100_FALLBACK=0, rows=4 | **PASS** |
| CRIT-2 — MCP unit-tests exits 0 with target tests GREEN (filter equality parquet + tpch_q1_sf10_2gpu) | Both target tests: exit 0, PASS. filter equality parquet=31 assertions; tpch_q1_sf10_2gpu=99 assertions. Pre-existing [mgpu-audit] SIGSEGV prevents full-suite exit 0. | **PARTIAL** (Phase 10 fix objective closed; one pre-existing failure remains) |
| CRIT-4 — AUDIT disjointedness REQUIRE (Plan 09-03 carryover) | tpch_q1_sf10_2gpu: GPU0=4 unique batch_ids, GPU1=3, cross-GPU intersection=0. set_intersection REQUIRE PASSES. SF100: GPU0=42, GPU1=29, intersection=0. | **PASS_CARRYOVER** |
| CRIT-6 — SF100 [mgpu-audit] scan_batch distribution + wall-clock | GPU0=42, GPU1=29, intersection=0, wall-clock=0:05.70, 219 audit entries | **PASS** |

## Phase 10 Fix Verification

| Test | Phase 9 Status | Phase 10 Status |
|------|---------------|-----------------|
| `gpu_execution - filter equality parquet` | SIGSEGV (assertion 19/31) | **PASS (exit 0, 31 assertions)** |
| `gpu_execution - tpch_q1_sf10_2gpu` | SF10 skipped; 16 assertions (gated) | **PASS (exit 0, 99 assertions — SF10 fully exercised)** |
| `gpu_execution - [mgpu-audit] per-GPU distribution on TPC-H Q1` | FAIL (pre-existing SIGSEGV) | **FAIL (same pre-existing SIGSEGV — not introduced by Phase 10)** |

Fix commit: `71fd623` — `feat(10-03): fix stream use-after-destroy SIGSEGV in parquet filter translation`

Root cause: `translation_stream` (local `rmm::cuda_stream`) destroyed at for-loop scope exit while scalars retain stale `cudaStream_t` handle; `cudaFreeAsync` fires on stale handle during `translated_expression` teardown at next QueryBegin.

Fix: moved `translation_stream` into `translated_expression::owned_stream` (declared before `owned_literals` for C++ reverse-destruction order), extending stream lifetime to match scalar lifetimes. 36 LOC.

## Static Invariants (post-run grep)

| Check | Baseline | Observed | Status |
|-------|----------|----------|--------|
| `grep -rn 'rmm::cuda_stream_default' src/` total | <= 41 | 40 | **PASS** |
| `grep -cnE 'cuda_set_device_raii.*target_gpu_id' src/op/scan/duckdb_scan_executor.cpp` | >= 2 | 2 | **PASS** |
| `grep -c 'set_preferred_device_id(target_gpu_id)' src/op/scan/duckdb_scan_executor.cpp` | 1 (Plan 09-01) | 1 | **PASS** |
| `grep -c '_batch_gpu_affinity\[' src/op/scan/duckdb_scan_executor.cpp` | >= 2 (Plan 09-02) | 2 | **PASS** |
| `grep -c 'set_intersection' test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp` | >= 1 (Plan 09-03) | 2 | **PASS** |
| Branch is `feature/single-node-multi-gpu2`; no merge to dev | required | confirmed | **PASS** |

All Phase 9 + Phase 10 source invariants PRESERVED. HYG-02 improved by 1 (41 → 40) by Phase 10-03 fix (explicit `rmm::cuda_stream` instead of any use of default).

## [mgpu-audit] Residual Attribution

**Pre-existing — NOT attributable to Phase 10 fix.**

Evidence chain:
1. `10-03-FIX.md` frontmatter explicitly states: `"[mgpu-audit] pre-existing SIGSEGV confirmed: fails identically on base branch before fix; not introduced by this plan"` (verified via `git stash; MCP run; git stash pop` during Plan 10-03 execution)
2. The MCP full-suite run in Plan 10-03 (post-fix) showed `full_suite_after: "exit -1, only [mgpu-audit] pre-existing SIGSEGV remains"` — same test, same location, identical failure mode
3. This Phase 10-04 run reproduces the same failure at the same location: `test_gpu_execution_tpch_mgpu_audit.cpp:200` (SIGSEGV)
4. The [mgpu-audit] test uses `attach_integration_duckdb(*con)` (DuckDB SF1 data) and calls `CALL gpu_execution(...)` — a completely different code path than the parquet filter translation fixed by Plan 10-03

The [mgpu-audit] SIGSEGV is a Phase-11 candidate for future investigation.

## Verdict

**PARTIAL** — The Phase 10 fix (stream use-after-destroy in parquet filter translation) is demonstrably closed and verified:
- `gpu_execution - filter equality parquet` exits 0 with 31 assertions (was: SIGSEGV at 19/31)
- `gpu_execution - tpch_q1_sf10_2gpu` exits 0 with 99 assertions (was: SF10 gate skipped; 16 assertions only)
- SF100 Q1 num_gpus=2 passes with byte-identical result vs 1-GPU baseline, 5.70s wall-clock, zero errors
- All CRIT-1, CRIT-4_carryover, and CRIT-6 criteria PASS outright
- CRIT-2 Phase 10 objective (close the two TABLE_FUNCTION SIGSEGV tests) is COMPLETE

The PARTIAL verdict is driven by one pre-existing `[mgpu-audit]` SIGSEGV that was present on the base branch before any Phase 10 work, is not related to the parquet filter translation fix, and is explicitly documented as out-of-scope for Phase 10.

v1.2 SF100 ship-gate (Criteria 1 + 6) PASSES. Phase 10 fix objective (Criterion 2 target tests) PASSES. The residual [mgpu-audit] SIGSEGV is an independently-scoped follow-on.

## Open Issue

### `[mgpu-audit]` per-GPU distribution SIGSEGV (Phase 11 candidate)

**Classification:** Pre-existing — present before Phase 10 changes, confirmed by Plan 10-03 documentation.

**Observed signature:** `SIGSEGV - Segmentation violation signal` at `test_gpu_execution_tpch_mgpu_audit.cpp:200` when running `CALL gpu_execution("SELECT ... FROM lineitem ...")` via the `attach_integration_duckdb` path. The SF10-scale `tpch_q1_sf10_2gpu` test (which uses `attach_sf10_tables()` via the parquet path) does NOT trigger this crash and PASSES with 99 assertions.

**Distinguishing factor:** The [mgpu-audit] test uses DuckDB-attached data (`attach_integration_duckdb`) rather than parquet files. It exercises the same `CALL gpu_execution(...)` form that SF100 uses (and which works). The crash is specific to the DuckDB-attach path with the AUDIT fixture setup at `test_gpu_execution_tpch_mgpu_audit.cpp:140-200`.

**Hypotheses:**
- **H-A:** The AUDIT test's `attach_integration_duckdb` path triggers a different code branch in the scan executor that has an unrelated stream/memory issue (not covered by the Phase 10 parquet filter translation fix)
- **H-B:** Catch2 test isolation issue: the mgpu-audit test is the last in the [mgpu-audit] group and runs after tpch_q1/q6/q12_sf10 tests. Context from prior tests may affect state in the audit test's DuckDB attach path.

**Impact:** Low — SF100 production workloads use the parquet path and are unaffected. The DuckDB-attach test path is exercised primarily for unit-test coverage.

**Suggested next actions (Phase 11 scope):**
1. Run `[mgpu-audit]` in isolation (first in suite, no prior test state) to confirm whether the crash is test-ordering dependent
2. If ordering-dependent: add `--rng-seed 0` or explicit isolation to the test setup
3. If not ordering-dependent: attach gdb to the crashing binary with `.claude/skills/debug-gdb/SKILL.md` and get a backtrace at the SIGSEGV frame inside the `attach_integration_duckdb` path

## Next Steps

**v1.2 status:** SF100 ship-gate (CRIT-1 + CRIT-6) PASSES. Phase 10 fix objective COMPLETE. The pre-existing [mgpu-audit] SIGSEGV is a Phase-11 follow-on if DuckDB-attach path coverage is required for v1.2 release criteria. Feature branch `feature/single-node-multi-gpu2` is preserved; no merge to dev until milestone-level decision.

**Phase 11 (if pursued):** Scope limited to the `[mgpu-audit]` DuckDB-attach SIGSEGV. Estimated scope: 1 plan (bisect + gdb + targeted fix or test isolation). Does not block v1.2 SF100 ship if CRIT-2's parquet-path tests are considered the authoritative gate.
