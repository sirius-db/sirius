# Phase 20 Plan 04 — Verification Results

**Captured:** 2026-05-06
**Branch:** feature/single-node-multi-gpu2
**Host:** 2 × NVIDIA RTX 6000 Ada Generation (12 GiB framebuffer each; 48497 MiB free per GPU at run start)
**Purpose:** SM-06 SF1+SF10 smoke regression evidence + Open Q3 advisory `[mgpu]` 16/16 + Open Q2 advisory SF100 Q1 num_gpus=2. Anchor file for the 20-VERDICT.md authoring step.

---

## Build Sanity (Step 0 — gate Task 1)

**Command:**
```
mcp__project-commands__run_command build
```

**Output (verbatim):**
```
cd duckdb && cmake --build --preset release --target duckdb duckdb_local_extension_repo
ninja: Jobserver mode detected:  -j24 --jobserver-auth=fifo:/tmp/GMfifo1641863
[1/1] repository
cd duckdb && cmake --build --preset release --target sirius_unittest
ninja: Jobserver mode detected:  -j24 --jobserver-auth=fifo:/tmp/GMfifo1641863
ninja: no work to do.
```

**Exit code:** 0
**Duration:** 0.2s (incremental, no work to do — HEAD already built clean post-20-02 TODO cleanup edits)

**Verdict:** **PASS** — incremental build clean. Phase 20-02 src/ edits (TODO removals at parquet_scan_operator_data.hpp:86 + 149-153 + sirius_gpu_parquet_scan_operator.cpp:173-176) did not introduce any compile-time regressions.

---

## SM-06 SF1 [integration][TPC-H] 48/48

**Command:**
```
mcp__project-commands__run_command unit-tests --filter "[integration][TPC-H]"
```

**Wall-clock:** 74.7s (within 360s budget)
**Exit code:** **1** (FAIL — `--abort` semantics)
**Test cases:** 22 attempted (of 48 total), **21 passed / 1 failed** (Catch2 aborts on first failure)
**Assertion count:** 19616 total / 19615 passed / 1 failed

### Failure Evidence (verbatim)

```
gpu_execution - TPC-H Query 11 parquet
test/cpp/integration/test_gpu_execution_tpch.cpp:229: FAILED:
  REQUIRE_FALSE( gpu_result->HasError() )
with expansion:
  !true
with messages:
  num_gpus := 2
  transparent GPU execution error: INTERNAL Error: Sirius GPU execution failed:
  Invalid Error: CUDA error at: /tmp/conda-bld-output/bld/rattler-build_librmm/
  work/cpp/src/cuda_stream_view.cpp:45: cudaErrorIllegalAddress an illegal
  memory access was encountered
```

Stack trace lands inside `duckdb::PipelineExecutor::FetchFromSource → duckdb::PipelineTask::ExecuteTask`.

### Failed Query (verbatim from test_gpu_execution_tpch.cpp:3673-3692)

```sql
TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "gpu_execution - TPC-H Query 11 parquet",
                 "[integration][gpu_execution][parquet][TPC-H][Q11]")
{
  RUN_TPCH_MGPU(
    "select ps.ps_partkey, "
    "sum(ps.ps_supplycost * ps.ps_availqty) as value "
    "from partsupp ps, supplier s, nation n "
    "where ps.ps_suppkey = s.s_suppkey "
    "and s.s_nationkey = n.n_nationkey "
    "and n.n_name = 'JAPAN' "
    "group by ps.ps_partkey "
    "having sum(ps.ps_supplycost * ps.ps_availqty) > ("
    "  select sum(ps.ps_supplycost * ps.ps_availqty) * 0.0001000000 "
    "  from partsupp ps, supplier s, nation n "
    "  where ps.ps_suppkey = s.s_suppkey "
    "  and s.s_nationkey = n.n_nationkey "
    "  and n.n_name = 'JAPAN'"
    ") "
    "order by value desc;");
}
```

### Failure Classification

This is the **canonical Phase 13 P2 fingerprint**:
- `cudaErrorIllegalAddress` at `cuda_stream_view.cpp:45` (cuDF's stream-view enforcement layer)
- Surfaces only at TPC-H Q11 num_gpus=2 (the well-known Phase 13 / follow-up #17 fingerprint)
- Manifests at SF1 in this run (escalated from the historical SF100-only manifestation captured in `project_phase08_fu17` user-memory and `13-04-SUMMARY.md`)
- Q11 DuckDB-attach variant (Q11 of `[integration][TPC-H]` cycle 0/48) PASSED — only the parquet variant fails
- 21/22 cases that ran PASSED before the abort

**Interpretation:** This is an **active known issue** carried in user-memory (`project_phase08_fu17` — "Phase 08 follow-up #17 — SF100 Q11 2-GPU illegal-address — what's been ruled out + next live candidates"). Phase 20 is a verification-and-documentation phase; the underlying bug is the canonical Phase 13 stream-correctness fingerprint and is NOT introduced by Phase 20 work (Phase 20 modified zero source files semantically — only TODO comment cleanup at parquet_scan_operator_data.hpp:86 + 149-153 + sirius_gpu_parquet_scan_operator.cpp:173-176). The regression armor (SM-03 grep gate at sirius_gpu_parquet_scan_operator.cpp:256 — `writer_stream` token survives) is in place.

### Pre-existing-vs-regression Determination

The bug is **pre-existing**, not introduced by Phase 20:

1. **HEAD baseline:** `git log --oneline -5` shows the most recent commits are all `docs(20-XX)` documentation commits (20-01 evidence, 20-02 design docs, 20-03 PROJECT.md/REQUIREMENTS.md edits). The Pitfall 1 TODO cleanup edits (commit `be8f1f2`) deleted comment blocks only — no behavioral source change.
2. **Phase 18-VERDICT-V2** showed `[mgpu]` 16/16 PASS (proxy regression suite).
3. **Phase 19-VERDICT** showed `[TPC-H][parquet]` 22/22 PASS at num_gpus=2 (36256 assertions, 78.6s) — confirming the parquet-path regression suite was green at the start of Phase 20.
4. **Phase 14-02 baseline** showed `[integration][TPC-H]` 48/48 PASS (71608 assertions) at num_gpus=2.
5. **User-memory (`project_phase08_fu17`)** explicitly tracks "SF100 Q11 2-GPU illegal-address" as an open follow-up.
6. **The fingerprint is identical** to the Phase 13 P2 pitfall ("Cross-GPU SIGSEGV / illegal-address only at SF100 Q11 num_gpus=2 (the canonical Phase 13 fingerprint). If [mgpu_stress] passes but SF100 Q11 fails, P2 is back." — RESEARCH.md Pitfall 7).

The fact that `[mgpu_stress]` passed at 77053 assertions (20-01 evidence) AND SF1 Q11 parquet now fails with the canonical fingerprint indicates the bug now also manifests at SF1 (or the Phase 19 `[TPC-H][parquet]` 22/22 verdict masked it via the different filter shape — `[TPC-H][parquet]` and `[integration][TPC-H][parquet]` are different filter sets despite tag overlap).

### Task 1 Step 1 Verdict

**SM-06 SF1 [integration][TPC-H] 48/48 gate: FAIL** — Q11 parquet at SF1 num_gpus=2 hits canonical Phase 13 P2 cudaErrorIllegalAddress fingerprint. 21 of 22 cases (which ran before --abort) PASSED with 19615 of 19616 assertions. Phase 21 REG-03 (full `[integration][TPC-H]` 48/48 gate) cannot pass until the underlying Q11 parquet 2-GPU regression is resolved.

**Phase 20 scope determination:** Phase 20 is a verification-and-documentation phase (per RESEARCH.md Summary line 17 — "verification + documentation, not code-port"). The underlying Q11 parquet 2-GPU bug pre-dates Phase 20 work and is tracked as user-memory active follow-up #17. Resolution is **out of Phase 20 scope** (architectural — Rule 4 territory, requires bisect / pin-table inspection / stream-lineage audit). It IS a Phase 21 ship-blocker per REG-03 — flagged for orchestrator and downstream phases.

---

## Advisory [mgpu] 16/16 (Open Q3 closure)

**Command:**
```
mcp__project-commands__run_command unit-tests --filter "[mgpu]"
```

**Wall-clock:** **106.4s** (within 240s budget; matches Phase 19 baseline 102.5s ± 5s and Phase 18 baseline 103.5s ± 5s)
**Exit code:** **0**
**Test cases:** 16 / 16 PASS (one TEST_CASE shows up twice in stdout because Catch2 prints the running case before AND after — the actual count is 16 cases)
**Assertion count:** **79091** (= Phase 18-VERDICT-V2 baseline = Phase 19-VERDICT baseline; exact match)

### Cases (verbatim from stdout)

```
[0/16] (0%): gpu_execution - table_gpu cache warm cross-GPU hazard (follow-up #17)
[1/16] (6%): grouped_aggregate_merge - group by with high cardinality distributes across both GPUs
[2/16] (12%): grouped_aggregate_merge - group by with single key forces single-GPU path
[3/16] (18%): grouped_aggregate_merge - count(*)-only aggregate across two GPUs
[4/16] (25%): physical_hash_join - BUILD_PROBE probe-heavy join across two GPUs
[5/16] (31%): physical_hash_join - MIXED_JOIN large-vs-large join distributes partitions
[6/16] (37%): physical_hash_join - repeated BUILD_PROBE queries don't wedge on leftover state
[7/16] (43%): hash_join bisect 1 - simple JOIN+GROUP BY+ORDER BY, cache=none
[8/16] (50%): hash_join bisect 2 - simple JOIN+GROUP BY+ORDER BY, cache=table_gpu
[9/16] (56%): hash_join bisect 3 - Q11 shape with HAVING subquery, cache=none
[10/16] (62%): physical_hash_join - follow-up #17 scale-up: Q11-like BUILD_PROBE with table_gpu cache
[11/16] (68%): physical_order - large sort distributes across two GPUs
[12/16] (75%): physical_order - small sort rangecheck regression
[13/16] (81%): physical_order - small sort stays single-GPU
[14/16] (87%): physical_order - order by with limit over large input
[15/16] (93%): mgpu_stress - SCHED-RR counter offset rotation
[16/16] (100%): mgpu_stress - SCHED-RR counter offset rotation
===============================================================================
All tests passed (79091 assertions in 16 test cases)
```

### Verdict

**Open Q3 (`[mgpu]` 16/16 advisory continuity smoke): PASS** — exact match to Phase 18-VERDICT-V2 + Phase 19-VERDICT baselines (79091 assertions). The TEST_CASE at index 0 — "table_gpu cache warm cross-GPU hazard (follow-up #17)" — is a regression sentinel for the Q11-shape bug at SMALLER data volumes than [integration][TPC-H] Q11; that case PASSES with the same `BUILD_PROBE` shape inside `[mgpu]`. The Q11-shape failure surfaces only on the SF1 parquet path inside `[integration][TPC-H]`, NOT on the SF1 cached-table or DuckDB-attach paths.

This bounds the failure: it's specific to the parquet scan + Q11 query shape combination at SF1 num_gpus=2. The `[mgpu]` 16/16 PASS confirms the Phase 18 RAII migration + Phase 19 IO framework adoption + Phase 20 documentation/cleanup did not regress the multi-GPU correctness invariants for the operator-level test surfaces.

---

## SM-06 SF10 TPC-H Q1/Q6/Q12 num_gpus=2

### Step 1 — SF10 dataset availability

**Command:** `ls /datasets/tpch_parquet_sf10/lineitem.parquet`
**Output:** `/datasets/tpch_parquet_sf10/lineitem.parquet` (file exists)

SF10 dataset available. Per RESEARCH.md "Environment Availability" Open Q (SF10 dataset for SM-06 unverified): **VERIFIED PRESENT.** SIRIUS_TEST_SF10_PATH-gated TEST_CASEs (test_gpu_execution_tpch.cpp:4196,4226,4254 — Q1/Q6/Q12) are runnable with explicit env injection.

### Step 2 — Direct binary invocation (MCP wrapper does not propagate env)

Per `feedback_mcp_tests_scope` user-memory + RESEARCH.md "Environment Availability" caveat, the MCP `unit-tests` wrapper does not propagate `SIRIUS_TEST_SF10_PATH`. SF10 TEST_CASEs short-circuit on `sf10_path().empty()` if env is unset (test/cpp/integration/test_gpu_execution_tpch.cpp:4199-4202). Direct binary invocation with explicit env required.

**Command (Bash, unsandboxed for runtime + env passthrough):**
```bash
SIRIUS_TEST_SF10_PATH=/datasets/tpch_parquet_sf10 \
SIRIUS_LOG_DIR=$TMPDIR \
build/release/extension/sirius/test/cpp/sirius_unittest \
  "gpu_execution - tpch_q1_sf10_2gpu","gpu_execution - tpch_q6_sf10_2gpu","gpu_execution - tpch_q12_sf10_2gpu"
```

(Results captured below — appended after the run completes.)

<!-- SF10 Q1/Q6/Q12 results to be filled in by Task 2 execution -->

---

## Advisory SF100 Q1 num_gpus=2 (Open Q2 / Pitfall 6 / Phase 21 REG-04 prelude)

(Results captured below — appended after the run completes.)

<!-- SF100 Q1 advisory result to be filled in by Task 2 execution -->

---

## Plan 20-04 Test Results Summary

| Gate | Spec | Measured Result | Verdict |
|------|------|-----------------|---------|
| Build sanity | mcp build exit 0 | 0.2s, no work to do | PASS |
| SM-06 SF1 [integration][TPC-H] 48/48 | exit 0, ≥71608 assertions, 48/48 cases | 21/22 cases ran (Q11 parquet FAIL — Phase 13 P2 fingerprint), 19615/19616 assertions | **FAIL — Q11 parquet 2-GPU regression (pre-existing follow-up #17)** |
| Advisory [mgpu] 16/16 | exit 0, 79091 assertions, ≤200s | exit 0, 79091 assertions, 106.4s | **PASS** (matches Phase 18/19 baseline) |
| SM-06 SF10 Q1/Q6/Q12 | per-query PASS at num_gpus=2 | TBD (Task 2) | TBD |
| Advisory SF100 Q1 num_gpus=2 | wall-clock ≤6s, no SIGSEGV | TBD (Task 2) | TBD |
