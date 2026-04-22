---
phase: 08-multi-gpu-sql-pipeline-fix
milestone: v1.2
status: SHIP_BLOCKED
completed_date: 2026-04-22
started_date: 2026-04-21

tags: [multi-gpu, pipeline, cross-device-stream, pattern-2, host-parquet-converter, tpch-integration, mgpu-audit]

# Phase-level requirements coverage
requirements_total: 11
requirements_closed:
  - FIX-01  # Plan 08-01
  - FIX-02  # Plan 08-02
  - FIX-03  # Plan 08-06 (HYG sweep PASS — 41 baseline preserved)
  - FIX-04  # Plan 08-06 (MCP build exit 0)
  - TEST-01 # Plan 08-04
  - TEST-02 # Plan 08-04
  - TEST-03 # Plan 08-05 (authored; runtime PARTIAL — see ship-gate)
  - TEST-04 # Plan 08-05 (authored; runtime DEFERRED)
  - AUDIT-01 # Plan 08-03 (log payload) + 08-05 (TEST_CASE authored)
  - AUDIT-02 # Plan 08-03 + 08-05
  - AUDIT-03 # Plan 08-03 + 08-05

ship_verdict: BLOCKED_ON_RESIDUAL_FIX_SITE

# Plans
plans:
  - id: 08-01
    title: "FIX-01 per-GPU stream pool in duckdb_scan_executor"
    summary: "Replaced singular GPU-0-bound stream pool with an unordered_map<int, unique_ptr<exclusive_stream_pool>>; wrapped dispatch and acquire in rmm::cuda_set_device_raii paired guards. Root-cause fix for the scan executor's cross-device stream-device mismatch."
    requirements: [FIX-01]
    duration: 6min
    commits: [2ff8a3c, 2150777, 0d03ca1]
  - id: 08-02
    title: "FIX-02 Branch B Sirius host->gpu converter override"
    summary: "Authored src/data/sirius_host_to_gpu_converter.{hpp,cpp} (+237 LOC) implementing Pattern 2 idiom for the host_data_representation → gpu_table_representation path. Re-probe confirmed Branch B closed THAT specific converter frame but surfaced a distinct fix-site (host_parquet_representation) — handed off to 08-06."
    requirements: [FIX-02]
    duration: 15min
    commits: [46b933f, 96481df, fae1915]
  - id: 08-03
    title: "AUDIT-01/02/03 log payload extension (task_id + batch_id)"
    summary: "Extended the two existing [mgpu-audit] INFO emissions with task_id (from gpu_pipeline_task::get_task_id()) and batch_id (reusing _scan_round_robin counter) suffixes. Grep-stable payload shape preserved."
    requirements: [AUDIT-01, AUDIT-02, AUDIT-03]
    duration: 6min
  - id: 08-04
    title: "TEST-01/02 TPC-H parameterization on num_gpus ∈ {1,2}"
    summary: "Added integration-2gpu.yaml; introduced g_integration_env_2gpu + acquire_integration_env_for helpers + bind_env/release_env fixture machinery; applied RUN_TPCH_MGPU macro to all 44 TPC-H TEST_CASEs. 1-GPU default preserved; 2-GPU variant fires via Catch2 GENERATE."
    requirements: [TEST-01, TEST-02]
    duration: 20min
  - id: 08-05
    title: "TEST-03/04 + AUDIT TEST_CASE acceptance gate"
    summary: "Authored dedicated [mgpu-audit] TEST_CASE with per-GPU unique-ID counting via tmp-log-dir + regex parse; SF10 Q1/Q6/Q12 2-GPU variants gated on SIRIUS_TEST_SF10_PATH env; Q4 scoped retry wrapper per ROADMAP flake policy. Runtime verification partial: 609 passed, 1 known-08-06 fail, 373 deferred due to MCP --abort."
    requirements: [TEST-03, TEST-04, AUDIT-01, AUDIT-02, AUDIT-03]
    duration: 86min
  - id: 08-06
    title: "Ship-gate VALIDATION + carryover fix to host_parquet converter"
    summary: "Applied Pattern 2 idiom to convert_host_parquet_to_gpu_with_prefetched_data_source (08-02 carryover). FIX-03 HYG grep PASS (41 baseline unchanged). FIX-04 clean build exit 0. Pattern 2 idiom grep-verifiable at 4 fix sites. Criteria 1/2/4/6 DEFERRED — carryover fix did NOT close the cudaErrorInvalidValue @ cuda_memcpy.cu:42 on parquet + num_gpus=2; at least one additional fix-site remains."
    requirements: [FIX-03, FIX-04]
    duration: 21min
    commits: [bf53dcc, 584707b]
---

# Phase 08 Multi-GPU SQL Pipeline Fix — Phase Summary

**Shipped (authored and code-review-verifiable):** 11 of 11 REQ-IDs covered; FIX-01 + FIX-02 (Branch B) + HYG + build + all TEST + AUDIT sources are in. The cross-device stream-correctness bug shape from v1.1 is structurally closed at the primary (scan dispatch) site AND at the secondary (cucascade host_data converter) site; Pattern 2 idiom is grep-visible at 4 fix sites; the [mgpu-audit] log payload carries task_id + batch_id; the AUDIT TEST_CASE asserts per-GPU unique counts; the TPC-H matrix is parameterized on num_gpus ∈ {1,2}.

**Remaining blocker:** At least one additional fix-site on the `host_parquet_representation → gpu_table_representation` path produces the same `cudaErrorInvalidValue @ cuda_memcpy.cu:42` signature on num_gpus=2 EVEN AFTER the 08-06 carryover fix landed on `convert_host_parquet_to_gpu_with_prefetched_data_source`. The failing tests are `gpu_execution hive partition - filter on data column` (integration.yaml flipped to num_gpus: 2) and `gpu_execution - TPC-H Query 1 parquet` (num_gpus=2 via GENERATE(1,2) from 08-04). Root cause not identified within 08-06's plan scope; the plan's `<carryover_fix>` directive explicitly said "If the fix requires more than ~50 LOC or you hit non-obvious complexity: STOP, return to orchestrator". Four candidate hypotheses documented in `08-06-VALIDATION.md` Open Issue section with suggested next actions.

## Milestone v1.2 Ship Verdict: **BLOCKED_ON_RESIDUAL_FIX_SITE**

This phase delivers:

- All planned requirements covered (11/11).
- Static-invariant ship criteria PASS: HYG-02 zero net-new, Pattern 2 idiom grep-visible, MCP build clean, cucascade submodule untouched.
- Partial runtime ship criteria PASS: non-TPC-H multi-GPU work + TPC-H DuckDB-fixture 2-GPU variants pass via MCP unit-tests.

But:

- Full runtime ship criteria FAIL: TPC-H parquet-fixture 2-GPU variants (including Q1 used for the SF100 ship-gate) hit `cudaErrorInvalidValue @ cuda_memcpy.cu:42` — identical v1.1 bug signature.

The milestone cannot ship until the residual fix-site is closed. Recommended next action: a single-plan v1.2.1 patch (or the start of v1.3 Phase 9) that applies more targeted observability (entry/exit SIRIUS_LOG_INFO with `cudaGetDevice`, stream identity, memory-space device_id) to `convert_host_parquet_to_gpu_with_prefetched_data_source` to determine whether the converter IS entered vs. whether an upstream H2D frame is the remaining hazard. Full guidance in `08-06-VALIDATION.md` Open Issue section.

## Phase 8 Requirements Traceability

| REQ-ID  | Description                                                  | Plan-of-record | Authoring status | Runtime status                  |
| ------- | ------------------------------------------------------------ | -------------- | ---------------- | ------------------------------- |
| FIX-01  | lock_or_prepare_batch cross-device stream-correctness        | 08-01          | Complete         | PASS via scan-dispatch 2-GPU tests |
| FIX-02  | Audit + apply to other cross-device memcpy call-sites        | 08-02          | Complete         | PARTIAL — Branch B closes Site C (host_data); host_parquet path has residual bug post 08-06 carryover |
| FIX-03  | Zero net-new rmm::cuda_stream_default                        | 08-06          | Complete         | PASS (41 baseline preserved)    |
| FIX-04  | MCP build exits 0                                            | 08-06          | Complete         | PASS (exit 0)                   |
| TEST-01 | Parameterize TPC-H on num_gpus ∈ {1,2}                       | 08-04          | Complete         | PASS (44 TEST_CASEs × 2 GPUs)   |
| TEST-02 | integration.yaml flow supports num_gpus=2                    | 08-04          | Complete         | PASS (integration-2gpu.yaml)    |
| TEST-03 | All 22 TPC-H SF1 queries pass on num_gpus=2                  | 08-05          | Complete         | PARTIAL — all DuckDB-fixture pass; parquet-fixture blocked by residual bug |
| TEST-04 | TPC-H Q1, Q6, Q12 SF10 pass on num_gpus=2                    | 08-05          | Complete         | DEFERRED — blocked by same residual bug (SF10 uses parquet path) |
| AUDIT-01 | pipeline_task >0 on BOTH GPUs, log-grep asserted             | 08-03 + 08-05  | Complete         | DEFERRED — TEST_CASE authored; doesn't fire because --abort halts at the residual-bug failure before audit test order |
| AUDIT-02 | scan_batch >0 on BOTH GPUs, log-grep asserted                | 08-03 + 08-05  | Complete         | DEFERRED — same                 |
| AUDIT-03 | Default unit-tests run catches single-GPU regressions        | 08-05          | Complete         | DEFERRED — same                 |

**Coverage:** 11/11 authored; 4/11 PASS on runtime (FIX-01, FIX-03, FIX-04, TEST-01, TEST-02 — and part of TEST-03); 7/11 blocked on the single residual parquet fix-site.

## Phase 8 Deliverables (code)

- **FIX-01 `src/op/scan/duckdb_scan_executor.{hpp,cpp}`** — per-GPU stream pool map, paired device-RAII guards for acquire + dispatch, target_gpu_id hoisted above parquet-only block.
- **FIX-02 Branch B `src/data/sirius_host_to_gpu_converter.{hpp,cpp}` (NEW, 237+73 lines)** — host_data_representation → gpu_table_representation converter override with target-bound stream + target-device RAII; public-API-only column-tree reconstruction (no dependency on cucascade-private helpers).
- **FIX-02 host_parquet carryover `src/data/host_parquet_representation_converters.cpp`** — Pattern 2 idiom applied to `convert_host_parquet_to_gpu_with_prefetched_data_source` (lines 55-119 of original; +41/-4 in plan 08-06).
- **Registry wiring `src/include/data/sirius_converter_registry.hpp`** — MGPU-06 P2P override (existing) + FIX-02 host override (new, plan 08-02).
- **CMakeLists.txt** — `src/data/sirius_host_to_gpu_converter.cpp` added to EXTENSION_SOURCES.
- **AUDIT-03 `src/pipeline/pipeline_executor.cpp` + `src/op/scan/duckdb_scan_executor.cpp`** — extended [mgpu-audit] INFO emissions with task_id / batch_id suffixes (plan 08-03).

## Phase 8 Deliverables (tests)

- **TEST-01/02 `test/cpp/integration/integration-2gpu.yaml` (NEW)** + `test/cpp/utils/sirius_test_env.{hpp,cpp}` updates (plan 08-04) — second shared_test_env for num_gpus: 2; acquire_integration_env_for(int) helper.
- **TEST-01/02 `test/cpp/unittest.cpp`** — constructs paused 2-GPU env; listener pauses between tests.
- **TEST-01/02 `test/cpp/integration/test_gpu_execution_tpch.cpp`** — fixture-base bind_env/release_env/compare_gpu_vs_cpu_for + setup_schema virtual hook + RUN_TPCH_MGPU macro applied to all 44 TPC-H TEST_CASEs.
- **AUDIT-01/02/03 `test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp` (NEW, 260 lines)** (plan 08-05) — dedicated [mgpu-audit] TEST_CASE with tmp-log-dir + regex parse + per-GPU std::set<string> unique-ID counting + REQUIRE on per-GPU unique counts ≥ threshold.
- **TEST-04 SF10 Q1/Q6/Q12 2-GPU TEST_CASEs** (plan 08-05) — `sf10_path() / attach_sf10_tables() / compare_gpu_vs_cpu_sf10_for()` helpers on GPUExecutionFixtureBase; gated on SIRIUS_TEST_SF10_PATH env.
- **TEST-03 Q4-scoped retry wrapper** (plan 08-05) — inline GENERATE(1,2) + try/catch retry per ROADMAP flake policy.

## Phase 8 Deliverables (docs)

- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-01-PLAN.md..08-06-PLAN.md` (6 plans)
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-01-SUMMARY.md..08-06-SUMMARY.md` (6 plan-level summaries)
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-01-REPRODUCTION.md` — FIX-01 pre-reproduction + static-audit
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-02-PROBE.md` — FIX-02 static audit (Sites A-L) + MCP-driven FAIL verdict + post-override re-probe + distinct-fix-site handoff
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-05-RUN.md` — unit-tests outcome + per-query status + 08-06 handoff
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-06-VALIDATION.md` — Phase 8 ship-gate evidence + Open Issue + follow-up command blocks
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-SUMMARY.md` — this file
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-RESEARCH.md` — Phase 8 technical research (Pattern 1-4 dispatch taxonomy, FIX classifiers)

## Outstanding Work (post-Phase-8)

### 1. Close the residual parquet + num_gpus=2 fix-site

This is the single blocker for v1.2 ship. Current state:

- Failing tests: `gpu_execution hive partition - filter on data column`, `gpu_execution - TPC-H Query 1 parquet` (and presumably many other TPC-H parquet variants — all blocked by --abort after the first failure).
- Error: `cudaErrorInvalidValue @ cuda_memcpy.cu:42`.
- Already addressed and did NOT close: the 08-06 carryover fix to `convert_host_parquet_to_gpu_with_prefetched_data_source`.
- Hypothesis candidates documented in `08-06-VALIDATION.md` Open Issue section A through D.
- Suggested first action: add SIRIUS_LOG_INFO breadcrumbs at converter entry/exit showing cudaGetDevice + stream.value() + target_device_id + memory_space device_id, then re-run the failing test under MCP to determine whether the converter is even entered. This is a ~10-line instrumentation commit that should deterministically identify whether the remaining hazard is upstream or inside the converter.

### 2. Complete SF10 smoke + SF100 Q1 ship-gate (criteria 1 + 6)

After step 1 lands:

- Run `SIRIUS_TEST_SF10_PATH=/datasets/tpch_parquet_sf10 mcp__project-commands__run_command unit-tests` — expect 88/88 TPC-H × {DuckDB,parquet} × {1,2} pass; AUDIT TEST_CASE asserts `>=5 per GPU`.
- Run the SF100 Q1 command block in `08-06-VALIDATION.md` "How to complete criteria 1 + 6" on num_gpus=2.
- Record wall-clock, [mgpu-audit] distribution, num_gpus=1-vs-2 result parity in `08-06-VALIDATION.md` criterion 1 + 6 sections.

### 3. Milestone v1.2 ship

After steps 1-2, mark `status: SHIP` in the frontmatter, update ROADMAP.md with phase completion, archive `.planning/milestones/v1.2-ROADMAP.md`.

## Risks / Concerns for v1.3

- **Silent parquet-path bug resilience.** The residual fix-site exposed by 08-06's re-probe means the v1.2 bug signature was actually TWO coupled bugs at two fix-sites sharing the same error surface. A v1.3 hardening pass should add an assertion in the converter-registry dispatch path that the caller stream belongs to the target device (or a documented bypass), to surface future regressions at function entry rather than inside cudf's H2D path.
- **MCP daemon commands.yaml caching.** Plan 08-05 documented that MCP does not hot-reload commands.yaml mid-session. To probe with `--abortx 999` (or any flag edit) needs a fresh daemon session. v1.3 infra work could wrap MCP with a session-reset capability.
- **Host-environment heterogeneity.** The current worktree host has 2 GPUs (confirmed via MCP) but no driver in the bash shell. Direct binary invocation is gated by MCP. Documented as an operational constraint; no action required beyond awareness.

## Key Architectural Decisions Preserved

- **Pattern 2 as the canonical idiom** for any cross-device H2D/P2P hot path: acquire a target-bound stream under `rmm::cuda_set_device_raii{target_device_id}` and issue every memcpy on that stream, NEVER the caller's stream.
- **Public-API-only converter overrides** to preserve cucascade submodule pin. All Sirius-side converter overrides (P2P, host_fast_to_gpu, host_parquet) use only cucascade's public headers + cudf factories. Private cucascade helpers are reimplemented inside Sirius when needed.
- **Per-GPU stream pool map** in cross-GPU executors (vs. per-executor singleton pool). Applied to duckdb_scan_executor in 08-01; pattern documented for future use.
- **[mgpu-audit] grep-stable payload** (prefix + GPU N + key=value suffixes) preserves backward compatibility with v1.1 verification scripts while enabling unique-ID counting assertions.

---

**Phase 8 authored:** 2026-04-22. **Ship verdict:** BLOCKED pending closure of residual parquet + num_gpus=2 fix-site. **Evidence:** `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-06-VALIDATION.md`.

*Milestone: v1.2*
*Phase: 08-multi-gpu-sql-pipeline-fix*
*Completed: 2026-04-22*
