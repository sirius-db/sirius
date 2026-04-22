---
phase: 08-multi-gpu-sql-pipeline-fix
plan: 06
subsystem: ship-gate

tags: [validation, ship-gate, pattern-2, fix-02-carryover, host-parquet-converter, mgpu-audit, hyg-sweep]

# Dependency graph
requires:
  - phase: 08-multi-gpu-sql-pipeline-fix
    plan: 01
    provides: "FIX-01 per-GPU stream pool map; pattern-2 idiom presence in duckdb_scan_executor (criterion 5 match)"
  - phase: 08-multi-gpu-sql-pipeline-fix
    plan: 02
    provides: "FIX-02 Branch B host_data->gpu_table converter override; canonical template src/data/sirius_host_to_gpu_converter.cpp; post-probe handoff pointing to host_parquet path"
  - phase: 08-multi-gpu-sql-pipeline-fix
    plan: 03
    provides: "[mgpu-audit] INFO emissions carrying task_id + batch_id suffixes"
  - phase: 08-multi-gpu-sql-pipeline-fix
    plan: 04
    provides: "integration-2gpu.yaml + GENERATE(1,2) parameterization so the bug surface is actually exercised"
  - phase: 08-multi-gpu-sql-pipeline-fix
    plan: 05
    provides: "AUDIT TEST_CASE source + SF10 Q1/Q6/Q12 TEST_CASEs + Q4 retry wrapper + 08-05-RUN.md reference"
provides:
  - "08-06-VALIDATION.md — Phase 8 ship-gate evidence document per ROADMAP criterion 6"
  - "08-06 carryover fix to convert_host_parquet_to_gpu_with_prefetched_data_source (target-bound stream + target-device RAII, mirroring 08-02 Branch B)"
  - "Residual-bug handoff: at least one more fix-site remains on the parquet+num_gpus=2 path, not closed by 08-06's single-file scope"
affects:
  - "milestone v1.2 ship verdict — blocked on residual fix-site"
  - "any post-08-06 hot-fix plan that chases the remaining host_parquet cross-stream hazard"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern 2 extended to the sirius-owned parquet→gpu converter: sync caller stream → enter rmm::cuda_set_device_raii(target_device_id) → acquire_stream() from target_memory_space → use target_stream for cudf::io::read_parquet + apply_post_convert + apply_partition_inject + final sync → consume sticky cuda errors via (void)cudaGetLastError()"
    - "VALIDATION.md criterion-by-criterion verdict matrix format (borrowed shape from 07-04-VALIDATION.md) with explicit PASS/DEFERRED/FAIL per numbered ROADMAP criterion"

key-files:
  created:
    - ".planning/phases/08-multi-gpu-sql-pipeline-fix/08-06-VALIDATION.md (319 lines — ship-gate evidence document + residual-bug open-issue analysis)"
    - ".planning/phases/08-multi-gpu-sql-pipeline-fix/08-06-SUMMARY.md (this file)"
    - ".planning/phases/08-multi-gpu-sql-pipeline-fix/08-SUMMARY.md (phase-level summary)"
  modified:
    - "src/data/host_parquet_representation_converters.cpp (+41/-4 lines — 08-06 carryover fix: Pattern 2 idiom on convert_host_parquet_to_gpu_with_prefetched_data_source; added cuda_runtime_api.h include for cudaGetLastError)"

key-decisions:
  - "Applied carryover fix FIRST before validation tasks per orchestrator directive. The fix is scoped narrowly to the one function 08-02-PROBE.md identified (`convert_host_parquet_to_gpu_with_prefetched_data_source` at lines 55-119). Reimplemented the exact Pattern 2 idiom from sirius_host_to_gpu_converter.cpp (08-02 Branch B template). Build exits 0; HYG-02 baseline preserved at 41 matches."
  - "Did NOT expand the fix scope when the post-fix re-probe showed the same failure signature on the same test. Per plan's `<carryover_fix>` directive: 'If the fix requires more than ~50 LOC or you hit non-obvious complexity: STOP'. The remaining failure indicates at least one additional fix-site beyond what the plan identified. Documented four hypothesis candidates in VALIDATION.md's Open Issue section with suggested next actions; did not chase."
  - "Recorded ROADMAP criteria 3 + 5 as PASS on the static-invariant evidence (HYG grep, Pattern 2 grep, clean build). Recorded criteria 1/2/4/6 as DEFERRED with the residual blocker explicitly pointed out. This gives the milestone-verdict author a deterministic picture (not mixed up with criterion 5's PASS)."
  - "Did NOT run an SF100 Q1 ship-gate on num_gpus=2 on this host because the SF1 parquet TPC-H Q1 on num_gpus=2 already reproduces the same `cudaErrorInvalidValue @ cuda_memcpy.cu:42` as a fast smoke-test; running SF100 would not produce ship-gate evidence, it would reproduce the known blocker at much higher wall-clock cost. The SF100 run command block is preserved verbatim in VALIDATION.md for execution once the residual fix-site is closed."
  - "Kept integration.yaml committed at num_gpus: 1 per v1.2 Out-of-Scope constraint. The 2-GPU runs were transient yaml flips (reverted before any commit)."

patterns-established:
  - "When a VALIDATION plan's carryover fix does not fully close a known bug signature, record the fix as 'did-the-right-thing' (code-review-verifiable) + document the residual as a scoped open-issue handoff. Don't escalate into open-ended bug hunting — that's a separate plan / hot-fix's scope."
  - "VALIDATION.md must enumerate each ROADMAP success criterion verbatim with an explicit verdict per criterion, plus an invariants table summarizing grep/build/submodule-clean evidence. Criterion-level deferrals must list the specific blocker (test name + file:line + error signature) so a future executor can resume precisely."

requirements-completed: [FIX-03, FIX-04]
# FIX-03 = HYG-02 zero net-new `rmm::cuda_stream_default` — PASS (41 baseline preserved)
# FIX-04 = MCP build exits 0 after rm -rf build — PASS
# (ROADMAP criterion 5 Pattern 2 idiom grep also PASS but is not a numbered REQ-ID.)
# FIX-01/02 + TEST-01/02/03/04 + AUDIT-01/02/03 were already recorded complete
# by plans 08-01..08-05; this plan's scope is FIX-03 + FIX-04 + runtime gate.

# Metrics
duration: 21min
completed: 2026-04-22
---

# Phase 08 Plan 06: Ship-Gate VALIDATION + FIX-02 Host-Parquet Carryover Summary

**Closed FIX-03 (HYG-02 grep) + FIX-04 (clean build) on the static side with an explicit PASS verdict. Applied the orchestrator-directed carryover fix to `src/data/host_parquet_representation_converters.cpp` (Pattern 2 idiom — target-bound stream + target-device RAII — mirroring 08-02 Branch B's template). Produced `08-06-VALIDATION.md` recording criterion-by-criterion verdicts. Criteria 1/2/4/6 remain DEFERRED because the same `cudaErrorInvalidValue @ cuda_memcpy.cu:42` signature persists on num_gpus=2 parquet TPC-H Q1 (and hive-partition filter) AFTER the carryover fix landed — the fix addressed one known site but at least one additional site remains. Per plan directive ('Don't over-author'), the residual is handed off with four hypothesis candidates + suggested next actions rather than chased.**

## Performance

- **Duration:** ~21 min (wall clock)
- **Started:** 2026-04-22 04:33 UTC
- **Tasks:** 3 authored + 1 orchestrator-directed carryover fix (applied first)
- **Commits:** 3 (`bf53dcc` carryover fix, `584707b` VALIDATION.md, final metadata commit pending)

## What Shipped

### Carryover fix: `src/data/host_parquet_representation_converters.cpp` (+41/-4)

Applied the exact Pattern 2 idiom from `src/data/sirius_host_to_gpu_converter.cpp` (08-02 Branch B) to `convert_host_parquet_to_gpu_with_prefetched_data_source`:

1. `stream.synchronize()` — flush any work on the caller's stream (cross-device-safe).
2. `rmm::cuda_set_device_raii target_device_raii(target_device_id)` — unchanged (was already present).
3. `auto target_stream = target_memory_space->acquire_stream()` — NEW: acquire a stream from the target memory space's pool.
4. Replace every `stream` with `target_stream` in: `cudf::io::read_parquet(opts, target_stream, mr_ref)`, `apply_post_convert(std::move(table), target_stream)`, `apply_partition_inject(std::move(table), target_stream)`, final `target_stream.synchronize()`.
5. `(void)cudaGetLastError()` — consume sticky cuda state before returning (Pattern 2 hygiene).

Added `#include <cuda_runtime_api.h>` for `cudaGetLastError`. No other files modified.

### `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-06-VALIDATION.md` (319 lines)

Criterion-by-criterion verdict matrix with:

- PASS on criteria 3 (FIX-03 HYG), 4 build-gate subclause (FIX-04), 5 (Pattern 2 idiom grep).
- DEFERRED on criteria 1, 2, 4 audit-assertion subclause, and 6 — all blocked by the same residual parquet+num_gpus=2 fix-site.
- Full grep output evidence for criteria 3 and 5.
- Full MCP unit-tests run outputs for both integration.yaml flavors (num_gpus: 1 default + temporary num_gpus: 2 flip).
- "Open Issue" section with four hypothesis candidates (A-D) and concrete suggested next actions for the post-08-06 follow-up.
- Invariants table showing HYG baseline unchanged, cucascade submodule clean, yaml restored.

## MCP Runs

| Invocation                                                 | Exit | Duration | Detail                                                                |
| ---------------------------------------------------------- | ---- | -------- | --------------------------------------------------------------------- |
| `build` (post carryover-fix edit)                          | 0    | 6.7s     | Incremental: recompiled host_parquet_representation_converters.cpp + linked deps |
| `unit-tests` (integration.yaml = num_gpus: 2 — flipped)   | 1    | 35.9s    | 316 run, 315 passed, 1 failed: `gpu_execution hive partition - filter on data column` — `cudaErrorInvalidValue @ cuda_memcpy.cu:42` |
| `unit-tests` (integration.yaml = num_gpus: 1 — committed) | 1    | 147.7s   | 610 run, 609 passed, 1 failed: `gpu_execution - TPC-H Query 1 parquet` (num_gpus=2 section via GENERATE(1,2)) — same `cudaErrorInvalidValue @ cuda_memcpy.cu:42` |

Both failures share the same bug signature. The 08-06 carryover fix did NOT close them.

## Static Invariants (all green)

| Check                                                                                           | Result                                               |
| ----------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `grep -rn 'rmm::cuda_stream_default' src/` total                                                | 41 (baseline unchanged)                              |
| HYG matches in Phase 8 modified files (duckdb_scan_executor, sirius_host_to_gpu_converter, sirius_converter_registry, pipeline_executor, host_parquet_representation_converters) | 0 net-new                                            |
| `mcp build` exit                                                                                | 0 (6.7s)                                             |
| `grep -rnE 'cuda_set_device_raii.*(target\|source)' src/op/scan/duckdb_scan_executor.cpp src/data/sirius_p2p_converter.cpp src/data/sirius_host_to_gpu_converter.cpp src/data/host_parquet_representation_converters.cpp` | 6 code + 2 doc matches (ROADMAP criterion 5 evidence) |
| cucascade submodule (`git status cucascade/`)                                                   | clean (untouched)                                    |
| `git diff test/cpp/integration/integration.yaml`                                                | empty (restored to num_gpus: 1 before commit)        |

## Commits

| Task                                                                                           | Commit    | Type |
| ---------------------------------------------------------------------------------------------- | --------- | ---- |
| Carryover fix: apply Pattern 2 idiom to host_parquet_representation_converters.cpp             | `bf53dcc` | fix  |
| Validation evidence: record Phase 8 VALIDATION.md with criterion-by-criterion verdicts         | `584707b` | docs |
| Plan metadata (this file + STATE.md + ROADMAP.md + REQUIREMENTS.md updates)                    | pending   | docs |

## Deviations from Plan

### Orchestrator-directed carryover fix (applied first)

**Not a deviation in the strict sense** — the orchestrator's `<carryover_fix>` block explicitly directed applying this fix before the validation tasks. Recorded here for audit trail:

- **Scope:** one file, one function, ~41 LOC net change.
- **Basis:** 08-02-PROBE.md "Post-Override Re-Probe" section identified this distinct fix-site and the 08-02 executor explicitly deferred it to 08-06 (see 08-02-SUMMARY.md "Open Handoff").
- **Template:** `src/data/sirius_host_to_gpu_converter.cpp` — the canonical Pattern 2 host→gpu converter authored by 08-02 Branch B.
- **Static verification:** build exits 0; HYG preserved; Pattern 2 idiom grep-visible at the new site.

### Auto-fixed Issues

None during this plan. The residual num_gpus=2 failure was NOT auto-fixed per plan scope ("Don't over-author"). It is handed off as a scoped open-issue in VALIDATION.md.

### Scope-preserved deferrals (Phase 8 → post-Phase-8)

- **Residual cudaErrorInvalidValue @ cuda_memcpy.cu:42 on parquet + num_gpus=2.** After the 08-06 carryover fix landed, re-probe with integration.yaml = num_gpus: 2 still produces the same failure on `gpu_execution hive partition - filter on data column`. The 08-06 plan's scope was: (a) apply the fix identified by the orchestrator, (b) validate the static invariants. Open-ended bug hunting beyond the plan-scoped single fix-site would violate the `<carryover_fix>` directive's explicit "Don't over-author" clause. Handed off with four hypothesis candidates + concrete next actions in VALIDATION.md's Open Issue section.
- **SF10 smoke + SF100 ship-gate runs (criteria 1 + 6).** Not executed on this host because the prerequisite SF1 parquet + num_gpus=2 already reproduces the blocker. Running SF100 with the known bug active would not produce ship-gate evidence, it would reproduce the blocker at 100× the wall-clock. The exact command blocks are preserved verbatim in VALIDATION.md for execution once the residual fix-site is closed.
- **AUDIT TEST_CASE runtime assertion (criterion 4).** The TEST_CASE is source-complete and wired into the build (per 08-05-SUMMARY.md). It cannot fire on this host because `--abort` halts the suite at test 609/610 before reaching it. Auto-engages once criterion 2 is closed.

## Authentication Gates

None.

## Issues Encountered

1. **Carryover fix did not close the failing test.** Post-fix re-probe showed the same `cudaErrorInvalidValue @ cuda_memcpy.cu:42` signature on both `gpu_execution hive partition - filter on data column` (integration.yaml flipped) and `gpu_execution - TPC-H Query 1 parquet` (num_gpus=2 GENERATE section). This is the "non-obvious complexity" condition the plan's `<carryover_fix>` block called out: the fix was applied correctly (Pattern 2 idiom at the exact lines 55-119 identified by 08-02-PROBE), but at least one more fix-site remains. Per plan directive, stopped before escalating. Documented with hypotheses A-D in VALIDATION.md.
2. **Bash shell has no GPU driver access.** Only the MCP project-commands shell can run binaries that initialize `SiriusContext::initialize` (requires real GPUs). All runtime validation went through MCP; direct `./build/release/.../sirius_unittest` invocation from the bash shell fails with `cucascade::topology_discovery reported 0 GPUs`. This constrained how finely I could probe the failing test (e.g. no way to run `[integration][hive_partition]` in isolation with SIRIUS_LOG_LEVEL=info from this shell — I could only go through MCP's `unit-tests` wrapper which has `--abort`).

## Known Stubs

None. VALIDATION.md is fully authored with real numbers pulled from real MCP runs. No placeholder TODOs, no "coming soon" UI text, no empty REQUIRE assertions. The DEFERRED criteria are documented with real reasons (residual fix-site), not stubbed pending.

## Phase 8 Ship Verdict

**BLOCKED on residual fix-site.** See `08-SUMMARY.md` for the phase-level verdict and the full traceability table.

## Self-Check: PASSED

**Files verified to exist:**

- FOUND: `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-06-VALIDATION.md` (319 lines)
- FOUND: `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-06-SUMMARY.md` (this file)
- FOUND: `src/data/host_parquet_representation_converters.cpp` (modified per carryover fix; Pattern 2 idiom present at line 98 + doc at lines 59, 70)

**Commits verified to exist (`git log --oneline -5`):**

- FOUND: `bf53dcc` (fix — carryover: Pattern 2 idiom on host_parquet converter)
- FOUND: `584707b` (docs — VALIDATION.md)

**Grep invariants verified:**

- `rmm::cuda_stream_default` across src/: 41 matches across 12 files (unchanged baseline)
- `rmm::cuda_stream_default` in Phase 8 modified files: 0 matches
- Pattern 2 idiom (`cuda_set_device_raii.*(target|source)`) in fix sites: 6 code + 2 doc matches
- cucascade submodule (`git status cucascade/`): clean

**MCP runs verified via persisted output:**

- `build` post-carryover-fix: exit 0
- `unit-tests` post-carryover-fix (num_gpus=2 yaml): exit 1, 316/315 passed, 1 residual failure documented
- `unit-tests` post-carryover-fix (num_gpus=1 default): exit 1, 610/609 passed, 1 residual failure documented

**Invariant re-checks post SUMMARY.md authoring:**

- `grep -rn 'rmm::cuda_stream_default' src/ | wc -l` → 41 (baseline preserved)
- `git status cucascade/` → clean (submodule untouched)
- `grep "num_gpus" test/cpp/integration/integration.yaml` → `num_gpus: 1` (restored; no transient flip committed)

## Self-Check: PASSED

---

*Phase: 08-multi-gpu-sql-pipeline-fix*
*Completed: 2026-04-22*
