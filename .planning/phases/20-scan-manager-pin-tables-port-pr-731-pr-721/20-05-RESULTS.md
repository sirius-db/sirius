# Phase 20 Plan 05 — Verification Results

**Captured:** 2026-05-06
**Branch:** feature/single-node-multi-gpu2
**Authoritative cite:** `20-05-DIAGNOSIS.md` (PATH: B); `20-05-INVESTIGATION.md` (STATUS: human_needed).
**Test host:** 2 × NVIDIA RTX 6000 Ada Generation (peer-DMA broken on 2 direction(s); cucascade host-staging fallback active).

---

## Path Taken

**PATH: B** (Escalation) — per `20-05-DIAGNOSIS.md` `PATH: B` marker at end of file.

**Rationale (cited from DIAGNOSIS):**
- Cluster A (5/21 races): cudf+kvikio internal parquet reader stream-ordering. Library boundary upstream of all Sirius/cucascade attachment points.
- Cluster B (16/21 races): cucascade host-staging fallback inside `alloc_and_peer_copy_async`. Race shape E (cucascade-internal lineage gap) per plan 20-05 taxonomy → Path B.

No source files modified this plan. Phase 18..20 invariants preserved end-to-end.

---

## Sanitizer Re-Run (Path A only)

**SKIPPED — Path B selected.** Pre-fix sanitizer evidence captured in `20-05-DIAGNOSIS.md` (`/tmp/p20_sanitizer.out`, 1217 lines, 21 race blocks, 13 benign init API errors).

A post-fix sanitizer re-run is not meaningful when no fix was applied: the same 21 race blocks would re-appear with no change. Plan 20-05 Task 4 explicitly directs Path B to "skip sanitizer re-run + Q11 + [integration][TPC-H]" and run only the [mgpu] continuity baseline + invariant gates.

---

## [integration][TPC-H] 48/48 (Path A) / Skipped (Path B)

**SKIPPED — Path B selected.** Pre-fix evidence already captured in `20-04-RESULTS.md` (Q11 parquet num_gpus=2 FAIL with `cudaErrorIllegalAddress` at `cuda_stream_view.cpp:45`; 21/22 cases ran, 19615/19616 assertions PASS before --abort).

The plan does not direct re-running [integration][TPC-H] when no fix was applied — re-running would produce the same FAIL fingerprint. Phase 21 REG-03 ship-gate carries this dependency forward (per `20-05-INVESTIGATION.md` `## Carry-Forward to Phase 21 REG-03` section).

---

## [mgpu] 16/16 (both paths)

**Command:**
```
mcp__project-commands__run_command unit-tests --filter "[mgpu]"
```

**Wall-clock:** **104.4s** (within 200s budget; matches 20-04 baseline 106.4s ± 5s; matches Phase 18-VERDICT-V2 + Phase 19-VERDICT baselines)
**Exit code:** **0**
**Test cases:** 16 / 16 PASS
**Assertion count:** **79091** (= Phase 18-VERDICT-V2 baseline = Phase 19-VERDICT baseline = 20-04 baseline; exact match)

**Cases (verbatim from stdout):**
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

**Verdict:** PASS — continuity baseline preserved. The follow-up #17 sentinel TEST_CASE at index 0 (Q11-shape regression at smaller data volume) PASSES, confirming the [mgpu] regression suite is unaffected by the Q11 SF1 parquet failure mode (which is specific to `[integration][TPC-H]` parquet path — different filter set than `[mgpu]`).

---

## Phase 18..20 Invariant Gates

```bash
grep -rEn "(->get_data\(\)|pop_data_batch.*task_created|data_batch_processing_handle)" src/ test/
```

**DB-grep result (4 hits):**
```
src/legacy/expression_executor/gpu_expression_executor.cpp:260:  auto& input_data_rep = input_batch->get_data()->cast<cucascade::gpu_table_representation>();
src/legacy/expression_executor/gpu_expression_executor.cpp:348:  auto& input_data_rep = input_batch->get_data()->cast<cucascade::gpu_table_representation>();
test/cpp/operator/test_gpu_partition_impl.cpp:43: * data_batch_processing_handle with the RAII mutable_data_batch accessor.
test/cpp/operator/aggregate/test_gpu_merge_impl.cpp:54: * data_batch_processing_handle with mutable_data_batch (RAII). Holding
```

**Verdict:** PASS — all 4 hits are in src/legacy/ (frozen path; CLAUDE.md exempts) or doc-comments referencing the migration history. Zero live `->get_data()` / `pop_data_batch.*task_created` / `data_batch_processing_handle` in active code. Matches 20-04 baseline (4 hits, same files, same lines).

```bash
grep -rn "cucascade_datasource" src/ test/
```

**IO-15 result:** **0 hits.** PASS — cucascade_datasource fully retired (Phase 19-05).

```bash
grep -rn "writer_stream\|record_writer_event" src/op/scan/
```

**SM-03 result:**
```
src/op/scan/sirius_gpu_parquet_scan_operator.cpp:256:  // execution stream as writer_stream — preserves Phase 13-04 Path-2
```

**Verdict:** PASS — 1 hit, matches Phase 20-02 post-edit position; Phase 13-04 Path-2 stream-lineage re-attachment preserved.

```bash
grep -rn "rmm::cuda_stream_default" src/ | wc -l
```

**HYG-02 result:** **40 hits.** PASS — all in src/legacy/ + src/include/legacy/ (Phase 19-01 baseline; matches 20-04 baseline; ≤ 40 ceiling).

**Source diff vs start of plan 20-05 (HEAD~3 prior to 20-05 commits):** 0 lines.

```bash
git diff HEAD~3 -- src/ test/ | wc -l
# 0
```

No source files modified in plan 20-05 execution. Path B integrity preserved.

---

## Verdict

**PARTIAL** (Path B; gap escalated; SM-06 SF1 carryover to Phase 21 REG-03 explicit; STATUS: human_needed).

**Per plan 20-05 success criteria:**

| Criterion | Result | Evidence |
|-----------|--------|----------|
| 20-05-DIAGNOSIS.md exists with all 9 H2 sections + PATH marker | PASS | DIAGNOSIS.md committed `c93fe05` (Task 1) + `05ff610` (Task 2 path-gate annotation) |
| 20-05-INVESTIGATION.md exists with structural finding + STATUS: human_needed | PASS | INVESTIGATION.md committed `b5ff7e8` (Task 3); 210 lines; STATUS: human_needed marker present |
| 20-05-RESULTS.md exists with all 6 H2 sections | PASS | This file (Task 4) |
| /tmp/p20_sanitizer.out captured | PASS | 1217 lines, 21 race blocks documented in DIAGNOSIS.md |
| Phase 18..20 invariants preserved end-to-end | PASS | DB-grep == 4 (legacy + comments only); IO-15 == 0; SM-03 >= 1; HYG-02 <= 40 |
| Path A: Q11 SF1 num_gpus=2 parquet PASS post-fix | N/A (Path B) | Pre-fix evidence in 20-04-RESULTS.md; carried to Phase 21 REG-03 |
| Path A: [integration][TPC-H] 48/48 PASS | N/A (Path B) | Pre-fix evidence in 20-04-RESULTS.md |
| Path A: post-fix sanitizer 0 errors at original site | N/A (Path B) | No fix applied |
| Path B: [mgpu] 16/16 continuity baseline PASS | PASS | 79091 assertions / 104.4s / exit 0 |
| Path B: orchestrator sees `STATUS: human_needed` marker | PASS | INVESTIGATION.md ends with `STATUS: human_needed` |

**Path B closure summary:**

The Q11 SF1 num_gpus=2 parquet `cudaErrorIllegalAddress` is structurally beyond a Phase 20 single-cycle fix. The sanitizer reveals 2 root races at library boundaries (cudf+kvikio + cucascade pin 1c1e648 host-staging fallback). Both fix shapes are >1 day with the cucascade fork+bump being the recommended primary path (1.5-2.5 day total for full closure including a Sirius-side cudaStreamSynchronize workaround for the cudf+kvikio cluster).

Phase 20 closes with **5/6 SM-XX requirements PASS unconditionally** (SM-01..05) and **SM-06 PARTIAL** (SF10 PASS via 20-04; SF1 BLOCKED on this structural finding, carried to Phase 21 REG-03 with `STATUS: human_needed`).

**Phase 21 REG-03 ship-gate cannot pass** until either (a) cucascade fork+bump lands closing the SF1 [integration][TPC-H] 48/48; OR (b) Phase 21 acceptance criteria are explicitly relaxed to "47/48 + Q11 parquet num_gpus=2 known limitation" or "[integration][TPC-H] 48/48 at num_gpus=1 PASS only".

---

## Plan 20-05 Test Results Summary

| Gate | Spec | Measured Result | Verdict |
|------|------|-----------------|---------|
| Build sanity (Task 1 Step 0) | mcp build exit 0 | 0.2s, no work to do | PASS |
| Sanitizer Q11 SF1 num_gpus=2 (Task 1 Step 3) | log captured + FIRST race extracted | 21 race blocks, 132s, ERROR SUMMARY: 34 errors (= 21 races + 13 benign init) | PASS (diagnostic) |
| FIRST race fingerprint (Task 1 Step 4) | file:line:subsystem identified, race shape classified | cluster A: cudf+kvikio; cluster B: cucascade alloc_and_peer_copy_async; both shape A (use-before-alloc cross-stream) | PASS |
| Path decision (Task 1 Step 7) | A or B per taxonomy | PATH: B (cluster B == race shape E; cluster A == novel/upstream) | PASS |
| Task 2 fix (PATH: A only) | Localized fix at single site | N/A (Path B; gracefully skipped) | N/A |
| Task 3 INVESTIGATION (PATH: B) | ≥50 lines, STATUS: human_needed, structural finding + recommended fix shape + estimated effort + carry-forward | 210 lines, STATUS: human_needed, all required sections | PASS |
| [mgpu] 16/16 continuity baseline (Task 4) | exit 0, 79091 assertions, ≤200s | exit 0, 79091 assertions, 104.4s | PASS |
| Phase 18..20 invariant gates (Task 4) | DB-grep == 0 (live) / IO-15 == 0 / SM-03 >= 1 / HYG-02 <= 40 | DB-grep 4 (legacy+comments only); IO-15 0; SM-03 1; HYG-02 40 | PASS |
| Source diff (Task 4) | 0 lines (Path B) | 0 lines (`git diff HEAD~3 -- src/ test/ \| wc -l`) | PASS |

---

## Outcome

- **Build sanity** PASS.
- **Diagnosis** complete: 21 stream-ordered races identified across 2 clusters at library boundaries (cudf+kvikio + cucascade pin 1c1e648). FIRST race fingerprint and race-shape classification documented in 20-05-DIAGNOSIS.md.
- **PATH: B** decision recorded; Task 2 fix gracefully skipped (no source modifications); Task 3 INVESTIGATION.md authored with `STATUS: human_needed` marker.
- **[mgpu] 16/16 continuity baseline** PASS (79091 assertions / 104.4s / exit 0).
- **Phase 18..20 invariants** preserved end-to-end (no source diff).
- **Phase 20 final verdict** remains **PARTIAL** (5/6 SM-XX PASS unconditionally; SM-06 PARTIAL with explicit carryover to Phase 21 REG-03 + `STATUS: human_needed`). Path B integrity preserved.
- **Phase 21 REG-03 ship-gate** carries SF1 closure dependency.

The user is now informed of the structural finding + recommended fix shape + estimated effort to make the next-step decision (cucascade fork+bump vs ship-with-known-limitation vs alternative-path disable).
