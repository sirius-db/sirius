---
phase: 23-update-cucascade-and-sirius-from-upstream
verified: 2026-05-12T21:00:00Z
status: gaps_found
score: 7/10 must-haves verified
gaps:
  - truth: "REG-05 [mgpu_stress] 500-iter PASS"
    status: failed
    reason: "cudaErrorInvalidValue at representation_converter.cpp:628 inside alloc_and_peer_copy_async host-staging HtoD cudaMemcpyAsync. Introduced by commit 8392c3d (Phase 23 Plan 02 rebase), which replaced cudf::pack/unpack with reconstruct_column_p2p→alloc_and_peer_copy_async in convert_gpu_to_gpu. Prior to Phase 23, [mgpu_stress] passed 500-iter in Phase 21, 22, 22.3. This is a Phase 23 regression, not a pre-existing limitation."
    artifacts:
      - path: "cucascade/src/data/representation_converter.cpp"
        issue: "Line 628 HtoD cudaMemcpyAsync in alloc_and_peer_copy_async lacks rmm::cuda_set_device_raii{dst_device} guard; invoked from reconstruct_column_p2p outside the convert_gpu_to_gpu outer target_guard scope"
    missing:
      - "Add rmm::cuda_set_device_raii{dst_device} around line 628 HtoD cudaMemcpyAsync in alloc_and_peer_copy_async (or pass device context through the call chain). Phase 24 fix."
  - truth: "REG-06 Leg 1: memcheck on [multi_gpu_foundation] 7/7 PASS, 0 violations"
    status: failed
    reason: "6/7 PASS; same root cause as REG-05. Test gpu_to_gpu round-trip preserves bytes on N>=2 hosts (MGPU-04 + MGPU-06) fails with cudaErrorInvalidValue at representation_converter.cpp:628. This test was 7/7 in Phase 21, 22, and 22.3."
    artifacts:
      - path: "cucascade/src/data/representation_converter.cpp"
        issue: "Same missing device context guard as REG-05 gap above"
    missing:
      - "Same fix as REG-05. Phase 24 fix."
  - truth: "GAUNTLET-23: All 17 invariant gates pass (REG-01..06, GATE-22.1-A/B/C, K.6, K.7, Cluster B, HYG-02, datasource_factory, tpch_sf10, mgpu-audit)"
    status: failed
    reason: "15/17 gates pass. REG-05 and REG-06 Leg 1 fail. REG-06 Leg 2 skipped due to Leg 1 failure. Gate failure is caused by the Phase 23 rebase introducing a new convert_gpu_to_gpu code path that is broken on this host's hardware."
    artifacts:
      - path: "cucascade/src/data/representation_converter.cpp"
        issue: "New convert_gpu_to_gpu column-walk path (commit 8392c3d) calls alloc_and_peer_copy_async without active dst device CUDA context"
    missing:
      - "Fix alloc_and_peer_copy_async HtoD context binding. Phase 24."
      - "Run REG-06 Leg 2 memcheck on [integration][gpu_execution][parquet][join] after fix to close the skipped gate."
      - "Update sanitizer_gate_22.sh to distinguish race findings from API-error backtraces (cluster_B false positive). Phase 24 gate maintenance."
---

# Phase 23: Update cucascade + sirius from upstream — Verification Report

**Phase Goal:** Rebase cucascade fork onto origin/main (PR #121 "Make host memory portable") + merge sirius origin/dev into feature/single-node-multi-gpu2 + verify no regression via Phase 22.x invariant gauntlet.
**Verified:** 2026-05-12T21:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Cucascade fork rebased onto bcddb89 (PR #121); 6 commits ahead; surgical split of 6236494 correct | VERIFIED | `git -C cucascade log --oneline origin/main..HEAD` = 6 commits at 1e889d7; 23-CUCASCADE-DIFF.md documents each commit's content |
| 2 | Sirius origin/dev merged into feature/single-node-multi-gpu2; 6 conflicts resolved; 12 upstream commits absorbed | VERIFIED | Merge commit 49b7b86 present in git log; 23-04-CONFLICT-LOG.md documents all 6 conflict resolutions with rationale |
| 3 | Build succeeds with zero new warnings | VERIFIED | State.md documents build green at post-merge head; intermediate gauntlet (Plan 23-03) confirms build clean |
| 4 | REG-01..04 and HYG-02 pass (multi-GPU, TPC-H, SF100 Q1, stream_default count) | VERIFIED | REG-01 16/16 79091 assertions; REG-02 22/22 36256 assertions; REG-03 49/49 71623 assertions; REG-04 3.048s byte-identical; HYG-02=40 all in src/legacy/ |
| 5 | REG-05 [mgpu_stress] 500-iter PASS | FAILED | 0/1 FAIL; cudaErrorInvalidValue at representation_converter.cpp:628; Phase 23 regression from commit 8392c3d |
| 6 | REG-06 Leg 1 memcheck on [multi_gpu_foundation] 7/7 PASS | FAILED | 6/7 FAIL; same root cause as REG-05 |
| 7 | GATE-22.1-A/B/C (kvikio-free, Cluster A=0, SF1 Q11 functional) preserved through merge | VERIFIED | bypass-grep=0 confirmed; cluster_A=0 confirmed; SF1 Q11 num_gpus=2 1/1 PASS 9011 assertions |
| 8 | K.6 and K.7 NO-REPRO status preserved | VERIFIED | K.6: SF100 Q11 exit 0, 0 cudaSetDevice(-1) errors; K.7: [tpch_sf10] 4/4 PASS including tpch_q11_sf10_2gpu test |
| 9 | Phase 22 Cluster B same-stream invariant holds (total_races=0) | VERIFIED | sanitizer total_races=0; cluster_B=1 in gate script is confirmed false positive (API-error backtraces, not race findings); Cluster B same-stream fix at 1e889d7 intact |
| 10 | 23-VERDICT.md and 23-CUCASCADE-DIFF.md written; ROADMAP/REQUIREMENTS/STATE updated | VERIFIED | All 5 files present; REQUIREMENTS.md traceability rows for MERGE-CC-23, MERGE-DEV-23, GAUNTLET-23 updated; ROADMAP.md Phase 23 row marked Complete (PARTIAL) |

**Score:** 7/10 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `cucascade/src/data/representation_converter.cpp` | P2P override + DMA probe + same-stream fix at 1e889d7 | WIRED (with regression) | Exists, substantive (383+ lines added by 8392c3d, 13 lines by 1e889d7); introduces REG-05/REG-06 breakage at line 628 |
| `.planning/phases/23.../23-VERDICT.md` | Phase verdict with pass/fail per gate | VERIFIED | Present; status=PARTIAL; all 17 gates documented |
| `.planning/phases/23.../23-CUCASCADE-DIFF.md` | CC-UPSTREAM-01 fork divergence doc, 6 commits | VERIFIED | Present; all 6 commits documented with SHA, files, rationale |
| `.planning/STATE.md` | Phase 23 complete; post-merge baseline | VERIFIED | Present; status=complete; stopped_at documents PARTIAL verdict |
| `.planning/ROADMAP.md` | Phase 23 Complete (PARTIAL); 5/5 plans checked | VERIFIED | Phase 23 row: "Complete (PARTIAL) 2026-05-12"; all 5 plan checkboxes present |
| `.planning/REQUIREMENTS.md` | MERGE-CC-23, MERGE-DEV-23, GAUNTLET-23 updated | VERIFIED | All 3 rows present in traceability table with Phase 23 disposition |
| `src/include/op/sirius_physical_partition.hpp` | Both set_min_num_partitions + no_history_peak_memory_estimate | VERIFIED | Lines 96 and 102 confirmed present |
| `src/sirius_engine.cpp` | drain_after_error + unfinalized-op warning + inserted_operators rename | VERIFIED | Lines 231/235/253 drain_after_error; line 454 inserted_operators |
| `src/op/scan/duckdb_scan_executor.cpp` | get_estimated_reservation_size_info + any_memory_space_in_tier_with_preference | VERIFIED | Lines 420/422 confirmed |
| `src/include/sirius_context.hpp` | thread header dropped; unordered_map/set/utility kept | VERIFIED | No #include <thread>; unordered_map line 46, unordered_set line 47 present |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| Cucascade fork commit 8392c3d | alloc_and_peer_copy_async host-staging HtoD | reconstruct_column_p2p call chain | BROKEN | Line 628 HtoD cudaMemcpyAsync executes outside dst-device context (target_guard at line 843 is in outer convert_gpu_to_gpu; inner call via reconstruct_column_p2p does not re-set device) |
| Phase 22 Cluster B same-stream fix (1e889d7) | alloc_and_peer_copy_async DtoH + HtoD ordering | Both on target_stream | WIRED | target_stream used for both copies; total_races=0 confirmed |
| origin/dev merge (49b7b86) | drain_after_error on success path | src/sirius_engine.cpp:253 | WIRED | drain_after_error preserved post-merge; Phase 22.2 UAF fix intact |
| GATE-22.1-A bypass-grep | 0 cudf::io::datasource::create calls in src/ | grep output | WIRED | Count=0 confirmed live |
| Cucascade gitlink | Post-rebase HEAD 1e889d7 | git submodule status | WIRED | SHA matches 23-CUCASCADE-DIFF.md; matches 23-VERDICT.md frontmatter |

---

### Data-Flow Trace (Level 4)

Not applicable. This is a merge/rebase phase — no new user-visible data-rendering artifacts. The relevant data flows are correctness chains through CUDA runtime calls verified by test execution and grep gates.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Cucascade gitlink is post-rebase SHA | git submodule status cucascade | 1e889d7e...cucascade (heads/fix/pinned-portable-flags) | PASS |
| 6 commits ahead of upstream bcddb89 | git -C cucascade log --oneline origin/main..HEAD | 6 commits listed (1e889d7, 89d6a3f, 085d917, 8392c3d, 0c0a4af, 9a23f4f) | PASS |
| HYG-02: rmm::cuda_stream_default count=40 | grep -rn rmm::cuda_stream_default src/ | 40 (all in src/legacy/) | PASS |
| GATE-22.1-A: kvikio bypass count=0 | grep -rn cudf::io::datasource::create\|source_info{ src/ | 0 | PASS |
| drain_after_error on success path | grep -n drain_after_error src/pipeline/task_scheduler.cpp | line 203 | PASS |
| CTE _types validator present | grep -n column count mismatch src/pipeline/gpu_pipeline_task.cpp | line 57 | PASS |
| SF10 Q11 regression test present | grep -n tpch_q11_sf10_2gpu test/cpp/integration/test_gpu_execution_tpch.cpp | line 4415 | PASS |
| downgrade_executor tier gate intact | grep src/downgrade/downgrade_executor.cpp | lines 79/89/182 | PASS |
| Both partition methods in sirius_physical_partition.hpp | grep set_min_num_partitions + no_history_peak_memory_estimate | lines 96 + 102 | PASS |
| Sirius is 0 commits behind origin/dev | git log HEAD..origin/dev | 0 | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| MERGE-CC-23 | 23-02-PLAN.md | Cucascade fork rebased onto bcddb89 (PR #121) with surgical split of 6236494 | PARTIAL | Rebase structurally correct; 6 commits ahead of bcddb89; surgical split confirmed by 23-CUCASCADE-DIFF.md. Partial because commit 8392c3d introduces a regression on this host's hardware (broken-peer-DMA path in convert_gpu_to_gpu). The rebase work itself is complete; the regression is a functional gap. |
| MERGE-DEV-23 | 23-04-PLAN.md | Sirius origin/dev merged into feature/single-node-multi-gpu2 | SATISFIED | Merge commit 49b7b86; 6 conflicts resolved per 23-04-CONFLICT-LOG.md; 12 upstream commits absorbed; all invariant grep gates preserved post-merge; build clean |
| GAUNTLET-23 | 23-05-PLAN.md | Full Phase 22.x invariant gauntlet passes post-merge | BLOCKED | 15/17 gates PASS. REG-05 [mgpu_stress] and REG-06 Leg 1 [multi_gpu_foundation] FAIL. REG-06 Leg 2 skipped. The 2 failing gates regressed from the prior baseline. |

**Orphaned requirements check:** REQUIREMENTS.md shows MERGE-CC-23, MERGE-DEV-23, and GAUNTLET-23 are the only Phase 23 requirements. All 3 are accounted for. No orphaned IDs.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| cucascade/src/data/representation_converter.cpp | 628 | `cudaMemcpyAsync` HtoD issued without active dst-device CUDA context | BLOCKER | Causes cudaErrorInvalidValue in [mgpu_stress] and [multi_gpu_foundation] on broken-peer-DMA hardware. Was not present in prior cucascade pin (c666b21). |

---

### Regression Triage: Is This a Phase 23 Regression or a Carry-Forward?

**Assessment: Phase 23 regression — correctly deferred to Phase 24.**

The triage rule is: "a phase that ships breakage where there was none before is a regression even if the user requested it; a phase that exposes a pre-existing limitation in a new code path may be carry-forward."

Applying that rule:

- Before Phase 23 (at c666b21), [mgpu_stress] 500-iter PASSED in Phase 21, 22, and 22.3. The failure is new.
- The breakage was introduced by commit 8392c3d (Phase 23 Plan 02 rebase), which changed `convert_gpu_to_gpu` from cudf::pack/unpack to `reconstruct_column_p2p → alloc_and_peer_copy_async`.
- The hardware limitation (broken peer DMA on 2 × RTX 6000 Ada) pre-existed Phase 23 — it was empirically probed and documented. But `alloc_and_peer_copy_async` had a working host-staging path (Phase 22 fixed its stream-ordering). The new failure is not the hardware limitation itself; it is a missing device-context guard at the HtoD step when called from the new code path.
- The cudf::pack/unpack → column-walk migration was a deliberate architectural change accepted as part of the cucascade PR #121 rebase. The missing `rmm::cuda_set_device_raii{dst_device}` guard is a latent bug in the new code path that only manifests when `reconstruct_column_p2p` is called from within `convert_gpu_to_gpu` (not from the outer function where `target_guard` is in scope).

**Verdict:** This is a Phase 23 regression. The phase shipped breakage where there was none before. The carry-forward framing in 23-VERDICT.md is accurate for the fix location (it is a single-line fix best done in a focused Phase 24 plan), but the classification of the gap must acknowledge it is a regression, not merely an exposed pre-existing limitation.

The user's question "should it be punted to Phase 24?" has the answer: punting the fix is acceptable given the surgical nature of the fix and the fact that all other 15 gates pass, but the `gaps_found` status is correct — the phase did not achieve its full gauntlet goal.

---

### Human Verification Required

None. All automated checks are definitive. The failing gates are reproduced with consistent error messages across 3 runs (documented in 23-VERDICT.md Section E).

---

### Gaps Summary

**Two related gaps, one root cause.**

REG-05 ([mgpu_stress] 500-iter) and REG-06 Leg 1 ([multi_gpu_foundation] gpu_to_gpu round-trip) both fail because Phase 23 Plan 02's cucascade rebase (commit 8392c3d) introduced a new `convert_gpu_to_gpu` implementation that calls `alloc_and_peer_copy_async` via `reconstruct_column_p2p`. On this host (2 × RTX 6000 Ada, peer DMA broken in both directions), the host-staging path executes — and the HtoD `cudaMemcpyAsync` at line 628 fails with `cudaErrorInvalidValue` because no `cuda_set_device_raii{dst_device}` is active at that call site (the outer `target_guard` at line 843 of `convert_gpu_to_gpu` is not in scope when `reconstruct_column_p2p` recursively calls `alloc_and_peer_copy_async`).

Prior to Phase 23, `[mgpu_stress]` and the `gpu_to_gpu round-trip` test both passed. The regression is confirmed. The minimal fix is adding a `rmm::cuda_set_device_raii{dst_device_id}` guard inside `alloc_and_peer_copy_async` before the HtoD `cudaMemcpyAsync` at line 628 (Phase 24).

**What passes:** All 15 other invariant gates hold — REG-01..04, [datasource_factory], [tpch_sf10], [mgpu-audit], GATE-22.1-A/B/C, K.6, K.7 NO-REPRO, Phase 22 Cluster B same-stream (total_races=0), HYG-02=40. The origin/dev merge is structurally complete and correct. The cucascade gitlink is correctly pinned to 1e889d7. Side-benefit confirmed: upstream commit 7cc7a79 closed the Phase 22.3 pin_table suite-run flake.

**What is deferred:** REG-06 Leg 2 memcheck on [integration][gpu_execution][parquet][join] was not run (skipped after Leg 1 failure). After the Phase 24 fix, Leg 2 must be run to close the REG-06 gate. The `sanitizer_gate_22.sh` false positive (cluster_B=1 when total_races=0) is a gate-script maintenance task for Phase 24.

---

_Verified: 2026-05-12T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
