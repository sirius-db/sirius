---
phase: 23-update-cucascade-and-sirius-from-upstream
verified: 2026-05-13T12:00:00Z
status: passed
score: 10/10 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 7/10
  gaps_closed:
    - "REG-05 [mgpu_stress] 500-iter PASS — closed by Plan 23-06 (dst_guard fix 37df815) + Plan 23-07 (probe-device-restore fix 9da4047)"
    - "REG-06 Leg 1 [multi_gpu_foundation] 7/7 PASS — closed by same two cucascade commits"
    - "GAUNTLET-23 all 17 gates PASS — REG-06 Leg 2 first-run PASS 42/42; sanitizer_gate_22.sh cluster_B false-positive closed by windowed awk counter"
  gaps_remaining: []
  regressions: []
---

# Phase 23: Update cucascade + sirius from upstream — Verification Report (Re-verification)

**Phase Goal:** Rebase cucascade fork onto origin/main (PR #121 "Make host memory portable") + merge sirius origin/dev into feature/single-node-multi-gpu2 + verify no regression via Phase 22.x invariant gauntlet.
**Verified:** 2026-05-13T12:00:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure (Plans 23-06 + 23-07)
**Sirius HEAD at verification:** 7062477
**Cucascade gitlink at verification:** 9da404756a8354d84d1dcd6bf3f3b46c29abfb3e

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Cucascade fork rebased onto bcddb89 (PR #121); surgical split of 6236494 correct; 8 commits ahead | VERIFIED | `git -C cucascade log --oneline origin/main..HEAD` = 8 commits at 9da4047; 23-CUCASCADE-DIFF.md documents all 8 commits including gap-closure commits 37df815 + 9da4047 |
| 2 | Sirius origin/dev merged into feature/single-node-multi-gpu2; 6 conflicts resolved; 12 upstream commits absorbed | VERIFIED | Merge commit 49b7b86 present in git log; 23-04-CONFLICT-LOG.md documents all 6 conflict resolutions |
| 3 | Build succeeds with zero new warnings | VERIFIED | Plan 23-07 smoke build [128/128] after bump to 9da4047; sirius_unittest linked; no new warnings |
| 4 | REG-01..04 and HYG-02 pass (multi-GPU, TPC-H, SF100 Q1, stream_default count) | VERIFIED | REG-01 16/16 79091 assertions; REG-02 22/22 36256 assertions; REG-03 49/49 71623 assertions; REG-04 3.048s byte-identical; HYG-02=40 (all in src/legacy/, 0 in active src/) |
| 5 | REG-05 [mgpu_stress] 500-iter PASS | VERIFIED | 1/1 PASS 77053 assertions 83.7s exit 0 — closed by 37df815 + 9da4047; stderr confirms host-staging path taken on broken-peer-DMA hardware; no cudaErrorInvalidValue |
| 6 | REG-06 Leg 1 [multi_gpu_foundation] 7/7 PASS; Leg 2 [parquet][join] memcheck 42/42 PASS | VERIFIED | Leg 1: 7/7 functional 38 assertions exit 0; Leg 2: 42/42 PASS 1,922,202 assertions 0 new violations; same root cause fix as REG-05 |
| 7 | GATE-22.1-A/B/C (kvikio-free, Cluster A=0, SF1 Q11 functional) preserved through merge | VERIFIED | GATE-22.1-A bypass-grep=0 (with exclusion filter, confirmed at verification time); cluster_A=0; SF1 Q11 1/1 9011 assertions |
| 8 | K.6 and K.7 NO-REPRO status preserved | VERIFIED | K.6: SF100 Q11 exit 0, 0 cudaSetDevice(-1) errors; K.7: [tpch_sf10] 4/4 64 assertions including tpch_q11_sf10_2gpu at test_gpu_execution_tpch.cpp:4415 |
| 9 | Phase 22 Cluster B same-stream invariant holds; sanitizer_gate_22.sh cluster_B=0 | VERIFIED | cluster_B=0 (windowed awk counter); cluster_A=0; total_races=0; P22_SELFTEST=1 exits 0 (verified at verification time); false positive root cause documented in Section J of 23-VERDICT.md |
| 10 | 23-VERDICT.md status:PASS; ROADMAP/REQUIREMENTS/STATE updated; all 3 REQ rows Complete | VERIFIED | 23-VERDICT.md status=PASS 17/17 gates; REQUIREMENTS.md: MERGE-CC-23, MERGE-DEV-23, GAUNTLET-23 all "Complete"; ROADMAP.md Phase 23 row "Complete 7/7 2026-05-13" |

**Score:** 10/10 truths verified

---

## Independent Check Results

The following checks were performed independently at sirius HEAD 7062477, cucascade gitlink 9da4047:

| Check | Command | Expected | Actual | Status |
|-------|---------|----------|--------|--------|
| Sirius gitlink = 9da4047 | `git submodule status cucascade` | 9da404756a8354d84d1dcd6bf3f3b46c29abfb3e | 9da404756a8354d84d1dcd6bf3f3b46c29abfb3e | PASS |
| Cucascade branch tip = 9da4047 | `git -C cucascade log --oneline -1 fix/pinned-portable-flags` | 9da4047 | 9da4047 | PASS |
| 37df815 is parent of 9da4047 | `git -C cucascade log --oneline 9da4047~1 -1` | 37df815 | 37df815 | PASS |
| dst_guard exists in alloc_and_peer_copy_async | grep -n dst_guard representation_converter.cpp | line 646 | line 646: `rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}};` | PASS |
| peer-DMA path cudaMemcpyPeerAsync exists | grep -n cudaMemcpyPeerAsync representation_converter.cpp | present | line 605: `CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(` | PASS |
| src_guard at ~619 | grep -n src_guard representation_converter.cpp | line ~619 | line 619: `rmm::cuda_set_device_raii src_guard{rmm::cuda_device_id{src_device}};` | PASS |
| rmm::cuda_stream_default count in representation_converter.cpp = 0 | grep -c rmm::cuda_stream_default cucascade/.../representation_converter.cpp | 0 | 0 | PASS |
| P22_SELFTEST=1 exits 0 | `P22_SELFTEST=1 bash test/scripts/sanitizer_gate_22.sh` | SELFTEST PASS exit 0 | SELFTEST PASS: windowed cluster_B counter is correct | PASS |
| HYG-02: total cuda_stream_default in src/+legacy/ = 40 | grep -rE rmm::cuda_stream_default src/ legacy/ | 40 | 40 | PASS |
| HYG-02: 0 in active src/ | grep -rE rmm::cuda_stream_default src/ | 0 | 0 | PASS |
| kvikio-free (with exclusion filter) | grep -rn 'cudf::io::datasource::create\|cudf::io::source_info{' src/ | grep -v 'data_source.get()\|datasource.get()' | 0 | 0 | PASS |
| No push of gap-closure commits | git show-ref origin/feature/single-node-multi-gpu2 | d573fc4 (pre-gap) | d573fc49f4dd1566741236b9a323e75b9d0872aa (= old merge commit, NOT 7062477) | PASS — not pushed |
| REQUIREMENTS.md MERGE-CC-23 Complete | grep MERGE-CC-23 REQUIREMENTS.md | Complete | Complete | PASS |
| REQUIREMENTS.md MERGE-DEV-23 Complete | grep MERGE-DEV-23 REQUIREMENTS.md | Complete | Complete | PASS |
| REQUIREMENTS.md GAUNTLET-23 Complete | grep GAUNTLET-23 REQUIREMENTS.md | Complete | Complete | PASS |
| 23-VERDICT.md status:PASS | grep status 23-VERDICT.md frontmatter | PASS | PASS | PASS |
| Sirius HEAD = 7062477 | git rev-parse HEAD | 7062477 | 7062477d5ff116d6cd3aa5d0953b5aa49bf2326c | PASS |

**Note on kvikio check:** Three `cudf::io::source_info{...}` occurrences exist in src/ but all pass a `datasource.get()` or `data_source.get()` pointer argument — a safe pre-materialized datasource*, not a file path. The exclusion filter `grep -v 'data_source\.get()\|datasource\.get()'` correctly eliminates all three. The GATE-22.1-A gate (as documented in sanitizer_gate_22.sh line 206) uses this same exclusion filter. Count after exclusion = 0.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `cucascade/src/data/representation_converter.cpp` | dst_guard at line 646; src_guard at line 619; peer-DMA path via cudaMemcpyPeerAsync; 0 cuda_stream_default | VERIFIED | All 4 checks pass independently at verification time |
| `.planning/phases/23.../23-VERDICT.md` | status:PASS; 17/17 gates; gap closure sections E/F.2/F.3/J present | VERIFIED | Frontmatter status=PASS; gap_closure_date=2026-05-13; 17 gate rows all PASS |
| `.planning/phases/23.../23-CUCASCADE-DIFF.md` | fork_head=9da4047; commits_ahead=8; commits 7+8 (37df815 + 9da4047) documented | VERIFIED | frontmatter fork_head=9da4047, commits_ahead=8; both gap-closure commits in graph |
| `test/scripts/sanitizer_gate_22.sh` | windowed awk cluster_B counter; P22_SELFTEST mode; Pitfall 7 comment | VERIFIED | windowed awk at lines 177-185; P22_SELFTEST block at lines 90-119; Pitfall 7 comment at lines 33-43; selftest exits 0 confirmed |
| `.planning/REQUIREMENTS.md` | MERGE-CC-23/MERGE-DEV-23/GAUNTLET-23 all "Complete" | VERIFIED | All 3 rows at lines 158-160 say Complete with post-gap-closure detail |
| `src/include/op/sirius_physical_partition.hpp` | Both set_min_num_partitions + no_history_peak_memory_estimate | VERIFIED | Confirmed present in initial verification (lines 96 + 102); no merge conflict affected this file |
| `src/sirius_engine.cpp` | drain_after_error + unfinalized-op warning + inserted_operators rename | VERIFIED | drain_after_error line 203 in task_scheduler.cpp confirmed; no regression from gap-closure commits |
| `src/op/scan/duckdb_scan_executor.cpp` | get_estimated_reservation_size_info + any_memory_space_in_tier_with_preference | VERIFIED | Confirmed present in initial verification; gap-closure touches only cucascade and sanitizer script |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| cucascade commit 37df815 | HtoD cudaMemcpyAsync in alloc_and_peer_copy_async | rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}} at line 646 | WIRED | grep confirmed at verification time: line 646 present, wraps HtoD path in host-staging branch |
| cucascade commit 9da4047 | run_p2p_probe_locked device context restore | cudaGetDevice saved_device + cudaSetDevice(saved_device) on exit | WIRED | Commit subject and 23-07-SUMMARY.md Task 1 document the fix; REG-05 + Leg 1 7/7 PASS confirm correctness |
| Sirius gitlink bump (commits 15c47f5 + 5c554d1) | cucascade 9da4047 | git submodule record in index | WIRED | `git submodule status cucascade` = 9da4047... confirmed at verification time |
| sanitizer_gate_22.sh windowed awk | cluster_B counter | awk state machine in_race tracking | WIRED | Script lines 177-185 confirmed; P22_SELFTEST=1 exit 0 confirms state machine correctness |
| Phase 22 Cluster B same-stream fix (1e889d7) | alloc_and_peer_copy_async DtoH + HtoD ordering | Both on target_stream | WIRED | Still present at line 619 (src_guard) and 646 (dst_guard); total_races=0 in 23-07 gauntlet |

---

### Data-Flow Trace (Level 4)

Not applicable. This is a merge/rebase phase — no new user-visible data-rendering artifacts. The relevant data flows are correctness chains through CUDA runtime calls verified by test execution and grep gates.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Cucascade gitlink is post-gap-closure SHA 9da4047 | `git submodule status cucascade` | 9da404756a8354d84d1dcd6bf3f3b46c29abfb3e cucascade (heads/fix/pinned-portable-flags) | PASS |
| 8 commits ahead of upstream bcddb89 | `git -C cucascade log --oneline origin/main..HEAD` | 8 commits: 9da4047, 37df815, 1e889d7, 89d6a3f, 085d917, 8392c3d, 0c0a4af, 9a23f4f | PASS |
| dst_guard in alloc_and_peer_copy_async | grep -n dst_guard representation_converter.cpp | line 646 present | PASS |
| HYG-02: rmm::cuda_stream_default total=40 | grep -rE rmm::cuda_stream_default src/ legacy/ | 40 | PASS |
| HYG-02: 0 in active src/ | grep -rE rmm::cuda_stream_default src/ | 0 | PASS |
| GATE-22.1-A kvikio bypass=0 | grep with datasource.get() exclusion filter | 0 | PASS |
| P22_SELFTEST windowed awk correct | P22_SELFTEST=1 bash test/scripts/sanitizer_gate_22.sh | SELFTEST PASS exit 0 | PASS |
| REG-05 [mgpu_stress] 77053 assertions | Plan 23-07 gauntlet Leg A | 1/1 PASS 77053 assertions 83.7s | PASS |
| REG-06 Leg 1 [multi_gpu_foundation] 7/7 | Plan 23-07 gauntlet Leg B | 7/7 PASS 38 assertions 5.7s | PASS |
| REG-06 Leg 2 [parquet][join] memcheck 42/42 | Plan 23-07 gauntlet Leg C | 42/42 PASS 1,922,202 assertions 0 new violations | PASS |
| Sirius has not been pushed beyond origin ref | git show-ref origin/feature/single-node-multi-gpu2 | d573fc4 (pre-gap-closure merge commit) | PASS |
| Sirius HEAD = 7062477 | git rev-parse HEAD | 7062477 | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| MERGE-CC-23 | 23-02-PLAN.md | Cucascade fork rebased onto bcddb89 (PR #121) with surgical split; convert_gpu_to_gpu regression closed | SATISFIED | Rebase structurally correct (8 commits ahead of bcddb89); surgical split confirmed by 23-CUCASCADE-DIFF.md; dst_guard + probe-restore fix (37df815 + 9da4047) close the REG-05/REG-06 regression |
| MERGE-DEV-23 | 23-04-PLAN.md | Sirius origin/dev merged into feature/single-node-multi-gpu2 | SATISFIED | Merge commit 49b7b86; 6 conflicts resolved per 23-04-CONFLICT-LOG.md; 12 upstream commits absorbed; build clean; all invariant grep gates preserved |
| GAUNTLET-23 | 23-05-PLAN.md | Full Phase 22.x invariant gauntlet passes post-merge | SATISFIED | 17/17 gates PASS per 23-VERDICT.md (updated 2026-05-13); REG-05 + REG-06 Leg 1 + Leg 2 all green; sanitizer_gate_22.sh cluster_B=0 with windowed awk |

**Orphaned requirements check:** REQUIREMENTS.md shows MERGE-CC-23, MERGE-DEV-23, and GAUNTLET-23 are the only Phase 23 requirements. All 3 are accounted for. No orphaned IDs.

---

### Anti-Patterns Found

None at re-verification. The BLOCKER anti-pattern from initial verification (missing dst_guard at line 628 in alloc_and_peer_copy_async) was closed by Plan 23-06 commit 37df815. The `dst_guard` now wraps the HtoD branch at line 646. The `src_guard` at line 619 remains intact (unchanged from Phase 22 D-07 fix). No new anti-patterns introduced by the gap-closure commits.

---

### Human Verification Required

None. All automated checks are definitive. The failing gates from initial verification are now reproducibly passing with documented assertion counts, wall-clocks, and exit codes.

---

## Re-verification Summary

**Initial verification (2026-05-12):** 7/10 — Truths 5, 6, 9 failed. Root cause: Phase 23 Plan 02 cucascade rebase (commit 8392c3d) changed `convert_gpu_to_gpu` to use `alloc_and_peer_copy_async` via `reconstruct_column_p2p`, but the HtoD branch at line 628 lacked a `cuda_set_device_raii{dst_device}` guard. On 2 x RTX 6000 Ada hardware (peer DMA broken on both directions), the host-staging path executes and the missing guard caused `cudaErrorInvalidValue`. Additionally, `run_p2p_probe_locked` ended with a hardcoded `cudaSetDevice(0)` that clobbered the caller's RAII device guard. The sanitizer_gate_22.sh cluster_B counter was a flat grep that matched API-error backtraces as well as race findings.

**Gap-closure (Plans 23-06 + 23-07, 2026-05-13):**
- Plan 23-06 (cucascade commit 37df815): Added `rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}}` around the HtoD `cudaMemcpyAsync` + `cudaStreamSynchronize` pair in `alloc_and_peer_copy_async`. Closes REG-05 root cause.
- Plan 23-07 deviation (cucascade commit 9da4047): Fixed `run_p2p_probe_locked` to save and restore device context (`cudaGetDevice` + `cudaSetDevice(saved_device)`), closing the second bug exposed after the first fix. Closes REG-06 Leg 1 root cause.
- Plan 23-07 Task 2 (sirius commit 0a3e2a7): Replaced flat grep with windowed awk state machine in `sanitizer_gate_22.sh`; added P22_SELFTEST=1 synthetic log validation. Closes cluster_B false-positive.
- Sirius gitlink bumped to 9da4047 (commits 15c47f5 + 5c554d1 on feature/single-node-multi-gpu2).

**Re-verification result:** 10/10 — All 3 previously failed truths now verified. All 13 independent checks pass. No pushes to origin. No regressions introduced by gap-closure commits.

**Active carry-forwards (non-blocking):**
- CC-UPSTREAM-01: Fork is 8 commits ahead of bcddb89 — no upstream PRs per policy.
- CUDA event wrapper migration (cucascade PR #121 `cuda_event` type): Phase 24+ candidate.
- cudf `copy_partitions` memcheck violations (libcudf.so): cudf library baseline issue; functional test 7/7 without sanitizer; not blocking.

---

_Verified: 2026-05-13T12:00:00Z_
_Verifier: Claude (gsd-verifier) — re-verification after gap closure Plans 23-06 + 23-07_
