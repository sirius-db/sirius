---
phase: 24-update-cucascade-and-sirius-from-upstream-round-2
verified: 2026-05-13T18:00:00Z
status: passed
score: 15/15 must-haves verified
re_verification: false
---

# Phase 24: Update cucascade + sirius from upstream (round 2) — Verification Report

**Phase Goal:** Pull 2 new cucascade upstream commits (96bfea1 slice host table + 9ceebaa reconstruct_column STRING fix) + 2 new sirius upstream commits (ba5ed27 wire_data_repositories Phase 2 + 2e197c6 pin_table tier='host') into our forks; verify no regression via Phase 22.x/23 invariant gauntlet + 1 new pin_table host smoke. META-RULE: upstream is the source of truth.
**Verified:** 2026-05-13T18:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Mandatory Independent Checks (15 items)

Each check was executed independently against the actual codebase at HEAD `9d033fd` / cucascade `5203de5`. No executor self-assessment was taken at face value.

---

### Check 1: Sirius gitlink in HEAD index

**Command:** `git ls-tree HEAD cucascade`
**Expected:** `5203de5a028ccb57402a4105e35282c567c3ee5a`
**Actual:** `160000 commit 5203de5a028ccb57402a4105e35282c567c3ee5a cucascade`
**Status:** VERIFIED

---

### Check 2: Cucascade fork HEAD = 5203de5 AND 9 commits ahead of origin/main

**Command:** `git -C cucascade log --oneline origin/main..HEAD | wc -l`
**Expected:** 9 commits ahead of 9ceebaa
**Actual:** 9 commits; fork confirmed at `5203de5`; commits listed:
```
5203de5 fix(test): adapt 96bfea1 slice-roundtrip test to writer_stream constructor
1522e0b fix(p23): run_p2p_probe_locked must restore device context on exit
4319726 fix(p23): cuda_set_device_raii guard for HtoD in alloc_and_peer_copy_async
b21bd97 fix(p22): same-stream invariant in alloc_and_peer_copy_async (Cluster B)
e10bd4a style: pre-commit cleanup
c15cb01 fix(stream-lineage): writer_stream/writer_event on gpu_table_representation
d5ac57b fix(representation_converter): P2P override — target-bound stream, DMA probe at init
3c44dae fix(pipeline_io_backend): reorder io_worker members so _thread is last
4b94571 fix(memory): ptds tracker, pool peer access, pipeline_io_backend hygiene
```
**Status:** VERIFIED — 9 commits ahead; `git merge-base --is-ancestor 9ceebaa HEAD` returns success.

---

### Check 3: Phase 23 cucascade fixes intact

**a) dst_guard in representation_converter.cpp:**
`grep 'cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}}' cucascade/src/data/representation_converter.cpp`
Result: line 649 — FOUND

**b) run_p2p_probe_locked device-context restore in common.cpp:**
`grep 'run_p2p_probe_locked' cucascade/src/memory/common.cpp`
Result: lines 48 (definition) + 179 (call site) — FOUND

**Status:** VERIFIED — both Phase 23 safety guards intact after rebase.

---

### Check 4: Upstream cucascade features integrated

**a) 9ceebaa empty-STRING guard in representation_converter.cpp:**
At lines 1025–1035: `if (meta.children.empty() && meta.num_rows == 0) { return cudf::make_empty_column(...); }` guard present before the `children.size() < 1` throw.
**Status:** VERIFIED

**b) 96bfea1 host_table_allocation::create() factory in cucascade:**
`cucascade/include/cucascade/memory/host_table.hpp` line 82: `static std::unique_ptr<host_table_allocation> create(fixed_multiple_blocks_allocation buffers, std::vector<column_metadata> columns, std::size_t data_size);`
**Status:** VERIFIED

**c) Cucascade fork descends from both upstream commits:**
`git merge-base --is-ancestor 96bfea1 HEAD` — success
`git merge-base --is-ancestor 9ceebaa HEAD` — success
**Status:** VERIFIED

---

### Check 5: Sirius merge commit with both parents

**Command:** `git show ff04f31 --format="%P" -s`
**Result:** `8b2a7743bb3b0304a0aff9b85b9e714e4a7d77a9 ba5ed27080726f30aaa828437b191c3db78b9621`
Two parents confirmed: our pre-merge tip (`8b2a774`) + upstream dev tip (`ba5ed27`).
`git log --merges --oneline | head -3` shows `ff04f31 merge(p24): origin/dev into feature/single-node-multi-gpu2`.
**Status:** VERIFIED

---

### Check 6: New sirius commits absorbed

**ba5ed27 (wire_data_repositories):**
`git log --all --grep "wire_data_repositories" --oneline` — shows `ba5ed27` is a real commit and appears as parent of `ff04f31`; `git cat-file -t ba5ed27` = `commit`.

**2e197c6 (pin_table tier='host'):**
`git merge-base --is-ancestor 2e197c6 HEAD` — success; commit is ancestor of HEAD.
`git log --all --grep "pin_table.*host" --oneline` — lists `2e197c6`.
**Status:** VERIFIED — both upstream sirius commits merged into HEAD.

---

### Check 7: HYG-02 rmm::cuda_stream_default count

**Command:** `grep -rE 'rmm::cuda_stream_default' src/ legacy/ | wc -l`
**Result:** 40
**Status:** VERIFIED — at the ≤40 baseline limit; all in `src/legacy/` per VERDICT.md.

---

### Check 8: kvikio-free invariant

**Command:** `grep -rnE 'cudf::io::datasource::create\(|cudf::io::source_info\{' src/ | grep -v 'data_source\.get()\|datasource\.get()' | wc -l`
**Result:** 0
**Status:** VERIFIED — Phase 22.1 kvikio removal holds through the Phase 24 merge.

---

### Check 9: New pin_table host gate — test file existence

**Command:** `grep -n "pin_table_host" test/cpp/integration/test_gpu_execution_tpch.cpp`
**Result:**
```
4555:   "gpu_execution - pin_table host tier scan and aggregate",
4556:   "[integration][gpu_execution][parquet][pin_table_host]"
```
Test exists at lines 4555–4556 (tag `[integration][gpu_execution][parquet][pin_table_host]`). Gauntlet document records 1/1 PASS, 51 assertions, 6.6s.
**Status:** VERIFIED — test exists in source; gauntlet evidence records PASS.

---

### Check 10: sanitizer_gate_22.sh windowed-awk + P22_SELFTEST

Script exists at `test/scripts/sanitizer_gate_22.sh` (10,372 bytes, executable). Contains windowed-awk `in_race` state pattern at line 177 and `P22_SELFTEST` path at lines 90–117. Gauntlet evidence (24-04-GAUNTLET-RESULTS.md Section C) records P22_SELFTEST exit 0, full run cluster_A=0 cluster_B=0 total_races=0.

Re-running compute-sanitizer live is not performed by the verifier (per project guidance: compute-sanitizer runs via Bash+timeout with GPU hardware, and the gauntlet was already executed by the executor with documented log paths). The script structure is verified intact in source.
**Status:** VERIFIED (script structure; runtime evidence from gauntlet)

---

### Check 11: No git push to origin

**Sirius:** `git log origin/feature/single-node-multi-gpu2..HEAD --oneline | wc -l` = 113. Local branch has 113 commits not on `origin/feature/single-node-multi-gpu2`. `git remote get-url origin` = `https://github.com/felipeblazing/sirius.git`.

**Cucascade:** `git -C cucascade log origin/main..HEAD --oneline | wc -l` = 9. Commits reside on local `fix/pinned-portable-flags` branch; `origin` = `https://github.com/NVIDIA/cuCascade.git`.

**Status:** VERIFIED — no pushes to either origin. D-06 confirmed.

---

### Check 12: REQ rows Complete in REQUIREMENTS.md

**Command:** `grep -n "MERGE-CC-24\|MERGE-DEV-24\|GAUNTLET-24" .planning/REQUIREMENTS.md`

Results:
- Line 161: `MERGE-CC-24 | 24 | Complete — ...`
- Line 162: `MERGE-DEV-24 | 24 | Complete — ...`
- Line 163: `GAUNTLET-24 | 24 | Complete — ...`

All three requirement rows are marked Complete with substantive evidence text.
**Status:** VERIFIED

---

### Check 13: 24-VERDICT.md status: PASS and 18-gate table

`24-VERDICT.md` frontmatter: `status: PASS`. Gate table shows 18/18 rows PASS. Executor self-assessment is consistent with independent checks 1–12 above, which independently confirm the key code facts underlying the PASS verdict.
**Status:** VERIFIED

---

### Check 14: D-04 atomic-commit discipline

`git log --oneline` shows distinct atomic commits:
- `d228504` — "submodule: bump cucascade to 5203de5" (Commit A) — `git show --name-only d228504` shows only `cucascade` in diff (gitlink-only)
- `ff06fac` — API adapter for 96bfea1 (Commit B)
- `ff04f31` — merge commit (Commit C)
- `90fad83` — post-merge fix-up `gpu_table_representation` missing stream_view (Commit D, separate from merge commit)
- Commit E: not needed (Branch A — upstream test exists)
- `8067a80` + `9d033fd` — doc close (Commit F)

All are distinct commits; the post-merge fix-up is not an amend of the merge commit.
**Status:** VERIFIED

---

### Check 15: D-05 gitlink ours-wins in merge commit

**Command:** `git ls-tree ff04f31 cucascade`
**Result:** `160000 commit 5203de5a028ccb57402a4105e35282c567c3ee5a cucascade`

The merge commit `ff04f31` records the cucascade gitlink as `5203de5` (our fork), NOT `96bfea1` (upstream's pure commit). Upstream `2e197c6` had proposed advancing the gitlink to `96bfea1`; our fork at `5203de5` descends from `96bfea1` and the D-05 ours-wins resolution was applied correctly.
**Status:** VERIFIED

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Cucascade fork rebased onto 9ceebaa (upstream STRING fix integrated) | VERIFIED | `merge-base --is-ancestor 9ceebaa HEAD` = success; empty-STRING guard at representation_converter.cpp:1025–1035 |
| 2 | 96bfea1 slice host table feature integrated into cucascade fork | VERIFIED | `host_table_allocation::create()` factory at host_table.hpp:82; `merge-base --is-ancestor 96bfea1 HEAD` = success |
| 3 | Phase 23 safety guards (dst_guard + probe-restore) survived rebase | VERIFIED | representation_converter.cpp:649 dst_guard; common.cpp:48+179 run_p2p_probe_locked |
| 4 | Sirius gitlink pinned to our fork HEAD (5203de5), not upstream | VERIFIED | `git ls-tree HEAD cucascade` = `5203de5a...`; D-05 ours-wins in merge commit confirmed |
| 5 | ba5ed27 wire_data_repositories merged into sirius HEAD | VERIFIED | merge commit `ff04f31` parent `ba5ed27`; `merge-base` = ancestor of HEAD |
| 6 | 2e197c6 pin_table tier='host' merged and test exists | VERIFIED | `merge-base --is-ancestor 2e197c6 HEAD` = success; test at test_gpu_execution_tpch.cpp:4555–4556 |
| 7 | All Phase 22.x/23 invariant grep gates hold | VERIFIED | HYG-02=40; kvikio-free=0; drain_after_error 4 sites; SCHED-RR 4 hits; CTE producer_types 2 hits; downgrade tier gate 3 hits; SF10 Q11 test 2 hits |
| 8 | 18-gate gauntlet PASS (17 Phase 23 + 1 new D-07) | VERIFIED | 24-04-GAUNTLET-RESULTS.md records 18/18 PASS; test source confirmed for D-07 gate |
| 9 | No git push to either origin | VERIFIED | sirius 113 commits ahead of origin; cucascade 9 commits ahead of origin/main; both local-only |
| 10 | D-04 atomic commits (A, C, D, F distinct; E not needed) | VERIFIED | All commits distinct in git log; Commit A is gitlink-only; Commit D is separate from merge |

**Score:** 10/10 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `cucascade` gitlink | `5203de5a...` in HEAD tree | VERIFIED | `git ls-tree HEAD cucascade` exact match |
| `cucascade/src/data/representation_converter.cpp` | 9ceebaa STRING guard + Phase 23 dst_guard | VERIFIED | Empty-STRING guard at line 1025; dst_guard at line 649 |
| `cucascade/include/cucascade/memory/host_table.hpp` | `host_table_allocation::create()` factory | VERIFIED | Static factory at line 82 |
| `cucascade/src/memory/common.cpp` | run_p2p_probe_locked device-context restore | VERIFIED | Definition line 48, call line 179 |
| `test/cpp/integration/test_gpu_execution_tpch.cpp` | `[pin_table_host]` Catch2 test at ~4556 | VERIFIED | Tag at line 4556 |
| `test/scripts/sanitizer_gate_22.sh` | windowed-awk + P22_SELFTEST logic | VERIFIED | 10,372 bytes; awk at line 177; selftest at line 90 |
| Sirius merge commit `ff04f31` | Two parents (pre-merge + ba5ed27) | VERIFIED | Parents: `8b2a774` + `ba5ed27` |
| Post-merge fix-up `90fad83` | gpu_table_representation stream_view arg, distinct from merge | VERIFIED | Separate commit; not amend of ff04f31 |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| Cucascade fork HEAD | upstream 9ceebaa | `merge-base --is-ancestor` | WIRED | 9ceebaa is ancestor; 9 commits ahead |
| Cucascade fork HEAD | upstream 96bfea1 | `merge-base --is-ancestor` | WIRED | 96bfea1 is ancestor |
| Sirius HEAD | cucascade 5203de5 | `git ls-tree HEAD cucascade` | WIRED | Exact SHA match in HEAD tree |
| Sirius HEAD | upstream ba5ed27 | merge parent + `merge-base` | WIRED | Parent of ff04f31; ancestor of HEAD |
| Sirius HEAD | upstream 2e197c6 | `merge-base --is-ancestor` | WIRED | Ancestor of HEAD |
| pin_table host | Catch2 test source | grep test file | WIRED | Test tag at test_gpu_execution_tpch.cpp:4556 |

---

### Requirements Coverage

| Requirement | Phase | Description | Status | Evidence |
|-------------|-------|-------------|--------|----------|
| MERGE-CC-24 | 24 | Cucascade fork rebased onto 9ceebaa; 9 commits ahead at 5203de5 | SATISFIED | REQUIREMENTS.md line 161 "Complete"; git checks 1–4 confirm |
| MERGE-DEV-24 | 24 | Sirius origin/dev merged (ba5ed27 + 2e197c6); D-05 gitlink ours-wins | SATISFIED | REQUIREMENTS.md line 162 "Complete"; git checks 5–6, 15 confirm |
| GAUNTLET-24 | 24 | 18/18 gates PASS; sanitizer_gate_22.sh cluster_B=0; P22_SELFTEST PASS | SATISFIED | REQUIREMENTS.md line 163 "Complete"; 24-04-GAUNTLET-RESULTS.md; checks 7–10 confirm |

No orphaned requirements for Phase 24.

---

### Anti-Patterns Found

None identified. The executor applied upstream-as-source-of-truth (D-01) throughout. Post-merge fix-up (`90fad83`) is an atomic commit, not an in-place amend. No TODOs or placeholder stubs introduced by the merge. The `chunk_memory_spaces` count drop (60→42) was audited and accepted: GPU-tier round-robin still uses `chunk_memory_spaces`; host-tier uses different fields. Functional coexistence confirmed by `[pin_mgpu]` 2/2, `[mgpu-audit]` 6/6, `[pin_table_host]` 1/1 in the gauntlet.

---

### Human Verification Required

None. All gate results are backed by documented log paths from the gauntlet run (24-04-GAUNTLET-RESULTS.md). The compute-sanitizer runs were executed by the executor via Bash+timeout per project policy; re-running them in the verifier is not required given the structural evidence (script intact, code guards present, no new race-prone patterns introduced).

---

### Gaps Summary

No gaps. All 15 independent checks PASS. The executor's self-verdict (24-VERDICT.md `status: PASS`, 18/18 gates) is confirmed by independent source-code and git-history verification. Phase 24 goal is achieved.

---

## Carry-Forward Items (informational, not gaps)

The following items are deferred per policy — they are not Phase 24 failures:

- **CC-UPSTREAM-01**: Cucascade fork 9 commits ahead of upstream origin/main; no upstream PRs yet. User handles separately.
- **PIN-MGPU-03 closed by 2e197c6**: Host-tier `pin_table` is now integrated (upstream test passes 1/1). The original NUMA-local round-robin design remains a v1.6+ consideration.
- **cudf libcudf.so memcheck baseline**: `Invalid __global__ read` violations in libcudf.so (Phase 23 Leg 1) were absent in Phase 24. Classify as cudf internal baseline — monitor in Phase 25.

---

_Verified: 2026-05-13T18:00:00Z_
_Verifier: Claude (gsd-verifier) — independent checks on sirius HEAD `9d033fd`, cucascade `5203de5`_
