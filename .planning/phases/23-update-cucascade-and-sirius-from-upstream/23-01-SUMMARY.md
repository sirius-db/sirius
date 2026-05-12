---
phase: 23-update-cucascade-and-sirius-from-upstream
plan: 01
subsystem: cucascade-rebase
tags: [git, rebase, cucascade, surgical-split, safety-net]
dependency_graph:
  requires: []
  provides: [cucascade-rebase-in-progress, cucascade-backup-branch, sirius-pre-merge-tag]
  affects: [cucascade/.git, sirius-refs]
tech_stack:
  added: []
  patterns: [surgical-git-rebase, edit-stop-split]
key_files:
  created:
    - /tmp/claude/p23_01_ours_only_hunks.diff
    - /tmp/claude/p23_01_cucascade_rebase_state.txt
  modified:
    - cucascade (refs only — fix/pinned-portable-flags-pre-phase23-backup branch)
    - cucascade (surgical commit 9a23f4f replaces 6236494 in rebase in-flight)
decisions:
  - "Re-scope: cucascade origin/main drifted from bcddb89 to 49134ff (CMake C-language cleanup only — cosmetic, no overlap with D-03/D-04 files)"
  - "Re-scope: Sirius origin/dev drifted from 12/393 to 12/395 (2 new commits: 8524c79 python fix + 16543e6 docs — low risk, no scope change)"
  - "D-04 KEEP confirmed: small_pinned_host_memory_resource.cpp NOT touched by PR #121 (bcddb89) — remains ours-only"
  - "D-03 DROP confirmed: all 4 files (common.hpp, common.cpp, memory_space.cpp, numa_region_pinned_host_allocator.cpp) touched by PR #121"
  - "Rebase stopped at 6236494 with conflict in numa_region_pinned_host_allocator.cpp (D-03 file) — resolved by taking --theirs (origin/main) then unstaging all D-03 files"
metrics:
  duration: 4min
  completed: 2026-05-12T17:58:00Z
  tasks: 2
  files: 0
---

# Phase 23 Plan 01: Safety Nets + Cucascade Surgical Split Summary

**One-liner:** Safety nets created (cucascade backup branch `fix/pinned-portable-flags-pre-phase23-backup` + Sirius tag `pre-phase23-merge`) and cucascade rebase is mid-flight at the surgical-split edit-stop with 9a23f4f replacing 6236494 (3 files, no D-03 portable-pinning hunks).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create recovery safety nets + inspect upstream | git refs only (no file commit) | cucascade backup branch, Sirius pre-merge tag |
| 2 | Run rebase, stop at 6236494, surgical split | cucascade: 9a23f4f (in-flight rebase) | pipeline_io_backend.cpp, reservation_aware_resource_adaptor.cpp, small_pinned_host_memory_resource.cpp |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Re-scope] Cucascade origin/main drifted from bcddb89 to 49134ff**

- **Found during:** Task 1 step 1 (divergence check)
- **Issue:** Plan expected `1<TAB>6` but got `2<TAB>6` because a new upstream commit `49134ff` ("Stop enabling C for C++ and CUDA builds, #123") landed on `origin/main` after the scaffold was written on 2026-05-08. The rebase target changed from `bcddb89` to `49134ff`.
- **Impact assessment:** `49134ff` only touches `CMakeLists.txt`, `CMakePresets.json`, and `benchmark/CMakeLists.txt` — no overlap with D-03 or D-04 files. The D-03/D-04 file split is unchanged. Rebase onto `49134ff` instead of `bcddb89` is safe.
- **Fix:** Re-scoped rebase target to `49134ff` (origin/main HEAD). All D-03/D-04 decisions still valid.

**2. [Rule 1 - Re-scope] Sirius origin/dev drifted from 12/393 to 12/395**

- **Found during:** Task 1 step 1 (divergence check)
- **Issue:** Plan expected `12<TAB>393` but got `12<TAB>395` because 2 new commits landed on `origin/dev`: `8524c79` (fix python extension bug) and `16543e6` (docs refresh).
- **Impact assessment:** Both are low-risk commits (per D-21 classification pattern). No scope change for Plan 23-01.
- **Fix:** Documented deviation. Plans 23-02/23-03 will handle the 2 additional origin/dev commits.

**3. [Rule 3 - Technique] Rebase edit-stop arrived as CONFLICT, not clean edit-stop**

- **Found during:** Task 2 step 1 (rebase execution)
- **Issue:** The plan specified marking `6236494` as `edit` and doing `git reset HEAD^` after the edit-stop. However, git stopped the rebase with a CONFLICT in `src/memory/numa_region_pinned_host_allocator.cpp` (a D-03 DROP file) rather than a clean edit-stop, because `6236494` conflicts with `49134ff`'s changes to that file.
- **Fix:** Instead of `git reset HEAD^` + selective re-stage, resolved the conflict by:
  1. Resolving the conflicted D-03 file with `git checkout --theirs` (taking origin/main version)
  2. Unstaging all 4 D-03 files with `git restore --staged`
  3. Restoring D-03 files' working tree to `HEAD` state (origin/main's version)
  4. Only D-04 files remained staged — committed directly with D-05 message
- **Result:** Identical end-state to the plan's intended outcome. Surgical commit has exactly 3 files, no D-03 hunks survive.

## Verification Results

### Divergence (re-scoped)
- Cucascade: `2/6` (2 behind origin/main due to `49134ff`; 6 ahead with our commits intact)
- Sirius: `12/395` (12 behind origin/dev; 395 ahead — 2 new low-risk commits)

### Safety Nets
- Backup branch: `fix/pinned-portable-flags-pre-phase23-backup` @ `c666b21926dec70b26a1febd509435635bea8deb`
- Pre-merge tag: `pre-phase23-merge` @ `b423a470a1b1e26082a8753cc88124ef6f2180e6`

### Upstream PR #121 Overlap Confirmation (D-03/D-04)
- **D-03 DROP confirmed (all 4 touched by bcddb89):**
  - `include/cucascade/memory/common.hpp` — YES (portable_pinning flag added)
  - `src/memory/common.cpp` — YES (portable variant factory)
  - `src/memory/memory_space.cpp` — YES (make_portable config field)
  - `src/memory/numa_region_pinned_host_allocator.cpp` — YES (portable ctor + cuda_host_flags)
- **D-04 KEEP confirmed (small_pinned_host_memory_resource.cpp):**
  - `git show bcddb89 -- src/memory/small_pinned_host_memory_resource.cpp` returned empty — NOT touched by PR #121
  - D-04 KEEP decision stands unchanged

### Surgical Commit (9a23f4f)
```
commit 9a23f4f0aa83ea25770b12177a4a28b4552a3842
fix(memory): ptds tracker, pool peer access, pipeline_io_backend hygiene

 src/data/pipeline_io_backend.cpp                  | 104 ++++++----------------
 src/memory/reservation_aware_resource_adaptor.cpp |  37 ++++++--
 src/memory/small_pinned_host_memory_resource.cpp  |  14 +--
 3 files changed, 64 insertions(+), 91 deletions(-)
```

- `git diff origin/main..HEAD -- include/cucascade/memory/ src/memory/common.cpp src/memory/memory_space.cpp src/memory/numa_region_pinned_host_allocator.cpp` = **0 lines** (D-03 files: no portable-pinning hunks survive)
- `git diff origin/main..HEAD -- src/memory/reservation_aware_resource_adaptor.cpp` = **73 lines** (non-empty, ptds tracker + pool peer access survives)
- `git diff origin/main..HEAD -- src/data/pipeline_io_backend.cpp` = **188 lines** (non-empty, 104-line cleanup survives)

### Rebase State
- Status: `interactive rebase in progress; onto 49134ff`
- Applied: 1 of 6 (surgical replacement for 6236494)
- Remaining: `a1778f9`, `995bf4e`, `1c1e648`, `42a01c4`, `c666b21` — Plan 23-02 picks up

### Reference Files
- Ours-only hunks diff: `/tmp/claude/p23_01_ours_only_hunks.diff` (306 lines)
- Post-split staged diff: `/tmp/claude/p23_01_post_split_cached.diff` (288 lines — identical content, 18 fewer header lines)
- Rebase state for Plan 23-02: `/tmp/claude/p23_01_cucascade_rebase_state.txt`

### Branch Confirmation
- Sirius parent: `feature/single-node-multi-gpu2`
- No `git push` executed

## Known Stubs

None — this plan is pure git operations with no code stubs.

## Self-Check

- [x] `/tmp/claude/p23_01_ours_only_hunks.diff` exists and is non-empty (14697 bytes, 306 lines)
- [x] `/tmp/claude/p23_01_cucascade_rebase_state.txt` exists and is non-empty
- [x] `fix/pinned-portable-flags-pre-phase23-backup` exists at `c666b21926dec70b26a1febd509435635bea8deb`
- [x] `pre-phase23-merge` tag exists at `b423a470a1b1e26082a8753cc88124ef6f2180e6`
- [x] Cucascade rebase in progress with 1 commit applied (9a23f4f)
- [x] 5 remaining commits queued for Plan 23-02

## Self-Check: PASSED
