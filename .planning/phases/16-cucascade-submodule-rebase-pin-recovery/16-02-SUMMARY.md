---
phase: 16-cucascade-submodule-rebase-pin-recovery
plan: 02
subsystem: infra
tags: [git, rebase, cherry-pick, cucascade, memory-hygiene, io-worker, conflict-resolution]

# Dependency graph
requires: ["16-01"]
provides:
  - "cucascade branch phase16-rebase-wip: 2 commits on top of 73d00c4 (Group 1 + Group 3)"
  - "Memory hygiene: Portable/Mapped pinning, ptds tracker, pool peer access re-applied on 73d00c4"
  - "Pipeline io_worker member-order fix: _thread is last member with MUST-be-last comment"
  - "16-rebase-log.md Round 1 + Round 2 resolution notes recorded"
affects: [16-03, 16-04, 16-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Cherry-pick-per-group onto rebased base: cherry-pick squashed groups individually instead of running full interactive rebase"
    - "Conflict resolution D-D1: prefer-ours for additive changes (take our version, keep their structural additions)"
    - "Conflict resolution for mixed-state auto-merge: when git auto-merges body sections inconsistently, write the complete target-tree version from the source commit"

key-files:
  created: []
  modified:
    - cucascade/src/memory/common.cpp            # enable_pool_peer_access_for_all_visible_devices added; no-capacity ctor in #else branch
    - cucascade/src/memory/memory_space.cpp      # pool priming removed; peer access call added in both branches
    - cucascade/src/data/pipeline_io_backend.cpp # cudaHostAllocPortable/Mapped; simplified stream/event; io_worker _thread last
    - cucascade/src/memory/small_pinned_host_memory_resource.cpp  # auto-applied Portable flags
    - cucascade/src/memory/numa_region_pinned_host_allocator.cpp  # auto-applied Portable flags
    - cucascade/include/cucascade/memory/common.hpp               # auto-applied
    - cucascade/src/memory/reservation_aware_resource_adaptor.cpp # auto-applied
    - .planning/phases/16-cucascade-submodule-rebase-pin-recovery/16-rebase-log.md

key-decisions:
  - "Conflict resolution: for pipeline_io_backend.cpp, git auto-merged inconsistently (ctor from ours, method bodies from 73d00c4). Wrote complete Group 1 tree version to resolve the inconsistent state. This is D-D1 applied at file level."
  - "MUST-be-last comment: original eda349a commit used a multi-line block comment; plan acceptance criteria required an inline // MUST be last comment on _thread. Added inline comment and amended Group 3 commit."
  - "Group 3 clean apply: since we already wrote the Group 1 version of pipeline_io_backend.cpp (which had _thread first, same as Group 3's base), the Group 3 patch applied with no conflicts."

# Metrics
duration: 4min
completed: 2026-05-04
---

# Phase 16 Plan 02: Cherry-pick Groups 1+3 onto 73d00c4 Summary

**Groups 1 (memory hygiene) and 3 (io_worker member-order) cherry-picked onto cucascade origin/main tip 73d00c4 — 3 conflicts resolved (common.cpp, memory_space.cpp, pipeline_io_backend.cpp), 4 non-conflict files auto-applied, and io_worker _thread confirmed last-declared member with MUST-be-last inline comment**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-04T23:15:29Z
- **Completed:** 2026-05-04T23:19:48Z
- **Tasks:** 3
- **Files modified:** 7 cucascade source files + 1 planning log file

## Accomplishments

- Created `phase16-rebase-wip` branch at `73d00c4`
- Cherry-picked `phase16-squashed-group1` onto `73d00c4` — resolved 3 conflicts (see Deviations section); resulting commit `6236494`
- Cherry-picked `phase16-squashed-group3` onto Group 1 tip — clean apply + amended with `// MUST be last` inline comment; resulting commit `a1778f9`
- All 4 named refs `phase16-squashed-group{1,2,3,4}` verified intact for downstream 16-03/16-04
- `73d00c4` ancestry preserved: `git merge-base --is-ancestor 73d00c4 HEAD` exits 0
- Grep gates verified: `enable_pool_peer_access_for_all_visible_devices` in common.cpp (3 occurrences) and memory_space.cpp (2 occurrences); `cudaHostAllocPortable` in small_pinned (1) and numa_region (1) and pipeline_io_backend (2)
- Pool priming argument removed: `cuda_async_memory_resource concrete_mr(config.memory_capacity)` → `cuda_async_memory_resource concrete_mr;`
- Groups 2 and 4 are NOT applied — `representation_converter.cpp` and `gpu_data_representation.hpp` unchanged from `73d00c4`
- Rebase log Round 1 and Round 2 sections updated with resolution notes and commit SHAs

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Cherry-pick Group 1 onto 73d00c4 | `6236494` (cucascade) | common.cpp, memory_space.cpp, pipeline_io_backend.cpp, small_pinned_host_memory_resource.cpp, numa_region_pinned_host_allocator.cpp, common.hpp, reservation_aware_resource_adaptor.cpp |
| 2 | Cherry-pick Group 3 (io_worker) onto Group 1 | `a1778f9` (cucascade, amended) | pipeline_io_backend.cpp |
| 3 | Update 16-rebase-log.md | `36605a8` (parent repo) | 16-rebase-log.md |

## Cucascade Commit Log (post-plan)

```
a1778f9 fix(pipeline_io_backend): reorder io_worker members so _thread is last
6236494 fix(memory): memory hygiene — Portable/Mapped pinning, ptds tracker, pool peer access
73d00c4 implement 3-class data_back model and get rid of state machine (#117)
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Mixed auto-merge state] pipeline_io_backend.cpp inconsistent auto-merge**
- **Found during:** Task 1 conflict resolution
- **Issue:** Git auto-merged the `pipeline_io_backend.cpp` file inconsistently — the constructor section conflicted (our ctor vs 73d00c4 ctor), but the method bodies were silently taken from `73d00c4`'s version which referenced `res.copy_stream` from the per-device approach. Our Group 1 patch had already replaced the per-device approach with a single `_copy_stream/_order_event`. The auto-merged file would have referenced `res.copy_stream` in methods while the constructor created `_copy_stream` — a broken mixed state.
- **Fix:** Wrote the complete Group 1 version of `pipeline_io_backend.cpp` from `git show phase16-squashed-group1:src/data/pipeline_io_backend.cpp`. This is D-D1 applied at file level — take our version entirely.
- **Files modified:** `cucascade/src/data/pipeline_io_backend.cpp`
- **Commit:** Included in `6236494` (Task 1)

**2. [Rule 2 - Missing acceptance gate] MUST-be-last inline comment absent from eda349a**
- **Found during:** Task 2 Group 3 cherry-pick
- **Issue:** The original `eda349a` commit used a multi-line block comment starting with "Members are constructed in declaration order... MUST be declared before _thread". The plan acceptance criteria required `grep -c "MUST be last"` to return >= 1, expecting an inline `// MUST be last` comment directly on the `std::thread _thread;` line.
- **Fix:** Added `// MUST be last — joins on destruction, must outlive _mutex/_cv` inline comment. Amended the Group 3 commit with `git commit --amend --no-edit`.
- **Files modified:** `cucascade/src/data/pipeline_io_backend.cpp`
- **Commit:** Included in `a1778f9` (Task 2, amended)

### Notes

- Plan expected 2 conflict-resolution rounds (common.cpp + memory_space.cpp for Group 1; pipeline_io_backend.cpp for Group 3). Actual: 3 conflicts in Group 1 round (pipeline_io_backend.cpp conflicted on the constructor). Group 3 was clean. The extra conflict was in scope per D-D1 and resolved within budget.
- D-A4 abort criterion not triggered. Total time ~25 min, well within 30 min budget per group.

## Verification Results

| Gate | Command | Expected | Actual | Status |
|------|---------|----------|--------|--------|
| rev-list count | `git -C cucascade rev-list --count 73d00c4..HEAD` | 2 | 2 | PASS |
| Group 1 log subject | `git -C cucascade log HEAD~1 --format=%s` | `fix(memory): memory hygiene...` | `fix(memory): memory hygiene — Portable/Mapped pinning, ptds tracker, pool peer access` | PASS |
| Group 3 log subject | `git -C cucascade log HEAD --format=%s` | `fix(pipeline_io_backend): reorder...` | `fix(pipeline_io_backend): reorder io_worker members so _thread is last` | PASS |
| enable_pool_peer_access in common.cpp | `grep -c "enable_pool_peer_access_for_all_visible_devices" cucascade/src/memory/common.cpp` | >= 1 | 3 | PASS |
| enable_pool_peer_access in memory_space.cpp | `grep -c "enable_pool_peer_access_for_all_visible_devices" cucascade/src/memory/memory_space.cpp` | >= 1 | 2 | PASS |
| Portable in small_pinned | `grep -c "cudaHostAllocPortable" cucascade/src/memory/small_pinned_host_memory_resource.cpp` | >= 1 | 1 | PASS |
| Portable in numa_region | `grep -c "cudaHostAllocPortable" cucascade/src/memory/numa_region_pinned_host_allocator.cpp` | >= 1 | 1 | PASS |
| Pool priming removed | `grep -nE "cuda_async_memory_resource concrete_mr\(config\.memory_capacity"` cucascade/src/memory/memory_space.cpp | 0 lines | 0 | PASS |
| MUST be last | `grep -c "MUST be last" cucascade/src/data/pipeline_io_backend.cpp` | >= 1 | 1 | PASS |
| _thread last in io_worker | awk scan of io_worker class, last member | _thread | _thread | PASS |
| Named refs | `git -C cucascade rev-parse phase16-squashed-group{1,2,3,4}` | exit 0 | exit 0 | PASS |
| 73d00c4 ancestry | `git -C cucascade merge-base --is-ancestor 73d00c4 HEAD` | exit 0 | exit 0 | PASS |
| Group 2/4 isolation (representation_converter.cpp) | diff 73d00c4..HEAD | 0 lines changed | 0 | PASS |
| Group 2/4 isolation (gpu_data_representation.hpp) | diff 73d00c4..HEAD | 0 lines changed | 0 | PASS |
| Round 1 in rebase log | grep "applied" in Round 1 section | present | present | PASS |
| Round 2 in rebase log | grep "applied" in Round 2 section | present | present | PASS |

## Known Stubs

None — this plan is pure git rebase work with no new features or data flows.

## Next Phase Readiness

- 16-03 can cherry-pick `phase16-squashed-group2` (stream/converter) onto `a1778f9` (current HEAD of `phase16-rebase-wip`)
- 16-04 can cherry-pick `phase16-squashed-group4` (stream-lineage) onto 16-03's tip
- The named refs `phase16-squashed-group2` and `phase16-squashed-group4` are intact and pointing at the correct squashed commits
- `phase16-rebase-wip` branch is at `a1778f9` — the starting point for 16-03

## Self-Check: PASSED

- FOUND: cucascade HEAD is `a1778f9` (`fix(pipeline_io_backend): reorder io_worker members so _thread is last`)
- FOUND: `git -C cucascade rev-list --count 73d00c4..HEAD` = 2
- FOUND: `grep -q "enable_pool_peer_access_for_all_visible_devices" cucascade/src/memory/common.cpp`
- FOUND: `grep -q "enable_pool_peer_access_for_all_visible_devices" cucascade/src/memory/memory_space.cpp`
- FOUND: `grep -q "MUST be last" cucascade/src/data/pipeline_io_backend.cpp`
- FOUND: all 4 named refs `phase16-squashed-group{1,2,3,4}` resolve
- FOUND: `git -C cucascade merge-base --is-ancestor 73d00c4 HEAD` exits 0
- FOUND: `16-rebase-log.md` Round 1 + Round 2 status = "applied"

---
*Phase: 16-cucascade-submodule-rebase-pin-recovery*
*Completed: 2026-05-04*
