---
phase: 16-cucascade-submodule-rebase-pin-recovery
plan: 03
subsystem: infra
tags: [git, rebase, cherry-pick, cucascade, p2p, stream-binding, conflict-resolution]

# Dependency graph
requires:
  - phase: "16-02"
    provides: "2 commits on top of 73d00c4 (Group 1 + Group 3); memory hygiene + io_worker fix"
provides:
  - "cucascade branch phase16-rebase-wip: 3 commits on top of 73d00c4 (Groups 1, 3, 2)"
  - "P2P peer-DMA probe block in common.cpp (run_p2p_probe_locked, probe_peer_dma_works)"
  - "target-bound stream in convert_host_fast_to_gpu and convert_gpu_to_gpu"
  - "p2p routing via probe_peer_dma_works in alloc_and_peer_copy_async"
  - "4 gpu_table_representation construction sites with stream as 3rd arg (provisional)"
  - "get_table() -> get_table_view() rename complete in representation_converter.cpp"
affects: [16-04, 16-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Conflict resolution D-D2: single conflict at forward-decl (HEAD full body vs Group 2 forward-decl) — take Group 2's forward-decl, discard HEAD's old cudf::pack body"
    - "3-arg ctor Option B: add stream arg to construction sites now, accepting build break until 16-04 adds the REQUIRED ctor param to header"
    - "API rename per #117 surface: get_table().view() -> get_table_view() in auto-merged bodies"

key-files:
  created: []
  modified:
    - cucascade/src/memory/common.cpp            # P2P probe block added (auto-merged by git)
    - cucascade/include/cucascade/memory/common.hpp  # probe_peer_dma_works declaration (auto-merged)
    - cucascade/src/data/representation_converter.cpp  # 1 conflict resolved; get_table() renamed; 4 3-arg ctor sites
    - .planning/phases/16-cucascade-submodule-rebase-pin-recovery/16-rebase-log.md

key-decisions:
  - "Conflict at forward-decl of convert_gpu_to_gpu: took Group 2's forward-decl (single line) and discarded HEAD's old cudf::pack full body. The column-tree-walk implementation appears later in the file and auto-merged correctly."
  - "3-arg ctor Option B selected: stream arg added to all 4 construction sites now, accepting build break. 16-04 will fix by adding writer_stream as REQUIRED 3rd arg to gpu_data_representation.hpp header."
  - "get_table().view() at line 838 (in auto-merged convert_gpu_to_gpu body) changed to get_table_view() per #117 API removal of get_table()."

patterns-established:
  - "probe_peer_dma_works(src, dst): public entry point for P2P DMA capability check; wraps cached probe result from run_p2p_probe_locked"
  - "alloc_and_peer_copy_async: routes to direct cudaMemcpyPeerAsync or explicit host-staged copy based on probe_peer_dma_works result"

requirements-completed: [CC-02]

# Metrics
duration: 15min
completed: 2026-05-04
---

# Phase 16 Plan 03: Cherry-pick Group 2 (P2P override + stream/converter) Summary

**Group 2 cherry-picked onto phase16-rebase-wip: P2P probe block in common.cpp, target-bound stream in 4 converter construction sites, probe_peer_dma_works routing in alloc_and_peer_copy_async — provisional state (build NOT clean; 16-04 finalizes header)**

## Performance

- **Duration:** 15 min
- **Started:** 2026-05-04T23:23:27Z
- **Completed:** 2026-05-04T23:38:45Z
- **Tasks:** 2
- **Files modified:** 4 (cucascade: 3 source files; planning: 1 log file)

## Accomplishments

- Cherry-picked `phase16-squashed-group2` (`2c1c844`) onto `a1778f9` — resolved 1 conflict
- `src/memory/common.cpp`: auto-merged correctly; P2P probe block now present alongside 16-02's `enable_pool_peer_access_for_all_visible_devices`
- `include/cucascade/memory/common.hpp`: auto-merged correctly; `probe_peer_dma_works()` declaration present
- `src/data/representation_converter.cpp`: 1 conflict at forward-decl of `convert_gpu_to_gpu` (HEAD had old cudf::pack body; Group 2 had forward-decl); resolved by taking forward-decl; column-tree-walk implementation auto-merged below
- `get_table().view()` at line 838 renamed to `get_table_view()` (per #117 API removal)
- All 4 `gpu_table_representation` construction sites updated to 3-arg form with stream
- `gpu_data_representation.hpp` still at `73d00c4` shape (no writer_stream — 16-04 task)
- Round 3 rebase-log section recorded with resolution notes and provisional-state caveat
- cucascade HEAD: `995bf4e` (3 commits above `73d00c4`)

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Cherry-pick Group 2 + resolve conflicts | `452aef1` (parent) / `995bf4e` (cucascade) | common.cpp, common.hpp, representation_converter.cpp |
| 2 | Update 16-rebase-log.md Round 3 section | `b4bab3c` (parent) | 16-rebase-log.md |

## Cucascade Commit Log (post-plan)

```
995bf4e fix(representation_converter): P2P override — target-bound stream, DMA probe at init
a1778f9 fix(pipeline_io_backend): reorder io_worker members so _thread is last
6236494 fix(memory): memory hygiene — Portable/Mapped pinning, ptds tracker, pool peer access
73d00c4 implement 3-class data_back model and get rid of state machine (#117)
```

## Files Created/Modified

- `cucascade/src/memory/common.cpp` — P2P probe block: `run_p2p_probe_locked`, `ensure_p2p_probed`, `p2p_dma_works_cached`, `probe_peer_dma_works` (auto-merged from Group 2)
- `cucascade/include/cucascade/memory/common.hpp` — `probe_peer_dma_works(int, int)` declaration (auto-merged)
- `cucascade/src/data/representation_converter.cpp` — 1 conflict resolved; `get_table().view()` → `get_table_view()`; 4 construction sites now 3-arg
- `.planning/phases/16-cucascade-submodule-rebase-pin-recovery/16-rebase-log.md` — Round 3 status = applied; resolution notes

## Decisions Made

- **Conflict resolution at convert_gpu_to_gpu forward-decl:** HEAD had the old cudf::pack-based full implementation; Group 2 had only a forward declaration (the column-tree-walk body is defined below `convert_gpu_to_host_fast`). Took Group 2's forward-decl — discarded HEAD's old body. This is the cleanest resolution per D-D2 (the old body is superseded by Group 2's implementation).
- **3-arg ctor Option B:** Per RESEARCH.md Section D Round 2 note, added stream as 3rd arg now (Option B), accepting the build break. The 4 sites use: `convert_host_to_gpu` → `stream`, `convert_gpu_to_gpu` → `target_stream`, `convert_host_fast_to_gpu` → `target_stream`, `convert_disk_to_gpu` → `stream`. 16-04 closes the break by adding `writer_stream` as REQUIRED to the header.
- **get_table_view() rename:** Group 2's auto-merged body used `gpu_source.get_table().view()` at one site (line 838 in convert_gpu_to_gpu). Changed to `gpu_source.get_table_view()` per #117 API (D-D2 — `get_table()` is gone at `73d00c4`).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] get_table().view() in auto-merged convert_gpu_to_gpu body**
- **Found during:** Task 1 — post-cherry-pick verification
- **Issue:** Group 2's `convert_gpu_to_gpu` implementation used `gpu_source.get_table().view()` but `get_table()` does not exist at `73d00c4` (replaced by `get_table_view()`). The plan's acceptance criteria required `grep -c "get_table()"` to return 0.
- **Fix:** Changed line 838 from `gpu_source.get_table().view()` to `gpu_source.get_table_view()`
- **Files modified:** `cucascade/src/data/representation_converter.cpp`
- **Committed in:** `452aef1` / `995bf4e` (Task 1 cherry-pick commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — API rename in auto-merged body)
**Impact on plan:** Required for correctness (the `get_table()` call would be a compile error at `73d00c4`). No scope creep.

## Issues Encountered

- The cherry-pick produced only 1 conflict (at the `convert_gpu_to_gpu` forward-declaration site). The column-tree-walk implementation, `alloc_and_peer_copy_async`, `reconstruct_column_p2p`, and the modified `convert_host_fast_to_gpu` all auto-merged correctly. This was simpler than expected — the diff between Group 2 and HEAD in `representation_converter.cpp` was entirely additive (new functions below the conflict site).

## Known Stubs

None — this plan is pure git rebase work with no new features or UI data flows.

## Build State

**NOT compile-clean.** Expected and documented. The `gpu_data_representation.hpp` header still has the 2-arg ctor from `73d00c4`. The 4 construction sites in `representation_converter.cpp` now pass 3 args (with stream). 16-04 will add `writer_stream` as a REQUIRED 3rd ctor arg to the header, closing the build gate.

## Next Phase Readiness

- 16-04 can cherry-pick `phase16-squashed-group4` (Phase 13 stream-lineage) onto `995bf4e` (current HEAD of `phase16-rebase-wip`)
- `phase16-squashed-group4` (Group 4, commit `4930652`) is intact and verified
- The 3 converter sites with `target_stream` will need `target_stream` passed to the ctor — 16-04 will also add the `record_writer_event(target_stream)` call after ctor construction in `convert_gpu_to_gpu`
- `gpu_data_representation.hpp` is ready for Group 4's 3-arg ctor addition

## Self-Check: PASSED

- FOUND: `git -C cucascade rev-list --count 73d00c4..HEAD` = 3
- FOUND: cucascade HEAD is `995bf4e` (`fix(representation_converter): P2P override — target-bound stream, DMA probe at init`)
- FOUND: `grep -q "p2p_dma_supported" cucascade/src/memory/common.cpp` (3 occurrences)
- FOUND: `grep -q "probe_peer_dma_works" cucascade/src/data/representation_converter.cpp` (1 call site in alloc_and_peer_copy_async)
- FOUND: `grep -c "get_table()" cucascade/src/data/representation_converter.cpp` = 0
- FOUND: all 4 make_unique<gpu_table_representation> sites have stream as 3rd arg
- FOUND: `git -C cucascade diff 73d00c4 HEAD -- include/cucascade/data/gpu_data_representation.hpp | wc -l` = 0
- FOUND: `git -C cucascade merge-base --is-ancestor 73d00c4 HEAD` exits 0
- FOUND: `grep -c "MUST be last" cucascade/src/data/pipeline_io_backend.cpp` = 1 (Group 3 from 16-02 preserved)
- FOUND: 16-rebase-log.md Round 3 status = "applied (provisional...)"

---
*Phase: 16-cucascade-submodule-rebase-pin-recovery*
*Completed: 2026-05-04*
