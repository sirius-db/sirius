# Phase 16 Rebase Log

**Started:** 2026-05-04 (date stamp at write time)
**Worktree:** /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272

## Decisions Recorded

- **D-A3:** Rebased cucascade history is local-only. Pin advances to a free-floating local hash (not pushed to fork or upstream). Future re-clones must redo the rebase locally OR receive a patch series. Accepted risk for v1.4. Captured in `CC-UPSTREAM-01` deferral.
- **D-A4:** Abort criterion not yet triggered. If conflict resolution exceeds ~2 hr total, fall back to `git merge origin/main` on the local cucascade branch and document here.

## Squash Mapping (16-01)

Backup ref: `phase16-pre-squash-backup` -> 62e0517

| Group | Squashed Commit (post-16-01) | Original commits |
|-------|------------------------------|------------------|
| 1 | 3147ecf | 1fff85d, 3743621, 2dcab24, ff14ff4, e23f3a2 |
| 2 | 2c1c844 | 7ed84f2, cc2a53d, e4db3d8 |
| 3 | d52a67e | eda349a |
| 4 | 4930652 | 7409c60, 62e0517 |

## Conflict Resolution Rounds

### Round 1 (Group 1 — memory hygiene) — 16-02
- Files: `src/memory/common.cpp`, `src/memory/memory_space.cpp`
- Status: pending
- Resolution notes: (filled by 16-02)

### Round 2 (Group 3 — pipeline) — 16-02
- Files: `src/data/pipeline_io_backend.cpp`
- Status: pending
- Resolution notes: (filled by 16-02)

### Round 3 (Group 2 — stream/converter) — 16-03
- Files: `src/data/representation_converter.cpp`
- Status: pending
- Resolution notes: (filled by 16-03)

### Round 4 (Group 4 — Phase 13 stream-lineage) — 16-04
- Files: `include/cucascade/data/gpu_data_representation.hpp`, `src/data/gpu_data_representation.cpp`, `src/data/representation_converter.cpp`, `include/cucascade/data/data_batch.hpp` (proxy add), `test/data/test_data_batch.cpp` (ctor call updates)
- Status: pending
- Resolution notes: (filled by 16-04)

## Pin Advance (16-05)

- Old pin: 62e0517 (HEAD before rebase)
- New pin: (filled by 16-05)
- Parent commit: (filled by 16-05)

## CC-04 Grep Gate Outcomes (16-05)

| Gate | Command | Expected | Actual | Status |
|------|---------|----------|--------|--------|
| 1 | `grep -rn "record_writer_event\|get_writer_event" cucascade/include/cucascade/data/` | non-empty | (16-05) | pending |
| 2 | `grep -rn "cudaHostAllocPortable" cucascade/src/memory/` | non-empty | (16-05) | pending |
| 3 | `grep -rn "task_created\|in_transit" cucascade/src/data/` | zero | (16-05) | pending |
| 4 | `git -C cucascade merge-base --is-ancestor 73d00c4 HEAD` | exit 0 | (16-05) | pending |
| 5 | `git -C cucascade rev-list --count 73d00c4..HEAD` | 4 | (16-05) | pending |

## ctest Outcome (16-05)

- Run: pending
- Result: pending
- Build dir: pending
