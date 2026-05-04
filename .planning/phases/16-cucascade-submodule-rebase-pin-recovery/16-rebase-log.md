# Phase 16 Rebase Log

**Started:** 2026-05-04 (date stamp at write time)
**Worktree:** /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272

## Decisions Recorded

- **D-A3:** Rebased cucascade history is local-only. Pin advances to a free-floating local hash (not pushed to fork or upstream). Future re-clones must redo the rebase locally OR receive a patch series. Accepted risk for v1.4. Captured in `CC-UPSTREAM-01` deferral.
- **D-A4:** Abort criterion not yet triggered. If conflict resolution exceeds ~2 hr total, fall back to `git merge origin/main` on the local cucascade branch and document here.

## Squash Mapping (16-01)

Backup ref: `phase16-pre-squash-backup` -> 62e0517

| Group | Squashed Commit (post-16-01) | Original commits | Rebased commit (phase16-rebase-wip) |
|-------|------------------------------|------------------|--------------------------------------|
| 1 | 3147ecf | 1fff85d, 3743621, 2dcab24, ff14ff4, e23f3a2 | 6236494 |
| 2 | 2c1c844 | 7ed84f2, cc2a53d, e4db3d8 | (pending — 16-03) |
| 3 | d52a67e | eda349a | a1778f9 |
| 4 | 4930652 | 7409c60, 62e0517 | (pending — 16-04) |

## Conflict Resolution Rounds

### Round 1 (Group 1 — memory hygiene) — 16-02
- Files: `src/memory/common.cpp`, `src/memory/memory_space.cpp`, `src/data/pipeline_io_backend.cpp`
- Status: applied
- Resulting commit: `6236494` (cherry-pick of phase16-squashed-group1 onto 73d00c4)
- Resolution notes:
  - `src/memory/common.cpp`: One conflict in `#else` branch of `CUCASCADE_RMM_HAS_MOVABLE_ANY_RESOURCE` (73d00c4 retained pool capacity arg; ours drops it and adds peer access). Resolved per D-D1 — took our version: no capacity arg, added `enable_pool_peer_access_for_all_visible_devices` call.
  - `src/memory/memory_space.cpp`: One conflict in same `#else` branch. Resolved per D-D1 — took our version: removed `config.memory_capacity` from `cuda_async_memory_resource` ctor, added `enable_pool_peer_access_for_all_visible_devices(pool_handle, config.device_id)`. `get_chunked_resource_info()` method from 73d00c4 preserved (non-conflict).
  - `src/data/pipeline_io_backend.cpp`: One conflict in `pipeline_io_backend` ctor (73d00c4 uses `cudaMallocHost`; ours uses `cudaHostAllocPortable | cudaHostAllocMapped`). Group 1 patch also simplifies from per-device resources to single `_copy_stream/_order_event`. Resolved per D-D1 — took full Group 1 version of `pipeline_io_backend.cpp` (preserving Portable+Mapped flags, simplified stream/event management).
  - Non-conflict files `small_pinned_host_memory_resource.cpp` and `numa_region_pinned_host_allocator.cpp`: auto-applied by git with Portable/Mapped flags intact.
  - Time spent: ~20 min

### Round 2 (Group 3 — pipeline) — 16-02
- Files: `src/data/pipeline_io_backend.cpp`
- Status: applied
- Resulting commit: `a1778f9` (cherry-pick of phase16-squashed-group3 onto Group 1 tip)
- Resolution notes:
  - No conflict — cherry-pick applied cleanly (the diff between Group 2 squash and Group 3 squash is only the `io_worker` member reorder, which is a clean patch on top of our Group 1 version).
  - Deviation: the original `eda349a` comment says "MUST be declared before _thread" (on the block comment); the plan acceptance criteria requires `// MUST be last` inline comment on the `_thread` line. Added `// MUST be last — joins on destruction, must outlive _mutex/_cv` inline comment on `std::thread _thread;` line. Commit amended with this addition.
  - `_thread` is confirmed last member: `_mutex`, `_cv`, `_pending_work`, `_pending_promise`, `_has_task`, `_shutdown`, `_thread`.
  - Time spent: ~5 min

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
