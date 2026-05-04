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
- Files: `src/memory/common.cpp`, `src/data/representation_converter.cpp`, `include/cucascade/memory/common.hpp`
- Status: applied (provisional convert_gpu_to_gpu — finalized in 16-04)
- Resulting commit: `995bf4e` (cherry-pick of phase16-squashed-group2 onto Group 3 tip a1778f9)
- Resolution notes:
  - `src/memory/common.cpp`: Auto-merged correctly by git. Group 2's P2P probe block (`run_p2p_probe_locked`, `p2p_dma_works_cached`, `ensure_p2p_probed`, `probe_peer_dma_works`) added alongside 16-02's `enable_pool_peer_access_for_all_visible_devices` helper. No manual intervention needed for this file.
  - `include/cucascade/memory/common.hpp`: Auto-merged correctly. `probe_peer_dma_works(int, int)` declaration added to the memory namespace header.
  - `src/data/representation_converter.cpp`: ONE conflict at lines 145–202. HEAD had the old cudf::pack-based `convert_gpu_to_gpu` full implementation; Group 2 had only a forward declaration (the column-tree-walk implementation is defined later in the file below `convert_gpu_to_host_fast`). Resolved by taking Group 2's forward declaration form (`rmm::cuda_stream_view stream);`) and discarding HEAD's old cudf::pack body. The full column-tree-walk implementation auto-merged in below `convert_gpu_to_host_fast`.
  - API rename: `get_table().view()` at line 838 (in the auto-merged `convert_gpu_to_gpu` body) changed to `get_table_view()` per #117 API surface (D-D2 — `get_table()` is gone at 73d00c4).
  - 3-arg ctor wiring: All 4 `gpu_table_representation` construction sites updated to pass stream as 3rd arg (Option B per Round 2 note in 16-03 PLAN): `convert_host_to_gpu` → `stream`, `convert_gpu_to_gpu` → `target_stream`, `convert_host_fast_to_gpu` → `target_stream`, `convert_disk_to_gpu` → `stream`.
  - Provisional convert_gpu_to_gpu: uses the column-tree walk from Group 2 + `probe_peer_dma_works` routing via `alloc_and_peer_copy_async`. Group 4 (16-04) will finalize this with `cudaStreamWaitEvent(target_stream, writer_event)` and the `writer_stream` ctor arg.
  - Build state at end of 16-03: NOT compile-clean. The `gpu_data_representation.hpp` header still has the 2-arg ctor from 73d00c4; all 4 construction sites now pass 3 args. 16-04 fixes the build by adding `writer_stream` as a REQUIRED 3rd ctor arg to the header.
  - Apply order on rebased branch: Group 1 → Group 3 → Group 2 (different from original chronological 1 → 2 → 3 → 4). Acceptable per CC-02 "preserves carry as 4 group commits" — the order within the rebased branch is permitted to differ from original chronology.
  - Time spent: ~15 min

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
