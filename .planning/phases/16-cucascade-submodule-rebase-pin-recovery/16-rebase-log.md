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
| 4 | 4930652 | 7409c60, 62e0517 | 1c1e648 |

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
- Files: `include/cucascade/data/gpu_data_representation.hpp`, `src/data/gpu_data_representation.cpp`, `src/data/representation_converter.cpp`, `include/cucascade/data/data_batch.hpp` (proxy add), `test/data/test_data_batch.cpp`, `test/data/test_representation_converter.cpp`, `test/data/test_disk_host_converters.cpp`, `test/data/test_gpu_disk_converters.cpp`, `test/data/test_data_representation.cpp`, `src/data/bandwidth_profiler.cpp`, `benchmark/benchmark_disk_converter.cpp`, `benchmark/benchmark_representation_converter.cpp`
- Status: applied
- Resulting commit: `1c1e648` (amended from initial 9dddf77 to include missed ctor sites)
- Resolution notes:
  - **Approach:** Full re-implementation per D-D2 (not a merge/cherry-pick conflict resolution). The prior session applied the cherry-pick and resolved all conflicts manually, then committed as 9dddf77. The wrap-up session (16-04 execution) amended this commit after build verification revealed missed ctor sites.
  - **gpu_data_representation.hpp:** Clean rewrite merging #117's owning_table_view variant shell (struct owning_table_view {std::any owner; size_t alloc_size; cudf::table_view view}; std::variant<unique_ptr<cudf::table>, owning_table_view> _table) with Group 4's writer_stream REQUIRED on both ctors (D-B2): 3-arg simple ctor + 5-arg template ctor for PR #116 cudf::table_view path. Added: record_writer_event(rmm::cuda_stream_view), [[nodiscard]] cudaEvent_t get_writer_event() const, ~gpu_table_representation() override, cudaEvent_t _writer_event{nullptr} member appended after _table. Includes: union of #117 (any, variant, cudf/table_view) + Group 4 (rmm/cuda_stream_view.hpp, cuda_runtime.h).
  - **gpu_data_representation.cpp:** Simple ctor body: init idata_representation + _table(std::move(table)) then record_writer_event(writer_stream). Template ctor: defined inline in header, initializes owning_table_view then calls record_writer_event. Destructor: if (_writer_event != nullptr) { cudaEventDestroy(_writer_event); }. record_writer_event impl: cudaEventCreateWithFlags + cudaEventRecord. get_writer_event: returns _writer_event. All variant-dispatch methods (get_size_in_bytes, get_uncompressed_data_size_in_bytes, get_table_view, release_table, clone) preserved verbatim from 73d00c4.
  - **representation_converter.cpp:** convert_gpu_to_gpu finalized: cudaStreamWaitEvent(target_stream.value(), source_repr.get_writer_event(), 0) added before issuing peer copies (Phase 13 STREAM-LINEAGE pass-1). Group 2's p2p_dma_supported routing via alloc_and_peer_copy_async preserved. Column-tree walk (not cudf::pack) from Group 2 retained. Other 3 converter functions (convert_host_to_gpu, convert_host_fast_to_gpu, convert_disk_to_gpu) retain their 16-03 3-arg ctor wiring.
  - **data_batch.hpp:** read_only_data_batch::get_writer_event() const proxy added per D-B3: auto* repr = get_data(); if (!repr) return nullptr; auto* gpu_repr = dynamic_cast<gpu_table_representation*>(repr); if (!gpu_repr) return nullptr; return gpu_repr->get_writer_event(). Added #include <cuda_runtime.h> at top. No circular include (gpu_data_representation.hpp does not include data_batch.hpp).
  - **Test files (~10 ctor sites):** test_data_batch.cpp (5 sites), test_representation_converter.cpp, test_disk_host_converters.cpp, test_gpu_disk_converters.cpp (4 sites) — all updated to 3-arg form with rmm::cuda_stream_view{} as writer_stream. test_data_representation.cpp: wrap_column helper updated to accept optional rmm::cuda_stream_view writer_stream = rmm::cuda_stream_view{} parameter; internal ctor call updated; all wrap_column call sites updated.
  - **Missed sites (caught by build verification):** bandwidth_profiler.cpp (1 site — uses bootstrap_stream), benchmark_disk_converter.cpp (11 sites — use stream.view() from local rmm::cuda_stream; 1 site in write_table_to_disk takes rmm::cuda_stream_view so uses stream directly), benchmark_representation_converter.cpp (6 sites — push_back loop sites use rmm::cuda_stream_view{}, warmup sites use warmup_stream.view(), setup-loop sites use setup_stream.view()). All caught in first build pass and amended into the commit.
  - **Build state at end of 16-04:** COMPILE-CLEAN. cmake --build exits 0 for library + tests + benchmarks. ctest deferred to 16-05 (sandboxed shell has no CUDA device in env; host has 2x RTX 6000 Ada).
  - **Time budget:** ~45 min total (source work in prior session ~30 min; wrap-up: .bak cleanup + build verify + fixes + SUMMARY + log + state updates ~15 min). Within D-A4 budget.
  - **4 commits on top of 73d00c4 confirmed:** 6236494 (Group 1), a1778f9 (Group 3), 995bf4e (Group 2), 1c1e648 (Group 4)

## Pin Advance (16-05)

- **Old pin:** `62e0517` (HEAD before Phase 16 rebase; pre-squash backup ref `phase16-pre-squash-backup`)
- **New pin:** `1c1e648a282a06747328c78f62d2d676ce51a8ce` (4 commits on top of `73d00c4`)
- **Intermediate pin:** `995bf4e` (3 groups: 1, 3, 2; advanced by 16-03 docs commit)
- **Final parent commit:** `5d1a8e0` (`docs(16-04): complete Group 4 stream-lineage plan — SUMMARY + audit log + STATE + ROADMAP`) — this commit also advanced the pin from `995bf4e` → `1c1e648` as part of the 16-04 metadata commit
- **Date:** 2026-05-05
- **Verification:** `git ls-tree HEAD cucascade` = `1c1e648a282a06747328c78f62d2d676ce51a8ce` ✓; `git submodule status cucascade` shows no leading `+` (clean state) ✓
- **Per D-A3:** Local-only pin — not pushed to `felipe` fork or upstream. Future re-clones must redo the rebase locally OR receive a patch series. Documented in this file.
- **Note:** The 16-04 docs commit (`5d1a8e0`) staged and committed `cucascade` gitlink as part of completing plan 16-04. Plan 16-05 Task 3 confirms the pin is already at the correct new SHA and no additional parent commit is needed.

## Phase 16 Final Status (16-05)

- **Phase:** COMPLETE
- **Date:** 2026-05-05
- **Requirements closed:** CC-01 (pin advanced), CC-02 (11 fixes as 4 group commits), CC-03 (writer_event re-attached), CC-04 (ctest + 8 grep gates green)
- **Cucascade branch:** `phase16-rebase-wip` at `1c1e648` (4 commits above `73d00c4`)
- **ctest:** PASSED (100% tests passed, 1/1, 13.91s)
- **Grep gates:** All 8 PASS
- **ROADMAP criteria:** All 5 PASS
- **D-A3 honored:** No push to any remote
- **D-A4 abort criterion:** NOT triggered (all 4 groups applied cleanly within budget)
- **Next phase:** Phase 17 — Sirius origin/dev merge (MERGE-01..05)

## CC-04 Grep Gate Outcomes (16-05)

All 8 grep gates run on 2026-05-05. All PASS.

| Gate | Command | Expected | Actual | Status |
|------|---------|----------|--------|--------|
| 1 | `grep -rn "record_writer_event\|get_writer_event" cucascade/include/cucascade/data/` | non-empty | 11 matches (data_batch.hpp + gpu_data_representation.hpp) | PASS |
| 2 | `grep -rn "cudaHostAllocPortable" cucascade/src/memory/` | non-empty | 2 matches (numa_region_pinned_host_allocator.cpp:45, small_pinned_host_memory_resource.cpp:57) | PASS |
| 3 | `grep -rn "task_created\|in_transit" cucascade/src/data/` AND `cucascade/include/cucascade/` | zero in BOTH | src/data/: 0, include/cucascade/: 0 (total: 0) | PASS |
| 4 | `git -C cucascade merge-base --is-ancestor 73d00c4 HEAD` | exit 0 | exit 0 | PASS |
| 5 | `git -C cucascade rev-list --count 73d00c4..HEAD` | 4 | 4 | PASS |
| 6 | `grep -nE "make_unique<gpu_table_representation>" cucascade/src/data/representation_converter.cpp` | >= 4 sites | 4 sites (lines 243, 886, 1136, 1738) | PASS |
| 7 | `_thread` is last-declared member in `io_worker` class (pipeline_io_backend.cpp) | _thread last | `std::thread _thread;  // MUST be last` at line 119, after _mutex (113) and _cv (114) | PASS |
| 8 | `grep -rn ".get_table()" cucascade/src/ cucascade/include/` | zero | 0 matches | PASS |

## ROADMAP Success Criteria (16-05)

All 5 ROADMAP Phase 16 success criteria PASS.

| Criterion | Description | Result |
|-----------|-------------|--------|
| ROADMAP-1 | 4 group commits with original-hash trailers | PASS (4 trailer/hash references in commit messages) |
| ROADMAP-2 | P2 writer_stream/cudaStreamWaitEvent survival in converter | PASS (cudaStreamWaitEvent at line 855; 4 ctor sites all pass writer_stream) |
| ROADMAP-3 | P9 Portable flag in memory pinning sites | PASS (2 matches: numa_region_pinned + small_pinned_host) |
| ROADMAP-4 | P8 io_worker _thread last member | PASS (_thread at line 119 after _mutex/114, _cv/114) |
| ROADMAP-5 | ctest passes + FSM removal across src/data/ + include/cucascade/ | PASS (ctest=100%, FSM hits=0 in both locations) |

## ctest Outcome (16-05)

- **Run:** 2026-05-05
- **Result:** PASS — 100% tests passed (1/1 test)
- **Build dir:** `cucascade/build/` (CTestTestfile.cmake present; built in plan 16-04)
- **Runtime:** 13.91s
- **Exit code:** 0
- **Log tail:**
  ```
  Test project /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade/build
      Start 1: cucascade_tests
  1/1 Test #1: cucascade_tests ..................   Passed   13.91 sec

  100% tests passed, 0 tests failed out of 1

  Total Test time (real) =  13.91 sec
  ```
