---
phase: 23-update-cucascade-and-sirius-from-upstream
plan: 02
subsystem: cucascade-rebase
tags: [git, rebase, cucascade, conflict-resolution, d07, d09, d10]
dependency_graph:
  requires: [23-01-cucascade-surgical-split]
  provides: [cucascade-rebase-complete, cucascade-new-head-sha]
  affects: [cucascade/.git, /tmp/claude/p23_02_new_cucascade_head.txt]
tech_stack:
  added: []
  patterns: [git-rebase-conflict-integration, both-sides-merge]
key_files:
  created:
    - /tmp/claude/p23_02_new_cucascade_head.txt
  modified:
    - cucascade/include/cucascade/memory/common.hpp (D-07 integration: PR #121 portable-pinning + 995bf4e P2P probe declarations)
    - cucascade/src/memory/common.cpp (D-07 integration: PR #121 portable factory + 995bf4e P2P probe implementation)
decisions:
  - "D-07 resolution: integrated both PR #121 portable-pinning (make_portable overloads) AND 995bf4e P2P/DMA probe helpers (anonymous namespace + public functions) — they are different functions/symbols with no semantic overlap"
  - "42a01c4 disposition: re-format + continue (clang-format applied cleanly to post-rebase tree; patch not empty)"
  - "a1778f9, 1c1e648, c666b21 applied cleanly without conflict (D-06, D-08, D-10 predictions correct)"
  - "alloc_and_peer_copy_async is in representation_converter.cpp, not reservation_aware_resource_adaptor.cpp — plan acceptance criteria text had file mislabeled; substance verified in correct file"
metrics:
  duration: 12min
  completed: 2026-05-12T18:10:00Z
  tasks: 2
  files: 2
---

# Phase 23 Plan 02: Cucascade Rebase Completion Summary

**One-liner:** Cucascade rebase completed — 6 commits ahead of origin/main (new HEAD 1e889d7) with PR #121 portable-pinning and 995bf4e DMA probe co-existing, Phase 13 stream-lineage and Phase 22 Cluster B same-stream invariant preserved.

## Tasks Completed

| Task | Name | Commit (cucascade) | Files |
|------|------|--------------------|-------|
| 1 | Resume rebase; apply a1778f9, 995bf4e, 1c1e648; D-07 conflict resolution | 0c0a4af, 8392c3d, 085d917 | common.hpp, common.cpp, representation_converter.cpp |
| 2 | Apply 42a01c4 (re-format + continue), c666b21 (Cluster B clean); capture SHA | 89d6a3f, 1e889d7 | common.cpp, representation_converter.cpp |

## Final Cucascade HEAD

```
1e889d7e67070de7dc88860c373622182afe35df
```

Written to: `/tmp/claude/p23_02_new_cucascade_head.txt`

## Commits Ahead of origin/main

```
1e889d7 fix(p22): same-stream invariant in alloc_and_peer_copy_async (Cluster B)
89d6a3f style: pre-commit cleanup (clang-format + codespell)
085d917 fix(stream-lineage): writer_stream/writer_event on gpu_table_representation + cudaStreamWaitEvent
8392c3d fix(representation_converter): P2P override — target-bound stream, DMA probe at init
0c0a4af fix(pipeline_io_backend): reorder io_worker members so _thread is last
9a23f4f fix(memory): ptds tracker, pool peer access, pipeline_io_backend hygiene
```

6 commits ahead of origin/main (49134ff). This matches the "5 or 6 commits ahead" prediction in the plan.

## Per-Commit Disposition

| Original SHA | Description | Disposition | Notes |
|---|---|---|---|
| `a1778f9` | pipeline_io_backend: reorder io_worker _thread to last | **Clean apply** — new SHA `0c0a4af` | D-06 prediction correct |
| `995bf4e` | representation_converter P2P + DMA probe at init | **Conflict on common.hpp + common.cpp — D-07 integrated both sides** | New SHA `8392c3d`; see D-07 section below |
| `1c1e648` | stream-lineage writer_stream/writer_event | **Clean apply** — new SHA `085d917` | D-08 prediction correct |
| `42a01c4` | clang-format + codespell cleanup | **Re-format + continue** — new SHA `89d6a3f` | D-09: patch applied without empty-patch skip |
| `c666b21` | Phase 22 Cluster B same-stream invariant | **Clean apply** — new SHA `1e889d7` | D-10 prediction correct |

## D-07 Conflict Resolution (995bf4e — memory/common)

The conflict in `include/cucascade/memory/common.hpp` and `src/memory/common.cpp` was between:
- **HEAD (PR #121 bcddb89/49134ff)**: Added `make_default_host_memory_resource(int, size_t, bool make_portable)` overload; updated `make_default_allocator_for_tier` lambda; updated `make_default_host_memory_resource` to forward to the bool overload.
- **995bf4e (ours)**: Added anonymous `namespace {}` block with P2P probe implementation (`g_p2p_supported`, `p2p_probe_mutex`, `run_p2p_probe_locked`, `ensure_p2p_probed`, `p2p_dma_works_cached`, `set_access_on_pool`); rewrote `enable_pool_peer_access_for_all_visible_devices` to use probe results; added public `probe_peer_dma_works` and `disable_peer_access_where_broken` functions + declarations.

**Resolution:** Took the full `995bf4e` anonymous namespace block + `enable_pool_peer_access_for_all_visible_devices` implementation, then kept PR #121's portable-pinning additions (`make_default_host_memory_resource` overloads, `make_default_allocator_for_tier` lambda) which were below the conflict region and were already in the post-conflict file area.

**Post-resolution grep verification:**
```
# PR #121 portable-pinning preserved:
src/memory/common.cpp:259:                                  bool make_portable)
src/memory/common.cpp:267:                                                                     make_portable)};
include/cucascade/memory/common.hpp:204:make_default_host_memory_resource(int device_id, std::size_t capacity, bool make_portable);

# 995bf4e DMA probe preserved:
src/memory/common.cpp:40:bool g_p2p_supported[kMaxDevices][kMaxDevices] = {};
src/memory/common.cpp:42:std::mutex& p2p_probe_mutex()
src/memory/common.cpp:285:bool probe_peer_dma_works(int src_device, int dst_device)
include/cucascade/memory/common.hpp:176:[[nodiscard]] bool probe_peer_dma_works(int src_device, int dst_device);
```

## D-09 Disposition (42a01c4 — clang-format/codespell)

`42a01c4` applied as re-format + continue. The formatter cleanup commit applied to the post-rebase tree without producing an empty patch — the clang-format changes (brace alignment, line-wrapping in `common.cpp` after the DMA probe additions) were non-trivial enough to still apply.

## D-10 Disposition (c666b21 — Cluster B same-stream)

Applied cleanly. The Phase 22 same-stream invariant in `alloc_and_peer_copy_async` (in `src/data/representation_converter.cpp`, NOT `reservation_aware_resource_adaptor.cpp`) survived intact.

**Note:** The plan's acceptance criteria text incorrectly listed `reservation_aware_resource_adaptor.cpp` as the file containing `alloc_and_peer_copy_async`. The actual function is in `representation_converter.cpp`. Verified in the correct file — all invariants pass.

## Grep Gate Results (Task 2 Step 4)

All 5 invariants pass:

**1. writer_stream/cudaStreamWaitEvent (non-zero required — Phase 13 stream-lineage):**
```
src/data/representation_converter.cpp:855:
  CUCASCADE_CUDA_TRY(cudaStreamWaitEvent(target_stream.value(), writer_event, 0));
```
PASS (non-zero)

**2. cudaHostAllocPortable in src/memory/ (non-zero required — D-04 surgical split preserved):**
```
src/memory/small_pinned_host_memory_resource.cpp:57:
  auto err = ::cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
src/memory/numa_region_pinned_host_allocator.cpp:43:
  ? static_cast<int>(cudaHostAllocPortable | cudaHostAllocMapped)
```
PASS (non-zero)

**3. record_writer_event/get_writer_event in include/cucascade/data/ (non-zero required — Phase 13):**
```
include/cucascade/data/gpu_data_representation.hpp:164: void record_writer_event(rmm::cuda_stream_view writer_stream);
include/cucascade/data/gpu_data_representation.hpp:178: [[nodiscard]] cudaEvent_t get_writer_event() const;
include/cucascade/data/data_batch.hpp:318: return gpu_repr->get_writer_event();
```
PASS (non-zero)

**4. FSM state machine (task_created/in_transit/data_batch_processing_handle/idata_batch_probe — must be 0):**
```
grep -rn "task_created|in_transit|data_batch_processing_handle|idata_batch_probe" src/ | wc -l
0
```
PASS (0 — old FSM state machine not re-introduced)

**5. io_worker _thread member order (must be last):**
```
awk '/class io_worker/,/^};/' src/data/pipeline_io_backend.cpp | grep -nE '^\s*[a-zA-Z_].*_thread' | tail -1
68:  std::thread _thread;  // MUST be last — joins on destruction, must outlive _mutex/_cv
```
PASS (_thread is last member)

**Phase 22 Cluster B same-stream invariant:**
```
grep -n "target_stream" src/data/representation_converter.cpp | head -3
597:                                                    rmm::cuda_stream_view target_stream,
600:  rmm::device_buffer buf(size, target_stream, target_mr);
606:  ..., target_stream.value()
grep -n "rmm::cuda_stream src_stream" src/data/representation_converter.cpp | wc -l
0  (local stream removed by c666b21 did NOT reappear)
grep -n "cudaStreamSynchronize(target_stream.value())" src/data/representation_converter.cpp
622:    CUCASCADE_CUDA_TRY(cudaStreamSynchronize(target_stream.value()));
630:  CUCASCADE_CUDA_TRY(cudaStreamSynchronize(target_stream.value()));
```
PASS (Phase 22 ordering pattern preserved)

## Branch Confirmation

- Cucascade branch: `fix/pinned-portable-flags`
- Sirius parent branch: `feature/single-node-multi-gpu2`
- Cucascade backup: `fix/pinned-portable-flags-pre-phase23-backup` @ `c666b21926dec70b26a1febd509435635bea8deb` (INTACT)
- Pre-merge Sirius tag: `pre-phase23-merge` @ `b423a470a1b1e26082a8753cc88124ef6f2180e6` (not checked in this plan — set in Plan 23-01)
- No `git push` executed

## Plan 23-03 Hand-off

Plan 23-03 reads `/tmp/claude/p23_02_new_cucascade_head.txt` to bump the Sirius gitlink:
```
1e889d7e67070de7dc88860c373622182afe35df
```

## Deviations from Plan

**1. [Rule 1 - Plan Mislabel] alloc_and_peer_copy_async file location**

- **Found during:** Task 2 grep gate verification
- **Issue:** Plan's Task 2 acceptance criteria specified `grep -n "target_stream" src/memory/reservation_aware_resource_adaptor.cpp` — but `alloc_and_peer_copy_async` lives in `src/data/representation_converter.cpp`. The plan had the wrong filename.
- **Fix:** Ran the verification against the correct file (`representation_converter.cpp`). All invariants pass.
- **Impact:** No code changes needed; verification only.

## Known Stubs

None — this plan is pure git operations with no code stubs.

## Self-Check

- [x] `fix/pinned-portable-flags-pre-phase23-backup` exists at `c666b21926dec70b26a1febd509435635bea8deb` (VERIFIED)
- [x] `git status` reports clean working tree on `fix/pinned-portable-flags` (VERIFIED)
- [x] `git log --oneline origin/main..HEAD | wc -l` = 6 (VERIFIED)
- [x] `/tmp/claude/p23_02_new_cucascade_head.txt` contains `1e889d7e67070de7dc88860c373622182afe35df` (VERIFIED)
- [x] All 5 grep gate invariants PASS (VERIFIED above)
- [x] DMA probe + PR #121 portable-pinning co-exist in common.{hpp,cpp} (VERIFIED)

## Self-Check: PASSED
