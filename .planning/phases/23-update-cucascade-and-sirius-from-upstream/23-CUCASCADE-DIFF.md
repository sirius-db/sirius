---
phase: 23-update-cucascade-and-sirius-from-upstream
type: cucascade-fork-divergence
upstream_base: bcddb89
upstream_base_subject: "Make host memory portable (PR #121)"
fork_head: 9da404756a8354d84d1dcd6bf3f3b46c29abfb3e
fork_head_short: 9da4047
fork_branch: fix/pinned-portable-flags
commits_ahead: 8
prior_pin: c666b21
last_updated: 2026-05-13
status: CC-UPSTREAM-01-carry
---

# Phase 23 cucascade Fork Divergence

## Background

Per CC-UPSTREAM-01 (established 2026-05-04, updated in Phase 22 and 22.1): Sirius carries a local cucascade fork pin on `feature/single-node-multi-gpu2`. Upstream cucascade PRs for the local fixes are deferred — the fixes are hardware-specific or operationally validated only against the Sirius workload, making upstream review a separate scoped effort.

Phase 23 Plan 02 rebased the local fork from `c666b21` (6 commits ahead of the old base `73d00c4`) onto `bcddb89` (PR #121 "Make host memory portable"), surgically splitting the prior squash `6236494` to drop the 4 files that PR #121 supersedes (portable/mapped pinning in `include/cucascade/memory/common.hpp`, `src/memory/common.cpp`, `src/memory/memory_space.cpp`, `src/memory/numa_region_pinned_host_allocator.cpp`) and keeping only the 3 ours-only files.

The result is **6 commits ahead of `bcddb89`** at `HEAD 1e889d7`.

```
bcddb89  upstream: PR #121 "Make host memory portable"
    |
    +-- 9a23f4f  fix(memory): ptds tracker, pool peer access, pipeline_io_backend hygiene
    |
    +-- 0c0a4af  fix(pipeline_io_backend): reorder io_worker members so _thread is last
    |
    +-- 8392c3d  fix(representation_converter): P2P override — target-bound stream, DMA probe at init
    |
    +-- 085d917  fix(stream-lineage): writer_stream/writer_event on gpu_table_representation + cudaStreamWaitEvent
    |
    +-- 89d6a3f  style: pre-commit cleanup (clang-format + codespell)
    |
    +-- 1e889d7  fix(p22): same-stream invariant in alloc_and_peer_copy_async (Cluster B)
    |
    +-- 37df815  fix(p23): cuda_set_device_raii guard for HtoD in alloc_and_peer_copy_async
    |
    +-- 9da4047  fix(p23): run_p2p_probe_locked must restore device context on exit
```

---

## Commit 1: 9a23f4f — Memory hygiene (ptds tracker, pool peer access, io_worker cleanup)

**Subject:** `fix(memory): ptds tracker, pool peer access, pipeline_io_backend hygiene`

**SHA:** `9a23f4f0aa83ea25770b12177a4a28b4552a3842`

**Files touched:**
- `src/data/pipeline_io_backend.cpp` (+90/-104 net — io_worker member cleanup, 104-line rewrite)
- `src/memory/reservation_aware_resource_adaptor.cpp` (+37/-8 net — ptds tracker + pool peer access)

**What this commit does:**

This is the surgical re-application of the former squash `6236494` (from Phase 22 and earlier) after dropping the 4 files superseded by upstream PR #121. Three ours-only functional hunks were retained:

1. **`pipeline_io_backend.cpp` io_worker cleanup** (104 lines): Removes an older constructor pattern that set `_thread = std::thread(...)` inline; replaces with a `start()` / `stop()` lifecycle. Prevents EINVAL races during concurrent io_worker teardown in multi-GPU test runs.

2. **`reservation_aware_resource_adaptor.cpp` per-instance ptds tracker**: Each `reservation_aware_resource_adaptor` instance gets its own `ptds_allocation_tracker` rather than sharing a global one. Required for correct multi-GPU memory reservation accounting when two GPU memory spaces are active simultaneously.

3. **`reservation_aware_resource_adaptor.cpp` pool peer access**: After pool construction, calls `cudaDeviceEnablePeerAccess` between all GPU pairs so that cross-GPU transfers in `alloc_and_peer_copy_async` can use the peer DMA path when available.

**Why not upstreamed:** Hardware-specific multi-GPU patterns (peer access enablement, per-instance ptds tracking) require upstream cucascade validation against their hardware matrix. CC-UPSTREAM-01 deferred pending coordinated upstream PR.

**Phase introduced:** Phase 23 Plan 02 (surgical rebase of Phase 22's work, originally Phase 16 Group 1 commits)

---

## Commit 2: 0c0a4af — io_worker member reordering (_thread last)

**Subject:** `fix(pipeline_io_backend): reorder io_worker members so _thread is last`

**SHA:** `0c0a4afb4df80cd0e65122d1b391393ecff5670b`

**Files touched:**
- `src/data/pipeline_io_backend.cpp` (+6/-1)

**What this commit does:**

`std::thread _thread` is declared AFTER `_mutex` and `_cv` in `io_worker`. C++ destroys members in reverse-declaration order; placing `_thread` last ensures `join()` inside `~io_worker` happens while `_mutex` and `_cv` are still alive. Without this fix, under parallel test teardown, `~io_worker` calls `_thread.join()` on a thread that's waiting on a destroyed `_cv`, causing EINVAL on mutex destruction.

Original commit: `eda349a` (Phase 11 hotfix — the io_worker EINVAL race was the Phase 11 AUDIT TEST_CASE SIGSEGV root cause half-1).

**Why not upstreamed:** Simple member-ordering fix that should be upstreamable; however, it's bundled with the other Phase 11/22 fixes in this commit history. Upstream PR would be the natural vehicle.

**Phase introduced:** Phase 11 original; carried through Phase 16 and Phase 23 rebase

---

## Commit 3: 8392c3d — P2P override, target-bound stream, DMA probe at init

**Subject:** `fix(representation_converter): P2P override — target-bound stream, DMA probe at init`

**SHA:** `8392c3d2892c4e5de0bc19abc551e82ec4834af3`

**Files touched:**
- `include/cucascade/memory/common.hpp` (+55)
- `src/data/representation_converter.cpp` (+383/-65)
- `src/memory/common.cpp` (+253)

**What this commit does:**

Three original commits squashed (`7ed84f2 cc2a53d e4db3d8`):

1. **Target-bound stream in host→gpu and gpu→gpu converters** (v1.1 P2P fix): All allocations and copies in `convert_host_to_gpu` / `convert_gpu_to_gpu` use `target_stream` (the destination GPU's execution stream) rather than a locally-created source stream. Closes cross-device stream-timeline ordering issue where the source GPU's stream was used for destination GPU operations.

2. **Source MR pass to cudf::pack + default-pool peer access**: Passes the source memory resource to cudf::pack calls so scratch allocations happen on the correct device. Adds default-pool peer access for zero-copy transfers on hardware where peer DMA works.

3. **Empirical DMA probe at init** (`probe_peer_dma_works`): During cucascade initialization, probes each (src, dst) GPU pair empirically by attempting a small device-to-device memcpy; stores result in a per-pair cache. `convert_gpu_to_gpu` routes to direct peer DMA (server hardware with full peer DMA) or host-staging (consumer hardware with broken peer DMA).

**Known regression (Phase 23 finding):** The new `convert_gpu_to_gpu` implementation in this commit uses `reconstruct_column_p2p` → `alloc_and_peer_copy_async` for the host-staging path. On hardware where peer DMA is broken (2 × RTX 6000 Ada), the HtoD `cudaMemcpyAsync` at `representation_converter.cpp:628` fails with `cudaErrorInvalidValue`. The old cucascade pin used `cudf::pack/unpack` and never triggered `alloc_and_peer_copy_async` from `convert_gpu_to_gpu`. This is a Phase 24 fix: add `rmm::cuda_set_device_raii{dst_device}` around the HtoD copy at line 628.

**Why not upstreamed:** Relies on empirical peer DMA probing and hardware-specific routing. The upstream cucascade architecture may prefer a different multi-GPU detection strategy. CC-UPSTREAM-01 deferred.

**Phase introduced:** Phase 23 Plan 02 (rebase of Phase 16 Groups 2+3 commits — `7ed84f2`, `cc2a53d`, `e4db3d8`)

---

## Commit 4: 085d917 — Stream lineage: writer_stream/writer_event + cudaStreamWaitEvent

**Subject:** `fix(stream-lineage): writer_stream/writer_event on gpu_table_representation + cudaStreamWaitEvent`

**SHA:** `085d917c3bc07f92fcae33391f095e502b6b4f57`

**Files touched:**
- `benchmark/benchmark_disk_converter.cpp` (+22/-4)
- `include/cucascade/data/gpu_table_representation.hpp` (+significant)
- `src/data/representation_converter.cpp` (+significant)

**What this commit does:**

Two original Phase 13 commits squashed (`7409c60 62e0517`):

1. **`record_writer_event` / `get_writer_event` on `gpu_table_representation`**: Adds CUDA event recording at the point a `gpu_table_representation` is written. When `convert_gpu_to_gpu` later copies the batch to another device, it first calls `cudaStreamWaitEvent(target_stream, src.get_writer_event())` to ensure the destination stream observes all writes to the source table.

2. **`writer_stream` as required ctor argument**: `gpu_table_representation(table, mem_space, stream)` — the `stream` argument is now compile-time enforced as a required parameter. This prevents accidental construction without a writer stream, which would leave `_writer_event` in an unrecorded state.

3. **Column-tree walk replaces cudf::pack in conversion**: `convert_gpu_to_gpu` switches from `cudf::pack / cudf::unpack` (which internally allocates on the wrong stream via `compute_splits`) to a column-by-column walk via `reconstruct_column_p2p`. This avoids the stream-ordered race in `compute_splits` scratch allocations that was the root cause of the Phase 13 SF100 Q11 illegal-address crash.

**Why not upstreamed:** Phase 13 stream-lineage is Sirius-specific (writer_stream is passed from Sirius pipeline execution; upstream cucascade uses a different data lifecycle model). Requires upstream review of the event-based synchronization design. CC-UPSTREAM-01 deferred.

**Phase introduced:** Phase 13 original; carried through Phase 16 and Phase 23 rebase

---

## Commit 5: 89d6a3f — Pre-commit formatting cleanup

**Subject:** `style: pre-commit cleanup (clang-format + codespell)`

**SHA:** `89d6a3f29b6c0ae03051e61531b3c2b292a95588`

**Files touched:** 15 source/header files (formatting only) + `docs/ARCHITECTURE.md` (typo)

**What this commit does:**

Pure formatting: `pre-commit run --all-files` applied to the fork after the functional commits. 15 files reformatted by clang-format (line-wrap, brace alignment). `docs/ARCHITECTURE.md`: `sytem` → `system` (codespell). No semantic changes.

**Why not upstreamed:** Formatting cleanup bundled with the local fork; will be included naturally if/when functional commits are upstreamed. CC-UPSTREAM-01 deferred.

**Phase introduced:** Phase 23 Plan 02 (new — ran pre-commit after completing the rebase)

---

## Commit 6: 1e889d7 — Same-stream invariant in alloc_and_peer_copy_async (Cluster B)

**Subject:** `fix(p22): same-stream invariant in alloc_and_peer_copy_async (Cluster B)`

**SHA:** `1e889d7e67070de7dc88860c373622182afe35df`

**Files touched:**
- `src/data/representation_converter.cpp` (+13/-6 in `alloc_and_peer_copy_async`)

**What this commit does:**

Phase 22 D-07 fix: collapses the host-staging fallback path in `alloc_and_peer_copy_async` onto a single CUDA stream (`target_stream`) so that `rmm::device_buffer::allocate_async` (line 600), the DtoH `cudaMemcpyAsync` (under `cuda_set_device_raii(src_device)`), and the HtoD `cudaMemcpyAsync` all observe a single stream timeline.

Before this fix, the function created an in-function `rmm::cuda_stream src_stream` and issued the DtoH copy on that stream, creating a stream-ordered race between the in-function stream's HtoD completion and the caller's `target_stream` usage of `buf`. The Phase 22.3 `sanitizer_gate_22.sh` Cluster B gate (`cluster_B=0`) verified this fix.

**Relationship to Commit 3 regression:** The same-stream invariant in this commit applies to the HOST-STAGING path (when `probe_peer_dma_works` returns false). The Phase 23 finding (REG-05 FAIL) is that the HtoD copy at line 628 fails with `cudaErrorInvalidValue` on this hardware — the invariant fix itself is correct (both copies are on `target_stream`), but the missing `cuda_set_device_raii{dst_device}` before the HtoD copy makes the copy fail. These are independent bugs: the Phase 22 fix addresses stream-ordering; the Phase 24 fix addresses device-context binding.

**Why not upstreamed:** Upstream cucascade PR review deferred per CC-UPSTREAM-01. This commit is the most self-contained of the 6 and the best upstream PR candidate once the Phase 24 `dst_device` guard fix is also applied (they should be submitted together so `alloc_and_peer_copy_async` is correct in one submission).

**Phase introduced:** Phase 22 Plan 03 (commit `c666b21` pre-rebase; rebased to `1e889d7` in Phase 23 Plan 02)

---

## Commit 7: 37df815 — dst_guard for HtoD in alloc_and_peer_copy_async (Phase 23 gap-closure)

**Subject:** `fix(p23): cuda_set_device_raii guard for HtoD in alloc_and_peer_copy_async`

**SHA:** `37df8153bf8330203954da99d341a139fcedd18c`

**Files touched:**
- `src/data/representation_converter.cpp` (+5/-1 in `alloc_and_peer_copy_async`)

**What this commit does:**

Wraps the HtoD `cudaMemcpyAsync` at line 628 (now ~629 after edit) in a new scope:
```cpp
{
  // Phase 23 gap-closure: set dst-device context before HtoD cudaMemcpyAsync
  rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}};
  cudaMemcpyAsync(buf.data(), host_buf, size, cudaMemcpyHostToDevice, target_stream.value());
}
```

Without this guard, the destination CUDA context is not active when `cudaMemcpyAsync(HtoD)` is called,
causing `cudaErrorInvalidValue` on hardware where peer DMA is broken (2 × RTX 6000 Ada). The function
already has a `rmm::cuda_set_device_raii src_guard{rmm::cuda_device_id{src_device}}` for the DtoH copy;
the HtoD copy needed a symmetric dst_guard.

**Phase introduced:** Phase 23 Plan 23-06 (gap-closure — fixes REG-05/REG-06 regression from commit 8392c3d)

**Upstream candidate:** Yes — should be submitted upstream alongside commit 6 (1e889d7) as a bundle.

---

## Commit 8: 9da4047 — probe device-restore in run_p2p_probe_locked (Phase 23 gap-closure)

**Subject:** `fix(p23): run_p2p_probe_locked must restore device context on exit`

**SHA:** `9da404756a8354d84d1dcd6bf3f3b46c29abfb3e`

**Files touched:**
- `src/data/representation_converter.cpp` (+4/-1 in `run_p2p_probe_locked`)

**What this commit does:**

`run_p2p_probe_locked` probes peer DMA between every (src, dst) GPU pair using `cudaMemcpyPeer`.
Before this fix, the function ended with a hardcoded `cudaSetDevice(0)`, clobbering any caller-held
RAII device guard. This left the active CUDA device as 0 after probe completion, causing downstream
`cudaEventRecord` calls for GPU 1's stream to fail with `cudaErrorInvalidResourceHandle` (the event
was created for GPU 1's context, but the current device was GPU 0).

Fix: save the current device at entry with `cudaGetDevice` and restore it at exit with `cudaSetDevice`.

**Discovery:** Found during Plan 23-07 smoke test: after applying commit 7 (37df815), the
[multi_gpu_foundation] 7/7 smoke still showed 6/7 FAIL with a different error
(`cudaErrorInvalidResourceHandle at gpu_data_representation.cpp:106`). The probe device-restore bug
was an independent second bug exposed by the first fix.

**Phase introduced:** Phase 23 Plan 23-07 (deviation Rule 1 auto-fix — blocking bug in smoke test)

**Upstream candidate:** Yes — straightforward correctness fix; should be submitted upstream together with commits 6+7.

---

## Upstreaming notes (CC-UPSTREAM-01)

Per CC-UPSTREAM-01 policy (established 2026-05-04):

- All 8 commits are carried in the local fork on `feature/single-node-multi-gpu2`
- No upstream PRs have been opened
- Upstream PR candidates when reviewed:
  - **Commit 2** (io_worker ordering) — most straightforward; clear correctness fix
  - **Commits 5+6+7+8 combined** (formatting + same-stream invariant + dst_guard + probe-restore) — the 4 Phase 22/23 `alloc_and_peer_copy_async` fixes form a logical unit for upstream PR submission
  - **Commit 4** (stream-lineage) — requires upstream agreement on `writer_stream` ctor design
  - **Commits 1+3** (memory hygiene + P2P override) — requires hardware-matrix validation; defer until server hardware available upstream

Commits 6, 7, and 8 together make `alloc_and_peer_copy_async` correct for broken-peer-DMA hardware:
- 6 (1e889d7): same-stream invariant — DtoH + HtoD on same target_stream
- 7 (37df815): dst_guard — HtoD requires dst-device CUDA context active
- 8 (9da4047): probe-restore — run_p2p_probe_locked must not clobber caller's device context

This document supersedes `22-CUCASCADE-DIFF.md` for the current fork state. The prior Phase 22 diff documented the 6-commit chain from base `73d00c4`; after the Phase 23 rebase, the new base is `bcddb89` and the fork carries 8 commits (6 from Phase 23 Plan 02 rebase + 2 gap-closure commits from Plans 23-06/23-07).
