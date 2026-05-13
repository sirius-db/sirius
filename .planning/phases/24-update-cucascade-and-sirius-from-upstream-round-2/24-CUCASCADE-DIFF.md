---
phase: 24-update-cucascade-and-sirius-from-upstream-round-2
type: cucascade-fork-divergence
upstream_base: 9ceebaa
upstream_base_subject: "Fix for: Invalid Error: reconstruct_column STRING column metadata must have at least one child (offsets) (#124)"
fork_head: 5203de5a028ccb57402a4105e35282c567c3ee5a
fork_head_short: 5203de5
fork_branch: fix/pinned-portable-flags
commits_ahead: 9
prior_pin: 9da4047
prior_commits_ahead: 8
last_updated: 2026-05-13
status: CC-UPSTREAM-01-carry
---

# Phase 24 cucascade Fork Divergence

## Background

Per CC-UPSTREAM-01 (established 2026-05-04, updated through Phases 22, 22.1, 22.2, 22.3, 23):
Sirius carries a local cucascade fork pin on `feature/single-node-multi-gpu2`. Upstream cucascade
PRs for the local fixes are deferred — the fixes are hardware-specific or operationally validated
only against the Sirius workload, making upstream review a separate scoped effort.

Phase 24 Plan 02 rebased the local fork from `9da4047` (8 commits ahead of the old base `bcddb89`)
onto `9ceebaa` (upstream `origin/main` tip including PR #122 "feat: adding the ability to slice
host table" as `96bfea1` and PR #124 "Fix for: Invalid Error: reconstruct_column STRING" as
`9ceebaa` itself).

One commit required D-02 RE-DERIVE triage (commit 3 — `representation_converter.cpp`). One
additional test-fix commit was added (commit 9 — `5203de5`) to handle an API mismatch introduced
by `96bfea1`'s new slice-roundtrip test. No commits were dropped as fully OBSOLETED.

The result is **9 commits ahead of `9ceebaa`** at `HEAD 5203de5`.

This document supersedes `23-CUCASCADE-DIFF.md` for the current fork state. The prior Phase 23
diff documented the 8-commit chain from base `bcddb89`; after the Phase 24 rebase the new base is
`9ceebaa` and the fork carries 9 commits (8 from Phase 23 survives + 1 new test-fix commit).

```
9ceebaa  upstream: PR #124 "Fix for: Invalid Error: reconstruct_column STRING (#124)"
    |       (preceded by: 96bfea1 "feat: adding the ability to slice host table (#122)")
    |
    +-- 4b94571  fix(memory): ptds tracker, pool peer access, pipeline_io_backend hygiene
    |
    +-- 3c44dae  fix(pipeline_io_backend): reorder io_worker members so _thread is last
    |
    +-- d5ac57b  fix(representation_converter): P2P override — target-bound stream, DMA probe at init
    |
    +-- c15cb01  fix(stream-lineage): writer_stream/writer_event on gpu_table_representation + cudaStreamWaitEvent
    |
    +-- e10bd4a  style: pre-commit cleanup (clang-format + codespell)
    |
    +-- b21bd97  fix(p22): same-stream invariant in alloc_and_peer_copy_async (Cluster B)
    |
    +-- 4319726  fix(p23): cuda_set_device_raii guard for HtoD in alloc_and_peer_copy_async
    |
    +-- 1522e0b  fix(p23): run_p2p_probe_locked must restore device context on exit
    |
    +-- 5203de5  fix(test): adapt 96bfea1 slice-roundtrip test to writer_stream constructor
```

---

## Dropped During Phase 24 Rebase per D-02 Step 4

No commits were dropped during the Phase 24 rebase. All 8 Phase 23 fork commits survived:

| Original Phase 23 SHA | Phase 24 Rebased SHA | Outcome |
|----------------------|---------------------|---------|
| `49134ff` (CMake C-language cleanup) | — | Had already been dropped in Phase 23 (already upstream) |
| `9a23f4f` | `4b94571` | CLEAN |
| `0c0a4af` | `3c44dae` | CLEAN |
| `8392c3d` | `d5ac57b` | RE-DERIVE (single-line conflict resolved) |
| `085d917` | `c15cb01` | CLEAN |
| `89d6a3f` | `e10bd4a` | CLEAN |
| `1e889d7` | `b21bd97` | CLEAN |
| `37df815` | `4319726` | CLEAN |
| `9da4047` | `1522e0b` | CLEAN |

The D-08 HIGH-risk prediction ("`96bfea1`'s 489-line slice-host-table refactor may inline, rewrite,
or remove `alloc_and_peer_copy_async`") resolved favorably: `alloc_and_peer_copy_async` and
`reconstruct_column_p2p` are **100% fork-only code** — not present in any upstream commit.
The only conflict was the `convert_host_fast_to_gpu()` parameter-type change for the
`allocation` field (unique_ptr → shared_ptr), not the P2P code blocks themselves.

## Phase 24 Re-derived Commits

One commit required re-derivation (D-02 step 3):

**Commit 3: `8392c3d` → `d5ac57b`** — P2P override + DMA probe at init

- **Original Phase 23 SHA:** `8392c3d2892c4e5de0bc19abc551e82ec4834af3`
- **Phase 24 SHA:** `d5ac57b...` (full SHA not captured; rebased commit)
- **What changed:** `96bfea1` changed `host_table_allocation::allocation` from
  `unique_ptr<multiple_blocks_allocation>` to `shared_ptr<multiple_blocks_allocation>`. The
  `convert_host_fast_to_gpu()` function now dereferences it via `*fast_table->allocation` (shared_ptr
  requires explicit deref to pass a reference). Our commit had used the old `fast_table->allocation`
  form (unique_ptr passed directly).
- **Resolution:** Take upstream's `*fast_table->allocation` dereference (D-01 upstream shape) AND
  keep our `target_stream` argument (unique multi-GPU behavior). All other hunks (P2P code insertion,
  `convert_gpu_to_gpu` stub, DMA probe, `alloc_and_peer_copy_async`, `reconstruct_column_p2p`)
  applied without conflict — upstream has none of these functions.

## Phase 24 New Commits

**Commit 9: `5203de5`** — fix(test): adapt 96bfea1 slice-roundtrip test to writer_stream constructor

This is a net-new commit added during Phase 24 rebase (not present in Phase 23 fork). See
"Commit 9" section below for full details.

---

## Commit 1: 4b94571 — Memory hygiene (ptds tracker, pool peer access, io_worker cleanup)

**Subject:** `fix(memory): ptds tracker, pool peer access, pipeline_io_backend hygiene`
**Phase 23 SHA:** `9a23f4f0aa83ea25770b12177a4a28b4552a3842`
**Phase 24 SHA:** `4b94571...` (rebased commit on top of 9ceebaa)

**Files touched:**
- `src/data/pipeline_io_backend.cpp` (io_worker member cleanup + lifecycle)
- `src/memory/reservation_aware_resource_adaptor.cpp` (ptds tracker + pool peer access)

**What this commit does:**

Surgical re-application of the ours-only functional hunks from Phase 22's squash commit after
dropping the 4 files superseded by upstream PR #121:

1. **`pipeline_io_backend.cpp` io_worker cleanup:** Removes the older inline `_thread = std::thread(...)`
   constructor pattern; replaces with `start()` / `stop()` lifecycle. Prevents EINVAL races during
   concurrent io_worker teardown in multi-GPU test runs.

2. **`reservation_aware_resource_adaptor.cpp` per-instance ptds tracker:** Each instance gets its
   own `ptds_allocation_tracker` rather than sharing a global one. Required for correct multi-GPU
   memory reservation accounting when two GPU memory spaces are active simultaneously.

3. **`reservation_aware_resource_adaptor.cpp` pool peer access:** After pool construction, calls
   `cudaDeviceEnablePeerAccess` between all GPU pairs so that cross-GPU transfers can use the peer
   DMA path when available.

**Phase introduced:** Originally Phase 22 (in squash commit `6236494`); surgically re-extracted in
Phase 23 Plan 02 (as `9a23f4f`); carried cleanly through Phase 24 rebase.

**Why not upstreamed:** Hardware-specific multi-GPU patterns (peer access enablement, per-instance
ptds tracking) require upstream cucascade validation against their hardware matrix. CC-UPSTREAM-01
deferred pending coordinated upstream PR.

---

## Commit 2: 3c44dae — io_worker member reordering (_thread last)

**Subject:** `fix(pipeline_io_backend): reorder io_worker members so _thread is last`
**Phase 23 SHA:** `0c0a4afb4df80cd0e65122d1b391393ecff5670b`
**Phase 24 SHA:** `3c44dae...`

**Files touched:**
- `src/data/pipeline_io_backend.cpp` (+6/-1)

**What this commit does:**

`std::thread _thread` is declared AFTER `_mutex` and `_cv` in `io_worker`. C++ destroys members in
reverse-declaration order; placing `_thread` last ensures `join()` inside `~io_worker` happens
while `_mutex` and `_cv` are still alive. Without this ordering, `~io_worker` calls `_thread.join()`
on a thread that's waiting on a destroyed `_cv`, causing EINVAL on mutex destruction.

Original commit: `eda349a` (Phase 11 hotfix).

**Why not upstreamed:** Simple member-ordering fix; natural upstream PR vehicle. CC-UPSTREAM-01
deferred — this is the most self-contained of the 9 commits and a strong upstream PR candidate.

**Phase introduced:** Phase 11 original; carried through Phase 23 Plan 02 rebase and Phase 24.

---

## Commit 3: d5ac57b — P2P override, target-bound stream, DMA probe at init (RE-DERIVED)

**Subject:** `fix(representation_converter): P2P override — target-bound stream, DMA probe at init`
**Phase 23 SHA:** `8392c3d2892c4e5de0bc19abc551e82ec4834af3`
**Phase 24 SHA:** `d5ac57b...` (RE-DERIVE — single-line conflict resolved per D-02)

**Files touched:**
- `include/cucascade/memory/common.hpp` (DMA probe declarations)
- `src/data/representation_converter.cpp` (P2P code + `convert_host_fast_to_gpu` conflict site)
- `src/memory/common.cpp` (`probe_peer_dma_works`, `run_p2p_probe_locked`)

**What this commit does:**

Three original commits squashed (in Phase 23 context: commits `7ed84f2`, `cc2a53d`, `e4db3d8`):

1. **Target-bound stream in converters:** All allocations and copies in `convert_host_to_gpu` /
   `convert_gpu_to_gpu` use `target_stream` (the destination GPU's execution stream) rather than a
   locally-created source stream. Closes cross-device stream-timeline ordering issue.

2. **Empirical DMA probe at init** (`probe_peer_dma_works`): During cucascade initialization, probes
   each (src, dst) GPU pair empirically by attempting a small device-to-device memcpy; stores result
   in a per-pair cache. `convert_gpu_to_gpu` routes to direct peer DMA (server hardware) or
   host-staging (consumer hardware with broken peer DMA: 2 × RTX 6000 Ada).

3. **P2P code insertion:** `alloc_and_peer_copy_async`, `alloc_and_peer_copy_sync`,
   `reconstruct_column_p2p`, `convert_gpu_to_gpu` column-walk implementation — entirely ours-only;
   upstream has no equivalent.

**Phase 24 RE-DERIVE detail:** `96bfea1` changed `host_table_allocation::allocation` from
`unique_ptr` to `shared_ptr`, so the `convert_host_fast_to_gpu()` dereference changed from
`fast_table->allocation` (unique_ptr passed directly) to `*fast_table->allocation` (explicit
deref for shared_ptr → reference). Our commit's `target_stream` was preserved. Result:
`reconstruct_column(col_meta, *fast_table->allocation, target_stream, mr, batch)`.

**Why not upstreamed:** Relies on empirical peer DMA probing and hardware-specific routing. The
upstream cucascade architecture may prefer a different multi-GPU detection strategy. CC-UPSTREAM-01
deferred; this commit must be submitted alongside commits 6+7+8 as a logical bundle.

**Phase introduced:** Phase 23 Plan 02 (rebase of Phase 16 Groups 2+3); re-derived in Phase 24.

---

## Commit 4: c15cb01 — Stream lineage: writer_stream/writer_event + cudaStreamWaitEvent

**Subject:** `fix(stream-lineage): writer_stream/writer_event on gpu_table_representation + cudaStreamWaitEvent`
**Phase 23 SHA:** `085d917c3bc07f92fcae33391f095e502b6b4f57`
**Phase 24 SHA:** `c15cb01...`

**Files touched:**
- `benchmark/benchmark_disk_converter.cpp`
- `include/cucascade/data/gpu_table_representation.hpp`
- `src/data/representation_converter.cpp`

**What this commit does:**

1. **`record_writer_event` / `get_writer_event` on `gpu_table_representation`:** Adds CUDA event
   recording at the point a `gpu_table_representation` is written. When `convert_gpu_to_gpu` later
   copies the batch to another device, it calls `cudaStreamWaitEvent(target_stream, src.get_writer_event())`
   to ensure the destination stream observes all writes to the source table.

2. **`writer_stream` as required ctor argument:** `gpu_table_representation(table, mem_space, stream)` —
   the stream argument is compile-time enforced. Prevents accidental construction without a writer
   stream (which would leave `_writer_event` in an unrecorded state).

3. **Column-tree walk replaces cudf::pack in conversion:** `convert_gpu_to_gpu` switches from
   `cudf::pack / cudf::unpack` to a column-by-column walk via `reconstruct_column_p2p`. Avoids
   the stream-ordered race in `compute_splits` scratch allocations (Phase 13 SF100 Q11 root cause).

**Phase introduced:** Phase 13 original; carried through Phase 23 Plan 02 rebase and Phase 24.

**Why not upstreamed:** Phase 13 stream-lineage is Sirius-specific (`writer_stream` is passed from
Sirius pipeline execution; upstream cucascade uses a different data lifecycle model). Requires
upstream review of the event-based synchronization design. CC-UPSTREAM-01 deferred.

---

## Commit 5: e10bd4a — Pre-commit formatting cleanup

**Subject:** `style: pre-commit cleanup (clang-format + codespell)`
**Phase 23 SHA:** `89d6a3f29b6c0ae03051e61531b3c2b292a95588`
**Phase 24 SHA:** `e10bd4a...`

**Files touched:** Source/header files (formatting only) + `docs/ARCHITECTURE.md` (typo fix)

**What this commit does:**

Pure formatting: `pre-commit run --all-files` applied to the fork after the functional commits.
clang-format reformats, codespell fixes typos. No semantic changes.

**Phase introduced:** Phase 23 Plan 02 (new — ran pre-commit after completing the Phase 23 rebase).

**Why not upstreamed:** Bundled with the local fork; will accompany functional commits if/when
submitted upstream. CC-UPSTREAM-01 deferred.

---

## Commit 6: b21bd97 — Same-stream invariant in alloc_and_peer_copy_async (Cluster B)

**Subject:** `fix(p22): same-stream invariant in alloc_and_peer_copy_async (Cluster B)`
**Phase 23 SHA:** `1e889d7e67070de7dc88860c373622182afe35df`
**Phase 24 SHA:** `b21bd97...`

**Files touched:**
- `src/data/representation_converter.cpp` (+13/-6 in `alloc_and_peer_copy_async`)

**What this commit does:**

Phase 22 D-07 fix: collapses the host-staging fallback path in `alloc_and_peer_copy_async` onto a
single CUDA stream (`target_stream`) so that `rmm::device_buffer::allocate_async`, the DtoH
`cudaMemcpyAsync` (under `cuda_set_device_raii(src_device)`), and the HtoD `cudaMemcpyAsync` all
observe a single stream timeline.

Before this fix, the function created an in-function `rmm::cuda_stream src_stream` and issued the
DtoH copy on that stream, creating a stream-ordered race between the in-function stream's HtoD
completion and the caller's `target_stream` usage of `buf`. The `sanitizer_gate_22.sh` Cluster B
gate (`cluster_B=0`) verifies this fix.

**Relationship to commits 7+8:** This commit (6) addresses stream-ordering (both copies on
`target_stream`). Commit 7 addresses device-context binding (dst_guard). Commit 8 addresses probe
device-context clobbering. Together they form the complete `alloc_and_peer_copy_async` correctness
fix for broken-peer-DMA hardware.

**Phase introduced:** Phase 22 Plan 03 (commit `c666b21` pre-rebase; rebased to `1e889d7` in
Phase 23 Plan 02; carried to `b21bd97` in Phase 24 Plan 02).

**Why not upstreamed:** Best upstream candidate when submitted alongside commits 7+8 as a bundle.
CC-UPSTREAM-01 deferred.

---

## Commit 7: 4319726 — dst_guard for HtoD in alloc_and_peer_copy_async (Phase 23 gap-closure)

**Subject:** `fix(p23): cuda_set_device_raii guard for HtoD in alloc_and_peer_copy_async`
**Phase 23 SHA:** `37df8153bf8330203954da99d341a139fcedd18c`
**Phase 24 SHA:** `4319726...`

**Files touched:**
- `src/data/representation_converter.cpp` (+5/-1 in `alloc_and_peer_copy_async`)

**What this commit does:**

Wraps the HtoD `cudaMemcpyAsync` in a new scope with a `rmm::cuda_set_device_raii dst_guard`:

```cpp
{
  // Phase 23 gap-closure: set dst-device context before HtoD cudaMemcpyAsync
  rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}};
  cudaMemcpyAsync(buf.data(), host_buf, size, cudaMemcpyHostToDevice, target_stream.value());
}
```

Without this guard, the destination CUDA context is not active when `cudaMemcpyAsync(HtoD)` is
called, causing `cudaErrorInvalidValue` on hardware where peer DMA is broken (2 × RTX 6000 Ada).
The function already had a symmetric `src_guard` for the DtoH copy; the HtoD copy needed the
symmetric `dst_guard`.

**Phase introduced:** Phase 23 Plan 23-06 (gap-closure fixing REG-05/REG-06 regression from
commit 8392c3d); carried cleanly through Phase 24 rebase.

**Upstream candidate:** Yes — should be submitted upstream alongside commit 6 (b21bd97) and
commit 8 (1522e0b) as a bundle.

---

## Commit 8: 1522e0b — probe device-restore in run_p2p_probe_locked (Phase 23 gap-closure)

**Subject:** `fix(p23): run_p2p_probe_locked must restore device context on exit`
**Phase 23 SHA:** `9da404756a8354d84d1dcd6bf3f3b46c29abfb3e`
**Phase 24 SHA:** `1522e0b...`

**Files touched:**
- `src/memory/common.cpp` (in `run_p2p_probe_locked`)

**What this commit does:**

`run_p2p_probe_locked` probes peer DMA between every (src, dst) GPU pair using `cudaMemcpyPeer`.
Before this fix, the function ended with a hardcoded `cudaSetDevice(0)`, clobbering any
caller-held RAII device guard. This left the active CUDA device as 0 after probe completion,
causing downstream `cudaEventRecord` calls for GPU 1's stream to fail with
`cudaErrorInvalidResourceHandle` (the event was created for GPU 1's context, but the current
device was GPU 0).

Fix: save the current device at entry with `cudaGetDevice` (`saved_device` at lines 56-57) and
restore it at exit with `cudaSetDevice` (line 146).

**Discovery:** Found during Phase 23 Plan 23-07 smoke test: after applying commit 7 (37df815),
the `[multi_gpu_foundation]` 7/7 smoke still showed 6/7 FAIL with a different error
(`cudaErrorInvalidResourceHandle at gpu_data_representation.cpp:106`). The probe device-restore
bug was an independent second bug exposed by the first fix.

**Phase introduced:** Phase 23 Plan 23-07 (deviation Rule 1 auto-fix); carried cleanly through
Phase 24 rebase.

**Upstream candidate:** Yes — straightforward correctness fix; should be submitted upstream
together with commits 6+7.

---

## Commit 9: 5203de5 — fix(test): adapt 96bfea1 slice-roundtrip test to writer_stream constructor

**Subject:** `fix(test): adapt 96bfea1 slice-roundtrip test to writer_stream constructor`
**Phase 23 SHA:** (did not exist in Phase 23 fork)
**Phase 24 SHA:** `5203de5a028ccb57402a4105e35282c567c3ee5a` (new commit added during Phase 24 rebase)

**Files touched:**
- `test/data/test_data_representation.cpp`

**What this commit does:**

Upstream commit `96bfea1` ("feat: adding the ability to slice host table") added a new Catch2
test `host_data_representation::slice round-trip` in `test_data_representation.cpp`. This test
constructs a `gpu_table_representation` using the old 2-argument constructor:
`gpu_table_representation(std::move(table), mem_space)`.

Our commit `c15cb01` (Phase 24 rebased from `085d917`) makes `writer_stream` a required 3rd
constructor argument, so the upstream test fails to compile: error on missing 3rd argument.

Fix: added `stream.view()` as the 3rd argument to the upstream test's `gpu_table_representation`
construction site. This is a test-only fix — no production code changes.

**Classification:** DEVIATION Rule 1 auto-fix (post-rebase compilation error in upstream test
file). This is NOT a conflict marker; it was a compilation error discovered when running `ctest`
after the rebase completed.

**Phase introduced:** Phase 24 Plan 02 (new commit; not present in Phase 23 fork).

**Upstream candidate:** This test fix should accompany commit 4 (`c15cb01` — the writer_stream
required-ctor change) in any upstream PR. Since `96bfea1` is already upstream but our commit 4
is not, this creates an ordering dependency: if commit 4 is ever submitted upstream, this fix
must accompany it (or the slice-roundtrip test would fail to compile in the upstream repo after
commit 4 lands).

---

## Upstreaming Notes (CC-UPSTREAM-01)

Per CC-UPSTREAM-01 policy (established 2026-05-04; updated through Phases 22–24):

- All 9 commits are carried in the local fork on `fix/pinned-portable-flags`
- No upstream PRs have been opened
- D-06 confirmed: no `git push origin` executed in Phase 24

**Recommended upstream PR groupings:**

| Group | Commits | Rationale |
|-------|---------|-----------|
| A: IO worker ordering | 2 (3c44dae) | Self-contained correctness fix; lowest dependency |
| B: P2P correctness bundle | 6+7+8 (b21bd97+4319726+1522e0b) | Together make `alloc_and_peer_copy_async` correct for broken-peer-DMA hardware |
| C: Stream lineage + test | 4+9 (c15cb01+5203de5) | writer_stream ctor change requires test adaptation |
| D: P2P override + probe | 1+3 (4b94571+d5ac57b) | Hardware-specific; requires upstream hardware-matrix validation |
| E: Formatting | 5 (e10bd4a) | Accompanies whichever functional group is submitted first |

Groups A and B are the most upstream-ready. Group D requires upstream agreement on the empirical
peer-DMA probing design (vs. static device capability queries).

This document supersedes `23-CUCASCADE-DIFF.md`. The prior Phase 23 diff documented the 8-commit
chain from base `bcddb89`; after the Phase 24 rebase the new base is `9ceebaa` and the fork carries
9 commits.
