# Phase 16: Cucascade Submodule Rebase + Pin Recovery — Research

**Researched:** 2026-05-04
**Domain:** git rebase mechanics + cucascade C++ API surgery (RAII DataBatch, stream-lineage, memory hygiene)
**Confidence:** HIGH — all findings from direct `git show` on live commits; no training-data assertions

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**A. Rebase Mechanics**
- D-A1: Squash 11 commits into 4 group commits before rebasing. Groups: (1) Memory hygiene — `1fff85d + 3743621 + 2dcab24 + ff14ff4 + e23f3a2`; (2) Stream/converter — `7ed84f2 + cc2a53d + e4db3d8`; (3) Pipeline — `eda349a`; (4) Phase 13 stream-lineage — `7409c60 + 62e0517`.
- D-A2: Resolve conflicts in-place per group commit. 4 rounds, not 11.
- D-A3: Local-only pin. No fork push, no upstream PR this milestone.
- D-A4: Abort criterion — switch to `git merge origin/main` if conflicts exceed ~2× budget (~2 hr total, ~30 min per group).

**B. writer_event Placement**
- D-B1: Keep `writer_stream`/`writer_event` on `gpu_table_representation` (not `data_batch`, not `idata_representation`).
- D-B2: `writer_stream` is a REQUIRED ctor parameter — compile-time enforced. Both ctors get it.
- D-B3: Expose `get_writer_event()` on `read_only_data_batch` as a proxy (~5 LOC). Implementation: `return _batch->get_data()->cast<gpu_table_representation>().get_writer_event()` (with appropriate downcast via `get_data()` on the private `data_batch`). The same proxy on `mutable_data_batch` is NOT required unless Phase 18 needs it.
- D-B4: Recording stays caller-controlled (explicit `record_writer_event(stream)` calls). No auto-record on `set_data()`.

**C. Carry-fix Granularity**
Resolved by D-A1: 4 group commits on top of `73d00c4`.

**D. Conflict Resolution Policy**
- D-D1: Additive collisions — prefer ours, re-apply on top of theirs.
- D-D2: Deletion conflicts — re-implement against new shape; document "obviated" fixes in `16-rebase-log.md`.
- D-D3: Signature-change conflicts — combine both intents; neither intent is dropped.

### Claude's Discretion
- Author attribution of 4 group commits: use `git config` user; add `Co-Authored-By` for original authors if helpful.
- Per-group commit messages: include "Squash of 11 fixes onto cucascade `73d00c4`"; cite original commit hashes.
- `mutable_data_batch::get_writer_event()` proxy: add only if Phase 18 needs it; YAGNI otherwise.
- Cucascade ctest failure handling: fix in-phase; if fix exceeds 1 hr, escalate with diagnostic.
- Updates to STREAM-LINEAGE comment block in `representation_converter.cpp`: refresh if time permits.
- PR #112 / PR #116 sanity check: light verification (test_data_batch.cpp passes; `gpu_data_representation` cudf::table_view ctor compiles).

### Deferred Ideas (OUT OF SCOPE)
- Upstream the 11 local fixes as cucascade PRs (CC-UPSTREAM-01, v1.5+).
- Sirius-side regression test for writer_event correctness.
- Refresh STREAM-LINEAGE comment block fully if scope grows.
- `mutable_data_batch::get_writer_event()` proxy unless Phase 18 needs it.
- Bandwidth profiler (PR #112) integration into Sirius observability.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CC-01 | Cucascade submodule pin advanced to a commit descended from `73d00c4` | Section F (submodule pin mechanics), Section D (rebase command sequence) |
| CC-02 | All 11 local fixes preserved on the new pin | Section A (per-file conflict shape), Section G (pitfall guards) |
| CC-03 | Phase 13 stream-lineage semantics re-attached under #117 RAII accessor model | Section B (API delta), Section C (proxy implementation), Section A-file-3 |
| CC-04 | Cucascade ctest passes + grep gates green | Section E (ctest scope), Section H (grep one-liners) |
</phase_requirements>

---

## Summary

Phase 16 is a pure git rebase operation inside `cucascade/`. The Sirius parent repo is untouched except for advancing the submodule pointer. The 11 local commits on `cucascade/HEAD` (`62e0517`, tip) diverged from merge-base `edd6f03` and must be squashed into 4 group commits then rebased onto `origin/main` tip `73d00c4` (PR #117 RAII DataBatch + PR #112 bandwidth profiler + PR #116 `gpu_data_representation` from `cudf::table_view`).

The dominant difficulty is a genuine semantic conflict between two independent sets of changes to the same files: `gpu_data_representation.hpp` and `representation_converter.cpp`. PR #117 reshaped both files deeply (deleting the FSM, removing `get_table()`, renaming `release_table()` to require a stream argument, adding the `owning_table_view` variant). Our Group 4 (Phase 13 stream-lineage) added `writer_stream` ctor arg, `record_writer_event`, `get_writer_event`, and `_writer_event` member — none of which appear at `73d00c4`. These must be re-planted by hand into the #117 shapes. Mechanical (line-shifted) conflicts in the memory files are easier: the Portable/Mapped flags are one-liner additions at `cudaMallocHost`/`cudaHostAlloc` call sites that #117 did not modify in those specific lines.

**Primary recommendation:** Treat `representation_converter.cpp` and `gpu_data_representation.hpp` as full re-implementations rather than merges. Start from `73d00c4`'s versions, hand-apply Group 2+4 additions, verify with greps before proceeding to ctest. All other conflict files are surgical line-level re-applications.

---

## Section A: Per-File Conflict Shape

### Conflict File 1: `include/cucascade/data/gpu_data_representation.hpp`

**Group responsible:** Group 4 (Phase 13 stream-lineage — `7409c60 + 62e0517`)

**What HEAD adds (not in `73d00c4`):**

| HEAD addition | Location |
|---------------|----------|
| `#include <rmm/cuda_stream_view.hpp>` | include block |
| `#include <cuda_runtime.h>` | include block |
| Constructor `gpu_table_representation(unique_ptr<cudf::table>, memory_space&, rmm::cuda_stream_view writer_stream)` — replaces the 2-arg ctor | public section |
| `~gpu_table_representation()` override — destroys `_writer_event` | public section |
| `record_writer_event(rmm::cuda_stream_view)` method | public section |
| `[[nodiscard]] cudaEvent_t get_writer_event() const` method | public section |
| `cudaEvent_t _writer_event{nullptr}` member | private section |
| STREAM-LINEAGE doxygen block on ctor | doxygen |

**What `73d00c4` adds (not in HEAD):**

| 73d00c4 addition | Location |
|-----------------|----------|
| `#include <cudf/table/table_view.hpp>` | include block |
| `#include <any>`, `#include <variant>`, `#include <cstddef>` | include block |
| Second ctor: `template<typename Owner> gpu_table_representation(cudf::table_view, Owner&&, size_t, memory_space&)` | public section (PR #116) |
| `get_table_view() const` replaces `get_table() const` | public section |
| `release_table(rmm::cuda_stream_view stream)` — signature changed (stream added) | public section |
| `owning_table_view` struct | private section |
| `std::variant<unique_ptr<cudf::table>, owning_table_view> _table` | private section |
| Explicit `= delete` for copy/move | public section |

**Conflict type:** SEMANTIC — both sides add to the class definition in the same regions (public methods, private members). The only shared base class is the 2-arg ctor and `get_table()` — both are replaced by different things. Resolution cannot be done by a merge tool; it must be authored by hand.

**Resolution approach (D-D1 / D-D3):**
1. Start from `73d00c4` version.
2. Add the REQUIRED `writer_stream` parameter to the 2-arg ctor (making it 3-arg), adding the stream after `memory_space&`.
3. Add the REQUIRED `writer_stream` parameter to the PR #116 template ctor too (D-B2: both ctors get it).
4. Add `record_writer_event`, `get_writer_event`, destructor declarations to the public section.
5. Add `_writer_event{nullptr}` member to the private section.
6. Keep `owning_table_view` variant from #117 — do not revert to `unique_ptr<cudf::table>` only.
7. Keep `get_table_view()` from #117 (HEAD's `get_table()` is gone; update converter to use `get_table_view()`).
8. Keep `release_table(stream)` from #117.

**Insertion point for Group 4 additions in the post-#117 file:**
- The 3-arg ctor goes at the position of the existing 2-arg ctor (replacing it).
- `record_writer_event` and `get_writer_event` go after `release_table` in the public section.
- `~gpu_table_representation()` override goes immediately after the ctor declarations.
- `_writer_event` member goes at the end of the private section, after `_table`.

**Critical note:** The `release_table(rmm::cuda_stream_view stream)` at `73d00c4` is a NEW signature. HEAD still has `release_table()` with no stream argument. After rebase, `release_table` must take the stream param (it's used in the `owning_table_view` path to synchronize before releasing the owner's lifetime). Check all call sites of `release_table` in the converter — there are none in the GROUP 4 commits (the converter only uses `get_table()` / `get_table_view()`, not `release_table`), so this is a non-issue for conflict resolution.

---

### Conflict File 2: `src/data/gpu_data_representation.cpp`

**Group responsible:** Group 4

**What HEAD adds:** Constructor body records `writer_event` via `record_writer_event(writer_stream)`. Destructor destroys `_writer_event` with `cudaEventDestroy`. `record_writer_event` and `get_writer_event` implementations.

**What `73d00c4` changes:** Constructor body handles the `owning_table_view` variant path. `get_table_view()` implementation traverses the variant. `release_table(stream)` synchronizes the stream for the owning path.

**Conflict type:** SEMANTIC — same constructor body, different purposes. Both intents must survive (D-D3).

**Resolution approach:**
- In the simple-table ctor: after `idata_representation(memory_space)` init, call `record_writer_event(writer_stream)` (same as HEAD).
- In the template ctor body (in the `.hpp` file, since it's a template): call `record_writer_event(writer_stream)` after `_table(...)` initialization.
- `get_size_in_bytes()` and `get_uncompressed_data_size_in_bytes()` — `73d00c4` adds variant-dispatch logic for the `owning_table_view` path; keep that. HEAD has no variant — keep `73d00c4`'s variant version.
- `get_table_view()` — keep `73d00c4`'s implementation (traverses variant). HEAD's `get_table()` is gone.
- `record_writer_event()` and `get_writer_event()` — re-apply from HEAD verbatim; they don't interact with the variant logic.
- Destructor — re-apply `cudaEventDestroy(_writer_event)` guard from HEAD.

---

### Conflict File 3: `src/data/representation_converter.cpp` — HIGHEST RISK

**Groups responsible:** Groups 2 AND 4 (dual-group conflict)

**Group 2 changes (commits `7ed84f2 + cc2a53d + e4db3d8`):**
- `convert_host_to_gpu`: passes `stream` to `gpu_table_representation` ctor.
- `convert_host_fast_to_gpu` (lines ~1088 in HEAD): passes `stream` to ctor.
- `convert_host_to_gpu` (old cudf::pack path): passes stream.
- `e4db3d8`: `p2p_dma_supported()` probe function and call in `convert_gpu_to_gpu` to decide host-staging vs. peer-DMA path (the entire ~120-line peer-DMA probe logic in `src/memory/common.cpp` is Group 2).

**Group 4 changes (commits `7409c60 + 62e0517`):**
- `convert_gpu_to_gpu`: full rewrite — replaces the `cudf::pack`-based path with a column-tree walk; adds `cudaStreamWaitEvent(target_stream, writer_event, 0)` as STREAM-LINEAGE pass 1; adds fallback `cudaDeviceSynchronize`; constructs result with 3-arg ctor passing `target_stream`.
- All other `convert_*` functions: pass `stream` as the 3rd ctor arg wherever `gpu_table_representation` is constructed.

**What `73d00c4` changes in this file:**
- At `73d00c4`, `convert_gpu_to_gpu` still uses the `cudf::pack`-based path (NOT the column-tree walk). It calls `get_table_view()` (not `get_table()`), `release_table(stream)` (with stream arg), and constructs `gpu_table_representation` with 2 args (no writer_stream).
- `convert_host_fast_to_gpu` at `73d00c4`: constructs `gpu_table_representation` with 2 args.
- The file is 1506 lines at `73d00c4` vs. 1798 lines at HEAD.

**Intra-our-side collision (Groups 2 AND 4 both touch the file):**
When Groups 2 and 4 are squashed together they will NOT collide with each other because:
- Group 2 primarily touches the P2P probe logic (now in `common.cpp`) and the stream wiring of converters OTHER than `convert_gpu_to_gpu`.
- Group 4 owns `convert_gpu_to_gpu` entirely and the 3-arg ctor wiring across all converter functions.
- There is no line-level overlap between Group 2's `convert_host_to_gpu` edits and Group 4's `convert_gpu_to_gpu` rewrite.

However, Group 2's P2P probe (`e4db3d8`) is in `src/memory/common.cpp`, NOT in `representation_converter.cpp`. So Group 2's contribution to `representation_converter.cpp` is only: target-bound stream in `convert_host_to_gpu` / `convert_gpu_to_gpu` (the earlier version before Group 4 rewrote `convert_gpu_to_gpu`).

**Since Group 4 fully rewrites `convert_gpu_to_gpu` anyway, Group 2's earlier version of that function is superseded by Group 4. When squashing Groups 2+4 (they are in separate group commits but both touch this file), the squash ordering is:**
- Group 2 commit contains the target-bound stream in `convert_host_to_gpu` and early `convert_gpu_to_gpu`.
- Group 4 commit then rewrites `convert_gpu_to_gpu` from scratch.
- The squashed Group 4 result naturally supersedes Group 2's `convert_gpu_to_gpu` changes.

**Conflict type with `73d00c4`:** SEMANTIC — `73d00c4`'s `convert_gpu_to_gpu` uses `cudf::pack` + 2-arg ctor; HEAD's uses column-tree walk + `cudaStreamWaitEvent` + 3-arg ctor. The two are logically incompatible: cannot merge, must re-implement.

**Resolution approach (per D-D2 / D-D3 — re-implement against new shape):**
1. For `convert_gpu_to_gpu`: start from HEAD's column-tree-walk implementation (Group 4). Re-read `73d00c4`'s version to confirm `get_table_view()` is used there; update HEAD's `get_table().view()` calls to `get_table_view()` (since `get_table()` is gone at `73d00c4`). Verify the 3-arg ctor call passes `target_stream`. The rest of the function body is Group 4's and is taken verbatim.
2. For `convert_host_to_gpu` (old cudf::pack path, ~line 246): `73d00c4` uses `get_table_view()` and 2-arg ctor; re-apply Group 2/4's 3-arg ctor addition.
3. For `convert_host_fast_to_gpu` (~line 853 at HEAD): `73d00c4` uses 2-arg ctor; re-apply the 3-arg ctor addition from Group 4.
4. For `convert_gpu_to_host_fast` (~line 585 at HEAD): `73d00c4` uses `get_table_view()`; HEAD uses `get_table().view()`. Update to `get_table_view()`.
5. For `convert_disk_to_gpu` (~line 1453 at HEAD): similar 2->3-arg ctor addition.
6. The STREAM-LINEAGE comment block (~40 lines at top of `convert_gpu_to_gpu`) — keep from HEAD, update the #117-era class name references.
7. **Key API rename:** everywhere in the file, `get_table()` → `get_table_view()`. This is mandated by #117 removing `get_table()` in favor of `get_table_view()`.

**P2P probe in `common.cpp`:** Group 2's `e4db3d8` adds ~200 lines to `src/memory/common.cpp`. At `73d00c4`, `common.cpp` does NOT contain the probe (the diff from HEAD→73d00c4 removes it entirely). This is the most substantial Group 2 content and lives in its own file — no collision with `73d00c4`'s `representation_converter.cpp` changes.

---

### Conflict File 4: `src/data/pipeline_io_backend.cpp`

**Group responsible:** Group 3 (single commit `eda349a`)

**What HEAD adds:** Reorders `io_worker` member declarations so `_mutex`, `_cv`, `_pending_work`, `_pending_promise`, `_has_task`, `_shutdown` come BEFORE `_thread`. Adds comment explaining why `_thread` must be last.

**What `73d00c4` has:** Same class structure but `_thread` is declared FIRST (before `_mutex`, `_cv`, etc.). This is the pre-fix state.

**Conflict type:** MECHANICAL — same class definition, same member names, different declaration order. The diff is pure member-reordering with a comment addition.

**Resolution approach (D-D1):**
Take `73d00c4`'s `io_worker` class structure and re-apply the member reordering from `eda349a`. The logic is identical — only the declaration order changes. After resolution, `io_worker` must end with:
```
  std::mutex _mutex;
  std::condition_variable _cv;
  std::function<void()> _pending_work;
  std::promise<void> _pending_promise;
  bool _has_task{false};
  bool _shutdown{false};
  std::thread _thread;  // MUST be last — joins on destruction, must outlive _mutex/_cv
```

**Important:** Verify that `73d00c4`'s `pipeline_io_backend.cpp` has no structural changes to the rest of the file (the non-`io_worker` content). The diff indicates `73d00c4` adds only the class and its member declarations; the rest of the file is functionally the same. If `73d00c4` adds new members or changes the class in other ways, incorporate both changes.

---

### Conflict File 5: `src/memory/common.cpp`

**Group responsible:** Group 1 (commit `e23f3a2` — cross-device pool peer access) + Group 2 (commit `e4db3d8` — P2P probe)

**What HEAD adds (Groups 1+2 combined):**
- `enable_pool_peer_access_for_all_visible_devices()` helper function (Group 1, ~30 lines)
- `p2p_dma_supported(src, dst)` function with empirical probe (~120 lines: `g_p2p_supported`, `g_p2p_probed`, `p2p_probe_mutex()`, `run_p2p_probe_locked()`, probe entry point) — this entire block is absent at `73d00c4`
- `#include <mutex>` (for the probe mutex)

**What `73d00c4` has:** `common.cpp` at `73d00c4` is SHORT (~18 lines of actual content per the diff showing HEAD→73d00c4 removes ~230 lines). The `enable_pool_peer_access_for_all_visible_devices()` helper and entire P2P probe block are gone.

**Conflict type:** ADDITIVE (our side adds, their side removes). Since `73d00c4` removes content that we added on top of an earlier `origin/main`, the 3-way merge will show all our additions as "ours only" — the baseline doesn't have them and neither does the target.

**Resolution approach (D-D1 — prefer ours):**
Re-apply the full content from HEAD's `common.cpp` additions. Specifically:
1. Re-add `#include <mutex>`.
2. Re-add `enable_pool_peer_access_for_all_visible_devices()` in the anonymous namespace.
3. Re-add the entire P2P probe block (`kMaxDevices`, `g_p2p_supported`, `g_p2p_probed`, `p2p_probe_mutex()`, `run_p2p_probe_locked()`, `p2p_dma_supported()`).

These additions have NO interaction with `73d00c4`'s `common.cpp` structure (which is minimal). Place them in the anonymous namespace at the top of the `cucascade::memory` namespace.

---

### Conflict File 6: `src/memory/memory_space.cpp`

**Group responsible:** Group 1 (commit `e23f3a2` — drop pool priming + cross-device pool peer access)

**What HEAD changes:**
- GPU memory space construction: removes the initial-pool-size argument to `cuda_async_memory_resource` (prevents pool priming from exhausting GPU memory across multiple spaces).
- Calls `enable_pool_peer_access_for_all_visible_devices(pool_handle, config.device_id)` after pool construction.

**What `73d00c4` has:** Pool construction passes `config.memory_capacity` as `initial_pool_size` (the pre-fix state that causes priming). Does NOT call `enable_pool_peer_access_for_all_visible_devices` (it doesn't exist there).

**Conflict type:** SEMANTIC — two different pool construction strategies. `73d00c4` passes initial capacity; our fix deliberately omits it plus adds peer-access calls.

**Resolution approach (D-D2 — our intent, translated against new shape):**
1. Keep `73d00c4`'s `concrete_mr` construction WITHOUT the `config.memory_capacity` initial_pool_size argument (our Group 1 fix).
2. Add back the `enable_pool_peer_access_for_all_visible_devices(pool_handle, config.device_id)` call after `pool_handle` is set.
3. The `get_chunked_resource_info()` method added by `73d00c4` — keep it; it has no interaction with our pool-priming fix.
4. The `get_default_allocator()` reformatting in `73d00c4` — accept their formatting change.

**Summary of the diff between `73d00c4` and the post-rebase target:**
```cpp
// 73d00c4 has (WRONG — primes pool):
rmm::mr::cuda_async_memory_resource concrete_mr(config.memory_capacity);
pool_handle = concrete_mr.pool_handle();
_allocator  = cuda::mr::any_resource<...>(std::move(concrete_mr));

// Post-rebase must have (our fix preserved):
rmm::mr::cuda_async_memory_resource concrete_mr;   // no initial_pool_size
pool_handle = concrete_mr.pool_handle();
enable_pool_peer_access_for_all_visible_devices(pool_handle, config.device_id);
_allocator  = cuda::mr::any_resource<...>(std::move(concrete_mr));
```

---

### Portable/Mapped Flag Files

**Groups responsible:** Group 1 (commits `1fff85d + 3743621 + 2dcab24 + ff14ff4`)

The Portable/Mapped flags live in `src/memory/small_pinned_host_memory_resource.cpp` and `src/memory/numa_region_pinned_host_allocator.cpp`. These files are NOT in the 6 conflict files listed in the additional context. Confirmation from direct inspection:

- At `73d00c4`, `small_pinned_host_memory_resource.cpp` uses `::cudaMallocHost(&ptr, bytes)` (no Portable flag).
- At HEAD, the same file uses `::cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped)`.
- At `73d00c4`, `numa_region_pinned_host_allocator.cpp` uses `cudaHostAllocDefault` and `cudaHostRegisterMapped` (no Portable).
- At HEAD, the same file uses `cudaHostAllocPortable | cudaHostAllocMapped` and `cudaHostRegisterPortable | cudaHostRegisterMapped`.

Since `73d00c4` touches NEITHER of these files (they are not in the 6 conflict files), the rebase will apply Group 1's changes to these files cleanly without conflicts. **These are NOT true conflict files** — the rebase engine will apply the Group 1 patch to them automatically.

**However, the `src/memory/common.cpp` and `src/memory/memory_space.cpp` files (which ARE conflict files) contain Group 1 content as well. See the conflict file analysis above for those.**

**Grep verification post-rebase:**
```bash
grep -n "cudaHostAllocPortable" cucascade/src/memory/small_pinned_host_memory_resource.cpp
grep -n "cudaHostAllocPortable" cucascade/src/memory/numa_region_pinned_host_allocator.cpp
```
Both must return non-empty. These are the canonical Portable flag sites.

---

## Section B: API Delta — HEAD vs. Post-#117 `gpu_data_representation.hpp`

### HEAD (`62e0517`) public API

```cpp
// Includes: rmm/cuda_stream_view.hpp, cuda_runtime.h
class gpu_table_representation : public idata_representation {
 public:
  // 3-arg ctor (our addition):
  gpu_table_representation(std::unique_ptr<cudf::table> table,
                           cucascade::memory::memory_space& memory_space,
                           rmm::cuda_stream_view writer_stream);

  ~gpu_table_representation() override;  // destroys _writer_event

  std::size_t get_size_in_bytes() const override;
  std::size_t get_uncompressed_data_size_in_bytes() const override;
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

  const cudf::table& get_table() const;          // GONE at 73d00c4 (replaced by get_table_view)
  std::unique_ptr<cudf::table> release_table();  // GONE at 73d00c4 (requires stream arg now)

  void record_writer_event(rmm::cuda_stream_view writer_stream);  // our addition
  [[nodiscard]] cudaEvent_t get_writer_event() const;             // our addition

 private:
  std::unique_ptr<cudf::table> _table;
  cudaEvent_t _writer_event{nullptr};  // our addition
};
```

### Post-#117 (`73d00c4`) public API

```cpp
// Includes: cudf/table/table_view.hpp, any, variant, cstddef
class gpu_table_representation : public idata_representation {
 public:
  // 2-arg simple ctor (base, no stream):
  gpu_table_representation(std::unique_ptr<cudf::table> table,
                           cucascade::memory::memory_space& memory_space);

  // PR #116 template ctor (no stream):
  template<typename Owner>
  gpu_table_representation(cudf::table_view table_view,
                           Owner&& owner,
                           std::size_t alloc_size,
                           cucascade::memory::memory_space& memory_space);

  std::size_t get_size_in_bytes() const override;
  std::size_t get_uncompressed_data_size_in_bytes() const override;
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

  cudf::table_view get_table_view() const;                             // NEW at #117
  std::unique_ptr<cudf::table> release_table(rmm::cuda_stream_view);  // stream arg added

  // NO record_writer_event, NO get_writer_event, NO _writer_event

 private:
  struct owning_table_view { std::any owner; std::size_t alloc_size{0}; cudf::table_view view; };
  std::variant<std::unique_ptr<cudf::table>, owning_table_view> _table;
};
```

### Target post-rebase API (combining D-B1/D-B2/D-D3)

```cpp
// Includes: all from both + rmm/cuda_stream_view.hpp, cuda_runtime.h
class gpu_table_representation : public idata_representation {
 public:
  // 3-arg ctor (our writer_stream REQUIRED, D-B2):
  gpu_table_representation(std::unique_ptr<cudf::table> table,
                           cucascade::memory::memory_space& memory_space,
                           rmm::cuda_stream_view writer_stream);

  // PR #116 template ctor WITH writer_stream (D-B2: both ctors get it):
  template<typename Owner>
  gpu_table_representation(cudf::table_view table_view,
                           Owner&& owner,
                           std::size_t alloc_size,
                           cucascade::memory::memory_space& memory_space,
                           rmm::cuda_stream_view writer_stream);

  ~gpu_table_representation() override;  // destroys _writer_event

  std::size_t get_size_in_bytes() const override;
  std::size_t get_uncompressed_data_size_in_bytes() const override;
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

  cudf::table_view get_table_view() const;                             // from #117
  std::unique_ptr<cudf::table> release_table(rmm::cuda_stream_view);  // from #117

  void record_writer_event(rmm::cuda_stream_view writer_stream);  // from our Group 4
  [[nodiscard]] cudaEvent_t get_writer_event() const;             // from our Group 4

 private:
  struct owning_table_view { std::any owner; std::size_t alloc_size{0}; cudf::table_view view; };
  std::variant<std::unique_ptr<cudf::table>, owning_table_view> _table;  // from #117
  cudaEvent_t _writer_event{nullptr};  // from our Group 4, appended after _table
};
```

**Key difference from HEAD that requires attention in the converter:**
`get_table()` is gone. All uses of `gpu_source.get_table()` in the converter must become `gpu_source.get_table_view()`. At HEAD there are 4 such call sites (lines 160, 545, 866, 1539). After rebase these must all use `get_table_view()`.

---

## Section C: `read_only_data_batch::get_writer_event()` Proxy Implementation

### How `read_only_data_batch` accesses its representation at `73d00c4`

From direct inspection of `data_batch.hpp` at `73d00c4`:

```cpp
class read_only_data_batch {
 public:
  idata_representation* get_data() const { return _batch->get_data(); }
  memory::memory_space* get_memory_space() const { return _batch->get_memory_space(); }
  // ...
 private:
  friend class data_batch;
  std::shared_ptr<data_batch> _batch;        // parent (destroyed second)
  std::shared_lock<std::shared_mutex> _lock; // shared lock (destroyed first)
};
```

`read_only_data_batch` has a `_batch` field (shared_ptr to `data_batch`). `data_batch` has a private `get_data()` method that returns `idata_representation*`. `read_only_data_batch` exposes `get_data()` as a public passthrough (`return _batch->get_data()`).

### Proxy implementation (D-B3)

The implementation decision from the context is:
```cpp
// In class read_only_data_batch (data_batch.hpp):
[[nodiscard]] cudaEvent_t get_writer_event() const
{
  auto* repr = get_data();  // returns idata_representation*
  if (!repr) { return nullptr; }
  auto* gpu_repr = dynamic_cast<gpu_table_representation*>(repr);
  if (!gpu_repr) { return nullptr; }
  return gpu_repr->get_writer_event();
}
```

**Is a downcast needed?** Yes. `get_writer_event()` lives on `gpu_table_representation` (D-B1), not on `idata_representation`. A dynamic_cast is the correct mechanism. The nullptr guard for non-GPU representations is essential because `read_only_data_batch` can hold any representation type (host, disk, etc.).

**Alternative: virtual `get_writer_event()` on `idata_representation`?**

If `idata_representation` exposed a virtual `get_writer_event()` with a default-nullptr return, no downcast would be needed. This would slightly relax D-B1 (putting the declaration on the interface, not just on the concrete class). This is a potential refinement flagged here as a low-priority alternative. The decision D-B1 explicitly places the API only on `gpu_table_representation`, so the downcast approach is the locked implementation. Do not modify D-B1 within Phase 16.

**Where to add this in `data_batch.hpp`:** After `get_memory_space()` in the `read_only_data_batch` public section, before the clone operations. Add `#include <cucascade/data/gpu_data_representation.hpp>` at the top of `data_batch.hpp` only if it is not already included (it is unlikely to be — check the include list of `data_batch.hpp` at `73d00c4`: it includes `common.hpp`, `representation_converter.hpp`, `memory/common.hpp` — no `gpu_data_representation.hpp`). If adding the include creates a circular dependency, move the proxy implementation to `data_batch.cpp` with a forward-declared downcast helper, or add a freestanding inline helper.

**P1 deadlock concern (from CONTEXT.md decisions):** The proxy calls `get_data()` which delegates to `_batch->get_data()`. This is a private method called only through a friend accessor (`read_only_data_batch`). The shared lock is already held by `_lock`. There is NO additional mutex acquisition in `get_data()` → `get_writer_event()` chain. The `_writer_event` member is a `cudaEvent_t` (a pointer-sized value). Reading it while holding the shared lock is safe. No deadlock risk in the proxy itself.

---

## Section D: Rebase Command Sequence

### Prerequisites

Working directory: `cucascade/` inside the Sirius worktree.

```bash
cd /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade
```

### Step 1: Confirm merge-base is `edd6f03`

```bash
git merge-base HEAD origin/main
# Expected output: edd6f03c5b3344812094756d0c4720f2de72fb40
```

The 11 local commits are between `edd6f03` (exclusive) and `62e0517` (HEAD, inclusive):
```bash
git log --oneline edd6f03..HEAD
# Expected: 11 commits in reverse-chronological order
```

### Step 2: Squash 11 commits into 4 group commits

```bash
git rebase -i edd6f03
```

In the interactive editor, reorder and squash as follows:
```
pick 1fff85d fix(memory): pin host memory with Portable flag for multi-GPU DMA
squash 3743621 fix(memory): make cudaMallocHost sites Portable-aware too
squash 2dcab24 fix(memory): add cudaHostAllocMapped to all pinned allocation sites
squash ff14ff4 fix(memory): per-instance ptds_allocation_tracker thread_local
squash e23f3a2 fix(memory): drop pool priming + add cross-device pool peer access

pick 7ed84f2 fix(representation_converter): use target-bound stream in host->gpu and gpu->gpu converters
squash cc2a53d wip(memory): pass source mr to cudf::pack + default-pool peer access
squash e4db3d8 fix(p2p): probe peer DMA at init; route convert_gpu_to_gpu accordingly

pick eda349a fix(pipeline_io_backend): reorder io_worker members so _thread is last

pick 7409c60 fix(stream-lineage): add gpu_table_representation::{record,get}_writer_event + cudaStreamWaitEvent in convert_gpu_to_gpu
squash 62e0517 fix(stream-lineage): require writer_stream in gpu_table_representation constructor
```

**Note on commit ordering:** `git log --oneline edd6f03..HEAD` shows commits in reverse order (newest first). The `git rebase -i` editor shows them in chronological order (oldest first, same as `git log --reverse`). The oldest commit (at the top of the rebase editor) is `1fff85d`.

**Commit message template for each group (use `--reword` or the squash editor):**

Group 1 message:
```
fix(memory): memory hygiene — Portable/Mapped pinning, ptds tracker, pool peer access

Squash of 5 fixes onto cucascade 73d00c4 for Sirius multi-GPU v1.4.
Original commits: 1fff85d 3743621 2dcab24 ff14ff4 e23f3a2

- Pin host memory with cudaHostAllocPortable for multi-GPU DMA accessibility
- Make cudaMallocHost sites Portable-aware
- Add cudaHostAllocMapped to all pinned allocation sites
- Per-instance ptds_allocation_tracker (thread_local, not process-global)
- Drop pool priming (prevents multi-space GPU memory exhaustion)
- Add cross-device pool peer access at construction time

Co-Authored-By: Felipe Aramburu <faramburu@nvidia.com>
```

Group 2 message:
```
fix(representation_converter): P2P override — target-bound stream, DMA probe at init

Squash of 3 fixes onto cucascade 73d00c4 for Sirius multi-GPU v1.4.
Original commits: 7ed84f2 cc2a53d e4db3d8

- Use target-bound stream in host->gpu and gpu->gpu converters (v1.1 P2P fix)
- Pass source mr to cudf::pack + default-pool peer access (WIP carry)
- Empirical P2P peer DMA probe at init; route convert_gpu_to_gpu to real peer
  DMA on server hardware and host-staging on consumer chipsets (Intel lying-enable)

Co-Authored-By: Felipe Aramburu <faramburu@nvidia.com>
```

Group 3 message:
```
fix(pipeline_io_backend): reorder io_worker members so _thread is last

Squash of 1 fix onto cucascade 73d00c4 for Sirius multi-GPU v1.4.
Original commit: eda349a

std::thread _thread must be declared AFTER _mutex and _cv. C++ destroys members
in reverse-declaration order; if _thread is first, it joins while _mutex is
still live but _cv may have been destroyed if the order were reversed. Avoids
SIGTERM/EINVAL on io_worker teardown under parallel test runs.

Co-Authored-By: Felipe Aramburu <faramburu@nvidia.com>
```

Group 4 message:
```
fix(stream-lineage): writer_stream/writer_event on gpu_table_representation + cudaStreamWaitEvent

Squash of 2 fixes onto cucascade 73d00c4 for Sirius multi-GPU v1.4.
Original commits: 7409c60 62e0517

Phase 13 fix: closes SF100 Q11 2-GPU illegal-address race.
- Add record_writer_event/get_writer_event accessors on gpu_table_representation
- require writer_stream as a REQUIRED ctor argument (compile-time enforced)
- convert_gpu_to_gpu: cudaStreamWaitEvent(target_stream, src.get_writer_event())
  before peer copy; fallback to cudaDeviceSynchronize for un-migrated callers
- Replaces cudf::pack path with column-tree walk (avoids stream-ordered race
  in compute_splits scratch allocations)

Co-Authored-By: Felipe Aramburu <faramburu@nvidia.com>
```

### Step 3: Rebase 4 group commits onto `origin/main`

After the squash succeeds (no conflicts expected at squash time — all 11 commits operate on the same base):
```bash
git rebase origin/main
```

This will pause at conflict-resolution checkpoints, one per group commit. Resolve in order:

**Round 1 (Group 1 — memory hygiene):**
Files with conflicts: `src/memory/common.cpp`, `src/memory/memory_space.cpp`.
Resolution per Section A above. Verify:
```bash
grep -n "enable_pool_peer_access_for_all_visible_devices\|p2p_dma_supported" src/memory/common.cpp
grep -n "enable_pool_peer_access_for_all_visible_devices" src/memory/memory_space.cpp
```
Then:
```bash
git add src/memory/common.cpp src/memory/memory_space.cpp
git rebase --continue
```

**Round 2 (Group 2 — stream/converter):**
Files with conflicts: `src/data/representation_converter.cpp`.
Resolution per Section A above: keep `73d00c4`'s `get_table_view()` API, re-apply P2P probe routing in `convert_gpu_to_gpu`. Note: Group 4 will later fully rewrite `convert_gpu_to_gpu`, so at this group it's acceptable to have a partial/provisional `convert_gpu_to_gpu` as long as the P2P routing logic compiles.
```bash
git add src/data/representation_converter.cpp
git rebase --continue
```

**Round 3 (Group 3 — pipeline):**
Files with conflicts: `src/data/pipeline_io_backend.cpp`.
Resolution per Section A above: reorder `io_worker` members. Verify `_thread` is last.
```bash
git add src/data/pipeline_io_backend.cpp
git rebase --continue
```

**Round 4 (Group 4 — stream-lineage):**
Files with conflicts: `include/cucascade/data/gpu_data_representation.hpp`, `src/data/gpu_data_representation.cpp`, `src/data/representation_converter.cpp`.
This is the most complex round. Resolution per Sections A and B above.
- Re-implement `gpu_data_representation.hpp` as shown in Section B "target post-rebase API".
- Re-implement `gpu_data_representation.cpp` with both RAII variant logic (from #117) and writer_event logic (from Group 4).
- Re-implement `representation_converter.cpp`'s `convert_gpu_to_gpu` as the column-tree walk from HEAD with `get_table_view()` (not `get_table()`).
- Also add the `get_writer_event()` proxy to `read_only_data_batch` in `data_batch.hpp` (D-B3) — this file has no conflict, but the addition is part of this group's intent.

```bash
git add include/cucascade/data/gpu_data_representation.hpp
git add src/data/gpu_data_representation.cpp
git add src/data/representation_converter.cpp
git add include/cucascade/data/data_batch.hpp   # proxy addition (no conflict, but modified)
git rebase --continue
```

### Step 4: Build verification inside cucascade

After rebase completes with 4 commits on top of `73d00c4`:
```bash
# Build cucascade inside pixi env
cd /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272
# Use MCP: mcp__project-commands__run_command build
# Or if building cucascade standalone:
cd cucascade && CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) cmake --preset release 2>/dev/null || \
  cmake -B build/release -DCMAKE_BUILD_TYPE=Release . && cmake --build build/release -j$(nproc)
```

Build failures expected if `get_table()` call sites remain anywhere. Fix each one by changing to `get_table_view()`.

### Step 5: Run grep gates (CC-04 pre-check)

```bash
# Must be non-empty:
grep -rn "record_writer_event\|get_writer_event" include/cucascade/data/

# Must be non-empty:
grep -rn "cudaHostAllocPortable" src/memory/

# Must return zero lines:
grep -rn "task_created\|in_transit" src/data/

# Must show writer_stream in converter construction sites:
grep -n "gpu_table_representation" src/data/representation_converter.cpp | grep -v "^.*//.*gpu_table"
```

### Step 6: Update submodule pin in parent Sirius repo

After the cucascade rebase is complete and build-verified:
```bash
cd /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272
# The submodule HEAD is now on the new 4-commit tip
git add cucascade
git commit -m "chore(submodule): advance cucascade pin to post-#117 rebase tip (Phase 16 CC-01)"
```

This advances the `.gitmodules`-tracked pin from `62e0517` to the new local hash.

### Author attribution mechanics

To author squashed commits with `Co-Authored-By` trailers, use the commit message HEREDOC approach during the rebase interactive session:

```bash
# When git rebase -i opens the EDITOR for a squash commit message:
# Clear the default squash message and replace with:
git commit --amend --reset-author -m "$(cat <<'EOF'
fix(memory): memory hygiene — Portable/Mapped pinning, ptds tracker, pool peer access
...
Co-Authored-By: Felipe Aramburu <faramburu@nvidia.com>
EOF
)"
```

Alternatively, prepare commit message files in `$TMPDIR` and use `git commit --amend -F /path/to/msg.txt` after each squash completes.

---

## Section E: Cucascade ctest Scope and Failure Modes

### ctest targets

At `73d00c4`, the cucascade test suite is a SINGLE ctest target (`cucascade_tests`) that runs all tests in one executable. From `test/CMakeLists.txt` at `73d00c4`:

```
add_test(NAME cucascade_tests COMMAND cucascade_tests)
```

The test sources at `73d00c4` include `test_bandwidth_profiler.cpp` (PR #112 addition) which is NOT in the current HEAD test list. The HEAD `test/CMakeLists.txt` does NOT include `test_bandwidth_profiler.cpp`.

**Full test source list at `73d00c4`:**
- `test/data/test_data_batch.cpp` (1558 lines — PR #117's RAII tests; **this is the main new test**)
- `test/data/test_bandwidth_profiler.cpp` (PR #112 addition)
- `test/data/test_data_repository.cpp`
- `test/data/test_data_repository_manager.cpp`
- `test/data/test_data_representation.cpp`
- `test/data/test_disk_io_backend.cpp`
- `test/data/test_disk_host_converters.cpp`
- `test/data/test_gpu_disk_converters.cpp`
- `test/data/test_representation_converter.cpp`
- `test/memory/test_memory_reservation_manager.cpp`
- `test/memory/test_small_pinned_host_memory_resource.cpp`
- `test/memory/test_topology_discovery.cpp`
- `test/memory/test_gpu_kernels.cu`
- `test/unittest.cpp`

**ctest run command:**
```bash
cd cucascade/build/release   # or wherever the build directory is
ctest --output-on-failure -j1
```

Run with `-j1` because GPU tests are not safe to parallelize (device isolation not guaranteed under ctest without explicit labels).

### Expected test failures if writer_event re-attachment is buggy

`test_data_batch.cpp` at `73d00c4` constructs `gpu_table_representation` with the 2-arg ctor:
```cpp
auto gpu_repr = std::make_unique<gpu_table_representation>(
  std::make_unique<cudf::table>(std::move(table)), *gpu_space);
```

After our rebase, the 2-arg ctor no longer exists — it becomes 3-arg (writer_stream required). This means ALL `test_data_batch.cpp` construction sites will fail to compile unless the writer_stream argument is added.

**This is expected and is the correct behavior** — Phase 16's intent is compile-time enforcement of the writer_stream requirement (D-B2). The fix is to add a default stream (e.g., `rmm::cuda_stream_view{}` or a test-local `rmm::cuda_stream`) to each construction call in the test file.

**Failure shapes to anticipate:**
1. **Compile error:** `gpu_table_representation` 2-arg ctor no longer exists → fix by adding `rmm::cuda_stream_view{}` as 3rd arg to each test site.
2. **Runtime: writer_event is nullptr in conversion tests:** If the test constructs with `rmm::cuda_stream_view{}` (null stream), no event is recorded (expected for test code that doesn't exercise P2P paths). The test itself won't fail — it will take the fallback `cudaDeviceSynchronize` path.
3. **Runtime: `test_representation_converter.cpp` GPU-to-GPU test fails:** If the 3-arg ctor is missing from the converter's construction call in `representation_converter.cpp`, `convert_gpu_to_gpu` produces a representation with `nullptr` writer_event. The ctest won't fail for correctness (the fallback sync fires), but the test's intent is weakened. Check with:
   ```bash
   grep -n "gpu_table_representation" cucascade/src/data/representation_converter.cpp
   ```
   Every construction call must have 3 args including the stream.
4. **Runtime: `test_data_batch.cpp` RAII accessor tests:** These test `to_read_only()`, `to_mutable()`, `readonly_to_mutable()`, etc. These tests do NOT involve `gpu_table_representation` directly — they use `mock_data_representation`. They should pass without modification.

### Expected runtime

Based on typical cucascade test suites on RTX 6000 Ada hardware with T4-class tests:
- Memory tests: ~5-10 seconds
- Data batch tests (RAII, no GPU data): ~5 seconds
- GPU conversion tests (H2D, D2H, GPU clone): ~15-30 seconds
- Disk I/O tests: ~30-60 seconds (pipeline backend with actual disk I/O)
- Total: 60-120 seconds for the full suite

### Hidden / disabled tests for stream-lineage

At `73d00c4`, `test_data_batch.cpp` does NOT contain any test for `record_writer_event` or `get_writer_event` — those are our Phase 16 additions. After the rebase, there are no pre-existing upstream tests for writer_event. Phase 16's CC-04 ceiling is "cucascade ctest passes" — not "stream-lineage is explicitly tested by cucascade tests." The stream-lineage is verified by the grep gate and later by SF100 Q11 in Phase 21.

No hidden tests need un-hiding for stream-lineage coverage within Phase 16 scope.

---

## Section F: Submodule Pin Update Mechanics

### After rebase: git sequence in parent worktree

```bash
# Inside cucascade/, verify the new tip:
git log --oneline -5
# Should show: 4 commits on top of 73d00c4

# Confirm ancestry:
git merge-base --is-ancestor 73d00c4 HEAD && echo "PASS: 73d00c4 is an ancestor"

# Return to parent worktree:
cd /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272

# Check submodule status:
git submodule status cucascade
# Should show: +<old_hash> cucascade (commits to advance)

# Stage the submodule pin advance:
git add cucascade

# Commit:
git commit -m "$(cat <<'EOF'
chore(submodule): advance cucascade pin to post-#117 rebase tip (Phase 16 CC-01)

Rebases 11 local cucascade fixes onto origin/main 73d00c4 (PR #117 RAII DataBatch
+ PR #112 bandwidth profiler + PR #116 gpu_data_representation from cudf::table_view).
Squashed into 4 group commits. Local-only pin per D-A3 (CC-UPSTREAM-01 deferred).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

### Local-only pin and `.gitmodules`

The `.gitmodules` entry for cucascade is:
```
[submodule "cucascade"]
    path = cucascade
    url = https://github.com/NVIDIA/cuCascade.git
    branch = main
```

The `branch = main` field is used by `git submodule update --remote` to know which remote branch to track. It does NOT prevent a local commit hash from being pinned. Git submodules always pin to a specific commit hash in the parent repo's tree object — the `branch` field is advisory for `--remote` only.

**D-A3 is fully supported:** A free-floating local commit hash (not pushed to any remote) works correctly as a submodule pin. `git submodule update --init --recursive` in a fresh clone will fail to fetch the hash (since it's not on the remote), but `git submodule update` in an existing worktree that already has the local cucascade repo will succeed if the cucascade remote is already cloned and the local commit exists in `.git/objects/`.

**Implication for re-clones:** Anyone who clones the Sirius repo fresh cannot resolve this pin from the cucascade remote (since the local hash is not pushed). They must redo the rebase locally. This is the accepted risk per D-A3. Document in `16-rebase-log.md`.

---

## Section G: Pitfalls Deep-Dive

### P2 — `writer_stream` Lost in `representation_converter.cpp`

**Where to grep for survival check:**
```bash
# Must be non-empty — checks converter construction sites:
grep -n "gpu_table_representation(" cucascade/src/data/representation_converter.cpp | grep -v "^.*//.*gpu_table"
```
Each matching line must show 3 arguments (the third being a stream variable like `target_stream`, `stream`, etc.), NOT just 2 arguments.

**Specific construction sites to verify (line numbers are HEAD-approximate; will shift after rebase):**
| Function | Expected 3rd arg |
|----------|-----------------|
| `convert_gpu_to_gpu` (~line 886 at HEAD) | `target_stream` |
| `convert_host_to_gpu` (~line 289 at HEAD) | `stream` |
| `convert_host_fast_to_gpu` (~line 853 at HEAD) | `stream` |
| `convert_disk_to_gpu` (~line 1453 at HEAD) | `stream` |

**Diff-line count indicator:** If the post-rebase `representation_converter.cpp` is shorter than 1798 lines (HEAD) by more than 100 lines, investigate — the column-tree-walk `convert_gpu_to_gpu` (Group 4, ~90 lines) may have been dropped. The post-rebase file should be in the range 1500-1900 lines.

**Warning sign:** `grep -n "gpu_table_representation.*stream" src/data/representation_converter.cpp` returns fewer than 4 matches.

---

### P8 — `io_worker` Member-Order

**Verification command post-rebase:**
```bash
# Extract io_worker class member declarations (last ~8 lines of private section):
grep -n "_mutex\|_cv\|_pending_work\|_pending_promise\|_has_task\|_shutdown\|_thread" \
  cucascade/src/data/pipeline_io_backend.cpp | tail -7
```

Expected output (line numbers vary, but ORDER must be):
```
N:  std::mutex _mutex;
N:  std::condition_variable _cv;
N:  std::function<void()> _pending_work;
N:  std::promise<void> _pending_promise;
N:  bool _has_task{false};
N:  bool _shutdown{false};
N:  std::thread _thread;   # MUST be last
```

**Destructor ordering matters:** The destructor does:
```cpp
{ _shutdown = true; } _cv.notify_one(); _thread.join();
```
It acquires `_mutex` to set `_shutdown`, then joins `_thread`. If `_thread` is declared BEFORE `_mutex`, then in destruction order `_thread` is destroyed first — joining happens BEFORE `_mutex` and `_cv` are destroyed, which is safe. But if `_mutex` is destroyed before `_thread` joins (reversed order), the running thread can call `pthread_mutex_lock` on a destroyed mutex → `EINVAL` → `std::terminate`. The fix (Group 3) moves `_thread` to last so it is destroyed last, ensuring `_mutex` and `_cv` outlive the join.

---

### P9 — Portable/Mapped Flags

**Verification commands:**
```bash
# small_pinned_host_memory_resource.cpp — must show Portable|Mapped:
grep -n "cudaHostAlloc\|cudaMallocHost" cucascade/src/memory/small_pinned_host_memory_resource.cpp
# Expected: cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped)

# numa_region_pinned_host_allocator.cpp — must show Portable|Mapped:
grep -n "cudaHostAlloc\|cudaMallocHost\|cudaHostRegister" cucascade/src/memory/numa_region_pinned_host_allocator.cpp
# Expected:
#   cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped)
#   cudaHostRegister(ptr, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped)
```

These two files are NOT in the 6 conflict files, so the rebase should apply Group 1's changes automatically. If the grep shows the old flags (`cudaHostAllocDefault` or plain `cudaMallocHost`), the rebase failed to apply the patch cleanly — investigate with `git log --all --oneline -- src/memory/small_pinned_host_memory_resource.cpp`.

**Why both flags matter:**
- `cudaHostAllocPortable`: makes memory DMA-accessible from ALL CUDA contexts (both GPU 0 and GPU 1). Without it, only the allocating context's GPU can DMA to/from the buffer.
- `cudaHostAllocMapped`: maps the host memory into the device address space for `cudaMemcpyAsync` zero-copy paths.

---

### P1 — RAII Lock Scope

Phase 16 introduces one new code site where RAII locking occurs: the `read_only_data_batch::get_writer_event()` proxy (D-B3). The proxy implementation:
```cpp
cudaEvent_t get_writer_event() const {
  auto* repr = get_data();
  if (!repr) { return nullptr; }
  auto* gpu_repr = dynamic_cast<gpu_table_representation*>(repr);
  if (!gpu_repr) { return nullptr; }
  return gpu_repr->get_writer_event();
}
```

This method is called while holding the shared lock (`_lock` in `read_only_data_batch`). The call chain is: `get_data()` → `_batch->get_data()` → returns `_data.get()` (no lock acquisition). Then `dynamic_cast` (no lock). Then `get_writer_event()` on `gpu_table_representation` → reads `_writer_event` (a single `cudaEvent_t` member — no lock). **No nested lock acquisition anywhere in the chain.** P1 self-deadlock cannot occur from this proxy.

The P1 risk is more relevant for Phase 18 (the Sirius-side RAII migration). In Phase 16, the only RAII API exposure is the proxy method above, which is deadlock-safe.

---

### P7 — PR #739 × #117 Ordering Guard

**After Phase 16 rebase completes, verify:**
```bash
# Inside cucascade/:
git log --oneline | head -10
# Must show our 4 group commits on top of 73d00c4, with NO commit that includes
# "Compat/update cucascade" or bumps to 0cd4a6a

# Ancestry check:
git merge-base --is-ancestor 73d00c4 HEAD && echo "PASS: 73d00c4 is ancestor of HEAD"

# No #739-shaped content on cucascade branch:
git log --oneline | grep -i "compat\|0cd4a6a"
# Must return EMPTY
```

**In the parent Sirius repo (for Phase 17 guard):**
```bash
# After Phase 16, before Phase 17 dev-merge:
git log --oneline --all | grep "468f6e1"
# Should show on origin/dev, NOT on feature/single-node-multi-gpu2
# PR #739 must NOT be cherry-picked onto our branch
```

---

## Section H: Verification Recipes (CC-04 Grep Gates)

### Gate 1: writer_event API exists in cucascade headers

```bash
grep -rn "record_writer_event\|get_writer_event" \
  /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade/include/cucascade/data/
```
Expected: non-empty (at minimum, `gpu_data_representation.hpp` and `data_batch.hpp` should match).

### Gate 2: Portable flags survive

```bash
grep -rn "cudaHostAllocPortable" \
  /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade/src/memory/
```
Expected: 2+ matches (one in `small_pinned_host_memory_resource.cpp`, one in `numa_region_pinned_host_allocator.cpp`).

### Gate 3: Old FSM state names gone

```bash
grep -rn "task_created\|in_transit" \
  /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade/src/data/
```
Expected: ZERO matches. `batch_state::task_created` and `batch_state::in_transit` were removed by PR #117.

### Gate 4: cucascade pin is descended from `73d00c4`

```bash
cd /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade
git merge-base --is-ancestor 73d00c4 HEAD && echo "CC-01 PASS" || echo "CC-01 FAIL"
```

### Gate 5: 4 group commits above `73d00c4`

```bash
cd /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade
git log --oneline 73d00c4..HEAD | wc -l
# Expected: 4
```

### Gate 6: writer_stream in converter construction sites

```bash
cd /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade
grep -n "make_unique<gpu_table_representation>" src/data/representation_converter.cpp
# Must show all construction sites use 3-arg ctor (stream argument present)
```

### Gate 7: io_worker _thread is last member

```bash
cd /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade
awk '/class io_worker/,/^};/' src/data/pipeline_io_backend.cpp | grep "_thread\|_mutex\|_cv"
# _thread must appear AFTER _mutex and _cv in the output
```

### Gate 8: get_table() removed (compiler-verified, but can grep too)

```bash
cd /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade
grep -rn "\.get_table()" src/ include/
# Must return ZERO (all usages should be get_table_view() post-rebase)
```

### Full CC-04 verification script

```bash
cd /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade

echo "=== CC-04 Grep Gates ==="

echo -n "Gate 1 (writer_event API): "
grep -rn "record_writer_event\|get_writer_event" include/cucascade/data/ | wc -l | \
  xargs -I{} bash -c '[ {} -gt 0 ] && echo PASS || echo FAIL'

echo -n "Gate 2 (Portable flags): "
grep -rn "cudaHostAllocPortable" src/memory/ | wc -l | \
  xargs -I{} bash -c '[ {} -gt 0 ] && echo PASS || echo FAIL'

echo -n "Gate 3 (no old FSM states): "
COUNT=$(grep -rn "task_created\|in_transit" src/data/ | wc -l)
[ "$COUNT" -eq 0 ] && echo PASS || echo "FAIL ($COUNT matches)"

echo -n "Gate 4 (73d00c4 ancestry): "
git merge-base --is-ancestor 73d00c4 HEAD && echo PASS || echo FAIL

echo -n "Gate 5 (4 group commits above 73d00c4): "
COUNT=$(git log --oneline 73d00c4..HEAD | wc -l)
[ "$COUNT" -eq 4 ] && echo PASS || echo "FAIL (found $COUNT commits)"
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Conflict resolution of `representation_converter.cpp` | A merge-tool 3-way merge | Manual re-implementation: start from `73d00c4`, hand-apply Group 4 changes | The file has 3 conflicting authors (73d00c4 + Group 2 + Group 4); merge tools can't determine semantic intent |
| Squash commit author attribution | A custom script | `git rebase -i` with manual EDITOR + `Co-Authored-By` trailers | Git native mechanism; avoids detached-HEAD complications |
| `get_writer_event()` proxy deadlock protection | A custom mutex or extra lock | Direct delegation through the existing shared lock already held | The RAII accessor already holds the shared lock; adding another lock creates P1 risk |
| `cudaEvent_t` thread-safety in proxy | A local mutex on `_writer_event` | Read-only access through shared lock is sufficient | `_writer_event` is written only in ctor (under exclusive lock via mutable accessor); reads via shared lock are safe |

---

## Common Pitfalls

### Pitfall P2: `writer_stream` Silently Dropped from `representation_converter.cpp`

**What goes wrong:** `convert_gpu_to_gpu` constructs `gpu_table_representation` with 2 args (old style). Compiles only if the 2-arg ctor still exists — since we're removing it, this is a compile error. BUT if the 3rd arg is added but points to the wrong stream (e.g., the caller's `stream` instead of `target_stream`), it compiles and runs but doesn't correctly order the cross-device copy.

**How to avoid:** After conflict resolution, re-read the function body manually. The construction site must use `target_stream` (not `stream`). The argument `stream` (the caller's stream) is the writer of the SOURCE representation — `target_stream` is the writer of the RESULT.

**Warning sign:** `grep -n "gpu_table_representation.*stream\b" src/data/representation_converter.cpp | grep "convert_gpu_to_gpu"` shows `stream` instead of `target_stream` as the 3rd arg.

### Pitfall P8: `io_worker` Member Order Silently Reverted

**What goes wrong:** Git's conflict resolution defaults to "theirs" for structural changes, reverting `_thread` to first position.

**Warning sign:** Test-ordering-dependent SIGSEGV appearing only in `[integration][TPC-H]` or `[mgpu_stress]`.

**Detection:** Run the Gate 7 check above before running any tests.

### Pitfall P9: Portable Flags in Non-Conflict Files Silently Skipped

**What goes wrong:** The rebase applies Group 1's patch to `small_pinned_host_memory_resource.cpp` but the patch doesn't apply cleanly (e.g., context lines mismatch due to `73d00c4` editing neighboring lines). Git reports a CONFLICT but with `--strategy-option=ours` or accidental use of `git checkout --theirs` the fix is dropped.

**Detection:** Gate 2 grep above. Run before ctest.

### Pitfall: `get_table()` Remaining in Converter Post-Rebase

**What goes wrong:** HEAD's converter uses `get_table()`. After rebase to `73d00c4`'s `gpu_table_representation` which only has `get_table_view()`, any remaining `get_table()` calls produce compile errors.

**How to avoid:** Gate 8 grep. This is also a compile-time catch.

### Pitfall: `test_data_batch.cpp` 2-arg ctor compile errors

**What goes wrong:** The 1558-line `test_data_batch.cpp` at `73d00c4` constructs `gpu_table_representation` with 2 args throughout. After our rebase makes writer_stream REQUIRED (D-B2), every test construction site becomes a compile error.

**Expected behavior:** This IS expected. Fix by adding `rmm::cuda_stream_view{}` as the 3rd arg at each test construction site. This is approximately 5-10 sites in `test_data_batch.cpp` and another 3-5 in `test_representation_converter.cpp`.

---

## Environment Availability

Step 2.6: SKIPPED — Phase 16 is a pure git rebase inside `cucascade/`. No external tools beyond `git` and the pixi environment (already confirmed working from v1.3 testing). The cucascade build uses the same pixi environment as the main Sirius build.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Catch2 v2.13.10 (fetched via FetchContent) |
| Config file | `cucascade/test/CMakeLists.txt` |
| Quick run command | `./cucascade_tests "[data_batch][gpu]"` (GPU data_batch tests only, ~15s) |
| Full suite command | `ctest --output-on-failure -j1` from `cucascade/build/release/` (~90s) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| CC-01 | Pin advanced to 73d00c4-descendant | shell gate | `git merge-base --is-ancestor 73d00c4 HEAD` | N/A |
| CC-02 | 11 fixes preserved | grep gates (H) | Section H one-liners | N/A |
| CC-03 | writer_event re-attached | grep + compile | Gate 1 + build success | ✅ (test_data_batch.cpp after fix) |
| CC-04 | ctest passes | integration | `ctest --output-on-failure -j1` | ✅ (test sources in cucascade/) |

### Sampling Rate
- **Per conflict-resolution round:** Grep gates (Gates 1-3, 6-8) relevant to the resolved group
- **After all 4 rounds:** Full gate battery (Section H script) + `cmake --build`
- **Phase gate:** `ctest --output-on-failure -j1` green before `/gsd:verify-work`

### Wave 0 Gaps
- Test compilation failures in `test_data_batch.cpp` (2-arg ctor sites) and `test_representation_converter.cpp` must be fixed as part of the rebase resolution (not deferred). These are not new test files — they exist but won't compile until the ctor signature mismatch is resolved.

---

## Sources

### Primary (HIGH confidence — direct git inspection)

- `cucascade git show 73d00c4:include/cucascade/data/gpu_data_representation.hpp` — post-#117 class shape (no writer_event, has owning_table_view variant, get_table_view() only)
- `cucascade git show 73d00c4:include/cucascade/data/data_batch.hpp` — RAII `read_only_data_batch`/`mutable_data_batch` class definitions; `_batch` field; `get_data()` passthrough
- `cucascade git show 73d00c4:src/data/representation_converter.cpp` — baseline converter at #117: `get_table_view()` usage, 2-arg ctor construction, `cudf::pack`-based `convert_gpu_to_gpu`
- `cucascade git show HEAD:src/data/representation_converter.cpp` — Phase 13 full `convert_gpu_to_gpu` column-tree-walk + `cudaStreamWaitEvent`
- `cucascade git show HEAD:src/data/pipeline_io_backend.cpp` — Group 3 member-reordering fix applied
- `cucascade git show 73d00c4:src/data/pipeline_io_backend.cpp` — pre-fix `io_worker` with `_thread` first
- `cucascade git diff HEAD 73d00c4 -- src/memory/common.cpp` — confirmed P2P probe block (~230 lines) absent at `73d00c4`
- `cucascade git diff HEAD 73d00c4 -- src/memory/memory_space.cpp` — confirmed pool-priming and peer-access call absent at `73d00c4`
- `cucascade git show 73d00c4:src/memory/small_pinned_host_memory_resource.cpp` — confirmed `cudaMallocHost` without Portable flag at `73d00c4`
- `cucascade git show 73d00c4:src/memory/numa_region_pinned_host_allocator.cpp` — confirmed `cudaHostAllocDefault` at `73d00c4`
- `cucascade git show 73d00c4:test/data/test_data_batch.cpp` — confirmed 1558 lines, uses 2-arg ctor at construction sites
- `cucascade git show 73d00c4:test/CMakeLists.txt` — confirmed single `cucascade_tests` ctest target + `test_bandwidth_profiler.cpp` addition
- `cucascade git merge-base HEAD origin/main` — confirmed `edd6f03` as merge-base
- `cucascade git log --oneline -5` — confirmed origin/main tip is `73d00c4`
- `.gitmodules` — confirmed `branch = main` advisory only; free-floating hash pin is supported

### Secondary (MEDIUM confidence)

- `.planning/research/PITFALLS.md` — Phase 16-relevant pitfalls P1, P2, P7, P8, P9 with full context
- `.planning/research/ARCHITECTURE.md` — v1.3 surface integration map; Surface 7 (stream-lineage) and Surface 8 (SCHED-RR)
- `.planning/research/SUMMARY.md` — overall v1.4 phase sequencing and confidence assessment
- `.planning/phases/16-cucascade-submodule-rebase-pin-recovery/16-CONTEXT.md` — locked decisions

---

## Metadata

**Confidence breakdown:**
- Per-file conflict shape (Section A): HIGH — derived from direct `git show` and `git diff` on all 6 conflict files
- API delta HEAD vs. #117 (Section B): HIGH — direct class member enumeration from both versions
- Proxy implementation (Section C): HIGH — derived from `data_batch.hpp` field inspection at `73d00c4`
- Rebase command sequence (Section D): HIGH — standard git mechanics; specific hashes verified
- ctest scope (Section E): HIGH — `test/CMakeLists.txt` read at `73d00c4`
- Submodule mechanics (Section F): HIGH — `.gitmodules` inspected directly
- Pitfall guards (Section G): HIGH — derived from PITFALLS.md + source confirmation
- Grep gates (Section H): HIGH — all greps derived from verified file content

**Research date:** 2026-05-04
**Valid until:** 2026-06-04 (cucascade origin/main assumed stable; revalidate if upstream pushes to main)
