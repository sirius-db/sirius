# Phase 24 — Upstream Cucascade Diff Triage

**Date:** 2026-05-13
**Analyst:** Claude (Plan 24-01)
**Base SHA (Phase 23 upstream base):** `bcddb89` (Make host memory portable #121)
**Upstream target:** `9ceebaa` (STRING fix #124, on top of `96bfea1` slice-host-table #122)
**Our fork branch:** `fix/pinned-portable-flags` at `9da4047` (8 commits ahead of `bcddb89`)
**Backup branch:** `fix/pinned-portable-flags-pre-phase24-backup` at `9da4047`

---

## D-10 Drift Check

`git log --oneline ^bcddb89 origin/main` result:

```
9ceebaa Fix for: Invalid Error: reconstruct_column: STRING column metadata must have at least one child (offsets) (#124)
96bfea1 feat: adding the ability to slice host table (#122)
49134ff Stop enabling C for C++ and CUDA builds (#123)
```

**Result: 3 commits beyond bcddb89 — within the 5-commit drift limit. Proceed.**

- `49134ff` Stop enabling C for C++ and CUDA builds: already absorbed in Phase 23 as the base (`bcddb89` sits on top of `49134ff` in the upstream log). No new action needed.
- `96bfea1`: new — slice host table feature. Triage below (Section A).
- `9ceebaa`: new — STRING reconstruct fix. Triage below (Section B).

---

## Section A: 96bfea1 — "feat: adding the ability to slice host table (#122)"

**Stat:** 5 files changed, 489 insertions(+), 113 deletions(-)

### Files touched

| File | Change | Risk to our fork |
|------|--------|-----------------|
| `include/cucascade/data/cpu_data_representation.hpp` | +22 lines: new `num_columns()`, `column_size()`, `slice()` methods on `host_data_representation` | LOW — pure additions |
| `include/cucascade/memory/host_table.hpp` | +58 lines: `host_table_allocation` converted from struct to class; constructor goes private; `create()` factory + `slice()` + `clone()` methods added; `allocation` member changes from `unique_ptr<...>` to `shared_ptr<...>` | **HIGH — API break for callers** |
| `src/data/cpu_data_representation.cpp` | +186 lines: implement new `host_table_allocation` methods; `host_data_representation::clone()` simplified via new `clone()` | MEDIUM — our fork doesn't call these directly |
| `src/data/representation_converter.cpp` | +133 lines, −113 lines: API update from `fixed_multiple_blocks_allocation` (unique_ptr alias) to dereferenced `multiple_blocks_allocation`; `host_table_allocation::create()` factory calls replace `make_unique<host_table_allocation>` constructions | **HIGH — direct collision with our commits 3/6/7** |
| `test/data/test_data_representation.cpp` | +130 lines: new slice tests, updated dereferences | LOW — test-only |

### Does `alloc_and_peer_copy_async` still exist post-96bfea1?

**YES — and it is COMPLETELY UNCHANGED.**

`alloc_and_peer_copy_async` and `alloc_and_peer_copy_sync` were introduced by our fork commit `8392c3d` and do NOT exist in the upstream code tree at all (verified: `git show 96bfea1:src/data/representation_converter.cpp | grep alloc_and_peer_copy` returns nothing). These are entirely our invention. Upstream's `convert_gpu_to_gpu` uses `cudf::pack/unpack`; our fork replaces that with column-tree peer-copy reconstruction.

The `96bfea1` diff against `bcddb89` only touches the HOST-tier code paths (GPU→HOST, HOST→GPU, HOST→DISK, DISK→HOST helpers). The P2P path (`alloc_and_peer_copy_async`, `reconstruct_column_p2p`, `convert_gpu_to_gpu` forward declaration) lives in our commits `8392c3d` through `9da4047` and is **not in upstream at all**.

### Does `run_p2p_probe_locked` still exist post-96bfea1?

**YES — and it is in `src/memory/common.cpp`, not in `representation_converter.cpp`.**

`run_p2p_probe_locked` was introduced by our fork commit `8392c3d` (in `common.cpp`) and repaired by `9da4047`. Upstream has no probe function in `common.cpp` at all (verified: `git show 96bfea1:src/memory/common.cpp` does not contain `run_p2p_probe_locked`). This function is 100% our fork's invention. Upstream's `96bfea1` does not touch `common.cpp` at all.

### Does the host-staging fallback path still exist?

The host-staging fallback (`cudaMallocHost` + DtoH + HtoD path in `alloc_and_peer_copy_async`) is present in our fork at lines 610-652. It does NOT exist in upstream — upstream's `convert_gpu_to_gpu` uses `cudf::pack/unpack` which has its own internal staging. Our host-staging path **only exists in our fork's additional code**.

### What does 96bfea1 actually change in `representation_converter.cpp`?

The 96bfea1 diff against bcddb89 touches lines 528–1214 of the `bcddb89` representation_converter.cpp. These are the HOST-tier utility functions:

1. **`collect_d2h_ops`** / **`collect_column_d2h_ops`**: parameter type `fixed_multiple_blocks_allocation&` (which was `unique_ptr<...>`) → `multiple_blocks_allocation&` (the raw type). Access changed from `alloc->method()` to `alloc.method()`.
2. **`alloc_and_schedule_h2d`** / **`alloc_and_copy_h2d_sync`**: same parameter type change; null check `!alloc` removed (now always a reference).
3. **`reconstruct_column`** (HOST→GPU path, NOT the P2P path): parameter type change.
4. **`write_host_buffer_to_disk`** / **`read_disk_buffer_to_host`** / **`write_column_buffers`** / **`read_column_buffers`**: same parameter type change.
5. **`convert_gpu_to_host_fast`**: uses `host_table_allocation::create()` instead of `make_unique<host_table_allocation>(...)`.
6. **`convert_host_fast_to_host_fast`**: same.
7. **`convert_disk_to_host_data`**: same.
8. **`convert_host_fast_to_gpu`**: dereference `*fast_table->allocation` when calling `reconstruct_column`.

### Collision surface with our commit 8392c3d

Our commit `8392c3d` introduces a large hunk (`@@ -132,66 +134,15 @@`) that **replaces the upstream `convert_gpu_to_gpu` body with a forward declaration** and another hunk (`@@ -618,6 +569,290 @@`) that **inserts 290 lines of new P2P functions** between `convert_gpu_to_host_fast` and the `convert_host_fast_to_gpu` area.

The `96bfea1` upstream commit modifies the `convert_gpu_to_host_fast` function and several functions after it (lines 606+). When git tries to replay `8392c3d` onto `96bfea1`, both patches target lines in the 500–900 range of the bcddb89 baseline. The precise conflict shape:

- `8392c3d` adds the P2P functions block BEFORE the region that `96bfea1` touches (between `convert_gpu_to_host_fast` and `convert_host_fast_to_gpu`).
- `96bfea1`'s changes to `convert_gpu_to_host_fast` and downstream functions (parameter type updates) may or may not conflict textually depending on whether git can reconcile the line offsets.
- Our commits `1e889d7` (same-stream invariant), `37df815` (dst_guard), and `9da4047` (probe-restore) only touch code that OUR fork added (`alloc_and_peer_copy_async`, `run_p2p_probe_locked`). Since upstream does NOT have these functions, upstream's `96bfea1` cannot have modified them. These three commits should apply cleanly as additions.

### Private constructor / shared_ptr pattern (per D-03)

`host_table_allocation` now has:
- Private constructor
- Factory: `static std::unique_ptr<host_table_allocation> create(fixed_multiple_blocks_allocation buffers, std::vector<column_metadata> columns, std::size_t data_size)`
- `allocation` member type: `shared_ptr<fixed_size_host_memory_resource::multiple_blocks_allocation>`

Sirius `2e197c6` (pin_table tier='host') depends on this API. Plan 24-03 must ensure sirius callers use `host_table_allocation::create()` and handle `shared_ptr` for `allocation`.

### Predicted line-number drift for dst_guard fix (37df815)

Our `dst_guard` is in `alloc_and_peer_copy_async` which is entirely our fork addition (not in upstream). Post-rebase, `alloc_and_peer_copy_async` will still be at whatever line our commits land on. **No line-number drift from 96bfea1** — the function is in our-code-only territory.

---

## Section B: 9ceebaa — "Fix for: Invalid Error: reconstruct_column: STRING column metadata must have at least one child (offsets) (#124)"

**Stat:** 3 files changed, 52 insertions(+)

### Files touched

| File | Change | Risk to our fork |
|------|--------|-----------------|
| `src/data/representation_converter.cpp` | +11 lines: two early-return guards for empty-STRING columns (`meta.children.empty() && meta.num_rows == 0`) in `reconstruct_column` and `reconstruct_column_from_disk` | LOW — our fork already has equivalent guard in `reconstruct_column_p2p` |
| `test/data/test_disk_host_converters.cpp` | +23 lines: 2 new empty-STRING round-trip tests | LOW — test only |
| `test/data/test_gpu_disk_converters.cpp` | +24 lines: 2 new empty-STRING disk→GPU tests | LOW — test only |

### 9ceebaa's STRING guard vs our reconstruct_column_p2p guard

`9ceebaa` adds this guard to `reconstruct_column` (HOST→GPU path, lines ~734 in `96bfea1` version):

```cpp
if (meta.children.empty() && meta.num_rows == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
}
```

Our fork already has an equivalent guard in `reconstruct_column_p2p` (GPU→GPU path, current lines 703-710):

```cpp
if (src.num_children() < 1) {
    // Empty / degenerate STRING column with no offsets child (cudf produces
    // these for empty intermediate results). ...
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
}
```

The two guards use different predicates because they work from different input types:
- `9ceebaa`'s guard: `meta.children.empty() && meta.num_rows == 0` (metadata-based, more precise)
- Our guard: `src.num_children() < 1` (column_view-based, catches the same case since cudf produces this shape for empty STRING columns)

**Interaction analysis:** `9ceebaa`'s guard is in `reconstruct_column` (HOST→GPU via host_table_allocation). Our `dst_guard` in `alloc_and_peer_copy_async` is called by `reconstruct_column_p2p` (GPU→GPU via peer copy), which is a completely different code path. `9ceebaa` does NOT add any guard to `reconstruct_column_p2p` — our path is untouched.

**Verdict:** No collision. The two STRING guards handle the same data shape but in different converter paths. They coexist cleanly.

Does `reconstruct_column_p2p` path still execute under 9ceebaa's guard? YES — `reconstruct_column_p2p` is only called by `convert_gpu_to_gpu`. `reconstruct_column` is called by `convert_host_fast_to_gpu`. These are separate converter registrations; `9ceebaa`'s guard cannot gate or short-circuit our P2P path.

---

## Section C: Per-fork-commit Classification Table

| # | SHA | Subject | Files Touched | Classification | Rationale |
|---|-----|---------|---------------|---------------|-----------|
| 1 | `9a23f4f` | fix(memory): ptds tracker, pool peer access, pipeline_io_backend hygiene | `src/memory/memory_space.cpp`, `src/data/pipeline_io_backend.cpp`, headers | **CLEAN** | Upstream 96bfea1/9ceebaa touch neither of these files. Replays cleanly as pure additions. |
| 2 | `0c0a4af` | fix(pipeline_io_backend): reorder io_worker members so _thread is last | `src/data/pipeline_io_backend.cpp` | **CLEAN** | Upstream touches nothing in pipeline_io_backend.cpp. Replays cleanly. |
| 3 | `8392c3d` | fix(representation_converter): P2P override — target-bound stream, DMA probe at init | `src/data/representation_converter.cpp`, `src/memory/common.cpp` | **RE-DERIVE (HIGH CONFLICT RISK)** | This commit replaces upstream's `convert_gpu_to_gpu` body with a forward declaration AND inserts 290 lines of P2P code. Upstream `96bfea1` modifies the HOST-tier functions immediately after the insertion site. Git will likely generate a merge conflict on `representation_converter.cpp` at the boundary between our P2P insertion and upstream's parameter-type changes. The function body we insert (`alloc_and_peer_copy_async`, `reconstruct_column_p2p`, `convert_gpu_to_gpu` impl) is entirely our code — no upstream analog — but the surrounding context lines may shift. Resolution: accept both; keep our P2P functions; take upstream's parameter-type changes to HOST-tier functions. |
| 4 | `085d917` | fix(stream-lineage): writer_stream/writer_event on gpu_table_representation | `src/data/gpu_data_representation.cpp`, headers | **CLEAN** | Upstream touches no GPU representation files. Replays cleanly. |
| 5 | `89d6a3f` | style: pre-commit cleanup (clang-format + codespell) | multiple files (formatting only) | **CLEAN** | Formatting-only. Will apply cleanly; any lines that upstream changed will produce minor formatting conflicts at worst, resolvable by re-running clang-format on the affected file sections. |
| 6 | `1e889d7` | fix(p22): same-stream invariant in alloc_and_peer_copy_async (Cluster B) | `src/data/representation_converter.cpp` (our P2P section only) | **CLEAN** | Modifies `alloc_and_peer_copy_async` which is 100% our fork code, not in upstream. Once commit 3 applies (creating the function), this commit patches inside it. No upstream overlap. |
| 7 | `37df815` | fix(p23): cuda_set_device_raii guard for HtoD in alloc_and_peer_copy_async | `src/data/representation_converter.cpp` (our P2P section only) | **CLEAN** | Same as commit 6 — modifies the dst_guard scope inside `alloc_and_peer_copy_async`, which only exists in our fork. After commit 3 applies cleanly, this commit patches inside our code. No upstream overlap. |
| 8 | `9da4047` | fix(p23): run_p2p_probe_locked must restore device context on exit | `src/memory/common.cpp` | **CLEAN** | `run_p2p_probe_locked` is in `common.cpp` which upstream's 96bfea1 and 9ceebaa do not touch. Applies cleanly after commit 3 created the function. |

**Summary:** 1 RE-DERIVE (commit 3), 7 CLEAN.

---

## Section D: Rebase Strategy and Predicted Order

### The single collision: Commit 3 (8392c3d) on representation_converter.cpp

Our commit `8392c3d` makes two large changes to `representation_converter.cpp` against the bcddb89 baseline:
1. **Lines ~132–197**: removes the old upstream `convert_gpu_to_gpu` body (66 lines) and replaces with 15 lines (forward declaration comment + stub).
2. **Lines ~618+**: inserts 290 lines of new P2P code (DMA probe, `alloc_and_peer_copy_async`, `alloc_and_peer_copy_sync`, `reconstruct_column_p2p`, and the full `convert_gpu_to_gpu` implementation) immediately after `convert_gpu_to_host_fast`.

The upstream `96bfea1` (rebased onto as new base) modifies lines 528–1214 of the bcddb89 version — the HOST-tier helpers that appear AFTER our P2P insertion point.

**Git conflict prediction:** When replaying `8392c3d` onto the new upstream base `9ceebaa` (which includes `96bfea1`'s changes), git will need to reconcile:
- Our removal of the original `convert_gpu_to_gpu` body (already gone in upstream's `96bfea1` — the body is still present there, but in the NEW shared_ptr form).
- Our insertion of 290 lines of P2P code at the `convert_gpu_to_host_fast` boundary.
- Upstream's changes to `collect_d2h_ops`, `alloc_and_schedule_h2d`, etc. (parameter type changes at the same line range).

**Resolution plan for Plan 24-02:**
1. When the conflict fires on commit 3 (`8392c3d`), open the conflicted `representation_converter.cpp`.
2. Accept the full upstream parameter-type changes to all HOST-tier functions (`collect_d2h_ops`, `collect_column_d2h_ops`, `alloc_and_schedule_h2d`, `alloc_and_copy_h2d_sync`, `reconstruct_column` HOST path, disk helpers).
3. Keep our entire P2P block (`alloc_and_peer_copy_async`, `alloc_and_peer_copy_sync`, `reconstruct_column_p2p`, `convert_gpu_to_gpu` impl with forward-decl).
4. Keep our removal of the upstream's old pack/unpack `convert_gpu_to_gpu` body.
5. Use `host_table_allocation::create()` instead of `make_unique<host_table_allocation>()` in the two places in our P2P path that construct host_table_allocation (these are NOT in the P2P path — our P2P path creates `gpu_table_representation`, not `host_table_allocation`; so this change doesn't affect our P2P code).
6. Mark `git add src/data/representation_converter.cpp` and `git rebase --continue`.

**Commits 4–8** should apply mechanically after commit 3 is resolved. All of them touch either:
- `gpu_data_representation.cpp` (commit 4 — no upstream change)
- Formatting (commit 5 — accept upstream formatting where it conflicts)
- `alloc_and_peer_copy_async` or `run_p2p_probe_locked` (commits 6, 7, 8 — our code, no upstream overlap)

### Predicted rebase outcome

```
Commit 1 (9a23f4f): CLEAN — applies
Commit 2 (0c0a4af): CLEAN — applies
Commit 3 (8392c3d): RE-DERIVE — CONFLICT on representation_converter.cpp
  → Plan 24-02 resolves: keep our P2P code + take upstream's HOST-tier type changes
Commit 4 (085d917): CLEAN — applies after 3 is resolved
Commit 5 (89d6a3f): CLEAN — may need re-run of clang-format on modified files
Commit 6 (1e889d7): CLEAN — applies (patches inside our alloc_and_peer_copy_async)
Commit 7 (37df815): CLEAN — applies (patches inside our alloc_and_peer_copy_async)
Commit 8 (9da4047): CLEAN — applies (patches common.cpp, no upstream change)
```

### Sirius-side impact (for Plan 24-03)

`2e197c6` (pin_table tier='host') uses `host_table_allocation::create()` because the constructor went private in `96bfea1`. After the cucascade rebase, our fork will have the new API. The sirius merge must handle:
- Any sirius-side code that directly constructed `host_table_allocation` (verify no such code exists in sirius — it wraps through cucascade's own converter path, not directly constructing `host_table_allocation`).
- `2e197c6`'s gitlink: per D-05, ours wins — resolve gitlink conflict to our fork HEAD.

---

*Generated: 2026-05-13 by Plan 24-01 analysis of /tmp/claude/p24_01_96bfea1_full.diff and /tmp/claude/p24_01_9ceebaa_full.diff*
