---
phase: 22-multi-gpu-pinning-stream-lineage-hardening
plan: 01
subsystem: scan_manager
tags: [pin_table, multi-gpu, memory_space, pinned_entry, PIN-MGPU-01]

# Dependency graph
requires:
  - phase: 20-scan-manager-pin-tables-port
    provides: pinned_entry struct + insert_pinned_entry single-pointer API; create_provider_for consumes entry.memory_space
provides:
  - "pinned_entry::chunk_memory_spaces — std::vector<cucascade::memory::memory_space*> parallel to data_batches_by_column inner vectors"
  - "insert_pinned_entry signature accepting per-chunk vector + precondition check (size == data_tables.size())"
  - "Same-row-count merge invariant: throws on chunk_memory_spaces size or pointer-by-pointer mismatch (Phase 22 Pitfall 3 closure)"
  - "Public [[nodiscard]] get_pinned_entries() const noexcept accessor for [pin_mgpu] distribution test in Plan 22-05"
affects: [22-02-pin-table-round-robin, 22-03-cucascade-cluster-b, 22-04-cached-split-provider, 22-05-pin-mgpu-distribution-test]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-chunk parallel vector on pinned_entry (D-03): chunk_memory_spaces[i] is the memory_space* for every column's chunk at index i"
    - "Merge-branch invariant check throws std::runtime_error on mismatched chunk_memory_spaces between existing and new entry (Pitfall 3)"
    - "Public const accessor for test-only read of internal map (encapsulation preserved — no setters exposed)"

key-files:
  created: []
  modified:
    - src/include/scan_manager/sirius_scan_manager.hpp
    - src/scan_manager/sirius_scan_manager.cpp

key-decisions:
  - "chunk_memory_spaces vector member name (matches snake_case sibling fields column_names, file_paths, data_batches_by_column, num_rows; no leading underscore — public-struct convention)"
  - "Precondition check uses data_tables.size() not a separate total_input_chunks counter — current code already enforces one cudf::table per chunk via the release() loop, so data_tables.size() == total chunk count"
  - "Public accessor returns const-reference to _pinned_entries map (zero-copy, callers cannot mutate) — minimal API surface needed for Plan 22-05 distribution test"
  - "Merge-branch invariant throws std::runtime_error (not std::invalid_argument) — caller did not pass an invalid argument; the merge VIOLATES the per-call deterministic-alignment invariant (D-02)"

patterns-established:
  - "Pattern 1: parallel-vector-to-data_batches — every per-chunk attribute on pinned_entry is a std::vector<...> with size == total chunks across all input data_tables"
  - "Pattern 2: precondition-then-merge-invariant-then-fresh-insert — the three-stage shape validates the per-call contract before mutating shared state"

requirements-completed:
  - PIN-MGPU-01

# Metrics
duration: 4min
completed: 2026-05-07
---

# Phase 22 Plan 01: Per-chunk memory_space vector on pinned_entry Summary

**pinned_entry now carries std::vector<cucascade::memory::memory_space*> chunk_memory_spaces parallel to data_batches_by_column inner vectors, plus a public get_pinned_entries() const accessor and a same-row-count merge invariant — Plan 22-02's PinTableFunction round-robin loop and Plan 22-05's [pin_mgpu] distribution test now have a place to write/read per-chunk GPU placement.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-05-07T22:55:44Z
- **Completed:** 2026-05-07T22:59:24Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Replaced the single `pinned_entry::memory_space` pointer with a per-chunk parallel vector — closes the structural prerequisite for D-03 multi-GPU chunk distribution.
- Added precondition + merge-invariant checks in `insert_pinned_entry` that loudly reject any caller (including future ones) that passes a misaligned `chunk_memory_spaces` vector — closes Pitfall 3 from Phase 22 RESEARCH.md.
- Added public read-only accessor `get_pinned_entries()` so the planned `[pin_mgpu]` Catch2 test (Plan 22-05) can assert chunk placement directly without friending or exposing the `_pinned_entries` member.

## Exact File:Line Pinpoints (per plan output spec)

1. **New `chunk_memory_spaces` member:** `src/include/scan_manager/sirius_scan_manager.hpp:67`
   ```cpp
   std::vector<cucascade::memory::memory_space*> chunk_memory_spaces;
   ```
2. **New `get_pinned_entries()` accessor:** `src/include/scan_manager/sirius_scan_manager.hpp:168`
   ```cpp
   [[nodiscard]] const std::unordered_map<std::string, pinned_entry>&
   get_pinned_entries() const noexcept { return _pinned_entries; }
   ```
3. **Merge-invariant throw site (Pitfall 3 closure):** `src/scan_manager/sirius_scan_manager.cpp:279-294`
   - Size check: `cpp:279` — `if (entry.chunk_memory_spaces.size() != chunk_memory_spaces.size())`
   - Pointer-by-pointer check: `cpp:288` — `if (entry.chunk_memory_spaces[i] != chunk_memory_spaces[i])`
   - Both throw `std::runtime_error` with diagnostic message naming the offending index.
4. **Build is intentionally broken at three Plan 22-02 hand-off sites** (verified by MCP `build` invocation, exit code 2):
   - `src/scan_manager/sirius_scan_manager.cpp:107` — `if (entry.memory_space == nullptr)` inside `create_provider_for`
   - `src/scan_manager/sirius_scan_manager.cpp:176` — `*entry.memory_space` inside the `cached_split_provider` ctor argument list
   - `src/sirius_extension.cpp:820` — `PinTableFunction` calls `insert_pinned_entry` with the old single-`mem_space` argument
   - **Plan 22-02 closes all three by:** (a) building the per-chunk vector in the round-robin loop in `PinTableFunction`, (b) flipping `cached_split_provider` to consume the per-chunk space at emission time, and (c) updating `create_provider_for` to read from the vector.
   - **Note:** the plan's stated expected-fail sites (`cached_split_provider.cpp:42-43,99` and `sirius_extension.cpp:749`) are slightly different from the observed reality. `cached_split_provider.cpp` itself does NOT break because its `_memory_space` member is constructed from `*entry.memory_space` at the caller (`sirius_scan_manager.cpp:176`). The CALLER (`create_provider_for` at `sirius_scan_manager.cpp:107,176`) breaks instead. This is the same logical hand-off — Plan 22-02 still closes it. Documented here for traceability.
5. **HYG-02 grep result:** 0 new `rmm::cuda_stream_default` introductions across both modified files (this plan touches no streams; headers + insert_pinned_entry body only).

## Task Commits

Each task was committed atomically with `--no-verify` per parallel-execution protocol:

1. **Task 1: Refactor pinned_entry header to per-chunk vector + public accessor** — `20dcf19` (refactor)
2. **Task 2: Migrate insert_pinned_entry implementation + merge invariant** — `81be803` (refactor)

Plan metadata commit (SUMMARY + STATE + ROADMAP) lands as the final commit of this plan.

## Files Created/Modified

- `src/include/scan_manager/sirius_scan_manager.hpp` — `pinned_entry::chunk_memory_spaces` field added (line 67), single-pointer `memory_space` member removed; `insert_pinned_entry` signature updated (line 159) to take `std::vector<cucascade::memory::memory_space*>`; public `get_pinned_entries()` accessor added (line 168).
- `src/scan_manager/sirius_scan_manager.cpp` — `insert_pinned_entry` implementation rewritten (lines 242-340) with: precondition check (chunk_memory_spaces.size() == data_tables.size()), same-row-count merge invariant (size + pointer-by-pointer), fresh-insert move-assign of the new vector, removal of the old `entry.memory_space = &memory_space` line.

## Decisions Made

- **Precondition uses `data_tables.size()` not a separate counter** — the existing function loop already iterates `data_tables` once-per-table-per-chunk via `table->release()`, and the call site (`PinTableFunction` chunk loop) emits exactly one `cudf::table` per `chunked_parquet_reader::read_chunk()` result. So `data_tables.size() == total chunks across all files`. Using a derived counter would have required scanning the vector twice; using `.size()` is O(1).
- **Merge invariant throws `std::runtime_error` not `std::invalid_argument`** — the precondition (size == data_tables.size()) is an argument-validity check; the merge-branch is a runtime-state-violation check (the caller passed a structurally valid vector that happens to disagree with the existing entry's vector — that's a deeper invariant violation deserving its own error type).
- **Public accessor exposed via inline definition in the header** — minimal API surface; the body is a 1-line return-by-reference. Inline-defining avoids a .cpp churn for a trivial getter.

## Deviations from Plan

None — plan executed exactly as written. Both Task 1 and Task 2 acceptance criteria pass on first MCP build attempt. The build failure at three Plan 22-02 hand-off sites is an EXPECTED outcome documented in the plan's `<verification>` block.

A small documentation-only divergence: the plan listed the expected-fail sites as `cached_split_provider.cpp:42-43,99` + `sirius_extension.cpp:749`, but reality has them at `sirius_scan_manager.cpp:107,176` (in `create_provider_for`, which lives in the same translation unit) + `sirius_extension.cpp:820` (the `insert_pinned_entry` call site, not line 749 which is a non-erroring local-binding). This is a minor planning-time mis-specification, not an execution deviation; the same logical fix-set in Plan 22-02 closes all three sites regardless.

## Issues Encountered

None.

## Plan-Level Verification (per plan `<verification>` block)

All three plan-level gates PASS:

1. **Live `.memory_space\b` references in scan_manager:** 2 hits, both in `create_provider_for` (sirius_scan_manager.cpp:107 + 176) — these are the documented Plan 22-02 hand-off. 0 hits in the `pinned_entry` struct or `insert_pinned_entry` body.
2. **`chunk_memory_spaces` total occurrence count:** 21 (target: ≥6). Distribution: 6 in header (member decl, comments, signature, doxygen), 15 in cpp (signature, precondition, merge-invariant comments + checks, fresh-insert assign).
3. **HYG-02 invariant:** 0 occurrences of `rmm::cuda_stream_default` across both modified files. This plan introduces no streams; HYG-02 baseline of 40 (legacy-only) preserved.

## Self-Check: PASSED

- File `src/include/scan_manager/sirius_scan_manager.hpp` exists and contains all 3 new constructs (member, accessor, signature). FOUND.
- File `src/scan_manager/sirius_scan_manager.cpp` exists and contains the new signature + precondition + merge-invariant + fresh-insert assignment. FOUND.
- Commit `20dcf19` exists in `git log`. FOUND.
- Commit `81be803` exists in `git log`. FOUND.

## Next Plan Readiness (Plan 22-02)

Plan 22-02 (PinTableFunction round-robin) can resume immediately:
- The per-chunk vector field exists on `pinned_entry`.
- The `insert_pinned_entry` signature accepts the vector.
- Plan 22-02's three required code edits are: (a) round-robin loop in `PinTableFunction` building `std::vector<cucascade::memory::memory_space*>`; (b) updating `create_provider_for` to read per-chunk `entry.chunk_memory_spaces` and pass it through to `cached_split_provider`; (c) updating `cached_split_provider` to emit the per-chunk space per emitted batch.

Plan 22-05 (`[pin_mgpu]` distribution test) can resume immediately:
- `get_pinned_entries()` accessor is live; the test can assert on `entry.chunk_memory_spaces` directly.

Plan 22-03 (cucascade Cluster B fix) is independent of this plan and runs in parallel (Wave 1).

---
*Phase: 22-multi-gpu-pinning-stream-lineage-hardening*
*Plan: 01*
*Completed: 2026-05-07*
