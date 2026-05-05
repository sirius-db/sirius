---
phase: 18-databatch-raii-migration-cucascade-117-surface
plan: 04
subsystem: pipeline
tags: [cucascade, raii, data_batch, mutable_data_batch, read_only_data_batch, read_only_operators, scan_layer, task_creator, debug_utils, pitfall-4, phase-18, db-02, db-03]

# Dependency graph
requires:
  - phase: 18-databatch-raii-migration-cucascade-117-surface
    plan: 02
    provides: get_cudf_table_view accepts const read_only_data_batch&; pipeline::prepare_and_acquire_mutable RAII helper; sirius_physical_operator base + gpu_pipeline_task storage migrated; convertible_* wrappers migrated; R2 size-estimator inline bodies migrated.
provides:
  - 6 read-only operator .cpp files migrated to scoped to_read_only() accessors
  - 5 scan-layer files migrated (parquet/duckdb/cpu_source tasks + scan_executor + sirius_gpu_parquet_scan_operator)
  - task_creator.cpp affinity/sizing loop migrated (mark_task_created Sirius method preserved as no-cucascade-FSM)
  - debug_utils.cpp + .hpp migrated (is_gpu_tier helper now takes read_only_data_batch&; public API const dropped from data_batch&)
  - Pitfall 4 closure (4 known sites): filter:60, projection:63, table_scan:176 (closed by 18-03), gpu_parquet_scan_operator:252 — all now 3-arg make_data_batch with operator stream as writer_stream
affects: [18-05, 18-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "R1 single-shot read-only: scoped to_read_only() per loop iteration; accessor destroyed at end-of-iteration releases shared lock"
    - "R1 mixed sample/batched read: single accessor vector held for the lifetime of derived cudf::table_view objects (sort_sample.cpp pattern — table_view is non-owning so accessor must outlive every read)"
    - "Clone migration: data_batch::clone() moved onto accessor classes under #117; callers take a scoped to_read_only() and clone via the accessor"
    - "Pitfall 4 closure pattern: every make_data_batch call site uses the 3-arg form with the operator's actual execution stream as writer_stream, preserving Phase 13-04 Path-2 stream-lineage contract"
    - "Long-lived memory_space pointer: memory_space objects are owned by the reservation manager (long-lived), so a memory_space* resolved through a scoped accessor remains valid after the accessor drops — used in partition.cpp execute() to avoid holding a shared lock across gpu_partition_impl::* dispatches"
    - "Public-API const drop: where to_read_only() is needed inside a function that takes 'const data_batch&', the public signature drops const (debug_utils pattern). Cucascade #117 declares to_read_only/to_mutable as non-const because they mutate lock state — conventionally would be const-with-mutable-mutex but cucascade does not"

key-files:
  created: []
  modified:
    - src/op/sirius_physical_filter.cpp (R1 + Pitfall 4 — line 60 now 3-arg make_data_batch)
    - src/op/sirius_physical_projection.cpp (R1 + Pitfall 4 — line 63 now 3-arg make_data_batch)
    - src/op/sirius_physical_limit.cpp (R1 — read-only get_table_view + memory_space probe through accessor)
    - src/op/sirius_physical_partition.cpp (R1 + R7 — memory_space probe via scoped accessor; 2-arg get_data_batch_by_id; size estimator R2)
    - src/op/sirius_physical_result_collector.cpp (R1 + R3 — clone path: to_read_only on input, clone via accessor, to_mutable on freshly-cloned batch for convert_to)
    - src/op/sirius_physical_sort_sample.cpp (R1 — accessor vector for table-view lifetime)
    - src/op/scan/sirius_gpu_parquet_scan_operator.cpp (R1 + Pitfall 4 — line 252 now 3-arg; cached path migrated; Phase 18 TODO removed)
    - src/op/scan/parquet_scan_task.cpp (R2 — output-bytes loop wrapped in scoped to_read_only)
    - src/op/scan/duckdb_scan_task.cpp (R2 — same pattern)
    - src/op/scan/cpu_source_task.cpp (R1 — null-data probe via scoped accessor, dropped before std::move)
    - src/op/scan/duckdb_scan_executor.cpp (R1 + clone — scoped accessor for shallow_clone dynamic_cast probe; clone via accessor)
    - src/creator/task_creator.cpp (R2 — affinity/sizing loop in manager_loop migrated; mark_task_created Sirius method preserved)
    - src/debug_utils.cpp (R1 — is_gpu_tier signature flipped to const read_only_data_batch&; 7 debug_* fns take scoped accessor)
    - src/include/debug_utils.hpp (Rule 3: const dropped from data_batch& parameters because to_read_only() is non-const under #117)

key-decisions:
  - "[18-04] Pitfall 4 closure verified at all 4 known sites: filter:60 + projection:63 + gpu_parquet_scan_operator:252 (this plan), table_scan:176 (parallel plan 18-03). Every 3-arg make_data_batch call passes the operator's actual execution stream — no rmm::cuda_stream_default, no default-constructed cuda_stream_view{}. Closes the Phase 13-04 P2 race re-introduction risk."
  - "[18-04] sort_sample.cpp accessor lifetime: holds a std::vector<read_only_data_batch> for the LIFETIME of the derived sample_views vector (table_view is a non-owning ref into the accessor's gpu_table_representation). Per-iteration scope would let the lock drop before cudf::concatenate runs — and while the underlying representation is owned by the long-lived shared_ptr<data_batch>, holding the shared lock for the duration documents intent and matches RESEARCH.md's R1 + 'lock for read duration' guidance."
  - "[18-04] partition.cpp execute() — extract memory_space pointer through a scoped to_read_only() accessor, then drop the accessor BEFORE calling gpu_partition_impl::* (which takes its own accessor internally — plan 18-03 territory). Concurrent shared locks on the same batch would also be safe (P1 only fails on shared+exclusive overlap), but explicit drop documents the boundary and matches the plan's 'one accessor per access block' R1 phrasing."
  - "[18-04] result_collector.cpp clone path: the pre-clone read uses a scoped to_read_only on input_batch (block-level scope), the clone() via that accessor produces a freshly-idle clone_batch, then to_mutable on clone_batch for the convert_to. The to_mutable runs AFTER the input_batch ro-accessor drops — distinct batches, no P1 overlap. The post-convert read uses to_read_only on whichever batch carries the host representation (input_batch if no GPU->HOST conversion happened, clone_batch otherwise) for the lifetime of the chunk-pushing loop."
  - "[18-04] debug_utils.hpp public API: const dropped from cucascade::data_batch& parameters (7 functions). Required because cucascade::data_batch::to_read_only() is non-const under #117 — without dropping const we cannot acquire the shared lock from the helper body. Test files at test/cpp/debug/test_debug_utils.cpp pass `*batch` (deref shared_ptr<data_batch>) which is already a non-const lvalue, so no test churn. Documented as Rule 3 (Blocking auto-fix) because the cpp file cannot compile without it."
  - "[18-04] task_creator.cpp mark_task_created (4 sites) preserved verbatim. Per RESEARCH.md classification: 'Sirius method, not the cucascade FSM. Compile-clean; no migration needed.' This Sirius pipeline-side bookkeeping method (declared at sirius_pipeline.hpp:184, defined at sirius_pipeline.cpp:358) is unrelated to cucascade::data_batch's deleted task_created state. Rename is out-of-scope for Phase 18."
  - "[18-04] Rule 3 deviation: ->clone() call sites at duckdb_scan_executor.cpp:292 + result_collector.cpp:172 surfaced AFTER tasks 1+2 committed because the build halted at parallel-plan errors. Both files were already in 18-04's files_modified list, so the fix is scope-conformant. Migrated through accessors (clone() moved off data_batch onto accessor classes under #117)."
  - "[18-04] gpu_pipeline_executor.cpp:301 try_to_create_task site: RESEARCH.md classified as 'comment-only' (line 276). Empirically the call exists at line 301. Plan 18-03's commit d63b406 closed this in scope (replaced with a comment). No 18-04 action needed."

patterns-established:
  - "Pattern: Pitfall 4 every-site verification — every make_data_batch call site in src/ now uses 3-arg form with operator stream. Acceptance grep `grep -rnE 'make_data_batch\\([^,]+,\\s*\\*[^,)]+\\)\\s*\\)' src/` returns 0."
  - "Pattern: Scope-narrowed accessor with explicit RAII destruction comment — every accessor in 18-04's edits is annotated with `ro destroyed at end of scope -> shared lock released` to make the reviewer's job easy."
  - "Pattern: Post-clone to_mutable for in-place conversion — when an old `data_batch::convert_to<T>(...)` call is migrated, the surrounding code must clone first (via the new accessor-side clone), then take to_mutable() on the freshly-idle clone, then convert_to via the mutable accessor. Used in result_collector.cpp."

requirements-completed: [DB-02, DB-03]

# Metrics
duration: 16min
completed: 2026-05-05
---

# Phase 18 Plan 04: Read-Only Operators + Scan Layer + task_creator + debug_utils RAII Migration Summary

**Migrated 13 production .cpp files (6 read-only operators + 5 scan-layer + task_creator + debug_utils) to the cucascade #117 RAII model, closing all 4 known Pitfall 4 (2-arg make_data_batch) sneak-back sites — every src/ make_data_batch call now passes the operator stream as writer_stream.**

## Performance

- **Duration:** 16min
- **Started:** 2026-05-05T15:54:00Z
- **Completed:** 2026-05-05T16:09:00Z
- **Tasks:** 3 / 3
- **Files modified:** 14 (13 from plan files_modified + 1 Rule 3 header ripple — debug_utils.hpp)

## Accomplishments

- **6 read-only operator .cpp files** migrated to scoped `to_read_only()` accessors (Recipe R1):
  - `sirius_physical_filter.cpp`: 1 get_data + 1 make_data_batch 2-arg upgraded to 3-arg with stream (Pitfall 4 site #1)
  - `sirius_physical_projection.cpp`: 1 get_data + 1 make_data_batch 2-arg upgraded to 3-arg with stream (Pitfall 4 site #2)
  - `sirius_physical_limit.cpp`: 1 get_data + memory_space access for slice — single accessor scope spans both
  - `sirius_physical_partition.cpp`: execute() memory_space probe via scoped accessor + size estimator R2 + 2-arg get_data_batch_by_id (R7)
  - `sirius_physical_result_collector.cpp`: clone path migrated (R1 + R3) — to_read_only on input, clone through accessor, to_mutable on freshly-cloned batch for convert_to, to_read_only on host carrier for chunk loop
  - `sirius_physical_sort_sample.cpp`: 2 sites — memory_space probe + accessor vector for table-view lifetime (table_view is non-owning, must outlive every read)
- **5 scan-layer files** migrated:
  - `sirius_gpu_parquet_scan_operator.cpp`: cached path scoped to_read_only spans gpu_rep + cached_view + memory_space resolution; line 252 make_data_batch upgraded to 3-arg (Pitfall 4 site #3)
  - `parquet_scan_task.cpp` + `duckdb_scan_task.cpp`: output-bytes loops wrapped in scoped to_read_only per iteration (R2)
  - `cpu_source_task.cpp`: scoped to_read_only for null-data probe, dropped before std::move(batch)
  - `duckdb_scan_executor.cpp`: scoped to_read_only for shallow_clone dynamic_cast probe + clone() through accessor
- **`task_creator.cpp`** affinity/sizing loop in `manager_loop` migrated to per-batch scoped to_read_only; `mark_task_created` Sirius method preserved at 4 sites (out-of-scope per RESEARCH.md classification).
- **`debug_utils.cpp`** + header migrated. `is_gpu_tier` helper now takes `cucascade::read_only_data_batch const&`. All 7 `debug_*` public functions take a scoped accessor at function entry. `debug_diff` holds two accessors (one per batch).
- **Pitfall 4 closure** verified at all 4 known sites: 3 in this plan + 1 in parallel plan 18-03 (table_scan:176). Every make_data_batch call in src/ now uses the 3-arg form with the operator's actual execution stream as writer_stream. The Phase 13-04 P2 race re-introduction risk is closed.
- **HYG-02 baseline preserved**: 0 `rmm::cuda_stream_default` introductions in any modified file. Final src/ HYG-02 = 0.

## Task Commits

Each task was committed atomically with `--no-verify` per parallel-execution protocol:

1. **Task 1: Migrate read-only operators (filter, projection, limit, partition, sort_sample, result_collector) + close 2 Pitfall 4 sites** — `1ab1ba6` (refactor)
2. **Task 2: Migrate scan layer (parquet/duckdb/cpu_source/scan_executor/sirius_gpu_parquet_scan_operator) + close 4th Pitfall 4 site** — `3680877` (refactor)
3. **Task 3: Migrate task_creator + debug_utils + final src/ build verification** — `b455ce3` (refactor)
4. **Rule 3 fix: ->clone() migration in duckdb_scan_executor + result_collector** — `4aefd19` (fix)

## Files Created/Modified

### Modified (14 files: 13 plan-targeted + 1 Rule 3 header ripple)

- `src/op/sirius_physical_filter.cpp` (R1 + Pitfall 4 closure)
- `src/op/sirius_physical_projection.cpp` (R1 + Pitfall 4 closure)
- `src/op/sirius_physical_limit.cpp` (R1)
- `src/op/sirius_physical_partition.cpp` (R1 + R2 + R7)
- `src/op/sirius_physical_result_collector.cpp` (R1 + R3)
- `src/op/sirius_physical_sort_sample.cpp` (R1)
- `src/op/scan/sirius_gpu_parquet_scan_operator.cpp` (R1 + Pitfall 4 closure)
- `src/op/scan/parquet_scan_task.cpp` (R2)
- `src/op/scan/duckdb_scan_task.cpp` (R2)
- `src/op/scan/cpu_source_task.cpp` (R1)
- `src/op/scan/duckdb_scan_executor.cpp` (R1 + clone migration)
- `src/creator/task_creator.cpp` (R2)
- `src/debug_utils.cpp` (R1; is_gpu_tier signature change to read_only_data_batch&)
- **(Rule 3)** `src/include/debug_utils.hpp` (const dropped from data_batch& parameters in 7 functions + debug_diff)

### Created

- `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/deferred-items.md` (logs surfaced inventory misses for orchestrator triage)

## Decisions Made

(See `key-decisions` in frontmatter for the canonical list.)

## Deviations from Plan

### Auto-fixed Issues (Rule 3 — Blocking)

**1. [Rule 3 — Blocking] debug_utils.hpp public API const drop**
- **Found during:** Task 3 implementation
- **Issue:** `cucascade::data_batch::to_read_only()` is non-const under PR #117 (acquires the shared lock — conventionally would be const-with-mutable-mutex but cucascade does not declare it that way). The plan instructs `debug_utils.cpp` to wrap in `to_read_only()` accessor, but the public API takes `cucascade::data_batch const&` — this combination cannot compile.
- **Fix:** Dropped `const` from 7 public function signatures + 2 debug_diff parameters in `src/include/debug_utils.hpp`. Test files (test/cpp/debug/test_debug_utils.cpp) pass `*batch` (deref shared_ptr<data_batch>) which is already a non-const lvalue, so no test churn.
- **Files modified:** `src/include/debug_utils.hpp`
- **Commit:** `b455ce3` (folded into Task 3)

**2. [Rule 3 — Blocking] ->clone() migration**
- **Found during:** Task 2/3 verification build
- **Issue:** After committing Tasks 1-3 as planned, the build progressed past my files but two more sites failed: `duckdb_scan_executor.cpp:292 batch->clone(...)` and `result_collector.cpp:172 input_batch->clone(...)`. Cucascade #117 moved `data_batch::clone()` onto the accessor classes (read_only_data_batch::clone, mutable_data_batch::clone); the bare `data_batch::clone` is gone.
- **Fix:** Migrated both call sites to take a scoped `to_read_only()` accessor and clone through it. Both files are within plan 18-04's files_modified list — the fix is scope-conformant.
- **Files modified:** `src/op/scan/duckdb_scan_executor.cpp`, `src/op/sirius_physical_result_collector.cpp`
- **Commit:** `4aefd19` (Rule 3 — Blocking)

### Inventory misses surfaced for orchestrator triage

After plan 18-04's commits land, the src/-side build still has 9 FAILED translation units. 1 is out of DB-01..05 scope (`uring_reactor.cpp` / Phase 19), and 8 are inventory misses by 18-RESEARCH.md (operator-impl files + scan_manager). All 8 have a mechanical R1 / R3 recipe; total estimated effort < 30 minutes. Logged in detail in `.planning/phases/18-databatch-raii-migration-cucascade-117-surface/deferred-items.md`.

The plan's stated success criterion `MCP build: 0 src/ errors` is therefore NOT strictly met by 18-04 alone — but the strict per-task acceptance criteria for the 13 files in 18-04's files_modified list ALL pass. The remaining 8 errors are in files that no plan currently owns; they were missed by RESEARCH.md's enumeration.

## Build Verification

- **Files in 18-04 files_modified compiling cleanly**: 13 / 13 (100%)
- **Pitfall 4 closure**: 4 / 4 sites in src/ (filter:60 ✓, projection:63 ✓, table_scan:176 ✓ via 18-03, gpu_parquet_scan_operator:252 ✓)
- **HYG-02 baseline (src/ rmm::cuda_stream_default count)**: 0 (preserved)
- **Deleted-FSM-symbol grep (src/, excluding /legacy/)**: 0
- **FSM-state literal grep (src/, excluding /legacy/)**: 0 after 18-03's commits land (was 4 before; closed by 18-03)
- **Final src/ build**: 9 FAILED files (1 Phase 19 / 8 inventory misses — see deferred-items.md). All 13 files in 18-04 scope build cleanly.
- **Test errors**: not yet attempted (test migration is plan 18-05 territory).

## Verification Gates Passed (Plan-scoped only)

| Gate | Target | Actual | Pass |
|------|--------|--------|------|
| 2-arg make_data_batch in filter+projection | 0 | 0 | yes |
| to_read_only count across 6 read-only operators | ≥ 7 | 9 (1+1+1+2+2+2) | yes |
| HYG-02 in 6 read-only operators | 0 | 0 | yes |
| 2-arg make_data_batch in sirius_gpu_parquet_scan_operator | 0 | 0 | yes |
| to_read_only count in 5 scan files | ≥ 5 | 5 | yes |
| TODO Phase 17/18 in sirius_gpu_parquet_scan_operator | 0 | 0 | yes |
| HYG-02 in 5 scan files | 0 | 0 | yes |
| HYG-02 in task_creator + debug_utils | 0 | 0 | yes |
| mark_task_created preserved in task_creator | unchanged | 4 (matches baseline) | yes |
| Final src/ HYG-02 gate | 0 | 0 | yes |
| Final src/ deleted-FSM gate | 0 | 0 | yes |
| Final src/ FSM-state literal gate | 0 | 0 (after 18-03 lands) | yes |
| Final src/ 2-arg make_data_batch | 0 | 0 | yes |
| MCP build: src/ errors = 0 | 0 | 8 inventory misses + 1 Phase 19 | partial — see Deviations |

## Self-Check: PASSED

All 4 commits land in `git log --oneline`:
- `1ab1ba6` Task 1 (refactor) — 6 read-only operators
- `3680877` Task 2 (refactor) — 5 scan-layer files
- `b455ce3` Task 3 (refactor) — task_creator + debug_utils + .hpp
- `4aefd19` Rule 3 fix — ->clone() migration in 2 plan-scope files

All 14 modified files exist on disk. The 13 in plan files_modified all compile cleanly. The 1 header ripple (debug_utils.hpp) is documented as Rule 3 deviation. HYG-02 = 0 across src/. All Pitfall 4 (2-arg make_data_batch) sites in src/ closed. All deleted-FSM-symbol gates pass. Test-side migration (plan 18-05) and 8 inventory-miss files (deferred-items.md) are the only remaining src/-tree-or-test items before clean build.
