---
phase: 13-q11-multi-gpu-illegal-address
plan: 04
subsystem: cucascade-and-sirius-stream-lineage
tags: [stream-lineage, mgpu, cucascade, race-fix, partial]
verdict: PARTIAL
requires: [13-02-race-site, 13-03-falsifiers]
provides: stream-lineage-writer-event-infrastructure
affects: [cucascade-pin, src/include/data/data_batch_utils.hpp, src/op/aggregate, src/op/sirius_physical_*, src/data/host_parquet_*, src/expression_executor]
tech-stack:
  added:
    - cucascade::gpu_table_representation::record_writer_event/get_writer_event accessors
    - sirius::make_data_batch overloads taking rmm::cuda_stream_view writer_stream
  patterns:
    - STREAM-LINEAGE: producer records cudaEvent on writer stream; consumer cudaStreamWaitEvent in convert_gpu_to_gpu
key-files:
  created: []
  modified:
    - cucascade (submodule pin: e4db3d8 → 7409c60)
    - src/include/data/data_batch_utils.hpp (+2 overloads, ~35 LOC)
    - src/op/aggregate/gpu_aggregate_impl.cpp (2 sites)
    - src/data/host_parquet_representation_converters.cpp
    - src/expression_executor/gpu_expression_executor.cpp
    - src/op/sirius_physical_limit.cpp
    - src/op/sirius_physical_table_scan.cpp
    - src/op/sirius_physical_top_n.cpp
    - src/op/sirius_physical_ungrouped_aggregate.cpp
decisions:
  - "Submodule bump REQUIRED per Wave 2's Recommended Fix Shape — cucascade-side gpu_table_representation extension landed on cucascade branch fix/q11-mgpu-stream-event-lineage @ 7409c60."
  - "PARTIAL fix verdict: sanitizer error count 433 → 328 (24% reduction), but cumulative-state Q11 hang under SCHED-RR still SIGTERMs at 1800s. Additional producer sites need migration (left_delim_join::sink confirmed un-migrated; other compute_task-rooted writer paths unconfirmed)."
  - "Phase 14 SCHED-RR working-tree diff and kIterations=20 bump applied locally for fix verification, then reverted before commit. Committed scope = cucascade pin + 8 Sirius source files only."
  - "Q11-alone passes cleanly (9011 assertions, 7s); cumulative-state fails. Confirms Phase 13 CONTEXT.md hypothesis that Q11 only fails after Q1-Q10 warm-up."
metrics:
  duration: 60min (capped per orchestrator constraint; orphan agent had previously consumed time on building+sanitizer; this resume agent ran one full SCHED-RR build + Q11-alone + Q1-Q22 cumulative test)
  completed: 2026-04-30T03:30:00Z
  sanitizer_errors_before: 433
  sanitizer_errors_after_fix: 328
  hyg_count: 40
  cumulative_state_q11_under_sched_rr: SIGTERM_AT_1800s
---

# Phase 13 Plan 04: Q11 Multi-GPU Race Fix Summary

## One-liner

Bumps cucascade pin to `7409c60` (writer-event lineage in `convert_gpu_to_gpu`) and migrates 7 Sirius operator producer sites to the new `make_data_batch(table, mem_space, writer_stream)` overload — closes the 13-02 race-site fingerprint at one writer (`gpu_aggregate_impl::local_grouped_aggregate`) but leaves additional un-migrated producer paths firing the same race shape; cumulative-state Q11-under-SCHED-RR still hangs.

## Cucascade SHA Bump

- **Before:** `e4db3d8` (peer-DMA probe at init)
- **After:** `7409c60` (`fix(stream-lineage): add gpu_table_representation::{record,get}_writer_event + cudaStreamWaitEvent in convert_gpu_to_gpu`)
- **Branch (cucascade):** `fix/q11-mgpu-stream-event-lineage`
- **Diff scope (cucascade):**
  - `cucascade/include/cucascade/data/gpu_data_representation.hpp`: adds `_writer_event` member, `record_writer_event/get_writer_event` accessors, dtor releases event handle.
  - `cucascade/src/data/gpu_data_representation.cpp`: implements accessors using `cudaEventCreateWithFlags(cudaEventDisableTiming)` + `cudaEventRecord`.
  - `cucascade/src/data/representation_converter.cpp`: in `convert_gpu_to_gpu` (line 801), calls `cudaStreamWaitEvent(target_stream, gpu_source.get_writer_event(), 0)` BEFORE the column-loop when writer_event is non-null; falls back to `cudaDeviceSynchronize` on source device when no writer event was recorded (legacy compatibility); records writer event on `target_stream` for the resulting representation so downstream readers observe ordering.

## Sirius-Side Caller Adjustments

`src/include/data/data_batch_utils.hpp` adds 2 new `make_data_batch` overloads:

```cpp
inline std::shared_ptr<cucascade::data_batch> make_data_batch(
  cudf::table&& table,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view writer_stream);

inline std::shared_ptr<cucascade::data_batch> make_data_batch(
  std::unique_ptr<cudf::table> table,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view writer_stream);
```

Both call `gpu_repr->record_writer_event(writer_stream)` immediately after wrapping the table in `gpu_table_representation`.

Migrated producer sites (7 files, 8 call sites):

| File | Sites | Notes |
|------|-------|-------|
| src/op/aggregate/gpu_aggregate_impl.cpp | 2 | `local_grouped_aggregate` + `local_ungrouped_aggregate` — confirmed FIRST-error writer in 13-02 |
| src/data/host_parquet_representation_converters.cpp | 1 | host_parquet → gpu output |
| src/expression_executor/gpu_expression_executor.cpp | 1 | expression eval output |
| src/op/sirius_physical_limit.cpp | 1 | LIMIT operator output |
| src/op/sirius_physical_table_scan.cpp | 1 | table_scan output |
| src/op/sirius_physical_top_n.cpp | 1 | TOP-N output |
| src/op/sirius_physical_ungrouped_aggregate.cpp | 1 | (final stage of ungrouped agg) |

## Verification

### Build
- `mcp__project-commands__run_command name=build` → exit 0, 40/40 targets, ~30s
- Built without `_no_pref_rr_counter` (Phase 14 patch reverted; verified `grep -c _no_pref_rr_counter src/include/pipeline/task_scheduler.hpp = 0`).

### Q11 Alone (no SCHED-RR)
- `unit-tests filter="[TPC-H][parquet] gpu_execution - TPC-H Query 11 parquet"` → exit 0, 9011 assertions, 7.0s
- Result: PASS — Q11 in isolation works (consistent with CONTEXT.md "Q11 parquet passes in isolation").

### Q1-Q22 Cumulative under SCHED-RR (Phase 14 working-tree-applied)
- Phase 14 SCHED-RR diff applied via direct Edit tool calls (not git apply, per Wave 1 deviation — diff in 14-CONTEXT.md is documentation-style, missing unified-diff headers).
- Build: PASS (40/40)
- `unit-tests filter="[TPC-H][parquet]"` (22 queries × 2-GPU) → **TIMED OUT after 1800s**
  - Q1-Q10 PASS (10 test cases, 7813 assertions OK)
  - Q11 SIGTERM at the 1800s test runner timeout (same fingerprint as Phase 13 CONTEXT.md authoritative repro)
  - test cases: 11, 10 passed, 1 failed
- Verdict: **The fix is INCOMPLETE.** Q11 cumulative-state hang under SCHED-RR is NOT closed by the writer-event lineage in cucascade + the 7 migrated Sirius sites.

### Sanitizer Pre-Fix vs Post-Fix
- Pre-fix (13-02 evidence): 433 errors, FIRST = peer-copy of allocation written by `gpu_aggregate_impl::local_grouped_aggregate`.
- Post-fix (this plan, partial run by previous orphan agent at /tmp/claude/13-04-sanitizer/sanitizer.out): 328 errors (24% reduction).
- Post-fix FIRST error: SAME backtrace shape — writer is still `local_grouped_aggregate`, reader is still `convert_gpu_to_gpu`. Indicates the writer event is being recorded but either (a) the source data_batch reaching `convert_gpu_to_gpu` is NOT the same gpu_table_representation instance whose writer event was recorded (perhaps wrapped/rewrapped through an un-migrated path), OR (b) there's a different aggregate output flowing into convert_gpu_to_gpu that originated from an un-migrated site. Inventory of writer functions in remaining backtraces:
  - `gpu_pipeline_task::compute_task` (17 occurrences) — generic; could include any operator's execute() path not yet migrated
  - `gpu_aggregate_impl::local_grouped_aggregate` (19 occurrences) — already migrated
  - `gpu_pipeline_task::publish_output` (2 occurrences) — through `sirius_physical_left_delim_join::sink` to a downstream `local_grouped_aggregate` — left_delim_join::sink itself is NOT migrated.

### Phase 14 Diff-Empty Assertion (post-revert, working tree)
- `git diff -- src/include/pipeline/task_scheduler.hpp src/pipeline/task_scheduler.cpp src/include/creator/task_creator.hpp test/cpp/operator/test_physical_hash_join_mgpu.cpp | wc -l` = **0**
- `grep -c "_no_pref_rr_counter" src/include/pipeline/task_scheduler.hpp` = 0
- `grep -c "_no_pref_rr_counter" src/pipeline/task_scheduler.cpp` = 0
- `sed -n '642p' test/cpp/operator/test_physical_hash_join_mgpu.cpp` = `  constexpr int kIterations = 3;` (baseline)

### HYG-02 Baseline
- `grep -rn "rmm::cuda_stream_default" src/ --include="*.cpp" --include="*.cu" --include="*.hpp" | wc -l` = **40** (preserved; same as pre-fix baseline)

### Test Filter Used for Verification
- Q11-alone: `[TPC-H][parquet] gpu_execution - TPC-H Query 11 parquet` (1 test case, 7s)
- Cumulative: `[TPC-H][parquet]` (22 test cases × 2-GPU, 1800s timeout reached at Q11)

### Wall-Clock Spent (this resume agent only)
- ~50 min total (one full SCHED-RR build at ~30s + one 30-min test timeout + analysis/commit)

## Why The Fix Is Incomplete (Diagnosis)

The cucascade-side primitive (writer event + cudaStreamWaitEvent) is correctly implemented per the cucascade source inspection. The Sirius-side migration covers the FIRST-error writer (`gpu_aggregate_impl::local_grouped_aggregate`), but Q11 batch-mode hang persists. Three contributing factors visible from the post-fix sanitizer log:

1. **`sirius_physical_left_delim_join::sink` is NOT migrated.** This sink is in the publish_output path (2 sanitizer errors) and produces output without recording a writer event. Q11's plan shape is exactly DELIM_JOIN over a 3-table join (CONTEXT.md: "DELIM_JOIN over the same three-table join"), so this is on the bug's critical path.
2. **`gpu_pipeline_task::compute_task` (17 occurrences in writer backtraces)** is a generic frame — it could be wrapping multiple operators (`sirius_physical_hash_join`, `sirius_physical_grouped_aggregate_merge`, `sirius_physical_nested_loop_join`, `sirius_physical_sort_partition`, `sirius_physical_merge_sort`, `gpu_partition_impl`, `gpu_merge_impl`, `gpu_order_impl`) — many of which still use the 2-arg `make_data_batch` (no writer_stream) per the grep inventory captured during this plan.
3. **`scan/duckdb_scan_executor.cpp` and `scan/parquet_scan_task.cpp`** construct `cucascade::data_batch` directly via `std::make_shared<cucascade::data_batch>(...)` rather than going through `make_data_batch` — these are scan-source paths that produce GPU-resident representations and therefore also need writer-event recording.

Full migration audit (NOT done in this plan due to time budget):
```bash
grep -rn "make_data_batch\|std::make_shared<cucascade::data_batch>\|std::make_unique<cucascade::gpu_table_representation>\|std::make_unique<gpu_table_representation>" src/ --include="*.cpp" --include="*.cu" --include="*.hpp"
```
The grep inventory done during this plan identified ~30 producer call sites; only 8 are migrated. The remaining ~22 are migration debt for a Plan 13-05 follow-up.

## Self-Check: PASSED

Files created/modified verified on disk:
- `cucascade @ 7409c60`: FOUND
- `src/include/data/data_batch_utils.hpp` (writer_stream overloads): FOUND
- `src/op/aggregate/gpu_aggregate_impl.cpp` (writer_stream call): FOUND (lines 114, 322)
- 6 other migrated files: FOUND (per `git diff HEAD~1 HEAD --stat`)
- `13-04-SUMMARY.md` (this file): FOUND

Commit `b487807` exists in `git log --oneline -3`: FOUND.

Phase 14 working-tree-only files diff-empty against base: 0 lines (verified post-revert).
HYG-02 count = 40 (verified post-revert).

## Recommendation for Wave 5 / Plan 13-05

Phase 13's user-facing acceptance criteria (Q11 cumulative-state passes under SCHED-RR) are NOT met. Two paths forward:

**Option A (recommended):** Spawn a Plan 13-05 fix-completion plan that:
1. Audits the full ~30 producer sites via the grep inventory above.
2. Migrates every site that produces a fresh GPU representation on a non-default writer stream.
3. Re-runs the sanitizer + the Q1-Q11 cumulative-state test under SCHED-RR.
4. Iterates until sanitizer error count is 0 (or all remaining errors are at sites where the writer is genuinely on the consumer's stream — i.e., same-device same-stream, where waiting is moot).

**Option B (riskier):** Make the writer-event recording the DEFAULT for the 2-arg `make_data_batch` overloads — but this requires every caller to thread the writer stream from somewhere, which means a larger API change. Reject — Option A is more minimal-patch-aligned with Phase 12's template.

**Phase 14 unblock status:** Phase 13 deliverable is INCOMPLETE; Phase 14 (SCHED-RR distribution) cannot ship until Plan 13-05 closes the residual writer-event coverage gap. The cucascade infrastructure is in place; only Sirius-side migration is incomplete.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Phase 14 SCHED-RR diff is documentation-style, not git-applicable**

- **Found during:** Step 5 (re-apply Phase 14 patch for verification)
- **Issue:** `14-CONTEXT.md` contains a unified diff but missing `--- a/...` / `+++ b/...` headers; `git apply` rejects.
- **Fix:** Applied the diff via direct `Edit` tool calls (per Wave 1's same deviation, recorded in `13-01-SUMMARY.md`). After verification, reverted via `git checkout -- <files>`.
- **Files modified (working tree only, NOT committed):** `src/include/pipeline/task_scheduler.hpp`, `src/pipeline/task_scheduler.cpp`.
- **Commit:** N/A (working-tree-only)

### Auth Gates

None — all work via MCP, no human verification or external auth required.

### Architectural Changes Asked

None Wave-2 directly — proceeded with the Wave 2 directive (cucascade-side fix, submodule bump). The Plan 13-05 follow-up scope IS an architectural decision (whether to migrate every producer site one-by-one OR refactor `make_data_batch` to ALWAYS take a writer stream) — but that decision is for the orchestrator, not this Wave 4 executor.

## Known Stubs

None — every file modified by this plan was wired to real call sites; no placeholder/TODO logic introduced.

## Deferred Issues

1. **~22 un-migrated producer sites** in `src/op/`, `src/op/scan/`, `src/op/merge/`, `src/op/order/`, `src/op/partition/`, `src/legacy/` — see grep inventory in "Why The Fix Is Incomplete" section above.
2. **Q11 cumulative-state under SCHED-RR still SIGTERMs at 1800s.** Phase 13 acceptance criterion is NOT met by this plan.
3. **`compute_task` writer-frame disambiguation:** the 17 generic backtraces need per-operator attribution to know exactly which operators must be migrated. Could be done by adding a new sanitizer log after Plan 13-05 migration progress is made, or by binary-search migration (migrate 4 most likely sites at a time, run sanitizer, observe error-count delta).

## Cumulative Phase 13 Status

| Plan | Verdict | Notes |
|------|---------|-------|
| 13-01 | DONE | Authoritative repro reproduces; cheap repro DEAD on consumer host |
| 13-02 | DONE | FIRST race localized to cucascade/src/data/representation_converter.cpp:801; recommended fix shape: writer-event lineage |
| 13-03 | DONE | All 4 CONTEXT.md hypotheses corroborated DEAD; Wave 4 fix scope unambiguous |
| 13-04 | **PARTIAL** | Cucascade infrastructure landed + 7 producer sites migrated; ~22 sites remain; Q11 cumulative-state STILL hangs |
| 13-05 | (not yet planned) | Recommended: plan to complete writer-event migration across remaining producer sites |
