---
phase: 02-mutation-paths-and-lifecycle
plan: 03
status: complete
started: 2026-04-22
completed: 2026-04-22
gap_closure: true
---

## Summary

Removed all 18 redundant `cucascade::data_batch::to_idle()` calls from the 3 Phase 2 files. The cucascade RAII accessor types (`read_only_data_batch`, `mutable_data_batch`) have destructors that automatically transition batches back to idle state, making every explicit `to_idle(std::move(ro))` call redundant and a source of `[[nodiscard]]` compile errors under `-Werror`.

## Changes

| File | to_idle Removed | to_read_only Preserved |
|------|----------------|----------------------|
| `src/op/sirius_physical_result_collector.cpp` | 11 | 2 |
| `src/include/data/convertible_data_batch.hpp` | 4 | 5 |
| `src/include/data/convertible_gpu_pipeline_task.hpp` | 3 | 5 |
| **Total** | **18** | **12** |

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Remove redundant to_idle() calls from all three Phase 2 files | ✓ Complete |

## Self-Check: PASSED

- [x] Zero `to_idle` calls remain in the 3 Phase 2 files
- [x] `to_read_only()` accessor acquisitions preserved
- [x] `to_mutable()` / `try_to_mutable()` patterns in convertible_data_batch.hpp preserved
- [x] No new code added — pure deletion of redundant cleanup calls

## Commits

- `73b6358a` — fix(02-03): remove 18 redundant to_idle() calls from Phase 2 files

## Key Files

### Modified
- `src/op/sirius_physical_result_collector.cpp` — 11 to_idle calls removed from sink_single_batch
- `src/include/data/convertible_data_batch.hpp` — 4 to_idle calls removed from bytes_in_space, get_bytes_in_space, try_get_batch
- `src/include/data/convertible_gpu_pipeline_task.hpp` — 3 to_idle calls removed from convert, bytes_in_space, has_matching_batches

## Deviations

None.

## Issues

None.
