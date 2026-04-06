---
phase: 02-request-execution-and-api
plan: 02
status: complete
started: 2026-04-06T15:30:00Z
completed: 2026-04-06T15:45:00Z
duration: ~15min
tasks_completed: 1
tasks_total: 1
deviations: 2
---

## Summary

Rewrote all 7 existing tests that called the removed `run_downgrade_pass` methods to use the new `request_free_memory`/`request_downgrade` API. Added 3 new tests covering async futures, predicate-driven dispatch, and partial fulfillment. Fixed two bugs in the executor discovered during testing.

## Key Files

### Modified
- `test/cpp/downgrade/test_downgrade_executor.cpp` — 12 tests total (7 rewritten + 2 unchanged + 3 new)
- `src/downgrade/downgrade_executor.cpp` — Bug fixes for stream creation and zero-target collection

## Deviations

1. **Unconditional CUDA stream creation**: `start()` previously only created `_stream` when `_memory_space != nullptr`. The request API needs the stream regardless of monitor loop status. Fixed to create unconditionally.

2. **Zero-target candidate collection**: `collect_all_candidates()` with `target_bytes == 0` (used by `request_downgrade`) broke immediately since `0 >= 0` is true. Added `has_byte_limit` guard so zero means "collect all."

3. **Monitor loop interference**: Tests that enabled the monitor loop (by passing non-null `gpu_space`) suffered from race conditions where the monitor pushed requests before the test's request. Fixed by passing `nullptr` for `gpu_space` in all request API tests — they don't test the monitor.

## Verification

- `pre-commit run -a` on changed files: PASS
- `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make`: PASS (0 errors)
- `sirius_unittest "[downgrade_executor]"`: 12/12 tests pass, 51 assertions
- `grep -c "run_downgrade_pass" test/cpp/downgrade/test_downgrade_executor.cpp`: 0
- `grep -c "add_new_repository" test/cpp/downgrade/test_downgrade_executor.cpp`: 9 occurrences

## Self-Check: PASSED
