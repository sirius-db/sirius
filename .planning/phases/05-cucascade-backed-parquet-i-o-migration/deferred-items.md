# Phase 05 Deferred Items

Issues discovered during execution but deferred because they fall outside the
plan's scope.

## From Plan 05-05 execution (2026-04-21)

### test_parquet_scan_task - single threaded small table — FAILS after Plan 05-04

- **Location:** `test/cpp/scan/test_parquet_scan_task.cpp:664` (and likely other
  `parquet_scan_task_global_state` direct constructions in that file).
- **Failure message:** `"[parquet_scan_task_global_state] No GPU io_backends configured — SiriusContext::initialize() must have populated at least one (Approach C seeding via task_creator required)."`
- **Root cause:** Plan 05-04's commit `787a15e` added a throw on empty
  `gpu_io_backends` in `parquet_scan_task_global_state`. Tests that directly
  construct the global state without going through `task_creator` (which is
  the Approach-C seeding site) pass an empty map and trigger the throw.
- **Scope:** This test is in the `parquet_scan_task` test file owned by
  Plan 05-04 (the same plan that introduced the requirement). Plan 05-05
  only touches `sirius_parquet_metadata_scan_operator.cpp` +
  `iceberg_scan_task.cpp` — this test failure is NOT caused by Plan 05-05's
  changes and the fix (seeding the map in the test) belongs to Plan 05-04
  or Plan 05-06 (final cleanup).
- **Observation during Plan 05-05:** The rest of the test suite passes
  (947/948). Plan 05-05's own tests (metadata scan operator) pass with the
  new `make_test_io_backend()` helper pattern; Plan 05-04 should adopt the
  same helper for `test_parquet_scan_task.cpp`.
- **Status:** Deferred to Plan 05-04 or Plan 05-06. Phase 5 sign-off requires
  this to be fixed before closing.
