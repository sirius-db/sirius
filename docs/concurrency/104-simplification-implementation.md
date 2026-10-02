# Concurrency simplification implementation

This journal tracks opportunities 1–4 from
[the simplification review](103-concurrency-stack-simplification-review.md).
Each numbered item is implemented in its own commit on `concurrency4`.

## 1. Mandatory lifecycle binding

- Creator, scheduler, GPU/base executor, downgrade executor, scan manager,
  prefetcher and spill-task wrappers require a registry reference at construction.
  The owning runtime or fixture must outlive them. Removed the mutable registry
  setters and null-registry execution paths.
- Per-query executor drain always leaves shared managers running. Submission and
  work-lease accounting now apply to standalone executor/spill tests too.
- `retain_resources()` rejects unknown query IDs. Synthetic plan fixtures register
  and retire their lifecycle entries as well as their repository managers.
- OOM tests route retries through a real scheduler and creator. GPU executors no
  longer retry privately when creator wiring is missing; that is a reported setup
  error. Recording-only creator mocks still receive a registry but publish no work.
- Kept repeated cleanup of unknown/retired IDs harmless. Kept the distinct
  submission and work barriers and the order of rejected-task destruction.

Validation: `pixi run make` passed; 114 targeted cases / 1,110 assertions and
6 standalone plan/merge cases / 311 assertions passed. Formatting/lint passed.
Logs: `/tmp/simplification-1-build.log`, `/tmp/simplification-1-tests.log`, and
`/tmp/simplification-1-plan-tests.log`. Multi-GPU cases require a multi-GPU host.

## 2. Publish complete pin metadata

- Removed `attach_mvcc_metadata()` and `attach_proven_unique_columns()` from the
  production interface. All metadata now enters through `pinned_entry_metadata`
  and the existing `apply_pin_metadata()` publication helper.
- Migrated uniqueness tests to publish proofs with data. The merge regression now
  supplies a proof for discarded incoming bytes and verifies it is ignored, while
  an existing proof for retained data survives.
- Replaced the mutation/epoch test with complete-publication checks. Replacement
  tests retain an old reader and verify both failure atomicity and stable old
  metadata after a successful replacement.
- Removed the missing-entry test specific to the deleted mutation API; SQL MVCC
  tests and publication tests cover the supported interface. Kept source identity,
  merge shape checks and generation ownership unchanged.

Validation: build and formatting/lint passed; 49 pin/uniqueness/MVCC cases passed
with 1,045 assertions. Logs: `/tmp/simplification-2-build.log` and
`/tmp/simplification-2-tests.log`.

## 3. Centralize retirement

- Added `SiriusContext::retire_query_work(query_id, mode)`: close publication and
  settle publishers, stop scan producers, validate completion or drain cancelled
  work, then wait for borrowers. Engine success/error/destruction and window
  cleanup/backstops all use it.
- Added a private `release_query_state()` phase shared by normal and best-effort
  cleanup. It runs only after retirement succeeds, releasing creator state, the
  retained plan, telemetry placements, repositories and scan providers in order.
- Creator state now remains available until asynchronous work has settled. Engine
  retirement does not release the retained plan prematurely: post-execution plan
  inspection and failed-cleanup retention still work.
- Kept scheduler barriers for standalone users and the final runtime work barrier
  for partial initialization. No new global mutex or retirement flags were added.
- Protected scheduler retirement logging so a diagnostic failure cannot abort
  mandatory cleanup. Successful queries still report unexpectedly queued tasks;
  they do not silently take the cancellation path.
- Added integration coverage for both retirement modes, ownership through window
  close, and repeated retirement before and after registry removal.

Validation: build and formatting/lint passed. The first test selection passed
73 cases / 1,083 assertions; its one failure was the hidden worker-pressure case
requiring an unset `SIRIUS_TEST_TPCH_DIR`. Re-running that case with the repository's
`test/cpp/integration/data/parquet` fixture passed (1 case / 6 assertions).
Logs: `/tmp/simplification-3-build.log`, `/tmp/simplification-3-tests.log`, and
`/tmp/simplification-3-pressure-tests.log`. This is regression coverage with the
bundled fixture, not a throughput qualification.
