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
