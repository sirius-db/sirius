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
