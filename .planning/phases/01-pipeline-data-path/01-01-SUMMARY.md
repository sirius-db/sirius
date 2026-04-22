---
phase: 01-pipeline-data-path
plan: 01
subsystem: pipeline
tags: [cucascade, data_batch, read_only_data_batch, mutable_data_batch, raii, locking]

# Dependency graph
requires: []
provides:
  - "read_only_pipelineable_operator_data class in sirius_physical_operator.hpp"
  - "read_only_partitioned_operator_data class in sirius_physical_operator.hpp"
  - "lock_or_prepare_batch returning optional<read_only_data_batch>"
  - "prepare_for_processing returning optional<vector<read_only_data_batch>>"
affects:
  - 01-02-PLAN
  - gpu_pipeline_task
  - downgrade_executor
  - all pipeline operators consuming pipelineable_operator_data

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "to_read_only() -> optional return on space match; readonly_to_mutable() + convert_to() + mutable_to_readonly() on space mismatch"
    - "move-only RAII wrapper types for locked batch collections"
    - "sibling class hierarchy: read_only_pipelineable_operator_data inherits operator_data (not pipelineable_operator_data)"

key-files:
  created: []
  modified:
    - src/include/op/sirius_physical_operator.hpp
    - src/include/pipeline/batch_lock_utils.hpp
    - src/op/sirius_physical_operator.cpp

key-decisions:
  - "read_only_pipelineable_operator_data inherits operator_data directly (D-05: sibling, not subclass of pipelineable_operator_data)"
  - "read_only_partitioned_operator_data inherits read_only_pipelineable_operator_data (D-06: mirrors partitioned -> pipelineable pattern)"
  - "lock_or_prepare_batch uses blocking to_read_only() then readonly_to_mutable() + convert_to() + mutable_to_readonly() for space mismatches (D-01, D-02, D-03)"
  - "prepare_for_processing return type changed from optional<vector<data_batch_processing_handle>> to optional<vector<read_only_data_batch>>"

patterns-established:
  - "RAII batch locking: always acquire read_only_data_batch via to_read_only() for shared access"
  - "Space mismatch conversion: readonly_to_mutable() -> convert_to<T>() -> mutable_to_readonly() avoids holding exclusive lock longer than needed"
  - "Move-only semantics on all RAII batch containers to prevent lock aliasing"

requirements-completed: [TYPE-01, TYPE-02, PIPE-01, PIPE-05]

# Metrics
duration: 3min
completed: 2026-04-22
---

# Phase 01 Plan 01: Foundational RAII Batch Types and lock_or_prepare_batch Rewrite Summary

**Two new move-only RAII wrapper types for read-only locked data batches and rewritten `lock_or_prepare_batch` using cucascade's 3-class API transitions (to_read_only / readonly_to_mutable / mutable_to_readonly)**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-04-22T03:17:09Z
- **Completed:** 2026-04-22T03:20:27Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added `read_only_pipelineable_operator_data` class as a sibling of `pipelineable_operator_data` (inherits `operator_data` directly), storing `vector<cucascade::read_only_data_batch>` with move-only semantics
- Added `read_only_partitioned_operator_data` class extending `read_only_pipelineable_operator_data` with partition index, mirroring the existing `partitioned_operator_data` structure
- Rewrote `lock_or_prepare_batch` to return `optional<read_only_data_batch>` using the new cucascade 3-class API: `to_read_only()` for space match, `readonly_to_mutable()` + `convert_to<T>()` + `mutable_to_readonly()` for space mismatch
- Updated `pipelineable_operator_data::prepare_for_processing` declaration and definition to return `optional<vector<read_only_data_batch>>` instead of `optional<vector<data_batch_processing_handle>>`
- Removed all references to deprecated API: `data_batch_processing_handle`, `lock_for_processing_status`, `wait_to_lock_for_processing`, `try_to_lock_for_in_transit`, `try_to_release_in_transit`

## Task Commits

Each task was committed atomically:

1. **Task 1: Define read_only_pipelineable_operator_data and read_only_partitioned_operator_data** - `c660ed25` (feat)
2. **Task 2: Rewrite lock_or_prepare_batch to return read_only_data_batch** - `f951f198` (feat)

## Files Created/Modified
- `src/include/op/sirius_physical_operator.hpp` - Added two new RAII wrapper type classes; updated `prepare_for_processing` declaration
- `src/include/pipeline/batch_lock_utils.hpp` - Rewrote `lock_or_prepare_batch` with new cucascade 3-class API transitions
- `src/op/sirius_physical_operator.cpp` - Updated `prepare_for_processing` implementation to use `read_only_data_batch`

## Decisions Made
- `read_only_pipelineable_operator_data` is a sibling (not subclass) of `pipelineable_operator_data` per D-05, inheriting `operator_data` directly to keep the two hierarchies independent
- `read_only_partitioned_operator_data` inherits `read_only_pipelineable_operator_data` per D-06, maintaining the same structural pattern as the existing `partitioned_operator_data`
- `lock_or_prepare_batch` uses blocking `to_read_only()` (not try-based) per D-01, consistent with CLAUDE.md constraint on non-blocking reads
- Space mismatch path: `readonly_to_mutable()` + `convert_to<T>()` + `mutable_to_readonly()` per D-02 — upgrades only when conversion is needed, then downgrades to minimize exclusive lock hold time

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. The cucascade submodule was not initialized in the worktree, but the main repo's cucascade headers were available for reference and the worktree's source files reference it correctly via CMake.

## Next Phase Readiness
- Plan 01-02 can proceed: the foundational types and lock function are in place
- All call sites that currently use `prepare_for_processing` will need updating to consume `vector<read_only_data_batch>` instead of `vector<data_batch_processing_handle>`
- `gpu_pipeline_task::compute_task` will need to accept `vector<read_only_data_batch>` input in Plan 02

---
*Phase: 01-pipeline-data-path*
*Completed: 2026-04-22*
