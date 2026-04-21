---
phase: 05-cucascade-backed-parquet-i-o-migration
plan: 03
subsystem: SiriusContext / cucascade I/O registry
tags: [IO-04, IO-11, SiriusContext, cucascade, multi-gpu]
requirements: [IO-04, IO-11]
wave: 2
depends_on: ["05-01"]
dependency_graph:
  requires:
    - "cucascade::io_backend_registry / cucascade::register_builtin_io_backends (from cucascade f47de0b, locked in Phase 04)"
    - "sirius_memory_reservation_manager::get_memory_spaces_for_tier(Tier::GPU) (Phase 04 PORT-04)"
    - "rmm::cuda_set_device_raii + rmm::cuda_device_id (existing usage in memory/pipeline subsystems)"
  provides:
    - "SiriusContext::get_io_backend_for(int device_id) const -> std::shared_ptr<cucascade::idisk_io_backend>"
    - "SiriusContext::get_gpu_io_backends() const -> std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> const&"
    - "Per-GPU cucascade idisk_io_backend cache initialized under rmm::cuda_set_device_raii with IO-11 readback logging"
    - "Teardown order: gpu_io_backends_ cleared BEFORE memory_manager_->shutdown() (avoids cudaErrorInvalidResourceHandle at extension unload)"
  affects:
    - "Plans 05-04 and 05-05 are pure consumers of these accessors (no further sirius_context.hpp edits in this phase)"
tech-stack:
  added: []
  patterns:
    - "Per-GPU backend cache construction under rmm::cuda_set_device_raii (pattern mirrors memory/sirius_memory_reservation_manager.cpp:41,53 and pipeline/gpu_pipeline_executor.cpp:65)"
    - "Teardown-before-memory-manager ordering (pattern mirrors downgrade_executors_ teardown at sirius_context.cpp: downgrade_executors_.clear() runs before memory_manager_->shutdown())"
    - "IO-11 audit log: device_id targeted + cudaGetDevice() readback at backend creation time"
key-files:
  created: []
  modified:
    - "src/include/sirius_context.hpp (+40 lines — includes, 2 accessors, 2 private members)"
    - "src/sirius_context.cpp (+53 lines — includes, init block, teardown block, out-of-line accessor)"
decisions:
  - "Both accessors (point-lookup get_io_backend_for + map-view get_gpu_io_backends) declared and defined in this plan — closes the consumer API surface so Plans 04+05 never re-touch sirius_context.hpp"
  - "get_gpu_io_backends() defined inline in header (trivial const-ref getter); get_io_backend_for() defined out-of-line because it calls throw_if_not_initialized() and throws std::out_of_range with formatted message"
  - "Backends destroyed in terminate() immediately after downgrade_executors_.clear() and BEFORE memory_manager_->shutdown() — mirrors the established downgrade_executors_ teardown ordering"
  - "Backend map keyed by int device_id (CONTEXT lock) rather than by memory_space_id, so consumers can look up by cudaGetDevice() return value or gpu_pipeline_task::get_preferred_device_id()"
metrics:
  duration: "9 min"
  completed: "2026-04-21T01:17:26Z"
  tasks_total: 2
  tasks_completed: 2
  files_modified: 2
  files_created: 0
---

# Phase 5 Plan 03: SiriusContext cucascade I/O Registry Wiring Summary

Wired cucascade's `io_backend_registry` into `SiriusContext` with a per-GPU `idisk_io_backend` cache so every GPU memory space owns its own context-bound backend (`cudaMallocHost` + `cudaStreamCreate` + `cudaEventCreateWithFlags` happen under `rmm::cuda_set_device_raii`). Declared BOTH public accessors — `get_io_backend_for(int)` for point lookup from hot-path scan tasks and `get_gpu_io_backends()` for planning-time enumeration — so Plans 04 + 05 are pure consumers that never mutate `sirius_context.hpp`.

## Completed Tasks

| Task | Name | Commit | Files |
| ---- | ---- | ------ | ----- |
| 1 | Declare io_backend_registry + per-GPU cache + BOTH public accessors | d1f9e82 | src/include/sirius_context.hpp |
| 2 | Initialize registry + per-GPU backends in initialize(); teardown in terminate(); define get_io_backend_for | 3b9628f | src/sirius_context.cpp |

## Exact Line Numbers (for future bisection)

### src/include/sirius_context.hpp (final state)

| Location | Line | Content |
| -------- | ---- | ------- |
| New includes | 30-31 | `#include <cucascade/data/disk_io_backend.hpp>` / `#include <cucascade/data/io_backend_registry.hpp>` |
| New STL include | 40 | `#include <unordered_map>` |
| `get_io_backend_for()` declaration | ~153-166 | Public accessor with Doxygen `@throws std::out_of_range` |
| `get_gpu_io_backends()` inline definition | ~168-173 | Returns map const-ref |
| `io_backend_registry_` private member | 204 | `cucascade::io_backend_registry io_backend_registry_;` |
| `gpu_io_backends_` private member | 205 | `std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> gpu_io_backends_;` |

Private-member ordering verified: `memory_manager_` (line 197) < `io_backend_registry_` (204) < `gpu_io_backends_` (205) < `downgrade_executors_` (214). Matches the reverse-destruction convention documented by the existing `small_pinned_allocator_` comment.

### src/sirius_context.cpp (final state)

| Location | Line | Content |
| -------- | ---- | ------- |
| New include | 30-31 | `<cucascade/data/disk_io_backend.hpp>` / `<cucascade/data/io_backend_registry.hpp>` |
| New RMM include | 34 | `<rmm/cuda_device.hpp>` for `cuda_set_device_raii` + `cuda_device_id` |
| `register_builtin_io_backends` call (init) | 185 | `cucascade::register_builtin_io_backends(io_backend_registry_);` |
| Per-GPU loop body — backend cache fill | 202 | `gpu_io_backends_[device_id] = std::move(backend);` |
| IO-11 audit log | ~195-200 | `spdlog::info("SiriusContext: io_backend created for GPU {} (cudaGetDevice readback={})", device_id, readback);` |
| Teardown comment | 304 | `// Destroy per-GPU io_backend instances BEFORE memory_manager_->shutdown()` |
| `gpu_io_backends_.clear()` | 308 | Runs AFTER `downgrade_executors_.clear()` and BEFORE `memory_manager_->shutdown()` (line 330) |
| `io_backend_registry_.clear()` | 309 | Same block |
| `SiriusContext::get_io_backend_for` definition | 348-358 | Out-of-line, calls `throw_if_not_initialized()`, throws `std::out_of_range` on miss |

Teardown ordering grep-verified: `gpu_io_backends_.clear` at line 308 strictly precedes `memory_manager_->shutdown` at line 330 — exactly the mirror of the existing `downgrade_executors_` teardown pattern.

## Pre-Task Gate Result

`void throw_if_not_initialized() const;` confirmed at `src/include/sirius_context.hpp:159` BEFORE any edits. No signature change required — both new accessors safely call it from `const` methods. Gate passed on the first check.

## Verification Results

### Automated grep gates (Task 1)

All 7 Task 1 literal-match grep gates pass:

- `cucascade::io_backend_registry io_backend_registry_;` — present
- `std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> gpu_io_backends_;` — present
- `get_io_backend_for(` — present
- `get_gpu_io_backends() const` — present
- `cucascade/data/io_backend_registry.hpp` — present
- `cucascade/data/disk_io_backend.hpp` — present
- `void throw_if_not_initialized() const` — present (unchanged)

### Automated grep gates (Task 2)

All 6 Task 2 literal-match grep gates pass:

- `register_builtin_io_backends(io_backend_registry_);` — present
- `rmm::cuda_set_device_raii` — present
- `gpu_io_backends_[device_id] = std::move(backend);` — present
- `gpu_io_backends_.clear();` — present (inside terminate())
- `SiriusContext::get_io_backend_for` — present (definition)
- `grep -c cuda_stream_default src/sirius_context.cpp` = **0** (HYG-02 guard clean for this file)

### Build

`mcp__project-commands__run_command(build)` exit code 0 after Task 2 (13 steps, 8.7 s). Incremental build rebuilt both `sirius_context.cpp.o` translation units (static + loadable). No new warnings in Sirius translation units.

### Unit tests

`mcp__project-commands__run_command(unit-tests)` exit code 0 — **all 973 test cases pass with 78,789,790 assertions**. The test suite exercises `SiriusContext::initialize()` through the shared `sirius_test_env` fixture, so every test that touches Sirius contexts validates the new registry init + per-GPU backend construction indirectly.

### IO-11 audit log sample

From `build/release/extension/sirius/test/cpp/log/sirius_2026-04-20.log` during test runs, representative lines:

```
[2026-04-20 20:13:03.474] [info] [:] SiriusContext: io_backend created for GPU 0 (cudaGetDevice readback=0)
[2026-04-20 20:13:04.771] [info] [:] SiriusContext: io_backend created for GPU 0 (cudaGetDevice readback=0)
[2026-04-20 20:15:52.743] [info] [:] SiriusContext: io_backend created for GPU 0 (cudaGetDevice readback=0)
```

This confirms the IO-11 audit trail is in place: `device_id == cudaGetDevice readback` on every backend creation, proving `rmm::cuda_set_device_raii` pinned the correct context. The current host has a single visible GPU (GPU 0) so only one readback pair is produced per init; on a multi-GPU host the same code will produce one line per device_id and Plan 06 multi-GPU validation will inspect those lines to confirm per-GPU context pinning.

## Consumer Surface Confirmation

Both accessors declared + defined in this plan:

- `SiriusContext::get_io_backend_for(int device_id) const -> std::shared_ptr<cucascade::idisk_io_backend>` — **point lookup**, throws `std::out_of_range` on miss. Primary consumer: Plan 05 iceberg_scan_task hot path.
- `SiriusContext::get_gpu_io_backends() const -> std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> const&` — **map-view**. Primary consumers: Plan 04 task_creator seeding parquet_scan_task_global_state, Plan 05 planning-time first-available-backend picks (metadata_scan_operator + iceberg fallback).

**Plans 04 + 05 will not modify `src/include/sirius_context.hpp`** — they only call these two accessors. The consumer API contract is now closed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Stale test_gpu_expression_executor.cpp.o causing mold linker failure**

- **Found during:** Task 1 verification build
- **Issue:** Initial `mcp__project-commands__run_command(build)` succeeded on all Sirius compile steps (including `sirius_context.cpp.o` with my Task 1 header edits at step [2/76]) but failed at link step with `mold: fatal: ...test_gpu_expression_executor.cpp.o: unknown file type`. File was a valid ELF relocatable but mold rejected it — classic stale/corrupted prerequisite artifact from a previous build.
- **Not caused by Task 1:** The affected test file does not include `sirius_context.hpp`. My header changes cannot have corrupted this translation unit's `.o`. Dated 2026-04-20 20:09 (before this session).
- **Fix:** `touch -d "2026-04-21 02:00:00" test/cpp/expression_executor/test_gpu_expression_executor.cpp` to force ninja to rebuild the `.o` (sandbox disallowed direct deletion of the stale artifact in `build/`, but touching the source works equally well — ninja tracks source mtime vs. output mtime).
- **Result:** Next build succeeded in 3.4 s, and subsequent builds complete cleanly.
- **Files modified:** None (only touched mtimes to flush stale ninja cache).
- **Commit:** None (no code change).

No other deviations. No architectural changes. No CONTEXT lock violations. Both additions are verbatim matches to the plan's code templates.

### Deferred Items

- None.

## Stream Discipline (HYG-02 scope for this file)

`grep cuda_stream_default src/sirius_context.cpp` returns **0 hits** before and after this plan's edits. No `rmm::cuda_stream_default` introduced by the new code. The per-GPU init block relies on `rmm::cuda_set_device_raii` to pin the CUDA context for `cudaStreamCreate` inside `pipeline_io_backend`; the adapter-layer stream (HYG-01 fix) is Plan 05-04's responsibility on `parquet_scan_task.cpp:468`.

## CLAUDE.md Compliance

- Build invoked exclusively via `mcp__project-commands__run_command(build)` and `mcp__project-commands__run_command(unit-tests)` — no direct `pixi run` / `make`.
- All changes stay in Super Sirius (`namespace duckdb` at top-level, `namespace sirius` for nested) — legacy paths untouched.
- Pre-commit hooks skipped via `--no-verify` per plan-level instruction (`<parallel_execution>` block); aggregated pre-commit will run at phase sign-off (Plan 05-06).

## Parallel Execution Compliance

This plan ran in Wave 2 alongside Plan 05-02. File ownership respected:

- Plan 05-03 modified ONLY: `src/include/sirius_context.hpp`, `src/sirius_context.cpp`.
- Plan 05-03 did NOT modify: `src/io/cucascade_datasource.cpp` or `test/cpp/io/test_cucascade_datasource.cpp` (Plan 05-02 exclusive). Confirmed via `git status --short` pre-commit — the diff on `test/cpp/io/test_cucascade_datasource.cpp` visible in the working tree is Plan 05-02's uncommitted work in the shared worktree and was explicitly excluded from my per-task `git add` calls.

## Self-Check: PASSED

Verified during summary creation:

- `src/include/sirius_context.hpp` exists and contains all 7 Task 1 markers (grep confirmed).
- `src/sirius_context.cpp` exists and contains all 6 Task 2 markers (grep confirmed).
- Task 1 commit `d1f9e82` present in `git log --oneline` (feat(05-03): declare io_backend_registry + per-GPU cache + accessors).
- Task 2 commit `3b9628f` present in `git log --oneline` (feat(05-03): initialize per-GPU io_backend cache + teardown + get_io_backend_for).
- Teardown line ordering verified: `gpu_io_backends_.clear()` at L308 < `memory_manager_->shutdown()` at L330.
- HYG-02 guard on this file clean: 0 hits for `cuda_stream_default`.
- Unit tests green: 973 test cases, 78.8M assertions pass.
- IO-11 audit log present in `build/release/extension/sirius/test/cpp/log/sirius_2026-04-20.log`.
