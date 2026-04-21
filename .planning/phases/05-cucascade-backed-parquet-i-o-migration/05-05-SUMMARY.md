---
phase: 05-cucascade-backed-parquet-i-o-migration
plan: 05
subsystem: op/scan (iceberg + metadata scan)
tags: [IO-05, IO-06, cucascade_datasource, iceberg, metadata-scan, approach-a]
requirements: [IO-05, IO-06]
wave: 3
depends_on: ["05-02", "05-03", "05-04"]
dependency_graph:
  requires:
    - "sirius::io::cucascade_datasource (Plans 05-01 + 05-02 adapter)"
    - "parquet_scan_task_global_state::get_gpu_io_backends() const-ref accessor (Plan 05-04 Approach C plumbing)"
    - "SiriusContext::get_gpu_io_backends() read-only accessor (Plan 05-03)"
  provides:
    - "Metadata scan operator migrated — datasource::create replaced by cucascade_datasource (IO-05, 1 call-site)"
    - "Iceberg positional + equality delete-file reads migrated to cucascade_datasource via source_info{&ds} (IO-06, 2 call-sites)"
    - "Iceberg helper signatures extended (Approach A) with std::shared_ptr<cucascade::idisk_io_backend> backend parameter"
    - "iceberg_scan_task_global_state ctor forwards gpu_io_backends map to base (closes Plan 05-04 handoff gap — iceberg no longer runtime-fails)"
    - "task_creator.cpp iceberg branch seeded with SiriusContext::get_gpu_io_backends() (completes Approach-C wiring for iceberg)"
  affects:
    - "Plan 05-06 (phase sign-off) — IO-05 + IO-06 grep gates now 0 globally; iceberg runtime-fail unblocker cleared"

tech-stack:
  added: []
  patterns:
    - "Approach A for iceberg delete-file helpers — helper signatures take explicit std::shared_ptr<cucascade::idisk_io_backend> backend parameter; caller resolves via inherited get_gpu_io_backends() accessor. Avoids exposing SiriusContext to the helpers while preserving single-responsibility (helpers stay pure functions of path + backend + output)"
    - "Metadata scan operator accepts optional io_backend ctor parameter (default nullptr) + throws in execute() if null. Matches the pattern used in task_creator for parquet_scan_task: caller resolves backend once at construction time, operator stores and uses it in hot path"
    - "Test helper make_test_io_backend() — static io_backend_registry + std::call_once construction of a shared default backend for tests that instantiate metadata_scan_operator without a full SiriusContext"

key-files:
  created:
    - ".planning/phases/05-cucascade-backed-parquet-i-o-migration/deferred-items.md (tracks Plan 05-04 test_parquet_scan_task single-threaded test failure — out of scope for 05-05)"
  modified:
    - "src/include/op/scan/sirius_parquet_metadata_scan_operator.hpp (+20 lines — cucascade include, io_backend ctor param, _io_backend member)"
    - "src/op/scan/sirius_parquet_metadata_scan_operator.cpp (+18 lines — io/cucascade_datasource include, <filesystem> include, cucascade_datasource construction, null-backend throw)"
    - "test/cpp/scan/test_metadata_gpu_scan_operators.cpp (+30 lines — cucascade io_backend_registry + disk_io_backend includes, <memory> + <mutex> includes, make_test_io_backend helper, 2 direct TEST_CASE + run_two_pipeline_scan helper updated to pass backend)"
    - "src/op/scan/iceberg_scan_task.cpp (+65 lines — io/cucascade_datasource + cucascade/data/disk_io_backend + <filesystem> includes, backend param added to both helpers, stack-local cucascade_datasource + source_info{&ds}, build_delete_pipeline resolves first-GPU backend via base get_gpu_io_backends(), ctor gpu_io_backends forwarded through delegating chain to base)"
    - "src/include/op/scan/iceberg_scan_task.hpp (+12 lines — cucascade/data/disk_io_backend + <unordered_map> includes, gpu_io_backends ctor param on public + private delegating ctors)"
    - "src/creator/task_creator.cpp (+10 lines — iceberg branch updated to resolve gpu_io_backends via sirius_ctx->get_gpu_io_backends() and pass to iceberg_scan_task_global_state ctor — mirrors parquet_scan_task branch pattern)"

decisions:
  - "Approach A (locked in frontmatter) — iceberg helper signatures extended with std::shared_ptr<cucascade::idisk_io_backend> backend parameter. No deviation; frontmatter approach_locked was 'A' and that's exactly what shipped"
  - "Metadata scan operator ctor accepts optional io_backend parameter (default nullptr). Rationale: the operator is not yet constructed from production planner code (tests only), so a mandatory param would force a deep test refactor. Optional with a runtime throw on null lets tests construct the operator for non-execute-path tests and still enforces correctness at the hot-path call site. Test helper make_test_io_backend() provides a standalone backend so test paths that do call execute() have a working backend"
  - "Plan 05-05 completes the iceberg handoff Plan 05-04 explicitly deferred. Plan 05-04 SUMMARY stated: 'Iceberg path will runtime-fail until Plan 05-05 ships'. Extending iceberg_scan_task_global_state ctor + updating task_creator.cpp iceberg branch closes this gap — these edits are in Plan 05-04's declared file scope, but Plan 05-04 was complete and explicitly acknowledged the handoff. Treated as Rule 3 (blocking issue) auto-fix"
  - "iceberg_scan_task.hpp modifications were required to thread the gpu_io_backends map through the iceberg ctor. Header is not claimed by any other plan; scope expansion is minimal and tightly coupled to the .cpp changes"

patterns-established:
  - "Sirius datasource-adapter construction flows through a per-GPU backend passed explicitly from caller — never looked up via a global/singleton. This consistent pattern (metadata scan ctor + parquet_scan_task.compute_task + iceberg helpers Approach A) makes the dependency graph explicit and keeps SiriusContext at the task-creator layer only"
  - "Iceberg protected-ctor chain forwards gpu_io_backends via std::move through both the public and private delegating constructors, then into the base parquet_scan_task_global_state. This matches how all other inherited Phase 4 state (approximate_batch_size, file_paths, selected_column_indices) is threaded"

requirements-completed: [IO-05, IO-06]

# Metrics
metrics:
  duration: "~20min"
  completed: "2026-04-21T01:44:46Z"
  tasks_total: 2
  tasks_completed: 2
  files_modified: 6
  files_created: 1
---

# Phase 5 Plan 05: Metadata Scan + Iceberg Delete-File Migration Summary

Migrated the 3 remaining parquet-I/O call sites not covered by Plan 05-04: `sirius_parquet_metadata_scan_operator.cpp:251` (IO-05, metadata scan) and `iceberg_scan_task.cpp:57 + :120` (IO-06, positional + equality delete reads). Iceberg helpers gain a `std::shared_ptr<cucascade::idisk_io_backend>` parameter (Approach A); the metadata scan operator gains an optional `io_backend` ctor parameter with a test helper (`make_test_io_backend()`) for isolated test paths. Also completed the iceberg-ctor handoff that Plan 05-04 explicitly deferred — `iceberg_scan_task_global_state` ctor now forwards `gpu_io_backends` through the delegating chain to the base class, and `task_creator.cpp`'s iceberg branch seeds the map from `SiriusContext::get_gpu_io_backends()`. Pure-consumer invariant on `src/include/sirius_context.hpp` upheld.

## Completed Tasks

| Task | Name | Commit | Files |
| ---- | ---- | ------ | ----- |
| 1 | Migrate sirius_parquet_metadata_scan_operator.cpp:251 to cucascade_datasource | 3d74113 | sirius_parquet_metadata_scan_operator.{hpp,cpp}, test_metadata_gpu_scan_operators.cpp |
| 2 | Migrate iceberg delete-file helpers to Approach A (backend parameter) + source_info{&ds} | 1c15063 | iceberg_scan_task.cpp |
| 2b (Rule 3 auto-fix) | Wire iceberg_scan_task_global_state for gpu_io_backends propagation (closes Plan 05-04 handoff) | ce387f7 | iceberg_scan_task.{hpp,cpp}, task_creator.cpp |

## Pre/Post Call-Site Counts (Approach A invariant)

| Helper | Pre-edit | Post-edit | Delta |
| --- | --- | --- | --- |
| `read_positional_delete_file(` | 2 | 2 | 0 (invariant) |
| `read_equality_delete_file(` | 2 | 2 | 0 (invariant) |

Each count is 1 declaration + 1 call-site in the .cpp anonymous namespace. No call sites silently dropped; no new ones introduced.

## Grep Gate Results

### `src/op/scan/sirius_parquet_metadata_scan_operator.cpp`

| Gate | Pattern | Result |
| --- | --- | --- |
| IO-05 | `cudf::io::datasource::create` | 0 |
| IO-05 adapter present | `sirius::io::cucascade_datasource` | 1 |
| Adapter include | `io/cucascade_datasource.hpp` | 1 |
| First-backend consumer | `_io_backend` (use — excludes member decl) | 1 |
| HYG-02 | `cuda_stream_default` | 0 |

### `src/op/scan/iceberg_scan_task.cpp`

| Gate | Pattern | Result |
| --- | --- | --- |
| IO-06 filepath source_info | `source_info\{[^}]*delete_file_path[^}]*\}` | 0 |
| IO-06 adapter present | `sirius::io::cucascade_datasource` | 4 (2 callsites + 2 helper-doc refs) |
| Non-owning ptr pattern | `source_info{&ds}` | 2 |
| Stack-local adapter | `sirius::io::cucascade_datasource ds\{` | 2 |
| IO-05 | `cudf::io::datasource::create` | 0 |
| HYG-02 | `cuda_stream_default` | 0 |
| Approach A helper signature | `std::shared_ptr<cucascade::idisk_io_backend> backend` | 2 |
| `read_positional_delete_file(` call count | invariant check | 2 |
| `read_equality_delete_file(` call count | invariant check | 2 |

### Pure-consumer invariant (cross-cutting)

- `git diff src/include/sirius_context.hpp` → **0 lines** (header is not modified by Plan 05-05 — Plan 05-03 remains sole owner).

## Consumption Confirmation

Both SiriusContext accessors consumed from Plan 05-03's header without modification:

- `get_gpu_io_backends()` — consumed in `task_creator.cpp` iceberg branch (seeds the map into `iceberg_scan_task_global_state`).
- Base-class `get_gpu_io_backends()` on `parquet_scan_task_global_state` (Plan 05-04 accessor) — consumed in `iceberg_scan_task_global_state::build_delete_pipeline()` for first-backend pick.

Metadata scan operator uses the ctor-injected backend directly — no SiriusContext reference in the operator's translation unit.

## Iceberg Test Results

Iceberg Catch2 tests in `test/cpp/integration/test_gpu_execution_multi_format.cpp` (14 tests across V1, V2 positional, V2 equality, equality-stress fixtures) compile cleanly and correctly enter the guard path that WARNs and returns when the community iceberg DuckDB extension is not available (`if (!iceberg_available) { WARN("iceberg extension not available — skipping"); return; }`). This is the standard Tier-A behavior for this host. Execution-level validation is deferred to Tier-B (a host with the iceberg extension loaded) per the Phase 5 two-tier validation contract.

The full `mcp__project-commands__run_command(unit-tests)` run shows 947/948 tests passing. The single failing test is `parquet_scan_task - single threaded small table` (`test_parquet_scan_task.cpp:373`), which fails with `"[parquet_scan_task_global_state] No GPU io_backends configured — SiriusContext::initialize() must have populated at least one (Approach C seeding via task_creator required)"`. This failure is caused by Plan 05-04's mandatory-backend throw (`787a15e`) combined with a test that constructs `parquet_scan_task_global_state` directly (bypassing `task_creator`). The test file is in Plan 05-04's declared file scope; the fix pattern (same `make_test_io_backend()` approach used here in `test_metadata_gpu_scan_operators.cpp`) is documented in `deferred-items.md` for Plan 05-04 or Plan 05-06 to apply.

## Build

- `mcp__project-commands__run_command(build)` after Task 1 — exit 0 (10 targets updated, 9.0 s).
- `mcp__project-commands__run_command(build)` after Task 2 — exit 0 (13 targets updated, 9.0 s).
- `mcp__project-commands__run_command(build)` after Task 2b (iceberg ctor wiring) — exit 0 (15 targets updated, 11.4 s).

## Deviations from Plan

### Plan-authorized scope expansions (not true deviations)

**1. Header modifications (metadata scan + iceberg_scan_task.hpp)**
Plan 05-05's `<parallel_execution>` block listed only `.cpp` file scope. Two header edits were required:
- `src/include/op/scan/sirius_parquet_metadata_scan_operator.hpp` — add `io_backend` ctor parameter. The plan's Task 1 action step 3 explicitly said: "plumb the context (or a pre-resolved first-backend shared_ptr) down to the loop" — which implied the caller-facing interface must change. Not claimed by Plan 05-04.
- `src/include/op/scan/iceberg_scan_task.hpp` — add `gpu_io_backends` ctor parameter. Required to thread the map through to the base class. Not claimed by Plan 05-04.

Both headers are the direct siblings of the `.cpp` files I own; no cross-plan ownership conflict.

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Iceberg ctor did not forward gpu_io_backends to base (Plan 05-04 handoff gap)**
- **Found during:** Task 2 validation — `iceberg_scan_task_global_state::build_delete_pipeline()` calls `this->get_gpu_io_backends()` which returns the inherited map from `parquet_scan_task_global_state`. That map was empty because the iceberg ctor delegated to the base with the default `= {}`.
- **Scope:** `task_creator.cpp` is in Plan 05-04's declared file scope, but Plan 05-04 was complete (commit `86ebd57` dated 2026-04-21) and its SUMMARY stated: *"Iceberg path will runtime-fail until Plan 05-05 ships"* — explicitly handing the iceberg wiring to Plan 05-05.
- **Fix:** 3-file patch (commit `ce387f7`):
  1. `src/include/op/scan/iceberg_scan_task.hpp` — add `gpu_io_backends` ctor parameter.
  2. `src/op/scan/iceberg_scan_task.cpp` — forward `gpu_io_backends` through delegating ctor chain to base.
  3. `src/creator/task_creator.cpp` — iceberg branch resolves `sirius_ctx->get_gpu_io_backends()` and passes the map to `iceberg_scan_task_global_state` (mirrors parquet_scan_task branch pattern).
- **Verification:** Build exit 0; pure-consumer invariant on `sirius_context.hpp` upheld; all grep gates on `iceberg_scan_task.cpp` still pass.

### Deferred Items

- **`test_parquet_scan_task - single threaded small table` fails after Plan 05-04.** Root-caused to Plan 05-04's mandatory-backend throw + a test that bypasses `task_creator`. Documented in `.planning/phases/05-cucascade-backed-parquet-i-o-migration/deferred-items.md`. Out of scope for Plan 05-05 (Plan 05-04 file; Plan 05-04 or Plan 05-06 should apply the same `make_test_io_backend()` pattern used here for the metadata scan tests). Phase 5 sign-off requires this to be fixed before closing.

## Known Stubs

None. Both helpers now read real parquet data via `sirius::io::cucascade_datasource`. The metadata scan operator constructs real cucascade-backed adapters during `execute()`. No placeholder/TODO markers left in the diff.

## Next Phase Readiness

- **IO-05 closed globally:** combined Plan 05-04 (`parquet_scan_task.cpp` lines 312 + 699) + Plan 05-05 (`sirius_parquet_metadata_scan_operator.cpp:251`) = all 3 `datasource::create` call-sites migrated. `grep -rn 'datasource::create' src/` returns 0 for paths controlled by this phase.
- **IO-06 closed:** both iceberg delete-file call-sites use non-owning `source_info{&ds}` with stack-local adapters. Approach A locked.
- **Iceberg runtime path unblocked:** the handoff gap from Plan 05-04 is closed. Iceberg global state now receives `gpu_io_backends` from task_creator, forwards to base, and delete-file helpers resolve their backend via `get_gpu_io_backends()`.
- **Plan 05-06 (phase sign-off) unblocked:** HYG-02 sweep + multi-GPU compute-sanitizer validation + SF10 regression measurement are the remaining work. All IO-* grep gates (IO-05, IO-06, IO-08) are clean for the files Plan 05-05 touched.
- **Single outstanding test failure** documented in `deferred-items.md` — Plan 05-04 or Plan 05-06 applies the `make_test_io_backend()` pattern to `test_parquet_scan_task.cpp`.

## Self-Check: PASSED

Verified during summary creation:

- `src/op/scan/sirius_parquet_metadata_scan_operator.cpp` — FOUND (migration at line 251 region).
- `src/include/op/scan/sirius_parquet_metadata_scan_operator.hpp` — FOUND (io_backend ctor param + _io_backend member).
- `src/op/scan/iceberg_scan_task.cpp` — FOUND (both helpers + build_delete_pipeline backend resolution).
- `src/include/op/scan/iceberg_scan_task.hpp` — FOUND (gpu_io_backends ctor param).
- `src/creator/task_creator.cpp` — FOUND (iceberg branch get_gpu_io_backends seeding).
- `test/cpp/scan/test_metadata_gpu_scan_operators.cpp` — FOUND (make_test_io_backend helper + updated call sites).
- `.planning/phases/05-cucascade-backed-parquet-i-o-migration/deferred-items.md` — FOUND.
- Commit `3d74113` (Task 1) — FOUND in `git log`.
- Commit `1c15063` (Task 2) — FOUND in `git log`.
- Commit `ce387f7` (Task 2b iceberg ctor handoff) — FOUND in `git log`.
- Pure-consumer invariant: `git diff src/include/sirius_context.hpp` returns 0 lines — PASS.
- Grep gates on both modified source files — all PASS (see tables above).
- Call-site count invariant for iceberg helpers — 2 == 2 for both — PASS.
- Build exit 0 after all 3 commits — PASS.

---
*Phase: 05-cucascade-backed-parquet-i-o-migration*
*Plan: 05*
*Completed: 2026-04-21*
