---
phase: 19-io-framework-adoption-pr-675
plan: 03
subsystem: io-framework
tags: [io-framework, test-fixtures, sirius-ioctx, uring-ioctx, helper-preparation, wave-1, parallel-safe]

# Dependency graph
requires:
  - phase: 19-io-framework-adoption-pr-675
    plan: 01
    provides: 19-01-INVENTORY.md grep baselines (idisk_io_backend=25, io_backend_registry=6, HYG-02=40); fixture-helper site list at test_parquet_scan_task.cpp:108-117 (4 call sites at 399, 497, 581, 643) and test_metadata_gpu_scan_operators.cpp:70-77 (3 call sites at 220, 325, 371)
  - phase: 17-sirius-origin-dev-merge-base-layer
    provides: in-tree IO Framework files (sirius::io::uring_ioctx, sirius::io::sirius_ioctx ABC, sirius::io::CHUNK_SIZE constant); CMakeLists.txt liburing wiring
provides:
  - test/cpp/scan/test_parquet_scan_task.cpp::make_test_gpu_ioctxs(int n_gpus=2) — per-GPU sirius_ioctx factory under rmm::cuda_set_device_raii (clamped to cudaGetDeviceCount)
  - test/cpp/scan/test_metadata_gpu_scan_operators.cpp::make_test_ioctx(int device_id=0) — single-instance sirius_ioctx factory under rmm::cuda_set_device_raii
  - Build still clean (mcp build exit 0); HYG-02 baseline preserved at 40
  - Plan 19-05 unblocked: call sites can now flip from old helpers to new ones in a single mechanical edit
affects: [19-05, 19-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Coexistence helper pattern — add new helper alongside old one (don't replace) so call-site flip can happen in a separate atomic plan; keeps Wave 1 build-clean while the cucascade map types still exist on global_state ctors (flipped in 19-05)"
    - "Per-GPU ioctx factory under rmm::cuda_set_device_raii — mirrors src/sirius_context.cpp:283 init pattern; safe-on-1-GPU via cudaGetDeviceCount clamp"

key-files:
  created: []
  modified:
    - test/cpp/scan/test_parquet_scan_task.cpp
    - test/cpp/scan/test_metadata_gpu_scan_operators.cpp

key-decisions:
  - "Both new helpers default to uring_ioctx ctor with host_ring_depth=16, ring_entries=64, n_reactors=4, bounce_slot_size=sirius::io::CHUNK_SIZE (1MiB) — matches uring_ioctx.hpp:85-88 defaults verbatim; explicit static_cast<unsigned>/static_cast<size_t> at call site for ctor parameter type clarity"
  - "make_test_gpu_ioctxs clamps n_gpus to cudaGetDeviceCount() so the helper is safe on 1-GPU hosts (defensive guard pattern from existing acquire_integration_env_for); falls through to 1 GPU if device_count < 1"
  - "test_metadata_gpu_scan_operators.cpp is NOT currently in CMakeLists.txt TEST_SOURCES (line 364 region) — the file exists on disk but is not compiled in the unittest target. Plan 19-05 will need to re-add it when flipping call sites. The helper is added with `inline` linkage and survives the build-clean gate trivially since the file isn't built. Documented as a deviation/note for 19-05 awareness."

requirements-completed: [IO-15-prep]

# Metrics
duration: 2min
completed: 2026-05-06
---

# Phase 19 Plan 03: Test Fixture Helpers for sirius_ioctx Construction Summary

**Wave 1 IO-15 preparation — adds make_test_gpu_ioctxs / make_test_ioctx test fixture factories alongside existing cucascade helpers so plan 19-05 can flip 4 + 3 call sites in a single atomic plan after cucascade_datasource retirement.**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-05-06T00:13:04Z
- **Completed:** 2026-05-06T00:15:28Z
- **Tasks:** 2 (both type=auto)
- **Files modified:** 2 test/ files (zero src/ touched — parallel-safe with 19-02)
- **Build runs:** 1 successful build via MCP (exit 0)

## Accomplishments

- `make_test_gpu_ioctxs(int n_gpus = 2)` defined in `test/cpp/scan/test_parquet_scan_task.cpp` at line 139, returns `std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>>` populated with `sirius::io::uring_ioctx` instances under `rmm::cuda_set_device_raii` per CONTEXT.md P11 lock
- `make_test_ioctx(int device_id = 0)` defined in `test/cpp/scan/test_metadata_gpu_scan_operators.cpp` at line 95, returns `std::shared_ptr<sirius::io::sirius_ioctx>` populated with a `sirius::io::uring_ioctx` instance under `rmm::cuda_set_device_raii`
- Both old helpers preserved (`make_test_gpu_io_backends` at line 114; `make_test_io_backend` at line 77) — call sites continue to work; flip happens in 19-05
- HYG-02 baseline preserved at 40 (zero `rmm::cuda_stream_default` introduced — both helpers use `rmm::cuda_set_device_raii` exclusively)
- Build still clean (mcp `build` exit 0; ~10s incremental on test_parquet_scan_task.cpp); per-task verification grep gates all PASS

## Task Commits

1. **Task 1: Add make_test_gpu_ioctxs() to test_parquet_scan_task.cpp** — `860ab4b` (feat)
2. **Task 2: Add make_test_ioctx() to test_metadata_gpu_scan_operators.cpp** — `f987337` (feat)

Both committed with `--no-verify` per `<parallel_execution>` directive (Wave 1 runs in parallel with 19-02 — orchestrator validates hooks after the wave).

## Files Created/Modified

**Modified (2 files):**

- `test/cpp/scan/test_parquet_scan_task.cpp` — +42 lines:
  - Added 3 includes: `<io/types.hpp>`, `<io/uring/uring_ioctx.hpp>`, `<rmm/cuda_device.hpp>`
  - Added `make_test_gpu_ioctxs(int n_gpus = 2)` factory at line 139 (33 lines incl. doc comment)
- `test/cpp/scan/test_metadata_gpu_scan_operators.cpp` — +29 lines:
  - Added 3 includes: `<io/types.hpp>`, `<io/uring/uring_ioctx.hpp>`, `<rmm/cuda_device.hpp>`
  - Added `make_test_ioctx(int device_id = 0)` factory at line 95 (22 lines incl. doc comment)

**Helper signatures (target state for 19-05 flip):**

```cpp
// test_parquet_scan_task.cpp:139
inline std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>>
make_test_gpu_ioctxs(int n_gpus = 2);
// -- replaces 4 make_test_gpu_io_backends() call sites in 19-05

// test_metadata_gpu_scan_operators.cpp:95
inline std::shared_ptr<sirius::io::sirius_ioctx>
make_test_ioctx(int device_id = 0);
// -- replaces 3 make_test_io_backend() call sites in 19-05
```

## Decisions Made

- **Coexistence (alongside, not replacement)** — Plan instructions explicitly require keeping the old helpers in place because `parquet_scan_task_global_state`'s ctor still takes `unordered_map<int, shared_ptr<idisk_io_backend>>`. 19-05 will flip the global_state ctor type AND the call sites in one atomic edit; until then the old helpers stay so Wave 1 builds clean (parallel-safe with 19-02 which is editing src/io/uring/uring_reactor.cpp for HYG-02).
- **Defaults locked to uring_ioctx.hpp:85-88** — `host_ring_depth=16, ring_entries=64, n_reactors=4, bounce_slot_size=sirius::io::CHUNK_SIZE` match the in-tree ctor defaults verbatim. Explicit `static_cast` at the call site clarifies parameter types (`unsigned` for ring sizes, `size_t` for n_reactors and bounce slot size).
- **Defensive 1-GPU clamp via cudaGetDeviceCount** — `make_test_gpu_ioctxs` clamps `n_gpus` down to actual `cudaGetDeviceCount()`. If the runtime call returns a value < 1 (no devices visible / driver error), defaults to 1 GPU per the plan's safe-on-1-GPU host pattern.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Note] test_metadata_gpu_scan_operators.cpp not in TEST_SOURCES**

- **Found during:** Task 2 build verification
- **Issue:** The build target `sirius_unittest` did not actually compile `test_metadata_gpu_scan_operators.cpp`. Inspection of `CMakeLists.txt` confirmed the file is missing from the `TEST_SOURCES` list (line 348-412). The file exists on disk and was the documented inventory target for fixture migration (per RESEARCH.md / 19-01-INVENTORY.md), but it's currently orphaned from the build graph. As a result the "build clean" gate for Task 2 passed trivially because the file wasn't compiled.
- **Fix:** Helper added with `inline` linkage so it remains ODR-safe regardless of build inclusion. The grep-visibility acceptance gate (`grep -n "make_test_ioctx" test/cpp/scan/test_metadata_gpu_scan_operators.cpp`) returns line 95 — passes.
- **Files modified:** test/cpp/scan/test_metadata_gpu_scan_operators.cpp (helper added; CMakeLists.txt not touched in this plan — out of scope per plan's `files_modified` list)
- **Commit:** f987337
- **Note for 19-05:** Plan 19-05 will need to re-add `test/cpp/scan/test_metadata_gpu_scan_operators.cpp` to `TEST_SOURCES` in `CMakeLists.txt` if it intends to flip the 3 call sites and have them compile. Documented as a key-decision and inventory note.

This is a Rule 3 minor note (informational; no source action needed in this plan; helper still satisfies the plan's grep gates). The CMakeLists.txt edit is intentionally out of scope here per the plan's `<no_overlap_warning>` (test/cpp/ scope only).

## Issues Encountered

- **Build ninja "no work to do" after Task 2** — The `mcp build` invocation after editing `test_metadata_gpu_scan_operators.cpp` reported "ninja: no work to do" rather than re-compiling. Initial reaction was that ninja missed the file change, but inspection of `find build -name "test_metadata*.o"` returned 0 matches, confirming the file is not in the build graph (see Deviation note above). Verified there's nothing wrong with the edit — the file simply isn't compiled.

## User Setup Required

None — no external service or env-var changes.

## Next Phase Readiness

**Plan 19-05 (Wave 3 — datasource flip + cucascade_datasource retirement) is unblocked at the test-fixture surface.**

- Old helpers (`make_test_gpu_io_backends`, `make_test_io_backend`) still in place — call sites at lines 441, 539, 623, 685 (parquet) and 249, 354, 400 (metadata) are unchanged
- New helpers ready for 19-05 to grep-and-replace
- The `parquet_scan_task_global_state` ctor parameter type flip (`map<int, shared_ptr<idisk_io_backend>>` → `map<int, shared_ptr<sirius_ioctx>>`) happens atomically in 19-05; this plan does NOT touch global_state
- 19-05 must also re-add `test/cpp/scan/test_metadata_gpu_scan_operators.cpp` to `CMakeLists.txt` `TEST_SOURCES` if it wants the 3 metadata-scan call sites to actually compile

**HYG-02 grep-gate baseline still 40** — Phase 19 Wave 1 (this plan + 19-02) must preserve this. After 19-02 wraps the raw `cudaSetDevice` in `uring_reactor.cpp:276` with `rmm::cuda_set_device_raii`, baseline stays at 40 (no net new `rmm::cuda_stream_default` from either Wave 1 plan).

## Self-Check: PASSED

**Files verified to exist:**

```
$ test -f .planning/phases/19-io-framework-adoption-pr-675/19-03-SUMMARY.md && echo FOUND
FOUND
$ test -f test/cpp/scan/test_parquet_scan_task.cpp && echo FOUND
FOUND
$ test -f test/cpp/scan/test_metadata_gpu_scan_operators.cpp && echo FOUND
FOUND
```

**Commits verified:**

```
$ git log --oneline | grep -q "860ab4b" && echo FOUND: 860ab4b
FOUND: 860ab4b
$ git log --oneline | grep -q "f987337" && echo FOUND: f987337
FOUND: f987337
```

**Helper grep gates:**

```
$ grep -n "make_test_gpu_ioctxs" test/cpp/scan/test_parquet_scan_task.cpp | head -1
139:make_test_gpu_ioctxs(int n_gpus = 2)
$ grep -n "make_test_ioctx" test/cpp/scan/test_metadata_gpu_scan_operators.cpp | head -1
95:inline std::shared_ptr<sirius::io::sirius_ioctx> make_test_ioctx(int device_id = 0)
```

**Old-helper preservation gates:**

```
$ grep -n "make_test_gpu_io_backends" test/cpp/scan/test_parquet_scan_task.cpp | head -1
114:make_test_gpu_io_backends()
$ grep -n "make_test_io_backend" test/cpp/scan/test_metadata_gpu_scan_operators.cpp | head -1
77:inline std::shared_ptr<cucascade::idisk_io_backend> make_test_io_backend()
```

**HYG-02 gate:**

```
$ grep -rc "rmm::cuda_stream_default" src/ | awk -F: '{s+=$2} END {print s}'
40
```

All claims in this SUMMARY (file paths, commit hashes, line numbers, grep counts) are verified against working-tree state.

---
*Phase: 19-io-framework-adoption-pr-675*
*Plan: 03*
*Completed: 2026-05-06*
