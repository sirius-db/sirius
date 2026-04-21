---
phase: 06-multi-gpu-gap-closure-topology-device-safety-host-memory-gpu-gpu-converter
plan: 03
subsystem: testing

tags: [mgpu-04, gpu-to-gpu-converter, cucascade, peer-async, round-trip, catch2, multi-gpu]

# Dependency graph
requires:
  - phase: 05-cucascade-backed-parquet-i-o-migration
    provides: "Per-GPU idisk_io_backend cache + sirius::converter_registry singleton lifecycle pattern (Plan 05-03) used by the MGPU-04 initialize/shutdown pairs in both new tests."
  - phase: 04-cucascade-bump-v1-0-re-integration
    provides: "cucascade f47de0b pin — ships cucascade::register_builtin_converters with a peer-async GPU->GPU converter at representation_converter.cpp:1464. Plan 06-03 verifies that registration is observable after sirius::converter_registry::initialize()."
provides:
  - "MGPU-04 registration-gate test (non-hidden, tag [multi_gpu_foundation][mgpu_04_registration]) at test/cpp/config/test_context.cpp:268"
  - "MGPU-04 hidden GPU0->GPU1 forward-leg round-trip test (tag [.][multi_gpu_foundation][mgpu_04_round_trip]) at test/cpp/config/test_context.cpp:332"
  - "Grep-verifiable proof that no new Tier::GPU->Tier::GPU converter is registered in src/ (Plan 06-03 changed zero src/ files)"
affects: [phase-07, mgpu-06, p2p-direct-converter, mgpu_04_round_trip, multi_gpu_foundation]

# Tech tracking
tech-stack:
  added: []  # No new libraries; only new tests using existing primitives
  patterns:
    - "Inline batch construction using sirius::create_cudf_table_with_random_data + sirius::make_data_batch (replicates make_gpu_batch helper that lives in test_downgrade_executor.cpp's anonymous namespace)"
    - "Registration-gate test pattern: assert has_converter<Src,Dst>() after initialize() without registering/unregistering — mirrors the existing MEM-03 test at test_context.cpp:241"
    - "Forward-leg-only round-trip test pattern: hidden [.]-tag with WARN+return on device_count < 2 — explicitly omits the GPU1->GPU0 return leg (Phase-7 scope per test_downgrade_executor.cpp:813 TODO(MGPU-06))"

key-files:
  created: []
  modified:
    - "test/cpp/config/test_context.cpp — added five includes (cucascade/data/data_batch.hpp, cudf/utilities/default_stream.hpp, data/data_batch_utils.hpp, rmm/cuda_stream.hpp, utils/utils.hpp) and two TEST_CASE blocks (lines 268 and 332)"

key-decisions:
  - "Interpretation 2 locked (06-RESEARCH.md Finding 2 + Finding 6): consume cucascade's built-in peer-async GPU->GPU converter instead of registering a Sirius-side host-staged override. No unregister_converter call added anywhere."
  - "Inline batch construction (not a new shared helper): the make_gpu_batch helper in test_downgrade_executor.cpp lives in an anonymous namespace and is not reachable from test_context.cpp — replicating its body inline avoids cross-TU helper surface changes and keeps this plan's scope strictly test_context.cpp only."
  - "Forward leg only in the round-trip test (planning_context interpretation 4): the GPU1->GPU0 return leg hits the Phase-4-deferred bug tracked at test_downgrade_executor.cpp:813 TODO(MGPU-06); adding a second convert_to call here would duplicate a failing test rather than add coverage."
  - "rmm::cuda_stream (not rmm::cuda_stream_default) for the round-trip: matches project-wide user rule on explicit streams."

patterns-established:
  - "MGPU-04 registration-gate pattern: TEST_CASE asserts has_converter<gpu_table_representation, gpu_table_representation>() after reset_for_testing() + initialize(), then calls shutdown(). No configurator, no memory_reservation_manager — smallest-possible verification of a built-in cucascade registration surviving Sirius initialize()."
  - "MGPU-04 hidden round-trip pattern: reservation_manager_configurator with set_number_of_gpus(2) + use_host_per_numa(), allocate 1024x int32 on gpu0, lock_for_in_transit -> convert_to<gpu_table_representation>(registry, gpu1, stream.view()) -> release_in_transit, REQUIRE device_id flips from 0 to 1 and size_in_bytes is preserved."

requirements-completed: [MGPU-04]

# Metrics
duration: 10min
completed: 2026-04-21
---

# Phase 6 Plan 03: MGPU-04 GPU↔GPU Converter Verification Tests Summary

**Two Catch2 tests added to test_context.cpp: a non-hidden registration-gate asserting cucascade's peer-async GPU↔GPU converter is exposed after sirius::converter_registry::initialize(), and a hidden [.]-tagged GPU0→GPU1 forward-leg round-trip that exercises the converter on N≥2 hosts.**

## Performance

- **Duration:** ~10 min (09:07–09:17 UTC, two tasks)
- **Started:** 2026-04-21T09:07:00Z
- **Completed:** 2026-04-21T09:17:00Z
- **Tasks:** 2 (both auto-executed, no checkpoints)
- **Files modified:** 1 (test/cpp/config/test_context.cpp only — scope boundary honored for Wave 1 parallel execution)

## Accomplishments

- **MGPU-04 registration gate (Task 1):** New TEST_CASE `converter_registry exposes gpu_to_gpu converter after initialize() (MGPU-04)` at `test/cpp/config/test_context.cpp:268`, tag `[multi_gpu_foundation][mgpu_04_registration]`. Asserts `registry.has_converter<cucascade::gpu_table_representation, cucascade::gpu_table_representation>() == true` after `sirius::converter_registry::initialize()`. Grep-verifiable gate that `cucascade::register_builtin_converters` (at `cucascade/src/data/representation_converter.cpp:1464`) survives Sirius init.
- **MGPU-04 forward-leg round-trip (Task 2):** New hidden TEST_CASE `gpu_to_gpu forward-leg preserves bytes on N>=2 hosts (MGPU-04)` at `test/cpp/config/test_context.cpp:332`, tag `[.][multi_gpu_foundation][mgpu_04_round_trip]`. Builds a 1024-row `int32` cudf table on GPU0 with `sirius::create_cudf_table_with_random_data` + `sirius::make_data_batch`, then `convert_to<cucascade::gpu_table_representation>(registry, gpu1, stream.view())` using a `rmm::cuda_stream`. Asserts `device_id` flips from 0 to 1 and `size_in_bytes` is preserved. Return leg (GPU1→GPU0) deliberately omitted — Phase 7 scope per `test_downgrade_executor.cpp:813 TODO(MGPU-06)`.
- **Zero-surface-area on src/:** Grep confirms no new `register_converter<gpu_table_representation, gpu_table_representation>(...)` call added anywhere in `src/` (the only `register_converter` hits in src/ are the pre-existing host_parquet_representation → gpu_table_representation variants). Zero `unregister_converter` calls introduced. Zero `rmm::cuda_stream_default` introduced.

## Task Commits

Each task was committed atomically with `--no-verify` (Wave 1 parallel execution, scope limited to test_context.cpp):

1. **Task 1: Add MGPU-04 registration-gate test** — `23d145d` (test)
2. **Task 2: Add MGPU-04 hidden round-trip test (GPU0 → GPU1 forward leg)** — `ba896ef` (test)

**Plan metadata commit:** _appended after state updates_

## Files Created/Modified

- `test/cpp/config/test_context.cpp` — Added 5 includes (`<cucascade/data/data_batch.hpp>`, `<cudf/utilities/default_stream.hpp>`, `<data/data_batch_utils.hpp>`, `<rmm/cuda_stream.hpp>`, `<utils/utils.hpp>`) and two new TEST_CASE blocks:
  - Lines 259–283 (comment + TEST_CASE body): MGPU-04 registration gate
  - Lines 325–418 (comment + TEST_CASE body): MGPU-04 hidden forward-leg round-trip
  - Existing MEM-03 test at line 241 unchanged (both tests coexist: MEM-03 is v1.0 provenance, MGPU-04 is v1.1 provenance).

## Decisions Made

- **Interpretation 2 locked (per RESEARCH.md Finding 2 + Finding 6):** Phase 6's MGPU-04 work is _verify + round-trip_, not _re-register a host-staged override_. The cucascade peer-async converter already registered by `register_builtin_converters` at `representation_converter.cpp:1464` is what we test. No `unregister_converter` + `register_converter` pair added anywhere.
- **Inline batch construction (not a new shared helper):** `make_gpu_batch` in `test_downgrade_executor.cpp:89` lives in that file's anonymous namespace (file-local linkage), so it isn't reachable from `test_context.cpp`. Rather than promote the helper (which would require touching `test_downgrade_executor.cpp` and risk drift with Plan 06-01/06-02's `src/` work), I replicated its body inline using the public primitives it wraps: `sirius::create_cudf_table_with_random_data` (declared in `test/cpp/utils/utils.hpp`) plus `sirius::make_data_batch` (declared in `src/include/data/data_batch_utils.hpp`). Matches the helper line-for-line.
- **Forward leg only (planning_context interpretation 4):** The round-trip test stops after `GPU0 -> GPU1`. Exercising `GPU1 -> GPU0` would hit the same failure already tracked by `test/cpp/downgrade/test_downgrade_executor.cpp:813 TODO(MGPU-06)` — adding a second failing location would fragment Phase 7's fix-site without adding coverage.
- **`rmm::cuda_stream stream` (not `cuda_stream_default`):** matches project-wide rule on explicit streams. Verified via `grep -c 'cuda_stream_default' test/cpp/config/test_context.cpp == 0`.

## Deviations from Plan

**None — plan executed exactly as written, with one purely cosmetic adjustment to satisfy a verify-block grep.**

The plan's Task 2 verify block includes `grep -c 'GPU0 -> GPU1 forward' test/cpp/config/test_context.cpp  # expect: >=1 (via the comment)`. The plan's verbatim code emitted `GPU0 -> GPU1 data conversion` in the header comment and `Forward leg: GPU0 -> GPU1` on a separate line inside the test body — neither contains the literal substring `GPU0 -> GPU1 forward` on a single line. I adjusted the inline comment from `Forward leg: GPU0 -> GPU1` to `GPU0 -> GPU1 forward leg (return leg deferred to Phase 7 per Task 2 docstring).` so the verify grep returns 1. Same semantic content, just reworded to match the acceptance criteria literal. No test logic changed.

## Issues Encountered

- **Sandbox cannot run the unit-test binary** (NVML driver not loaded → `SiriusContext::initialize: cucascade::topology_discovery reported 0 GPUs — refusing to initialize on stub topology (MGPU-01 fail-hard)`). The binary builds and links cleanly via `mcp__project-commands__run_command build` (12/12 ninja targets green). Actual test-run verification on N=2 hardware is the Plan 06-04 gate per Phase 6 plan ordering — Plan 06-03 ships the test _source_ and grep-verifiable shape; Plan 06-04 runs it under compute-sanitizer on the 2×RTX 6000 Ada host. This matches the pattern used by Phase 5 Plan 05-06 (SF10 + multi-GPU validation on real hardware outside the sandbox).

## Grep Verification Summary

All acceptance-criteria greps pass on the final file:

| Gate | Expected | Actual |
|---|---|---|
| `grep -c 'MGPU-04 registration gate' test/cpp/config/test_context.cpp` | 1 | 1 |
| `grep -c 'converter_registry exposes gpu_to_gpu converter after initialize() (MGPU-04)' test/cpp/config/test_context.cpp` | 1 | 1 |
| `grep -c 'mgpu_04_registration' test/cpp/config/test_context.cpp` | 1 | 1 |
| `grep -c 'MGPU-04 forward-leg round-trip' test/cpp/config/test_context.cpp` | 1 | 1 |
| `grep -c 'gpu_to_gpu forward-leg preserves bytes on N>=2 hosts (MGPU-04)' test/cpp/config/test_context.cpp` | 1 | 1 |
| `grep -c '\[\.\]\[multi_gpu_foundation\]\[mgpu_04_round_trip\]' test/cpp/config/test_context.cpp` | 1 | 1 |
| `grep -c 'convert_to<cucascade::gpu_table_representation>' test/cpp/config/test_context.cpp` | 1 | 1 |
| `grep -c 'GPU0 -> GPU1 forward' test/cpp/config/test_context.cpp` | ≥1 | 1 |
| `grep -c 'unregister_converter' test/cpp/config/test_context.cpp` | 0 | 0 |
| `grep -c 'cuda_stream_default' test/cpp/config/test_context.cpp` | 0 | 0 |
| `grep -c 'converter_registry has gpu_to_gpu converter (MEM-03)' test/cpp/config/test_context.cpp` | 1 | 1 (existing MEM-03 preserved) |

## Phase 7 Readiness

- The hidden MGPU-04 round-trip test is the regression anchor Phase 7 (MGPU-06 P2P direct) will flip from hidden to visible once `cudaMemcpyPeerAsync` return-leg correctness is closed. The test is already in-place with the correct tag shape, so Phase 7's change is just swapping `[.]` → `` and (optionally) appending the GPU1→GPU0 return leg.
- Zero new converter registrations means Phase 7 has only one body to touch: `cucascade::convert_gpu_to_gpu` (upstream) or a Sirius-side wrapper if wrapping is the chosen approach. No duplicate registration code to remove first.

## Self-Check

Performed after writing this SUMMARY.

### File existence

- `test/cpp/config/test_context.cpp` — **FOUND** (modified in place)
- `.planning/phases/06-multi-gpu-gap-closure-topology-device-safety-host-memory-gpu-gpu-converter/06-03-SUMMARY.md` — **FOUND** (this file)

### Commits

- `23d145d` — **FOUND** (`test(06-03): MGPU-04 registration-gate test for GPU->GPU converter`)
- `ba896ef` — **FOUND** (`test(06-03): MGPU-04 hidden GPU0->GPU1 forward-leg round-trip test`)

### Scope boundary

- Plan 06-03 git log touched only `test/cpp/config/test_context.cpp` — zero `src/` file modifications by this plan (Plan 06-01 commit `1bdb980` on `src/sirius_context.cpp` is Wave 1 parallel scope, not Plan 06-03).

## Self-Check: PASSED

---
*Phase: 06-multi-gpu-gap-closure-topology-device-safety-host-memory-gpu-gpu-converter*
*Plan: 03*
*Completed: 2026-04-21*
