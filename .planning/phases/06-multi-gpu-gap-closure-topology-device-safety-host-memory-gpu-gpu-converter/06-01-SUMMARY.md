---
phase: 06-multi-gpu-gap-closure-topology-device-safety-host-memory-gpu-gpu-converter
plan: 01
subsystem: context-initialization
tags: [topology, mgpu-01, mgpu-05, numa, fail-hard, spdlog, grep-gate]

# Dependency graph
requires:
  - phase: 05-cucascade-backed-parquet-i-o-migration
    provides: "SiriusContext::initialize() cucascade io_backend registry + per-GPU idisk_io_backend cache at src/sirius_context.cpp lines 185-204 — Plan 06-01 inserts MGPU-01 block before this and MGPU-05 block between memory_manager_ ctor and the io_backend cache"
  - phase: 04-cucascade-bump-v1-0-re-integration
    provides: "sirius_config::sirius_config() topology_discovery call at src/sirius_config.cpp:267 + cached _hw_topology + SiriusContext::get_hw_topology() accessor at src/include/sirius_context.hpp:117 — Plan 06-01 is a pure consumer, no re-discovery call added"
provides:
  - "Topology fail-hard at SiriusContext::initialize() entry — throws std::runtime_error when cached _hw_topology.num_gpus == 0 (stub-topology guard)"
  - "Info-level startup log: 'SiriusContext: topology summary — N GPU(s), M NUMA node(s), host=...' plus one line per GPU (id, name, numa, pci)"
  - "Post-memory_manager host-space log: 'SiriusContext: X host memory space(s) created for M NUMA node(s)' — warn (not throw) when X != M and M > 0"
  - "MGPU-01 sweep gate: Super Sirius (src/ \\ src/cuda/) is clean of raw cudaGetDeviceCount / numa_node_of_cpu / numa_available callsites; single documented legacy hit preserved at src/cuda/allocator.cu:70"
  - "MGPU-05 provenance comment above existing builder.use_host_per_numa() in sirius_config.cpp linking configurator intent to the initialize()-side assertion"
affects:
  - "06-02 — device-guard enforcement plan: consumes the same get_hw_topology() accessor; no new topology calls expected"
  - "06-03 — context/configurator unit tests: can assert on the log lines and MGPU-01 throw behaviour"
  - "06-04 — phase validation: /proc/PID/numa_maps evidence block references the MGPU-05 log line for N=2 host capture"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Topology cache + validated accessor (RESEARCH.md Pattern 1): validate cached topology once at initialize() entry — never re-discover"
    - "Narrative-vs-gate separation: audit-gate greps must not self-trip from narrative comments that contain the forbidden token names (rephrased MGPU-01 block comment to avoid literal tokens)"
    - "Per-NUMA host-space assertion (RESEARCH.md Pattern 3): warn-not-throw on count mismatch so non-NUMA CI hosts with num_numa_nodes == 0 still load"

key-files:
  created: []
  modified:
    - "src/sirius_context.cpp (initialize() body +49 / −0 net of rephrase: MGPU-01 block at lines 174-198, MGPU-05 block at lines 201-225)"
    - "src/sirius_config.cpp (host_mem_config::setup_configurator +6 comment lines at 216-220 above existing builder.use_host_per_numa() at line 221)"

key-decisions:
  - "Reuse existing config_.get_hw_topology() accessor — NO new SiriusContext::get_topology() alias, NO second topology_discovery.discover() call (RESEARCH.md Anti-Patterns §1)"
  - "Warn-not-throw on num_numa_nodes != host_spaces.size() (only when num_numa_nodes > 0) — non-NUMA single-socket dev hosts legitimately report num_numa_nodes == 0"
  - "Rephrased MGPU-01 block comment to use 'raw CUDA/NUMA device-enumeration APIs' instead of the literal forbidden tokens — narrative would otherwise self-trip the sweep gate (Rule 1 bug caught during Task 2 verification)"
  - "MGPU-05 comment is provenance-only in sirius_config.cpp — no behaviour change; existing use_host_per_numa() call at its existing line preserved"

patterns-established:
  - "Fail-hard initialization gate: validate cached topology at initialize() entry BEFORE any memory_manager_ / io_backend / downgrade_executor construction — downstream failures no longer mask a stub topology"
  - "Audit-log twin: every MGPU-* requirement gets a dedicated spdlog::info line that the phase validation plan consumes as evidence (mirrors IO-11 pattern from Plan 05-03)"

requirements-completed: [MGPU-01, MGPU-05]

# Metrics
duration: 6min
completed: 2026-04-21
---

# Phase 06 Plan 01: Topology Fail-Hard + Per-NUMA Host-Space Assertion Summary

**SiriusContext::initialize() now throws on zero-GPU topology, emits info-level startup log summarising the cached cucascade topology, and logs host-space count vs NUMA node count — closing MGPU-01 and MGPU-05 without touching cucascade or re-discovering NVML.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-04-21T14:07:16Z
- **Completed:** 2026-04-21T14:13:20Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Topology fail-hard wired at `SiriusContext::initialize()` entry — `std::runtime_error` is thrown if cached `_hw_topology.num_gpus == 0`, before any memory_manager / io_backend construction.
- Startup info log now lists total GPU count, NUMA count, host name, and one line per GPU with `id/name/numa_node/pci_bus_id` — provides the auditable evidence MGPU-01 requires.
- Post-memory_manager log reports `host_spaces.size()` against `topo.num_numa_nodes`, warning (not throwing) on mismatch so non-NUMA CI hosts still initialize cleanly.
- Provenance comment in `sirius_config.cpp` above the existing `builder.use_host_per_numa();` explicitly links the configurator call to the `SiriusContext::initialize()` assertion, preventing future refactors from silently breaking MGPU-05.
- MGPU-01 hand-rolled CUDA/NUMA sweep gate now clean across Super Sirius (`src/` excluding `src/cuda/`): 0 hits. Single legacy hit at `src/cuda/allocator.cu:70` preserved and documented.

## Task Commits

Each task was committed atomically with `--no-verify` (Wave 1 parallel execution with plans 06-02 / 06-03):

1. **Task 1: Add topology fail-hard + startup log in SiriusContext::initialize()** — `1bdb980` (feat)
2. **Task 2: Verify sirius_config use_host_per_numa() path + MGPU-01 sweep gate** — `097b8c0` (chore)

## Files Created/Modified

- `src/sirius_context.cpp` — Added two blocks inside `SiriusContext::initialize()`:
  - Lines 174-198: MGPU-01 block — topology validation + fail-hard throw + info-level topology summary log + per-GPU info lines.
  - Lines 201-225: MGPU-05 block — post-memory_manager host-space count log + warn-on-mismatch branch.
- `src/sirius_config.cpp` — Added provenance comment (lines 216-220) above the existing `builder.use_host_per_numa();` in `host_mem_config::setup_configurator`, linking the configurator intent to the initialize()-side assertion. No behaviour change.

## Verification Results

### Acceptance greps — Task 1 (src/sirius_context.cpp)

```
$ grep -c 'MGPU-01: Topology fail-hard' src/sirius_context.cpp
1
$ grep -c 'MGPU-05: Per-NUMA host memory space assertion' src/sirius_context.cpp
1
$ grep -c 'num_gpus == 0' src/sirius_context.cpp
1
$ grep -c 'SiriusContext: topology summary' src/sirius_context.cpp
1
$ grep -c 'SiriusContext: .* host memory space' src/sirius_context.cpp
1
$ grep -c 'cuda_stream_default' src/sirius_context.cpp
0
$ grep -c 'topology_discovery.discover' src/sirius_context.cpp
0
$ grep -c 'SiriusContext::get_topology' src/sirius_context.cpp
0
```

### Acceptance greps — Task 2 (src/sirius_config.cpp)

```
$ grep -c 'MGPU-05: cucascade builds one' src/sirius_config.cpp
1
$ grep -c 'builder.use_host_per_numa();' src/sirius_config.cpp
1
```

### MGPU-01 sweep gates

```
$ grep -rn 'cudaGetDeviceCount\|numa_node_of_cpu\|numa_available' src/ \
    --include='*.cpp' --include='*.hpp' --include='*.cu' --include='*.cuh' \
    | grep -v '^src/cuda/' | wc -l
0

$ grep -rn 'cudaGetDeviceCount\|numa_node_of_cpu\|numa_available' src/cuda/
src/cuda/allocator.cu:70:  err = cudaGetDeviceCount(&nDevices);

$ grep -rn 'numa_node_of_cpu' src/
(empty)

$ grep -c 'cudaGetDeviceCount' src/cuda/allocator.cu
1
$ grep -c 'numa_node_of_cpu' src/cuda/allocator.cu
0
```

All MGPU-01 gates pass. The single `src/cuda/allocator.cu:70` hit is the documented legacy-path exclusion per Phase 6 CONTEXT §Deferred — `namespace duckdb` legacy code frozen for v1.1 per PROJECT.md Out-of-Scope.

### Build

```
mcp__project-commands__run_command build
Exit code: 0  (9.9s first pass, 2.1s incremental re-build after sirius_config.cpp touch)
```

### Unit tests

```
mcp__project-commands__run_command unit-tests
Exit code: 0
All tests passed (78,789,792 assertions in 973 test cases)
```

No behavioural regression on the current host — GPU count > 0 path was taken, so fail-hard branch was not exercised. Fail-hard path is testable by construction (throws if `num_gpus == 0` on the cached topology); 06-03 owns the unit-test coverage per file scope.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] MGPU-01 block comment self-tripped the sweep gate**

- **Found during:** Task 2 (MGPU-01 sweep gate verification after Task 2 comment insert)
- **Issue:** The Task 1 MGPU-01 block comment in `src/sirius_context.cpp` contained the literal strings `cudaGetDeviceCount` and `numa_node_of_cpu` as descriptive prose ("Super Sirius files must not call cudaGetDeviceCount / numa_node_of_cpu directly…"). The MGPU-01 sweep gate is a text-based grep that does not distinguish narrative from code, so Gate 1 reported 2 hits (lines 181 + 182) and Gate 3 reported 1 hit on the self-referential prose.
- **Fix:** Rephrased the comment to use "the raw CUDA/NUMA device-enumeration APIs" instead of the literal forbidden tokens, preserving the audit narrative while letting the grep gate remain text-simple (per RESEARCH.md Anti-Patterns — avoid complicating the gate with `grep -v 'comment'` exclusions).
- **Files modified:** `src/sirius_context.cpp` (MGPU-01 block comment lines 181-183, same commit as Task 2).
- **Commit:** `097b8c0` (bundled with the Task 2 MGPU-05 provenance comment and the sweep-gate hygiene change).
- **Scope check:** Fix stayed within the plan's allowed file scope (`src/sirius_context.cpp` + `src/sirius_config.cpp`). No expansion into 06-02 or 06-03 territory.

### Auth Gates

None.

## Known Stubs

None. Both code paths are fully wired:
- MGPU-01 throw path uses the already-cached `_hw_topology` (populated by `sirius_config::sirius_config()`); no placeholder.
- MGPU-05 assertion consumes `memory_manager_->get_memory_spaces_for_tier(Tier::HOST)`, which is real (built by the configurator via `builder.use_host_per_numa()` already wired at Plan 04's YAML config work).

## Deferred Issues

None from Plan 06-01. Items deferred to other Phase 6 plans per the wave plan:
- Device-guard enforcement on raw `cudaSetDevice` callsites → Plan 06-02 (gpu_pipeline_executor.cpp:58, downgrade_executor.cpp:61).
- Phase-6 validation artifact capture (`/proc/PID/numa_maps` evidence on N=2 host) → Plan 06-04 (or the phase SUMMARY, per orchestrator choice).
- Unit-test coverage of the MGPU-01 throw path + MGPU-05 warn branch → Plan 06-03 (test/cpp/config/test_context.cpp).

## Self-Check: PASSED

- ✓ `src/sirius_context.cpp` exists (modified): FOUND — 49 lines added.
- ✓ `src/sirius_config.cpp` exists (modified): FOUND — 6 lines added (comment-only).
- ✓ Commit `1bdb980` exists: `git log --oneline --all | grep 1bdb980` → FOUND.
- ✓ Commit `097b8c0` exists: `git log --oneline --all | grep 097b8c0` → FOUND.
- ✓ MGPU-01 sweep gate (Super Sirius): 0 hits.
- ✓ MGPU-05 provenance comment present in `src/sirius_config.cpp`.
- ✓ File scope respected: only `src/sirius_context.cpp` + `src/sirius_config.cpp` touched. `src/pipeline/gpu_pipeline_executor.cpp` + `src/downgrade/downgrade_executor.cpp` (Plan 06-02) untouched. `test/cpp/config/test_context.cpp` (Plan 06-03) untouched.
- ✓ Both commits created with `--no-verify` per Wave 1 parallel-execution directive.
- ✓ Build green, unit-tests 973/973 green.
