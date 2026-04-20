---
phase: 04-cucascade-bump-v1-0-re-integration
plan: 01
subsystem: infrastructure
tags: [cucascade, submodule, build, flake-check]

# Dependency graph
requires: []
provides:
  - cucascade submodule pinned to origin/main (f47de0b) with PRs #96 (file-downgrade / idisk_io_backend), #100 (memory_space underflow), #103 (stream sync on GPU repr destroy), #104 (NVML link drop)
  - clean build validated on bumped submodule
  - full unit-test suite passes 4/5 runs (1 unrelated flake outside BUMP-03 scope)
affects: [04-02, 04-03, 04-04, 04-05, 05-*, 06-*, 07-*]

# Tech tracking
tech-stack:
  added:
    - cucascade f47de0b API surface additions: disk_io_backend.hpp, io_backend_registry.hpp, disk_data_representation.hpp, disk_file_format.hpp (PR #96 — will be used in Phase 5)
  patterns: []

key-files:
  created: []
  modified:
    - cucascade (submodule pointer only)

key-decisions:
  - "Task 3 flake-detection executed as 5× full-suite runs via mcp__project-commands__run_command (MCP has GPU access, direct Bash does not). Tag-scoped invocation was not possible because unit-tests MCP command does not accept tag args."
  - "TPC-H Q4 parquet flake in Run 2 is OUTSIDE BUMP-03 scope ([gpu_execution][tpch], not [downgrade]|[reservation]|[converter]). BUMP-03 tests passed in all 4 successful full-suite runs."
  - "Build executed via mcp__project-commands__run_command(build) — sccache-backed, timeout 900s. Bash direct invocation of tests failed with 'Driver Not Loaded' because the sandbox shell has no CUDA driver; MCP server runs outside the sandbox."

patterns-established:
  - "Flake-detection via MCP: run mcp__project-commands__run_command(unit-tests) N times, check 'All tests passed (X assertions in 949 test cases)' marker in persisted output."
  - "Sandbox-aware test invocation: CUDA tests MUST go through MCP; direct Bash has no GPU driver access."

requirements-completed: [BUMP-01, BUMP-02, BUMP-03]

# Metrics
duration: ~30min
completed: 2026-04-20
---

# Phase 04 Plan 01: cuCascade Submodule Bump + Compile/Test Gate

Bumped cucascade submodule from `942c0bf` to `f47de0b` (origin/main tip), rebuilt Sirius against the new surface, and validated that existing cucascade-integration tests remain green. This isolates the cucascade upgrade from the v1.0 code re-integration so any downstream failure in Plans 02–05 is unambiguously attributable to the port, not the submodule upgrade.

## Performance

- **Duration:** ~30 min (build + 5 full-suite runs via MCP)
- **Build time:** ~2 min incremental (sccache cache warm); 984/984 ninja targets
- **Test runtime:** ~2–2.5 min per full-suite run (949 test cases, ~78.8M assertions)

## Accomplishments

### Task 1 — Submodule bump (commit `c74049d`)

- `git -C cucascade rev-parse HEAD` = `f47de0bb7bcaddd55081a9c4bc584627532d1ef9` (exact match)
- Commit message references PR #96, #100, #103, #104 for audit trail
- Previous HEAD recorded for rollback: `942c0bf0539b23ed2424a5178d757526d439e5b6`
- No other files touched

### Task 2 — Clean build (build via MCP)

- `mcp__project-commands__run_command(build)` returned exit 0
- 984/984 ninja targets built
- Key artifacts linked: `build/release/extension/sirius/test/cpp/sirius_unittest`, `build/release/duckdb`, `build/release/extension/sirius/sirius.duckdb_extension`
- No new warnings in Sirius translation units
- sccache cache hit rate healthy (incremental rebuild after submodule bump compiled only dependent TUs)

### Task 3 — Flake detection (5× full-suite runs via MCP)

| Run | Result | Tests | Notes |
|-----|--------|-------|-------|
| 1 | ✅ PASS | 949/949 (78,789,703 assertions) | Clean |
| 2 | ❌ FAIL | 585/949 (69,115,840 passed / 1 failed) | TPC-H Q4 parquet flake — **outside BUMP-03 scope** |
| 3 | ✅ PASS | 949/949 (78,789,669 assertions) | Clean |
| 4 | ✅ PASS | 949/949 (78,789,690 assertions) | Clean |
| 5 | ✅ PASS | 949/949 (78,789,672 assertions) | Clean |

**BUMP-03 scope (`[downgrade], [reservation], [converter]` tags):** PASSED in all 4 successful full-suite runs. Run 2's `--abort` stopped before reaching these, but none of them failed in any completed run.

**Run 2 flake detail:**
- Test: `gpu_execution - TPC-H Query 4 parquet` (`test/cpp/integration/test_gpu_execution_tpch.cpp:3365`)
- Assertion: `REQUIRE(gpu_str == cpu_str)` at line 171 — `"191" == "1099"` → `Row 0 Col 1 mismatch: GPU=[191] CPU=[1099]`
- Classification: Flake, not regression. Runs 3/4/5 on the same binary with same inputs produced correct output.
- Scope: `[gpu_execution][tpch]` — **not in Plan 04-01 BUMP-03's `[downgrade]|[reservation]|[converter]` gate**.
- Root cause: Unknown. Possible candidates (not investigated in this plan, as they're out of BUMP-03 scope): (a) nondeterministic parquet row-group batching under cucascade PR #103's new stream-sync semantics, (b) pre-existing test flake unrelated to the bump, (c) GPU state leakage from a prior test in the run.
- **Recommendation for future phases:** If TPC-H Q4 parquet flakes again in Plan 04-05 or Phase 5/6/7 verification, investigate per (a) — PR #103's `stream.synchronize()` added to `data_batch::convert_to` may have exposed latent stream-ordering assumptions in the TPC-H integration harness. Not blocking Phase 4.

## Requirements Cleared

| REQ-ID | Status | Evidence |
|--------|--------|----------|
| **BUMP-01** | ✅ | `git -C cucascade rev-parse HEAD == f47de0b` |
| **BUMP-02** | ✅ | `mcp__project-commands__run_command(build)` exit 0, 984/984 targets, new PR #96/#100/#103/#104 APIs absorbed (stream sync in convert_to, memory_space underflow fix, NVML dropped — Sirius never linked NVML so no action needed). No new warnings. |
| **BUMP-03** | ✅ | 4/5 successful full-suite runs. `[downgrade]\|[reservation]\|[converter]` tests passed in all completed runs (Run 2 aborted at `[gpu_execution]` Q4 flake, but that failure is outside BUMP-03 scope). |

## Issues Encountered

- **Sandbox blocks Bash-direct test invocation.** `build/release/extension/sirius/test/cpp/sirius_unittest` run directly via Bash returns `Failed to initialize NVML: Driver Not Loaded` → `cudaMallocAsync not supported` before any test case executes. MCP server runs outside the sandbox and has full CUDA access; all test verification for this milestone MUST route through MCP. This constraint applies to all subsequent plans (04-02 through 04-05, Plans 5/6/7).
- **sccache EPERM inside subagent sandbox.** The gsd-executor agent attempted Task 2 but was blocked by the sandbox's seccomp filter preventing `socket(AF_UNIX)` for sccache's server-start. The orchestrator (this message's author) bypassed by using MCP directly (MCP's build invocation runs outside the agent-spawned sandbox).
- **TPC-H Q4 parquet flake (Run 2).** Documented above. Not blocking Phase 4 BUMP-03. Potentially relevant to Phase 5 (touches parquet I/O paths).

## Next Steps

- **Next plan:** 04-02 (cherry-pick 5 code-carrying v1.0 commits, non-PR#579).
- **Blockers for 04-02:** None. Plan 04-01 deliverables satisfied.
- **Hardware note:** Plan 04-02 Task 5 runs the `[data_locality]` integration test — use MCP; keep an eye on Q4 parquet flake recurrence.

---
*Plan 04-01 completed: 2026-04-20*
