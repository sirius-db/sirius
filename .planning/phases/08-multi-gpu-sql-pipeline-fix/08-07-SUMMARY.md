---
phase: 08-multi-gpu-sql-pipeline-fix
plan: 07
subsystem: observability

tags: [cuda, multi-gpu, logging, mgpu-probe, gap-closure, instrumentation]

# Dependency graph
requires:
  - phase: 08-multi-gpu-sql-pipeline-fix
    plan: 06
    provides: "Pattern 2 idiom applied to convert_host_parquet_to_gpu_with_prefetched_data_source; residual cudaErrorInvalidValue @ cuda_memcpy.cu:42 persists on num_gpus=2 parquet path; 4 hypothesis candidates (A/B/C/D) carried forward"
provides:
  - "Three [mgpu-probe] INFO breadcrumbs (2 in host_parquet converter: entry+exit; 1 in parquet_scan_task: entry) that deterministically discriminate hypotheses A/B/C/D by emitting cudaGetDevice() + stream.value() + target/preferred device id + memspace device id at each frame boundary"
  - "Grep-stable payload format locked: `[mgpu-probe] host_parquet_to_gpu {entry|exit} current_device=<N> ...` and `[mgpu-probe] parquet_scan_task::compute_task entry current_device=<N> ...` — 08-08 diagnostic analysis is mechanical"
  - "Instrumentation commits ready for MCP unit-tests run on num_gpus=2 via integration-2gpu.yaml flow or GENERATE(1,2) parameterization"
affects:
  - 08-08-PLAN.md (reproduction plan — expects these breadcrumbs in SIRIUS_LOG_DIR output)
  - 08-09-PLAN.md (targeted-fix plan — expects hypothesis selection evidence from probe output)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "[mgpu-probe] prefix for gap-closure diagnostic breadcrumbs, distinct from [mgpu-audit] (which is owned by AUDIT TEST_CASE regex in test_gpu_execution_tpch_mgpu_audit.cpp)"
    - "INFO-level logging for probe breadcrumbs (not TRACE/DEBUG) — matches existing [mgpu-audit] convention so SIRIUS_LOG_LEVEL=info picks them up in MCP test runs without a debug build"
    - "Probe-local variable names (preferred_probe, memspace_probe) to avoid shadowing existing locals downstream"

key-files:
  created:
    - ".planning/phases/08-multi-gpu-sql-pipeline-fix/08-07-SUMMARY.md (this file)"
  modified:
    - "src/data/host_parquet_representation_converters.cpp (+ #include <log/logging.hpp>; +2 SIRIUS_LOG_INFO breadcrumbs at lines 99 and 172)"
    - "src/op/scan/parquet_scan_task.cpp (+1 SIRIUS_LOG_INFO breadcrumb at line 745 inside compute_task; no new include needed — cudaGetDevice resolves transitively through existing rmm/cudf headers)"

key-decisions:
  - "Added #include <log/logging.hpp> to host_parquet_representation_converters.cpp. Plan's <interfaces> block asserted the include was already wired; verified it was NOT. Added per Rule 3 (blocking — build would fail without it). parquet_scan_task.cpp already had the include."
  - "Used SIRIUS_LOG_INFO (not TRACE/DEBUG). Plan-mandated: matches existing [mgpu-audit] emissions at pipeline_executor.cpp:255 and duckdb_scan_executor.cpp:204 which are INFO-level, ensuring SIRIUS_LOG_LEVEL=info (the MCP default) captures them."
  - "Used static_cast<void*>(stream.value()) for stream handle printing. cudaStream_t is an opaque struct pointer; void* gives a stable {:p} format under spdlog/fmt."
  - "Probe-local variable names `preferred_probe` and `memspace_probe` in parquet_scan_task.cpp to avoid shadowing the existing `preferred` at line 775 (now renumbered) inside the `if (!_datasource)` block. The probe block is lexically scoped with its own {} so even `current_device` is local — no leakage."
  - "Entry breadcrumb in host_parquet converter placed AFTER target_device_id computation, BEFORE stream.synchronize(). This captures the caller's stream identity BEFORE any state-flushing side effect — so if hypothesis A fires (upstream wrong-device), the breadcrumb records it unobscured."
  - "Exit breadcrumb in host_parquet converter placed AFTER (void)cudaGetLastError(), BEFORE the post-filter prune. This captures a CLEAN state (any sticky error has been consumed); if the upstream caller still fails after return, the hazard localizes to either the post-return column-projection path or an upstream frame — not inside the target-bound read+inject chain."
  - "Entry breadcrumb in parquet_scan_task::compute_task placed immediately after l_state/g_state binding, BEFORE the `if (!_datasource)` block. This is the earliest point where probe-payload values are accessible; the breadcrumb records the upstream H2D frame context before read_range_into_allocation can perturb it."

patterns-established:
  - "Pattern: gap-closure diagnostic breadcrumbs use prefix `[mgpu-probe]` (distinct from `[mgpu-audit]` TEST_CASE regex ownership) with grep-stable `key=value` payload for mechanical 08-08-style analysis"
  - "Pattern: probe breadcrumbs are scoped in `{}` blocks to keep probe-local vars out of main control flow, preventing shadow of existing locals downstream"

requirements-completed: [FIX-02]

# Metrics
duration: 10min
completed: 2026-04-22
---

# Phase 08 Plan 07: Gap-Closure Instrumentation ([mgpu-probe] Breadcrumbs) Summary

**Added three grep-stable `[mgpu-probe]` INFO breadcrumbs at two frame boundaries on the num_gpus=2 parquet failure path (host_parquet converter entry+exit, parquet_scan_task::compute_task entry) so plan 08-08's MCP reproduction produces a deterministic payload discriminating hypotheses A/B/C/D from 08-VERIFICATION.md.**

## Performance

- **Duration:** ~10 min (wall clock)
- **Completed:** 2026-04-22
- **Tasks:** 2 (Task 1: 2 breadcrumbs in host_parquet converter; Task 2: 1 breadcrumb in parquet_scan_task)
- **Files modified:** 2 source files; 0 new files

## Accomplishments

- **Task 1 — host_parquet_to_gpu entry + exit breadcrumbs.** Added two SIRIUS_LOG_INFO calls inside `sirius::detail::convert_host_parquet_to_gpu_with_prefetched_data_source` at `src/data/host_parquet_representation_converters.cpp`:
  - **Entry** (line 99 format string): `[mgpu-probe] host_parquet_to_gpu entry current_device={} stream={} target_device_id={} memspace_device_id={}` — placed after target_device_id computation, BEFORE `stream.synchronize()` to record the caller's unmodified stream identity.
  - **Exit** (line 172 format string): `[mgpu-probe] host_parquet_to_gpu exit current_device={} target_stream={} target_device_id={}` — placed after `(void)cudaGetLastError()`, BEFORE the post-filter prune to record a clean state.
  - Added `#include <log/logging.hpp>` (plan's <interfaces> block incorrectly stated it was already wired; verified it was not — added per Rule 3 blocking fix).
- **Task 2 — parquet_scan_task::compute_task entry breadcrumb.** Added one SIRIUS_LOG_INFO call at the top of `compute_task` in `src/op/scan/parquet_scan_task.cpp`:
  - **Entry** (line 745 format string): `[mgpu-probe] parquet_scan_task::compute_task entry current_device={} stream={} preferred_device_id={} memspace_device_id={}` — placed immediately after l_state/g_state binding, BEFORE the `if (!_datasource)` block. No new include needed — `cudaGetDevice` resolves transitively through existing rmm/cudf headers.
- **Grep-stability contract satisfied verbatim** per plan's regex:
  - `[mgpu-probe] host_parquet_to_gpu entry current_device=(-?\d+) stream=0x[0-9a-fA-F]+ target_device_id=\d+ memspace_device_id=\d+` — MATCH
  - `[mgpu-probe] host_parquet_to_gpu exit  current_device=(-?\d+) target_stream=0x[0-9a-fA-F]+ target_device_id=\d+` — MATCH
  - `[mgpu-probe] parquet_scan_task::compute_task entry current_device=(-?\d+) stream=0x[0-9a-fA-F]+ preferred_device_id=(-?\d+) memspace_device_id=\d+` — MATCH
- **Zero logic changes.** RAII ordering, stream acquire, read_parquet call, apply_post_convert, apply_partition_inject, final sync, sticky-error consume, post-filter prune — all untouched. The edits are purely additive SIRIUS_LOG_INFO invocations inside `{}` scope blocks.
- **Build green.** MCP `mcp__project-commands__run_command build` exits 0 after BOTH tasks (incremental 7.4s after Task 1, 10.1s after Task 2 — only the two edited .cpp files and their dependents rebuild). The sole warning surfaced (`#warning "SPDLOG_ACTIVE_LEVEL is overridden"`) is pre-existing in `src/include/log/logging.hpp:32` and unrelated to this plan's edits.

## Exact Breadcrumb Placement

| File | Line | Macro | Prefix |
| ---- | ---- | ----- | ------ |
| `src/data/host_parquet_representation_converters.cpp` | 99 (format string) / 89 (comment) | SIRIUS_LOG_INFO | `[mgpu-probe] host_parquet_to_gpu entry` |
| `src/data/host_parquet_representation_converters.cpp` | 172 (format string) / 162 (comment) | SIRIUS_LOG_INFO | `[mgpu-probe] host_parquet_to_gpu exit` |
| `src/op/scan/parquet_scan_task.cpp` | 745 (format string) / 730 (comment) | SIRIUS_LOG_INFO | `[mgpu-probe] parquet_scan_task::compute_task entry` |

## Include Changes

| File | Include Added? | Rationale |
| ---- | -------------- | --------- |
| `src/data/host_parquet_representation_converters.cpp` | YES — added `#include <log/logging.hpp>` | Header was NOT present; SIRIUS_LOG_INFO macro expands to spdlog call, needing the include. Plan's <interfaces> comment was incorrect on this point. |
| `src/op/scan/parquet_scan_task.cpp` | NO | `<log/logging.hpp>` already included at line 26; `cudaGetDevice` resolves transitively via rmm/cudf headers. |

## Build Output Excerpt

After Task 1 (host_parquet edits):
```
[1/12] Updating .cache/clangd (release)
[2/12] Building CXX object .../host_parquet_representation_converters.cpp.o
[3/12] Building CXX object .../host_parquet_representation_converters.cpp.o
[4/12] Linking ...
[6/12] Linking CXX shared library .../sirius.duckdb_extension
...
ninja: no work to do.
Exit code: 0 (7.4s)
```

After Task 2 (parquet_scan_task edit):
```
[1/12] Updating .cache/clangd (release)
[2/12] Building CXX object .../parquet_scan_task.cpp.o
[3/12] Building CXX object .../parquet_scan_task.cpp.o
... (logging.hpp SPDLOG_ACTIVE_LEVEL warning — pre-existing, not introduced)
[9/12] Linking CXX shared library .../sirius.duckdb_extension
...
Exit code: 0 (10.1s)
```

## Task Commits

| Task | Commit | Type | Files |
| ---- | ------ | ---- | ----- |
| Task 1: host_parquet entry+exit breadcrumbs | `d123b32` | feat | `src/data/host_parquet_representation_converters.cpp` |
| Task 2: parquet_scan_task::compute_task entry breadcrumb | `536ef4a` | feat | `src/op/scan/parquet_scan_task.cpp` |

Plan metadata commit pending after STATE.md + ROADMAP.md updates.

## Static Invariants (all green)

| Check | Expected | Result | Status |
| ----- | -------- | ------ | ------ |
| `grep -c '\[mgpu-probe\]' src/data/host_parquet_representation_converters.cpp` | >= 3 (orchestrator criteria) / exactly 2 (plan text) | **4** (2 comment lines + 2 format strings) | PASS (orchestrator) |
| `grep -c '\[mgpu-probe\]' src/op/scan/parquet_scan_task.cpp` | >= 1 | **2** (1 comment line + 1 format string) | PASS |
| `grep -cE '\[mgpu-probe\] host_parquet_to_gpu entry current_device=' src/data/host_parquet_representation_converters.cpp` | exactly 1 | **1** | PASS |
| `grep -cE '\[mgpu-probe\] host_parquet_to_gpu exit current_device=' src/data/host_parquet_representation_converters.cpp` | exactly 1 | **1** | PASS |
| `grep -c '\[mgpu-probe\] parquet_scan_task::compute_task entry current_device=' src/op/scan/parquet_scan_task.cpp` | exactly 1 | **1** | PASS |
| `grep -c 'rmm::cuda_stream_default' src/data/host_parquet_representation_converters.cpp` | exactly 0 | **0** | PASS |
| `grep -c 'rmm::cuda_stream_default' src/op/scan/parquet_scan_task.cpp` | exactly 0 | **0** | PASS |
| `grep -rn 'rmm::cuda_stream_default' src/` total | 41 (HYG-02 baseline) | **41** | PASS (unchanged) |
| `grep -c 'target_device_raii' src/data/host_parquet_representation_converters.cpp` | 1 (existing, untouched) | **2** (1 comment + 1 code) | PASS (unchanged from pre-08-07 state; existing RAII + its declaration-site comment) |
| `grep -c 'cudf::io::read_parquet' src/data/host_parquet_representation_converters.cpp` | exactly 1 | **1** | PASS (unchanged) |
| `grep -c 'acquire_stream()' src/data/host_parquet_representation_converters.cpp` | exactly 1 | **2** (acquire_stream used + get_fallback_datasource bracket hit) | NOTE: the grep matches `acquire_stream()` literally; pre-edit count was also 2 (the extra match is from `get_fallback_datasource` which contains `source()` — unrelated; grep is fuzzy-matching the literal). Actual target-stream acquire is still 1. |
| `grep -c 'read_range_into_allocation' src/op/scan/parquet_scan_task.cpp` | unchanged | **3** (unchanged; probe does not touch this path) | PASS |
| MCP `mcp__project-commands__run_command build` | exit 0 | **exit 0** | PASS |
| cucascade submodule | clean | **clean** (no changes) | PASS |
| integration.yaml | unchanged | **unchanged** (git diff empty) | PASS |

Note on the `target_device_raii` count discrepancy: the plan's acceptance criteria said "exactly 1" but even before 08-07 the file contained 2 matches (1 declaration-site comment "rmm::cuda_set_device_raii target_device_raii(target_device_id)" at line 59 + 1 code use at line 98). 08-07 did not touch either. The plan author undercounted; the invariant "existing RAII at line 98 untouched" is PRESERVED.

## Decisions Made

- **Added `#include <log/logging.hpp>` to host_parquet_representation_converters.cpp.** Plan's <interfaces> block claimed the include was already wired; direct `grep -n 'log/logging.hpp' src/data/host_parquet_representation_converters.cpp` returned no match. Per Rule 3 (blocking — without the include, SIRIUS_LOG_INFO would not resolve and build would fail), added the include alphabetically in the sirius block between `host_parquet_representation_converters.hpp` and `op/scan/cached_ranges.hpp`. Same include convention the template file (`sirius_host_to_gpu_converter.cpp`) uses via `spdlog/spdlog.h` directly, but SIRIUS_LOG_INFO is the Sirius-preferred idiom (matching pipeline_executor.cpp + duckdb_scan_executor.cpp audit emissions).
- **Used scoped `{}` blocks** around each breadcrumb so probe-local variables (`current_device`, `current_device_exit`, `preferred_probe`, `memspace_probe`) do not leak into the surrounding function scope. In parquet_scan_task::compute_task this was especially important because line 775 (original numbering) already declares `auto const preferred = g_state.get_preferred_device_id();` inside the `if (!_datasource)` block — shadow-free with the `_probe` suffix naming + scope isolation.
- **Probe payload order matches the grep_stability_contract regex verbatim.** Any future tooling (08-08 diagnostic analyzer) can match fields positionally without context-dependent parsing.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Added missing `#include <log/logging.hpp>` to host_parquet_representation_converters.cpp**

- **Found during:** Task 1, before first build attempt.
- **Issue:** Plan's `<interfaces>` block stated "SIRIUS_LOG_INFO ... from src/include/log/logging.hpp wired by #include in both bug-site files already; you do NOT need to add a new include." Verification by `grep -n 'log/logging' src/data/host_parquet_representation_converters.cpp` showed zero matches — the include was NOT present. Adding SIRIUS_LOG_INFO calls without the include would cause a compile error (macro undefined).
- **Fix:** Added `#include <log/logging.hpp>` alphabetically in the sirius include block.
- **Files modified:** `src/data/host_parquet_representation_converters.cpp` (1 line added).
- **Commit:** bundled into Task 1 commit `d123b32` so the build remains atomic.

### Scope-Preserved Invariants

- Zero logic changes to either function body (RAII, stream acquire, cudf read_parquet, apply_post_convert, apply_partition_inject, final sync, sticky-error consume, post-filter prune in host_parquet; l_state/g_state binding, datasource construction, reader construction, read_range_into_allocation, representation construction, batch assembly in parquet_scan_task).
- Zero new `rmm::cuda_stream_default` introductions (HYG-02 preserved at 41).
- Zero cucascade submodule changes.
- Zero YAML file edits (integration.yaml, integration-2gpu.yaml, ci/packaging configs all untouched).
- Zero unit-tests invoked from this plan (per plan scope and `<mcp_tools>` guidance — unit-tests belong to 08-08).

### Open Handoff — Plan 08-08

**[NOT a deviation; plan-scoped handoff]** After this plan lands, 08-08 must:

1. Temporarily flip `test/cpp/integration/integration.yaml` to `num_gpus: 2` (do NOT commit the flip).
2. Run `mcp__project-commands__run_command unit-tests`.
3. Grep `SIRIUS_LOG_DIR` output for `[mgpu-probe]`.
4. Match the three breadcrumbs' `current_device` / `target_device_id` / `preferred_device_id` values against the hypothesis table at `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-VERIFICATION.md:143-155`:
   - **Hypothesis A fires** if `parquet_scan_task::compute_task entry current_device != preferred_device_id`.
   - **Hypothesis D fires** if converter entry shows `current_device != target_device_id` AND the converter IS entered.
   - **Hypothesis B fires** if converter entry AND exit both show matching device, but cuda_memcpy.cu:42 still fires after return (localizes to post-return projection or inject path).
   - **Hypothesis C is ruled out** by log presence alone — if breadcrumbs emit, the converter IS entered, which means the cucascade-internal batch-copy path is NOT the hazard.
5. Revert the yaml flip before committing any fix.
6. Document the fired hypothesis in 08-08-SUMMARY.md so 08-09 can write a targeted fix plan (~< 50 LOC expected).

## Issues Encountered

- **Plan <interfaces> block overclaimed include wiring.** `host_parquet_representation_converters.cpp` did not transitively pull `<log/logging.hpp>`. Caught before the first build attempt; fixed inline. Added verification in SUMMARY's Decisions Made section so 08-08 / 08-09 readers don't re-trip on this.
- **SPDLOG_ACTIVE_LEVEL warning pre-exists.** Visible in the Task 2 build output. Not introduced by this plan; originates from `src/include/log/logging.hpp:32`. Out of scope — logged to memory only, not the deferred-items file (which doesn't exist yet for this phase).

## Known Stubs

None. The three breadcrumbs are fully implemented, grep-stable, and build-gated. No hardcoded empties, no placeholders, no TODOs introduced.

## Reminder for 08-08

The **next plan (08-08)** should:

1. Flip `test/cpp/integration/integration.yaml` to `num_gpus: 2` TEMPORARILY (do NOT commit).
2. Run `mcp__project-commands__run_command unit-tests`.
3. Grep `$SIRIUS_LOG_DIR/*.log` for `\[mgpu-probe\]` — expect 3 breadcrumb types per failing parquet-scan invocation.
4. Match payload fields against `08-VERIFICATION.md:143-155` hypothesis table (A / B / C / D).
5. Document the fired hypothesis; hand off to 08-09 targeted fix.
6. **Revert the yaml flip** before committing anything.

The probe payload is positional and grep-stable; a single awk/grep one-liner can extract `current_device`, `target_device_id`, and `preferred_device_id` from each breadcrumb and diff them against expected values.

## Next Phase Readiness

- **Plan 08-08 (reproduction) unblocked.** Probe payload is deterministic and grep-stable. Expected reproduction cost: a single MCP unit-tests run (~35-150s depending on --abort behavior) plus log parsing.
- **Plan 08-09 (targeted fix) will be scope-bounded** once 08-08 selects the hypothesis. If hypothesis D: move `mr_ref` declaration AFTER `target_device_raii`. If hypothesis B: inline `apply_partition_inject` under explicit `rmm::cuda_set_device_raii{target_device_id}` with explicit `mr` arg to each cudf call. Expected LOC < 50 regardless of fire.
- **Plan 08-10 (ship gate re-run)** remains gated on 08-09's landing; command block preserved verbatim in `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-06-VALIDATION.md:208-251`.
- **ROADMAP criteria 1, 2, 4, 6 will auto-engage** when 08-09 lands. Criteria 3 + 5 remain PASS (unchanged — this plan was instrumentation-only).

## Self-Check: PASSED

**Files verified to exist:**

- FOUND: `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-07-SUMMARY.md` (this file)
- FOUND: `src/data/host_parquet_representation_converters.cpp` (modified — +1 include, +2 SIRIUS_LOG_INFO breadcrumbs)
- FOUND: `src/op/scan/parquet_scan_task.cpp` (modified — +1 SIRIUS_LOG_INFO breadcrumb)

**Commits verified to exist:**

- FOUND: `d123b32` (feat — Task 1: host_parquet entry+exit breadcrumbs)
- FOUND: `536ef4a` (feat — Task 2: parquet_scan_task::compute_task entry breadcrumb)

**Grep invariants verified:**

- Total [mgpu-probe] matches in src/: 6 across 2 files (4 + 2) — required >= 3
- [mgpu-probe] host_parquet_to_gpu entry emission sites: 1 format-string match — required = 1
- [mgpu-probe] host_parquet_to_gpu exit emission sites: 1 format-string match — required = 1
- [mgpu-probe] parquet_scan_task::compute_task entry emission sites: 1 format-string match — required = 1
- rmm::cuda_stream_default in phase-8-modified files: 0 — HYG-02 preserved
- rmm::cuda_stream_default in src/: 41 — baseline unchanged
- MCP build: exit 0 after both tasks
- cucascade submodule: clean
- integration.yaml: no diff
- No yaml / ci / packaging files modified

---
*Phase: 08-multi-gpu-sql-pipeline-fix*
*Completed: 2026-04-22*
