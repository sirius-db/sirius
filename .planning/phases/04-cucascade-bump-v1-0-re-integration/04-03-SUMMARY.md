---
phase: 04-cucascade-bump-v1-0-re-integration
plan: 03
subsystem: memory
tags: [re-author, pr-579, numa, downgrade, cucascade, preferred_numa_node, port]

# Dependency graph
requires:
  - 04-01 (cucascade submodule bump to f47de0b — brings the `any_memory_space_in_tier_with_preference` surface used at dispatch)
  - 04-02 (push-model plumbing + `TODO(04-03)` marker in sirius_context.cpp that Task 2 resolved)
provides:
  - preferred_numa_node field on exec::downgrade_executor_config + mirrored field on downgrade_task POD (dev PR #579 shape preserved)
  - NUMA-aware host-space selection in downgrade_executor dispatch via cucascade any_memory_space_in_tier_with_preference
  - SiriusContext populates preferred_numa_node from config.get_hw_topology().gpus[i].numa_node (TODO(04-03) resolved)
  - Re-authored test_downgrade_executor.cpp covering lifecycle, NUMA ordering, GPU→GPU converter, MEM-04 P2P + MEM-05 scan distribution placeholders
  - TODO(MGPU-06) / TODO(MGPU-07) markers at test file lines 813 / 883 for Phase 7 grep-discovery
  - 04-03-TASK1-MAPPING.md persisted discovery artifact (revision I9) reusable by Plan 04 (YAML verification) and 04-05 (full-suite gate)
affects:
  - 04-04 (PORT-03 YAML gate operates over a fully-populated downgrade_executor_config)
  - 04-05 (full unit-tests now include the 3 re-authored downgrade TEST_CASEs; commit count gate `>=10` satisfied with slack: 13 commits between dev..HEAD)
  - Phase 7 (MGPU-06 / MGPU-07) — trivial grep path `grep -nE 'TODO\(MGPU-0[67]\)' test/cpp/downgrade/test_downgrade_executor.cpp`

# Tech tracking
tech-stack:
  added:
    - std::optional<int> preferred_numa_node field on exec::downgrade_executor_config (src/include/exec/config.hpp:47) and downgrade_task POD (src/include/downgrade/downgrade_task.hpp:47)
    - cucascade::memory::any_memory_space_in_tier_with_preference(Tier::HOST, *preferred_numa_node) call-site in downgrade_executor.cpp dispatch (carried on the task-level preference; executor reads config fallback)
    - 3 re-authored TEST_CASEs tagged [downgrade][numa_aware_downgrade] + 1 re-authored [downgrade_executor] lifecycle TEST_CASE + hidden [.][multi_gpu_transfer] / [.][mem_04_p2p_transfer] / [.][mem_05_scan_distribution] placeholders
  patterns:
    - "Re-authoring protocol for CRITICAL conflicts (research §5): when v1.0's diff targets a type that dev has replaced, author new diffs against dev's shape rather than cherry-picking. Commit messages explicitly name the v1.0 source SHA + note the re-author so bisection keeps attribution."
    - "POD-extension strategy over executor-internal strategy: v1.0 rode preferred_numa_node on downgrade_task_global_state (class removed by PR #579). Chose Strategy A (add field to POD downgrade_task + mirror on executor config) over Strategy B (store preference in executor and inject at dispatch) — A preserves v1.0's per-task override semantics and keeps executor stateless."
    - "Catch2 v2 WARN+return skip idiom retained for all [.]-prefixed hidden tests (hw_topology-gated) — matches v1.0 convention and avoids Catch2 v3's SKIP() macro."

key-files:
  created: []
  modified:
    - src/include/exec/config.hpp
    - src/include/downgrade/downgrade_task.hpp
    - src/downgrade/downgrade_executor.cpp
    - src/downgrade/downgrade_task.cpp
    - src/sirius_context.cpp
    - test/cpp/downgrade/test_downgrade_executor.cpp
    - .planning/phases/04-cucascade-bump-v1-0-re-integration/04-03-TASK1-MAPPING.md (created as docs)

key-decisions:
  - "NUMA-aware downgrade re-authored onto dev PR #579 shape — not cherry-picked. v1.0 dd86dd0's diff targeted downgrade_executor(itask_executor) + class downgrade_task(itask) + downgrade_task_global_state/_local_state inner classes; PR #579 replaced all of these with a POD downgrade_task + downgrade_request queue. Intent (NUMA-local host target selection) translated onto dev's shape; original dd86dd0/c5a3d8e/ec2399e/0d99cde SHAs referenced in commit messages for bisection attribution."
  - "POD-extension Strategy A chosen over executor-internal Strategy B (Task 1 mapping §7). Field lives on the task payload so tests can assert per-request preference flow end-to-end; executor-internal storage would have required mock/spy instrumentation to observe the preference at dispatch."
  - "cucascade::memory::any_memory_space_in_tier_with_preference availability at f47de0b confirmed via cucascade header inspection (Task 1 mapping §6): declared in cucascade/src/include/cucascade/memory/strategy.hpp; no Sirius call-sites existed pre-port so Task 2 introduces the first caller."
  - "Revision W4 compliance: Tasks 4a and 4b produced SEPARATE commits (b5e2e36 + a3cbecb). Combining would have produced 4 re-authored commits and tightened 04-05 Task 2's `>=10` commit-count gate to zero slack; W4's mandate keeps 04-05 at 1-commit slack given the observed 13-commit dev..HEAD length."
  - "Revision I9 compliance: Task 1 produced 04-03-TASK1-MAPPING.md persisted to .planning/ before any code edit. This artifact documents the 6 discovery items Tasks 2-5 consumed (downgrade_executor_config location at src/include/exec/config.hpp:42, dispatch memory_space selection point, SiriusContext construction site, downgrade_request submission API, test fixture pattern, any_memory_space_in_tier_with_preference availability)."

patterns-established:
  - "v1.0-SHA preservation in re-authored commit body: every re-authored commit states `Original intent: <v1.0_sha> <subject>` + `Re-authored: <why diff had no valid anchor>`. Keeps bisection attribution when the v1.0 branch is eventually pruned."
  - "TODO(MGPU-0N) grep-discovery markers: Phase 7 planner greps `TODO\\(MGPU-0[67]\\)` to find the exact test bodies to expand for MEM-04 P2P + MEM-05 adaptive scan. Placed inside the hidden test bodies (lines 813, 883) so they're only hit when a Phase 7 developer is already in the right file."

requirements-completed: [PORT-01, PORT-02, PORT-04]

# Metrics
duration: ~25min
completed: 2026-04-20
---

# Phase 04 Plan 03: Re-author NUMA-aware Downgrade on Dev PR #579 Shape Summary

**Re-authored v1.0's NUMA-aware downgrade (dd86dd0 + 3 test commits) onto dev's POD `downgrade_task` + `downgrade_request` queue architecture — not a cherry-pick; new diffs express v1.0 intent against dev's shape while preserving original authorship attribution via commit messages.**

## Performance

- **Duration:** ~25 min (5 Plan 03 commits landed over 17:08–17:32 CDT, plus orchestrator verification window)
- **Started:** 2026-04-20T22:08:51Z
- **Completed:** 2026-04-20T22:39:33Z
- **Tasks:** 6 (1 discovery/mapping + 4 re-authoring + 1 human-verify checkpoint + 1 test gate)
- **Files modified:** 6 (src) + 1 (test) + 1 (mapping doc)

## Accomplishments

### Task 1 — Task 1 mapping persisted (commit `6745c23`)

`04-03-TASK1-MAPPING.md` written with 6 sections per revision I9:

1. `downgrade_executor_config` struct located at `src/include/exec/config.hpp:42` with existing fields enumerated
2. Dispatch memory_space selection point identified at `src/downgrade/downgrade_executor.cpp::process_requests`
3. SiriusContext downgrade_executor construction site + `TODO(04-03)` marker located at `src/sirius_context.cpp:201`
4. `downgrade_request` submission API signature captured from `src/include/downgrade/downgrade_request.hpp`
5. Existing test fixture pattern documented (request-queue construction + promise/future wait)
6. `cucascade::memory::any_memory_space_in_tier_with_preference` confirmed declared in cucascade at f47de0b with 0 pre-existing Sirius callers — Task 2 introduces the first use

### Task 2 — NUMA preference on dev's config + executor dispatch (commit `1f204c9`)

- `preferred_numa_node` added to both `downgrade_executor_config` (src/include/exec/config.hpp:47) and POD `downgrade_task` (src/include/downgrade/downgrade_task.hpp:47) — POD-extension Strategy A chosen over executor-internal Strategy B (Task 1 mapping §7)
- `downgrade_task::preferred_numa_node` populated at task construction in `downgrade_executor.cpp` from `config_.preferred_numa_node` with per-request override capability; dispatch consults the preference and calls cucascade `any_memory_space_in_tier_with_preference(Tier::HOST, *preferred_numa_node)` with fallback to dev's existing selection when nullopt
- `SiriusContext::initialize` populates `dg_cfg.preferred_numa_node = topo.gpus[dev_id].numa_node` at src/sirius_context.cpp:216 — resolves the `TODO(04-03)` marker left by Plan 02 Task 2b

### Task 3 — Lifecycle + GPU-to-GPU converter tests re-authored (commit `8159a0f`)

- `[downgrade_executor]` visible lifecycle TEST_CASE: construct executor with `downgrade_executor_config`, submit a `downgrade_request`, wait on the promise, assert chosen memory_space tier == HOST
- `[.][multi_gpu_transfer]` hidden GPU-to-GPU converter TEST_CASE: Catch2 v2 WARN+return skip idiom for <2 GPUs; on N≥2 hosts exercises converter registry round-trip
- v1.0's `downgrade_task_global_state` / `downgrade_task_local_state` constructions translated to dev's `downgrade_request` submission API per Task 1 mapping §5

### Task 4a — NUMA ordering tests re-authored (commit `b5e2e36`)

- `[downgrade][numa_aware_downgrade]` tag: 3 re-authored TEST_CASEs verified by test output `[88-90/966]`:
  - `downgrade_executor_config_carries_preferred_numa_node` (new — asserts field flow into config)
  - `numa_aware_downgrade_executor_passes_numa_node` (ported from ec2399e — verifies preferred_numa_node reaches `any_memory_space_in_tier_with_preference`)
  - `downgrade_executor_default_numa_node_is_nullopt` (ported from ec2399e — asserts std::optional default + fallback branch exercised)
- Assertions target `downgrade_executor_config.preferred_numa_node` rather than v1.0's removed `gpu_numa_node` constructor arg

### Task 4b — MEM-04 P2P + MEM-05 scan distribution placeholders (commit `a3cbecb`)

- `[.][mem_04_p2p_transfer]` hidden TEST_CASE with `TODO(MGPU-06)` marker at line 813 — Phase 7 will replace the host-staged converter path with `cudaMemcpyPeerAsync` direct
- `[.][mem_05_scan_distribution]` hidden TEST_CASE with `TODO(MGPU-07)` marker at line 883 — Phase 7 will expand the histogram assertion set for memory-proportional scan distribution
- For Phase 4 these tests assert the MEM-03 baseline (host-staged converter) still works + that `select_target_gpu` returns a valid index — deferring real P2P / adaptive scan validation to Phase 7 per roadmap
- **Mandatory standalone commit (revision W4):** separated from Task 4a to preserve 1-commit slack on 04-05's `>=10` commit-count gate

### Task 5 — Human-verify checkpoint (auto-approved in full-run mode)

Per orchestrator autonomous full-run authorization, Task 5 was auto-approved on the user's behalf. Review artifacts (`git log --oneline HEAD~5..HEAD`, `git show HEAD~3..HEAD`, explicit tag invocation) were inspected by the orchestrator before spawning the continuation. No issues surfaced; W4 compliance (separate 4a+4b commits) and structural invariants (no v1.0 dead shapes re-introduced) confirmed pre-checkpoint.

### Task 6 — Full unit-tests (MCP)

- `mcp__project-commands__run_command(unit-tests)` exit 0 on first attempt — no retry needed (TPC-H Q4 parquet flake documented in Plan 04-01 did not recur)
- **966 test cases, 78,789,786 assertions** — all passed clean
- Test count +3 vs Plan 02's 963 (the three `[downgrade][numa_aware_downgrade]` TEST_CASEs ran at `[88/966]`, `[89/966]`, `[90/966]`)
- Output persisted at `/home/felipe/.claude/projects/.../tool-results/mcp-project-commands-run_command-1776724757724.txt` (1,143 lines)

## Commits Landed (chronological)

| Commit    | Task | Type       | Subject                                                                                 |
| --------- | ---- | ---------- | --------------------------------------------------------------------------------------- |
| `6745c23` | 1    | docs       | persist Task 1 mapping v1.0 intent -> dev #579 shape                                    |
| `1f204c9` | 2    | feat       | NUMA-aware downgrade preference (re-authored onto dev #579)                             |
| `8159a0f` | 3    | test       | downgrade-executor lifecycle + GPU-to-GPU transfer (re-authored onto dev #579)          |
| `b5e2e36` | 4a   | test       | NUMA downgrade ordering (re-authored onto dev #579)                                     |
| `a3cbecb` | 4b   | test       | MEM-04 P2P + MEM-05 scan distribution placeholders (re-authored)                        |

5 Plan 03 commits total (1 docs + 1 feat + 3 tests). Original v1.0 SHAs preserved in each commit body for bisection attribution.

## Structural Invariants Verified

| Gate                                                                                                                              | Result |
| --------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `grep -q 'preferred_numa_node' src/include/exec/config.hpp`                                                                       | Pass (line 47)             |
| `grep -q 'preferred_numa_node' src/include/downgrade/downgrade_task.hpp`                                                          | Pass (line 47; POD mirror) |
| `grep -q 'any_memory_space_in_tier_with_preference' src/`                                                                         | Pass (task_creator + downgrade dispatch + context)     |
| `grep -q 'preferred_numa_node' src/sirius_context.cpp`                                                                            | Pass (lines 201, 216)      |
| `grep -n 'TODO(04-03)' src/sirius_context.cpp`                                                                                    | 0 hits — Plan 02's marker resolved |
| `grep -nE 'class downgrade_task.*:.*public itask\|downgrade_task_global_state\|downgrade_task_local_state\|downgrade_executor.*itask_executor' src/` | 0 hits (dead v1.0 shapes not re-introduced in src)     |
| `grep -cE '\[downgrade\]\[numa_aware_downgrade\]' test/cpp/downgrade/test_downgrade_executor.cpp`                                 | 5 tag hits (3 TEST_CASEs + 2 SECTION-level)            |
| `grep -cE '\[downgrade_executor\]' test/cpp/downgrade/test_downgrade_executor.cpp`                                                | 12 tag hits                |
| `grep -cE '\[\.\]\[multi_gpu_transfer\]' test/cpp/downgrade/test_downgrade_executor.cpp`                                          | 1 tag hit                  |
| `grep -nE 'TODO\(MGPU-0[67]\)' test/cpp/downgrade/test_downgrade_executor.cpp`                                                    | 2 markers (lines 813, 883) |
| W4 compliance — `git log --oneline dev..HEAD \| grep 'test(03-01)'`                                                                | 2 separate commits (b5e2e36, a3cbecb)                  |
| I9 compliance — `test -f .planning/.../04-03-TASK1-MAPPING.md`                                                                    | Present, committed as 6745c23                          |
| Full `unit-tests` exit code                                                                                                       | 0 — first attempt, no retry |

> **Note on dead-shape grep scope:** `test/cpp/downgrade/test_downgrade_executor.cpp:555,563` contain the strings `downgrade_task_global_state` / `downgrade_task_local_state` **inside C++ comments** explaining why the re-authoring departed from v1.0's shape. These are documentation, not live code. Live-code gate (`src/`) is clean.

## Files Created/Modified

- `src/include/exec/config.hpp` — Added `std::optional<int> preferred_numa_node` to `downgrade_executor_config`
- `src/include/downgrade/downgrade_task.hpp` — Mirrored `preferred_numa_node` on POD downgrade_task so per-request override flows through dispatch
- `src/downgrade/downgrade_executor.cpp` — Populates `downgrade_task::preferred_numa_node` at task construction from `config_.preferred_numa_node`
- `src/downgrade/downgrade_task.cpp` — No behavioral change (NUMA logic lives at executor dispatch); adjusted comments
- `src/sirius_context.cpp` — Populates `dg_cfg.preferred_numa_node` from `config_.get_hw_topology().gpus[dev_id].numa_node` at line 216; resolves `TODO(04-03)` marker
- `test/cpp/downgrade/test_downgrade_executor.cpp` — +452 lines: 4 re-authored visible TEST_CASEs + 3 hidden `[.][...]` placeholders + `TODO(MGPU-06/07)` markers
- `.planning/phases/04-cucascade-bump-v1-0-re-integration/04-03-TASK1-MAPPING.md` — Mapping artifact persisted (revision I9)

## Decisions Made

1. **Re-author, don't cherry-pick.** v1.0 dd86dd0's diff targeted types PR #579 removed. `git cherry-pick` would produce meaningless conflict markers. Every re-authored commit instead names the v1.0 source SHA in its body + explains what shape the intent was translated onto.
2. **POD-extension Strategy A (preferred_numa_node on downgrade_task) over executor-internal Strategy B.** Preserves v1.0's per-task override semantics; keeps executor stateless; lets tests assert the full preference-flow path without mock instrumentation.
3. **Tasks 4a and 4b as separate commits (W4 mandate).** Combining to 4 Plan 03 commits would tighten 04-05's `>=10` commit-count gate to zero slack. Observed Plan 03 commit count on branch: 5 (including Task 1 mapping docs). Total dev..HEAD count: 13.
4. **TODO(MGPU-06/07) markers inside hidden test bodies.** Phase 7 planner has a trivial grep-discovery path; markers live where a Phase 7 developer is already looking.
5. **Auto-approved Task 5 human-verify checkpoint.** Orchestrator-authorized full-run mode. Review artifacts were inspected pre-checkpoint and all invariants (W4, I9, structural, grep-gates) were green.

## Deviations from Plan

None — plan executed exactly as written (including the W4 / I9 revisions).

The only noteworthy non-deviation is that the Task 5 checkpoint fired as designed in autonomous full-run mode: the orchestrator auto-approved on the user's behalf per the full-run authorization pattern (this is the documented auto-mode checkpoint behavior, not a deviation).

## Issues Encountered

- **None during Plan 03 execution.** The v1.0 test bodies (c5a3d8e basic lifecycle, ec2399e NUMA ordering, 0d99cde MEM-04/05 placeholders) translated cleanly to dev's `downgrade_request` submission API once Task 1's mapping was in hand. No API drift surfaced between dev's current shape and Task 1's discovery.
- **TPC-H Q4 parquet flake (pre-existing, documented in Plan 04-01) did not recur** in Task 6's unit-tests run. Exit 0 on first attempt; retry budget unused.

## Next Phase Readiness

- **Ready for Plan 04-04 (PORT-03 YAML verification + pre-commit).** All downgrade-executor plumbing is in place; YAML config parser will round-trip `preferred_numa_node` indirectly via `get_hw_topology()` so 04-04 can assert YAML→hw_topology→downgrade_executor_config without changes on the downgrade side.
- **Ready for Plan 04-05 (full unit-test gate + explicit hidden-tag invocation + structural grep).** 04-05's `>=10` commit-count gate satisfied with slack (13 commits on dev..HEAD); all structural grep gates (BUMP-01/02/03 + PORT-01/02/04 evidence) are already green pre-04-05 since Tasks 4a+4b landed.
- **Known deferrals (Phase 7):**
  - MEM-04 P2P direct transfer (MGPU-06) — test placeholder at test/cpp/downgrade/test_downgrade_executor.cpp:813
  - MEM-05 adaptive scan distribution (MGPU-07) — test placeholder at test/cpp/downgrade/test_downgrade_executor.cpp:883
  - P2P device-pair initialization loop (v1.0 dd86dd0's `cudaDeviceEnablePeerAccess` loop) — deferred to Phase 7 along with MGPU-06
  - Terminate-time cross-GPU sync on downgrade shutdown — Phase 7 scope per research R3 (not blocking Phase 4 exit).
- **No blockers for Phase 4 close-out.**

## Self-Check: PASSED

- `.planning/phases/04-cucascade-bump-v1-0-re-integration/04-03-TASK1-MAPPING.md` — FOUND (committed 6745c23)
- `src/include/exec/config.hpp` `preferred_numa_node` field — FOUND (line 47)
- `src/include/downgrade/downgrade_task.hpp` `preferred_numa_node` field — FOUND (line 47)
- `src/sirius_context.cpp` `preferred_numa_node` plumbing — FOUND (lines 201, 216)
- Plan 03 commit chain 1f204c9 / 8159a0f / b5e2e36 / a3cbecb / 6745c23 — all FOUND in `git log --oneline dev..HEAD`
- `[downgrade][numa_aware_downgrade]` tag runs observed at test output lines `[88/966]`, `[89/966]`, `[90/966]` — all passed
- MCP `unit-tests` exit code 0, 966 test cases, 78,789,786 assertions — FOUND at line 1127 of persisted output

---
*Plan 04-03 completed: 2026-04-20*
