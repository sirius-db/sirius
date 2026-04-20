---
phase: 04-cucascade-bump-v1-0-re-integration
plan: 02
subsystem: scheduling
tags: [cherry-pick, push-model, data-locality, multi-gpu, port]

# Dependency graph
requires:
  - 04-01 (cucascade submodule bump to f47de0b)
provides:
  - preferred_device_id plumbing on both pipeline_task local and global state
  - task_creator data-locality score + NUMA→GPU mapping
  - management_eventloop push-model routing (pops task, routes by preferred_device_id, waits on preferred GPU at capacity)
  - duckdb_scan_executor.select_target_gpu for cross-GPU scan distribution
  - [multi_gpu_foundation] test_context.cpp cases against dev YAML fixtures
  - [data_locality] integration test file test_gpu_execution_locality.cpp
affects:
  - 04-03 (inherits preferred_device_id surface; will re-author NUMA-aware downgrade on dev's downgrade_request shape)
  - 04-04 (PORT-03 YAML + pre-commit gate runs over these files)
  - 04-05 (full unit-test gate)

# Tech tracking
tech-stack:
  added:
    - std::optional<int> preferred_device_id on gpu_pipeline_task_local_state and gpu_pipeline_task_global_state (v1.0 commit 59bc284)
    - gpu_pipeline_task::get_preferred_device_id() accessor (local wins over global)
    - task_creator::compute_data_locality_score (SCHED-01/02) + numa_to_gpu map
    - pipeline_executor::management_eventloop push-model dispatch with wait_on_preferred_device sentinel (v1.0 commit dd9264b)
    - duckdb_scan_executor::select_target_gpu (SCHED-04; v1.0 commit 7f18e66)
    - test/cpp/integration/test_gpu_execution_locality.cpp (v1.0 commit 2e6ba26)
  patterns:
    - "Cherry-pick authorship preservation: original author + date retained via `git cherry-pick` (moderate conflicts) or `git commit --author=... --date=...` (manual conflict resolution for CMakeLists.txt)."
    - "Push-model dispatch: gpu_pipeline_executor no longer publishes task_requests; management_eventloop pops from _task_queue and routes by preferred_device_id. Capacity control stays in gpu_pipeline_executor::manager_loop (bounded_pool->reserve()) — at-capacity tasks wait on the preferred GPU's queue rather than falling back (STATE.md locked invariant)."

key-files:
  created:
    - test/cpp/integration/test_gpu_execution_locality.cpp
  modified:
    - src/include/pipeline/sirius_pipeline_task_states.hpp
    - src/include/pipeline/gpu_pipeline_task.hpp
    - src/creator/task_creator.cpp
    - src/include/creator/task_creator.hpp
    - src/pipeline/pipeline_executor.cpp
    - src/pipeline/gpu_pipeline_executor.cpp
    - src/op/scan/duckdb_scan_executor.cpp
    - src/include/op/scan/duckdb_scan_executor.hpp
    - src/sirius_context.cpp
    - test/cpp/config/test_context.cpp
    - test/cpp/pipeline/test_gpu_pipeline_executor.cpp
    - test/cpp/pipeline/test_oom_reschedule.cpp
    - CMakeLists.txt

key-decisions:
  - "Wait-on-preferred-device invariant confirmed in Task 3b human-verify checkpoint: management_eventloop routes tasks to the preferred GPU executor whose manager_loop handles capacity internally; there is NO fallback re-dispatch to a different GPU."
  - "Test adaptation over re-authoring: test_gpu_pipeline_executor.cpp and test_oom_reschedule.cpp were adapted to the push-model by scheduling tasks directly on the executor (removing the `request_channel.get()` wait loops). The channel wiring is kept for fixture/constructor compatibility."
  - "API drift fix (5764cbc): `cfg.tier` → `std::holds_alternative<gpu_memory_space_config>(cfg)` because bumped cucascade f47de0b makes memory_space_config a std::variant; `local_state->_input_data->get_data_batches()` → use the pre-move `pipelineable_input` raw pointer because get_data_batches lives on the derived pipelineable_operator_data."

patterns-established:
  - "Post-v1.0 test adaptation: when a cherry-picked behavior change (push-model) breaks a pre-existing test that depends on the old pull-model, the adapter fix goes in as its own commit (not an amend) referencing the cherry-pick it adapts to."

requirements-completed:
  - PORT-01 (plumbing cherry-picks landed)
  - PORT-02 (no DuckDB vocabulary re-introduction; full unit-tests green; structural grep gates green)
  - PORT-04 (management_eventloop push-model routing with wait_on_preferred_device sentinel)

# Metrics
duration: ~2h (executor + orchestrator-side build/test loop)
completed: 2026-04-20
---

# Phase 04 Plan 02: Cherry-pick v1.0 Non-#579 Commits (Push-Model Plumbing)

Landed the 5 code-carrying v1.0 commits that do not touch the PR #579 downgrade rewrite. This brings push-model task dispatch plumbing (`preferred_device_id` on local + global state), `task_creator` locality scoring + NUMA→GPU mapping, `management_eventloop` push-model routing, `duckdb_scan_executor.select_target_gpu` cross-GPU scan distribution, and the `[data_locality]` integration test file onto the current branch preserving v1.0 authorship. PR #579-colliding commits stay deferred to Plan 04-03 for proper re-authoring against dev's new `downgrade_request` queue + POD `downgrade_task` shape.

## Commits Landed (chronological)

| Commit    | Task | Author (v1.0 preserved) | Files                                                                                                                                                                   |
| --------- | ---- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `3fab217` | 1    | Felipe Aramburu         | test/cpp/config/test_context.cpp                                                                                                                                        |
| `a1efc11` | 2a   | Felipe Aramburu         | sirius_pipeline_task_states.hpp, gpu_pipeline_task.hpp, task_creator.{cpp,hpp}                                                                                          |
| `c9b74cd` | 2b   | Felipe Aramburu         | src/sirius_context.cpp (carve-out; downgrade_executor hunks deferred to 04-03)                                                                                          |
| `90dc104` | 3a   | Felipe Aramburu         | pipeline_executor.cpp, gpu_pipeline_executor.cpp (rebased — wait_on_preferred_device sentinel added)                                                                    |
| `5764cbc` | fix  | Felipe Aramburu (me)    | task_creator.cpp, test_context.cpp (post-bump API drift adaptations)                                                                                                    |
| `5e8e9b7` | 4    | Felipe Aramburu         | duckdb_scan_executor.{cpp,hpp}                                                                                                                                          |
| `2c28d4f` | 5    | Felipe Aramburu         | CMakeLists.txt, test/cpp/integration/test_gpu_execution_locality.cpp                                                                                                    |
| `6f13b97` | 6    | Felipe Aramburu (me)    | test/cpp/pipeline/test_gpu_pipeline_executor.cpp (push-model adaptation)                                                                                                |
| `3b5c029` | 6    | Felipe Aramburu (me)    | test/cpp/pipeline/test_oom_reschedule.cpp (push-model adaptation)                                                                                                       |

- `e1dab76` was inspected per research OQ-1 and found to be empty on its target file (no fixup needed).
- All v1.0-authored commits retain the original author + date.

## Accomplishments by Task

### Task 1 — Cherry-pick 3777645 → `3fab217`
Resolved `test/cpp/config/test_context.cpp` conflict by translating v1.0's env-var fixtures to dev's YAML test pattern (PR #565). Both the visible `[multi_gpu_foundation]` TEST_CASE and the hidden `[.][multi_gpu_foundation]` (requires >=2 GPUs) TEST_CASE are preserved. Catch2 v2 `WARN+return` skip idiom retained at line 262.

### Task 2a — Cherry-pick 59bc284 plumbing → `a1efc11`
Added `std::optional<int> preferred_device_id` on both local and global task states, `get_preferred_device_id()` accessor on `gpu_pipeline_task`, and `compute_data_locality_score` + `_numa_to_gpu` on `task_creator`. `sirius_context.cpp` carved out to Task 2b.

### Task 2b — sirius_context.cpp carve-out → `c9b74cd`
Threaded `system_topology_info*` into `task_creator`'s constructor so the locality score has access to NUMA topology. Downgrade_executor and P2P-loop hunks deferred to Plan 04-03 with `TODO(04-03)` marker in place.

### Task 3a — Cherry-pick dd9264b → `90dc104` (rebased)
Push-model `management_eventloop`: pop task → `dynamic_cast<gpu_pipeline_task*>` → `get_preferred_device_id()` → route to `_gpu_executors.at(target_device_id)`. Added `wait_on_preferred_device` comment sentinel so the locked STATE.md invariant ("wait on preferred GPU rather than falling back") is grep-discoverable. Removed `task_request_publisher.send()` from `gpu_pipeline_executor.cpp` (pull-model is dormant).

### Task 3b — Human-verify checkpoint (approved)
User verified routing semantics by inspecting `90dc104`'s pipeline_executor.cpp diff (lines 225-260). Enqueueing on the preferred executor's queue means at-capacity tasks wait on THAT executor (capacity gated by `bounded_pool->reserve()` in gpu_pipeline_executor::manager_loop), not a different GPU. No fallback re-dispatch path exists.

### Task 4 — Cherry-pick 7f18e66 → `5e8e9b7`
`duckdb_scan_executor::select_target_gpu` distributes scan batches across GPUs proportional to available GPU memory using `any_memory_space_in_tier_with_preference`. Pure cherry-pick, no conflict.

### Task 5 — Cherry-pick 2e6ba26 → `2c28d4f`
Added `test/cpp/integration/test_gpu_execution_locality.cpp` with `[data_locality]` tag. CMakeLists.txt conflict resolved by keeping both dev's `test_logical_type.cpp` and v1.0's new locality test.

### Task 6 — Full unit-test gate (MCP)
- Build: exit 0 after each cherry-pick block.
- Tests: All 963 test cases passed (78,789,759 assertions) — clean run.
- Two additional adapter commits (6f13b97, 3b5c029) were required because `test_gpu_pipeline_executor` and `test_oom_reschedule` depended on the pull-model and blocked forever on `request_channel.get()`. Adapted both tests to schedule tasks directly onto the executor.
- Earlier unit-test runs exhibited the same TPC-H parquet flake documented in Plan 04-01 (Run 2 of 5 there failed on TPC-H Q4 parquet, same shape here on Q6 then group-by-key parquet). Confirmed flake by running to green after the test adaptations.

## Performance

- Build (incremental): ~7-30s after each commit
- Full build (after new CMakeLists entry): ~285s (cold)
- Full unit-test suite: ~220s on exit 0; flaky TPC-H-parquet runs aborted earlier (~138–220s)
- PORT-02 structural grep gates: all pass
  - `preferred_device_id` in `sirius_pipeline_task_states.hpp`: 4 occurrences
  - `preferred_device_id` in `gpu_pipeline_task.hpp`: 6 occurrences
  - `compute_data_locality_score`/`preferred_device` in `task_creator.cpp`: 9 occurrences
  - `wait_on_preferred_device` in `pipeline_executor.cpp`: present
  - `select_target_gpu` in `duckdb_scan_executor.cpp`: 2 occurrences
  - `[multi_gpu_foundation]` in `test_context.cpp`: 5 TEST_CASE tag hits
  - `[data_locality]` in `test_gpu_execution_locality.cpp`: 11 TEST_CASE tag hits
  - `LogicalType::(INTEGER|BIGINT|VARCHAR)` in any cherry-picked file: 0 (no DuckDB vocabulary re-introduction)
  - `libconfig` in test_context.cpp: 0

## Issues Encountered

- **Executor agent MCP access:** The spawned executor agent did not have `mcp__project-commands__run_command` in its tool catalog, so it blocked on build/test verification. Orchestrator ran builds/tests directly via MCP. Next plan executions should ensure the MCP server is attached before spawning executors.
- **cucascade API drift:** Two unrelated compile fixes surfaced in 5764cbc after the cherry-picks: (1) `memory_space_config` is now a variant, so tier queries go through `std::holds_alternative`; (2) `get_data_batches()` lives on the derived `pipelineable_operator_data`, not `operator_data` base — reuse the pre-move dynamic_cast raw pointer.
- **Pull-model test dependencies:** Two tests (`test_gpu_pipeline_executor`, `test_oom_reschedule`) deadlocked on `request_channel.get()` after push-model landed. Adapted in-place rather than reverting the v1.0 change. The channel wiring remains so the fixture/constructor API doesn't change.
- **TPC-H parquet flake:** Pre-existing flake documented in Plan 04-01 (outside BUMP-03 scope). Re-appeared on early 04-02 test runs under different symptoms (Q6 SIGSEGV, Q1-family parquet SIGSEGV). Not blocking — one of the 04-01 runs also hit it.
