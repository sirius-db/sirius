# Plan 04-03 Task 1 Mapping: v1.0 intent → dev (PR #579) shape

**Created:** 2026-04-20
**Persisted per revision I9** so the 6 discovery items survive context interruption and Tasks 2-5 can consume them directly without re-doing discovery.

This mapping re-authors v1.0's NUMA-aware downgrade intent (commits `dd86dd0`, `c5a3d8e`, `ec2399e`, `0d99cde`) onto dev's post-PR-#579 architecture. v1.0's diff targets a class hierarchy that no longer exists on dev: `itask_executor`-inheriting `downgrade_executor`, `itask`-inheriting `downgrade_task`, `downgrade_task_global_state` / `downgrade_task_local_state` inner classes, and a `run_downgrade_pass()` entry-point. Dev replaced the whole subsystem with a `downgrade_request` queue architecture and a POD `downgrade_task` struct.

## 1. downgrade_executor_config definition

- **File:** `src/include/exec/config.hpp`
- **Lines:** 32-38
- **Current fields:**
  - `exec::thread_pool_config thread_pool` (default: 4 threads, prefix "downgrade")
  - `uint64_t monitor_period_ms` (default: 10; 0 disables monitor loop)
- **New field to add (Task 2):** `std::optional<int> preferred_numa_node`
  - Semantic: preferred HOST memory_space device_id (NUMA node) for GPU→HOST downgrade dispatch selection.
  - Default: `std::nullopt` (no preference; dispatch uses existing `any_memory_space_in_tier{Tier::HOST}` fallback path).

## 2. Downgrade dispatch memory_space-selection point

- **File:** `src/downgrade/downgrade_task.cpp`
- **Lines:** 49-67 (inside `downgrade_task::execute`)
- **Containing function:** `bool downgrade_task::execute(rmm::cuda_stream_view stream)`
- **Current selection logic:**
  ```cpp
  auto reservation = res_mgr.request_reservation(
    cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST}, data_size);
  ```
  The dispatch currently uses the unpreferred `any_memory_space_in_tier` strategy — whichever HOST memory_space has capacity is picked, with no NUMA-locality hint.

- **How the preference flows in (Task 2 re-authoring plan):**
  v1.0 carried the NUMA preference on `downgrade_task_global_state`, which the task read inside its `execute`. That class no longer exists. Dev's POD `downgrade_task` is `{batch, res_mgr}` only — it doesn't know about executor config. To thread the preference through, two equivalent strategies work:

  **Strategy A (preferred, minimal surface change):** Expand the POD `downgrade_task` struct in `src/include/downgrade/downgrade_task.hpp` to carry `std::optional<int> preferred_numa_node`. In `src/downgrade/downgrade_executor.cpp::processing_loop`, when constructing the task at line 147, pass `_config.preferred_numa_node` into the POD. Then `downgrade_task::execute` (line 51) calls `any_memory_space_in_tier_with_preference{Tier::HOST, preferred_numa_node}` when the field has a value, falling back to `any_memory_space_in_tier{Tier::HOST}` otherwise.

  **Strategy B (do-not-touch-task):** Keep POD `downgrade_task` as-is. Move the reservation-request logic into `processing_loop` itself (or a helper on `downgrade_executor`) so it can read `_config.preferred_numa_node` directly, construct the reservation, pass the chosen `memory_space*` into a new `downgrade_task` field. More invasive — changes dispatch lifecycle.

  **Decision for Task 2:** Strategy A. Adds one field to the POD, one parameter at construction, one branch inside `execute`. The POD shape stays a POD (no polymorphism, no inheritance — same invariant the research §2 "CRITICAL (rewrite)" row requires).

- **Where the preference is READ at dispatch time:** `src/downgrade/downgrade_task.cpp` lines 50-52 (the `res_mgr.request_reservation(...)` call). Task 2 rewrites this hunk to branch on `preferred_numa_node`.

## 3. downgrade_executor construction in SiriusContext

- **File:** `src/sirius_context.cpp`
- **Lines:** 195-213 (`create_executors_for_tier` lambda), 222-224 (the `TODO(04-03)` marker), 225-228 (task_creator construction immediately below)
- **Current constructor arg shape:**
  ```cpp
  downgrade_executor(dg_cfg,
                     *data_repository_manager_,
                     space->get_id(),
                     const_cast<cucascade::memory::memory_space*>(space),
                     *memory_manager_);
  ```
  (`dg_cfg` is `exec::downgrade_executor_config&` from `config_.get_downgrade_executor_config()` at line 200 — it is captured by-reference as `const` so the current lambda cannot mutate it per-executor. Task 2 must copy into a local `downgrade_executor_config` per executor to attach the per-GPU `preferred_numa_node`.)
- **TODO(04-03) marker line:** `src/sirius_context.cpp:222` (with supporting context on 223-224). Plan 02 Task 2b placed this marker; Task 2 of this plan resolves it.
- **hw_topology access pattern:** `config_.get_hw_topology().gpus[device_id].numa_node` — dev's `system_topology_info` still exposes `gpus[i].numa_node` (confirmed identical shape to v1.0). Access path used by plan_02 commit `c9b74cd` is `const_cast<system_topology_info*>(&config_.get_hw_topology())`.

## 4. downgrade_request submission API

- **Request struct:** `src/include/downgrade/downgrade_executor.hpp:50-58`
  ```cpp
  struct downgrade_request {
    size_t target_bytes{0};
    std::function<bool()> predicate;
    std::promise<size_t> result;
    std::atomic<size_t> bytes_freed{0};
    std::atomic<size_t> batches_downgraded{0};
    std::atomic<bool> satisfied{false};
    bool is_monitor_request{false};
  };
  ```
- **Submission functions on `downgrade_executor`** (`src/include/downgrade/downgrade_executor.hpp`):
  - `std::future<size_t> request_free_memory(size_t bytes)` (line 122) — async
  - `size_t request_free_memory_and_wait(size_t bytes)` (line 132) — sync wrapper
  - `std::future<size_t> request_downgrade(size_t target_bytes, std::function<bool()> predicate)` (line 145) — predicate-driven async
- **Completion signal:** `std::promise<size_t>` resolves to bytes actually freed. `request_free_memory_and_wait` calls `.get()`. Monitor requests (is_monitor_request=true) are fire-and-forget.
- **Example submission call-site:** `src/downgrade/downgrade_executor.cpp:186-208` (`monitor_loop` constructing a fire-and-forget request); `src/downgrade/downgrade_executor.cpp:348-360` (`request_free_memory` constructing a promise-backed request).
- **Internal processing:** `downgrade_executor::processing_loop` (`src/downgrade/downgrade_executor.cpp:109-185`) pops requests off the MPMC queue, collects candidates via `collect_all_candidates`, dispatches per-batch work onto `_pool`, fulfills the promise with total bytes freed.

## 5. Test fixture pattern for downgrade_executor

- **Example TEST_CASE:** `test/cpp/downgrade/test_downgrade_executor.cpp:149-164` (`"Single downgrade task executes correctly"`, tag `[downgrade_executor]`)
- **make_test_memory_manager helper:** `test/cpp/downgrade/test_downgrade_executor.cpp:53-75` — builds a 1-GPU config via `cucascade::memory::reservation_manager_configurator`.
- **make_test_executor helper:** `test/cpp/downgrade/test_downgrade_executor.cpp:106-113`:
  ```cpp
  sirius::exec::downgrade_executor_config config{
    .thread_pool = {.num_threads = 1, .thread_name_prefix = "downgrade"},
    .monitor_period_ms = 0};
  return downgrade_executor(config, repo_mgr, GPU_SPACE_ID, gpu_space, mem_mgr);
  ```
- **How a downgrade is initiated in tests:**
  ```cpp
  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();
  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);
  // check batch->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST
  executor.stop();
  ```
- **How assertions wait for completion:** Either `request_free_memory_and_wait` (sync) returns and tests inspect the batch's tier directly, OR tests call `request_free_memory(...).get()` on the returned `std::future<size_t>`.
- **Direct task test pattern:** `test_downgrade_executor.cpp:149-164` constructs `downgrade_task{batch, *mem_mgr}` directly (POD struct) and calls `task.execute(stream)`. This bypasses the executor entirely — useful for testing NUMA-preference selection deterministically.
- **Multi-GPU manager helper:** v1.0's ec2399e/c5a3d8e built a local `make_multi_gpu_memory_manager()` that sets `set_number_of_gpus(2)` + `use_host_per_gpu()`. The current test file has only `make_test_memory_manager()` (single GPU). Task 4a will add the multi-GPU variant.
- **Mock/spy patterns used:** NONE on dev. v1.0 tests used direct inspection (examining the memory_space device_id the batch landed on) rather than mocking. Task 4a will preserve this pattern — assert on `batch->get_memory_space()->get_device_id()` after downgrade to verify the NUMA preference took effect.

## 6. cucascade::memory::any_memory_space_in_tier_with_preference availability at f47de0b

- **Grep command result:**
  ```
  grep -rn 'any_memory_space_in_tier_with_preference' src/ test/ cucascade/
  ```
  - `src/op/scan/duckdb_scan_executor.cpp:297` — existing Sirius usage (plan 02 Task 4 landed commit 5e8e9b7)
  - `cucascade/src/memory/memory_reservation_manager.cpp:65` — definition
  - `cucascade/include/cucascade/memory/memory_reservation_manager.hpp:79` — declaration

- **Header where declared:** `cucascade/include/cucascade/memory/memory_reservation_manager.hpp`, lines 75-90.

- **Signature:**
  ```cpp
  struct any_memory_space_in_tier_with_preference : public reservation_request_strategy {
    Tier tier;
    std::optional<size_t> preferred_device_id;  // Optional preferred device within tier

    explicit any_memory_space_in_tier_with_preference(Tier t,
                                                      std::optional<size_t> device_id = std::nullopt)
      : reservation_request_strategy(true), tier(t), preferred_device_id(device_id)
    {
    }

    std::vector<memory_space*> get_candidates(memory_reservation_manager& manager) const override;
  };
  ```
  Namespace: `cucascade::memory`. Include: `<cucascade/memory/memory_reservation_manager.hpp>`.

- **Availability confirmed at f47de0b:** Yes. Directly visible in the submodule at the pinned commit (bumped in plan 04-01). `get_candidates(...)` orders candidates so the preferred device_id appears first; unrelated devices in the same tier appear after. Used already by Sirius in `duckdb_scan_executor.cpp:297` for scan-target NUMA-local HOST reservation, so the pattern is proven to link + run on bumped cucascade.

- **Call shape Task 2 will use (re-authoring dd86dd0's intent):**
  Inside `src/downgrade/downgrade_task.cpp::execute`, replace the current `any_memory_space_in_tier{Tier::HOST}` call:
  ```cpp
  auto reservation = [&]() {
    if (preferred_numa_node.has_value()) {
      return res_mgr.request_reservation(
        cucascade::memory::any_memory_space_in_tier_with_preference{
          cucascade::memory::Tier::HOST,
          std::optional<size_t>{static_cast<size_t>(*preferred_numa_node)}},
        data_size);
    }
    return res_mgr.request_reservation(
      cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST}, data_size);
  }();
  ```
  (conditional so the nullopt case runs identical to dev's current behavior — essential for the `downgrade_executor_default_numa_node_is_nullopt` test in Task 4a).

## Validation

All 6 items are present on dev. None diverges materially from research §2's prediction:

- [x] `downgrade_executor_config` exists (`src/include/exec/config.hpp:32`).
- [x] The downgrade-dispatch memory_space selection point is located (`src/downgrade/downgrade_task.cpp:51`).
- [x] SiriusContext's downgrade_executor construction site is located with the Plan 02 TODO marker in place (`src/sirius_context.cpp:222`).
- [x] The `downgrade_request` submission API is characterized with promise/future + MPMC queue semantics.
- [x] Dev's test fixture pattern is captured (`make_test_executor` at line 106; example tests at 149, 166, etc.).
- [x] `cucascade::memory::any_memory_space_in_tier_with_preference` exists at f47de0b (declaration at `cucascade/include/cucascade/memory/memory_reservation_manager.hpp:79`, definition at `cucascade/src/memory/memory_reservation_manager.cpp:65`).

**Ready for Tasks 2-4b to consume.**
