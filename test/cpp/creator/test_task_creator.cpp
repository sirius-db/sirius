/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "catch.hpp"
#include "creator/task_creator.hpp"
#include "exec/config.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/task_scheduler.hpp"
#include "utils/telemetry_utils.hpp"

#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/main/database.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

using namespace sirius::creator;
using namespace sirius::exec;
using namespace sirius::parallel;
using namespace sirius::pipeline;
using namespace sirius::op::scan;
using namespace std::chrono_literals;
using namespace sirius::op;
using namespace sirius;

//===----------------------------------------------------------------------===//
// Mock GPU Physical Operator
//===----------------------------------------------------------------------===//

/**
 * @brief A mock GPU physical operator for testing get_operator_for_next_task.
 *
 * This mock allows configuring the hint that get_next_task_hint() returns,
 * enabling controlled testing of different scheduling scenarios.
 */
class mock_sirius_physical_operator : public sirius_physical_operator {
 public:
  mock_sirius_physical_operator(
    SiriusPhysicalOperatorType op_type = SiriusPhysicalOperatorType::PROJECTION)
    : sirius_physical_operator(op_type, {}, 0), _use_custom_hint(false), _custom_hint(std::nullopt)
  {
  }

  /**
   * @brief Enable custom hint mode and set the hint to return.
   *
   * When custom hint mode is enabled, get_next_task_hint() returns the
   * configured hint instead of computing one from ports.
   */
  void set_custom_hint(std::optional<sirius::op::task_creation_hint> hint)
  {
    _use_custom_hint = true;
    _custom_hint     = std::move(hint);
  }

  /**
   * @brief Disable custom hint mode (use default port-based behavior).
   */
  void clear_custom_hint() { _use_custom_hint = false; }

  /**
   * @brief Override to return configured hint when in custom mode.
   */
  std::optional<task_creation_hint> get_next_task_hint() override
  {
    if (_use_custom_hint) { return _custom_hint; }
    // Fall back to parent implementation
    return sirius_physical_operator::get_next_task_hint();
  }

 private:
  bool _use_custom_hint;
  std::optional<sirius::op::task_creation_hint> _custom_hint;
};

/**
 * @brief A mock GPU pipeline for testing FULL barrier scenarios.
 *
 * This class allows controlling the return value of is_pipeline_finished()
 * for testing purposes.
 */
class mock_gpu_pipeline : public sirius_pipeline {
 public:
  explicit mock_gpu_pipeline(const pipeline::pipeline_build_context& ctx)
    : sirius_pipeline(ctx), _finished(false)
  {
  }

  void set_finished(bool finished) { _finished = finished; }

  bool is_pipeline_finished() const override { return _finished; }

 private:
  bool _finished;
};

/**
 * @brief A mock GPU pipeline for testing.
 *
 * This sets up ports directly on operators to control get_next_task_hint()
 * behavior.
 */
class mock_pipeline_builder {
 public:
  /**
   * @brief Create a mock pipeline with specified source and operators.
   *
   * The tests only need operator ports, not fully converted pipelines.
   */
  static void setup_operator_with_pipeline_port(mock_sirius_physical_operator& op,
                                                const std::string& port_id,
                                                MemoryBarrierType barrier_type,
                                                cucascade::shared_data_repository* repo,
                                                std::shared_ptr<sirius_pipeline> src_pipeline,
                                                std::shared_ptr<sirius_pipeline> dest_pipeline)
  {
    auto port           = std::make_unique<sirius_physical_operator::port>();
    port->type          = barrier_type;
    port->repo          = repo;
    port->src_pipeline  = src_pipeline;
    port->dest_pipeline = dest_pipeline;
    op.add_port(port_id, std::move(port));
  }
};

//===----------------------------------------------------------------------===//
// Testable Task Creator
//===----------------------------------------------------------------------===//

/**
 * @brief A testable subclass of task_creator that tracks scheduled tasks.
 *
 * This class overrides schedule() to record what operators objects
 * were scheduled, allowing tests to verify correct scheduling behavior.
 */
class testable_task_creator : public task_creator {
 public:
  testable_task_creator(int num_threads,
                        duckdb::ClientContext& client_context,
                        task_scheduler& task_sched,
                        sirius::memory::sirius_memory_reservation_manager& mem_res_mgr,
                        sirius::query_id_t query_id = sirius::make_query_id(1))
    : task_creator(
        creator::task_creator_config{
          .thread_pool = {.num_threads = num_threads, .thread_name_prefix = "task_creator"}},
        mem_res_mgr),
      _query_id(query_id)
  {
    // Binding a client context is what registers the query's state.
    this->set_client_context(query_id, client_context);
    this->set_task_scheduler(task_sched);
  }

  [[nodiscard]] sirius::query_id_t query_id() const noexcept { return _query_id; }

  void schedule(op::sirius_physical_operator* request) override
  {
    std::lock_guard<std::mutex> lock(_scheduled_mutex);
    if (request) {
      _scheduled_nodes.push_back(request);
      _scheduled_pipelines.push_back(request->get_pipeline());
    }
    _schedule_count++;
  }

  size_t get_schedule_count() const { return _schedule_count.load(); }

  std::vector<sirius_physical_operator*> get_scheduled_nodes()
  {
    std::lock_guard<std::mutex> lock(_scheduled_mutex);
    return _scheduled_nodes;
  }

  std::vector<std::shared_ptr<sirius_pipeline>> get_scheduled_pipelines()
  {
    std::lock_guard<std::mutex> lock(_scheduled_mutex);
    return _scheduled_pipelines;
  }

  void clear_scheduled()
  {
    std::lock_guard<std::mutex> lock(_scheduled_mutex);
    _scheduled_nodes.clear();
    _scheduled_pipelines.clear();
    _schedule_count.store(0);
  }

  [[nodiscard]] bool is_running() const { return _running.load(); }

  // Expose protected method for testing
  using task_creator::get_operator_for_next_task;

 private:
  std::atomic<size_t> _schedule_count{0};
  std::vector<sirius_physical_operator*> _scheduled_nodes;
  std::vector<std::shared_ptr<sirius_pipeline>> _scheduled_pipelines;
  sirius::query_id_t _query_id;
  std::mutex _scheduled_mutex;
};

//===----------------------------------------------------------------------===//
// Test Fixture Helper
//===----------------------------------------------------------------------===//

/**
 * @brief Helper class to set up minimal test infrastructure.
 */
class test_fixture {
 public:
  test_fixture()
    : db(nullptr),
      con(db),
      memory_manager([] {
        cucascade::memory::reservation_manager_configurator builder;
        const size_t gpu_capacity  = 2ull << 27;
        const double limit_ratio   = 0.75;
        const size_t host_capacity = 4ull << 27;

        builder.set_number_of_gpus(1)
          .set_gpu_usage_limit(gpu_capacity)
          .set_reservation_fraction_per_gpu(limit_ratio)
          .set_per_numa_region_capacity(host_capacity)
          .use_gpu_id_as_host_id()
          .set_reservation_fraction_per_numa_region(limit_ratio);

        // Build configuration with topology detection
        auto space_configs = builder.build();
        return std::make_unique<sirius::memory::sirius_memory_reservation_manager>(
          std::move(space_configs));
      }()),
      pipeline_exec(exec::thread_pool_config{.num_threads = 1},
                    *memory_manager,
                    sirius::test::make_test_telemetry_context()),
      empty_pipelines()
  {
  }

  /**
   * @brief Create a mock GPU pipeline with controllable finished state.
   */
  std::shared_ptr<mock_gpu_pipeline> create_mock_pipeline()
  {
    return std::make_shared<mock_gpu_pipeline>(build_ctx);
  }

  duckdb::DuckDB db;
  duckdb::Connection con;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory_manager;
  pipeline::pipeline_build_context build_ctx{nullptr, true};
  task_scheduler pipeline_exec;
  std::vector<std::shared_ptr<sirius_pipeline>> empty_pipelines;
};

//===----------------------------------------------------------------------===//
// task_creator Thread Pool Tests
//===----------------------------------------------------------------------===//

TEST_CASE("task_creator thread pool starts and stops", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, *fixture.con.context, fixture.pipeline_exec, *fixture.memory_manager);

  SECTION("Creator starts not running") { REQUIRE_FALSE(creator.is_running()); }

  SECTION("start_thread_pool creates threads")
  {
    creator.start_thread_pool();
    REQUIRE(creator.is_running());

    // Give threads time to start
    std::this_thread::sleep_for(10ms);

    creator.stop_thread_pool();
    REQUIRE_FALSE(creator.is_running());
  }

  SECTION("stop_thread_pool joins threads gracefully")
  {
    creator.start_thread_pool();

    // Stop should complete without hanging
    auto start_time = std::chrono::steady_clock::now();
    creator.stop_thread_pool();
    auto duration = std::chrono::steady_clock::now() - start_time;

    REQUIRE(duration < std::chrono::seconds(5));
    REQUIRE_FALSE(creator.is_running());
  }
}

TEST_CASE("task_creator thread pool is idempotent", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, *fixture.con.context, fixture.pipeline_exec, *fixture.memory_manager);

  SECTION("Multiple start_thread_pool calls don't create extra threads")
  {
    creator.start_thread_pool();
    creator.start_thread_pool();
    creator.start_thread_pool();

    REQUIRE(creator.is_running());

    creator.stop_thread_pool();
  }

  SECTION("Multiple stop_thread_pool calls don't crash")
  {
    creator.start_thread_pool();

    REQUIRE_NOTHROW(creator.stop_thread_pool());
    REQUIRE_NOTHROW(creator.stop_thread_pool());
    REQUIRE_NOTHROW(creator.stop_thread_pool());
  }

  SECTION("Can restart after stop")
  {
    creator.start_thread_pool();
    creator.stop_thread_pool();
    REQUIRE_FALSE(creator.is_running());

    creator.start_thread_pool();
    REQUIRE(creator.is_running());

    creator.stop_thread_pool();
  }
}

TEST_CASE("task_creator destructor stops thread pool", "[task_creator]")
{
  test_fixture fixture;

  {
    testable_task_creator creator(
      2, *fixture.con.context, fixture.pipeline_exec, *fixture.memory_manager);
    creator.start_thread_pool();
    // Destructor should stop threads
  }

  // If we get here without hanging, the destructor worked
  SUCCEED("Destructor completed without hanging");
}

//===----------------------------------------------------------------------===//
// get_operator_for_next_task Tests
//===----------------------------------------------------------------------===//

TEST_CASE("get_operator_for_next_task records every pipeline the hint walk visits",
          "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, *fixture.con.context, fixture.pipeline_exec, *fixture.memory_manager);

  auto pipeline_a = std::make_shared<mock_gpu_pipeline>(fixture.build_ctx);
  auto pipeline_b = std::make_shared<mock_gpu_pipeline>(fixture.build_ctx);

  // Upstream operator whose hint sweep finds nothing to do — the case where
  // get_next_task_hint() may have just drained its ports.
  auto op_b = std::make_unique<mock_sirius_physical_operator>();
  op_b->set_pipeline(pipeline_b);
  op_b->set_custom_hint(std::nullopt);

  auto op_a = std::make_unique<mock_sirius_physical_operator>();
  op_a->set_pipeline(pipeline_a);
  op_a->set_custom_hint(
    task_creation_hint{.hint = TaskCreationHint::WAITING_FOR_INPUT_DATA, .producer = op_b.get()});

  std::vector<std::shared_ptr<sirius_pipeline>> visited;
  auto* next = creator.get_operator_for_next_task(op_a.get(), visited);

  REQUIRE(next == nullptr);
  // The caller re-evaluates every visited pipeline on the nullptr path. If the
  // walk reported only the requesting pipeline, an upstream pipeline whose
  // tasks all completed earlier would never be marked finished — no later
  // mark_task_completed() exists to re-evaluate it — and its consumers would
  // wait forever.
  REQUIRE(std::find(visited.begin(), visited.end(), pipeline_a) != visited.end());
  REQUIRE(std::find(visited.begin(), visited.end(), pipeline_b) != visited.end());
}

TEST_CASE("get_operator_for_next_task with monostate hint and empty priority_scans",
          "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, *fixture.con.context, fixture.pipeline_exec, *fixture.memory_manager);

  // Create a mock operator with no ports (will return monostate)
  auto mock_op = std::make_unique<mock_sirius_physical_operator>();

  // process_next_task should just return nullptr because there its not really connected to anything
  // and has no data
  std::vector<std::shared_ptr<sirius_pipeline>> visited;
  auto next_op = creator.get_operator_for_next_task(mock_op.get(), visited);

  // Nothing should be scheduled
  REQUIRE(next_op == nullptr);
}

TEST_CASE("get_operator_for_next_task for operator with data returns the operator",
          "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, *fixture.con.context, fixture.pipeline_exec, *fixture.memory_manager);

  // Create the source operator that we will call process_next_task on
  auto source_op = std::make_unique<mock_sirius_physical_operator>();

  // Create the hint operator that should be scheduled
  auto hint_op = std::make_unique<mock_sirius_physical_operator>();

  // Create a data repository for the port
  auto data_repo = std::make_unique<cucascade::shared_data_repository>();

  // Set up the hint operator with a "default" port that has a dest_pipeline
  // For this test, we'll set the dest_pipeline to nullptr since we're testing
  // through the testable_task_creator which captures what gets scheduled
  mock_pipeline_builder::setup_operator_with_pipeline_port(
    *hint_op,
    "default",
    MemoryBarrierType::PIPELINE,
    data_repo.get(),
    nullptr,  // src_pipeline
    nullptr   // dest_pipeline - will be captured by schedule()
  );

  // Configure source_op to return hint_op as the hint
  source_op->set_custom_hint(
    task_creation_hint{.hint = TaskCreationHint::READY, .producer = hint_op.get()});

  // Following source_op's READY hint must land on hint_op.
  std::vector<std::shared_ptr<sirius_pipeline>> visited;
  auto next_op = creator.get_operator_for_next_task(source_op.get(), visited);

  REQUIRE(next_op == hint_op.get());
}
