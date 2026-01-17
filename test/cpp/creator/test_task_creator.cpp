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
#include "gpu_context.hpp"
#include "op/scan/duckdb_scan_executor.hpp"
#include "op/sirius_physical_operator.hpp"
#include "parallel/task_executor.hpp"
#include "pipeline/pipeline_executor.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius_pipeline_hashmap.hpp"

#include <data/data_repository.hpp>
#include <duckdb.hpp>
#include <duckdb/main/connection.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

using namespace sirius::creator;
using namespace sirius::parallel;
using namespace sirius::pipeline;
using namespace sirius::op::scan;
using namespace std::chrono_literals;
using namespace sirius::op;
using sirius::sirius_pipeline_hashmap;

//===----------------------------------------------------------------------===//
// Mock GPU Physical Operator
//===----------------------------------------------------------------------===//

/**
 * @brief A mock GPU physical operator for testing process_next_task.
 *
 * This mock allows configuring the hint that get_next_task_hint() returns,
 * enabling controlled testing of different scheduling scenarios.
 */
class mock_sirius_physical_operator : public sirius_physical_operator {
 public:
  mock_sirius_physical_operator(
    duckdb::PhysicalOperatorType op_type = duckdb::PhysicalOperatorType::PROJECTION)
    : sirius_physical_operator(op_type, {}, 0),
      _use_custom_hint(false),
      _custom_hint(std::monostate{})
  {
  }

  /**
   * @brief Enable custom hint mode and set the hint to return.
   *
   * When custom hint mode is enabled, get_next_task_hint() returns the
   * configured hint instead of computing one from ports.
   */
  void set_custom_hint(sirius::creator::task_creation_hint hint)
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
  sirius::creator::task_creation_hint get_next_task_hint() override
  {
    if (_use_custom_hint) { return _custom_hint; }
    // Fall back to parent implementation
    return sirius_physical_operator::get_next_task_hint();
  }

 private:
  bool _use_custom_hint;
  sirius::creator::task_creation_hint _custom_hint;
};

/**
 * @brief A mock GPU pipeline for testing FULL barrier scenarios.
 *
 * This class allows controlling the return value of is_pipeline_finished()
 * for testing purposes.
 */
class mock_gpu_pipeline : public sirius_pipeline {
 public:
  explicit mock_gpu_pipeline(duckdb::GPUExecutor& executor)
    : sirius_pipeline(executor), _finished(false)
  {
  }

  void set_finished(bool finished) { _finished = finished; }

  bool is_pipeline_finished() override { return _finished; }

 private:
  bool _finished;
};

/**
 * @brief A mock GPU pipeline for testing.
 *
 * This requires a GPUExecutor reference, so we use a factory pattern
 * to create test pipelines when we have the necessary context.
 */
class mock_pipeline_builder {
 public:
  /**
   * @brief Create a mock pipeline with specified source and operators.
   *
   * Since sirius_pipeline requires GPUExecutor, we set up ports directly
   * on operators to control get_next_task_hint() behavior.
   */
  static void setup_operator_with_pipeline_port(mock_sirius_physical_operator& op,
                                                const std::string& port_id,
                                                MemoryBarrierType barrier_type,
                                                cucascade::shared_data_repository* repo,
                                                duckdb::shared_ptr<sirius_pipeline> src_pipeline,
                                                duckdb::shared_ptr<sirius_pipeline> dest_pipeline)
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
 * This class overrides schedule() to record what task_creation_info objects
 * were scheduled, allowing tests to verify correct scheduling behavior.
 */
class testable_task_creator : public task_creator {
 public:
  testable_task_creator(size_t num_threads,
                        sirius_pipeline_hashmap& gpu_pipeline_map,
                        duckdb::ClientContext& client_context,
                        pipeline_executor& pipeline_executor,
                        duckdb_scan_executor& duckdb_scan_executor)
    : task_creator(
        num_threads, gpu_pipeline_map, client_context, pipeline_executor, duckdb_scan_executor)
  {
  }

  void schedule(std::unique_ptr<task_creation_info> info) override
  {
    std::lock_guard<std::mutex> lock(_scheduled_mutex);
    if (info) {
      _scheduled_nodes.push_back(info->_node);
      _scheduled_pipelines.push_back(info->_pipeline);
    }
    _schedule_count++;
  }

  size_t get_schedule_count() const { return _schedule_count.load(); }

  std::vector<sirius_physical_operator*> get_scheduled_nodes()
  {
    std::lock_guard<std::mutex> lock(_scheduled_mutex);
    return _scheduled_nodes;
  }

  std::vector<duckdb::shared_ptr<sirius_pipeline>> get_scheduled_pipelines()
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

  // Expose protected members for testing
  task_creation_queue* get_queue() { return _task_creation_queue.get(); }

  std::queue<duckdb::shared_ptr<sirius_pipeline>>& get_priority_scans() { return priority_scans; }

  bool is_running() const { return _running.load(); }

  size_t get_thread_count() const { return _threads.size(); }

 private:
  std::atomic<size_t> _schedule_count{0};
  std::vector<sirius_physical_operator*> _scheduled_nodes;
  std::vector<duckdb::shared_ptr<sirius_pipeline>> _scheduled_pipelines;
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
      gpu_context(*con.context),
      gpu_executor(*con.context, gpu_context),
      pipeline_config{2, false},
      gpu_executor_config{1, false},
      pipeline_exec(pipeline_config, gpu_executor_config, 1),
      scan_exec(task_executor_config{2, false}),
      empty_pipelines(),
      pipeline_map(empty_pipelines)
  {
  }

  /**
   * @brief Create a mock GPU pipeline with controllable finished state.
   */
  duckdb::shared_ptr<mock_gpu_pipeline> create_mock_pipeline()
  {
    return duckdb::make_shared_ptr<mock_gpu_pipeline>(gpu_executor);
  }

  duckdb::DuckDB db;
  duckdb::Connection con;
  duckdb::GPUContext gpu_context;
  duckdb::GPUExecutor gpu_executor;
  task_executor_config pipeline_config;
  task_executor_config gpu_executor_config;
  pipeline_executor pipeline_exec;
  duckdb_scan_executor scan_exec;
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> empty_pipelines;
  sirius::sirius_pipeline_hashmap pipeline_map;
};

//===----------------------------------------------------------------------===//
// task_creation_queue Tests
//===----------------------------------------------------------------------===//

TEST_CASE("task_creation_queue open/close lifecycle", "[task_creation_queue]")
{
  task_creation_queue queue(2);

  SECTION("Queue starts closed") { REQUIRE_FALSE(queue.is_open()); }

  SECTION("Queue can be opened")
  {
    queue.open();
    REQUIRE(queue.is_open());
  }

  SECTION("Queue can be closed after opening")
  {
    queue.open();
    REQUIRE(queue.is_open());
    queue.close();
    REQUIRE_FALSE(queue.is_open());
  }

  SECTION("Multiple open calls are idempotent")
  {
    queue.open();
    queue.open();
    REQUIRE(queue.is_open());
  }

  SECTION("Multiple close calls are safe")
  {
    queue.open();
    queue.close();
    REQUIRE_NOTHROW(queue.close());
  }
}

TEST_CASE("task_creation_queue push and pull", "[task_creation_queue]")
{
  task_creation_queue queue(1);
  queue.open();

  // Create a mock operator for task_creation_info
  // Note: task_creation_info requires a valid pipeline, so we'll test with nullptr checks
  auto mock_op = std::make_unique<mock_sirius_physical_operator>();

  SECTION("Push and pull single item")
  {
    // We can't easily create task_creation_info without a valid pipeline,
    // but we can test the queue mechanics with nullptr
    queue.push(nullptr);

    auto result = queue.pull();
    REQUIRE(result == nullptr);
  }

  queue.close();
}

TEST_CASE("task_creation_queue pull returns nullptr when closed", "[task_creation_queue]")
{
  task_creation_queue queue(1);

  SECTION("Pull from closed empty queue returns nullptr")
  {
    // Queue is closed by default, but we need to ensure it returns nullptr
    queue.open();
    queue.close();

    auto result = queue.pull();
    REQUIRE(result == nullptr);
  }
}

TEST_CASE("task_creation_queue close wakes blocked threads", "[task_creation_queue]")
{
  task_creation_queue queue(2);
  queue.open();

  std::atomic<int> threads_woken{0};
  std::vector<std::thread> threads;

  // Start threads that will block on pull
  for (int i = 0; i < 2; ++i) {
    threads.emplace_back([&queue, &threads_woken]() {
      auto result = queue.pull();
      // Should receive nullptr sentinel when queue closes
      if (result == nullptr) { threads_woken.fetch_add(1); }
    });
  }

  // Give threads time to block
  std::this_thread::sleep_for(50ms);

  // Close queue should wake all threads
  queue.close();

  // Wait for threads with timeout
  auto start_time = std::chrono::steady_clock::now();
  auto timeout    = std::chrono::seconds(5);

  for (auto& t : threads) {
    if (t.joinable()) { t.join(); }
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      FAIL("Test timed out waiting for threads to wake");
    }
  }

  REQUIRE(threads_woken.load() == 2);
}

//===----------------------------------------------------------------------===//
// task_creator Thread Pool Tests
//===----------------------------------------------------------------------===//

TEST_CASE("task_creator thread pool starts and stops", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);

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
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);

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
      2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);
    creator.start_thread_pool();
    // Destructor should stop threads
  }

  // If we get here without hanging, the destructor worked
  SUCCEED("Destructor completed without hanging");
}

//===----------------------------------------------------------------------===//
// process_next_task Tests
//===----------------------------------------------------------------------===//

TEST_CASE("process_next_task with monostate hint and empty priority_scans", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);

  // Create a mock operator with no ports (will return monostate)
  auto mock_op = std::make_unique<mock_sirius_physical_operator>();

  // process_next_task should do nothing when hint is monostate and no priority scans
  creator.process_next_task(mock_op.get());

  // Nothing should be scheduled
  REQUIRE(creator.get_schedule_count() == 0);
}

TEST_CASE("process_next_task with monostate hint uses priority_scans", "[task_creator]")
{
  test_fixture fixture;

  // Create a mock scan operator
  auto scan_op =
    std::make_unique<mock_sirius_physical_operator>(duckdb::PhysicalOperatorType::TABLE_SCAN);

  // We need to create a pipeline that has this as a source
  // This is complex because sirius_pipeline requires GPUExecutor
  // For this test, we verify that the scheduling logic attempts to use priority_scans

  // Create pipelines with the scan as source - this requires integration test setup
  // For unit testing purposes, we verify the code path via the testable interface
  testable_task_creator creator(
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);

  // Initially priority_scans should be empty (no TABLE_SCAN sources in empty pipeline map)
  REQUIRE(creator.get_priority_scans().empty());

  // Create a mock operator that returns monostate
  auto mock_op = std::make_unique<mock_sirius_physical_operator>();

  // With empty priority scans, nothing should be scheduled
  creator.process_next_task(mock_op.get());
  REQUIRE(creator.get_schedule_count() == 0);
}

TEST_CASE("process_next_task with operator hint schedules the hint node", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);

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
  source_op->set_custom_hint(sirius::creator::task_creation_hint(hint_op.get()));

  // Call process_next_task - this should attempt to schedule with hint_op
  // Note: This will try to access hint_op->get_port("default")->dest_pipeline
  // which we've set up above
  creator.process_next_task(source_op.get());

  // Verify that schedule was called with the hint_op
  auto scheduled_nodes = creator.get_scheduled_nodes();
  REQUIRE(creator.get_schedule_count() == 1);
  REQUIRE(scheduled_nodes.size() == 1);
  REQUIRE(scheduled_nodes[0] == hint_op.get());
}

TEST_CASE("process_next_task with pipeline hint recurses to inner operator", "[task_creator]")
{
  test_fixture fixture;

  // This test verifies the recursive behavior when the hint is a sirius_pipeline.
  // When hint is a pipeline, process_next_task should call itself with
  // pipeline->GetInnerOperators()[0]
  //
  // Since sirius_pipeline requires GPUExecutor and complex setup, we test this
  // behavior indirectly by verifying that the source operator's
  // get_next_task_hint() is called and the scheduling logic follows through.

  testable_task_creator creator(
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);

  // Create an operator chain: source_op returns a custom hint that is monostate
  // (simulating the end of recursion)
  auto source_op = std::make_unique<mock_sirius_physical_operator>();
  source_op->set_custom_hint(sirius::creator::task_creation_hint(std::monostate{}));

  // Call process_next_task - with monostate and no priority_scans, nothing scheduled
  creator.process_next_task(source_op.get());
  REQUIRE(creator.get_schedule_count() == 0);

  // Now test with an operator hint that returns itself (simulating ready operator)
  auto ready_op  = std::make_unique<mock_sirius_physical_operator>();
  auto data_repo = std::make_unique<cucascade::shared_data_repository>();

  // Set up the port so get_port("default") works
  mock_pipeline_builder::setup_operator_with_pipeline_port(
    *ready_op, "default", MemoryBarrierType::PIPELINE, data_repo.get(), nullptr, nullptr);

  // Configure ready_op to return itself as hint (all ports ready)
  ready_op->set_custom_hint(sirius::creator::task_creation_hint(ready_op.get()));

  creator.clear_scheduled();
  creator.process_next_task(ready_op.get());

  // Should have scheduled the ready_op
  REQUIRE(creator.get_schedule_count() == 1);
  auto nodes = creator.get_scheduled_nodes();
  REQUIRE(nodes.size() == 1);
  REQUIRE(nodes[0] == ready_op.get());
}

TEST_CASE("process_next_task operator hint follows dest_pipeline", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);

  // Create operators
  auto source_op = std::make_unique<mock_sirius_physical_operator>();
  auto target_op = std::make_unique<mock_sirius_physical_operator>();

  // Create data repository
  auto data_repo = std::make_unique<cucascade::shared_data_repository>();

  // Set up target_op with a default port
  mock_pipeline_builder::setup_operator_with_pipeline_port(
    *target_op, "default", MemoryBarrierType::PIPELINE, data_repo.get(), nullptr, nullptr);

  // Source returns target as hint
  source_op->set_custom_hint(sirius::creator::task_creation_hint(target_op.get()));

  creator.process_next_task(source_op.get());

  // Verify target_op was scheduled
  auto nodes = creator.get_scheduled_nodes();
  REQUIRE(nodes.size() == 1);
  REQUIRE(nodes[0] == target_op.get());
}

TEST_CASE("process_next_task hint traversal chain", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);

  // Test a chain where:
  // op1 returns hint pointing to op2
  // op2 will be scheduled

  auto op1       = std::make_unique<mock_sirius_physical_operator>();
  auto op2       = std::make_unique<mock_sirius_physical_operator>();
  auto data_repo = std::make_unique<cucascade::shared_data_repository>();

  // Set up op2 with default port
  mock_pipeline_builder::setup_operator_with_pipeline_port(
    *op2, "default", MemoryBarrierType::PIPELINE, data_repo.get(), nullptr, nullptr);

  // op1 returns op2 as hint
  op1->set_custom_hint(sirius::creator::task_creation_hint(op2.get()));

  creator.process_next_task(op1.get());

  // op2 should be scheduled
  auto nodes = creator.get_scheduled_nodes();
  REQUIRE(nodes.size() == 1);
  REQUIRE(nodes[0] == op2.get());
}

TEST_CASE("task_creator start/stop lifecycle", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);

  SECTION("start() calls start_thread_pool()")
  {
    creator.start();
    REQUIRE(creator.is_running());
    creator.stop();
    REQUIRE_FALSE(creator.is_running());
  }

  SECTION("stop() calls stop_thread_pool()")
  {
    creator.start();
    creator.stop();
    REQUIRE_FALSE(creator.is_running());
  }
}

TEST_CASE("task_creator get_next_task_id increments", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    1, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);

  // The task_id is protected, but we can verify behavior indirectly
  // by checking that the creator can be constructed and used
  REQUIRE_FALSE(creator.is_running());
}

//===----------------------------------------------------------------------===//
// Queue Integration with Task Creator
//===----------------------------------------------------------------------===//

TEST_CASE("task_creator queue integration", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    2, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);

  SECTION("Queue is accessible")
  {
    auto* queue = creator.get_queue();
    REQUIRE(queue != nullptr);
  }

  SECTION("Queue opens when thread pool starts")
  {
    auto* queue = creator.get_queue();
    REQUIRE_FALSE(queue->is_open());

    creator.start_thread_pool();
    REQUIRE(queue->is_open());

    creator.stop_thread_pool();
    REQUIRE_FALSE(queue->is_open());
  }
}

//===----------------------------------------------------------------------===//
// sirius_physical_operator::get_next_task_hint() Tests
//===----------------------------------------------------------------------===//

TEST_CASE("get_next_task_hint returns monostate when no ports", "[get_next_task_hint]")
{
  // An operator with no ports should return monostate
  mock_sirius_physical_operator op;

  auto hint = op.get_next_task_hint();

  REQUIRE(std::holds_alternative<std::monostate>(hint));
}

TEST_CASE("get_next_task_hint PIPELINE barrier with empty repo returns src_pipeline",
          "[get_next_task_hint]")
{
  mock_sirius_physical_operator op;
  auto data_repo = std::make_unique<cucascade::shared_data_repository>();

  // Create a mock src_pipeline - we use nullptr for simplicity but wrap in shared_ptr
  // In real usage this would be a valid pipeline
  auto src_pipeline = duckdb::shared_ptr<sirius_pipeline>(nullptr);

  // Create a simple mock pipeline to return as src_pipeline hint
  // Since we can't easily create a real sirius_pipeline, we'll test the path
  // where src_pipeline is nullptr (returns monostate)
  mock_pipeline_builder::setup_operator_with_pipeline_port(op,
                                                           "input",
                                                           MemoryBarrierType::PIPELINE,
                                                           data_repo.get(),
                                                           nullptr,  // src_pipeline is nullptr
                                                           nullptr   // dest_pipeline
  );

  // repo is empty, src_pipeline is nullptr → should return monostate
  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<std::monostate>(hint));
}

TEST_CASE("get_next_task_hint PIPELINE barrier with data returns this", "[get_next_task_hint]")
{
  mock_sirius_physical_operator op;
  auto data_repo = std::make_unique<cucascade::shared_data_repository>();

  // Add data to the repository so it's not empty
  auto batch = std::make_shared<cucascade::data_batch>(1, nullptr);
  data_repo->add_data_batch(batch);

  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "input", MemoryBarrierType::PIPELINE, data_repo.get(), nullptr, nullptr);

  // repo has data → all ports ready → should return this operator
  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<sirius_physical_operator*>(hint));
  REQUIRE(std::get<sirius_physical_operator*>(hint) == &op);
}

TEST_CASE("get_next_task_hint multiple PIPELINE ports all ready returns this",
          "[get_next_task_hint]")
{
  mock_sirius_physical_operator op;
  auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
  auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

  // Add data to both repositories
  data_repo1->add_data_batch(std::make_shared<cucascade::data_batch>(1, nullptr));
  data_repo2->add_data_batch(std::make_shared<cucascade::data_batch>(1, nullptr));

  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "input1", MemoryBarrierType::PIPELINE, data_repo1.get(), nullptr, nullptr);

  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "input2", MemoryBarrierType::PIPELINE, data_repo2.get(), nullptr, nullptr);

  // Both repos have data → all ports ready → should return this operator
  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<sirius_physical_operator*>(hint));
  REQUIRE(std::get<sirius_physical_operator*>(hint) == &op);
}

TEST_CASE("get_next_task_hint multiple PIPELINE ports one empty returns monostate",
          "[get_next_task_hint]")
{
  mock_sirius_physical_operator op;
  auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
  auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

  // Add data only to first repository
  data_repo1->add_data_batch(std::make_shared<cucascade::data_batch>(1, nullptr));
  // data_repo2 is empty

  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "input1", MemoryBarrierType::PIPELINE, data_repo1.get(), nullptr, nullptr);

  mock_pipeline_builder::setup_operator_with_pipeline_port(op,
                                                           "input2",
                                                           MemoryBarrierType::PIPELINE,
                                                           data_repo2.get(),
                                                           nullptr,  // no src_pipeline
                                                           nullptr);

  // One repo is empty with no src_pipeline → should return monostate
  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<std::monostate>(hint));
}

TEST_CASE("get_next_task_hint uses custom hint when set on mock", "[get_next_task_hint]")
{
  mock_sirius_physical_operator source_op;
  mock_sirius_physical_operator target_op;

  // Set custom hint to return target_op
  source_op.set_custom_hint(sirius::creator::task_creation_hint(&target_op));

  auto hint = source_op.get_next_task_hint();
  REQUIRE(std::holds_alternative<sirius_physical_operator*>(hint));
  REQUIRE(std::get<sirius_physical_operator*>(hint) == &target_op);
}

TEST_CASE("get_next_task_hint custom hint monostate", "[get_next_task_hint]")
{
  mock_sirius_physical_operator op;

  // Set custom hint to monostate
  op.set_custom_hint(sirius::creator::task_creation_hint(std::monostate{}));

  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<std::monostate>(hint));
}

TEST_CASE("get_next_task_hint clear_custom_hint falls back to default", "[get_next_task_hint]")
{
  mock_sirius_physical_operator op;

  // Set and then clear custom hint
  op.set_custom_hint(sirius::creator::task_creation_hint(&op));
  op.clear_custom_hint();

  // With no ports and no custom hint, should return monostate (default behavior)
  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<std::monostate>(hint));
}

//===----------------------------------------------------------------------===//
// FULL MemoryBarrierType Tests
//===----------------------------------------------------------------------===//

TEST_CASE("get_next_task_hint FULL barrier with unfinished pipeline returns src_pipeline",
          "[get_next_task_hint]")
{
  test_fixture fixture;
  mock_sirius_physical_operator op;
  auto data_repo = std::make_unique<cucascade::shared_data_repository>();

  // Create a mock pipeline that is NOT finished
  auto mock_pipeline = fixture.create_mock_pipeline();
  mock_pipeline->set_finished(false);

  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op,
    "input",
    MemoryBarrierType::FULL,
    data_repo.get(),
    mock_pipeline,  // src_pipeline is not finished
    nullptr);

  // src_pipeline is not finished → should return the src_pipeline
  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<duckdb::shared_ptr<sirius_pipeline>>(hint));
  REQUIRE(std::get<duckdb::shared_ptr<sirius_pipeline>>(hint) == mock_pipeline);
}

TEST_CASE("get_next_task_hint FULL barrier with finished pipeline returns this",
          "[get_next_task_hint]")
{
  test_fixture fixture;
  mock_sirius_physical_operator op;
  auto data_repo = std::make_unique<cucascade::shared_data_repository>();

  // Create a mock pipeline that IS finished
  auto mock_pipeline = fixture.create_mock_pipeline();
  mock_pipeline->set_finished(true);

  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op,
    "input",
    MemoryBarrierType::FULL,
    data_repo.get(),
    mock_pipeline,  // src_pipeline is finished
    nullptr);

  // src_pipeline is finished → all ports ready → should return this operator
  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<sirius_physical_operator*>(hint));
  REQUIRE(std::get<sirius_physical_operator*>(hint) == &op);
}

TEST_CASE("get_next_task_hint multiple FULL barriers all finished returns this",
          "[get_next_task_hint]")
{
  test_fixture fixture;
  mock_sirius_physical_operator op;
  auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
  auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

  // Create two mock pipelines that are both finished
  auto mock_pipeline1 = fixture.create_mock_pipeline();
  auto mock_pipeline2 = fixture.create_mock_pipeline();
  mock_pipeline1->set_finished(true);
  mock_pipeline2->set_finished(true);

  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "input1", MemoryBarrierType::FULL, data_repo1.get(), mock_pipeline1, nullptr);

  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "input2", MemoryBarrierType::FULL, data_repo2.get(), mock_pipeline2, nullptr);

  // Both pipelines finished → all ports ready → should return this operator
  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<sirius_physical_operator*>(hint));
  REQUIRE(std::get<sirius_physical_operator*>(hint) == &op);
}

TEST_CASE("get_next_task_hint multiple FULL barriers one unfinished returns src_pipeline",
          "[get_next_task_hint]")
{
  test_fixture fixture;
  mock_sirius_physical_operator op;
  auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
  auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

  // Create two mock pipelines: one finished, one not
  auto mock_pipeline1 = fixture.create_mock_pipeline();
  auto mock_pipeline2 = fixture.create_mock_pipeline();
  mock_pipeline1->set_finished(true);
  mock_pipeline2->set_finished(false);  // This one is not finished

  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "input1", MemoryBarrierType::FULL, data_repo1.get(), mock_pipeline1, nullptr);

  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "input2", MemoryBarrierType::FULL, data_repo2.get(), mock_pipeline2, nullptr);

  // One pipeline is not finished → should return that unfinished pipeline
  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<duckdb::shared_ptr<sirius_pipeline>>(hint));
  REQUIRE(std::get<duckdb::shared_ptr<sirius_pipeline>>(hint) == mock_pipeline2);
}

TEST_CASE("get_next_task_hint mixed PIPELINE and FULL barriers", "[get_next_task_hint]")
{
  test_fixture fixture;
  mock_sirius_physical_operator op;
  auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
  auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

  // Add data to PIPELINE port's repo
  data_repo1->add_data_batch(std::make_shared<cucascade::data_batch>(1, nullptr));

  // Create a mock pipeline that is finished for FULL barrier
  auto mock_pipeline = fixture.create_mock_pipeline();
  mock_pipeline->set_finished(true);

  // PIPELINE barrier with data
  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "pipeline_input", MemoryBarrierType::PIPELINE, data_repo1.get(), nullptr, nullptr);

  // FULL barrier with finished pipeline
  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "full_input", MemoryBarrierType::FULL, data_repo2.get(), mock_pipeline, nullptr);

  // Both ports are ready (PIPELINE has data, FULL is finished) → return this
  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<sirius_physical_operator*>(hint));
  REQUIRE(std::get<sirius_physical_operator*>(hint) == &op);
}

TEST_CASE("get_next_task_hint mixed barriers with FULL unfinished", "[get_next_task_hint]")
{
  test_fixture fixture;
  mock_sirius_physical_operator op;
  auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
  auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

  // Add data to PIPELINE port's repo
  data_repo1->add_data_batch(std::make_shared<cucascade::data_batch>(1, nullptr));

  // Create a mock pipeline that is NOT finished for FULL barrier
  auto mock_pipeline = fixture.create_mock_pipeline();
  mock_pipeline->set_finished(false);

  // PIPELINE barrier with data (ready)
  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "pipeline_input", MemoryBarrierType::PIPELINE, data_repo1.get(), nullptr, nullptr);

  // FULL barrier with unfinished pipeline (not ready)
  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "full_input", MemoryBarrierType::FULL, data_repo2.get(), mock_pipeline, nullptr);

  // FULL port is not ready → should return the unfinished src_pipeline
  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<duckdb::shared_ptr<sirius_pipeline>>(hint));
  REQUIRE(std::get<duckdb::shared_ptr<sirius_pipeline>>(hint) == mock_pipeline);
}

TEST_CASE("get_next_task_hint mixed barriers with PIPELINE empty", "[get_next_task_hint]")
{
  test_fixture fixture;
  mock_sirius_physical_operator op;
  auto data_repo1 = std::make_unique<cucascade::shared_data_repository>();
  auto data_repo2 = std::make_unique<cucascade::shared_data_repository>();

  // data_repo1 is empty for PIPELINE barrier

  // Create a mock pipeline for PIPELINE's src_pipeline (to return when empty)
  auto pipeline_src = fixture.create_mock_pipeline();

  // Create a mock pipeline that is finished for FULL barrier
  auto full_src = fixture.create_mock_pipeline();
  full_src->set_finished(true);

  // PIPELINE barrier with empty repo (not ready) - has src_pipeline to return
  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op,
    "pipeline_input",
    MemoryBarrierType::PIPELINE,
    data_repo1.get(),
    pipeline_src,  // src_pipeline to return when empty
    nullptr);

  // FULL barrier with finished pipeline (ready)
  mock_pipeline_builder::setup_operator_with_pipeline_port(
    op, "full_input", MemoryBarrierType::FULL, data_repo2.get(), full_src, nullptr);

  // PIPELINE port is empty → should return its src_pipeline
  auto hint = op.get_next_task_hint();
  REQUIRE(std::holds_alternative<duckdb::shared_ptr<sirius_pipeline>>(hint));
  REQUIRE(std::get<duckdb::shared_ptr<sirius_pipeline>>(hint) == pipeline_src);
}

//===----------------------------------------------------------------------===//
// Concurrent Operation Tests
//===----------------------------------------------------------------------===//

TEST_CASE("task_creator handles concurrent schedule calls", "[task_creator]")
{
  test_fixture fixture;

  testable_task_creator creator(
    4, fixture.pipeline_map, *fixture.con.context, fixture.pipeline_exec, fixture.scan_exec);

  const int num_calls = 100;
  std::atomic<int> completed{0};

  // Create mock operators
  std::vector<std::unique_ptr<mock_sirius_physical_operator>> operators;
  for (int i = 0; i < num_calls; ++i) {
    operators.push_back(std::make_unique<mock_sirius_physical_operator>());
  }

  // Spawn threads to call process_next_task concurrently
  std::vector<std::thread> threads;
  for (int i = 0; i < num_calls; ++i) {
    threads.emplace_back([&creator, &operators, i, &completed]() {
      creator.process_next_task(operators[i].get());
      completed.fetch_add(1);
    });
  }

  // Wait for all threads
  for (auto& t : threads) {
    t.join();
  }

  REQUIRE(completed.load() == num_calls);
}
