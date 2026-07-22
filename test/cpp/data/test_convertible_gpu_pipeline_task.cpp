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
#include "operator/operator_test_utils.hpp"

#include <rmm/cuda_stream.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/convertible_gpu_pipeline_task.hpp>
#include <exec/inspectable_priority_queue.hpp>
#include <op/sirius_physical_operator.hpp>
#include <parallel/task.hpp>
#include <pipeline/gpu_pipeline_task.hpp>
#include <pipeline/sirius_pipeline_task_states.hpp>
#include <utils/telemetry_utils.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

// Shared test environment: initialize memory manager once for all tests in this file.
// Uses rmm::cuda_stream (a real non-default CUDA stream) because the cuCascade
// converter uses cudaMemcpyBatchAsync which requires a non-default stream.
struct test_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;
  rmm::cuda_stream conv_stream;

  test_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0)),
      conv_stream()
  {
  }

  rmm::cuda_stream_view stream() { return conv_stream.view(); }
};

test_env& env()
{
  static test_env e;
  return e;
}

/// Minimal dummy task that is NOT a gpu_pipeline_task — used for predicate filtering tests.
class dummy_task_local_state : public sirius::parallel::itask_local_state {};
class dummy_task : public sirius::parallel::itask {
 public:
  dummy_task() : itask(0, std::make_unique<dummy_task_local_state>(), nullptr) {}
  void execute(rmm::cuda_stream_view /*stream*/) override {}
};

/// Helper: create a gpu_pipeline_task with the given data batches.
/// Batches remain in idle state (the new cucascade API default).
std::unique_ptr<sirius::pipeline::gpu_pipeline_task> make_test_gpu_task(
  uint64_t task_id, std::vector<std::shared_ptr<cucascade::data_batch>> batches)
{
  auto op_data = std::make_unique<sirius::op::pipelineable_operator_data>(std::move(batches));
  auto local =
    std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(op_data));
  auto global = std::make_shared<sirius::pipeline::gpu_pipeline_task_global_state>(
    nullptr, sirius::test::make_test_telemetry_context());

  return std::make_unique<sirius::pipeline::gpu_pipeline_task>(
    task_id,
    std::vector<cucascade::shared_data_repository*>{},
    std::move(local),
    std::move(global));
}

/// Helper: get the tier of a data_batch using a temporary read-only lock.
inline cucascade::memory::Tier get_batch_tier(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_tier();
}

/// Helper: get the size in bytes of a data_batch's data using a temporary read-only lock.
inline size_t get_batch_size(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_data()->get_size_in_bytes();
}

}  // anonymous namespace

TEST_CASE("RAII returns task to queue on destruction", "[convertible_gpu_pipeline_task]")
{
  auto& e = env();

  sirius::exec::inspectable_priority_queue<sirius::parallel::itask> queue;
  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);
  auto task = make_test_gpu_task(1, {batch});
  REQUIRE(queue.push(std::move(task)));
  REQUIRE(queue.size() == 1);

  {
    sirius::convertible_gpu_pipeline_task_provider provider(queue);
    auto cd = provider.get_next_convertible(e.gpu_space, false);
    REQUIRE(cd != nullptr);
    REQUIRE(queue.is_empty());
    // cd destroyed here, task returned to queue via RAII
  }

  REQUIRE(queue.size() == 1);
}

TEST_CASE("RAII returns task after successful convert", "[convertible_gpu_pipeline_task]")
{
  auto& e = env();

  sirius::exec::inspectable_priority_queue<sirius::parallel::itask> queue;
  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{10, 20, 30}, cudf::type_id::INT32);
  auto task = make_test_gpu_task(1, {batch});
  REQUIRE(queue.push(std::move(task)));

  {
    sirius::convertible_gpu_pipeline_task_provider provider(queue);
    auto cd = provider.get_next_convertible(e.gpu_space, false);
    REQUIRE(cd != nullptr);

    auto result = cd->convert({e.host_space}, e.stream(), *e.mgr, true);
    REQUIRE(result.has_value());
    // cd destroyed here, task returned to queue via RAII
  }

  REQUIRE(queue.size() == 1);

  // Verify the batch is now in HOST tier
  auto returned_task = queue.try_pop();
  REQUIRE(returned_task != nullptr);
  auto* gpt = dynamic_cast<sirius::pipeline::gpu_pipeline_task*>(returned_task.get());
  REQUIRE(gpt != nullptr);
  auto* ls = dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(gpt->local_state());
  REQUIRE(ls != nullptr);
  auto* pod = dynamic_cast<sirius::op::pipelineable_operator_data*>(ls->_input_data.get());
  REQUIRE(pod != nullptr);
  REQUIRE(pod->get_data_batches().size() == 1);
  REQUIRE(get_batch_tier(*pod->get_data_batches()[0]) == cucascade::memory::Tier::HOST);
}

TEST_CASE("RAII returns task on exception", "[convertible_gpu_pipeline_task]")
{
  auto& e = env();

  sirius::exec::inspectable_priority_queue<sirius::parallel::itask> queue;
  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{4, 5, 6}, cudf::type_id::INT32);
  auto task = make_test_gpu_task(1, {batch});
  REQUIRE(queue.push(std::move(task)));

  {
    sirius::convertible_gpu_pipeline_task_provider provider(queue);
    auto cd = provider.get_next_convertible(e.gpu_space, false);
    REQUIRE(cd != nullptr);
    REQUIRE(queue.is_empty());

    try {
      throw std::runtime_error("test exception");
    } catch (...) {
      // Exception caught, wrapper still alive
    }
    // cd destroyed normally here, task returned to queue
  }

  REQUIRE(queue.size() == 1);
}

TEST_CASE("non-gpu_pipeline_task skipped by predicate", "[convertible_gpu_pipeline_task]")
{
  auto& e = env();

  sirius::exec::inspectable_priority_queue<sirius::parallel::itask> queue;
  REQUIRE(queue.push(std::make_unique<dummy_task>()));
  REQUIRE(queue.size() == 1);

  sirius::convertible_gpu_pipeline_task_provider provider(queue);
  auto cd = provider.get_next_convertible(e.gpu_space, false);
  REQUIRE(cd == nullptr);
  REQUIRE(queue.size() == 1);  // dummy task still in queue
}

TEST_CASE("wrong memory_space skipped by predicate", "[convertible_gpu_pipeline_task]")
{
  auto& e = env();

  sirius::exec::inspectable_priority_queue<sirius::parallel::itask> queue;
  // Create GPU task with batch in gpu_space
  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);
  auto task = make_test_gpu_task(1, {batch});
  REQUIRE(queue.push(std::move(task)));

  sirius::convertible_gpu_pipeline_task_provider provider(queue);
  // Search for tasks in host_space — should find nothing (batch is in gpu_space)
  auto cd = provider.get_next_convertible(e.host_space, false);
  REQUIRE(cd == nullptr);
  REQUIRE(queue.size() == 1);
}

TEST_CASE("non-idle batch skipped by predicate", "[convertible_gpu_pipeline_task]")
{
  auto& e = env();

  sirius::exec::inspectable_priority_queue<sirius::parallel::itask> queue;
  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{7, 8, 9}, cudf::type_id::INT32);

  auto task = make_test_gpu_task(1, {batch});
  REQUIRE(queue.push(std::move(task)));

  // Hold an exclusive lock so batch is not idle — provider should skip it
  auto exclusive_lock = batch->to_mutable();

  sirius::convertible_gpu_pipeline_task_provider provider(queue);
  auto cd = provider.get_next_convertible(e.gpu_space, false);
  REQUIRE(cd == nullptr);
  REQUIRE(queue.size() == 1);

  // Release lock
  {
    auto discard = std::move(exclusive_lock);
  }
}

TEST_CASE("matching task selected by predicate", "[convertible_gpu_pipeline_task]")
{
  auto& e = env();

  sirius::exec::inspectable_priority_queue<sirius::parallel::itask> queue;
  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);
  auto task = make_test_gpu_task(1, {batch});  // batch is idle
  REQUIRE(queue.push(std::move(task)));

  sirius::convertible_gpu_pipeline_task_provider provider(queue);
  auto cd = provider.get_next_convertible(e.gpu_space, false);
  REQUIRE(cd != nullptr);
  REQUIRE(queue.is_empty());
}

TEST_CASE("convert GPU task to HOST", "[convertible_gpu_pipeline_task]")
{
  auto& e = env();

  sirius::exec::inspectable_priority_queue<sirius::parallel::itask> queue;
  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{100, 200, 300}, cudf::type_id::INT32);
  auto task = make_test_gpu_task(1, {batch});
  REQUIRE(queue.push(std::move(task)));

  sirius::convertible_gpu_pipeline_task_provider provider(queue);
  auto cd = provider.get_next_convertible(e.gpu_space, false);
  REQUIRE(cd != nullptr);

  auto result = cd->convert({e.host_space}, e.stream(), *e.mgr, true);
  REQUIRE(result.has_value());
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);
}

TEST_CASE("bytes_in_space returns correct size", "[convertible_gpu_pipeline_task]")
{
  auto& e = env();

  sirius::exec::inspectable_priority_queue<sirius::parallel::itask> queue;
  auto batch1 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);
  auto batch2 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{4, 5, 6, 7, 8}, cudf::type_id::INT32);

  auto batch1_size = get_batch_size(*batch1);
  auto batch2_size = get_batch_size(*batch2);

  auto task = make_test_gpu_task(1, {batch1, batch2});
  REQUIRE(queue.push(std::move(task)));

  sirius::convertible_gpu_pipeline_task_provider provider(queue);
  auto cd = provider.get_next_convertible(e.gpu_space, false);
  REQUIRE(cd != nullptr);

  REQUIRE(cd->bytes_in_space(e.gpu_space) == batch1_size + batch2_size);
  REQUIRE(cd->bytes_in_space(e.host_space) == 0);
}

TEST_CASE("get_all_convertible extracts all matching tasks", "[convertible_gpu_pipeline_task]")
{
  auto& e = env();

  sirius::exec::inspectable_priority_queue<sirius::parallel::itask> queue;

  for (uint64_t i = 0; i < 3; ++i) {
    auto batch = sirius::test::operator_utils::make_numeric_batch(
      *e.gpu_space, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);
    auto task = make_test_gpu_task(i + 1, {batch});
    REQUIRE(queue.push(std::move(task)));
  }
  REQUIRE(queue.size() == 3);

  sirius::convertible_gpu_pipeline_task_provider provider(queue);
  auto all = provider.get_all_convertible(e.gpu_space, false);
  REQUIRE(all.size() == 3);
  REQUIRE(queue.is_empty());

  // Destroy all wrappers — tasks should return to queue
  all.clear();
  REQUIRE(queue.size() == 3);
}

TEST_CASE("RAII on interrupted queue does not crash", "[convertible_gpu_pipeline_task]")
{
  auto& e = env();

  sirius::exec::inspectable_priority_queue<sirius::parallel::itask> queue;
  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);
  auto task = make_test_gpu_task(1, {batch});
  REQUIRE(queue.push(std::move(task)));

  {
    sirius::convertible_gpu_pipeline_task_provider provider(queue);
    auto cd = provider.get_next_convertible(e.gpu_space, false);
    REQUIRE(cd != nullptr);
    REQUIRE(queue.is_empty());

    // Interrupt the queue before destroying the wrapper
    queue.interrupt();
    // cd destroyed here — push() returns false, task is destroyed, SIRIUS_LOG_WARN fires
  }

  // Queue is interrupted and empty — task was destroyed, not pushed
  REQUIRE(queue.is_empty());
}
