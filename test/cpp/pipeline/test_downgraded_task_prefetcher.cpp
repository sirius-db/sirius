/*
 * Copyright 2026, Sirius Contributors.
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
#include <data/convertible_data_batch.hpp>
#include <exec/multi_index_priority_queue.hpp>
#include <op/sirius_physical_operator.hpp>
#include <parallel/task.hpp>
#include <pipeline/downgraded_task_prefetcher.hpp>
#include <pipeline/gpu_pipeline_task.hpp>
#include <pipeline/sirius_pipeline_task_states.hpp>
#include <utils/telemetry_utils.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;

// Shared test environment: initialize memory manager once for all tests in this
// file (mirrors test_convertible_gpu_pipeline_task.cpp).
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

/// Minimal dummy task that is NOT a gpu_pipeline_task — the prefetcher must skip it.
class dummy_task_local_state : public sirius::parallel::itask_local_state {};
class dummy_task : public sirius::parallel::itask {
 public:
  dummy_task() : itask(0, std::make_unique<dummy_task_local_state>(), nullptr) {}
  void execute(rmm::cuda_stream_view /*stream*/) override {}
};

/// Helper: create a gpu_pipeline_task with the given data batches (idle state).
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

/// Helper: make a batch that is resident on the HOST tier (created on GPU, then
/// downgraded — the same path the downgrade executor takes).
std::shared_ptr<cucascade::data_batch> make_host_batch(test_env& e,
                                                       std::vector<int32_t> values = {1, 2, 3})
{
  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::move(values), cudf::type_id::INT32);
  sirius::convertible_data_batch downgrader(batch);
  auto result = downgrader.convert({e.host_space}, e.stream(), *e.mgr, /*blocking=*/true);
  REQUIRE(result.has_value());
  return batch;
}

inline cucascade::memory::Tier get_batch_tier(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_tier();
}

/// Poll until the prefetcher reports @p n upgraded batches, or 5s elapse.
bool wait_for_prefetched(sirius::pipeline::downgraded_task_prefetcher& p, std::size_t n)
{
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (p.batches_prefetched() >= n) { return true; }
    std::this_thread::sleep_for(2ms);
  }
  return p.batches_prefetched() >= n;
}

}  // anonymous namespace

TEST_CASE("prefetcher upgrades HOST input of queued task to GPU", "[downgraded_task_prefetcher]")
{
  auto& e = env();

  sirius::exec::multi_index_priority_queue<sirius::parallel::itask> queue(
    [](const sirius::parallel::itask&) {
      return sirius::exec::index_keys{
        0, sirius::op::SiriusPhysicalOperatorType::INVALID, 0, sirius::exec::no_preferred_device};
    });
  auto batch = make_host_batch(e);
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);
  queue.push((make_test_gpu_task(1, {batch})));

  sirius::pipeline::downgraded_task_prefetcher::config cfg{};
  cfg.num_threads       = 1;
  cfg.min_free_fraction = 0.05;
  sirius::pipeline::downgraded_task_prefetcher prefetcher(cfg, queue, *e.mgr, e.gpu_space);

  REQUIRE(wait_for_prefetched(prefetcher, 1));
  prefetcher.stop();

  // The batch is back on GPU, idle, and the task never left the queue.
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::GPU);
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);
  REQUIRE(queue.size() == 1);
  REQUIRE(prefetcher.bytes_prefetched() > 0);
}

TEST_CASE("prefetcher upgrades multiple queued tasks in dispatch order",
          "[downgraded_task_prefetcher]")
{
  auto& e = env();

  sirius::exec::multi_index_priority_queue<sirius::parallel::itask> queue(
    [](const sirius::parallel::itask&) {
      return sirius::exec::index_keys{
        0, sirius::op::SiriusPhysicalOperatorType::INVALID, 0, sirius::exec::no_preferred_device};
    });
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (int i = 0; i < 3; ++i) {
    auto batch = make_host_batch(e, {i, i + 1, i + 2});
    batches.push_back(batch);
    queue.push((make_test_gpu_task(i + 1, {batch})));
  }

  sirius::pipeline::downgraded_task_prefetcher::config cfg{};
  cfg.num_threads       = 1;
  cfg.min_free_fraction = 0.05;
  sirius::pipeline::downgraded_task_prefetcher prefetcher(cfg, queue, *e.mgr, e.gpu_space);

  REQUIRE(wait_for_prefetched(prefetcher, 3));
  prefetcher.stop();

  for (auto& batch : batches) {
    REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::GPU);
    REQUIRE(batch->get_state() == cucascade::batch_state::idle);
  }
  REQUIRE(queue.size() == 3);
}

TEST_CASE("prefetcher leaves GPU-resident inputs alone", "[downgraded_task_prefetcher]")
{
  auto& e = env();

  sirius::exec::multi_index_priority_queue<sirius::parallel::itask> queue(
    [](const sirius::parallel::itask&) {
      return sirius::exec::index_keys{
        0, sirius::op::SiriusPhysicalOperatorType::INVALID, 0, sirius::exec::no_preferred_device};
    });
  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);
  queue.push((make_test_gpu_task(1, {batch})));

  sirius::pipeline::downgraded_task_prefetcher::config cfg{};
  cfg.num_threads       = 1;
  cfg.min_free_fraction = 0.05;
  sirius::pipeline::downgraded_task_prefetcher prefetcher(cfg, queue, *e.mgr, e.gpu_space);

  std::this_thread::sleep_for(100ms);
  prefetcher.stop();

  REQUIRE(prefetcher.batches_prefetched() == 0);
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::GPU);
}

TEST_CASE("headroom floor blocks prefetch", "[downgraded_task_prefetcher]")
{
  auto& e = env();

  sirius::exec::multi_index_priority_queue<sirius::parallel::itask> queue(
    [](const sirius::parallel::itask&) {
      return sirius::exec::index_keys{
        0, sirius::op::SiriusPhysicalOperatorType::INVALID, 0, sirius::exec::no_preferred_device};
    });
  auto batch = make_host_batch(e);
  queue.push((make_test_gpu_task(1, {batch})));

  sirius::pipeline::downgraded_task_prefetcher::config cfg{};
  cfg.num_threads = 1;
  // available <= max_memory always, so a floor of the full space blocks every upgrade.
  cfg.min_free_fraction = 1.0;
  sirius::pipeline::downgraded_task_prefetcher prefetcher(cfg, queue, *e.mgr, e.gpu_space);

  std::this_thread::sleep_for(100ms);
  prefetcher.stop();

  REQUIRE(prefetcher.batches_prefetched() == 0);
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);
}

TEST_CASE("busy batch is skipped, then picked up once released", "[downgraded_task_prefetcher]")
{
  auto& e = env();

  sirius::exec::multi_index_priority_queue<sirius::parallel::itask> queue(
    [](const sirius::parallel::itask&) {
      return sirius::exec::index_keys{
        0, sirius::op::SiriusPhysicalOperatorType::INVALID, 0, sirius::exec::no_preferred_device};
    });
  auto batch = make_host_batch(e);
  queue.push((make_test_gpu_task(1, {batch})));

  // Hold the exclusive lock: the prefetcher's try_to_mutable must skip.
  auto exclusive_lock = std::make_optional(batch->to_mutable());

  sirius::pipeline::downgraded_task_prefetcher::config cfg{};
  cfg.num_threads       = 1;
  cfg.min_free_fraction = 0.05;
  sirius::pipeline::downgraded_task_prefetcher prefetcher(cfg, queue, *e.mgr, e.gpu_space);

  std::this_thread::sleep_for(100ms);
  REQUIRE(prefetcher.batches_prefetched() == 0);

  // Release the lock — the next sweep should convert it.
  exclusive_lock.reset();
  REQUIRE(wait_for_prefetched(prefetcher, 1));
  prefetcher.stop();

  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::GPU);
}

TEST_CASE("non-gpu_pipeline_task entries are tolerated", "[downgraded_task_prefetcher]")
{
  auto& e = env();

  sirius::exec::multi_index_priority_queue<sirius::parallel::itask> queue(
    [](const sirius::parallel::itask&) {
      return sirius::exec::index_keys{
        0, sirius::op::SiriusPhysicalOperatorType::INVALID, 0, sirius::exec::no_preferred_device};
    });
  queue.push((std::make_unique<dummy_task>()));
  auto batch = make_host_batch(e);
  queue.push((make_test_gpu_task(1, {batch})));

  sirius::pipeline::downgraded_task_prefetcher::config cfg{};
  cfg.num_threads       = 1;
  cfg.min_free_fraction = 0.05;
  sirius::pipeline::downgraded_task_prefetcher prefetcher(cfg, queue, *e.mgr, e.gpu_space);

  REQUIRE(wait_for_prefetched(prefetcher, 1));
  prefetcher.stop();

  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::GPU);
  REQUIRE(queue.size() == 2);
}
