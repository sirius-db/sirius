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

// The downgrade executor's task_downgraded events: one per queued task whose inputs tier 2 moves
// off the GPU, none for repository (tier 1) moves or moves that start below the GPU.

#include "catch.hpp"
#include "data/convertible_data_batch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/data_repository_manager_registry.hpp"
#include "data/sirius_converter_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "event/downgraded_task.hpp"
#include "event/query_event_publisher.hpp"
#include "event/query_event_subscriber.hpp"
#include "exec/multi_index_priority_queue.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "query_id.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/telemetry_utils.hpp"
#include "utils/utils.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

using cucascade::memory::Tier;
using sirius::parallel::downgrade_executor;

namespace {

/// Every task_downgraded event delivered to it. Drains its mailbox on stop, so reading the events
/// after the downgrade request has returned sees all of them.
class task_downgrade_recorder final : public sirius::event::query_event_subscriber {
 public:
  struct entry {
    sirius::query_id_t query;
    sirius::event::downgraded_task task;
  };

  explicit task_downgrade_recorder(sirius::event::query_event_publisher& publisher)
    : query_event_subscriber(
        publisher, {sirius::event::event_type::task_downgraded}, /*drain_on_stop=*/true)
  {
    start();
  }

  ~task_downgrade_recorder() override { stop(); }

  [[nodiscard]] std::string_view name() const noexcept override { return "task-downgrades"; }

  /// Stop the subscriber, delivering everything already published, and return what arrived.
  std::vector<entry> collect()
  {
    stop();
    std::lock_guard lock(_mutex);
    return _entries;
  }

 private:
  void on_task_downgraded(sirius::event::event_id_t,
                          sirius::event::timestamp_t,
                          sirius::query_id_t query_id,
                          sirius::event::downgraded_task const& task) noexcept override
  {
    std::lock_guard lock(_mutex);
    _entries.push_back({query_id, task});
  }

  std::mutex _mutex;
  std::vector<entry> _entries;
};

/// Memory manager with one GPU, one HOST space and, when @p disk_dir is non-empty, a DISK space.
std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_manager(
  std::filesystem::path const& disk_dir = {})
{
  sirius::converter_registry::reset_for_testing();
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(2ull << 30)
    .set_reservation_fraction_per_gpu(0.75)
    .set_per_numa_region_capacity(4ull << 30)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(0.75);
  if (!disk_dir.empty()) { builder.set_disk_mounting_point(0, 4ull << 30, disk_dir.string()); }
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
  sirius::converter_registry::initialize();
  return manager;
}

std::shared_ptr<cucascade::data_batch> make_gpu_batch(cucascade::memory::memory_space& gpu_space,
                                                      std::size_t num_rows = 1000)
{
  auto stream                                            = cudf::get_default_stream();
  std::vector<cudf::data_type> col_types                 = {cudf::data_type{cudf::type_id::INT32}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {std::make_pair(0, 100000)};
  auto table = sirius::create_cudf_table_with_random_data(
    num_rows, col_types, ranges, stream, gpu_space.get_default_allocator());
  stream.sync();
  return sirius::make_data_batch(
    std::move(table), gpu_space, stream, sirius::telemetry::batch_telemetry_info{});
}

Tier batch_tier(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_tier();
}

std::size_t batch_size(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_data()->get_size_in_bytes();
}

std::unique_ptr<downgrade_executor> make_executor(
  sirius::exec::query_lifecycle_registry& lifecycle,
  sirius::data::data_repository_manager_registry& repo_registry,
  sirius::memory::sirius_memory_reservation_manager& mgr,
  cucascade::memory::memory_space& source,
  sirius::exec::multi_index_priority_queue<sirius::parallel::itask>& task_queue)
{
  sirius::exec::downgrade_executor_config config{
    .thread_pool    = {.num_threads = 1, .thread_name_prefix = "downgrade"},
    .monitor_period = std::chrono::milliseconds{0}};
  return std::make_unique<downgrade_executor>(
    lifecycle, config, repo_registry, source.get_id(), &source, mgr, &task_queue);
}

/// Global state of tasks of pipeline @p pipeline_id of @p query, with a scheduling priority and,
/// optionally, a preferred device. The pipeline has no operators, so it cannot account a task's
/// completion: it is detached on destruction, and must therefore be declared after the queue
/// that destroys the tasks.
struct task_global_state {
  std::shared_ptr<sirius::pipeline::gpu_pipeline_task_global_state> state;

  task_global_state(sirius::query_id_t query,
                    std::size_t pipeline_id,
                    sirius::exec::queue_priority priority,
                    std::optional<int> preferred_device)
  {
    const sirius::pipeline::pipeline_build_context build_ctx{nullptr, true};
    auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(build_ctx);
    pipeline->set_query_id(query);
    pipeline->set_pipeline_id(pipeline_id);
    state = std::make_shared<sirius::pipeline::gpu_pipeline_task_global_state>(
      std::move(pipeline), sirius::test::make_test_telemetry_context());
    state->set_priority(priority);
    if (preferred_device) { state->set_preferred_device_id(*preferred_device); }
  }

  ~task_global_state() { state->set_pipeline(nullptr); }

  task_global_state(const task_global_state&)            = delete;
  task_global_state& operator=(const task_global_state&) = delete;

  std::unique_ptr<sirius::pipeline::gpu_pipeline_task> make_task(
    std::uint64_t task_id, std::vector<std::shared_ptr<cucascade::data_batch>> batches) const
  {
    auto op_data = std::make_unique<sirius::op::pipelineable_operator_data>(std::move(batches));
    auto local =
      std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(op_data));
    return std::make_unique<sirius::pipeline::gpu_pipeline_task>(
      task_id, std::vector<cucascade::shared_data_repository*>{}, std::move(local), state);
  }
};

task_downgrade_recorder::entry const& find_task(
  std::vector<task_downgrade_recorder::entry> const& entries, std::uint64_t task_id)
{
  auto it = std::find_if(
    entries.begin(), entries.end(), [&](auto const& e) { return e.task.task_id == task_id; });
  REQUIRE(it != entries.end());
  return *it;
}

}  // namespace

TEST_CASE("tier 2 publishes one task_downgraded per converted task, with the inputs it moved",
          "[preload][downgrade_task_events]")
{
  auto mgr        = make_manager();
  auto* gpu_space = mgr->get_memory_space(Tier::GPU, 0);
  auto* host      = mgr->get_memory_space(Tier::HOST, 0);
  REQUIRE(gpu_space != nullptr);
  REQUIRE(host != nullptr);

  auto const q = sirius::make_query_id(7);
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(q);
  sirius::data::data_repository_manager_registry repo_registry;
  sirius::exec::multi_index_priority_queue<sirius::parallel::itask> queue(
    &sirius::pipeline::index_keys_for);
  auto const pinned_priority =
    static_cast<sirius::exec::queue_priority>(sirius::query_priority_bits(q) | 3);
  auto const free_priority =
    static_cast<sirius::exec::queue_priority>(sirius::query_priority_bits(q) | 4);
  // Declared after the queue so the pipelines are detached before the queue destroys the tasks.
  task_global_state pinned(q, /*pipeline_id=*/11, pinned_priority, /*preferred_device=*/0);
  task_global_state unpinned(q, /*pipeline_id=*/12, free_priority, std::nullopt);

  auto moved1 = make_gpu_batch(*gpu_space);
  auto moved2 = make_gpu_batch(*gpu_space, 1500);
  auto moved3 = make_gpu_batch(*gpu_space, 500);
  // Already off the GPU: not converted, so not reported.
  auto resident = make_gpu_batch(*gpu_space, 200);
  {
    // The chunked GPU->HOST copy rejects the default stream.
    rmm::cuda_stream conversion_stream;
    sirius::convertible_data_batch to_host(resident);
    REQUIRE(to_host.convert({host}, ::cuda::stream_ref{conversion_stream.value()}, *mgr, true)
              .has_value());
    conversion_stream.synchronize();
  }
  // A repository batch: tier 1 moves it and reports nothing.
  auto in_repository = make_gpu_batch(*gpu_space);
  {
    auto repo = std::make_unique<cucascade::shared_data_repository>();
    repo->add_data_batch(in_repository);
    repo_registry.create_for_query(q)->add_new_repository(1, "out", std::move(repo));
  }
  auto const gpu_bytes =
    std::vector<std::size_t>{batch_size(*moved1), batch_size(*moved2), batch_size(*moved3)};
  REQUIRE(queue.push(pinned.make_task(1, {moved1, resident, moved2})));
  REQUIRE(queue.push(unpinned.make_task(2, {moved3})));

  auto publisher = std::make_shared<sirius::event::query_event_publisher>();
  task_downgrade_recorder recorder(*publisher);
  auto executor = make_executor(lifecycle, repo_registry, *mgr, *gpu_space, queue);
  executor->set_query_event_publisher(*publisher);
  executor->start();
  REQUIRE(executor->request_free_memory_and_wait(1ull << 30) > 0);
  executor->stop();

  REQUIRE(queue.size() == 2);
  REQUIRE(batch_tier(*in_repository) == Tier::HOST);
  auto const entries = recorder.collect();
  REQUIRE(entries.size() == 2);

  auto const& first = find_task(entries, 1);
  CHECK(first.query == q);
  CHECK(first.task.pipeline_id == 11);
  CHECK(first.task.priority == pinned_priority);
  CHECK(first.task.preferred_device == 0);
  REQUIRE(first.task.inputs.size() == 2);

  auto const& second = find_task(entries, 2);
  CHECK(second.query == q);
  CHECK(second.task.pipeline_id == 12);
  CHECK(second.task.priority == free_priority);
  CHECK_FALSE(second.task.preferred_device.has_value());
  REQUIRE(second.task.inputs.size() == 1);

  struct expected {
    sirius::event::downgraded_input const* input;
    cucascade::data_batch* batch;
    std::size_t gpu_bytes;
  };
  // In the order the task's conversion moved them.
  for (auto const& want : {expected{&first.task.inputs[0], moved1.get(), gpu_bytes[0]},
                           expected{&first.task.inputs[1], moved2.get(), gpu_bytes[1]},
                           expected{&second.task.inputs[0], moved3.get(), gpu_bytes[2]}}) {
    REQUIRE(batch_tier(*want.batch) == Tier::HOST);
    CHECK(want.input->batch_id == want.batch->get_batch_id());
    CHECK(want.input->batch.lock().get() == want.batch);
    CHECK(want.input->to == host->get_id());
    CHECK(want.input->gpu_bytes == want.gpu_bytes);
  }
}

TEST_CASE("tier 2 publishes nothing for moves below the GPU or while no one subscribes",
          "[preload][downgrade_task_events]")
{
  sirius::test::scratch_dir const disk_dir{"downgrade_task_events"};
  auto mgr        = make_manager(disk_dir.path());
  auto* gpu_space = mgr->get_memory_space(Tier::GPU, 0);
  auto* host      = mgr->get_memory_space(Tier::HOST, 0);
  REQUIRE(gpu_space != nullptr);
  REQUIRE(host != nullptr);

  auto const q = sirius::make_query_id(1);
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(q);
  sirius::data::data_repository_manager_registry repo_registry;
  sirius::exec::multi_index_priority_queue<sirius::parallel::itask> queue(
    &sirius::pipeline::index_keys_for);
  auto const priority =
    static_cast<sirius::exec::queue_priority>(sirius::query_priority_bits(q) | 1);
  task_global_state global(q, /*pipeline_id=*/5, priority, /*preferred_device=*/0);

  auto publisher     = std::make_shared<sirius::event::query_event_publisher>();
  auto gpu_executor  = make_executor(lifecycle, repo_registry, *mgr, *gpu_space, queue);
  auto host_executor = make_executor(lifecycle, repo_registry, *mgr, *host, queue);
  gpu_executor->set_query_event_publisher(*publisher);
  host_executor->set_query_event_publisher(*publisher);
  gpu_executor->start();
  host_executor->start();

  // Nobody subscribes yet: the move is neither recorded nor published.
  auto unobserved = make_gpu_batch(*gpu_space);
  REQUIRE(queue.push(global.make_task(1, {unobserved})));
  REQUIRE_FALSE(publisher->has_subscribers(sirius::event::event_type::task_downgraded));
  REQUIRE(gpu_executor->request_free_memory_and_wait(1ull << 30) > 0);
  REQUIRE(batch_tier(*unobserved) == Tier::HOST);

  task_downgrade_recorder recorder(*publisher);
  // HOST to DISK: the task's input was not on a GPU, so preload has nothing to learn.
  REQUIRE(host_executor->request_free_memory_and_wait(1ull << 30) > 0);
  REQUIRE(batch_tier(*unobserved) == Tier::DISK);

  // The decision is taken per request: this move is seen.
  auto observed = make_gpu_batch(*gpu_space);
  REQUIRE(queue.push(global.make_task(2, {observed})));
  REQUIRE(gpu_executor->request_free_memory_and_wait(1ull << 30) > 0);
  REQUIRE(batch_tier(*observed) == Tier::HOST);
  host_executor->stop();
  gpu_executor->stop();

  auto const entries = recorder.collect();
  REQUIRE(entries.size() == 1);
  CHECK(entries.front().task.task_id == 2);
  REQUIRE(entries.front().task.inputs.size() == 1);
  CHECK(entries.front().task.inputs.front().batch_id == observed->get_batch_id());
}
