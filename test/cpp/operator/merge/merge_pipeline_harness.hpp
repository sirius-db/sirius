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

#pragma once

// A MERGE_SORT as the source of a real pipeline, with a real task scheduler and GPU executor, for
// the operator-level merge tests. Only the task creator's queue is replaced, by a pump thread that
// does what a creation worker does with each `schedule(op)` request.

#include "catch.hpp"
#include "creator/task_creator.hpp"
#include "exec/config.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/query_retirement_test_utils.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/task_scheduler.hpp"
#include "utils/telemetry_utils.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::test::merge_utils {

/// Longest a merge may take before the test gives up on it, so a stalled pipeline fails fast.
inline constexpr std::chrono::seconds kPipelineWatchdog{20};

/// Pipeline sink with the materialized result collector's publication contract: one batch per
/// positioned publication, every position at most once, and no mix of positioned and
/// unpositioned results. Keeps the chunks by position.
class ordered_collector final : public ::sirius::op::sirius_physical_operator {
 public:
  ordered_collector()
    : sirius_physical_operator(::sirius::op::SiriusPhysicalOperatorType::RESULT_COLLECTOR, {}, 0)
  {
  }

  sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }

  bool is_sink() const override { return true; }

  void sink(const ::sirius::op::operator_data& input, ::cuda::stream_ref) override
  {
    auto const& data    = dynamic_cast<const ::sirius::op::pipelineable_operator_data&>(input);
    auto const& batches = data.get_data_batches();
    if (batches.empty()) { return; }
    auto const position = data.get_position();
    std::lock_guard<std::mutex> lock(_mutex);
    if (position && batches.size() != 1) {
      throw sirius::internal_exception("Positioned result requires one output batch");
    }
    if (_positioned && *_positioned != position.has_value()) {
      throw sirius::internal_exception("Cannot mix positioned and unpositioned query results");
    }
    _positioned = position.has_value();
    if (!position) {
      throw sirius::internal_exception("MERGE_SORT published a result without a position");
    }
    if (!_chunks.emplace(*position, batches.front()).second) {
      throw sirius::internal_exception("Duplicate query result position");
    }
    _arrival.push_back(*position);
  }

  /// The published chunks, by position.
  [[nodiscard]] std::map<std::size_t, std::shared_ptr<cucascade::data_batch>> chunks() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _chunks;
  }

  /// Positions in the order they were published.
  [[nodiscard]] std::vector<std::size_t> arrival() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _arrival;
  }

  void clear()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _chunks.clear();
  }

 private:
  mutable std::mutex _mutex;
  std::optional<bool> _positioned;
  std::map<std::size_t, std::shared_ptr<cucascade::data_batch>> _chunks;
  std::vector<std::size_t> _arrival;
};

/// The upstream SORT_PARTITION pipeline: already finished, so the FULL barrier is open.
class finished_pipeline final : public ::sirius::pipeline::sirius_pipeline {
 public:
  finished_pipeline() : sirius_pipeline{::sirius::pipeline::pipeline_build_context{nullptr}} {}
  [[nodiscard]] bool is_pipeline_finished() const override { return true; }
};

class merge_pipeline;

/// Task creator whose requests are served by a merge_pipeline, the way a creation worker serves
/// them: ask the operator for its hint, then turn every input it hands out into a real task.
class pumping_task_creator final : public ::sirius::creator::task_creator {
 public:
  pumping_task_creator(::sirius::exec::query_lifecycle_registry& lifecycle,
                       ::sirius::memory::sirius_memory_reservation_manager& manager,
                       merge_pipeline& harness)
    : task_creator(lifecycle, ::sirius::creator::task_creator_config{}, manager), _harness(harness)
  {
  }

  void schedule(::sirius::op::sirius_physical_operator* request) override;

 private:
  merge_pipeline& _harness;
};

struct merge_pipeline_options {
  std::size_t executor_threads = 1;
  std::string thread_name_prefix{"merge-pipeline"};
  /// Called on the requesting thread for every request for the MERGE_SORT, before it is queued.
  std::function<void()> on_request;
  /// Extra text for the watchdog's failure message.
  std::function<std::string()> describe;
};

/**
 * @brief A MERGE_SORT as the source of a real pipeline whose sink is an ordered_collector, with a
 * real task scheduler and GPU executor of `executor_threads` threads per GPU.
 *
 * Tasks are real gpu_pipeline_tasks: reserved, prepared, executed and published by the executor.
 * The MERGE_SORT reads @p input through a FULL port fed by a finished pipeline, with the residency
 * policy (and the downgrade priority) repository wiring gives it.
 *
 * @p merge, @p input and @p manager must outlive this object; the destructor pops what is left in
 * every partition of @p input.
 */
class merge_pipeline {
 public:
  merge_pipeline(::sirius::op::sirius_physical_merge_sort& merge,
                 cucascade::shared_data_repository& input,
                 ::sirius::memory::sirius_memory_reservation_manager& manager,
                 merge_pipeline_options options = {})
    : _merge(merge), _input(input), _options(std::move(options))
  {
    _merge.operator_id    = 1;
    collector.operator_id = 2;

    pipeline = std::make_shared<::sirius::pipeline::sirius_pipeline>(
      ::sirius::pipeline::pipeline_build_context{nullptr});
    pipeline->set_pipeline_id(7);
    ::sirius::pipeline::sirius_pipeline_build_state state;
    state.set_pipeline_source(*pipeline, _merge);
    state.set_pipeline_operators(*pipeline, {_merge});
    state.set_pipeline_sink(*pipeline, &collector, 1);
    _merge.set_pipeline(pipeline);

    // What repository wiring does for the input of a sort.
    auto const policy      = _merge.input_residency_policy("default");
    auto upstream          = std::make_unique<::sirius::op::sirius_physical_operator::port>();
    upstream->type         = ::sirius::op::MemoryBarrierType::FULL;
    upstream->repo         = &_input;
    upstream->src_pipeline = producer;
    upstream->residency    = policy;
    _merge.add_port("default", std::move(upstream));
    _input.set_downgrade_priority(policy.downgrade_priority);

    global = std::make_shared<::sirius::pipeline::sirius_pipeline_task_global_state>(
      pipeline, sirius::test::make_test_telemetry_context());
    global->set_completion_handler(completion);

    ::sirius::exec::thread_pool_config config;
    config.num_threads        = static_cast<int>(_options.executor_threads);
    config.thread_name_prefix = _options.thread_name_prefix;
    lifecycle.open_query(pipeline->get_query_id());
    scheduler = std::make_unique<::sirius::pipeline::task_scheduler>(
      lifecycle, config, manager, sirius::test::make_test_telemetry_context());
    creator = std::make_unique<pumping_task_creator>(lifecycle, manager, *this);
    creator->set_task_scheduler(*scheduler);
    scheduler->set_task_creator(*creator);
    pipeline->set_task_creator(creator.get());
    scheduler->start();
    _pump = std::thread([this] { pump(); });
  }

  merge_pipeline(const merge_pipeline&)            = delete;
  merge_pipeline& operator=(const merge_pipeline&) = delete;

  ~merge_pipeline()
  {
    stop_pump();
    if (!_retired) {
      try {
        sirius::test::retire_scheduler_query(
          lifecycle, *scheduler, pipeline->get_query_id(), false, creator.get());
      } catch (...) {
      }
    }
    scheduler->stop();
    // Task output buffers retain executor streams, so free them before the scheduler's stream pool.
    collector.clear();
    for (std::size_t partition = 0; partition < _input.num_partitions(); ++partition) {
      while (_input.pop_next_data_batch(partition)) {}
    }
  }

  /**
   * @brief Hand the MERGE_SORT to the task creator, as the upstream pipeline does when it
   * finishes, and wait under a watchdog until the pipeline has finished.
   *
   * Every later task exists only because the operator scheduled itself again.
   */
  void run()
  {
    creator->schedule(&_merge);
    auto const deadline = std::chrono::steady_clock::now() + kPipelineWatchdog;
    while (!pipeline->is_pipeline_finished() && !completion->has_error()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        FAIL("the merge did not finish within the watchdog: "
             << (_options.describe ? _options.describe() : std::string{})
             << pipeline->get_tasks_created() << " tasks created, "
             << pipeline->get_tasks_completed() << " completed, " << _input.total_size()
             << " batches left in the input repository");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds{2});
    }
    if (completion->has_error()) {
      std::string what = "unknown error";
      if (completion_result.wait_for(std::chrono::seconds{5}) == std::future_status::ready) {
        try {
          completion_result.get();
        } catch (const std::exception& e) {
          what = e.what();
        } catch (...) {
        }
      }
      FAIL("a merge task failed: " << what);
    }
    REQUIRE(pipeline->get_tasks_created() == pipeline->get_tasks_completed());
    stop_pump();
    sirius::test::retire_scheduler_query(
      lifecycle, *scheduler, pipeline->get_query_id(), true, creator.get());
    _retired = true;
  }

  // Called by pumping_task_creator::schedule, from any thread.
  void request(::sirius::op::sirius_physical_operator* node)
  {
    // The pipeline's consumers are notified through the same creator; only the MERGE_SORT has
    // tasks to create here.
    if (node != &_merge) { return; }
    if (_options.on_request) { _options.on_request(); }
    {
      std::lock_guard<std::mutex> lock(_pump_mutex);
      ++_pending;
    }
    _pump_wake.notify_one();
  }

 private:
  ::sirius::op::sirius_physical_merge_sort& _merge;
  cucascade::shared_data_repository& _input;
  merge_pipeline_options _options;

 public:
  ordered_collector collector;
  std::shared_ptr<finished_pipeline> producer = std::make_shared<finished_pipeline>();
  std::shared_ptr<::sirius::pipeline::sirius_pipeline> pipeline;
  std::shared_ptr<::sirius::pipeline::completion_handler> completion =
    std::make_shared<::sirius::pipeline::completion_handler>();
  std::future<void> completion_result = completion->get_awaitable();
  std::shared_ptr<::sirius::pipeline::sirius_pipeline_task_global_state> global;
  ::sirius::exec::query_lifecycle_registry lifecycle;
  std::unique_ptr<::sirius::pipeline::task_scheduler> scheduler;
  std::unique_ptr<pumping_task_creator> creator;

 private:
  void pump()
  {
    while (true) {
      {
        std::unique_lock<std::mutex> lock(_pump_mutex);
        _pump_wake.wait(lock, [this] { return _pending > 0 || _stop; });
        if (_stop) { return; }
        --_pending;
      }
      try {
        create_tasks();
      } catch (...) {
        completion->report_error(std::current_exception());
      }
    }
  }

  /// One creation worker's turn for the MERGE_SORT (task_creator::manager_loop).
  void create_tasks()
  {
    auto const hint = _merge.get_next_task_hint();
    if (hint.has_value() && hint->hint == ::sirius::op::TaskCreationHint::READY) {
      while (!_merge.all_ports_empty()) {
        auto task_lock = pipeline->get_task_creation_lock();
        if (pipeline->is_pipeline_finished()) { break; }
        auto data          = _merge.get_next_task_input_data();
        auto* pipelineable = dynamic_cast<::sirius::op::pipelineable_operator_data*>(data.get());
        if (!data || (pipelineable && pipelineable->get_data_batches().empty())) { break; }
        auto const device = data->get_preferred_device_id();
        auto local =
          std::make_unique<::sirius::pipeline::gpu_pipeline_task_local_state>(std::move(data));
        local->set_preferred_device_id(device.value_or(0));
        auto task = std::make_unique<::sirius::pipeline::gpu_pipeline_task>(
          creator->get_next_task_id(),
          std::vector<cucascade::shared_data_repository*>{},
          std::move(local),
          global);
        task_lock.unlock();
        scheduler->schedule(std::move(task));
      }
    }
    pipeline->update_pipeline_status(false);
  }

  void stop_pump()
  {
    {
      std::lock_guard<std::mutex> lock(_pump_mutex);
      _stop = true;
    }
    _pump_wake.notify_one();
    if (_pump.joinable()) { _pump.join(); }
  }

  std::mutex _pump_mutex;
  std::condition_variable _pump_wake;
  std::size_t _pending = 0;
  bool _stop           = false;
  bool _retired        = false;
  std::thread _pump;
};

inline void pumping_task_creator::schedule(::sirius::op::sirius_physical_operator* request)
{
  _harness.request(request);
}

}  // namespace sirius::test::merge_utils
