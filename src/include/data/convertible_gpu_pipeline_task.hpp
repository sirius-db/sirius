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

#pragma once

#include "data/convertible_data.hpp"
#include "data/convertible_data_batch.hpp"
#include "data/sirius_converter_registry.hpp"
#include "exec/inspectable_priority_queue.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_operator.hpp"
#include "parallel/task.hpp"
#include "pipeline/gpu_pipeline_task.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace sirius {

/**
 * @brief Concrete convertible_data wrapping a gpu_pipeline_task with RAII queue ownership.
 *
 * Takes exclusive ownership of an itask via unique_ptr. On destruction, pushes the task
 * back into the originating task queue unless the queue has been closed (shutdown). This
 * enables temporary extraction from the queue for conversion, with guaranteed return on
 * all code paths.
 *
 * The convert() method uses the RAII mutable lock pattern: for each data_batch in the
 * task's input data, it acquires a mutable_data_batch via to_mutable() or try_to_mutable(),
 * attempts conversion to a target memory space, and relies on the mutable_data_batch RAII
 * destructor to restore idle state on all exit paths (success, failure, exception).
 */
class convertible_gpu_pipeline_task : public convertible_data {
 public:
  /**
   * @brief Construct from a task extracted from the queue.
   * @param task The task to wrap (exclusive ownership taken).
   * @param queue The originating task queue (task is returned here on destruction).
   */
  convertible_gpu_pipeline_task(
    std::unique_ptr<sirius::parallel::itask> task,
    sirius::exec::inspectable_priority_queue<sirius::parallel::itask>& queue)
    : _task(std::move(task)), _queue(queue)
  {
  }

  // Non-copyable, move-constructible only (reference member prevents move assignment)
  convertible_gpu_pipeline_task(const convertible_gpu_pipeline_task&)            = delete;
  convertible_gpu_pipeline_task& operator=(const convertible_gpu_pipeline_task&) = delete;
  convertible_gpu_pipeline_task(convertible_gpu_pipeline_task&&)                 = default;
  convertible_gpu_pipeline_task& operator=(convertible_gpu_pipeline_task&&)      = delete;

  /**
   * @brief Destructor: returns the task to the queue via RAII.
   *
   * If the task has been moved-from (nullptr), does nothing.
   * Pushes the task back to the queue; if the queue is closed (shutdown),
   * the task is destroyed -- this is expected during query teardown.
   */
  ~convertible_gpu_pipeline_task() override
  {
    if (_task) { (void)_queue.push(std::move(_task)); }
  }

  /**
   * @brief Convert this task's data batches to reside in one of the target memory spaces.
   *
   * Iterates the data_batches from the task's pipelineable_operator_data input.
   * For each batch not already at a target space, delegates to convertible_data_batch
   * which acquires a mutable_data_batch (blocking or non-blocking), attempts the conversion,
   * and releases automatically via RAII.
   *
   * @param target_spaces  Candidate destination memory spaces (tried in order).
   * @param stream         CUDA stream for asynchronous memory operations.
   * @param res_mgr        Reservation manager for acquiring memory in the target space.
   * @param blocking       When true, uses to_mutable() (blocks until exclusive lock acquired).
   *                       When false, uses try_to_mutable() (returns nullopt immediately if
   *                       the lock is unavailable).
   * @return A vector of bytes converted per target space index on success, or nullopt if
   *         no batches were converted.
   */
  std::optional<std::vector<std::size_t>> convert(
    const std::vector<const cucascade::memory::memory_space*>& target_spaces,
    rmm::cuda_stream_view stream,
    sirius::memory::sirius_memory_reservation_manager& res_mgr,
    bool blocking = true) override
  {
    auto* operator_data = get_pipelineable_data();
    if (!operator_data) { return std::nullopt; }

    const auto& batches = operator_data->get_data_batches();
    std::vector<std::size_t> totals(target_spaces.size(), 0);
    bool any_converted = false;

    for (const auto& batch : batches) {
      if (!batch) { continue; }

      // Quick check: skip batches already at a target space (avoid unnecessary locking)
      {
        auto ro           = batch->to_read_only();
        auto* batch_space = ro.get_memory_space();
        bool at_target    = false;
        for (const auto* ts : target_spaces) {
          if (batch_space == ts) {
            at_target = true;
            break;
          }
        }
        if (at_target) { continue; }
      }

      // Delegate to convertible_data_batch which handles to_mutable() internally
      sirius::convertible_data_batch batch_converter(batch);
      auto result = batch_converter.convert(target_spaces, stream, res_mgr, blocking);
      if (result) {
        any_converted = true;
        for (std::size_t i = 0; i < result->size(); ++i) {
          totals[i] += (*result)[i];
        }
      }
    }

    if (!any_converted) { return std::nullopt; }
    return totals;
  }

  /**
   * @brief Get the size in bytes of this task's data in the specified memory space.
   *
   * Sums bytes across all data_batches in the task's input that reside in the
   * given memory space. Accesses memory space via to_read_only() as required by
   * the new cucascade API.
   *
   * @param space The memory space to query.
   * @return Total size in bytes, or 0 if no data resides in that space.
   */
  std::size_t bytes_in_space(cucascade::memory::memory_space* space) const override
  {
    auto* operator_data = get_pipelineable_data();
    if (!operator_data) { return 0; }

    std::size_t total = 0;
    for (const auto& ro : operator_data->get_read_only_batches(false)) {
      if (ro.get_memory_space() == space) { total += ro.get_data()->get_size_in_bytes(); }
    }
    return total;
  }

  /**
   * @brief Navigate the dynamic_cast chain to reach pipelineable_operator_data.
   *
   * The chain is: itask -> gpu_pipeline_task -> local_state ->
   * gpu_pipeline_task_local_state -> _input_data -> pipelineable_operator_data.
   * Returns nullptr at any point if the cast fails (e.g., the task is not a
   * gpu_pipeline_task, or has no pipelineable operator_data input data).
   *
   * Public and static so that both the wrapper and the provider can share this
   * logic without duplicating the dynamic_cast chain.
   *
   * @param task The task to inspect.
   * @return Pointer to pipelineable_operator_data, or nullptr if unreachable.
   */
  static sirius::op::pipelineable_operator_data* get_pipelineable_data(
    sirius::parallel::itask& task)
  {
    auto* gpt = dynamic_cast<sirius::pipeline::gpu_pipeline_task*>(&task);
    if (!gpt) { return nullptr; }

    auto* ls = gpt->local_state();
    if (!ls) { return nullptr; }

    auto* gpt_ls = dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(ls);
    if (!gpt_ls) { return nullptr; }

    return dynamic_cast<sirius::op::pipelineable_operator_data*>(gpt_ls->_input_data.get());
  }

 private:
  /// @brief Convenience overload for the owned task.
  sirius::op::pipelineable_operator_data* get_pipelineable_data() const
  {
    return _task ? get_pipelineable_data(*_task) : nullptr;
  }

  std::unique_ptr<sirius::parallel::itask> _task;
  sirius::exec::inspectable_priority_queue<sirius::parallel::itask>& _queue;
};

/**
 * @brief Concrete convertible_data_provider that discovers convertible tasks in a task queue.
 *
 * Filters tasks by checking data_batches for batch_state::idle and a matching memory space.
 * Only batches in batch_state::idle are considered for conversion -- the new cucascade API
 * requires idle state before acquiring a mutable or read-only accessor.
 *
 * Non-gpu_pipeline_tasks and tasks without pipelineable_operator_data are silently skipped.
 */
class convertible_gpu_pipeline_task_provider : public convertible_data_provider {
 public:
  /**
   * @brief Construct from a reference to the task queue.
   * @param queue The task queue to search (non-owning reference).
   */
  explicit convertible_gpu_pipeline_task_provider(
    sirius::exec::inspectable_priority_queue<sirius::parallel::itask>& queue)
    : _queue(queue)
  {
  }

  /**
   * @brief Get the next task whose data_batches match the given memory space.
   *
   * Extracts the first matching task from the queue using mutable_pop_if().
   * The extracted task is wrapped in a convertible_gpu_pipeline_task which
   * returns the task to the queue via RAII on destruction.
   *
   * Only tasks with at least one idle data_batch in the given memory space
   * are returned. Iteration direction is respected (front_to_back).
   *
   * @param space           The memory space to filter by.
   * @param front_to_back   Iteration direction.
   * @return A convertible_gpu_pipeline_task wrapping the matching task, or nullptr.
   */
  std::unique_ptr<convertible_data> get_next_convertible(cucascade::memory::memory_space* space,
                                                         bool front_to_back) override
  {
    auto task = _queue.mutable_pop_if(
      [space](sirius::parallel::itask& t) { return has_matching_batches(t, space); },
      front_to_back);
    if (!task) { return nullptr; }
    return std::make_unique<convertible_gpu_pipeline_task>(std::move(task), _queue);
  }

  /**
   * @brief Get all convertible tasks matching the given memory space.
   *
   * Repeatedly extracts matching tasks from the queue until no more are found.
   * Each extracted task is wrapped in a convertible_gpu_pipeline_task which
   * returns the task to the queue via RAII on destruction.
   *
   * @param space           The memory space to filter by.
   * @param front_to_back   Iteration direction.
   * @param ignore_subscribed  Unused here: a queued task always subscribes to its own input
   *                           batches, so honoring this filter would disable task-queue downgrade
   *                           entirely. The subscriber filter applies to repository batches (see
   *                           convertible_data_batch_provider), not a task's own held batches.
   * @return A vector of convertible_gpu_pipeline_task wrappers (may be empty).
   */
  std::vector<std::unique_ptr<convertible_data>> get_all_convertible(
    cucascade::memory::memory_space* space,
    bool front_to_back,
    [[maybe_unused]] bool ignore_subscribed = true) override
  {
    std::vector<std::unique_ptr<convertible_data>> results;
    while (true) {
      auto task = _queue.mutable_pop_if(
        [space](sirius::parallel::itask& t) { return has_matching_batches(t, space); },
        front_to_back);
      if (!task) { break; }
      results.push_back(std::make_unique<convertible_gpu_pipeline_task>(std::move(task), _queue));
    }
    return results;
  }

 private:
  /**
   * @brief Predicate: does this task contain data_batches in the given space with
   *        batch_state::idle?
   *
   * Lightweight -- only performs dynamic_casts and state checks. Accesses memory
   * space via to_read_only().
   *
   * @param task  The task to inspect.
   * @param space The memory space to match.
   * @return true if the task has at least one matching batch in batch_state::idle.
   */
  static bool has_matching_batches(sirius::parallel::itask& task,
                                   cucascade::memory::memory_space* space)
  {
    auto* operator_data = convertible_gpu_pipeline_task::get_pipelineable_data(task);
    if (!operator_data) { return false; }

    for (const auto& batch : operator_data->get_data_batches()) {
      if (!batch) { continue; }
      if (batch->get_state() != cucascade::batch_state::idle) { continue; }
      auto ro = batch->try_to_read_only();
      if (!ro) { continue; }
      if (ro->get_memory_space() == space) { return true; }
    }

    return false;
  }

  sirius::exec::inspectable_priority_queue<sirius::parallel::itask>& _queue;
};

}  // namespace sirius
