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
#include "data/sirius_converter_registry.hpp"
#include "exec/inspectable_mpsc.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_operator.hpp"
#include "parallel/task.hpp"
#include "pipeline/gpu_pipeline_task.hpp"

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace sirius {

/**
 * @brief Concrete convertible_data wrapping a gpu_pipeline_task with RAII queue ownership.
 *
 * Takes exclusive ownership of an itask via unique_ptr. On destruction, pushes the task
 * back into the originating inspectable_mpsc queue unless the queue has been interrupted
 * (shutdown). This enables temporary extraction from the queue for conversion, with
 * guaranteed return on all code paths.
 *
 * The convert() method follows the same save/lock/convert/restore pattern as
 * convertible_data_batch: for each data_batch in the task's input data, it saves the
 * previous state, locks for in_transit, attempts conversion to a target memory space,
 * and restores state on all paths (success, failure, exception).
 */
class convertible_gpu_pipeline_task : public convertible_data {
 public:
  /**
   * @brief Construct from a task extracted from the queue.
   * @param task The task to wrap (exclusive ownership taken).
   * @param queue The originating queue (task is returned here on destruction).
   */
  convertible_gpu_pipeline_task(
    std::unique_ptr<sirius::parallel::itask> task,
    sirius::exec::inspectable_mpsc<sirius::parallel::itask>& queue)
    : _task(std::move(task)), _queue(queue)
  {
  }

  // Non-copyable (unique_ptr member)
  convertible_gpu_pipeline_task(const convertible_gpu_pipeline_task&)            = delete;
  convertible_gpu_pipeline_task& operator=(const convertible_gpu_pipeline_task&) = delete;

  // Movable (unique_ptr transfers naturally)
  convertible_gpu_pipeline_task(convertible_gpu_pipeline_task&&)            = default;
  convertible_gpu_pipeline_task& operator=(convertible_gpu_pipeline_task&&) = default;

  /**
   * @brief Destructor: returns the task to the queue via RAII.
   *
   * If the task has been moved-from (nullptr), does nothing.
   * If push() returns false (queue interrupted during shutdown), logs a warning
   * and lets the task be destroyed — this is expected during query teardown.
   */
  ~convertible_gpu_pipeline_task() override
  {
    if (_task) {
      if (!_queue.push(std::move(_task))) {
        SIRIUS_LOG_WARN(
          "convertible_gpu_pipeline_task: queue interrupted, task destroyed");
      }
    }
  }

  /**
   * @brief Convert this task's data batches to reside in one of the target memory spaces.
   *
   * Iterates the data_batches from the task's pipelineable_operator_data input.
   * For each batch:
   * - Skips batches already in a target space (no conversion needed)
   * - Skips batches not in task_created state (busy or already processing)
   * - Locks for in_transit, requests a reservation in each target space, converts
   * - Restores the previous batch state on all paths
   *
   * @param target_spaces  Candidate destination memory spaces (tried in order).
   * @param stream         CUDA stream for asynchronous memory operations.
   * @param res_mgr        Reservation manager for acquiring memory in the target space.
   * @return true if at least one batch was converted, false otherwise.
   */
  bool convert(const std::vector<cucascade::memory::memory_space*>& target_spaces,
               rmm::cuda_stream_view stream,
               sirius::memory::sirius_memory_reservation_manager& res_mgr) override
  {
    auto* pipelineable = get_pipelineable_data();
    if (!pipelineable) { return false; }

    const auto& batches = pipelineable->get_data_batches();
    bool any_converted = false;

    for (const auto& batch : batches) {
      if (!batch) { continue; }

      // Skip batches already at a target space — no conversion needed
      auto* batch_space = batch->get_memory_space();
      bool already_at_target = false;
      for (auto* ts : target_spaces) {
        if (batch_space == ts) {
          already_at_target = true;
          break;
        }
      }
      if (already_at_target) { continue; }

      // Only convert batches in task_created state
      if (batch->get_state() != cucascade::batch_state::task_created) { continue; }

      auto prev_state = batch->get_state();

      if (!batch->try_to_lock_for_in_transit()) { continue; }

      try {
        auto data_size = batch->get_data()->get_size_in_bytes();
        bool space_succeeded = false;

        for (auto* space : target_spaces) {
          auto reservation = res_mgr.request_reservation(
            cucascade::memory::specific_memory_space{space->get_tier(),
                                                     space->get_id().device_id},
            data_size);
          if (!reservation) { continue; }

          auto* mem_space =
            res_mgr.get_memory_space(reservation->tier(), reservation->device_id());
          if (!mem_space) { continue; }

          auto& registry = sirius::converter_registry::get();

          switch (space->get_tier()) {
            case cucascade::memory::Tier::HOST:
              batch->convert_to<cucascade::host_data_representation>(
                registry, mem_space, stream);
              break;
            case cucascade::memory::Tier::GPU:
              batch->convert_to<cucascade::gpu_table_representation>(
                registry, mem_space, stream);
              break;
            default: continue;
          }

          batch->try_to_release_in_transit(
            std::optional<cucascade::batch_state>{prev_state});
          any_converted = true;
          space_succeeded = true;
          break;
        }

        if (!space_succeeded) {
          // No target space succeeded for this batch — restore state
          batch->try_to_release_in_transit(
            std::optional<cucascade::batch_state>{prev_state});
        }
      } catch (...) {
        batch->try_to_release_in_transit(
          std::optional<cucascade::batch_state>{prev_state});
        throw;
      }
    }

    return any_converted;
  }

  /**
   * @brief Get the size in bytes of this task's data in the specified memory space.
   *
   * Sums bytes across all data_batches in the task's input that reside in the
   * given memory space.
   *
   * @param space The memory space to query.
   * @return Total size in bytes, or 0 if no data resides in that space.
   */
  std::size_t bytes_in_space(cucascade::memory::memory_space* space) const override
  {
    auto* pipelineable = get_pipelineable_data();
    if (!pipelineable) { return 0; }

    std::size_t total = 0;
    for (const auto& batch : pipelineable->get_data_batches()) {
      if (batch && batch->get_memory_space() == space) {
        total += batch->get_data()->get_size_in_bytes();
      }
    }
    return total;
  }

 private:
  /**
   * @brief Navigate the dynamic_cast chain to reach pipelineable_operator_data.
   *
   * The chain is: itask -> gpu_pipeline_task -> local_state ->
   * gpu_pipeline_task_local_state -> _input_data -> pipelineable_operator_data.
   * Returns nullptr at any point if the cast fails (e.g., the task is not a
   * gpu_pipeline_task, or has no pipelineable input data).
   */
  sirius::op::pipelineable_operator_data* get_pipelineable_data() const
  {
    auto* gpt = dynamic_cast<sirius::pipeline::gpu_pipeline_task*>(_task.get());
    if (!gpt) { return nullptr; }

    auto* ls = gpt->local_state();
    if (!ls) { return nullptr; }

    auto* gpt_ls =
      dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(ls);
    if (!gpt_ls) { return nullptr; }

    return dynamic_cast<sirius::op::pipelineable_operator_data*>(
      gpt_ls->_input_data.get());
  }

  std::unique_ptr<sirius::parallel::itask> _task;
  sirius::exec::inspectable_mpsc<sirius::parallel::itask>& _queue;
};

/**
 * @brief Concrete convertible_data_provider that discovers convertible tasks in an
 *        inspectable_mpsc queue.
 *
 * Uses mutable_pop_if to extract tasks whose data_batches match a given memory space
 * and batch_state::task_created. Each extracted task is wrapped in a
 * convertible_gpu_pipeline_task, which returns the task to the queue on destruction
 * via RAII.
 *
 * Non-gpu_pipeline_tasks and tasks without pipelineable_operator_data are silently
 * skipped by the predicate (they remain in the queue).
 */
class convertible_gpu_pipeline_task_provider : public convertible_data_provider {
 public:
  /**
   * @brief Construct from a reference to the task queue.
   * @param queue The inspectable_mpsc queue to search (non-owning reference).
   */
  explicit convertible_gpu_pipeline_task_provider(
    sirius::exec::inspectable_mpsc<sirius::parallel::itask>& queue)
    : _queue(queue)
  {
  }

  /**
   * @brief Get the next task whose data_batches match the given memory space.
   *
   * Calls mutable_pop_if with a predicate that navigates the dynamic_cast chain
   * and checks for data_batches in the target space with batch_state::task_created.
   *
   * @param space           The memory space to filter by.
   * @param front_to_back   Iteration direction.
   * @return A convertible_gpu_pipeline_task wrapping the matching task, or nullptr.
   */
  std::unique_ptr<convertible_data> get_next_convertible(
    cucascade::memory::memory_space* space, bool front_to_back) override
  {
    auto result = _queue.mutable_pop_if(
      [space](sirius::parallel::itask& task) {
        return has_matching_batches(task, space);
      },
      front_to_back);

    if (!result) { return nullptr; }
    return std::make_unique<convertible_gpu_pipeline_task>(
      std::move(result), _queue);
  }

  /**
   * @brief Get all tasks whose data_batches match the given memory space.
   *
   * Repeatedly calls mutable_pop_if until no more matching tasks are found.
   * Each extracted task is wrapped in a convertible_gpu_pipeline_task for RAII
   * queue return.
   *
   * @param space           The memory space to filter by.
   * @param front_to_back   Iteration direction.
   * @return A vector of convertible_gpu_pipeline_task instances (may be empty).
   */
  std::vector<std::unique_ptr<convertible_data>> get_all_convertible(
    cucascade::memory::memory_space* space, bool front_to_back) override
  {
    std::vector<std::unique_ptr<convertible_data>> results;

    while (true) {
      auto result = _queue.mutable_pop_if(
        [space](sirius::parallel::itask& task) {
          return has_matching_batches(task, space);
        },
        front_to_back);

      if (!result) { break; }
      results.push_back(std::make_unique<convertible_gpu_pipeline_task>(
        std::move(result), _queue));
    }

    return results;
  }

  /**
   * @brief Get the total byte size of task data in the given memory space.
   *
   * Returns 0 because precise byte counting would require temporarily removing
   * and re-inserting tasks, which is unsafe under concurrent producers. Callers
   * needing exact totals should use get_all_convertible() + bytes_in_space().
   *
   * @param space The memory space to query.
   * @return Always 0 (see rationale above).
   */
  std::size_t get_bytes_in_space(
    cucascade::memory::memory_space* /*space*/) const override
  {
    return 0;
  }

 private:
  /**
   * @brief Predicate: does this task contain data_batches in the given space with
   *        batch_state::task_created?
   *
   * Lightweight — only performs dynamic_casts and state checks. Suitable for use
   * under the queue mutex per inspectable_mpsc contract (T-07-01 mitigation).
   *
   * @param task  The task to inspect (mutable reference from mutable_pop_if).
   * @param space The memory space to match.
   * @return true if the task has at least one matching batch.
   */
  static bool has_matching_batches(sirius::parallel::itask& task,
                                   cucascade::memory::memory_space* space)
  {
    auto* gpt = dynamic_cast<sirius::pipeline::gpu_pipeline_task*>(&task);
    if (!gpt) { return false; }

    auto* ls = gpt->local_state();
    if (!ls) { return false; }

    auto* gpt_ls =
      dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(ls);
    if (!gpt_ls) { return false; }

    auto* pipelineable =
      dynamic_cast<sirius::op::pipelineable_operator_data*>(
        gpt_ls->_input_data.get());
    if (!pipelineable) { return false; }

    for (const auto& batch : pipelineable->get_data_batches()) {
      if (batch && batch->get_memory_space() == space &&
          batch->get_state() == cucascade::batch_state::task_created) {
        return true;
      }
    }

    return false;
  }

  sirius::exec::inspectable_mpsc<sirius::parallel::itask>& _queue;
};

}  // namespace sirius
