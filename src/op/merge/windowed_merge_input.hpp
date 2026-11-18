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

#include "op/merge/windowed_merge.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <functional>
#include <vector>

namespace sirius::op::merge {

/**
 * @brief Task input of one windowed MERGE_SORT task.
 *
 * Carries CPU-only cursors into the runs of one partition instead of locked batches: the task
 * resolves and locks batches itself, window by window. Estimates are window-sized
 * (estimate_task_bytes), not partition-sized, and leave out the window buffers a merger kept
 * from the partition's previous task already holds (set_kept_buffer_runs). Use
 * operator_data::set_preferred_device_id for locality.
 */
class windowed_merge_input final : public operator_data {
 public:
  /**
   * @param partition The sort partition being merged.
   * @param cursors One cursor per run the task reads.
   * @param final_pass False when the task writes an intermediate run (fan-in pass).
   * @param first_chunk Chunk index of the task's output within the partition.
   * @param cfg Window rows and output target of the task.
   * @param row_width Estimated row width in bytes of the fetched columns.
   * @param key_row_width Estimated row width in bytes of the key columns.
   * @param gpu_runs How many of the runs are sliced in place on the GPU and so need no window
   *        buffer, as far as their residency when the task was planned tells. The estimates
   *        still charge the window buffers of one of them (see get_estimated_size_in_bytes()).
   * @param fallback_bytes Sum, over the runs, of the largest batch of the run that may be
   *        converted whole to the GPU tier.
   * @param compressed_runs How many of the runs hold a compressed batch (read through decode
   *        cursors that keep scratch between refills).
   * @param fetch_columns Columns the task fetches. Both estimates use it: 0 leaves the validity
   *        masks of the window buffers out of the reservation.
   * @param string_columns How many of the fetched columns are STRING. Both estimates use it: 0
   *        leaves the offsets and the chars slack of the window buffers out of the reservation.
   * @param string_row_width Sum of the estimated widths of the STRING fetched columns; 0 = not
   *        known (estimate_window_slot_bytes then takes the slack on the whole row).
   * @param round_follows_output Whether cfg.round_rows was sized from the output target
   *        (round_rows_for); shrink_for_retry() then sizes it again from the halved target.
   */
  windowed_merge_input(std::size_t partition,
                       std::vector<merge_input_cursor> cursors,
                       bool final_pass,
                       std::size_t first_chunk,
                       window_config cfg,
                       std::size_t row_width,
                       std::size_t key_row_width,
                       std::size_t gpu_runs,
                       std::size_t fallback_bytes,
                       std::size_t compressed_runs  = 0,
                       std::size_t fetch_columns    = 0,
                       std::size_t string_columns   = 0,
                       std::size_t string_row_width = 0,
                       bool round_follows_output    = false);

  [[nodiscard]] operator_data_type get_type() const override
  {
    return operator_data_type::WINDOWED_MERGE;
  }

  /// Captures the task's memory space only; locks no batch.
  void prepare_for_processing(const cucascade::memory::memory_space* requested_memory_space,
                              ::cuda::stream_ref stream) override;

  /// estimate_task_bytes(...) plus estimate_reader_state_bytes(...) of the task's current
  /// window_config. The runs of set_kept_buffer_runs() are charged no window buffer, and
  /// neither are the runs sliced in place (gpu_runs) but one: a run on the GPU when the task was
  /// planned stays spillable until the task locks it, so the window buffers of one such run are
  /// reserved as a spare. More than one moving in that time is left to the retry (see
  /// shrink_for_retry()).
  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override;

  /// Same as get_estimated_size_in_bytes().
  [[nodiscard]] std::size_t get_estimated_working_set_size_in_bytes() const override;

  /// Prepares the retry of a task that ran out of memory.
  ///
  /// A task that left window buffers out of its estimate for runs sliced in place (gpu_runs > 0)
  /// may have failed because those runs were spilled before it locked them. Its first retry
  /// keeps its sizes and charges every run instead: gpu_runs is reset to 0, the recharge
  /// observer is called and the call returns true (the estimate changed, so the executor keeps
  /// the reservation floor from before the OOM, like after a shrink).
  ///
  /// Otherwise halves window_rows (floor kShrinkWindowRowsFloor) and output_target_bytes (floor
  /// 16 MiB), sizes round_rows again, calls the shrink observer and returns true; returns false
  /// when both sizes are already at their floors.
  ///
  /// Either way the kept buffer runs are reset to 0: the task that failed destroyed the kept
  /// merger, so the retry allocates every window buffer.
  bool shrink_for_retry() override;

  /// How many of the runs have their window buffers in the merger the partition kept from its
  /// previous task: device memory that is already allocated, which this task reuses and does not
  /// allocate. The estimates charge no window buffer for them. 0 until set.
  void set_kept_buffer_runs(std::size_t runs) noexcept;

  /// Called, with no argument, each time shrink_for_retry() shrinks the task; the operator that
  /// made the input counts its shrinks through it. May be called from any thread.
  void set_shrink_observer(std::function<void()> observer);

  /// Called, with no argument, when shrink_for_retry() charges the runs sliced in place instead
  /// of shrinking. May be called from any thread.
  void set_recharge_observer(std::function<void()> observer);

  /// Runs still left out of the estimate as sliced in place: the constructor's gpu_runs, or 0
  /// once a retry charged them.
  [[nodiscard]] std::size_t gpu_runs() const noexcept;

  /// The sort partition being merged.
  [[nodiscard]] std::size_t partition() const noexcept;

  /// One cursor per run the task reads.
  [[nodiscard]] const std::vector<merge_input_cursor>& cursors() const noexcept;

  /// False when the task writes an intermediate run.
  [[nodiscard]] bool final_pass() const noexcept;

  /// Chunk index of the task's output within the partition.
  [[nodiscard]] std::size_t first_chunk() const noexcept;

  /// Window rows and output target, as shrunk so far.
  [[nodiscard]] const window_config& config() const noexcept;

  /// @throws std::runtime_error if prepare_for_processing did not supply a GPU space.
  [[nodiscard]] cucascade::memory::memory_space& gpu_memory_space() const;

 private:
  std::size_t _partition;
  std::vector<merge_input_cursor> _cursors;
  bool _final_pass;
  std::size_t _first_chunk;
  window_config _cfg;
  std::size_t _row_width;
  std::size_t _key_row_width;
  std::size_t _gpu_runs;
  std::size_t _fallback_bytes;
  std::size_t _compressed_runs;
  std::size_t _fetch_columns;
  std::size_t _string_columns;
  std::size_t _string_row_width;
  bool _round_follows_output;
  std::size_t _kept_buffer_runs{0};
  std::function<void()> _shrink_observer;
  std::function<void()> _recharge_observer;
  cucascade::memory::memory_space* _gpu_memory_space = nullptr;
};

}  // namespace sirius::op::merge
