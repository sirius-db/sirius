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

#include "op/merge/bounded_kway_merge.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/row_range_read.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace sirius::op::merge {

/**
 * @brief A run: batches in key order (one batch, or an intermediate run's batches).
 *
 * batch_ids and batch_rows are parallel: batch_rows[i] is the row count of batch batch_ids[i].
 */
struct merge_run {
  std::vector<std::uint64_t> batch_ids;
  std::vector<std::size_t> batch_rows;

  /// Total rows of the run (Σ batch_rows).
  [[nodiscard]] std::size_t total_rows() const noexcept
  {
    return std::accumulate(batch_rows.begin(), batch_rows.end(), std::size_t{0});
  }
};

/**
 * @brief CPU-only position inside a run: the next unread row is `row` of batch `batch`.
 *
 * Holds no device state, so a task that fails (e.g. OOM) can be retried from the same cursor.
 */
struct merge_input_cursor {
  merge_run run;
  std::size_t batch{0};
  std::size_t row{0};

  /// True when every row of the run has been read (batch == run.batch_ids.size()).
  [[nodiscard]] bool exhausted() const noexcept { return batch == run.batch_ids.size(); }

  /// Unread rows of the run from the cursor to the end.
  [[nodiscard]] std::size_t remaining() const noexcept
  {
    if (batch >= run.batch_rows.size()) { return 0; }
    const auto tail = std::accumulate(run.batch_rows.begin() + static_cast<std::ptrdiff_t>(batch),
                                      run.batch_rows.end(),
                                      std::size_t{0});
    return tail - row;
  }

  /**
   * @brief Move the cursor forward by @p rows, crossing batch boundaries.
   *
   * The cursor is left normalized: it never rests at the end of a batch (nor on an empty batch),
   * so it becomes exhausted() as soon as the last row is passed.
   *
   * @throws std::out_of_range if @p rows exceeds remaining(); the cursor is left unchanged.
   */
  void advance(std::size_t rows)
  {
    if (rows > remaining()) {
      throw std::out_of_range("merge_input_cursor::advance past the end of the run");
    }
    row += rows;
    while (batch < run.batch_rows.size() && row >= run.batch_rows[batch]) {
      row -= run.batch_rows[batch];
      ++batch;
    }
  }
};

/**
 * @brief Sizes of one windowed merge task.
 */
struct window_config {
  std::size_t window_rows{0};          ///< n: maximum rows per input window
  std::size_t output_target_bytes{0};  ///< stop a task once its output reaches this size
};

/**
 * @brief Counters accumulated by windowed_merger::run.
 */
struct merge_round_stats {
  std::size_t rounds{0};                ///< bounded merges performed
  std::size_t rows_out{0};              ///< rows emitted
  std::size_t refill_rows{0};           ///< rows read into window buffers
  std::size_t refill_bytes{0};          ///< data bytes read into window buffers
  std::size_t fallback_conversions{0};  ///< batches converted whole to the GPU tier
};

/// Signature of bounded_kway_merge; lets tests substitute the merge kernel.
using bounded_merge_fn = std::function<bounded_merge_result(std::span<const cudf::table_view>,
                                                            const merge_key_spec&,
                                                            std::span<const std::uint8_t>,
                                                            cudf::size_type,
                                                            rmm::cuda_stream_view,
                                                            rmm::device_async_resource_ref)>;

/**
 * @brief Runs rounds of a bounded k-way merge for one task.
 *
 * GPU-tier batches are windows via cudf::slice of a read-locked view (locked for the whole
 * run()); other tiers use a detail::window_buffer per run, read-locked only during each refill.
 * A window never spans two batches of a run (a short window at a batch end is "truncated").
 * Stops when the accumulated output reaches cfg.output_target_bytes or every cursor is
 * exhausted, and returns ONE table (0 rows only if all inputs were already exhausted).
 * `cursors` are written back only when run() returns normally. Batches without a reader go
 * through detail::convert_run_batch_to_gpu once. Move-only.
 */
class windowed_merger {
 public:
  /**
   * @param runs The resolved batches of each run: runs[i] are the batches of cursors[i].run, in
   *        the same order.
   * @param cursors One cursor per run; must outlive the merger. Read at construction, written
   *        back when run() returns normally.
   * @param fetch_cols Source column indices: keys + projected payload.
   * @param fetch_types Types of the fetched columns, in fetch_cols order.
   * @param keys Sort keys; key_cols index into fetch_cols order.
   * @param cfg Window rows and output target of this task.
   * @param readers Row-range reader registry; must outlive the merger.
   * @param gpu_space The task's GPU memory space; must outlive the merger.
   * @param merge The bounded merge kernel (tests may substitute it).
   */
  windowed_merger(std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> runs,
                  std::span<merge_input_cursor> cursors,
                  std::vector<std::size_t> fetch_cols,
                  std::vector<cudf::data_type> fetch_types,
                  merge_key_spec keys,
                  window_config cfg,
                  const cucascade::row_range_reader_registry& readers,
                  cucascade::memory::memory_space& gpu_space,
                  bounded_merge_fn merge = bounded_kway_merge);

  windowed_merger(windowed_merger&&) noexcept;
  windowed_merger& operator=(windowed_merger&&) noexcept;
  ~windowed_merger();

  /**
   * @brief Merge until the output target is reached or every cursor is exhausted.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used for windows, scratch and the output table.
   * @param stats Optional counters, added to (not reset); may be nullptr.
   * @return ONE table with the fetched columns in fetch_cols order.
   */
  std::unique_ptr<cudf::table> run(rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr,
                                   merge_round_stats* stats = nullptr);

  /// Merger state; defined by the merger's source.
  struct impl;

 private:
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::merge
