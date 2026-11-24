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
  /// n: maximum rows per window of a run read through a window buffer (and its row capacity).
  std::size_t window_rows{0};
  std::size_t output_target_bytes{0};  ///< stop a task once its output reaches this size
  /// Rows one round emits at most, and the most rows a run sliced in place offers a round;
  /// 0 = window_rows. Sized from the output target (round_rows_for) a round fills the task's
  /// output in one bounded merge when its runs are GPU-resident.
  std::size_t round_rows{0};

  /// round_rows, or window_rows when it is 0.
  [[nodiscard]] std::size_t effective_round_rows() const noexcept
  {
    return round_rows == 0 ? window_rows : round_rows;
  }

  /// Whether two configs size a merger alike (a merger is continued only by a task of the same
  /// sizes).
  [[nodiscard]] bool same_sizes(const window_config& other) const noexcept
  {
    return window_rows == other.window_rows && output_target_bytes == other.output_target_bytes &&
           effective_round_rows() == other.effective_round_rows();
  }
};

/**
 * @brief Counters accumulated by windowed_merger::run.
 */
struct merge_round_stats {
  std::size_t rounds{0};                ///< bounded merges performed
  std::size_t rows_out{0};              ///< rows emitted
  std::size_t refill_rows{0};           ///< rows read into window buffers
  std::size_t refill_bytes{0};          ///< data bytes read into window buffers
  std::size_t refill_truncated{0};      ///< refills that a full STRING slot stopped short
  std::size_t window_growths{0};        ///< times a window buffer grew its STRING slots
  std::size_t compacted_rows{0};        ///< buffered rows copied to the front before a refill
  std::size_t fallback_conversions{0};  ///< batches converted whole to the GPU tier
  std::size_t refill_bytes_host{0};     ///< of refill_bytes, read from HOST (uncompressed) runs
  std::size_t refill_bytes_disk{0};     ///< of refill_bytes, read from DISK (uncompressed) runs
  /// Of refill_bytes, read from compressed runs (device, host or disk).
  std::size_t refill_bytes_compressed{0};
  /// (run, run() call) pairs in which a refill read a compressed batch. Not a distinct-run count:
  /// a run compressed over several calls counts once per call, so this is an upper bound on the
  /// compressed runs, equal to it only for a single run() call.
  std::size_t runs_read_compressed{0};
  std::size_t round_rows_min{0};  ///< smallest rows emitted by one round (0 = no round yet)
  std::size_t round_rows_max{0};  ///< largest rows emitted by one round
};

/**
 * @brief Add the counters of @p from to @p into.
 *
 * Every counter is summed (runs_read_compressed too, so a run counts once per call), except
 * round_rows_min / round_rows_max, which take the minimum and the maximum of the two (a
 * round_rows_min of 0 means "no round yet" and never wins).
 */
inline void accumulate(merge_round_stats& into, const merge_round_stats& from) noexcept
{
  into.rounds += from.rounds;
  into.rows_out += from.rows_out;
  into.refill_rows += from.refill_rows;
  into.refill_bytes += from.refill_bytes;
  into.refill_truncated += from.refill_truncated;
  into.window_growths += from.window_growths;
  into.compacted_rows += from.compacted_rows;
  into.fallback_conversions += from.fallback_conversions;
  into.refill_bytes_host += from.refill_bytes_host;
  into.refill_bytes_disk += from.refill_bytes_disk;
  into.refill_bytes_compressed += from.refill_bytes_compressed;
  into.runs_read_compressed += from.runs_read_compressed;
  if (from.round_rows_min != 0 &&
      (into.round_rows_min == 0 || from.round_rows_min < into.round_rows_min)) {
    into.round_rows_min = from.round_rows_min;
  }
  if (from.round_rows_max > into.round_rows_max) { into.round_rows_max = from.round_rows_max; }
}

/// Signature of bounded_kway_merge; lets tests substitute the merge kernel.
using bounded_merge_fn = std::function<bounded_merge_result(std::span<const cudf::table_view>,
                                                            const merge_key_spec&,
                                                            std::span<const std::uint8_t>,
                                                            cudf::size_type,
                                                            rmm::cuda_stream_view,
                                                            rmm::device_async_resource_ref)>;

/// Signature of detail::convert_run_batch_to_gpu; lets tests substitute the whole-batch fallback.
using run_fallback_fn = std::function<bool(
  cucascade::data_batch&, cucascade::memory::memory_space&, rmm::cuda_stream_view)>;

/**
 * @brief Runs rounds of a bounded k-way merge over the runs of one partition, one task per run().
 *
 * Each round merges one window per run: rows of the run's current batch, from its cursor on, and
 * emits at most cfg.effective_round_rows() rows. A window never spans two batches of a run (a
 * short window at a batch end is "truncated"). A batch that is an uncompressed table in the
 * task's GPU space is windowed in place, as a view of at most effective_round_rows() of its rows
 * under a read lock held until the batch is finished or run() ends: the view costs no memory, so
 * its size follows the round, not n. Any other batch is read through its row-range reader into a
 * detail::window_buffer of the run, at most window_rows rows at a time, read-locked only during
 * each refill; a batch without a reader goes through
 * detail::convert_run_batch_to_gpu once. The null count of a window's columns is an upper bound
 * (exact only when 0): the merge kernel must read the masks.
 *
 * Memory: the window buffers, the merge scratch and the output come from the `mr` of run(). The
 * readers allocate from the default allocator of the GPU space, on the stream of the run() that
 * reads: the temporaries of a refill and, for a compressed batch, the decode state its reader
 * keeps with the run between refills (dropped with the run's rows, and by the reader when a
 * later run() uses another stream). The operator passes that same allocator as `mr`, so all of
 * it is charged to the task whose reservation is attached to the stream.
 *
 * run() stops when its output reaches cfg.output_target_bytes or every cursor is exhausted, and
 * returns ONE table (0 rows only if all inputs were already exhausted). `cursors` are written
 * back only when run() returns normally.
 *
 * The merger may be kept for several run() calls (the tasks of one partition). The window
 * buffers live as long as the merger: a later run() continues with the rows an earlier one read
 * but did not emit, as long as the cursors are where that run() left them (rebase_consumed()
 * keeps them consistent); rows buffered for another position are dropped. A run() that throws
 * drops and frees every buffer, so a retry reads them again from the unchanged cursors. No read
 * lock outlives a run().
 *
 * Lifetime: the cursor span, the reader registry, the GPU space and the memory resource the
 * buffers were allocated from (the `mr` of the run() calls, which must be the same resource in
 * every call) must outlive the merger. Each run() may use its own stream; the stream of the
 * previous run() must still be a live stream when the next run() starts, and the stream of the
 * last run() when the merger is destroyed. Not thread-safe: one call at a time. Move-only.
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
   * @param cfg Window rows, round rows and output target of this task.
   * @param readers Row-range reader registry; must outlive the merger.
   * @param gpu_space The task's GPU memory space; must outlive the merger. Batches are converted
   *        to it when they cannot be read by row range, and the readers allocate from its default
   *        allocator.
   * @param merge The bounded merge kernel (tests may substitute it).
   * @param convert The whole-batch fallback; empty for detail::convert_run_batch_to_gpu (tests
   *        may substitute it).
   */
  windowed_merger(std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> runs,
                  std::span<merge_input_cursor> cursors,
                  std::vector<std::size_t> fetch_cols,
                  std::vector<cudf::data_type> fetch_types,
                  merge_key_spec keys,
                  window_config cfg,
                  const cucascade::row_range_reader_registry& readers,
                  cucascade::memory::memory_space& gpu_space,
                  bounded_merge_fn merge  = bounded_kway_merge,
                  run_fallback_fn convert = {});

  windowed_merger(windowed_merger&&) noexcept;
  windowed_merger& operator=(windowed_merger&&) noexcept;
  ~windowed_merger();

  /**
   * @brief Merge until the output target is reached or every cursor is exhausted.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches; may differ
   *        from one call to the next.
   * @param mr Device memory resource used for windows, merge scratch and the output table; the
   *        same resource in every call, and it must outlive the merger. The readers do not use
   *        it: they allocate from the GPU space's default allocator.
   * @param stats Optional counters, added to (not reset); may be nullptr.
   * @return ONE table with the fetched columns in fetch_cols order.
   */
  std::unique_ptr<cudf::table> run(rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr,
                                   merge_round_stats* stats = nullptr);

  /**
   * @brief Forgets the batches every run has read to the end, after a run() that returned.
   *
   * For each run i, drops the merger's references to the first cursors[i].batch batches, erases
   * them from cursors[i].run and sets cursors[i].batch to 0. Rows already buffered are kept, so
   * the next run() continues without reading them again. The caller may then pop those batches.
   * An exhausted run is left with no batches and stays exhausted. Touches no device memory and
   * does not throw.
   */
  void rebase_consumed();

  /// Merger state; defined by the merger's source.
  struct impl;

 private:
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::merge
