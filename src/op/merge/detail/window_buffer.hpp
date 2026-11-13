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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/row_range_read.hpp>
#include <cucascade/memory/memory_reservation.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace sirius::op::merge::detail {

/**
 * @brief Thrown by window_buffer::refill when one source row does not fit a STRING slot, also
 * after the slot was grown once. The slot stays grown, so calling refill again keeps doubling
 * it; consuming the rows the window still holds makes room too.
 */
class window_row_too_large_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief Device window over one non-GPU run of a windowed merge.
 *
 * Two preallocated slots ("ping-pong") holding up to window_rows rows of the fetched columns of
 * ONE non-GPU run. Fixed-width, STRING (int32 offsets unless chars could exceed 2 GiB, then
 * int64), null masks. Nested types unsupported (std::invalid_argument). Move-only.
 *
 * Successive calls may use different streams: work on a new stream is ordered after the work
 * enqueued before it, without blocking the host. The stream of the previous call must still be a
 * live stream then, and the stream of the last call when the window is destroyed. The memory
 * resource must outlive the window. Only refill() can block the host (see there).
 */
class window_buffer {
 public:
  /**
   * @param fetch_types Types of the fetched columns, in fetch order.
   * @param window_rows Row capacity of each slot.
   * @param est_chars_bytes_per_row Estimated chars bytes per row of each fetched column, one
   *        entry per fetched column in fetch order, from batch metadata. Each STRING column's
   *        slots are sized from its own entry (plus a quarter of slack); the entries of the
   *        other columns are ignored.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource both slots are allocated from.
   * @throws std::invalid_argument if a fetched column has a nested type, or the estimates do
   *         not have one entry per fetched column.
   */
  window_buffer(std::vector<cudf::data_type> fetch_types,
                std::size_t window_rows,
                std::vector<std::size_t> est_chars_bytes_per_row,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr);

  /// Same, with one estimate @p est_string_bytes_per_row for every STRING column.
  window_buffer(std::vector<cudf::data_type> fetch_types,
                std::size_t window_rows,
                std::size_t est_string_bytes_per_row,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr);

  window_buffer(window_buffer&&) noexcept;
  window_buffer& operator=(window_buffer&&) noexcept;
  ~window_buffer();

  /**
   * @brief The active slot as a table of size() rows. Invalidated by consume() and refill().
   *
   * Touches no device memory. The null count of a column is an UPPER BOUND, exact only when it
   * is 0: a column declared null-free (set_null_free) has no mask and a null count of 0; every
   * other column carries the slot's mask and a null count of size(). Consumers must read the
   * mask, not the count.
   */
  [[nodiscard]] cudf::table_view view() const;

  /**
   * @brief Declares the columns that hold no NULL in the rows read from now on.
   *
   * @param null_free One flag per fetched column, in fetch order. Stays in force until the next
   *        call; a new window treats every column as possibly holding NULLs.
   * @throws std::logic_error if the window holds rows.
   * @throws std::invalid_argument if @p null_free does not have one flag per fetched column.
   */
  void set_null_free(std::vector<bool> null_free);

  /// Makes @p stream the stream of the window: its work is ordered after the work enqueued by
  /// earlier calls, so a reader of view() on @p stream sees the rows. Does not block the host.
  void use_stream(rmm::cuda_stream_view stream);

  /// Rows currently held in the active slot.
  [[nodiscard]] std::size_t size() const noexcept;

  /// Row capacity of each slot (window_rows).
  [[nodiscard]] std::size_t capacity_rows() const noexcept;

  /// Device bytes the two slots hold now. After refill() grew the chars of the active slot, the
  /// other slot has no chars until the next consume() that carries rows allocates them.
  [[nodiscard]] std::size_t device_bytes() const noexcept;

  /// Times refill() grew the chars of the STRING columns since the window was built.
  [[nodiscard]] std::size_t chars_growths() const noexcept;

  /**
   * @brief Drops the first @p consumed rows; the rest move to the front of the window.
   *
   * Invalidates view(). Enqueues the copy on @p stream without blocking the host; dropping every
   * row touches no device memory. After a refill() that grew the STRING chars, the first call
   * that keeps rows allocates the chars of the other slot from the window's memory resource (and
   * can throw its out-of-memory error; the window then still holds its rows).
   *
   * @throws std::out_of_range if @p consumed exceeds size().
   */
  void consume(std::size_t consumed, rmm::cuda_stream_view stream);

  /**
   * @brief Appends rows read from @p batch starting at source row @p row_begin.
   *
   * Reads up to capacity_rows() - size() rows via cucascade::read_rows_into (caller holds the
   * read lock). Returns the reader result; fewer rows than asked for, with the source not
   * exhausted, means a STRING slot ran out of chars. If a single row does not fit a STRING slot,
   * doubles the chars of the active slot and retries once. The reader does not report which
   * column was short, so every STRING column grows; the growth is allocated from the window's
   * memory resource, whose out-of-memory error propagates. Blocks the host after a consume() that
   * kept rows of a window with STRING columns, until that consume() has run on the device (the
   * chars end is read back); work enqueued after that consume() is not waited for.
   *
   * @param readers Row-range reader registry to dispatch through.
   * @param batch Read lock on the source batch, held by the caller for the duration of the call.
   * @param fetch_cols Source column indices, one per fetched column, in fetch order.
   * @param row_begin First source row to read.
   * @param reader_state Optional slot for reader state kept between calls; may be nullptr.
   * @param scratch Optional reservation for reader temporaries; may be nullptr.
   * @param stream CUDA stream all device work is enqueued on.
   * @throws cucascade::row_range_unsupported_error if the batch's representation has no reader.
   * @throws window_row_too_large_error if one row does not fit a STRING slot after growing once.
   */
  cucascade::row_range_result refill(
    const cucascade::row_range_reader_registry& readers,
    const cucascade::read_only_data_batch& batch,
    std::span<const std::size_t> fetch_cols,
    std::size_t row_begin,
    std::unique_ptr<cucascade::row_range_reader_state>* reader_state,
    cucascade::memory::reservation* scratch,
    rmm::cuda_stream_view stream);

  /// Slot storage; defined by the window buffer's CUDA source.
  struct impl;

 private:
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::merge::detail
