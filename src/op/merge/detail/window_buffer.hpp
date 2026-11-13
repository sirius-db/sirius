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
#include <vector>

namespace sirius::op::merge::detail {

/**
 * @brief Device window over one non-GPU run of a windowed merge.
 *
 * Two preallocated slots ("ping-pong") holding up to window_rows rows of the fetched columns of
 * ONE non-GPU run. Fixed-width, STRING (int32 offsets unless chars could exceed 2 GiB, then
 * int64), null masks. Nested types unsupported (std::invalid_argument). Move-only.
 */
class window_buffer {
 public:
  /**
   * @param fetch_types Types of the fetched columns, in fetch order.
   * @param window_rows Row capacity of each slot.
   * @param est_string_bytes_per_row Estimated chars bytes per row of each STRING column, from
   *        batch metadata; sizes the STRING slots.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource both slots are allocated from.
   * @throws std::invalid_argument if a fetched column has a nested type.
   */
  window_buffer(std::vector<cudf::data_type> fetch_types,
                std::size_t window_rows,
                std::size_t est_string_bytes_per_row,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr);

  window_buffer(window_buffer&&) noexcept;
  window_buffer& operator=(window_buffer&&) noexcept;
  ~window_buffer();

  /// The active slot as a table of size() rows. Invalidated by consume() and refill().
  [[nodiscard]] cudf::table_view view() const;

  /// Rows currently held in the active slot.
  [[nodiscard]] std::size_t size() const noexcept;

  /// Row capacity of each slot (window_rows).
  [[nodiscard]] std::size_t capacity_rows() const noexcept;

  /// Device bytes allocated by both slots.
  [[nodiscard]] std::size_t device_bytes() const noexcept;

  /**
   * @brief Drops the first @p consumed rows.
   *
   * Copies [consumed, size()) to the front of the other slot (unaligned null-mask shift, STRING
   * chars copy + offset rebase) and makes it active.
   */
  void consume(std::size_t consumed, rmm::cuda_stream_view stream);

  /**
   * @brief Appends rows read from @p batch starting at source row @p row_begin.
   *
   * Reads up to capacity_rows() - size() rows via cucascade::read_rows_into (caller holds the
   * read lock). Returns the reader result. If a single row does not fit a STRING slot, grows
   * both slots' chars (doubling) and retries once.
   *
   * @param readers Row-range reader registry to dispatch through.
   * @param batch Read lock on the source batch, held by the caller for the duration of the call.
   * @param fetch_cols Source column indices, one per fetched column, in fetch order.
   * @param row_begin First source row to read.
   * @param state Optional slot for reader state kept between calls; may be nullptr.
   * @param scratch Optional reservation for reader temporaries; may be nullptr.
   * @param stream CUDA stream all device work is enqueued on.
   * @throws cucascade::row_range_unsupported_error if the batch's representation has no reader.
   */
  cucascade::row_range_result refill(const cucascade::row_range_reader_registry& readers,
                                     const cucascade::read_only_data_batch& batch,
                                     std::span<const std::size_t> fetch_cols,
                                     std::size_t row_begin,
                                     std::unique_ptr<cucascade::row_range_reader_state>* state,
                                     cucascade::memory::reservation* scratch,
                                     rmm::cuda_stream_view stream);

  /// Slot storage; defined by the window buffer's CUDA source.
  struct impl;

 private:
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::merge::detail
