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

// cudf
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <op/scan/batch_coalescer.hpp>
#include <op/scan/gpu_ingestible_types.hpp>

// rmm
#include "io/io_context.hpp"

#include <rmm/cuda_stream_view.hpp>

// standard library
#include <concepts>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// cucascade (forward-declare to keep this header light; full include in .cpp)
namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace sirius::scan_manager {
class sirius_scan_manager;
}  // namespace sirius::scan_manager

namespace sirius::op {
class operator_data;

namespace scan {

class gpu_ingestible;
// Forward-declared to break the gpu_ingestible.hpp <-> sirius_gpu_scan_operator_data.hpp
// include cycle; only used by const-reference below. Full definition pulled in by .cpp.
class scan_operator_input;

//===----------------------------------------------------------------------===//
// gpu_ingestible
//===----------------------------------------------------------------------===//
/**
 * @brief Abstract source of cudf tables. One implementation per data format.
 *
 * Composed by @c scan_manager::split_provider, which drives the metadata
 * worker pool via @ref has_more_splits and @ref next_split_provider, and by
 * @c sirius::op::scan::sirius_gpu_scan_operator, which calls
 * @ref materialize_table (and conditionally @ref post_filter_and_project)
 * on each split it pulls off its connector.
 *
 * Implementations today: @c parquet_gpu_ingestible,
 * @c duckdb_native_gpu_ingestible.
 */
class gpu_ingestible : public std::enable_shared_from_this<gpu_ingestible> {
 public:
  using metadata_scan_task_t = std::function<std::unique_ptr<scan_info>()>;

  virtual ~gpu_ingestible() = default;

  gpu_ingestible(gpu_ingestible const&)            = delete;
  gpu_ingestible& operator=(gpu_ingestible const&) = delete;
  gpu_ingestible(gpu_ingestible&&)                 = delete;
  gpu_ingestible& operator=(gpu_ingestible&&)      = delete;

  filtered_table materialize_table(const op::scan::scan_operator_input& split,
                                   rmm::cuda_stream_view stream);

  virtual std::unique_ptr<batch_coalescer> create_batch_coalescer() const = 0;

  /**
   * @brief Snapshot check for remaining work. Thread-safe.
   *
   * Called by @c split_provider::run on the driver thread before claiming
   * the next batch. Implementations typically compare an atomic batch
   * index against a precomputed total.
   */
  [[nodiscard]] virtual bool has_processed_all_metadata() const = 0;

  /**
   * @brief Atomically claim the next batch and return a callable that
   *        produces its operator_data splits. Thread-safe.
   *
   * Splitting the claim from the work lets @c split_provider::run enqueue
   * one task per batch onto the scan_manager's worker pool. The callable
   * returns the splits as a vector of operator_data; an empty vector or a
   * null callable indicates no work was claimed (the driver loop skips
   * empty handoffs).
   */
  virtual metadata_scan_task_t next_split_provider(io::ioctx_resolver resolve) = 0;

  /**
   * @brief Materialize the cudf table for one split. Called by
   *        @c sirius_gpu_scan_operator::execute on the task-local stream.
   *
   * @param mem_space Destination memory space for decoded columns.
   *
   * Implementations allocate through this space's allocator. The caller must
   * make its device current before calling this method. I/O uses the datasource
   * attached to the scan metadata.
   */
  virtual filtered_table materialize_metadata_to_table(
    const scan_info& info,
    const cucascade::memory::memory_space& mem_space,
    rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Apply post-decode filter and/or projection to the materialized
   *        table. Called by @c sirius_gpu_scan_operator::execute whenever
   *        @ref materialize_metadata_to_table did not return
   *        @c filter_state::ROW_FILTERED_AND_PROJECTED.
   *
   * Takes the input by rvalue so implementations can move it through their filter and assembly
   * steps without an extra copy. The result is returned in the handle's natural state — an owned
   * table when a row filter gathered fresh columns, a view selection otherwise — and
   * @c sirius_gpu_scan_operator::execute decides whether to forward that view zero-copy or
   * materialize it.
   */
  virtual owning_table_view post_filter_and_project(
    filtered_table&& input,
    const cucascade::memory::memory_space& mem_space,
    rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Whether this ingestible holds a row-filter expression that
   *        @ref post_filter_and_project applies to splits not already
   *        row-filtered. Drives the working-set estimate of resident
   *        (pinned-cache) splits, which always reach post-filter unfiltered.
   */
  [[nodiscard]] virtual bool has_row_filter() const noexcept { return false; }

  [[nodiscard]] virtual const ingestible_table_info& table_info() const noexcept = 0;

  /// Column primary (storage) indices in the exact order @ref materialize_table emits
  /// them — output columns first (in output order), then pure-filter columns; partition
  /// and virtual columns excluded. This is the layout @ref post_filter_and_project assumes
  /// (its index-based filter refs and projection are expressed in this order).
  ///
  /// The pinned-cache scan path serves cached columns in this order — instead of raw
  /// column_ids order — so a cached batch is laid out identically to a fresh disk read and
  /// @ref post_filter_and_project resolves the same columns on both paths.
  [[nodiscard]] virtual std::vector<std::size_t> materialized_column_order() const = 0;

  /// Columns this scan can consume as a decode-time BOOL8 predicate mask rather
  /// than as values, keyed by column primary (storage) index; the mapped value is
  /// the constant set to test against.
  ///
  /// A column qualifies when its whole pushed-down filter is an equality / IN
  /// over string constants *and* it is never projected — the mask replaces the
  /// column's values, so a projected column could not survive the substitution.
  ///
  /// A source that can exploit this (a Simpatico-compressed pin, whose dictionary
  /// answers the predicate off its key set without gathering the decoded chars)
  /// attaches it via @c sirius::decode_equality_pushdown; every other source
  /// supplies the column normally. @ref post_filter_and_project copes with either
  /// by inspecting the batch it is handed, so the two need not agree.
  ///
  /// Empty by default: an ingestible whose filter path does not implement the
  /// substitution must not advertise candidates.
  [[nodiscard]] virtual std::unordered_map<std::size_t, std::vector<std::string>>
  decode_predicate_candidates() const
  {
    return {};
  }

 protected:
  gpu_ingestible() noexcept = default;
};

}  // namespace scan
}  // namespace sirius::op
