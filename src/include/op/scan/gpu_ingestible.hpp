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
#include <op/scan/scan_filter_analysis.hpp>

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
   * Takes the input by owning unique_ptr so implementations that call
   * @c assemble_scan_output (which consumes its input by rvalue) can
   * move-forward without an extra view→owning copy on the dominant
   * fresh-read + assembly path.
   */
  virtual std::unique_ptr<cudf::table> post_filter_and_project(
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

  /// Whether @ref post_filter_and_project's assembly is a leading-identity
  /// projection: output column k is materialized column k, and no partition or
  /// other synthesized column joins the output. When true, a decode-row-filtered
  /// batch whose width already matches the output arity needs no assembly at
  /// all, so the scan's transactional carrier-cast steal may bypass
  /// post_filter_and_project for it. A width match alone cannot prove this:
  /// trailing pure-filter columns can offset missing synthesized columns.
  /// Conservative default: false (the steal then falls back to the generic
  /// materialize/project path, which is always correct).
  /// Implementations must derive this from their assembly configuration
  /// (parquet: `!needs_output_assembly`) or return a structural constant only
  /// while the invariant is type-level (duckdb-native synthesizes no
  /// partition/virtual output columns and projects via `std::iota` — see its
  /// `post_filter_and_project`). An override returning a stale `true` silently
  /// corrupts decode-row-filtered steals.
  [[nodiscard]] virtual bool output_assembly_is_leading_identity() const noexcept { return false; }

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

  /// This scan's pushed-down filter, digested into the work a decompressor can
  /// do for it: which columns can be answered off a dictionary instead of
  /// decoded, and which rows can be dropped while decoding.
  ///
  /// A source that can exploit this (a Simpatico-compressed pin, whose
  /// dictionary answers an equality off its key set without gathering the
  /// decoded chars) hands it to the decoder as a
  /// @c sirius::pushdown_request; every other source supplies the columns
  /// normally. @ref post_filter_and_project copes with either by reading what
  /// the batch it is handed reports, so the two need not agree.
  ///
  /// Empty by default: an ingestible whose filter path does not implement this
  /// must not advertise work it cannot honour.
  [[nodiscard]] virtual scan_filter_analysis const& filter_analysis() const
  {
    static scan_filter_analysis const none;
    return none;
  }

 protected:
  gpu_ingestible() noexcept = default;
};

}  // namespace scan
}  // namespace sirius::op
