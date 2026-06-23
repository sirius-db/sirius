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

#include <cucascade/data/gpu_data_representation.hpp>
#include <op/scan/batch_coalecer.hpp>
#include <op/scan/gpu_ingestible_types.hpp>

// rmm
#include "io/io_context.hpp"

#include <rmm/cuda_stream_view.hpp>

// standard library
#include <concepts>
#include <functional>
#include <memory>
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
 * @c duckdb_native_gpu_ingestible, @c cached_parquet_gpu_ingestible.
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

  virtual std::unique_ptr<batch_coalecer> create_batch_coalecer() const = 0;

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
  virtual metadata_scan_task_t next_split_provider(std::shared_ptr<io::sirius_ioctx> io_ctx) = 0;

  /**
   * @brief Materialize the cudf table for one split. Called by
   *        @c sirius_gpu_scan_operator::execute on the task-local stream.
   *
   * @p mem_space carries both the allocator (via
   * @c get_default_allocator) and the device_id used to select a per-GPU
   * sirius_ioctx for the read — implementations route the read through
   * that ioctx so per-GPU CUDA contexts bind correctly.
   */
  virtual filtered_table materialize_metadata_to_table(
    const scan_info& info,
    const cucascade::memory::memory_space& mem_space,
    rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Apply post-decode filter and/or projection to the materialized
   *        table. Called by @c sirius_gpu_scan_operator::execute when the
   *        split carries a non-null @ref post_filter_and_projection_info,
   *        or when a pinned-cache batch needs filter/assembly.
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

  [[nodiscard]] virtual const ingestible_table_info& table_info() const noexcept = 0;

 protected:
  gpu_ingestible() noexcept = default;
};

}  // namespace scan
}  // namespace sirius::op
