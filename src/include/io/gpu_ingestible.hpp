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

// rmm
#include <rmm/cuda_stream_view.hpp>

// cucascade (forward-declare to keep this header light; full include in .cpp)
namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

// standard library
#include <functional>
#include <memory>
#include <vector>

namespace sirius::op {
class operator_data;
}  // namespace sirius::op

namespace sirius::scan_manager {
class sirius_scan_manager;
class split_connector;
}  // namespace sirius::scan_manager

namespace sirius::io {

class gpu_ingestible;

//===----------------------------------------------------------------------===//
// ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-table bind data carrier; polymorphic factory for gpu_ingestible.
 *
 * Built by the pipeline converter when it lowers a DuckDB scan binding,
 * parked on the gpu scan operator until prepare_for_query, then handed to
 * @ref make_gpu_ingestible (or directly to a cached gpu_ingestible when a
 * pinned-cache match wins). Implementations: parquet_ingestible_table_info,
 * duckdb_native_ingestible_table_info.
 */
class ingestible_table_info {
 public:
  virtual ~ingestible_table_info() = default;

  ingestible_table_info(ingestible_table_info const&)            = delete;
  ingestible_table_info& operator=(ingestible_table_info const&) = delete;

  /**
   * @brief Build the format-specific gpu_ingestible for this table.
   *
   * Called once during sirius_scan_manager::prepare_for_query after the
   * pinned-cache short-circuit misses. The override is expected to move
   * format-specific fields off *this into the constructed ingestible;
   * @p self is the owning unique_ptr of *this, threaded through so the
   * ingestible base can co-own the bind data (so accessors like
   * @ref table_info remain valid post-construction).
   *
   * @param self        Owning handle of *this. The override moves it into
   *                    the constructed ingestible's base.
   * @param mgr         Owning scan_manager. Forwarded so the ingestible
   *                    can route each path to its backend via
   *                    @c sirius_scan_manager::io_ctx_shared_for.
   */
  virtual std::shared_ptr<gpu_ingestible> make_ingestible(
    std::unique_ptr<ingestible_table_info> self, scan_manager::sirius_scan_manager const& mgr) = 0;

  /**
   * @brief Resolved file paths captured at bind time.
   *
   * Used by sirius_scan_manager to match an incoming scan against pinned
   * entries before falling back to @ref make_ingestible. Returned span
   * must remain valid for the lifetime of @c *this.
   */
  [[nodiscard]] virtual std::span<std::string const> file_paths() const = 0;

 protected:
  ingestible_table_info() = default;
};

//===----------------------------------------------------------------------===//
// scan_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-split scan descriptor. Polymorphic; each gpu_ingestible
 *        implementation defines its own subclass with the per-split
 *        information its @ref gpu_ingestible::materialize_table requires.
 *
 * Distinct from per-table bind data (@ref ingestible_table_info): one
 * ingestible produces many @c scan_info instances during its lifetime —
 * one per emitted split.
 */
class scan_info {
 public:
  virtual ~scan_info() = default;

  /**
   * @brief Estimated uncompressed byte count for the split.
   *
   * Read by @c scan_operator_input::get_estimated_size_in_bytes for the
   * reservation system; should reflect the GPU memory the materialize step
   * will allocate (parquet: sum of reserved_uncompressed_bytes across the
   * batch's row-group slices; duckdb-native: sum of row_group counts ×
   * column widths). Returns 0 for splits with no a-priori size estimate.
   */
  [[nodiscard]] virtual std::size_t estimated_bytes() const noexcept { return 0; }
};

//===----------------------------------------------------------------------===//
// post_filter_and_projection_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-split post-decode filter + projection description. Optional.
 *
 * Carried on @ref scan_and_filter_metadata when the materialized table needs
 * any of: post-decode DuckDB-expression evaluation (filter pushdown into the
 * parquet reader failed), hive-partition injection, or projection down to
 * the scan's output arity. Cached splits use this for their (rare) filter
 * application path.
 */
class post_filter_and_projection_info {
 public:
  virtual ~post_filter_and_projection_info() = default;
};

//===----------------------------------------------------------------------===//
// filter_state / filtered_table
//===----------------------------------------------------------------------===//
/**
 * @brief How much of the per-split filter + projection work the ingestible
 *        already absorbed during @ref gpu_ingestible::materialize_table.
 *
 * Returned alongside the materialized table so the scan operator can skip
 * a redundant @ref gpu_ingestible::post_filter_and_project call when the
 * ingestible already applied both the row-level filter and projection
 * inline (the parquet reader-side pushdown path).
 */
enum class filter_state {
  UNFILTERED,
  ROWGROUP_FILTERED,
  ROW_FILTERED,
  ROW_FILTERED_AND_PROJECTED,
};

/**
 * @brief Result of @ref gpu_ingestible::materialize_table.
 *
 * Bundles the materialized cudf::table with a tag describing how much of
 * the split's filter + projection work was applied during materialization.
 */
struct filtered_table {
  std::unique_ptr<cudf::table> table;
  filter_state state{filter_state::UNFILTERED};
};

//===----------------------------------------------------------------------===//
// scan_and_filter_metadata
//===----------------------------------------------------------------------===//
/**
 * @brief Pair of per-split scan descriptor and (optional) post-decode
 *        filter/projection description.
 *
 * Owned by @c sirius::op::scan::scan_operator_input — the operator_data
 * pushed into the scan operator's connector for fresh (non-cached) splits.
 * The scan operator inspects @ref has_filter to decide whether to call the
 * ingestible's @ref gpu_ingestible::post_filter_and_project after the
 * materialize step.
 */
class scan_and_filter_metadata {
 public:
  scan_and_filter_metadata(std::unique_ptr<scan_info> scan,
                           std::unique_ptr<post_filter_and_projection_info> filter_and_project)
    : _scan(std::move(scan)), _filter_and_project(std::move(filter_and_project))
  {
  }

  [[nodiscard]] bool has_filter() const noexcept { return _filter_and_project != nullptr; }
  [[nodiscard]] scan_info const& scan() const noexcept { return *_scan; }
  [[nodiscard]] post_filter_and_projection_info const& filter_and_project() const noexcept
  {
    return *_filter_and_project;
  }

 private:
  std::unique_ptr<scan_info> _scan;
  std::unique_ptr<post_filter_and_projection_info> _filter_and_project;
};

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
  explicit gpu_ingestible(std::unique_ptr<ingestible_table_info> info)
    : _table_info(std::move(info))
  {
  }

  virtual ~gpu_ingestible() = default;

  gpu_ingestible(gpu_ingestible const&)            = delete;
  gpu_ingestible& operator=(gpu_ingestible const&) = delete;
  gpu_ingestible(gpu_ingestible&&)                 = delete;
  gpu_ingestible& operator=(gpu_ingestible&&)      = delete;

  /**
   * @brief Snapshot check for remaining work. Thread-safe.
   *
   * Called by @c split_provider::run on the driver thread before claiming
   * the next batch. Implementations typically compare an atomic batch
   * index against a precomputed total.
   */
  [[nodiscard]] virtual bool has_more_splits() const = 0;

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
  virtual std::function<std::vector<std::unique_ptr<op::operator_data>>()>
  next_split_provider() = 0;

  /**
   * @brief Materialize the cudf table for one split. Called by
   *        @c sirius_gpu_scan_operator::execute on the task-local stream.
   *
   * @p mem_space carries both the allocator (via
   * @c get_default_allocator) and the device_id used to select a per-GPU
   * sirius_ioctx for the read — implementations route the read through
   * that ioctx so per-GPU CUDA contexts bind correctly.
   */
  virtual filtered_table materialize_table(scan_info const& info,
                                           ::cucascade::memory::memory_space const& mem_space,
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
    std::unique_ptr<cudf::table> input,
    post_filter_and_projection_info const& info,
    ::cucascade::memory::memory_space const& mem_space,
    rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Produce the next per-task input from the connector.
   *
   * Called on the scan operator's single consumer thread. The default returns
   * the next split unchanged; @c duckdb_native_gpu_ingestible coalesces
   * row-group ranges into cap-sized batches.
   *
   * @return The next per-task operator_data, or nullptr once the connector is
   *         closed and all buffered work has been served.
   */
  virtual std::unique_ptr<op::operator_data> consume_next_input(
    scan_manager::split_connector& connector);

  /**
   * @brief Whether the ingestible still holds buffered consumer-side work.
   *
   * Default: true (no buffer). The duckdb-native override returns false while
   * its coalescer still holds batches, so the scan is not reported done before
   * its last batch is served.
   */
  [[nodiscard]] virtual bool consumer_drained() const { return true; }

  [[nodiscard]] ingestible_table_info const& table_info() const noexcept { return *_table_info; }

 protected:
  std::unique_ptr<ingestible_table_info> _table_info;
};

//===----------------------------------------------------------------------===//
// make_gpu_ingestible
//===----------------------------------------------------------------------===//
/**
 * @brief Free-function wrapper around @ref ingestible_table_info::make_ingestible.
 *
 * Single call site used by @c sirius_scan_manager::prepare_for_query to
 * build the format-specific ingestible from the operator's bind data.
 * Throws when @p info is null.
 */
std::shared_ptr<gpu_ingestible> make_gpu_ingestible(std::unique_ptr<ingestible_table_info> info,
                                                    scan_manager::sirius_scan_manager const& mgr);

}  // namespace sirius::io
