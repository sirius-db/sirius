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

#include "exec/config.hpp"
#include "exec/thread_pool.hpp"
#include "scan_manager/split_provider.hpp"

// Phase 20.6 IO-MGPU-02: forward-declare sirius_ioctx via <io/types.hpp>
// for the gpu_ioctxs map type used by prepare_for_query / create_provider_for.
#include <io/types.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {
class sirius_gpu_parquet_scan_operator;
}  // namespace sirius::op::scan

namespace sirius::planner {
class query;
}  // namespace sirius::planner

namespace sirius::scan_manager {

/**
 * @brief A single pinned-table entry, keyed by table name in the scan_manager.
 *
 * Stores the column projection captured at pin time (so the scan side knows
 * which columns the user pinned) along with the data batches making up the
 * pinned table. The vector may be empty until splits are populated.
 */
struct pinned_entry {
  std::vector<std::string> column_names;
  /// Resolved (globbed) file paths captured at pin time. The scan_manager uses
  /// this list to match an incoming scan operator's parquet_scan_info::file_paths
  /// against this entry, so it can swap in a cached split provider.
  std::vector<std::string> file_paths;
  std::unordered_map<std::string, std::vector<std::shared_ptr<cudf::column>>>
    data_batches_by_column;
  /// Memory space the pinned columns reside in. Captured at pin time so the
  /// cached_split_provider can wrap copied tables as data_batch instances.
  cucascade::memory::memory_space* memory_space{nullptr};
  /// Total number of rows across all pinned chunks. Used by insert_pinned_entry
  /// to decide whether a re-insert merges into the existing entry (same row
  /// count → add unique columns) or replaces it (different row count).
  std::size_t num_rows{0};
};

/**
 * @brief Manages scan-side preparation for a query.
 *
 * The scan manager owns a configurable-size thread pool and is given a chance
 * to set up per-scan state before a query runs (via prepare_for_query).
 */
class sirius_scan_manager {
 public:
  /**
   * @brief Construct a new source manager.
   *
   * @param config Configuration for the thread pool (thread count, name prefix, CPU affinity).
   */
  explicit sirius_scan_manager(exec::thread_pool_config config);

  ~sirius_scan_manager();

  // Non-copyable and non-movable
  sirius_scan_manager(const sirius_scan_manager&)            = delete;
  sirius_scan_manager& operator=(const sirius_scan_manager&) = delete;
  sirius_scan_manager(sirius_scan_manager&&)                 = delete;
  sirius_scan_manager& operator=(sirius_scan_manager&&)      = delete;

  /// \brief Prepare per-scan state for the given query.
  ///
  /// Walks @p query 's pipelines in scan-operator order. For each GPU parquet
  /// scan source, the factory builds a split_provider from the operator's
  /// scan_info, installs a fresh split_connector on the operator, and stores
  /// the provider in a map keyed by the operator. A driver thread then runs
  /// the providers SEQUENTIALLY in registration order: provider[0] starts,
  /// when its future completes provider[1] starts, and so on. Consumers (the
  /// gpu scan operators) block in split_connector::get_next_split until splits
  /// arrive or the connector is closed, so no separate wake-up channel is
  /// needed.
  ///
  /// @param query        The query whose scan operators must be prepared.
  /// @param gpu_ioctxs   Per-GPU sirius_ioctx instances (Phase 20.6 IO-MGPU-02).
  ///                     Forwarded to parquet_split_provider so that footer
  ///                     metadata reads route through io_uring instead of
  ///                     cudf's bundled kvikio path. Empty map is permitted
  ///                     for callers (test harnesses) that pre-populate
  ///                     scan_info another way; production callers
  ///                     (SiriusContext::create_query) MUST pass the map
  ///                     from SiriusContext::get_gpu_ioctxs().
  void prepare_for_query(
    const sirius::planner::query& query,
    std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& gpu_ioctxs = {});

  /// \brief Clear the providers map and join the driver thread if it is
  ///        still running.
  void reset();

  /// \brief Start the worker thread pool. Idempotent.
  void start();

  /// \brief Stop the worker thread pool and the driver. Idempotent.
  void stop();

  /// \brief Pin the entry for a table.
  ///
  /// Releases the columns of each input @p data_tables into the entry's per-column
  /// map, keyed by @p column_names (the i-th column of every table is appended to
  /// @c data_batches_by_column[column_names[i]]). Tables become empty after this
  /// call.
  ///
  /// Re-insert semantics:
  ///   - If no entry exists for @p name, a fresh one is created.
  ///   - If an entry exists and its @c num_rows equals the new total row count,
  ///     only columns whose names are not already present are merged in;
  ///     duplicates are dropped. The existing file_paths and memory_space are
  ///     preserved.
  ///   - If row counts differ, the existing entry is dropped and replaced.
  ///
  /// \param name          Table name key.
  /// \param column_names  Column names in the order returned by the parquet read.
  /// \param file_paths    Resolved file paths captured at pin time (used to match scan ops).
  /// \param data_tables   Cudf tables produced by chunked parquet reads (may be empty).
  /// \param memory_space  Memory space the columns reside in.
  void insert_pinned_entry(const std::string& name,
                           std::vector<std::string> column_names,
                           std::vector<std::string> file_paths,
                           std::vector<std::unique_ptr<cudf::table>> data_tables,
                           cucascade::memory::memory_space& memory_space);

  /// \brief Remove the pinned entry for @p name. No-op if absent.
  void remove_pinned_entry(const std::string& name);

 private:
  /// \brief Build a split_provider for @p op by reading its parquet scan_info
  ///        and installing the resulting hive-partition inject_fn (if any) on
  ///        the operator.
  ///
  /// @param op           The parquet scan operator.
  /// @param gpu_ioctxs   Per-GPU sirius_ioctx map forwarded to
  ///                     parquet_split_provider (Phase 20.6 IO-MGPU-02).
  std::unique_ptr<split_provider> create_provider_for(
    op::scan::sirius_gpu_parquet_scan_operator* op,
    std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& gpu_ioctxs);

  /// \brief Run providers sequentially: start each, wait on its future, advance.
  void run_driver_loop();

  exec::thread_pool_config _config;
  std::unique_ptr<exec::thread_pool> _thread_pool;
  std::unordered_map<op::scan::sirius_gpu_parquet_scan_operator*, std::unique_ptr<split_provider>>
    _providers_by_op;
  std::vector<op::scan::sirius_gpu_parquet_scan_operator*> _scan_op_order;
  std::unordered_map<std::string, pinned_entry> _pinned_entries;
  std::thread _driver_thread;
};

}  // namespace sirius::scan_manager
