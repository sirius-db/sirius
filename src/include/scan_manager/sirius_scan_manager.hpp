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
#include "exec/scoped_dispatcher.hpp"
#include "exec/thread_pool.hpp"
#include "scan_manager/split_provider.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

namespace cucascade::memory {
class fixed_size_host_memory_resource;
}  // namespace cucascade::memory

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::io {
class sirius_ioctx;
class buffer_pool;
}  // namespace sirius::io

namespace sirius::op::scan {
class sirius_gpu_parquet_scan_operator;
struct scan_info;
}  // namespace sirius::op::scan

namespace sirius::planner {
class query;
}  // namespace sirius::planner

namespace sirius::scan_manager {

/**
 * @brief Configuration for the scan_manager.
 *
 * @c use_sirius_datasource controls whether the manager builds a
 * @c sirius_ioctx and routes parquet reads through @c sirius_datasource.
 * Set to @c false to fall back to @c cudf::io::datasource::create() at
 * every read site (e.g. when the sirius IO path is misbehaving).
 */
struct scan_manager_config {
  exec::thread_pool_config thread_pool{.num_threads = 8, .thread_name_prefix = "scan_manager"};
  bool use_sirius_datasource{false};
  /// Number of @c uring_reactor instances in the ioctx pool.  Ignored when
  /// @c use_sirius_datasource is false.
  std::size_t uring_n_reactors{4};
  /// io_uring submission/completion queue depth per reactor.  Ignored when
  /// @c use_sirius_datasource is false.
  unsigned uring_ring_entries{64};
  /// Enable the prefetching cache.  Requires @c use_sirius_datasource=true;
  /// when true, the scan_manager allocates a pinned-host buffer_pool and
  /// initializes the ioctx's cache.  Off by default.
  bool enable_prefetch_cache{false};
  /// Total pinned-host bytes reserved for the prefetch cache.  Rounded
  /// up to the nearest 500 MiB slab.  Ignored when
  /// @c enable_prefetch_cache is false.
  std::size_t prefetch_buffer_pool_bytes{20ULL << 30};
  /// Maximum chunks the cache may have in flight at once (admission
  /// control).  Ignored when @c enable_prefetch_cache is false.
  std::size_t prefetch_inflight_budget_chunks{2048};
};

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
  /// this list to match an incoming scan operator's scan_info::file_paths
  /// against this entry, so it can swap in a cached split provider.
  std::vector<std::string> file_paths;
  /// GPU-tier storage: one chunk vector per pinned column name. Populated by
  /// @ref sirius_scan_manager::insert_pinned_entry. Empty when @ref tier is HOST.
  std::unordered_map<std::string, std::vector<std::shared_ptr<cudf::column>>>
    data_batches_by_column;
  /// HOST-tier storage: one host_data_representation per chunk, each holding all
  /// pinned columns. The cached_split_provider slices these by column index when
  /// serving a particular scan. Populated by @ref insert_pinned_entry_host.
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks;
  /// Tier the pinned data resides in. Drives which storage member above is used
  /// and which cached_split_provider variant @ref create_provider_for builds.
  cucascade::memory::Tier tier{cucascade::memory::Tier::GPU};
  /// Memory space the pinned data resides in. Captured at pin time so the
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
   * @param config Scan-manager configuration (thread pool + sirius_datasource toggle).
   * @param host_mr Host memory resource backing the prefetch buffer_pool.  Required
   *        when @c config.enable_prefetch_cache is true; ignored otherwise.
   */
  explicit sirius_scan_manager(
    scan_manager_config config,
    cucascade::memory::fixed_size_host_memory_resource* host_mr = nullptr);

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
  void prepare_for_query(const sirius::planner::query& query);

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

  /// \brief Pin the entry for a table on the host tier.
  ///
  /// Each entry in @p host_chunks describes one batch's worth of pinned data
  /// (covering all pinned columns) as a host_data_representation. The
  /// cached_split_provider built from this entry slices each chunk by column
  /// index at scan time. Re-insert with a different row count drops the
  /// existing entry; otherwise the call replaces the entry's chunks.
  ///
  /// \param name          Table name key.
  /// \param column_names  Column names in the order the chunks were captured (i.e.
  ///                      the i-th column in each host_data_representation
  ///                      corresponds to @c column_names[i]).
  /// \param file_paths    Resolved file paths captured at pin time.
  /// \param host_chunks   One host_data_representation per emitted batch.
  /// \param memory_space  Host memory space the chunks reside in.
  void insert_pinned_entry_host(
    const std::string& name,
    std::vector<std::string> column_names,
    std::vector<std::string> file_paths,
    std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
    cucascade::memory::memory_space& memory_space);

  /// \brief Remove the pinned entry for @p name. No-op if absent.
  void remove_pinned_entry(const std::string& name);

  /// \brief Process-wide ioctx used to mint @c sirius_datasource instances.
  ///        Returns nullptr when the manager was configured with
  ///        @c use_sirius_datasource=false.
  [[nodiscard]] sirius::io::sirius_ioctx* io_ctx() const noexcept { return _io_ctx.get(); }

  /// \brief Same ioctx as @ref io_ctx(), shared so format-specific scan_info
  ///        subclasses can attach it to a split_provider's emitted slices.
  [[nodiscard]] std::shared_ptr<sirius::io::sirius_ioctx> shared_io_ctx() const noexcept
  {
    return _io_ctx;
  }

 private:
  /// \brief Build a split_provider for @p op by reading its scan_info.
  ///        Tries the pinned-cache short-circuit (format-agnostic; uses only
  ///        the common scan_info fields) and otherwise dispatches through
  ///        scan_info::make_provider() to the format-specific provider.
  std::unique_ptr<split_provider> create_provider_for(
    op::scan::sirius_gpu_parquet_scan_operator* op);

  /// \brief Try to short-circuit @p info to a cached_split_provider when a
  ///        pinned entry matches the scan's file paths. Returns nullptr on
  ///        miss so the caller can fall through to per-format dispatch.
  ///        Format-agnostic: uses only fields on the scan_info base.
  ///
  /// @param info   Scan bind-data — file_paths, column_ids, names, types,
  ///               projection_ids, partition_indices, table_filters,
  ///               scan_output_arity. Read-only; not consumed.
  /// @param op_id  Operator id for diagnostic logging.
  std::unique_ptr<split_provider> try_cached_for(op::scan::scan_info const& info,
                                                 std::size_t op_id);

  /// \brief Run providers sequentially: start each, wait on its future, advance.
  void start_metadata_processing();

  scan_manager_config _config;
  /// Pinned-host buffer pool backing the ioctx's prefetching cache.
  /// Constructed only when @c _config.enable_prefetch_cache is set.
  /// MUST be declared before @c _io_ctx so the ioctx (and its cache,
  /// which references the pool) is destroyed first.
  std::unique_ptr<sirius::io::buffer_pool> _buffer_pool;
  exec::static_thread_pool _thread_pool;
  std::unique_ptr<exec::scoped_dispatcher> _dispatcher;
  std::shared_ptr<sirius::io::sirius_ioctx> _io_ctx;
  std::unordered_map<op::scan::sirius_gpu_parquet_scan_operator*, std::unique_ptr<split_provider>>
    _providers_by_op;
  std::vector<op::scan::sirius_gpu_parquet_scan_operator*> _scan_op_order;
  std::unordered_map<std::string, pinned_entry> _pinned_entries;
};

}  // namespace sirius::scan_manager
