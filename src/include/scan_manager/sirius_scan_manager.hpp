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
#include "io/s3/s3_ioctx.hpp"
#include "scan_manager/split_provider.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

namespace cucascade::memory {
class fixed_size_host_memory_resource;
}  // namespace cucascade::memory

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::io {
class sirius_ioctx;
class buffer_pool;
}  // namespace sirius::io

namespace sirius::op::scan {
class sirius_gpu_parquet_scan_operator;
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

  /// When true (default — current behavior), parquet_split_provider prewarms
  /// per-row-group column-chunk byte ranges via @c cache->insert(obj,
  /// metadata, ranges).  When false, prewarm is skipped: insert is called
  /// with empty ranges (metadata-only, as in §24 describe_parquet).  Lets
  /// the B1 micro-bench A/B compare prefetch overlap on SF10.  Ignored when
  /// @c enable_prefetch_cache is false (no cache → no prewarm regardless).
  bool enable_chunk_prewarm{true};

  /// S3 backend opt-in. When set, scan_manager constructs an @c s3_ioctx
  /// alongside the uring_ioctx using these credentials/knobs. Default
  /// construction (empty optional) leaves the S3 backend disabled.
  /// SiriusContext populates this from object_store_config during
  /// initialize() when the engine config requests S3.
  std::optional<sirius::io::s3::s3_ioctx_config> s3_config{};

  /// Thread pool config for S3 async workers. Ignored when @c s3_config
  /// is empty. Separate from the main @c thread_pool because S3 I/O has
  /// different concurrency characteristics (more threads, network-bound,
  /// not CPU-bound). Injected into @c s3_ioctx_config::async_thread_pool
  /// before constructing the s3_ioctx so async S3 paths bypass detached
  /// std::thread fallbacks.
  exec::thread_pool_config s3_thread_pool{.num_threads = 8, .thread_name_prefix = "s3_io"};
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
  /// this list to match an incoming scan operator's parquet_scan_info::file_paths
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
 * @brief Bind-time result of @ref sirius_scan_manager::describe_parquet.
 *
 * Carries the column types and names a parquet file's footer yields, ready to
 * be copied into a DuckDB table function's bind out-parameters, plus the total
 * object size in bytes.
 */
struct parquet_bind_result {
  duckdb::vector<duckdb::LogicalType> return_types;
  duckdb::vector<std::string> names;
  std::size_t object_size{0};
  std::size_t total_num_rows{0};
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
  ///        Returns the FIRST registered ioctx (currently uring) for
  ///        backward compatibility with callers that don't yet do
  ///        per-path dispatch. Returns nullptr when @c use_sirius_datasource
  ///        is false AND no S3 backend is configured.
  [[nodiscard]] sirius::io::sirius_ioctx* io_ctx() const noexcept
  {
    return _io_ctxs.empty() ? nullptr : _io_ctxs.front().get();
  }

  /// \brief Per-path dispatch: returns the first registered ioctx whose
  ///        @c supports(path) is true. Returns nullptr when no backend
  ///        claims @p path. Used by @c parquet_split_provider::run_batch
  ///        to route each file to its supporting backend (local-disk
  ///        paths to @c uring_ioctx, @c s3:// paths to @c s3_ioctx, etc.).
  [[nodiscard]] sirius::io::sirius_ioctx* io_ctx_for(std::string_view path) const noexcept;

  /// \brief Same as @c io_ctx_for but returns a shared_ptr — needed by
  ///        @c parquet_split_provider to thread ioctx ownership through
  ///        each emitted @c row_group_slice. Returns an empty shared_ptr
  ///        when no backend supports @p path.
  [[nodiscard]] std::shared_ptr<sirius::io::sirius_ioctx> io_ctx_shared_for(
    std::string_view path) const noexcept;

  /// \brief Whether parquet_split_provider should prewarm column-chunk byte
  /// ranges via @c cache->insert(obj, metadata, ranges). Mirrors
  /// @c scan_manager_config::enable_chunk_prewarm. False disables the
  /// prewarm (insert is called with empty ranges — metadata-only, §24
  /// describe_parquet shape), letting B1 micro-bench A/B prefetch overlap.
  [[nodiscard]] bool chunk_prewarm_enabled() const noexcept { return _config.enable_chunk_prewarm; }

  /// \brief Probe a parquet file's schema for the SQL bind path.
  ///
  /// Resolves @p uri to a backend via @c io_ctx_for, fetches only the parquet
  /// footer (no full-file download), and infers the column types and names.
  /// When the resolved backend has a prefetch cache, the parsed footer is
  /// inserted as metadata-only so a subsequent scan reuses it instead of
  /// fetching and parsing the footer a second time.
  ///
  /// This is the C++ entry point behind the @c sirius_read_parquet table
  /// function's bind callback.
  ///
  /// \throws std::runtime_error when no backend supports @p uri, or when the
  ///         footer fetch / schema inference fails.
  [[nodiscard]] parquet_bind_result describe_parquet(std::string const& uri);

 private:
  /// \brief Build a split_provider for @p op by reading its parquet scan_info.
  ///        Returns a cached_split_provider when a pinned entry matches, otherwise
  ///        a parquet_split_provider; in both cases the provider carries the
  ///        scan_plan that the operator's execute() consults for output assembly.
  std::unique_ptr<split_provider> create_provider_for(
    op::scan::sirius_gpu_parquet_scan_operator* op);

  /// \brief Run providers sequentially: start each, wait on its future, advance.
  void start_metadata_processing();

  scan_manager_config _config;
  /// Pinned-host buffer pool backing the ioctx's prefetching cache.
  /// Constructed only when @c _config.enable_prefetch_cache is set.
  /// MUST be declared before @c _io_ctxs so the ioctxs (and their cache,
  /// which references the pool) are destroyed first.
  std::unique_ptr<sirius::io::buffer_pool> _buffer_pool;
  exec::static_thread_pool _thread_pool;
  /// Dedicated thread pool for S3 async paths. Constructed only when
  /// @c _config.s3_config is set. Owned by scan_manager; injected into
  /// the s3_ioctx_config before constructing the s3 backend so async
  /// paths bypass detached std::thread fallbacks. MUST be declared before
  /// @c _io_ctxs so it outlives any in-flight S3 task on stop.
  std::unique_ptr<exec::static_thread_pool> _s3_thread_pool;
  std::unique_ptr<exec::scoped_dispatcher> _dispatcher;
  /// All registered ioctx backends, in priority order. The first entry is
  /// typically the local-file backend (@c uring_ioctx) and subsequent
  /// entries are object-store backends (@c s3_ioctx). Per-path dispatch
  /// in @c io_ctx_for / @c io_ctx_shared_for walks the vector and returns
  /// the first whose @c supports(path) is true.
  std::vector<std::shared_ptr<sirius::io::sirius_ioctx>> _io_ctxs;
  std::unordered_map<op::scan::sirius_gpu_parquet_scan_operator*, std::unique_ptr<split_provider>>
    _providers_by_op;
  std::vector<op::scan::sirius_gpu_parquet_scan_operator*> _scan_op_order;
  std::unordered_map<std::string, pinned_entry> _pinned_entries;
};

}  // namespace sirius::scan_manager
