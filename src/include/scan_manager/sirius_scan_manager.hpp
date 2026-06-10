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
#include "io/gpu_ingestible.hpp"
#include "io/s3/s3_blocking_ioctx.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "scan_manager/gpu_ingestible_factory.hpp"
#include "scan_manager/split_provider.hpp"

// Forward-declare sirius_ioctx via <io/types.hpp> for the gpu_ioctxs map type
// used by prepare_for_query / create_provider_for.
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <io/types.hpp>

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
class sirius_gpu_scan_operator;
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
 * every read site (e.g. when the sirius IO path is misbehaving). Only
 * valid in single-GPU configurations — when more than one GPU is
 * configured, @c sirius_config::enforce_sirius_datasource_for_multi_gpu()
 * forces this field to @c true because kvikio's per-FileHandle CUDA-context
 * binding breaks multi-GPU residency.
 */
struct scan_manager_config {
  exec::thread_pool_config thread_pool{.num_threads = 8, .thread_name_prefix = "scan_manager"};
  bool use_sirius_datasource{false};
  /// Reserved (not currently consumed). Intended size of the @c uring_reactor
  /// pool, but the production @c uring_ioctx is built by @c SiriusContext, which
  /// scales the reactor count with the number of GPUs the NUMA node serves
  /// (@c clamp(4 * devices, 4, 16)) rather than reading this field. Parsed from
  /// YAML and kept for forward compatibility / tests; setting it has no effect
  /// on the engine today.
  std::size_t uring_n_reactors{4};
  /// Reserved (not currently consumed). Intended io_uring submission/completion
  /// queue depth per reactor; the production @c uring_ioctx hardcodes 64. Parsed
  /// from YAML and kept for forward compatibility / tests; setting it has no
  /// effect on the engine today.
  unsigned uring_ring_entries{64};
  /// Enable the prefetching cache.  Requires @c use_sirius_datasource=true;
  /// when true, SiriusContext (S6) allocates a pinned-host buffer_pool and
  /// initializes the cache on the IO backends it owns (the per-NUMA urings and
  /// the s3_ioctx).  Off by default.
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

  /// S3 backend opt-in. When set, SiriusContext (S6) constructs an @c s3_ioctx
  /// from these credentials/knobs and hands it to the scan_manager as a borrowed
  /// backend (the scan_manager itself constructs nothing). Default construction
  /// (empty optional) leaves the S3 backend disabled. SiriusContext populates
  /// this from object_store_config during initialize() when the engine config
  /// requests S3.
  std::optional<sirius::io::s3::s3_ioctx_config> s3_config{};

  /// Selects which S3 backend the scan_manager builds when @c s3_config is set.
  /// Mirrors @c object_store_config::s3_use_async_backend (SiriusContext copies
  /// it across in initialize()). When true (default) the manager builds the
  /// async @c s3_ioctx (libcurl-multi reactor: concurrent GETs + pipelined
  /// device reads); when false it builds the serial @c s3_blocking_ioctx fed by
  /// @c s3_thread_pool. Ignored when @c s3_config is empty.
  bool s3_use_async_backend{true};

  /// Thread pool config for the blocking S3 backend's async workers. Ignored
  /// when @c s3_config is empty or when @c s3_use_async_backend is true (the
  /// async @c s3_ioctx reactor owns its own worker thread). Separate from the
  /// main @c thread_pool because S3 I/O has different concurrency
  /// characteristics (more threads, network-bound, not CPU-bound). Injected
  /// into @c s3_ioctx_config::async_thread_pool before constructing the
  /// @c s3_blocking_ioctx so its async paths bypass detached std::thread
  /// fallbacks.
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
  /// this list to match an incoming scan operator's scan_info::file_paths
  /// against this entry, so it can swap in a cached split provider.
  std::vector<std::string> file_paths;
  /// GPU-tier storage: one chunk vector per pinned column name. Populated by
  /// @ref sirius_scan_manager::insert_pinned_entry. Empty when @ref tier is HOST.
  std::unordered_map<std::string, std::vector<std::shared_ptr<cudf::column>>>
    data_batches_by_column;
  /// Per-chunk memory space placement. Parallel to the inner vectors of
  /// data_batches_by_column: chunk_memory_spaces[i] is the memory_space*
  /// for every column's chunk at index i. All columns at chunk index i
  /// share the same memory_space because they came from the same
  /// chunked_parquet_reader::read_chunk() call.
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces;
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
  /// True when the pin was created with a row-count budget (e.g.
  /// `CALL pin_table(..., n_rows=N)`) that capped the captured rows below
  /// the full file content. Partial entries MUST NOT serve cached reads
  /// because a subsequent full scan of the same file paths would silently
  /// return only the pinned prefix.
  bool is_partial{false};
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
   * @brief Construct a new scan manager.
   *
   * The scan_manager owns a single io_context (uring_ioctx) and optionally
   * an S3 backend and a prefetch buffer pool, all created from @p config.
   *
   * @param config   Scan-manager configuration (thread pool, io settings,
   *                 S3 credentials, prefetch cache knobs).
   * @param host_fsmr  Host fixed-size memory resource for S3 backend and
   *                   prefetch cache buffer pool. May be nullptr when prefetch
   *                   cache is disabled and no S3 backend is configured.
   */
  explicit sirius_scan_manager(
    scan_manager_config config,
    cucascade::memory::fixed_size_host_memory_resource* host_fsmr = nullptr);

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
  /// @param gpu_memory_spaces device_id -> GPU memory_space lookup used by the
  ///                     HOST-tier cached_split_provider to materialize host
  ///                     chunks onto the executing GPU. Empty map disables the
  ///                     HOST-tier cache path (queries against a host pin fall
  ///                     through to parquet).
  void prepare_for_query(
    const sirius::planner::query& query,
    std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces = {});

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
  ///     duplicates are dropped. The existing file_paths and chunk_memory_spaces
  ///     are preserved (merge MUST verify chunk_memory_spaces alignment
  ///     between existing and new entry).
  ///   - If row counts differ, the existing entry is dropped and replaced.
  ///
  /// \param name                  Table name key.
  /// \param column_names          Column names in the order returned by the parquet read.
  /// \param file_paths            Resolved file paths captured at pin time (used to match scan
  /// ops).
  /// \param data_tables           Cudf tables produced by chunked parquet reads (may be empty).
  /// \param chunk_memory_spaces   Per-chunk memory space placement (size MUST equal total chunk
  ///                              count across data_tables; value at index i is shared by all
  ///                              columns at chunk i).
  /// \param is_partial            True when the caller capped row capture below the full file
  ///                              content (e.g. pin_table n_rows budget). Partial entries
  ///                              must NOT serve cached reads — see pinned_entry::is_partial.
  void insert_pinned_entry(const std::string& name,
                           std::vector<std::string> column_names,
                           std::vector<std::string> file_paths,
                           std::vector<std::unique_ptr<cudf::table>> data_tables,
                           std::vector<cucascade::memory::memory_space*> chunk_memory_spaces,
                           bool is_partial = false);

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
  /// \param is_partial    True when the caller capped row capture below the full file
  ///                      content (e.g. pin_table n_rows budget). Partial entries
  ///                      must NOT serve cached reads — see pinned_entry::is_partial.
  void insert_pinned_entry_host(
    const std::string& name,
    std::vector<std::string> column_names,
    std::vector<std::string> file_paths,
    std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
    cucascade::memory::memory_space& memory_space,
    bool is_partial = false);

  /// \brief Remove the pinned entry for @p name. No-op if absent.
  void remove_pinned_entry(const std::string& name);

  /// \brief Public read-accessor for the pinned-entries map. Used by unit
  /// tests to assert per-chunk memory_space placement after CALL pin_table.
  /// Const-only — callers cannot mutate the map.
  [[nodiscard]] const std::unordered_map<std::string, pinned_entry>& get_pinned_entries()
    const noexcept
  {
    return _pinned_entries;
  }

  [[nodiscard]] std::shared_ptr<sirius::io::sirius_datasource> create_datasource(
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
  /// \brief Run providers sequentially: start each, wait on its future, advance.
  void start_metadata_processing();

  scan_manager_config _config;
  exec::static_thread_pool _thread_pool;
  std::unique_ptr<exec::scoped_dispatcher> _dispatcher;
  std::unique_ptr<sirius::io::buffer_pool> _prefetch_buffer_pool;
  std::unique_ptr<sirius::exec::static_thread_pool> _s3_thread_pool;
  /// Owned io_context backends, in priority order. The first entry is the
  /// local-file backend (uring_ioctx); subsequent entries are object-store
  /// backends (s3_ioctx). Per-path dispatch in io_ctx_for / io_ctx_shared_for
  /// walks the vector and returns the first whose supports(path) is true.
  std::vector<std::shared_ptr<sirius::io::sirius_ioctx>> _io_ctxs;
  std::unordered_map<op::scan::sirius_gpu_scan_operator*, std::unique_ptr<split_provider>>
    _providers_by_op;
  std::vector<op::scan::sirius_gpu_scan_operator*> _scan_op_order;
  std::unordered_map<std::string, pinned_entry> _pinned_entries;
  /// Produces the right gpu_ingestible per scan source during
  /// prepare_for_query. Holds a borrowed reference to @c _pinned_entries
  /// — declared AFTER it so its constructor's reference is bound to a
  /// fully-constructed member.
  gpu_ingestible_factory _factory;
};

}  // namespace sirius::scan_manager
