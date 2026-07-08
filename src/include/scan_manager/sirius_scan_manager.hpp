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

#include "exec/scoped_dispatcher.hpp"
#include "exec/thread_pool.hpp"
#include "io/datasource_factory.hpp"
#include "io/sirius_datasource.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/duckdb_mvcc_metadata.hpp"
#include "scan_manager/load_balancing_scan_batch_coalescer.hpp"
#include "scan_manager/split_provider.hpp"

// Forward-declare sirius_ioctx via <io/types.hpp> for the gpu_ioctxs map type
// used by prepare_for_query / create_provider_for.
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <io/types.hpp>

namespace cucascade::memory {
class fixed_size_host_memory_resource;
}  // namespace cucascade::memory

#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cucascade::memory {
class memory_reservation_manager;
}  // namespace cucascade::memory

namespace sirius::memory {
class topology_index;
}  // namespace sirius::memory

namespace sirius::io {
class sirius_ioctx;
namespace cache {
class buffer_pool;
}  // namespace cache
}  // namespace sirius::io

namespace sirius::op::scan {
class sirius_gpu_scan_operator;
class gpu_ingestible;
}  // namespace sirius::op::scan

namespace sirius::scan_manager {
class load_balancing_scan_batch_coalescer;
}  // namespace sirius::scan_manager

namespace sirius::planner {
class query;
}  // namespace sirius::planner

namespace sirius::scan_manager {

/// Lightweight descriptor of a pinned table's cache identity + column layout,
/// stored on @ref pinned_entry in place of the read-side ingestible_table_info.
/// Captures only what serving needs — the table's identity (parquet file set OR
/// duckdb catalog/schema/table name), the cached columns (by primary/storage
/// index), and their names (aligned with @c column_ids) for the GPU gather — and
/// owns the match logic that @ref sirius_scan_manager::try_assign_cached_entries consults.
class cache_entry_info {
 public:
  std::vector<std::string> resolved_file_paths;    ///< parquet identity (file set)
  std::string catalog_name;                        ///< duckdb identity: catalog (attach alias)
  std::string schema_name;                         ///< duckdb identity: schema
  std::string table_name;                          ///< duckdb identity: table
  duckdb::vector<duckdb::ColumnIndex> column_ids;  ///< cached columns, by primary index
  std::vector<std::string> names;                  ///< aligned with column_ids; gather keys

  /// Build the cache descriptor from a read-side ingestible_table_info (parquet
  /// or duckdb-native): captures the format's identity, the kept @c column_ids,
  /// and the @c column_ids-aligned column names.
  [[nodiscard]] static cache_entry_info from(const op::scan::ingestible_table_info& info);

  /// Gather projection (positions into @c column_ids) that lets this cached entry
  /// serve @p other — matching identity (same parquet file set / same duckdb
  /// catalog.schema.table) AND a superset of @p other's requested columns. The projection
  /// reproduces @p other's requested column order. Empty when this entry cannot
  /// serve @p other (different format, identity, or a missing column).
  [[nodiscard]] std::vector<std::size_t> can_serve_with_columns(
    const op::scan::ingestible_table_info& other) const;

  /// Column names in @c column_ids order — the keys @c data_batches_by_column uses.
  [[nodiscard]] const std::vector<std::string>& column_names() const { return names; }
};

/**
 * @brief A single pinned-table entry, keyed by table name in the scan_manager.
 *
 * Stores the column projection captured at pin time (so the scan side knows
 * which columns the user pinned) along with the data batches making up the
 * pinned table. The vector may be empty until splits are populated.
 */
struct pinned_entry {
  /// Cache identity + column layout for this pinned table. Drives the cache-hit
  /// match (@ref cache_entry_info::can_serve_with_columns) and the per-column
  /// gather; replaces the heavyweight read-side ingestible_table_info.
  cache_entry_info cache_info;

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
  /// MVCC snapshot metadata for duckdb-native pins, attached by
  /// @ref sirius_scan_manager::attach_mvcc_metadata right after insert. nullptr
  /// for parquet pins (immutable sources need no visibility reconciliation).
  std::unique_ptr<duckdb_mvcc_metadata> mvcc;
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
   * @param config Scan-manager configuration (thread pool + sirius_datasource toggle).
   * @param reservation_manager Memory reservation manager for GPU memory.
   * @param topology_index Hardware GPU/NUMA topology index.  Drives round-robin
   *        GPU assignment for scans and is forwarded to the prefetching cache.
   */
  sirius_scan_manager(const scan_manager_config& config,
                      cucascade::memory::memory_reservation_manager& reservation_manager,
                      std::shared_ptr<const sirius::memory::topology_index> topology_index);

  ~sirius_scan_manager();

  // Non-copyable and non-movable
  sirius_scan_manager(const sirius_scan_manager&)            = delete;
  sirius_scan_manager& operator=(const sirius_scan_manager&) = delete;
  sirius_scan_manager(sirius_scan_manager&&)                 = delete;
  sirius_scan_manager& operator=(sirius_scan_manager&&)      = delete;

  using ingestible_table_info = op::scan::ingestible_table_info;

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
  void prepare_for_query(const sirius::planner::query& query);

  /// \brief Clear the providers map and join the driver thread if it is
  ///        still running.
  void reset();

  /// \brief Start the worker thread pool. Idempotent.
  void start();

  /// \brief Stop the worker thread pool and the driver. Idempotent.
  void stop();

  /// \brief Pin (or extend) the GPU-tier entry for a table.
  ///
  /// Releases the columns of each input @p data_tables into the entry's per-column
  /// map, keyed by the column names carried in @p cache_info (the i-th column of
  /// every table is appended to @c data_batches_by_column[cache_info.column_names()[i]]).
  /// Tables become empty after this call.
  ///
  /// Re-insert semantics (keyed by @p name):
  ///   - If no entry exists for @p name, a fresh one is created.
  ///   - If an entry exists and its @c num_rows equals the new total row count, the
  ///     incoming columns whose names are not already present are merged in
  ///     (duplicate columns are dropped), and the entry's @c cache_info is extended
  ///     to the union of pinned columns so later cache-hit matching can serve them.
  ///     The existing cache identity is preserved; the merge requires the incoming
  ///     @p chunk_memory_spaces to be identical to the existing entry's and rejects
  ///     any mismatch.
  ///   - If row counts differ, the existing entry is dropped and replaced. (An
  ///     n_rows-capped "partial" pin therefore never merges with a full pin of the
  ///     same table, since their row counts differ.)
  ///
  /// \param name                  Table name key.
  /// \param cache_info            Cache identity (parquet file set or duckdb
  ///                              catalog.schema.table) plus the cached columns by
  ///                              primary index and their @c column_ids-aligned names;
  ///                              drives later cache-hit matching and the per-column gather.
  /// \param data_tables           Cudf tables produced by chunked reads, one per chunk
  ///                              (may be empty). Each table's column count MUST equal
  ///                              the number of columns described by @p cache_info.
  /// \param chunk_memory_spaces   Per-chunk memory space placement; size MUST equal
  ///                              data_tables.size() (the value at index i is shared by
  ///                              all columns at chunk i).
  void insert_pinned_entry(const std::string& name,
                           cache_entry_info cache_info,
                           std::vector<std::unique_ptr<cudf::table>> data_tables,
                           std::vector<cucascade::memory::memory_space*> chunk_memory_spaces);

  /// \brief Pin the host-tier entry for a table.
  ///
  /// Each entry in @p host_chunks describes one batch's worth of pinned data
  /// (covering all pinned columns) as a host_data_representation. The
  /// cached_split_provider built from this entry slices each chunk by column
  /// index at scan time. This path always REPLACES any existing entry for @p name
  /// — there is no per-column merge analog to the GPU path because the
  /// chunk-vs-column dimensions are flipped (each chunk already holds every column).
  ///
  /// \param name          Table name key.
  /// \param cache_info    Cache identity plus the cached columns and their
  ///                      @c column_ids-aligned names (the i-th column in each
  ///                      host_data_representation corresponds to
  ///                      @c cache_info.column_names()[i]); drives cache-hit matching.
  /// \param host_chunks   One host_data_representation per emitted batch.
  /// \param memory_space  Representative host memory space the chunks reside in
  ///                      (metadata only; each chunk carries its own per-GPU
  ///                      NUMA-local memory_space).
  void insert_pinned_entry_host(
    const std::string& name,
    cache_entry_info cache_info,
    std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
    cucascade::memory::memory_space& memory_space);

  /// \brief Attach MVCC snapshot metadata to the pinned entry for @p name.
  ///
  /// Called by the duckdb-format pin path immediately after insert_pinned_entry /
  /// insert_pinned_entry_host. Overwrites any previous metadata: on a re-pin that
  /// merged into an existing entry, the refreshed (newer) v_base is the more
  /// conservative snapshot fence for every cached column, and the refreshed
  /// per-chunk counts stay valid for every column because the merge path rejects
  /// materializations whose per-chunk row counts differ from the existing
  /// chunks'. Throws std::invalid_argument when no entry exists for @p name.
  void attach_mvcc_metadata(const std::string& name, duckdb_mvcc_metadata metadata);

  /// \brief Remove the pinned entry for @p name. No-op if absent.
  void remove_pinned_entry(const std::string& name);

  void visit_pinned_entries(
    const std::function<bool(std::string_view, const pinned_entry&)>& visitor) const;

  parquet_bind_result describe_parquet(std::string const& uri);

  /// \brief Process-wide ioctx used to mint @c sirius_datasource instances.
  ///        Returns nullptr when the manager was configured with
  ///        @c use_sirius_datasource=false.
  [[nodiscard]] sirius::io::sirius_ioctx* io_ctx() const noexcept { return _io_ctx.get(); }

  [[nodiscard]] std::shared_ptr<sirius::io::sirius_datasource> create_datasource(
    std::string_view path);

 private:
  /// \brief Run providers sequentially: start each, wait on its future, advance.
  void start_metadata_processing();

  /// \brief Attach a cached batch_provider to @p op if a pinned entry can serve
  ///        it. Returns true when a cache hit was assigned (the caller then skips
  ///        the disk-reading split_provider for this operator).
  bool try_assign_cached_entries(op::scan::sirius_gpu_scan_operator* op);

  /// Resolve the ioctx that should serve @p path (normalized internally, so callers
  /// — including the scan resolver — may pass a raw `file://` / `s3://` URI),
  /// building it once per backend on first use.  Routes by path through the registry
  /// so an `s3://` URI reaches the rest_ioctx even when the local default `_io_ctx`
  /// is uring/kvikio.  Returns nullptr when no backend supports the path.
  std::shared_ptr<sirius::io::sirius_ioctx> ioctx_for_path(std::string_view path);

  scan_manager_config _config;
  cucascade::memory::memory_reservation_manager& _reservation_manager;
  /// Hardware GPU/NUMA topology, shared with the prefetching cache.  Source of
  /// the GPU id set fed to the round-robin scan-balancing strategy.
  std::shared_ptr<const sirius::memory::topology_index> _topology_index;
  exec::static_thread_pool _thread_pool;
  std::unique_ptr<exec::scoped_dispatcher> _dispatcher;
  std::shared_ptr<sirius::io::sirius_ioctx> _io_ctx;
  /// Lazily-built per-backend ioctxs for path-routed datasources (e.g. an s3://
  /// rest_ioctx alongside the local uring/kvikio `_io_ctx`).  Built exactly once
  /// per type: `_routed_io_ctxs_build_mtx` serializes construction (reactor
  /// threads + cache allocation happen outside the map mutex), while
  /// `_routed_io_ctxs_mtx` guards only map lookup/insert; drained + torn down
  /// in the dtor.
  std::mutex _routed_io_ctxs_build_mtx;
  std::mutex _routed_io_ctxs_mtx;
  std::unordered_map<sirius::io::io_context_type, std::shared_ptr<sirius::io::sirius_ioctx>>
    _routed_io_ctxs;
  std::unordered_map<op::scan::sirius_gpu_scan_operator*, std::unique_ptr<split_provider>>
    _providers_by_op;
  std::vector<op::scan::sirius_gpu_scan_operator*> _scan_op_order;
  std::unordered_map<std::string, pinned_entry> _pinned_entries;

  /// Per-query sequencer for opportunistic fadvise calls.  Built fresh
  /// in @ref prepare_for_query, gets one @c pipeline_slot per non-cached
  /// parquet scan (allocated by @ref create_provider_for when it builds
  /// a parquet_split_provider).  The sequencer task is enqueued on the
  /// per-query @c _dispatcher, which injects its own stop_token; the
  /// dispatcher's @c request_stop() in @ref reset() therefore tears the
  /// sequencer down without an extra side-channel.
  std::unique_ptr<load_balancing_scan_batch_coalescer> _metadata_processor;
  io::io_context_registry _ioctx_registry;
};

}  // namespace sirius::scan_manager
