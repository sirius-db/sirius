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

#include "duckdb/planner/table_filter.hpp"
#include "exec/scoped_dispatcher.hpp"
#include "exec/thread_pool.hpp"
#include "io/datasource_factory.hpp"
#include "io/s3/s3_list_parser.hpp"
#include "io/sirius_datasource.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "pin_table.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/duckdb_mvcc_metadata.hpp"
#include "scan_manager/insert_delta_job.hpp"
#include "scan_manager/load_balancing_scan_batch_coalescer.hpp"
#include "scan_manager/mvcc_mask_job.hpp"
#include "scan_manager/pinned_chunk_stats.hpp"
#include "scan_manager/split_provider.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <compression/compressed_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <io/types.hpp>

namespace cucascade::memory {
class fixed_size_host_memory_resource;
}  // namespace cucascade::memory

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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

namespace sirius::telemetry {
struct batch_telemetry_info;
}  // namespace sirius::telemetry

namespace sirius::scan_manager {

/// Lightweight descriptor of a pinned table's cache identity + column layout,
/// stored on @ref pinned_entry in place of the read-side ingestible_table_info.
/// Captures only what serving needs — the table's identity (parquet file set OR
/// duckdb catalog/schema/table name), the cached columns (by primary/storage
/// index), and their names (aligned with @c column_ids) for the GPU gather — and
/// owns the match logic that @ref sirius_scan_manager::try_match_cached_entry consults.
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

  /// Duckdb-identity check shared by can_serve_with_columns and the plan-time
  /// MVCC guards — one matcher, so the probe and prepare can never drift.
  /// False for parquet entries (empty table_name).
  [[nodiscard]] bool matches_duckdb_table(std::string_view catalog,
                                          std::string_view schema,
                                          std::string_view table) const;

  /// Column-superset gather over @p requested_ids (requested order): for each
  /// requested column, its position within the cached @c column_ids. Empty
  /// when the cache cannot serve — a requested rowid/virtual/empty/
  /// field-identifier column (never cached), or a primary index absent from
  /// the cached set.
  [[nodiscard]] std::vector<std::size_t> column_projection_for(
    duckdb::vector<duckdb::ColumnIndex> const& requested_ids) const;

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
  /// coalesced batch.
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces;
  /// HOST-tier storage: one chunk per emitted batch in emission order, each
  /// holding all pinned columns. Every element is either a
  /// cucascade::host_data_representation (uncompressed) or a
  /// sirius::compressed_host_representation (Simpatico-compressed) — a single
  /// pinned table may mix the two, since compression is decided per chunk. The
  /// cached provider dispatches on the concrete type per chunk (slice() for the
  /// uncompressed form, select_columns() for the compressed form). Populated by
  /// @ref insert_pinned_entry_host.
  std::vector<std::shared_ptr<cucascade::idata_representation>> host_chunks;
  /// GPU-tier compression-enabled storage: one @ref device_pin_chunk per emitted
  /// batch, in emission order. Each chunk is either Simpatico-compressed (served
  /// as a compressed_device_representation, decompressed on demand) or
  /// uncompressed (served directly as a gpu_table_representation) — a single pin
  /// may interleave the two, since compression is decided per chunk. The cached
  /// provider dispatches on the populated form per chunk. Populated by
  /// @ref insert_pinned_entry_device. Takes priority over data_batches_by_column
  /// (the plain, non-compression GPU pin path) when non-empty.
  std::vector<sirius::device_pin_chunk> device_chunks;
  /// Tier the pinned data resides in. Drives which storage member above is used
  /// and which fetch path the cached provider takes.
  cucascade::memory::Tier tier{cucascade::memory::Tier::GPU};
  /// Representative memory space of a HOST-tier entry; the MVCC path expands
  /// it into a per-chunk vector. GPU-tier entries leave it null.
  cucascade::memory::memory_space* memory_space{nullptr};
  /// Total number of rows across all pinned chunks. Used by insert_pinned_entry
  /// to decide whether a re-insert merges into the existing entry (same row
  /// count → add unique columns) or replaces it (different row count).
  std::size_t num_rows{0};
  /// Zone-map sidecar: pin-time DuckDB types + per-chunk min/max statistics,
  /// positional with cache_info.column_ids. Absent (never prunes) when the
  /// capture was statless or degraded; see @ref pinned_zone_maps for the
  /// invariant and merge semantics.
  pinned_zone_maps zone_maps;
  /// MVCC snapshot metadata for duckdb-native pins, attached by
  /// @ref sirius_scan_manager::attach_mvcc_metadata right after insert. nullptr
  /// for parquet pins (immutable sources need no visibility reconciliation).
  std::unique_ptr<duckdb_mvcc_metadata> mvcc;
};

/// Validate that @p entry can serve @p selected_columns (positions into
/// @c entry.cache_info.column_ids) without truncation. GPU tier: every
/// selected column must be present in @c data_batches_by_column with exactly
/// n_chunks non-null chunks, and @c chunk_memory_spaces must cover every chunk
/// with non-null spaces. HOST tier: every host chunk must be non-null.
/// Zero-chunk entries are legitimate and pass. Throws std::runtime_error
/// naming the offending column/condition.
///
/// Serve-time defense against malformed entries: the cached serving loop reads
/// a nullptr batch as end-of-stream, so a column with fewer chunks (or a null
/// chunk) would silently end the scan early — fewer rows than requested, no
/// error. @ref sirius_scan_manager::try_match_cached_entry calls this before
/// recording the assignment and converts a throw into a disk-read fallback.
void validate_pinned_entry_for_serving(pinned_entry const& entry,
                                       std::span<std::size_t const> selected_columns);

/**
 * @brief Cache-serve-time survivor plan for one cached scan.
 */
struct cached_scan_plan {
  std::vector<std::size_t> survivor_chunk_indices;  ///< indices of chunks that survived pruning
  std::size_t pruned{0};
};

/// Build the cached-serving databatch_provider for @p entry over
/// @p selected_columns (positions into @c entry.cache_info.column_ids, in the
/// scan's materialized order). @p plan lists the zone-map survivor chunks the
/// provider serves (the identity plan when nothing was pruned). @p mvcc_masks
/// is the provider's own copy of the per-chunk MVCC keep-mask set, paired with
/// each chunk it yields (slot i masks chunk i; a default slot — or an empty
/// set, the parquet-pin case — serves the chunk unmasked). Declared here so
/// the chunk↔mask pairing is unit-testable; the provider type itself stays
/// internal to the scan manager.
///
/// The provider CO-OWNS @p entry for its whole life: it reads the entry's chunks on every
/// get_next_batch, and with concurrent queries another connection may unpin (or replace)
/// the entry mid-scan. Shared ownership is what keeps that from dangling — an unpin drops
/// the scan manager's map slot, and the data dies only once the last serving provider does.
std::unique_ptr<databatch_provider> make_provider_for_pinned_entry(
  std::shared_ptr<pinned_entry const> entry,
  std::span<std::size_t const> selected_columns,
  cached_scan_plan plan,
  const telemetry::batch_telemetry_info& telemetry_info,
  mvcc_chunk_mask_set mvcc_masks               = {},
  std::vector<insert_delta_split> delta_splits = {});

/**
 * @brief Build the survivor plan for serving @p entry to a scan into @p requiested_column_ids with
 * @p table_filters applied. A chunk is pruned when any usable filter proves it empty against the
 * pinned entry's zone-map statistics.
 */
[[nodiscard]] cached_scan_plan build_cached_scan_plan(
  pinned_entry const& entry,
  duckdb::TableFilterSet const* table_filters,
  duckdb::vector<duckdb::ColumnIndex> const* requested_column_ids);

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

/// Number of concurrent queries the scan manager sizes its thread pool for.
///
/// Each query's coalescer runs exactly ONE sequencer task that BLOCKS in
/// queue.wait_dequeue, and is unblocked only by that query's own split_provider tasks
/// running on the same pool. So every concurrent query needs a parked thread on top of the
/// working budget. With Q concurrent queries and a pool of size P, Q >= P is a hard
/// deadlock: every thread parked in a sequencer, none left to feed them.
///
/// Left at 1 so the pool stays exactly the size it was before per-query state landed
/// (num_threads + 1, the old single-sequencer allowance) — this makes the concurrency
/// refactor behaviorally neutral for existing single-query runs.
///
/// TODO: promote to a real option on @ref scan_manager_config (scan_manager/config.hpp),
/// parsed alongside thread_pool.num_threads in sirius_config.cpp's from_yaml, and RAISE IT
/// before enabling more than a couple of concurrent queries. Nothing else should read this
/// constant once it moves.
/// Deprecated compile-time bound, kept only as the default for
/// scan_manager_config::max_concurrent_queries. Read the config value, not this.
inline constexpr int k_default_max_concurrent_queries = 1;

/**
 * @brief Manages scan-side preparation for a query.
 *
 * The scan manager owns a configurable-size thread pool shared by every query, and holds
 * per-query scan state keyed by query id (see @c query_scan_manager_state) so concurrent
 * queries prepare, run and tear down independently.
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
  /// Registers a NEW per-query state entry keyed by @p query 's id; it does NOT touch any
  /// other query's entry. Call reset(query_id) when the query finishes. A query with no GPU
  /// scan operators registers nothing, so its reset is a harmless no-op.
  ///
  /// @param query                           The query whose scan operators must be prepared.
  /// @param enable_pinned_zone_map_pruning  Per-query snapshot of the serve-side pruning flag (the
  ///                                        manager's _config is a construction-time copy, so SET
  ///                                        changes must be forwarded per query by the caller).
  ///                                        Consulted by try_assign_cached_entries when building
  ///                                        the survivor plan.
  void prepare_for_query(const sirius::planner::query& query, bool enable_pinned_zone_map_pruning);

  /// \brief Drop everything held for @p query_id: stop and join its scan work, then destroy
  ///        its providers and coalescer. Other queries are untouched. No-op for an unknown id.
  ///
  /// Blocking — waits out this query's in-flight reads. The wait happens OUTSIDE the state
  /// mutex, so another connection's prepare_for_query is never parked behind it.
  void reset(sirius::query_id_t query_id);

  /// \brief Drop every query's state. Teardown only (stop(), the destructor, and the
  ///        failed-query backstop).
  void reset_all();

  /// \brief Number of queries currently registered. Observability and tests.
  [[nodiscard]] std::size_t num_active_queries() const noexcept;

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
  ///   - Zone-map types/statistics mirror the data decisions: kept when a merge
  ///     drops duplicate columns, appended for new columns that received chunks,
  ///     and degraded to statless when the union column set cannot stay
  ///     positionally aligned (data serving is unaffected).
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
  /// \param column_types          Pin-time DuckDB type of each cached column, positional
  ///                              with @p cache_info's column_ids; empty pins statless.
  /// \param chunk_stats           Per-chunk zone-map stats (chunk_stats[c][i] = column i
  ///                              of chunk c, as compute_pinned_chunk_stats emits).
  void insert_pinned_entry(
    const std::string& name,
    cache_entry_info cache_info,
    std::vector<std::unique_ptr<cudf::table>> data_tables,
    std::vector<cucascade::memory::memory_space*> chunk_memory_spaces,
    duckdb::vector<duckdb::LogicalType> column_types                                 = {},
    std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats = {});

  /// \brief Pin the host-tier entry for a table.
  ///
  /// Each entry in @p host_chunks describes one batch's worth of pinned data
  /// (covering all pinned columns), in emission order. A chunk is either a
  /// cucascade::host_data_representation (uncompressed) or a
  /// sirius::compressed_host_representation (Simpatico-compressed); a single pin
  /// may mix the two, since compression is decided per chunk. The cached provider
  /// built from this entry projects each chunk by column index at scan time,
  /// dispatching on the concrete type. This path always REPLACES any existing
  /// entry for @p name — there is no per-column merge analog to the GPU path
  /// because the chunk-vs-column dimensions are flipped (each chunk already holds
  /// every column).
  ///
  /// \param name          Table name key.
  /// \param cache_info    Cache identity plus the cached columns and their
  ///                      @c column_ids-aligned names (the i-th column in each
  ///                      chunk corresponds to @c cache_info.column_names()[i]);
  ///                      drives cache-hit matching.
  /// \param host_chunks   One representation per emitted batch (uncompressed or
  ///                      compressed).
  /// \param memory_space  Representative host memory space the chunks reside in
  ///                      (metadata only; each chunk carries its own per-GPU
  ///                      NUMA-local memory_space).
  /// \param column_types  Pin-time DuckDB type of each cached column, positional
  ///                      with @p cache_info's column_ids; empty pins statless.
  /// \param chunk_stats   Per-chunk zone-map stats (chunk_stats[c][i] = column i of chunk c, as
  ///                      compute_pinned_chunk_stats emits).
  void insert_pinned_entry_host(
    const std::string& name,
    cache_entry_info cache_info,
    std::vector<std::shared_ptr<cucascade::idata_representation>> host_chunks,
    cucascade::memory::memory_space& memory_space,
    duckdb::vector<duckdb::LogicalType> column_types                                 = {},
    std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats = {});

  /// \brief Pin the entry for a table on the GPU tier from a compression-enabled pin.
  ///
  /// Each entry in @p chunks is one emitted batch, in emission order, holding all
  /// pinned columns — either Simpatico-compressed (decompressed on demand at scan
  /// time) or uncompressed (served directly); a single pin may interleave the two.
  /// The cached provider prefers @c device_chunks over @c data_batches_by_column
  /// when non-empty. Always REPLACES any existing entry for @p name — there is no
  /// per-column merge analog (each chunk already holds every column).
  ///
  /// \param name          Table name key.
  /// \param cache_info    Cache identity plus the cached columns and their
  ///                      @c column_ids-aligned names.
  /// \param chunks        One @ref device_pin_chunk per batch (compressed or not).
  /// \param memory_space  Representative GPU memory space (metadata only).
  void insert_pinned_entry_device(const std::string& name,
                                  cache_entry_info cache_info,
                                  std::vector<sirius::device_pin_chunk> chunks,
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

  /// Visit every pinned entry under the pin-table lock. The visitor runs WHILE the lock is
  /// held, so it must be quick and must not call back into the scan manager. The reference
  /// it receives is only guaranteed for the duration of the call — a visitor that stashes
  /// the address for later use is racing any concurrent unpin.
  void visit_pinned_entries(
    const std::function<bool(std::string_view, const pinned_entry&)>& visitor) const;

  /// The pinned entry whose duckdb identity matches catalog.schema.table, or nullptr.
  /// OWNING: the returned shared_ptr keeps the entry alive for as long as the caller holds
  /// it, so a concurrent unpin on another connection cannot pull it out from under the
  /// plan-time MVCC guards that read it. First match wins if one table was pinned under two
  /// names.
  [[nodiscard]] std::shared_ptr<const pinned_entry> find_pinned_entry_for_duckdb_table(
    std::string_view catalog_name, std::string_view schema_name, std::string_view table_name) const;

  parquet_bind_result describe_parquet(std::string const& uri);

  /// \brief Process-wide ioctx used to mint @c sirius_datasource instances.
  ///        Holds a @c uring_ioctx, or a @c kvikio_context when the manager
  ///        was configured with @c use_sirius_datasource=false.
  [[nodiscard]] sirius::io::sirius_ioctx* io_ctx() const noexcept { return _io_ctx.get(); }

  [[nodiscard]] std::shared_ptr<sirius::io::sirius_datasource> create_datasource(
    std::string_view path, sirius::io::open_hint hint = sirius::io::open_hint::generic);

  /// \brief Stream ListObjectsV2 pages for @p s3_prefix_uri ("s3://bucket/prefix")
  ///        to @p sink, one call per page; @p sink returns false to stop early.
  ///        Routes via @ref ioctx_for_path to the object-store backend (throws a
  ///        clear error when the path does not resolve to one). page_size /
  ///        early-stop semantics are the backend's (@c rest_ioctx::list_objects_paged).
  ///        @p max_scanned unset → the backend's configured cap
  ///        (@c rest.list_max_scanned); a value overrides it.
  void list_objects_paged(
    std::string const& s3_prefix_uri,
    std::size_t page_size,
    std::function<bool(sirius::io::s3::list_objects_v2_page const&)> const& sink,
    std::optional<std::size_t> max_scanned = std::nullopt);

  /// \brief The configured glob-match cap (@c rest.list_max_matches) for the
  ///        backend @p s3_uri routes to — the glob layer bounds its match set
  ///        with it. Throws a clear error for a non-object-store path.
  [[nodiscard]] std::size_t s3_list_max_matches(std::string const& s3_uri);

 private:
  /**
   * @brief Everything the scan manager holds on behalf of ONE query.
   *
   * Handed out as a `shared_ptr` keyed by query id: a caller resolves it under the state
   * mutex and uses it outside, so an erase racing a reader cannot pull the state out from
   * under them. The manager's shared members (ioctxs, registry, pinned entries, thread pool)
   * stay outside — they are process-wide and outlive every query. What lives here is exactly
   * what the old global `reset()` used to wipe, and wiping it globally is what made
   * finishing query A tear down query B's scan work.
   *
   * Operator ids restart at 0 for every query, so `metadata_processor`'s slot map (keyed by
   * `scan_op->get_operator_id()`) is unique only *within* an entry; one shared coalescer
   * would let two queries' scans collide on the same slot.
   *
   * Member order is teardown order: `dispatcher` is declared LAST so it is destroyed FIRST,
   * while the coalescer and providers its tasks captured are still alive. The destructor's
   * `drain()` makes that safe even if a member is later added below it.
   */
  struct query_scan_manager_state {
    ~query_scan_manager_state() { drain(); }

    query_scan_manager_state()                                           = default;
    query_scan_manager_state(const query_scan_manager_state&)            = delete;
    query_scan_manager_state& operator=(const query_scan_manager_state&) = delete;
    query_scan_manager_state(query_scan_manager_state&&)                 = delete;
    query_scan_manager_state& operator=(query_scan_manager_state&&)      = delete;

    //! Stop and join every task this query put on the shared pool. Idempotent, and safe to
    //! call outside the state mutex — it can block for as long as an in-flight read.
    void drain() noexcept;

    //! One scan operator and the disk-reading provider built for it (null when the operator
    //! matched a pinned entry and is served from the cache instead). A vector rather than a
    //! map plus a parallel order vector: the only traversal is registration order, and one
    //! container makes the "already registered" guard cover cache-matched operators too.
    struct scan_entry {
      op::scan::sirius_gpu_scan_operator* op{nullptr};
      std::unique_ptr<split_provider> provider;  ///< null for a cache-served operator
    };
    std::vector<scan_entry> scans;

    //! Per-query snapshot of the serve-side pruning flag; the manager's _config is a
    //! construction-time copy, so SET changes arrive per query via prepare_for_query.
    bool pruning_enabled{true};

    //! One mask computation per distinct pinned entry matched by THIS query (recorded by
    //! try_match_cached_entry, deduped by entry name); run block-in-prepare, then copied
    //! into each provider and cleared. Per-query by nature: two queries over the same entry
    //! see different MVCC snapshots and each needs its own masks.
    std::vector<mvcc_mask_job_request> pending_mvcc_mask_jobs;

    //! One insert-delta job per distinct pinned entry matched by this query, with the same
    //! dedup and lifecycle as the mask jobs above.
    std::vector<insert_delta_job_request> pending_insert_delta_jobs;

    //! This query's sequencer for opportunistic fadvise calls; gets one pipeline slot per
    //! scan. Its slot map is keyed by operator id, which restarts at 0 per query.
    std::unique_ptr<load_balancing_scan_batch_coalescer> metadata_processor;

    //! This query's scope for every scan task it puts on the SHARED pool. request_stop()
    //! here stops only this query's work; the pool and every other query keep running.
    std::unique_ptr<exec::scoped_dispatcher> dispatcher;
  };

  /// \brief Run @p state 's providers sequentially: start each, wait on its future, advance.
  void start_metadata_processing(query_scan_manager_state& state);

  //! Resolve a query's state, or nullptr when it has already been reset.
  [[nodiscard]] std::shared_ptr<query_scan_manager_state> get_query_state(
    sirius::query_id_t query_id) const;

  /// One matched (scan op ← pinned entry) pairing from the cache-match pass.
  /// Provider construction is deferred to after run_mvcc_mask_jobs so each
  /// provider takes its own copy of the entry's completed mask set. The
  /// assignment SHARES OWNERSHIP of the entry across that gap: the mask and
  /// insert-delta jobs block for as long as their IO takes, and with concurrent
  /// queries another connection may unpin in that window.
  struct cached_assignment {
    op::scan::sirius_gpu_scan_operator* op{nullptr};
    std::shared_ptr<pinned_entry const> entry;
    std::vector<std::size_t> columns;  ///< selected columns, materialized order
    std::string entry_name;            ///< handoff key into _pending_mvcc_mask_jobs
    cached_scan_plan plan;             ///< zone-map survivor plan, moved into the provider
  };

  /// \brief Match @p op against the pinned entries. On a hit, validates the
  ///        entry for serving, queues the entry's MVCC mask job unless one is
  ///        already pending (a self-join queues ONE job per entry), and
  ///        returns the assignment for the post-mask-run provider handoff.
  ///        Returns nullopt on a miss (the caller then builds the
  ///        disk-reading split_provider for this operator).
  ///
  ///        Mask and insert-delta jobs are queued into @p state, and @p state 's
  ///        pruning_enabled decides whether the survivor plan applies zone-map filters. The
  ///        dedup is therefore per-query, which is the point: two concurrent queries over
  ///        the same pinned entry each queue their own job and take their own mask copies.
  [[nodiscard]] std::optional<cached_assignment> try_match_cached_entry(
    op::scan::sirius_gpu_scan_operator* op, query_scan_manager_state& state);

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
  /// The pin table. Shared across every query and outliving all of them, so entries are
  /// held by shared_ptr rather than by value: a matched scan takes a reference for its
  /// whole duration, and an unpin from another connection drops only the map slot, leaving
  /// the data alive until the last serving provider releases it.
  std::unordered_map<std::string, std::shared_ptr<pinned_entry>> _pinned_entries;
  /// Guards the STRUCTURE of _pinned_entries (lookup/insert/erase) only, never the serving
  /// of an entry — shared ownership covers that. Held only for map manipulation and the
  /// snapshot in try_match_cached_entry, never across match/validate/plan work, so
  /// concurrent prepares do not serialize behind each other.
  mutable std::mutex _pinned_entries_mutex;

  //! One entry per in-flight query. Guarded by _query_states_mutex; the shared_ptr is
  //! resolved under the lock and used outside it, so an erase racing a reader cannot pull
  //! the state out from under them.
  std::map<sirius::query_id_t, std::shared_ptr<query_scan_manager_state>> _query_states;
  mutable std::mutex _query_states_mutex;

  io::io_context_registry _ioctx_registry;
};

}  // namespace sirius::scan_manager
