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
#include "scan_manager/memory_prefetcher.hpp"
#include "scan_manager/mvcc_mask_job.hpp"
#include "scan_manager/pinned_chunk_stats.hpp"
#include "scan_manager/split_provider.hpp"

namespace sirius::op {
class sirius_dynamic_filter_set;  // membership pushdown channel (op/sirius_dynamic_filter.hpp)
}

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <compression/compressed_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/storage_lock.hpp>
#include <io/types.hpp>

namespace cucascade::memory {
class fixed_size_host_memory_resource;
}  // namespace cucascade::memory

#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
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

  /// Parquet-identity check shared by can_serve_with_columns and the plan-time
  /// residency gate — one matcher, so the probe and prepare can never drift.
  /// Same file set irrespective of order (both sides sorted, byte-exact compare).
  /// False for duckdb entries (empty resolved_file_paths) and for an empty @p files.
  [[nodiscard]] bool matches_parquet_files(std::span<std::string const> files) const;

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
  /// Chunk-major stored-column metadata, positional with the cached data:
  /// column_storage[c][i] records the carrier, pin-time native mapping, and
  /// narrowing marker for cached column i in chunk c. Recorded by the pin driver
  /// at the moment of storage; insertion requires a matrix covering every chunk
  /// and column and cross-checks recorded carriers against
  /// uncompressed storage. An empty matrix reads as all-native — a legitimate
  /// state for a zero-chunk or hand-built entry, which is why the serving
  /// validator still accepts it. The plan-time narrowing folds and the
  /// serve-time conversion sizing read this matrix and never introspect storage.
  sirius::pinned_column_storage_matrix column_storage;
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

/// Validate the shape of @p matrix alone: it must hold @p expected_chunks rows of
/// @p expected_columns cells each. @p allow_empty admits the empty matrix, which reads as
/// all-native — @ref validate_pinned_entry_for_serving passes true because a zero-chunk or
/// hand-built entry is legitimately empty, while insertion passes false because the pin driver
/// always records coverage. @p context prefixes the thrown message so the caller is named.
/// Throws std::invalid_argument.
void validate_column_storage_shape(sirius::pinned_column_storage_matrix const& matrix,
                                   std::size_t expected_chunks,
                                   std::size_t expected_columns,
                                   std::string_view context,
                                   bool allow_empty);

/// Report a recorded carrier that contradicts the type storage actually holds. The single throw
/// site of @ref validate_recorded_column_storage's cross-check; declared here only because that
/// validator is a template.
[[noreturn]] void throw_recorded_carrier_mismatch(std::string_view context,
                                                  std::size_t chunk_idx,
                                                  std::size_t column_idx,
                                                  cudf::data_type recorded,
                                                  cudf::data_type stored);

/// Validate @p matrix against the storage about to be cached: it must cover every chunk and every
/// cached column, and every recorded carrier must equal the stored type wherever storage can
/// report one. @p stored_type answers (chunk, column) with the stored column's cuDF type, or
/// `std::nullopt` when the form is opaque — a Simpatico-compressed chunk, whose recorded carrier is
/// correct by construction (the pin driver recorded exactly what compress_with_plan received) and
/// whose end-to-end defense is serve-time normalization. Also `std::nullopt` for a chunk or column
/// the storage does not hold, which the cross-check simply skips. Throws std::invalid_argument.
template <std::invocable<std::size_t, std::size_t> StoredType>
  requires std::same_as<std::invoke_result_t<StoredType, std::size_t, std::size_t>,
                        std::optional<cudf::data_type>>
void validate_recorded_column_storage(sirius::pinned_column_storage_matrix const& matrix,
                                      std::size_t expected_chunks,
                                      std::size_t expected_columns,
                                      std::string_view context,
                                      StoredType const& stored_type)
{
  validate_column_storage_shape(
    matrix, expected_chunks, expected_columns, context, /*allow_empty=*/false);
  for (std::size_t chunk_idx = 0; chunk_idx < matrix.size(); ++chunk_idx) {
    for (std::size_t column_idx = 0; column_idx < matrix[chunk_idx].size(); ++column_idx) {
      auto const stored = stored_type(chunk_idx, column_idx);
      if (!stored) { continue; }
      if (matrix[chunk_idx][column_idx].carrier != *stored) {
        throw_recorded_carrier_mismatch(
          context, chunk_idx, column_idx, matrix[chunk_idx][column_idx].carrier, *stored);
      }
    }
  }
}

/// True when @p entry's storage metadata shows the cached column at @p entry_position (a position
/// into cache_info.column_ids) narrowed in every chunk. False for an empty matrix, a zero-chunk
/// entry, or an out-of-range position. `pinned_column_narrow_carrier` uses this in the plan-time
/// residency gate: a passing column needs at most a same-family widening per chunk; a failing
/// column stays native instead of requiring per-query range verification and downcasts.
[[nodiscard]] bool pinned_column_narrowed_in_all_chunks(pinned_entry const& entry,
                                                        std::size_t entry_position);

/**
 * @brief Derives the widest validated narrow carrier recorded for a cached column.
 *
 * Reads only @p entry's column-storage metadata. Returns no target unless @p entry_position is
 * covered by the cached-column identity and every chunk marks the column narrowed, records
 * @p native_type as its pin-time native type, and records a strict same-family narrowing of that
 * type. Chunks narrower than the returned target widen when served.
 *
 * Call only while @p entry is protected from concurrent pin or unpin; this function does not
 * retain it.
 *
 * @param entry Pinned entry to inspect.
 * @param entry_position Position in `entry.cache_info.column_ids`.
 * @param native_type Current scan's native cuDF type.
 * @return Widest valid recorded carrier, or `std::nullopt`.
 */
[[nodiscard]] std::optional<cudf::data_type> pinned_column_narrow_carrier(
  pinned_entry const& entry, std::size_t entry_position, cudf::data_type native_type);

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
/// set, the parquet-pin case — serves the chunk unmasked). @p normalization_targets
/// is the scan's carrier targets in output order
/// (@c sirius_gpu_scan_operator::normalization_targets); served slot k is output
/// column k, so a slot past the end is a pure-filter column that reaches no
/// target. Together with @p has_physical_overrides it decides which columns each
/// served chunk reports as converting, and how many destination bytes that
/// conversion allocates. Declared here so the chunk↔mask pairing and the
/// conversion sizing are unit-testable; the provider type itself stays internal
/// to the scan manager.
/// @p pushdown_req is the scan's filter as a decompressor can use it, parallel
/// to @p selected_columns: a GPU-tier compressed chunk may answer an equality/IN
/// filter off its dictionary (the column then arrives as the boolean answer) and
/// may drop rows against the ranges while decoding, handing back an
/// already-filtered batch. See @c sirius::pushdown_request; an empty request
/// (the default) leaves every chunk decoding unfiltered.
/// @p dynamic_filters is the operator's dynamic-filter channel (join builds
/// publish into it mid-scan); the provider snapshots it PER BATCH onto the
/// attached scan, so later batches legitimately see more filters. Null (the
/// default) disables it.
std::unique_ptr<databatch_provider> make_provider_for_pinned_entry(
  pinned_entry const& entry,
  std::span<std::size_t const> selected_columns,
  cached_scan_plan plan,
  const telemetry::batch_telemetry_info& telemetry_info,
  mvcc_chunk_mask_set mvcc_masks                                         = {},
  std::vector<insert_delta_split> delta_splits                           = {},
  std::vector<cudf::data_type> normalization_targets                     = {},
  bool has_physical_overrides                                            = false,
  sirius::pushdown_request pushdown_req                                  = {},
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> dynamic_filters = nullptr);

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
  /// @param query                           The query whose scan operators must be prepared.
  /// @param enable_pinned_zone_map_pruning  Per-query snapshot of the serve-side pruning flag (the
  ///                                        manager's _config is a construction-time copy, so SET
  ///                                        changes must be forwarded per query by the caller).
  ///                                        Consulted by try_assign_cached_entries when building
  ///                                        the survivor plan.
  /// @param allocated_gpu_ids GPU IDs allocated to this query (from
  ///        SiriusContext::compute_allocated_gpu_ids). Scan splits are
  ///        distributed round-robin across this subset instead of all GPUs.
  void prepare_for_query(const sirius::planner::query& query,
                         bool enable_pinned_zone_map_pruning,
                         const std::vector<int>& allocated_gpu_ids);

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
  /// \param column_storage        Chunk-major stored-column metadata as the pin driver
  ///                              recorded it; must cover every chunk and cached column. A
  ///                              recorded carrier that contradicts a stored column type throws.
  void insert_pinned_entry(
    const std::string& name,
    cache_entry_info cache_info,
    std::vector<std::unique_ptr<cudf::table>> data_tables,
    std::vector<cucascade::memory::memory_space*> chunk_memory_spaces,
    duckdb::vector<duckdb::LogicalType> column_types,
    std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats,
    sirius::pinned_column_storage_matrix column_storage);

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
  /// \param column_storage Chunk-major stored-column metadata as the pin driver recorded it;
  ///                      must cover every chunk and cached column. A recorded carrier that
  ///                      contradicts an uncompressed chunk's stored type throws; a compressed
  ///                      chunk's types are unreadable here, so its cells are trusted.
  void insert_pinned_entry_host(
    const std::string& name,
    cache_entry_info cache_info,
    std::vector<std::shared_ptr<cucascade::idata_representation>> host_chunks,
    cucascade::memory::memory_space& memory_space,
    duckdb::vector<duckdb::LogicalType> column_types,
    std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats,
    sirius::pinned_column_storage_matrix column_storage);

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
  /// \param column_storage Chunk-major stored-column metadata as the pin driver recorded it;
  ///                      must cover every chunk and cached column. A recorded carrier that
  ///                      contradicts an uncompressed chunk's stored type throws; a compressed
  ///                      chunk's types are unreadable here, so its cells are trusted.
  void insert_pinned_entry_device(const std::string& name,
                                  cache_entry_info cache_info,
                                  std::vector<sirius::device_pin_chunk> chunks,
                                  cucascade::memory::memory_space& memory_space,
                                  sirius::pinned_column_storage_matrix column_storage);

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

  /// The pinned entry whose duckdb identity matches catalog.schema.table, or
  /// nullptr. Non-owning; obtain and read it inside one slot-scoped window, and
  /// never hold it across a pin or unpin. First match wins if one table was
  /// pinned under two names. Read by the plan-time MVCC guards.
  [[nodiscard]] pinned_entry const* find_pinned_entry_for_duckdb_table(
    std::string_view catalog_name, std::string_view schema_name, std::string_view table_name) const;

  /// The pinned entry whose parquet identity matches @p resolved_file_paths
  /// (cache_entry_info::matches_parquet_files), or nullptr. Non-owning; obtain
  /// and read it inside one slot-scoped window, and never hold it across a pin
  /// or unpin. First match wins if one file set was pinned under two names.
  /// Read by the plan-time compressed-materialization residency gate.
  [[nodiscard]] pinned_entry const* find_pinned_entry_for_parquet_files(
    std::span<std::string const> resolved_file_paths) const;

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
  /// \brief Run providers sequentially: start each, wait on its future, advance.
  void start_metadata_processing();

  /// One matched (scan op ← pinned entry) pairing from the cache-match pass.
  /// Provider construction is deferred to after run_mvcc_mask_jobs so each
  /// provider takes its own copy of the entry's completed mask set; the entry
  /// pointer stays valid for the whole prepare (pin/unpin is
  /// query-lifecycle-serialized).
  struct cached_assignment {
    op::scan::sirius_gpu_scan_operator* op{nullptr};
    pinned_entry const* entry{nullptr};
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
  [[nodiscard]] std::optional<cached_assignment> try_match_cached_entry(
    op::scan::sirius_gpu_scan_operator* op);

  /// Build and start the per-query host->GPU memory prefetcher when enabled
  /// via the sirius.executor.scan_manager.memory_prefetcher config block (see
  /// memory_prefetcher.hpp). No-op otherwise.
  void maybe_start_memory_prefetcher();

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
  bool _pruning_enabled{true};

  /// One mask computation per distinct pinned entry matched this query
  /// (recorded by try_match_cached_entry, deduped by entry name); executed
  /// block-in-prepare by run_mvcc_mask_jobs, after which the provider handoff
  /// copies each completed set out and the vector is cleared (also cleared in
  /// reset() for the prepare-threw case).
  std::vector<mvcc_mask_job_request> _pending_mvcc_mask_jobs;

  /// One insert-delta job per distinct pinned entry matched this query, with
  /// the same dedup and lifecycle as the mask jobs above. Later operators
  /// union their columns into the pending request. The job no-ops when the
  /// table has no rows beyond the pinned prefix.
  std::vector<insert_delta_job_request> _pending_insert_delta_jobs;

  /// Prevents DuckDB checkpoints from replacing row groups between pinned
  /// query validation and completion.
  std::vector<duckdb::unique_ptr<duckdb::StorageLockKey>> _checkpoint_locks;

  /// Per-query sequencer for opportunistic fadvise calls.  Built fresh
  /// in @ref prepare_for_query, gets one @c pipeline_slot per scan,
  /// registered before the pinned-cache match.  The
  /// sequencer task is enqueued on the
  /// per-query @c _dispatcher, which injects its own stop_token; the
  /// dispatcher's @c request_stop() in @ref reset() therefore tears the
  /// sequencer down without an extra side-channel.
  std::unique_ptr<load_balancing_scan_batch_coalescer> _metadata_processor;
  /// Per-query background host->GPU upgrader for queued pinned-cache splits
  /// (built in start_metadata_processing when the memory_prefetcher config
  /// block enables it, torn down in reset()).
  std::unique_ptr<memory_prefetcher> _prefetcher;
  io::io_context_registry _ioctx_registry;
};

}  // namespace sirius::scan_manager
