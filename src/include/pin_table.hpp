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

#include <cudf/types.hpp>

#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {

struct PinTableArgs {
  std::string path;
  std::string tier;
  std::string name;
  std::optional<std::vector<std::string>> cols;
  /// Resolved at bind time: "parquet" or "duckdb". Chosen from an explicit
  /// `format` named parameter, else inferred from the path extension
  /// (.parquet -> parquet, .db/.duckdb -> duckdb).
  std::string format;
  /// duckdb-native only: schema that contains the table named by `name`. Defaults
  /// to "main"; the catalog search path (current/USE'd database) still applies.
  std::string schema = "main";
};

void pin_table_to(const PinTableArgs& args);

void unpin_table_to(const std::string& name);

// Test-only helpers used by unit tests to verify pin_table forwards the
// correct parameters to pin_table_to. Not part of the stable public API.
namespace pin_table_testing {
const std::vector<PinTableArgs>& recorded_calls();
void clear_recorded_calls();

const std::vector<std::string>& recorded_unpin_calls();
void clear_recorded_unpin_calls();
}  // namespace pin_table_testing

}  // namespace duckdb

namespace cudf {
class table;
class column;
}  // namespace cudf

namespace cucascade {
class host_data_representation;
class idata_representation;
}  // namespace cucascade

namespace sirius {
class compressed_host_representation;
class compressed_device_representation;
}  // namespace sirius

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace sirius {

namespace op::scan {
class gpu_ingestible;
class scan_info;
}  // namespace op::scan
namespace io {
class ioctx;
}  // namespace io

/**
 * @brief Physical-type metadata for one cached column in one pinned chunk.
 *
 * For uncompressed storage, `carrier` is the stored cuDF type. For Simpatico-compressed storage,
 * it is the type passed to `compress_with_plan` and therefore reproduced by decompression.
 * `native` records the declared logical type's native cuDF mapping when available, otherwise the
 * decoded type. Planning uses it to validate narrow sidecars, and DuckDB cache serving uses it to
 * reject projected type drift.
 */
struct pinned_column_storage_meta {
  cudf::data_type carrier{cudf::type_id::EMPTY};  ///< actual stored / compress-input cuDF type
  bool narrowed{false};  ///< carrier is narrower than the pin-time native mapping
  cudf::data_type native{cudf::type_id::EMPTY};  ///< declared-native mapping or decoded fallback
};

/// Chunk-major stored-column metadata: element [c][i] describes cached column i of chunk c.
using pinned_column_storage_matrix = std::vector<std::vector<pinned_column_storage_meta>>;

/// GPU-resident tables produced by driving a @c gpu_ingestible to completion, with
/// each table's GPU placement recorded in @c chunk_memory_spaces (parallel to
/// @c tables). Consumed by the pin_table path: fed directly into
/// @c sirius_scan_manager::insert_pinned_entry, or, after host conversion of each
/// table, into @c insert_pinned_entry_host.
struct materialized_pin {
  std::vector<std::unique_ptr<cudf::table>> tables;
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces;
  /// Chunk-major stored-column metadata, parallel to @c tables and their columns.
  pinned_column_storage_matrix column_storage;
  /// Row count of each materialized chunk (parallel to @c tables). For duckdb-native
  /// pins these become @c duckdb_mvcc_metadata::base_row_count_per_chunk — the
  /// positional chunk→rowid-range map query-time MVCC merge relies on.
  std::vector<std::size_t> base_row_count_per_chunk;
  /// Per-chunk zone-map capture: chunk_stats[c][i] = stats of batch column i of
  /// chunk c (null = none). Parallel to @c tables when capture ran; empty when
  /// capture was skipped (no pinned column types). Fed together with the
  /// pin-time column types into @c sirius_scan_manager::insert_pinned_entry.
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats;
};

/// Pin-time validation of the coalescer invariant the MVCC delta merge relies on
/// (#819): every duckdb-native batch must be a contiguous run of WHOLE row groups
/// starting at @p rows_before_chunk, and its row-group metadata must cover exactly
/// @p chunk_rows (the decoded cudf row count) — so the pinned chunks partition the
/// decoded rowid prefix [0, N_cache) at row-group boundaries and per-chunk
/// visibility masks can be assembled at row-group granularity. Throws
/// std::runtime_error on violation; batches that are not duckdb_native_scan_info
/// (parquet) are skipped. Called by the pin materialization driver on every
/// emitted batch; exposed here so its failure paths are unit-testable.
void validate_duckdb_pin_chunk(const op::scan::scan_info& batch,
                               std::size_t chunk_rows,
                               std::size_t rows_before_chunk);

/**
 * @brief Selects the pin-time native cuDF type recorded for a cached column.
 *
 * The declared logical mapping is preferred because a decoder may materialize another physical
 * type for the same logical type, such as a wider Parquet decimal. The decoded type is used when
 * @p declared_type is null or has no cuDF mapping.
 *
 * @param decoded_type Physical type produced by the decoder.
 * @param declared_type Optional, non-owning pointer to the declared DuckDB type.
 * @return Declared type's native cuDF mapping when available; otherwise @p decoded_type.
 */
[[nodiscard]] cudf::data_type pin_native_type(cudf::data_type decoded_type,
                                              duckdb::LogicalType const* declared_type);

/// Per-pin materialization behavior, sampled from config at pin time.
struct pin_materialization_options {
  bool capture_chunk_stats               = true;   ///< capture zone-map statistics per chunk
  bool enable_compressed_materialization = false;  ///< narrow eligible numeric and DATE carriers
};

/// Drive @p ingestible 's metadata walk + batch coalescer to completion on @p io_ctx,
/// materializing every emitted batch into a GPU-resident cudf::table and round-robining
/// placement across @p gpu_spaces. Single-threaded with deterministic placement, so
/// re-pinning the same source yields identical @c chunk_memory_spaces (required by
/// @c sirius_scan_manager::insert_pinned_entry 's merge path).
///
/// \param ingestible          Source ingestible (parquet or duckdb-native). Consumed by repeated
///                            next_split_provider / materialize calls.
/// \param gpu_spaces          Non-empty set of GPU memory spaces to round-robin across.
/// \param io_ctx              IO context the metadata reads run on (owned by the scan manager).
/// \param pinned_column_types Pin-time DuckDB type of each batch column, in batch-column
///                            (column_ids) order. When carrier narrowing is enabled, this vector
///                            must contain one type per materialized column. Otherwise, an empty
///                            vector skips zone-map capture (statless pin).
/// \param options             Per-pin materialization behavior (zone-map capture, carrier
///                            narrowing).
materialized_pin materialize_all_batches(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  io::ioctx& io_ctx,
  duckdb::vector<duckdb::LogicalType> const& pinned_column_types,
  pin_materialization_options options = {});

/// Result of driving a host-tier pin with optional Simpatico compression.
/// Compression is decided per chunk (a batch below the size threshold, or one
/// that fails to compress usefully, is pinned uncompressed), so a single pinned
/// table may mix the two forms. @c chunks holds one representation per emitted
/// batch in emission order — each element is either a @c cucascade::
/// host_data_representation (uncompressed) or a @c sirius::
/// compressed_host_representation; both derive from @c cucascade::idata_representation.
struct host_pin_result {
  std::vector<std::shared_ptr<cucascade::idata_representation>> chunks;
  /// Chunk-major stored-column metadata, recorded from each GPU table right before it is
  /// compressed or converted to host storage. Parallel to @c chunks and their columns.
  pinned_column_storage_matrix column_storage;
  /// Row count of each materialized batch, in emission order (covers compressed
  /// and uncompressed chunks alike); becomes duckdb_mvcc_metadata::
  /// base_row_count_per_chunk for duckdb-format pins.
  std::vector<std::size_t> base_row_count_per_chunk;
  /// Per-chunk zone-map capture (taken on the GPU table before host conversion /
  /// compression); chunk_stats[c][i] = stats of batch column i of chunk c (null =
  /// none). Parallel to @c chunks when capture ran; empty when capture was skipped
  /// (no pinned column types). Fed with the pin-time column types into
  /// @c sirius_scan_manager::insert_pinned_entry_host to drive zone-map pruning.
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats;
};

/// Optional compression settings for @ref materialize_pin_to_host
/// and @ref materialize_all_batches_compressed.
struct compression_pin_config {
  bool enabled{false};
  std::string plan_dsl;
  std::size_t min_batch_size_bytes{0};
  /// Keep the compressed form only if header+payload <= this fraction of the
  /// batch's original device size; otherwise pin uncompressed. See
  /// compression_config::max_compressed_fraction.
  double max_compressed_fraction{0.95};
  std::vector<std::string> column_names;
};

/// One GPU-tier chunk of a compression-enabled pin, in emission order. Exactly
/// one form is populated: a batch that qualified for compression carries
/// @c compressed (a compressed_device_representation holding every pinned column);
/// a batch pinned uncompressed carries @c columns (one device column per pinned
/// column, positional with the pin's column_ids) plus the @c memory_space it
/// resides in. Serving dispatches per chunk on which form is set, so a single
/// pin may freely interleave the two — mirrors the host-tier host_chunks design.
struct device_pin_chunk {
  std::shared_ptr<sirius::compressed_device_representation> compressed;
  std::vector<std::shared_ptr<cudf::column>> columns;
  cucascade::memory::memory_space* memory_space{nullptr};
};

/// Result of driving a GPU-tier pin with optional Simpatico compression. Each
/// emitted batch becomes one @ref device_pin_chunk in emission order — compressed
/// when it qualified, uncompressed otherwise — so compressed and uncompressed
/// chunks may be interleaved within a single pin.
struct device_pin_result {
  std::vector<device_pin_chunk> chunks;
  /// Chunk-major stored-column metadata, recorded from each GPU table right before it is
  /// compressed or split into device columns. Parallel to @c chunks and their columns.
  pinned_column_storage_matrix column_storage;
  /// Row count of each materialized batch, in emission order (covers compressed
  /// and uncompressed chunks alike); becomes duckdb_mvcc_metadata::
  /// base_row_count_per_chunk for duckdb-format pins.
  std::vector<std::size_t> base_row_count_per_chunk;
};

/// Drive @p ingestible to completion, streaming each emitted batch to pinned host memory
/// (peak GPU residency ~one batch), optionally compressing each batch with Simpatico first.
///
/// \param pinned_column_types Pin-time DuckDB type of each batch column, in batch-column
///                            (column_ids) order — drives the per-chunk zone-map capture,
///                            which runs on the decode GPU before host conversion /
///                            compression (so compressed and uncompressed pins alike get
///                            zone maps). When carrier narrowing is enabled, this vector must
///                            contain one type per materialized column. Otherwise, an empty vector
///                            skips capture (statless pin).
/// \param compression         Per-table Simpatico settings; disabled pins every chunk
///                            uncompressed.
/// \param options             Per-pin materialization behavior (zone-map capture, carrier
///                            narrowing). Narrowing runs before compression, so a compressed
///                            chunk stores narrow columns and decompresses straight back to them.
/// \return The pinned host chunks in materialization (round-robin) order — one per emitted
///         batch — plus their per-chunk row counts, stored-column metadata, and zone-map
///         captures.
host_pin_result materialize_pin_to_host(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  const std::unordered_map<int, cucascade::memory::memory_space*>& host_space_by_gpu,
  io::ioctx& io_ctx,
  duckdb::vector<duckdb::LogicalType> const& pinned_column_types,
  compression_pin_config const& compression,
  pin_materialization_options options = {});

/// Drive @p ingestible to completion like @ref materialize_all_batches, optionally
/// compressing each batch with Simpatico and keeping the compressed payload in GPU
/// (device) memory. A batch that does not qualify (below the size threshold, or it
/// fails to compress usefully) is kept as an uncompressed device chunk instead, so
/// @c device_pin_result::chunks may interleave the two forms in emission order.
/// Carrier narrowing runs before compression exactly as in @ref materialize_pin_to_host.
/// Zone-map capture is forced off:
/// @c device_pin_result carries no statistics and @c insert_pinned_entry_device stores
/// none, so device pins keep the statless-pin serving behavior.
///
/// \param pinned_column_types Pin-time DuckDB type of each batch column, in batch-column
///                            (column_ids) order — drives the carrier narrowing. Empty is
///                            valid only when narrowing is off.
/// \param compression         Per-table Simpatico settings; disabled pins every chunk
///                            uncompressed.
/// \param options             Per-pin materialization behavior (carrier narrowing).
device_pin_result materialize_all_batches_compressed(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  io::ioctx& io_ctx,
  duckdb::vector<duckdb::LogicalType> const& pinned_column_types,
  compression_pin_config const& compression,
  pin_materialization_options options = {});

}  // namespace sirius
