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
}  // namespace cudf

namespace cucascade {
class host_data_representation;
}  // namespace cucascade

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace sirius {

namespace op::scan {
class gpu_ingestible;
class scan_info;
}  // namespace op::scan
namespace io {
class sirius_ioctx;
}  // namespace io

/// GPU-resident tables produced by driving a @c gpu_ingestible to completion, with
/// each table's GPU placement recorded in @c chunk_memory_spaces (parallel to
/// @c tables). Consumed by the pin_table path: fed directly into
/// @c sirius_scan_manager::insert_pinned_entry, or, after host conversion of each
/// table, into @c insert_pinned_entry_host.
struct materialized_pin {
  std::vector<std::unique_ptr<cudf::table>> tables;
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces;
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

/// Host-pinned chunks produced by @ref materialize_pin_to_host — one
/// host_data_representation per emitted batch, with the batch row counts captured
/// alongside (parallel to @c host_chunks), mirroring @ref materialized_pin.
struct materialized_host_pin {
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks;
  std::vector<std::size_t> base_row_count_per_chunk;
  /// Per-chunk zone-map capture (taken on the GPU before the host conversion);
  /// chunk_stats[c][i] = stats of batch column i of chunk c (null = none).
  /// Parallel to @c host_chunks when capture ran; empty when capture was skipped.
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
///                            (column_ids) order — drives the per-chunk zone-map capture
///                            (compute_pinned_chunk_stats). Empty skips capture (statless pin).
materialized_pin materialize_all_batches(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  io::sirius_ioctx& io_ctx,
  duckdb::vector<duckdb::LogicalType> const& pinned_column_types);

/// Drive @p ingestible to completion like @ref materialize_all_batches, but stream each
/// emitted batch straight to pinned host memory instead of collecting GPU-resident tables:
/// materialize one batch on its round-robin GPU, convert it to a @c host_data_representation
/// on that GPU's NUMA-local host space, then free the GPU table before the next batch. Peak
/// GPU residency is therefore ~one batch (governed by @c scan_task_batch_size), so a host pin
/// never needs the whole table to fit in GPU memory.
///
/// \param ingestible        Source ingestible (parquet or duckdb-native).
/// \param gpu_spaces        Non-empty set of GPU memory spaces to round-robin materialization
/// across.
/// \param host_space_by_gpu   Maps each GPU device id to the host memory_space its batches should
///                            be pinned on (NUMA-local). Must contain an entry for every device id
///                            in @p gpu_spaces.
/// \param io_ctx              IO context the metadata reads run on (owned by the scan manager).
/// \param pinned_column_types Pin-time DuckDB type of each batch column, in batch-column
///                            (column_ids) order — drives the per-chunk zone-map capture, which
///                            runs on the decode GPU before the host conversion. Empty skips
///                            capture (statless pin).
/// \return The pinned host chunks in materialization (round-robin) order — one per emitted
///         batch — plus their per-chunk row counts and zone-map captures.
materialized_host_pin materialize_pin_to_host(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  const std::unordered_map<int, cucascade::memory::memory_space*>& host_space_by_gpu,
  io::sirius_ioctx& io_ctx,
  duckdb::vector<duckdb::LogicalType> const& pinned_column_types);

}  // namespace sirius
