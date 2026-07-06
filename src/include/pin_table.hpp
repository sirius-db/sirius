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
};

/// Drive @p ingestible 's metadata walk + batch coalescer to completion on @p io_ctx,
/// materializing every emitted batch into a GPU-resident cudf::table and round-robining
/// placement across @p gpu_spaces. Single-threaded with deterministic placement, so
/// re-pinning the same source yields identical @c chunk_memory_spaces (required by
/// @c sirius_scan_manager::insert_pinned_entry 's merge path).
///
/// \param ingestible  Source ingestible (parquet or duckdb-native). Consumed by repeated
///                    next_split_provider / materialize calls.
/// \param gpu_spaces  Non-empty set of GPU memory spaces to round-robin across.
/// \param io_ctx      IO context the metadata reads run on (owned by the scan manager).
materialized_pin materialize_all_batches(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  io::sirius_ioctx& io_ctx);

/// Result of driving a host-tier pin with optional Simpatico compression.
/// When every batch compresses successfully, @c compressed_chunks is non-empty
/// and @c host_chunks is empty; otherwise uncompressed chunks land in @c host_chunks.
struct host_pin_result {
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks;
  std::vector<std::shared_ptr<sirius::compressed_host_representation>> compressed_chunks;
};

/// Optional compression settings for @ref materialize_pin_to_host_with_compression
/// and @ref materialize_pin_to_device_with_compression.
struct compression_pin_config {
  bool enabled{false};
  std::string plan_dsl;
  std::size_t min_batch_size_bytes{0};
  std::vector<std::string> column_names;
};

/// Result of driving a GPU-tier pin with optional Simpatico compression.
/// When every batch compresses successfully, @c compressed_chunks is non-empty
/// and @c tables is empty; otherwise the uncompressed GPU tables (with their
/// per-chunk placement) land in @c tables / @c chunk_memory_spaces for the plain
/// insert_pinned_entry path.
struct device_pin_result {
  std::vector<std::unique_ptr<cudf::table>> tables;
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces;
  std::vector<std::shared_ptr<sirius::compressed_device_representation>> compressed_chunks;
};

/// Drive @p ingestible to completion like @ref materialize_pin_to_host, optionally
/// compressing each batch with Simpatico before storing it in host memory.
host_pin_result materialize_pin_to_host_with_compression(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  const std::unordered_map<int, cucascade::memory::memory_space*>& host_space_by_gpu,
  io::sirius_ioctx& io_ctx,
  compression_pin_config const& compression);

/// Drive @p ingestible to completion like @ref materialize_all_batches, optionally
/// compressing each batch with Simpatico and keeping the compressed payload in GPU
/// (device) memory. When compression is disabled or a batch does not qualify, the
/// uncompressed GPU table is retained instead (in @c device_pin_result::tables), so
/// the caller can fall back to the plain GPU pin.
device_pin_result materialize_pin_to_device_with_compression(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  io::sirius_ioctx& io_ctx,
  compression_pin_config const& compression);

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
/// \param host_space_by_gpu Maps each GPU device id to the host memory_space its batches should
///                          be pinned on (NUMA-local). Must contain an entry for every device id
///                          in @p gpu_spaces.
/// \param io_ctx            IO context the metadata reads run on (owned by the scan manager).
/// \return The pinned host chunks in materialization (round-robin) order — one per emitted batch.
std::vector<std::shared_ptr<cucascade::host_data_representation>> materialize_pin_to_host(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  const std::unordered_map<int, cucascade::memory::memory_space*>& host_space_by_gpu,
  io::sirius_ioctx& io_ctx);

}  // namespace sirius
