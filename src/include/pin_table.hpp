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

}  // namespace sirius
