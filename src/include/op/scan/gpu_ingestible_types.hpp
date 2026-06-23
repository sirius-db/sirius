/*
 * Copyright 2026, Sirius Contributors.
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

#include "op/scan/owning_table_view.hpp"

#include <io/sirius_datasource.hpp>

#include <memory>
#include <span>
#include <vector>

#pragma once

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-table bind data carrier; polymorphic factory for gpu_ingestible.
 *
 * Built by the pipeline converter when it lowers a DuckDB scan binding,
 * parked on the gpu scan operator until prepare_for_query, then handed to
 * @ref make_gpu_ingestible (or directly to a cached gpu_ingestible when a
 * pinned-cache match wins). Implementations: parquet_ingestible_table_info,
 * duckdb_native_ingestible_table_info.
 */
class ingestible_table_info {
 public:
  virtual ~ingestible_table_info() = default;

  ingestible_table_info(ingestible_table_info const&)            = delete;
  ingestible_table_info& operator=(ingestible_table_info const&) = delete;

  /// True iff this (cached) table info can serve @p other — i.e. it reads a
  /// superset of @p other's data so a pinned scan of this can feed @p other
  /// (after the @ref column_projections gather). Implementations define the
  /// per-format match (e.g. same files + column superset).
  [[nodiscard]] virtual std::vector<size_t> can_serve_with_columns(
    const ingestible_table_info& other) const
  {
    return {};
  }

  [[nodiscard]] virtual std::span<std::string const> column_names() const { return {}; }

  /**
   * @brief Resolved file paths captured at bind time.
   *
   * Used by sirius_scan_manager to match an incoming scan against pinned
   * entries before falling back to @ref make_ingestible. Returned span
   * must remain valid for the lifetime of @c *this.
   */
  [[nodiscard]] virtual std::span<std::string const> file_paths() const = 0;

 protected:
  ingestible_table_info() = default;
};

//===----------------------------------------------------------------------===//
// scan_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-split scan descriptor. Polymorphic; each gpu_ingestible
 *        implementation defines its own subclass with the per-split
 *        information its @ref gpu_ingestible::materialize_table requires.
 *
 * Distinct from per-table bind data (@ref ingestible_table_info): one
 * ingestible produces many @c scan_info instances during its lifetime —
 * one per emitted split.
 */
class scan_info : public std::enable_shared_from_this<scan_info> {
 public:
  /// One unit of work pushed by a provider.  A null @c datasource is the
  /// closure sentinel: the sequencer treats it as "slot done, move on".
  struct fadvise_entry {
    std::shared_ptr<sirius::io::sirius_datasource> datasource;
    std::vector<cudf::io::text::byte_range_info> ranges;
  };

  virtual ~scan_info() = default;

  virtual std::vector<fadvise_entry> fadvise_entries() const { return {}; }

  /**
   * @brief Estimated uncompressed byte count for the split.
   *
   * Read by @c scan_operator_input::get_estimated_size_in_bytes for the
   * reservation system; should reflect the GPU memory the materialize step
   * will allocate (parquet: sum of reserved_uncompressed_bytes across the
   * batch's row-group slices; duckdb-native: sum of row_group counts ×
   * column widths). Returns 0 for splits with no a-priori size estimate.
   */
  [[nodiscard]] virtual std::size_t estimated_bytes() const noexcept { return 0; }
};

//===----------------------------------------------------------------------===//
// filter_state / filtered_table
//===----------------------------------------------------------------------===//
/**
 * @brief How much of the per-split filter + projection work the ingestible
 *        already absorbed during @ref gpu_ingestible::materialize_table.
 *
 * Returned alongside the materialized table so the scan operator can skip
 * a redundant @ref gpu_ingestible::post_filter_and_project call when the
 * ingestible already applied both the row-level filter and projection
 * inline (the parquet reader-side pushdown path).
 */
enum class filter_state {
  UNFILTERED,                  // pinned table is an example of this
  ROWGROUP_FILTERED,           // hybrid_scan materialize is an example of this
  ROW_FILTERED,                // read_parquet is an example of this
  ROW_FILTERED_AND_PROJECTED,  // table for the particular query is cached
};

/**
 * @brief Result of @ref gpu_ingestible::materialize_table.
 *
 * Bundles the materialized cudf::table with a tag describing how much of
 * the split's filter + projection work was applied during materialization.
 */
struct filtered_table {
  owning_table_view table;
  filter_state state{filter_state::UNFILTERED};
};

}  // namespace sirius::op::scan
