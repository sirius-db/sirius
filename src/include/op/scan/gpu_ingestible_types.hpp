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
#include <utility>
#include <vector>

#pragma once

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-table bind data carrier; polymorphic factory for gpu_ingestible.
 *
 * Built from a scan binding by the plan generator or by pin_table, then passed
 * to make_ingestible. prepare_for_query reads it back to match pinned entries.
 * Implementations: parquet_ingestible_table_info,
 * duckdb_native_ingestible_table_info.
 */
class ingestible_table_info {
 public:
  virtual ~ingestible_table_info() = default;

  ingestible_table_info(ingestible_table_info const&)            = delete;
  ingestible_table_info& operator=(ingestible_table_info const&) = delete;

  [[nodiscard]] virtual std::span<std::string const> column_names() const = 0;

  /**
   * @brief Resolved file paths captured at bind time.
   *
   * Used by sirius_scan_manager to match an incoming scan against pinned
   * entries. Returned span
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
   * @brief Estimated decoded bytes for projected data columns before row filtering.
   *
   * Read by @c scan_operator_input::get_estimated_size_in_bytes for the
   * reservation system and execution history. A format may use decoded read
   * columns as a nonzero history basis when no data column is projected.
   * Returns 0 for splits with no a-priori size estimate.
   */
  [[nodiscard]] virtual std::size_t estimated_bytes() const noexcept { return 0; }

  /**
   * @brief Estimated decoded column buffers needed to materialize the split.
   *
   * Defaults to the projected-column estimate. Formats that decode additional
   * transient columns, such as parquet pure-filter columns, override this to
   * expose that memory separately from the execution-history basis. Decoder
   * scratch and synthesized columns are not included.
   */
  [[nodiscard]] virtual std::size_t estimated_working_set_bytes() const noexcept
  {
    return estimated_bytes();
  }

 protected:
  template <typename RangeFactory>
  static void append_fadvise_entry(std::vector<fadvise_entry>& entries,
                                   std::shared_ptr<sirius::io::sirius_datasource> const& datasource,
                                   RangeFactory&& make_ranges)
  {
    if (!datasource || !datasource->uses_prefetching_cache()) { return; }
    auto ranges = std::forward<RangeFactory>(make_ranges)();
    if (ranges.empty()) { return; }
    entries.push_back(fadvise_entry{datasource, std::move(ranges)});
  }
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
  /// Positions in @c table delivered as a BOOL8 predicate result rather than
  /// values (see sirius::pushdown_outcome::predicate_columns). Their filter
  /// conjunct is already answered, so post_filter_and_project references the
  /// column instead of re-expressing the comparison. Empty on every path that
  /// substitutes nothing.
  std::vector<std::size_t> predicate_columns;
  /// The decode also applied those conjuncts to the rows (see
  /// sirius::pushdown_outcome::predicates_enforced), so post_filter_and_project
  /// drops them from the residual instead of referencing the answer.
  bool predicates_enforced{false};
};

}  // namespace sirius::op::scan
