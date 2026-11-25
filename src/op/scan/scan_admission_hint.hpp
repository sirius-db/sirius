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

#pragma once

#include "helper/logical_type.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Plan-time facts about a scan that GPU admission (planner/gpu_admission.hpp) uses in
 *        place of the planner's filtered cardinality and the configured average string width.
 *
 * Filled by the plan generator from the Parquet footers' row counts and string sizes and the
 * table function's own cardinality and statistics callbacks. Every field is optional: admission
 * falls back to the planner estimate and `avg_variable_column_bytes` wherever a field is absent.
 */
struct scan_admission_hint {
  /// Rows of the whole table or file set, before any filter: the Parquet footers' row counts,
  /// or what the table function reports. The planner's `estimated_cardinality` applies DuckDB's
  /// filter selectivity, which under-counts range and LIKE filters by 4-5x on TPC-H; admission
  /// only ever takes the larger of the two, because under-estimating puts a large query on too
  /// few GPUs while over-estimating merely keeps today's allocation.
  std::optional<uint64_t> unfiltered_rows;
  /// Bytes per row of each output column (plan order) where a source could say: the Parquet
  /// footers' decoded string size per value, or the table statistics' maximum string length,
  /// each plus the offset word. Only consulted for variable-width columns; fixed-width carriers
  /// are exact already.
  std::vector<std::optional<uint64_t>> column_widths;
  /// Types of the columns the scan reads only to evaluate pushed-down filters. The scan decodes
  /// them like its output columns, so they count towards its size, but they are not part of its
  /// output: a `NOT LIKE` on a long string column can be most of a scan that outputs two keys.
  std::vector<sirius::logical_type> filter_only_types;
  /// Bytes per row of each filter-only column where a source could say, parallel to
  /// `filter_only_types` and read like `column_widths`.
  std::vector<std::optional<uint64_t>> filter_only_widths;
};

}  // namespace sirius::op::scan
