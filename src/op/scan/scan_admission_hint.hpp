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

#include <cstdint>
#include <optional>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Plan-time facts about a scan that GPU admission (planner/gpu_admission.hpp) uses in
 *        place of the planner's filtered cardinality and the configured average string width.
 *
 * Filled by the plan generator from the table function's own cardinality and statistics
 * callbacks and, for Parquet, from the first file's footer. Every field is optional: admission
 * falls back to the planner estimate and `avg_variable_column_bytes` wherever a field is absent.
 */
struct scan_admission_hint {
  /// Rows the table function reports for the whole table or file set, before any filter. The
  /// planner's `estimated_cardinality` applies DuckDB's filter selectivity, which under-counts
  /// range and LIKE filters by 4-5x on TPC-H; admission only ever takes the larger of the two,
  /// because under-estimating puts a large query on too few GPUs while over-estimating merely
  /// keeps today's allocation.
  std::optional<uint64_t> unfiltered_rows;
  /// Bytes per row of each output column (plan order) where a source could say: Parquet column
  /// chunks' uncompressed size per value for columns stored verbatim, or the table statistics'
  /// maximum string length plus the offset word. Only consulted for variable-width columns;
  /// fixed-width carriers are exact already.
  std::vector<std::optional<uint64_t>> column_widths;
};

}  // namespace sirius::op::scan
