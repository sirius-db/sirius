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

// sirius
#include <helper/logical_type.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/planner/table_filter.hpp>

namespace sirius::op {

/**
 * @brief Build a mapping from column_ids index to batch column position.
 *
 * The parquet scan (make_selected_column_indices) produces batch columns in
 * column_ids order, but only for indices present in projection_ids.
 * For example, if column_ids has 5 entries and projection_ids = {1, 3}:
 *   batch position 0 → column_ids[1]
 *   batch position 1 → column_ids[3]
 *
 * When projection_ids is empty, every column_ids entry maps to its own index.
 *
 * Returns a vector of size column_ids_count where:
 *   result[i] = batch position of column_ids[i], or idx_t(-1) if not projected.
 */
std::vector<duckdb::idx_t> build_batch_column_map(
  const duckdb::vector<duckdb::idx_t>& projection_ids, duckdb::idx_t column_ids_count);

/**
 * @brief Convert a DuckDB TableFilterSet into a single bound DuckDB expression (conjunction of
 * all filters), suitable for passing to gpu_expression_translator::translate_expression().
 *
 * Returns nullptr if the filter set is empty or contains only unsupported filter types.
 */
duckdb::unique_ptr<duckdb::Expression> convert_table_filters_to_expression(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::vector<duckdb::idx_t>& batch_column_map);

}  // namespace sirius::op
