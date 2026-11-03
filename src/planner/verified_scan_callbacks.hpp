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

#include <duckdb/function/table_function.hpp>

#include <tuple>

namespace sirius::planner::detail {
// Compare semantic callbacks and scan settings before inspecting provider-owned info.
// Audited against DuckDB 88ecd0a BaseTableFunction: only get_metrics, to_string,
// and table_scan_progress are excluded as presentation/profiling. Info identity
// and layout are checked separately by the provider-specific caller.
// The host's operator== omits semantic fields and includes presentation hooks,
// so it cannot replace this comparison. Bound call values are read-identity input.
template <class Function>
bool same_scan_callbacks(duckdb::TableFunction const& reference, Function const& candidate)
{
  return std::tie(reference.function,
                  reference.bind,
                  reference.bind_replace,
                  reference.bind_operator,
                  reference.init_global,
                  reference.init_local,
                  reference.in_out_function,
                  reference.in_out_function_final,
                  reference.statistics,
                  reference.statistics_extended,
                  reference.dependency,
                  reference.cardinality,
                  reference.pushdown_complex_filter,
                  reference.pushdown_expression,
                  reference.get_partition_data,
                  reference.get_bind_info,
                  reference.get_table_entry,
                  reference.to_sql,
                  reference.projection_expression_pushdown,
                  reference.get_multi_file_reader,
                  reference.supports_pushdown_type,
                  reference.supports_pushdown_extract,
                  reference.get_partition_info,
                  reference.get_partition_stats,
                  reference.get_virtual_columns,
                  reference.get_row_id_columns,
                  reference.set_scan_order,
                  reference.serialize,
                  reference.deserialize,
                  reference.verify_serialization,
                  reference.projection_pushdown,
                  reference.filter_pushdown,
                  reference.filter_prune,
                  reference.sampling_pushdown,
                  reference.late_materialization,
                  static_cast<duckdb::BaseTableFunction const&>(reference).return_type,
                  reference.call_return_type,
                  reference.order_preservation_type,
                  reference.global_initialization,
                  reference.GetSignature(),
                  reference.GetName(),
                  reference.is_repeatable,
                  reference.set_partitions_to_scan,
                  reference.parallelism) ==
         std::tie(candidate.function,
                  candidate.bind,
                  candidate.bind_replace,
                  candidate.bind_operator,
                  candidate.init_global,
                  candidate.init_local,
                  candidate.in_out_function,
                  candidate.in_out_function_final,
                  candidate.statistics,
                  candidate.statistics_extended,
                  candidate.dependency,
                  candidate.cardinality,
                  candidate.pushdown_complex_filter,
                  candidate.pushdown_expression,
                  candidate.get_partition_data,
                  candidate.get_bind_info,
                  candidate.get_table_entry,
                  candidate.to_sql,
                  candidate.projection_expression_pushdown,
                  candidate.get_multi_file_reader,
                  candidate.supports_pushdown_type,
                  candidate.supports_pushdown_extract,
                  candidate.get_partition_info,
                  candidate.get_partition_stats,
                  candidate.get_virtual_columns,
                  candidate.get_row_id_columns,
                  candidate.set_scan_order,
                  candidate.serialize,
                  candidate.deserialize,
                  candidate.verify_serialization,
                  candidate.projection_pushdown,
                  candidate.filter_pushdown,
                  candidate.filter_prune,
                  candidate.sampling_pushdown,
                  candidate.late_materialization,
                  static_cast<duckdb::BaseTableFunction const&>(candidate).return_type,
                  candidate.call_return_type,
                  candidate.order_preservation_type,
                  candidate.global_initialization,
                  candidate.GetSignature(),
                  candidate.GetName(),
                  candidate.is_repeatable,
                  candidate.set_partitions_to_scan,
                  candidate.parallelism);
}
}  // namespace sirius::planner::detail
