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

#include "duckdb/common/types.hpp"
#include "duckdb/parser/expression/window_expression.hpp"
#include "expression/ast/node.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/window/whole_partition.hpp"
#include "sirius_config.hpp"

#include <memory>
#include <vector>

namespace duckdb {
class LogicalWindow;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

struct whole_partition_expression {
  aggregate_id function;
  duckdb::idx_t argument_slot;
  duckdb::LogicalType argument_type;
  // Keep DuckDB's exact result type, including HUGEINT and DECIMAL(38, scale).
  duckdb::LogicalType result_type;
  duckdb::WindowBoundary start;
  duckdb::WindowBoundary end;
  duckdb::idx_t output_slot;
};

// Planning IR only: not an executable operator. All slots index the prepared input
// (original columns followed by hidden expressions), except output_slot, which
// indexes the visible output (original columns followed by window results).
struct whole_partition_window_plan {
  duckdb::idx_t input_width;
  duckdb::vector<duckdb::LogicalType> prepared_types;
  duckdb::vector<duckdb::LogicalType> output_types;
  std::vector<duckdb::idx_t> partition_slots;
  std::vector<std::unique_ptr<ast::node>> hidden_expressions;
  std::vector<whole_partition_expression> expressions;
};

// Keep integer window results out of consumers that would apply the generic HUGEINT -> INT64
// expression mapping. Passthrough references and ordering remain admitted.
void validate_whole_partition_window_consumers(duckdb::LogicalOperator const& plan);

// Requires resolved column references. Does not consume or modify the logical
// plan. Throws NotImplementedException for unsupported combinations, even on
// empty input. Hidden columns must survive through reduction/expansion and be
// removed when the visible output is published.
whole_partition_window_plan plan_whole_partition_window(duckdb::LogicalWindow const& window);

// Build only prepared-input -> PARTITION -> CONCAT (concat_all = true). The result
// exposes prepared_types, without SQL window results. The executable WINDOW adapter
// consumes these complete prepared buckets after pipeline wrapper insertion.
duckdb::unique_ptr<op::sirius_physical_operator> build_whole_partition_window_input(
  duckdb::unique_ptr<op::sirius_physical_operator> child,
  whole_partition_window_plan const& plan,
  std::size_t estimated_cardinality,
  uint64_t hash_partition_bytes = config::DEFAULT_HASH_PARTITION_BYTES);

// Insert only the exchange/assembly around an already prepared child. SQL lowering calls this
// after compressed-schema propagation, alongside the other pipeline wrapper insertions.
duckdb::unique_ptr<op::sirius_physical_operator> build_whole_partition_window_exchange(
  duckdb::unique_ptr<op::sirius_physical_operator> child,
  std::vector<duckdb::idx_t> const& partition_slots,
  std::size_t estimated_cardinality,
  uint64_t hash_partition_bytes);

}  // namespace sirius::planner
