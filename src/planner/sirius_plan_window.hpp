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
#include "op/window/ranking.hpp"
#include "op/window/whole_partition.hpp"
#include "sirius_config.hpp"

#include <memory>
#include <vector>

namespace duckdb {
class LogicalWindow;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

// The fields beyond function and argument_slot describe the expression for later window stages.
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

struct ranking_window_expression {
  op::ranking_function function;
  duckdb::LogicalType result_type;
  duckdb::idx_t output_slot;
};

// Planning IR only: not an executable operator. All slots index the prepared input
// (original columns followed by hidden expressions), except output_slot, which
// indexes the visible output (original columns followed by window results).
// A window holds either `expressions` (aggregates over the whole partition) or `rankings`
// (ranking functions over `ranking_order`), never both.
struct whole_partition_window_plan {
  duckdb::idx_t input_width;
  duckdb::vector<duckdb::LogicalType> prepared_types;
  duckdb::vector<duckdb::LogicalType> output_types;
  std::vector<duckdb::idx_t> partition_slots;
  std::vector<std::unique_ptr<ast::node>> hidden_expressions;
  std::vector<whole_partition_expression> expressions;
  std::vector<ranking_window_expression> rankings;
  // Order key slots, direction and NULL placement shared by all `rankings`.
  op::ranking_order ranking_order;
};

// Keep integer window results out of consumers that would apply the generic HUGEINT -> INT64
// expression mapping. Passthrough references and ordering remain admitted. An integer SUM is an
// exact DECIMAL128 with scale 0 on the GPU, and that mapping can truncate a sum wider than 64 bits.
// Carrying HUGEINT as DECIMAL128 everywhere is tracked in
// https://github.com/sirius-db/sirius/issues/2067
void validate_whole_partition_window_consumers(duckdb::LogicalOperator const& plan);

// Requires resolved column references. Does not consume or modify the logical
// plan. Throws NotImplementedException for unsupported combinations, even on
// empty input. Hidden columns must survive through reduction/expansion and be
// removed when the visible output is published.
whole_partition_window_plan plan_whole_partition_window(duckdb::LogicalWindow const& window);

/**
 * @brief Build the input side of a whole-partition window: prepare, PARTITION, then CONCAT.
 *
 * Adds a projection on @p child that passes its columns through and appends the plan's hidden
 * expressions. Each `PARTITION BY` key or aggregate argument that is not a plain column reference
 * is computed once and added after them, so that PARTITION and WINDOW address every key and
 * argument as a column slot. PARTITION then hashes on the key slots and CONCAT (with
 * `concat_all` set) assembles each bucket into one batch. The result exposes `prepared_types`,
 * without the SQL window results. The executable WINDOW adapter consumes these complete prepared
 * buckets after pipeline wrapper insertion.
 *
 * @param child Producer of the window's input columns.
 * @param plan Plan from `plan_whole_partition_window`.
 * @param estimated_cardinality Estimated number of input rows.
 * @param hash_partition_bytes Target size of a bucket in bytes. Must be positive.
 * @return The root of the prepared input.
 * @throws duckdb::InternalException if the child's schema differs from the plan's input schema.
 * @throws std::invalid_argument if @p hash_partition_bytes is zero.
 */
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
