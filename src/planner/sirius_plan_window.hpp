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
#include "op/window/aggregate_types.hpp"
#include "op/window/cumulative.hpp"
#include "op/window/frame.hpp"
#include "op/window/framed.hpp"
#include "op/window/ranking.hpp"
#include "op/window/whole_partition.hpp"
#include "planner/sirius_plan_sort_exchange.hpp"
#include "sirius_config.hpp"

#include <cstdint>
#include <memory>
#include <optional>
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

// An aggregate over a frame that is not the whole partition, evaluated over input sorted on the
// partition and order keys.
struct framed_window_expression {
  aggregate_id function;
  duckdb::idx_t argument_slot;
  duckdb::LogicalType argument_type;
  // DuckDB's exact result type, as for whole_partition_expression.
  duckdb::LogicalType result_type;
  op::window_frame frame;
  duckdb::idx_t output_slot;
};

/// The kinds of window expression. A stage holds one kind.
enum class window_kind : std::uint8_t { whole_partition, ranking, cumulative, framed };

// Planning IR only: not an executable operator. All slots index the prepared input
// (original columns followed by hidden expressions), except output_slot, which
// indexes the visible output (original columns followed by window results).
// `kind` says what a window holds: `expressions` with aggregates over the whole partition,
// `rankings` with ranking functions over `order`, `expressions` with running aggregates over
// `order` (the frame `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`), or `framed` with
// aggregates over other frames of `order`.
struct window_plan {
  duckdb::idx_t input_width;
  duckdb::vector<duckdb::LogicalType> prepared_types;
  duckdb::vector<duckdb::LogicalType> output_types;
  std::vector<duckdb::idx_t> partition_slots;
  std::vector<std::unique_ptr<ast::node>> hidden_expressions;
  std::vector<whole_partition_expression> expressions;
  // How the evaluator computes `expressions`, set for aggregates over the whole partition.
  std::optional<op::whole_partition_plan> aggregate_plan;
  std::vector<ranking_window_expression> rankings;
  std::vector<framed_window_expression> framed;
  // How the framed evaluator computes `framed`, set when there are any.
  std::optional<op::framed_window_plan> framed_plan;
  // Order key slots, direction and NULL placement shared by all `rankings`, all running
  // aggregates or all `framed`.
  op::window_order order;
  window_kind kind = window_kind::whole_partition;
};

/**
 * @brief Plan all expressions of a window as one stage.
 *
 * A stage holds aggregates over the whole partition, ranking functions, running aggregates (the
 * frame `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` where the running evaluator supports
 * the function and argument), or aggregates over other frames that are not the whole partition.
 * Its expressions must have equivalent `PARTITION BY` keys and, except for aggregates over the
 * whole partition, the same `ORDER BY`. Framed aggregates may each have their own frame. The
 * physical plan generator splits a window that does not fit one stage and plans each stage with
 * this rule. The tests use this function to check the plan of a single stage.
 *
 * A key or an argument that is not a column reference becomes a hidden column of the prepared
 * input. Each distinct expression is evaluated once for the stage, and the hidden columns are
 * removed when the visible output is published.
 *
 * The plan is refused, even when the input is empty, if:
 * - an expression is not a form the window evaluators run, such as `DISTINCT`, `FILTER`,
 *   `EXCLUDE`, or an unsupported function or result type
 * - an aggregate's frame is not the whole partition and the window has no `ORDER BY`, or is not
 *   a running aggregate and has no `PARTITION BY`, uses `GROUPS` or `RANGE` offsets, or has a frame
 * offset that is not a non-negative `BIGINT` constant after optimization
 * - the expressions mix kinds, or do not share one specification
 * - a key or an argument is volatile, cannot be translated, or has an unsupported type
 *
 * @param window The window to plan. Its column references must be resolved. It must have one child
 * and at least one expression.
 * @return The plan: the prepared input schema, the partition key slots, the aggregate expressions,
 * the ranking functions or the framed aggregates with their order keys, and the output schema.
 * @throws duckdb::NotImplementedException if the window is refused. The query then falls back to
 * the CPU.
 */
window_plan plan_window(duckdb::LogicalWindow const& window);

/**
 * @brief Build the input side of a window: prepare, PARTITION, then CONCAT.
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
 * @param plan Plan from `plan_window`.
 * @param estimated_cardinality Estimated number of input rows.
 * @param hash_partition_bytes Target size of a bucket in bytes. Must be positive.
 * @return The root of the prepared input.
 * @throws duckdb::InternalException if the child's schema differs from the plan's input schema.
 * @throws std::invalid_argument if @p hash_partition_bytes is zero.
 */
duckdb::unique_ptr<op::sirius_physical_operator> build_window_input(
  duckdb::unique_ptr<op::sirius_physical_operator> child,
  window_plan const& plan,
  std::size_t estimated_cardinality,
  uint64_t hash_partition_bytes = config::DEFAULT_HASH_PARTITION_BYTES);

// Insert only the exchange/assembly around an already prepared child. SQL lowering calls this
// after compressed-schema propagation, alongside the other pipeline wrapper insertions.
duckdb::unique_ptr<op::sirius_physical_operator> build_window_exchange(
  duckdb::unique_ptr<op::sirius_physical_operator> child,
  std::vector<duckdb::idx_t> const& partition_slots,
  std::size_t estimated_cardinality,
  uint64_t hash_partition_bytes);

/**
 * @brief Build the sorted input of a framed window: a distributed sort cut on the partition keys.
 *
 * Puts `MERGE_SORT → SORT_PARTITION → SORT_SAMPLE → ORDER_BY` on the prepared @p child, built by
 * `build_sort_exchange`. ORDER_BY sorts on the partition keys followed by @p order. Range
 * boundaries are sampled and compared on the partition keys only, and rows equal on the compared
 * keys are never split between ranges, so every partition lands whole in one range. MERGE_SORT
 * publishes each range sorted on the full key.
 *
 * Only adjacency of equal partition keys matters to the window, so the partition keys are sorted
 * `ASC NULLS FIRST`. This is the only place that chooses that order.
 *
 * Insert after compressed-schema propagation, alongside the other pipeline wrapper insertions.
 *
 * @param child The prepared window input.
 * @param partition_slots The partition key slots of the prepared input. Must not be empty.
 * @param order The window ORDER BY over slots of the prepared input.
 * @param estimated_cardinality Estimated number of input rows.
 * @param options Sort sizing. Its boundary key count is replaced by the number of partition keys.
 * @return The MERGE_SORT at the root of the input.
 * @throws std::invalid_argument if @p child is null, @p partition_slots is empty, or a slot is
 * outside the child's schema.
 */
duckdb::unique_ptr<op::sirius_physical_operator> build_window_sort_exchange(
  duckdb::unique_ptr<op::sirius_physical_operator> child,
  std::vector<duckdb::idx_t> const& partition_slots,
  op::window_order const& order,
  std::size_t estimated_cardinality,
  sort_exchange_options options);

}  // namespace sirius::planner
