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

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_expression_get.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalExpressionGet& op)
{
  D_ASSERT(op.children.size() == 1);
  auto plan = create_plan(*op.children[0]);

  // Evaluate all expressions at plan time using DuckDB's ExpressionExecutor
  // and materialize results into a ColumnDataCollection. This mirrors DuckDB's
  // own foldable optimization in plan_expression_get.cpp.
  auto collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(context, op.types);

  auto& allocator = duckdb::Allocator::Get(context);
  duckdb::DataChunk chunk;
  chunk.Initialize(allocator, op.types);

  duckdb::ColumnDataAppendState append_state;
  collection->InitializeAppend(append_state);
  for (std::size_t expression_idx = 0; expression_idx < op.expressions.size(); expression_idx++) {
    duckdb::ExpressionExecutor executor(context, op.expressions[expression_idx]);
    chunk.Reset();
    executor.Execute(chunk);
    collection->Append(append_state, chunk);
  }

  auto chunk_scan = duckdb::make_uniq<sirius::op::sirius_physical_column_data_scan>(
    op.types,
    sirius::op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN,
    op.expressions.size(),
    std::move(collection));

  return std::move(chunk_scan);
}

}  // namespace sirius::planner
