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
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/operator/logical_expression_get.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalExpressionGet& op)
{
  D_ASSERT(op.children.size() == 1);
  auto plan = create_plan(*op.children[0]);

  auto& allocator = duckdb::Allocator::Get(context);

  // Evaluate all constant expressions and materialize into a ColumnDataCollection.
  auto collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(context, op.types);

  duckdb::DataChunk chunk;
  chunk.Initialize(allocator, op.types);

  duckdb::ColumnDataAppendState append_state;
  collection->InitializeAppend(append_state);

  for (duckdb::idx_t expression_idx = 0; expression_idx < op.expressions.size(); expression_idx++) {
    chunk.Reset();
    auto& expr_list = op.expressions[expression_idx];

    duckdb::ExpressionExecutor executor(context, expr_list);
    executor.Execute(chunk);

    collection->Append(append_state, chunk);
  }

  SIRIUS_LOG_DEBUG("[plan_expression_get] Materialized {} rows into ColumnDataCollection",
                   collection->Count());

  auto chunk_scan = duckdb::make_uniq<sirius::op::sirius_physical_column_data_scan>(
    op.types,
    sirius::op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN,
    op.expressions.size(),
    std::move(collection));

  return std::move(chunk_scan);
}

}  // namespace sirius::planner
