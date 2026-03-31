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

#include "util/plan_extraction.hpp"

#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"

namespace sirius::util {

duckdb::unique_ptr<duckdb::LogicalOperator> extract_optimized_plan(duckdb::ClientContext& context,
                                                                   const std::string& query,
                                                                   bool enable_optimizer)
{
  optimizer_guard guard(context, enable_optimizer);

  duckdb::Parser parser(context.GetParserOptions());
  parser.ParseQuery(query);
  if (parser.statements.empty()) { throw duckdb::BinderException("Empty or invalid query"); }
  if (parser.statements.size() > 1) {
    throw duckdb::BinderException("Only a single statement is supported");
  }

  duckdb::Planner planner(context);
  planner.CreatePlan(std::move(parser.statements[0]));
  D_ASSERT(planner.plan);

  auto plan = std::move(planner.plan);

  if (context.config.enable_optimizer) {
    duckdb::Optimizer optimizer(*planner.binder, context);
    plan = optimizer.Optimize(std::move(plan));
  }

  // After optimization, refresh types before column binding resolution
  // to ensure types are consistent (some optimizers may have set stale types)
  plan->ResolveOperatorTypes();

  duckdb::ColumnBindingResolver resolver;
  duckdb::ColumnBindingResolver::Verify(*plan);
  resolver.VisitOperator(*plan);

  return plan;
}

}  // namespace sirius::util
