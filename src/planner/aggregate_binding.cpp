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

#include "planner/aggregate_binding.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/main/client_context.hpp"

#include <utility>

namespace sirius::planner {

duckdb::unique_ptr<duckdb::BoundAggregateExpression> bind_builtin_aggregate(
  duckdb::ClientContext& context,
  char const* name,
  duckdb::LogicalType const& type,
  duckdb::unique_ptr<duckdb::Expression> argument,
  duckdb::unique_ptr<duckdb::Expression> filter)
{
  auto& entry =
    duckdb::Catalog::GetSystemCatalog(context).GetEntry<duckdb::AggregateFunctionCatalogEntry>(
      context, DEFAULT_SCHEMA, name);
  auto function = entry.functions.GetFunctionByArguments(context, {type});
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
  children.push_back(std::move(argument));
  duckdb::FunctionBinder binder(context);
  return binder.BindAggregateFunction(std::move(function), std::move(children), std::move(filter));
}

bool is_plain_aggregate(duckdb::BoundAggregateExpression const& aggregate)
{
  return !aggregate.IsDistinct() && !aggregate.GetFilter() && !aggregate.GetOrderBys() &&
         aggregate.StateExportMode() != duckdb::AggregateStateExportMode::STATE_EXPORT;
}

}  // namespace sirius::planner
