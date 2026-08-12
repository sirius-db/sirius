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

#include "planner/dynamic_filter/build_filter_evidence.hpp"

#include "duckdb/planner/operator/logical_get.hpp"

namespace sirius::planner {

bool build_subtree_is_filtering(duckdb::LogicalOperator const& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_GET:
      return !op.Cast<duckdb::LogicalGet>().table_filters.filters.empty();
    case duckdb::LogicalOperatorType::LOGICAL_FILTER:
    case duckdb::LogicalOperatorType::LOGICAL_TOP_N: return true;
    default: break;
  }
  for (auto const& child : op.children) {
    if (child && build_subtree_is_filtering(*child)) { return true; }
  }
  return false;
}

bool build_relation_is_derived(duckdb::LogicalOperator const& op)
{
  switch (op.type) {
    // Childless derived leaves: the filtering walk bottoms out here, so its "unfiltered" verdict
    // means "cannot see", not "whole key domain".
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_GET:
    case duckdb::LogicalOperatorType::LOGICAL_CTE_REF:
    // Reducing operators: their output key set is a subset of the domain even when no predicate
    // appears anywhere below them.
    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_ANY_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
    case duckdb::LogicalOperatorType::LOGICAL_DISTINCT:
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT:
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT: return true;
    default: break;
  }
  for (auto const& child : op.children) {
    if (child && build_relation_is_derived(*child)) { return true; }
  }
  return false;
}

}  // namespace sirius::planner
