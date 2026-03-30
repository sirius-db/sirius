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

#include "transparent/sirius_optimizer_extension.hpp"

#include "config.hpp"
#include "log/logging.hpp"
#include "sirius_context.hpp"

#include <duckdb/common/enums/logical_operator_type.hpp>

namespace sirius::transparent {

bool is_acceleratable_query(const duckdb::LogicalOperator& root)
{
  // Check if this node's operator type is supported by sirius_physical_plan_generator.
  // This mirrors the switch statement in sirius_physical_plan_generator::create_plan().
  switch (root.type) {
    // Supported operators:
    case duckdb::LogicalOperatorType::LOGICAL_GET:
    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
    case duckdb::LogicalOperatorType::LOGICAL_FILTER:
    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
    case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY:
    case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
    case duckdb::LogicalOperatorType::LOGICAL_TOP_N:
    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_DUMMY_SCAN:
    case duckdb::LogicalOperatorType::LOGICAL_EMPTY_RESULT:
    case duckdb::LogicalOperatorType::LOGICAL_CHUNK_GET:
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_GET:
    case duckdb::LogicalOperatorType::LOGICAL_EXPRESSION_GET:
    case duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
    case duckdb::LogicalOperatorType::LOGICAL_CTE_REF: break;

    // Everything else is unsupported — skip GPU execution.
    default: return false;
  }

  // Recursively check all children.
  for (auto& child : root.children) {
    if (!is_acceleratable_query(*child)) { return false; }
  }
  return true;
}

void sirius_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                           duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  if (!duckdb::Config::ENABLE_TRANSPARENT_EXECUTION) { return; }

  auto& context = input.context;

  auto ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!ctx || !ctx->is_initialized()) { return; }

  if (!is_acceleratable_query(*plan)) {
    SIRIUS_LOG_DEBUG("Transparent execution: query not acceleratable, skipping GPU");
    return;
  }

  try {
    auto plan_copy = plan->Copy(context);
    ctx->set_captured_logical_plan(std::move(plan_copy));
    SIRIUS_LOG_DEBUG("Transparent execution: logical plan captured for GPU execution");
  } catch (duckdb::NotImplementedException&) {
    SIRIUS_LOG_DEBUG("Transparent execution: logical plan not serializable, skipping GPU");
  } catch (std::exception& e) {
    SIRIUS_LOG_DEBUG("Transparent execution: failed to copy logical plan: {}", e.what());
  }
}

}  // namespace sirius::transparent
