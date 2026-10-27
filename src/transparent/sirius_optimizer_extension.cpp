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

#include "op/scan/table_scan/bound_read_view.hpp"
#include "planner/connector_registry.hpp"
#include "sirius_context.hpp"
#include "transparent/connection_provenance.hpp"

#include <duckdb/common/enums/optimizer_type.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/planner/binder.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <log/logging.hpp>
#include <util/duckdb_error_message.hpp>

#include <cstddef>
#include <exception>
#include <utility>
#include <vector>

namespace sirius::transparent {

namespace {

bool gpu_execution_enabled(const duckdb::ClientContext& context)
{
  duckdb::Value setting;
  auto lookup_result = context.TryGetCurrentSetting("gpu_execution", setting);
  return lookup_result && !setting.IsNull() && setting.GetValue<bool>();
}

void preserve_parquet_bindings(duckdb::LogicalOperator const& from,
                               duckdb::LogicalOperator& to,
                               duckdb::ClientContext& context)
{
  if (from.type != to.type || from.children.size() != to.children.size()) {
    throw duckdb::NotImplementedException("Plan copy changed the bound scan structure");
  }
  if (from.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto const& original = from.Cast<duckdb::LogicalGet>();
    auto& copied         = to.Cast<duckdb::LogicalGet>();
    auto const* source   = sirius::planner::lookup_connector(original, context);
    if (source &&
        (source->function_name == "read_parquet" || source->function_name == "parquet_scan")) {
      if (original.table_index != copied.table_index ||
          sirius::planner::lookup_connector(copied, context) != source) {
        throw duckdb::NotImplementedException("Plan copy changed the Parquet source");
      }
      // Serialization rebinds the wildcard and loses Hive pruning. Copy the verified
      // bind data instead; DuckDB gives the copy its own materialized file list.
      copied.bind_data = original.bind_data->Copy();
    }
  }
  for (std::size_t i = 0; i < from.children.size(); ++i) {
    preserve_parquet_bindings(*from.children[i], *to.children[i], context);
  }
}

}  // namespace

void sirius_pre_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                               duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  if (!plan) { return; }
  auto& context = input.context;
  auto ctx      = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!ctx) { return; }
  auto conn_state = duckdb::get_sirius_connection_state(context);
  if (!conn_state || conn_state->is_internal_query_active()) { return; }
  // Decide before optimization can remove scans, including when GPU execution is off.
  (void)should_use_duckdb(context, plan.get(), &input.optimizer.binder.GetStatementProperties());
  // DuckDB v2 derives the single-table restrictions implied by an OR-ed join condition itself
  // (JoinDependentFilterRule now also runs on INNER and SEMI join conditions, which is where the
  // TPC-H q19 and q7 disjunctions end up). This hook used to derive them too, for DuckDB 1.5, which
  // missed q19's. Deriving both made every pushed restriction appear twice: TPC-H q19's part scan
  // evaluated each column's OR filter twice and its FILTER carried two copies of the per-branch
  // restriction, which cost about 0.1 s at SF1000. The hook therefore only classifies the query.
}

bool copy_cardinality_estimates(duckdb::LogicalOperator const& from, duckdb::LogicalOperator& to)
{
  if (from.type != to.type || from.children.size() != to.children.size()) { return false; }
  to.estimated_cardinality     = from.estimated_cardinality;
  to.has_estimated_cardinality = from.has_estimated_cardinality;
  bool matched                 = true;
  for (std::size_t i = 0; i < from.children.size(); ++i) {
    if (!copy_cardinality_estimates(*from.children[i], *to.children[i])) { matched = false; }
  }
  return matched;
}

duckdb::unique_ptr<duckdb::LogicalOperator> copy_logical_plan(duckdb::LogicalOperator const& plan,
                                                              duckdb::ClientContext& context)
{
  auto copy = plan.Copy(context);
  preserve_parquet_bindings(plan, *copy, context);
  if (!copy_cardinality_estimates(plan, *copy)) {
    SIRIUS_LOG_DEBUG(
      "Transparent execution: plan copy differs in shape from the original, some cardinality "
      "estimates were not copied");
  }
  return copy;
}

void sirius_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                           duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  auto& context = input.context;

  auto ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!ctx) { return; }
  auto conn_state = duckdb::get_sirius_connection_state(context);
  if (!conn_state || conn_state->is_internal_query_active()) { return; }
  // Preserve the pre-hook's decline; optimized plans may no longer contain the scans.
  if (should_use_duckdb(context, nullptr, nullptr) != decline_reason::none) { return; }
  if (!gpu_execution_enabled(context) || !ctx->is_initialized()) { return; }

  // Copy the optimized plan into THIS connection's per-connection state,
  // stamped with the current planning generation. OnFinalizePrepare will
  // attempt create_plan() on this copy — that's the single source of truth for
  // GPU support. If the plan contains unsupported operators, create_plan()
  // throws and we fall back to CPU. A capture whose planning attempt never
  // reaches finalize (e.g. Connection::ExtractPlan) is structurally rejected
  // at the next attempt by the generation check.
  //
  // Plan-copy failures make the query ineligible for GPU execution. Optimizer
  // hooks must not throw, so log a readable message and decline the plan.
  try {
    auto const* captured = plan.get();
    if (plan->type == duckdb::LogicalOperatorType::LOGICAL_EXECUTE ||
        plan->type == duckdb::LogicalOperatorType::LOGICAL_PREPARE) {
      if (plan->children.size() != 1) { return; }
      captured = plan->children[0].get();
    }
    conn_state->set_captured_original_views(
      sirius::op::scan::capture_bound_read_views(*captured, context));
    conn_state->set_captured_plan(copy_logical_plan(*captured, context));
  } catch (duckdb::NotImplementedException& e) {
    // Plan not serializable — skip GPU. Logged because a silent skip here is
    // indistinguishable from "GPU ran and was slow": the query still returns correct
    // CPU results, so an unserializable table function looks like a perf mystery.
    SIRIUS_LOG_DEBUG("Transparent execution: plan not serializable, skipping GPU: {}", e.what());
  } catch (std::exception& e) {
    SIRIUS_LOG_DEBUG("Transparent execution: failed to copy logical plan: {}",
                     sirius::sanitized_message(e));
  }
}

}  // namespace sirius::transparent
