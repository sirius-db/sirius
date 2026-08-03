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

#include "sirius_context.hpp"

#include <duckdb/common/enums/optimizer_type.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/config.hpp>
#include <log/logging.hpp>
#include <util/duckdb_error_message.hpp>

#include <exception>
#include <utility>

namespace sirius::transparent {

namespace {

bool gpu_execution_enabled(const duckdb::ClientContext& context)
{
  duckdb::Value setting;
  auto lookup_result = context.TryGetCurrentSetting("gpu_execution", setting);
  return lookup_result && !setting.IsNull() && setting.GetValue<bool>();
}

}  // namespace

duckdb::unique_ptr<duckdb::LogicalOperator> copy_logical_plan(duckdb::LogicalOperator const& plan,
                                                              duckdb::ClientContext& context)
{
  return plan.Copy(context);
}

void sirius_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                           duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  if (!gpu_execution_enabled(input.context)) { return; }

  auto& context = input.context;

  auto ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!ctx || !ctx->is_initialized()) { return; }
  auto conn_state = duckdb::get_sirius_connection_state(context);
  if (!conn_state || conn_state->is_internal_query_active()) { return; }

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
    conn_state->set_captured_plan(copy_logical_plan(*plan, context));
  } catch (std::exception& e) {
    SIRIUS_LOG_DEBUG("Transparent execution: failed to copy logical plan: {}",
                     sirius::sanitized_message(e));
  }
}

}  // namespace sirius::transparent
