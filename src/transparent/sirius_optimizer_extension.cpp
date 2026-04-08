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
#include "sirius_context.hpp"

#include <duckdb/common/enums/optimizer_type.hpp>
#include <duckdb/main/config.hpp>
#include <spdlog/spdlog.h>

namespace sirius::transparent {

void sirius_pre_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                               duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  if (!duckdb::Config::ENABLE_GPU_EXECUTION) { return; }

  auto& context = input.context;
  auto ctx      = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!ctx || !ctx->is_initialized()) { return; }

  // Disable optimizers that produce DuckDB-internal functions Sirius can't handle.
  // The post-hook re-enables them so non-GPU queries aren't affected.
  auto& disabled = duckdb::DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(duckdb::OptimizerType::IN_CLAUSE);
  disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
}

void sirius_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                           duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  if (!duckdb::Config::ENABLE_GPU_EXECUTION) { return; }

  auto& context = input.context;

  auto ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!ctx || !ctx->is_initialized()) { return; }

  // Re-enable the disabled optimizers so they don't leak to non-GPU queries.
  auto& disabled = duckdb::DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.erase(duckdb::OptimizerType::IN_CLAUSE);
  disabled.erase(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);

  // Copy the optimized plan. OnFinalizePrepare will attempt create_plan() on this
  // copy — that's the single source of truth for GPU support. If the plan contains
  // unsupported operators, create_plan() throws and we fall back to CPU.
  try {
    auto plan_copy = plan->Copy(context);
    ctx->set_captured_logical_plan(std::move(plan_copy));
  } catch (duckdb::NotImplementedException&) {
    // Plan not serializable — skip GPU.
  } catch (std::exception& e) {
    spdlog::debug("Transparent execution: failed to copy logical plan: {}", e.what());
  }
}

}  // namespace sirius::transparent
