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

#include "planner/duckdb_join_filter_candidate_adapter.hpp"
#include "sirius_context.hpp"

#include <duckdb/common/enums/optimizer_type.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/config.hpp>
#include <log/logging.hpp>

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
  auto copy = plan.Copy(context);
  planner::duckdb_join_filter_candidate_adapter::preserve_dynamic_filter_metadata(plan, *copy);
  return copy;
}

void sirius_pre_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                               duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  if (!gpu_execution_enabled(input.context)) { return; }

  auto& context = input.context;
  auto ctx      = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!ctx || !ctx->is_initialized() || ctx->is_internal_query_active()) { return; }

  auto disabled = duckdb::DBConfig::GetConfig(context).options.disabled_optimizers;
  ctx->set_transparent_original_disabled_optimizers(disabled);

  // Transparent execution disables optimizers that introduce DuckDB-internal
  // plan shapes or source operators the rebind path cannot yet execute.
  disabled.insert(duckdb::OptimizerType::IN_CLAUSE);
  disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
  // Keep STATISTICS_PROPAGATION enabled. Its constant-folded
  // EXPRESSION_GET/COLUMN_DATA_SCAN and DUMMY_SCAN sources are handled by
  // GPU_VALUES. If the user disabled it explicitly, it remains in `disabled`.
  // LATE_MATERIALIZATION rewrites `ORDER BY ... LIMIT N` over a scan into a
  // self-RIGHT_SEMI_JOIN keyed on the parquet virtual columns `file_index` /
  // `file_row_number` (TOP_N picks the N rows, the semi-join re-fetches them
  // by row id). Sirius's parquet scan path drops virtual columns silently
  // (src/op/scan/scan_plan.cpp:194), so the join's key_col_indices reference
  // columns that don't exist at runtime. Until the scan path threads virtual
  // columns through (or sirius_plan_get falls back on them), disable this
  // pass so the small-sort / ORDER-BY-LIMIT plans stay on the standard
  // sort path. See PR #732 comment 3242605041.
  disabled.insert(duckdb::OptimizerType::LATE_MATERIALIZATION);

  duckdb::DBConfig::GetConfig(context).options.disabled_optimizers = std::move(disabled);
}

void sirius_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                           duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  if (!gpu_execution_enabled(input.context)) { return; }

  auto& context = input.context;

  auto ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!ctx || !ctx->is_initialized() || ctx->is_internal_query_active()) { return; }

  // Restore the original connection setting so transparent execution does not
  // leak optimizer changes into later CPU queries.
  ctx->restore_transparent_disabled_optimizers(context);

  // Copy the optimized plan. OnFinalizePrepare will attempt create_plan() on this
  // copy — that's the single source of truth for GPU support. If the plan contains
  // unsupported operators, create_plan() throws and we fall back to CPU.
  try {
    ctx->set_captured_logical_plan(copy_logical_plan(*plan, context));
  } catch (duckdb::NotImplementedException& e) {
    // Plan not serializable — skip GPU. Logged because a silent skip here is
    // indistinguishable from "GPU ran and was slow": the query still returns correct
    // CPU results, so an unserializable table function looks like a perf mystery.
    SIRIUS_LOG_DEBUG("Transparent execution: plan not serializable, skipping GPU: {}", e.what());
  } catch (std::exception& e) {
    SIRIUS_LOG_DEBUG("Transparent execution: failed to copy logical plan: {}", e.what());
  }
}

}  // namespace sirius::transparent
