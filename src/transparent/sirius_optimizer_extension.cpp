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
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <log/logging.hpp>

namespace sirius::transparent {

namespace {

bool gpu_execution_enabled(const duckdb::ClientContext& context)
{
  duckdb::Value setting;
  auto lookup_result = context.TryGetCurrentSetting("gpu_execution", setting);
  return lookup_result && !setting.IsNull() && setting.GetValue<bool>();
}

}  // namespace

namespace detail {

duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> clone_filter_pushdown_info(
  duckdb::JoinFilterPushdownInfo const& src)
{
  auto dst                   = duckdb::make_uniq<duckdb::JoinFilterPushdownInfo>();
  dst->join_condition        = src.join_condition;
  dst->build_side_has_filter = src.build_side_has_filter;
  dst->probe_info.reserve(src.probe_info.size());
  for (auto const& pi : src.probe_info) {
    duckdb::JoinFilterPushdownFilter copy_pi;
    copy_pi.dynamic_filters = pi.dynamic_filters;  // share — preserves route-key identity
    copy_pi.columns         = pi.columns;
    dst->probe_info.push_back(std::move(copy_pi));
  }
  // min_max_aggregates intentionally omitted: Sirius does not read them, and cloning
  // unique_ptr<Expression> trees requires deep-copying every node type.
  return dst;
}

void preserve_dynamic_filter_metadata(duckdb::LogicalOperator& original,
                                      duckdb::LogicalOperator& copy,
                                      preserved_counts& counts)
{
  // Parallel walk requires structural alignment. If Copy ever rearranges children, bail rather
  // than risk attaching mismatched metadata (safe failure: copy stays in its post-Copy state).
  if (original.type != copy.type || original.children.size() != copy.children.size()) { return; }

  if (original.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto& o = original.Cast<duckdb::LogicalGet>();
    auto& c = copy.Cast<duckdb::LogicalGet>();
    if (o.dynamic_filters) {
      c.dynamic_filters = o.dynamic_filters;  // share the same DynamicTableFilterSet
      ++counts.gets;
    }
  } else if (original.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    auto& o = original.Cast<duckdb::LogicalComparisonJoin>();
    auto& c = copy.Cast<duckdb::LogicalComparisonJoin>();
    if (o.filter_pushdown) {
      c.filter_pushdown = clone_filter_pushdown_info(*o.filter_pushdown);
      ++counts.joins;
    }
  }

  for (std::size_t i = 0; i < original.children.size(); ++i) {
    preserve_dynamic_filter_metadata(*original.children[i], *copy.children[i], counts);
  }
}

}  // namespace detail

duckdb::unique_ptr<duckdb::LogicalOperator> copy_logical_plan(duckdb::LogicalOperator& plan,
                                                              duckdb::ClientContext& context,
                                                              detail::preserved_counts* counts)
{
  auto copy = plan.Copy(context);
  detail::preserved_counts local_counts;
  detail::preserved_counts& target_counts = counts ? *counts : local_counts;
  detail::preserve_dynamic_filter_metadata(plan, *copy, target_counts);
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
    detail::preserved_counts counts;
    auto plan_copy = copy_logical_plan(*plan, context, &counts);
    if (counts.joins > 0 || counts.gets > 0) {
      SIRIUS_LOG_DEBUG(
        "Transparent execution: preserved dynamic-filter metadata on {} join(s), {} get(s)",
        counts.joins,
        counts.gets);
    }
    ctx->set_captured_logical_plan(std::move(plan_copy));
  } catch (duckdb::NotImplementedException&) {
    // Plan not serializable — skip GPU.
  } catch (std::exception& e) {
    SIRIUS_LOG_DEBUG("Transparent execution: failed to copy logical plan: {}", e.what());
  }
}

}  // namespace sirius::transparent
