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
#include "planner/delim_cte_proof.hpp"
#include "planner/delim_cte_rewrites.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_delim_aggregate_reuse.hpp"
#include "planner/sirius_plan_partial_aggregate_pushdown.hpp"
#include "sirius_context.hpp"
#include "transparent/connection_provenance.hpp"
#include "transparent/parquet_footer_statistics.hpp"
#include "transparent/push_narrowing_expressions.hpp"

#include <duckdb/common/enums/statement_type.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/planner/binder.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/filter/expression_filter.hpp>
#include <duckdb/planner/filter/table_filter_functions.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_top_n.hpp>
#include <log/logging.hpp>
#include <util/duckdb_error_message.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::transparent {

namespace {

bool gpu_execution_enabled(const duckdb::ClientContext& context)
{
  duckdb::Value setting;
  auto lookup_result = context.TryGetCurrentSetting("gpu_execution", setting);
  return lookup_result && !setting.IsNull() && setting.GetValue<bool>();
}

bool parquet_footer_statistics_enabled(const duckdb::ClientContext& context)
{
  duckdb::Value setting;
  auto lookup_result = context.TryGetCurrentSetting("parquet_footer_statistics", setting);
  return lookup_result && !setting.IsNull() && setting.GetValue<bool>();
}

/// Whether a plan optimization is on: its `operator_params` switch as `query_operator_options`
/// resolves it, without creating the session's overrides. Those snapshot the global
/// configuration, which would stop the session from seeing later changes to it.
bool plan_optimization_enabled(duckdb::ClientContext& context,
                               bool sirius::operator_params::* option)
{
  auto const state = duckdb::get_sirius_connection_state(context);
  if (state && state->execution_options) { return (*state->execution_options).*option; }
  if (state && state->operator_overrides) { return (*state->operator_overrides).*option; }
  auto const runtime = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  return runtime && runtime->get_config().get_operator_params().*option;
}

/// The smallest GPU's downgrade threshold: the bytes a GPU holds before Sirius starts moving data
/// off it. 0 when no GPU capacity is configured.
std::uint64_t gpu_memory_budget(duckdb::ClientContext& context)
{
  auto const runtime = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!runtime) { return 0; }
  std::uint64_t budget = 0;
  for (auto const& space : runtime->get_config().get_memory_space_configs()) {
    auto const* gpu = std::get_if<cucascade::memory::gpu_memory_space_config>(&space);
    if (gpu == nullptr || gpu->memory_capacity == 0) { continue; }
    std::uint64_t const threshold = gpu->downgrade_trigger_threshold();
    budget                        = budget == 0 ? threshold : std::min(budget, threshold);
  }
  return budget;
}

using top_n_filter_copies = std::unordered_map<duckdb::DynamicFilterData const*,
                                               duckdb::shared_ptr<duckdb::DynamicFilterData>>;

void copy_top_n_states(duckdb::LogicalOperator const& from,
                       duckdb::LogicalOperator& to,
                       top_n_filter_copies& filters)
{
  if (from.type == duckdb::LogicalOperatorType::LOGICAL_TOP_N) {
    auto const& original = from.Cast<duckdb::LogicalTopN>().dynamic_filter;
    if (original) {
      duckdb::annotated_lock_guard<duckdb::annotated_shared_mutex> guard(original->lock);
      auto fresh = duckdb::make_shared_ptr<duckdb::DynamicFilterData>(original->comparison_type,
                                                                      original->constant);
      to.Cast<duckdb::LogicalTopN>().dynamic_filter = fresh;
      filters.emplace(original.get(), std::move(fresh));
    }
  }
  for (size_t i = 0; i < from.children.size(); ++i)
    copy_top_n_states(*from.children[i], *to.children[i], filters);
}

void reconnect_top_n_expression(duckdb::Expression const& from,
                                duckdb::Expression& to,
                                top_n_filter_copies const& filters)
{
  if (from.GetExpressionClass() != to.GetExpressionClass())
    throw duckdb::NotImplementedException("Plan copy changed the runtime filter expression");
  if (from.GetExpressionClass() == duckdb::ExpressionClass::BOUND_FUNCTION) {
    auto const& original = from.Cast<duckdb::BoundFunctionExpression>();
    auto& copied         = to.Cast<duckdb::BoundFunctionExpression>();
    if (original.Function().GetName() != copied.Function().GetName())
      throw duckdb::NotImplementedException("Plan copy changed the runtime filter function");
    // Optional filters keep their predicate in bind data, not in the argument list.
    auto optional_child = duckdb::ExpressionFilter::GetOptionalFilterChild(from);
    if (optional_child && copied.BindInfo()) {
      duckdb::Expression* target = nullptr;
      if (original.Function().GetName() == duckdb::OptionalFilterScalarFun::NAME)
        target = copied.BindInfoMutable()
                   ->Cast<duckdb::OptionalFilterFunctionData>()
                   .child_filter_expr.get();
      else if (original.Function().GetName() == duckdb::SelectivityOptionalFilterScalarFun::NAME)
        target = copied.BindInfoMutable()
                   ->Cast<duckdb::SelectivityOptionalFilterFunctionData>()
                   .child_filter_expr.get();
      if (target) reconnect_top_n_expression(*optional_child, *target, filters);
    }

    if (original.Function().GetName() == duckdb::DynamicFilterScalarFun::NAME &&
        original.BindInfo()) {
      auto const& state =
        original.BindInfo()->Cast<duckdb::DynamicFilterFunctionData>().filter_data;
      auto found = filters.find(state.get());
      if (found != filters.end()) {
        copied.BindInfoMutable() =
          duckdb::make_uniq<duckdb::DynamicFilterFunctionData>(found->second);
      }
    }
  }
  std::vector<duckdb::Expression const*> children;
  duckdb::ExpressionIterator::EnumerateChildren(
    from, [&](duckdb::Expression const& child) { children.push_back(&child); });
  size_t index = 0;
  duckdb::ExpressionIterator::EnumerateChildren(to, [&](duckdb::Expression& child) {
    if (index >= children.size())
      throw duckdb::NotImplementedException("Plan copy changed the runtime filter children");
    reconnect_top_n_expression(*children[index++], child, filters);
  });
  if (index != children.size())
    throw duckdb::NotImplementedException("Plan copy changed the runtime filter children");
}

void reconnect_top_n_filters(duckdb::LogicalOperator const& from,
                             duckdb::LogicalOperator& to,
                             top_n_filter_copies const& filters)
{
  if (from.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto& copied = to.Cast<duckdb::LogicalGet>().table_filters;
    for (auto const& entry : from.Cast<duckdb::LogicalGet>().table_filters) {
      auto& target = copied.GetFilterByColumnIndexMutable(entry.GetIndex());
      reconnect_top_n_expression(*entry.Filter().Cast<duckdb::ExpressionFilter>().expr,
                                 *target.Cast<duckdb::ExpressionFilter>().expr,
                                 filters);
    }
  }
  for (size_t i = 0; i < from.children.size(); ++i)
    reconnect_top_n_filters(*from.children[i], *to.children[i], filters);
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

// The delim CTE rewrites that have to run before DuckDB orders the joins. Only for queries headed
// to the GPU: with GPU execution off, DuckDB plans exactly what it plans without Sirius. Each
// edit is decided and built in full before it changes the plan, with moves that cannot fail, so
// the plan is valid whenever a rewrite throws. There is no rollback: edits made before the throw
// stay, including an `EXISTS` already answered when the search for the next one fails.
//
// Returns whether the plan may have changed. A rewrite that throws counts, since it can have made
// edits first.
bool rewrite_delim_ctes(duckdb::OptimizerExtensionInput& input, duckdb::LogicalOperator& plan)
{
  auto& context = input.context;
  bool changed  = false;
  // First: once the keys reach a second table, the subquery no longer has the single key join the
  // comparison of the two subqueries looks for.
  if (plan_optimization_enabled(context, &sirius::operator_params::enable_exists_from_aggregate)) {
    try {
      if (sirius::planner::answer_exists_from_aggregate(context, plan) > 0) { changed = true; }
    } catch (std::exception& e) {
      changed = true;
      SIRIUS_LOG_DEBUG("Stopped answering EXISTS from aggregate subqueries: {}",
                       sirius::sanitized_message(e));
    }
  }
  if (plan_optimization_enabled(context, &sirius::operator_params::enable_subquery_key_carry)) {
    try {
      if (sirius::planner::carry_subquery_keys_across_equalities(
            context, input.optimizer.binder, plan) > 0) {
        changed = true;
      }
    } catch (std::exception& e) {
      changed = true;
      SIRIUS_LOG_DEBUG("Subquery keys not carried across equalities: {}",
                       sirius::sanitized_message(e));
    }
  }
  return changed;
}

// What Sirius needs from the plan before DuckDB optimizes it: its decision on the query, scan
// statistics, the delim CTE rewrites, and the proofs about the delim CTEs they leave.
void prepare_for_sirius(duckdb::OptimizerExtensionInput& input, duckdb::LogicalOperator& plan)
{
  auto& context = input.context;
  auto ctx      = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!ctx) { return; }
  auto conn_state = duckdb::get_sirius_connection_state(context);
  if (!conn_state || conn_state->is_internal_query_active()) { return; }
  // Decide before optimization can remove scans, including when GPU execution is off.
  auto const& properties = input.optimizer.binder.GetStatementProperties();
  auto const declined    = should_use_duckdb(context, &plan, &properties);
  if (parquet_footer_statistics_enabled(context)) {
    auto const scans = add_parquet_footer_statistics(context, plan);
    if (scans > 0) { SIRIUS_LOG_DEBUG("Gave {} Parquet scans their footer statistics", scans); }
  }
  bool const gpu_execution = gpu_execution_enabled(context);
  // Only a statement that can reach the GPU is rewritten; anything else must get the plan DuckDB
  // makes alone. That takes a statement that returns rows, and not the second planning pass of a
  // query Sirius has already left to DuckDB. `INSERT ... SELECT` and `CREATE TABLE AS` return
  // none, and Sirius never runs them. A `COPY` returns none either, so the Parquet export Sirius
  // does run is planned without these rewrites.
  bool const headed_to_gpu = declined == decline_reason::none &&
                             !conn_state->replanning_for_cpu() && ctx->is_initialized() &&
                             properties.return_type == duckdb::StatementReturnType::QUERY_RESULT &&
                             gpu_execution;
  // After the statistics, which give Parquet scans the row counts one rewrite's size guard reads;
  // before the proofs, which describe the plan the built-in optimizers will receive. A query that
  // Sirius turns down later is planned again without the rewrites (`mark_plan_rewritten`).
  if (headed_to_gpu && rewrite_delim_ctes(input, plan)) { conn_state->mark_plan_rewritten(); }
  // DuckDB clears its delim-CTE restriction proof before this hook runs, and never sets it for
  // Parquet scans. Re-derive it before the built-in optimizers restructure the delim CTEs; the
  // optimizer hook keeps it only together with the plan of this same run.
  conn_state->stage_cte_reader_proofs(nullptr);
  if (gpu_execution) {
    try {
      conn_state->stage_cte_reader_proofs(std::make_shared<sirius::planner::delim_cte_proofs const>(
        sirius::planner::collect_delim_cte_proofs(context, plan)));
    } catch (std::exception& e) {
      SIRIUS_LOG_DEBUG("Delim CTE reader proofs unavailable: {}", sirius::sanitized_message(e));
    }
  }
  // DuckDB v2 derives the single-table restrictions implied by an OR-ed join condition itself
  // (JoinDependentFilterRule now also runs on INNER and SEMI join conditions, which is where the
  // TPC-H q19 and q7 disjunctions end up). This hook used to derive them too, for DuckDB 1.5, which
  // missed q19's. Deriving both made every pushed restriction appear twice: TPC-H q19's part scan
  // evaluated each column's OR filter twice and its FILTER carried two copies of the per-branch
  // restriction, which cost about 0.1 s at SF1000. The hook therefore only classifies the query.
}

}  // namespace

void sirius_pre_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                               duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  if (!plan) { return; }
  prepare_for_sirius(input, *plan);
  // DuckDB clears every delim CTE's filter_dependency before it calls a pre-optimize hook, because
  // it cannot vouch for what an extension does to the plan. Without it, DuckDB's CTE filter pushers
  // skip delim CTEs, in its CPU plans too. The rewrites above keep each delim CTE's two readers and
  // a join-back that keeps an outer row only through that row, so give the dependency back,
  // re-derived with DuckDB's own rule on the plan as it now is, for every query, GPU execution on
  // or off.
  try {
    sirius::planner::restore_cte_filter_dependencies(input.context, *plan);
  } catch (std::exception& e) {
    SIRIUS_LOG_DEBUG("Delim CTE filter dependencies not restored: {}",
                     sirius::sanitized_message(e));
  }
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

namespace {

// A plan copy re-binds every Parquet scan, re-parsing the footers DuckDB's metadata cache would
// hold. The re-bound data is replaced by `preserve_parquet_bindings`, so only the cost matters.
// Enables the cache for the copy and restores the connection's own setting afterwards.
class scoped_parquet_metadata_cache {
 public:
  explicit scoped_parquet_metadata_cache(duckdb::ClientContext& context) : _context(context)
  {
    auto& db_config = duckdb::DBConfig::GetConfig(context);
    duckdb::optional_ptr<duckdb::ConfigurationOption const> option;
    auto const index = db_config.TryGetSettingIndex("parquet_metadata_cache", option);
    if (!index.IsValid()) { return; }
    _index   = index.GetIndex();
    _was_set = context.config.user_settings.IsSet(_index);
    if (_was_set) {
      context.config.user_settings.TryGetSetting(db_config.user_settings, _index, _previous);
    }
    context.config.user_settings.SetUserSetting(_index, duckdb::Value::BOOLEAN(true));
    _active = true;
  }
  ~scoped_parquet_metadata_cache()
  {
    if (!_active) { return; }
    if (_was_set) {
      _context.config.user_settings.SetUserSetting(_index, _previous);
    } else {
      _context.config.user_settings.ClearSetting(_index);
    }
  }
  scoped_parquet_metadata_cache(scoped_parquet_metadata_cache const&)            = delete;
  scoped_parquet_metadata_cache& operator=(scoped_parquet_metadata_cache const&) = delete;

 private:
  duckdb::ClientContext& _context;
  duckdb::idx_t _index = 0;
  duckdb::Value _previous;
  bool _was_set = false;
  bool _active  = false;
};

}  // namespace

duckdb::unique_ptr<duckdb::LogicalOperator> copy_logical_plan(duckdb::LogicalOperator const& plan,
                                                              duckdb::ClientContext& context)
{
  auto copy = [&] {
    scoped_parquet_metadata_cache const cache{context};
    return plan.Copy(context);
  }();
  preserve_parquet_bindings(plan, *copy, context);
  // Serialization omits runtime state. Reconnect the original bindings with
  // independent, uninitialized filters for this execution.
  top_n_filter_copies filters;
  copy_top_n_states(plan, *copy, filters);
  if (!filters.empty()) reconnect_top_n_filters(plan, *copy, filters);
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
  // The second planning pass of a query Sirius has already left to DuckDB: nothing to capture.
  if (conn_state->replanning_for_cpu()) { return; }
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
    // The subquery rewrites on @p copy; returns the subqueries the EXISTS rewrite replaced. A
    // rewrite that fails is left out by starting over from a fresh copy. The delim reuse runs
    // first, so its failure starts over before the EXISTS rewrite has replaced anything.
    auto const rewrite_subqueries =
      [&](duckdb::unique_ptr<duckdb::LogicalOperator>& copy) -> std::size_t {
      if (plan_optimization_enabled(context,
                                    &sirius::operator_params::enable_delim_aggregate_reuse)) {
        try {
          sirius::planner::reuse_delim_rows_for_subquery_aggregates(
            context, input.optimizer.binder, copy);
        } catch (std::exception& e) {
          SIRIUS_LOG_WARN(
            "Transparent execution: delim aggregate reuse failed, planning without it: {}",
            sirius::sanitized_message(e));
          copy = copy_logical_plan(*captured, context);
        }
      }
      if (!plan_optimization_enabled(context,
                                     &sirius::operator_params::enable_exists_minmax_rewrite)) {
        return 0;
      }
      try {
        return sirius::planner::rewrite_correlated_inequality_exists(
          context, input.optimizer.binder, copy, gpu_memory_budget(context));
      } catch (std::exception& e) {
        SIRIUS_LOG_WARN(
          "Transparent execution: EXISTS MIN/MAX rewrite failed, planning without it: {}",
          sirius::sanitized_message(e));
        copy = copy_logical_plan(*captured, context);
        return 0;
      }
    };
    auto copy                     = copy_logical_plan(*captured, context);
    auto exists_rewrites          = rewrite_subqueries(copy);
    std::size_t pushed_aggregates = 0;
    if (plan_optimization_enabled(context,
                                  &sirius::operator_params::enable_partial_aggregate_below_join)) {
      try {
        pushed_aggregates = sirius::planner::push_partial_aggregates_below_joins(
          context, input.optimizer.binder, copy);
      } catch (std::exception& e) {
        SIRIUS_LOG_WARN(
          "Transparent execution: partial aggregate pushdown failed, planning without it: {}",
          sirius::sanitized_message(e));
        copy            = copy_logical_plan(*captured, context);
        exists_rewrites = rewrite_subqueries(copy);
      }
    }
    if (plan_optimization_enabled(context,
                                  &sirius::operator_params::enable_narrowing_expression_pushdown)) {
      push_narrowing_expressions(*copy, input.optimizer.binder);
    }
    // Recorded once the plan is final, so the counts describe the plan that is captured.
    ctx->record_exists_minmax_rewrites(exists_rewrites);
    ctx->record_partial_aggregate_pushdowns(pushed_aggregates);
    conn_state->set_captured_plan(std::move(copy));
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
