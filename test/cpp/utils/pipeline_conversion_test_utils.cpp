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

#include "utils/pipeline_conversion_test_utils.hpp"

#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"
#include "sirius_engine.hpp"
#include "sirius_interface.hpp"

#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/main/settings.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace sirius::test {

namespace {

//! RAII: applies the optimizer disables of `SiriusTableFunctionData::PrepareConnection`,
//! restoring the prior settings on destruction (including on exceptions).
class optimizer_disable_guard {
 public:
  explicit optimizer_disable_guard(duckdb::ClientContext& context)
    : context_(context),
      original_config_(context.config),
      original_disabled_optimizers_(
        duckdb::DBConfig::GetConfig(context).options.disabled_optimizers)
  {
    auto& dbconfig = duckdb::DBConfig::GetConfig(context_);
    auto disabled  = dbconfig.options.disabled_optimizers;
    disabled.insert(duckdb::OptimizerType::IN_CLAUSE);
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
#ifdef DEBUG
    disabled.insert(duckdb::OptimizerType::COLUMN_LIFETIME);
#endif
    dbconfig.options.disabled_optimizers = disabled;
  }

  ~optimizer_disable_guard()
  {
    duckdb::DBConfig::GetConfig(context_).options.disabled_optimizers =
      original_disabled_optimizers_;
    context_.config = original_config_;
  }

  optimizer_disable_guard(const optimizer_disable_guard&)            = delete;
  optimizer_disable_guard& operator=(const optimizer_disable_guard&) = delete;

 private:
  duckdb::ClientContext& context_;
  duckdb::ClientConfig original_config_;
  std::set<duckdb::OptimizerType> original_disabled_optimizers_;
};

struct extracted_plan {
  duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan;
  duckdb::shared_ptr<duckdb::PreparedStatementData> prepared;
};

//! Parse + plan + optimize + resolve a SQL query, mirroring the sirius-specific order of
//! `SiriusTableFunctionData::ExtractPlan`: `ResolveOperatorTypes` BEFORE `ColumnBindingResolver`.
//! DuckDB's `Connection::ExtractPlan` uses the reverse order, which trips sirius plan
//! generation with an "inequal types" binder error on some queries.
extracted_plan extract_logical_plan_sirius_order(duckdb::ClientContext& context,
                                                 const std::string& query)
{
  duckdb::Parser parser(context.GetParserOptions());
  parser.ParseQuery(query);
  auto statement_type = parser.statements[0]->type;

  duckdb::Planner planner(context);
  planner.CreatePlan(std::move(parser.statements[0]));
  D_ASSERT(planner.plan);

  auto prepared       = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(statement_type);
  prepared->names     = planner.names;
  prepared->types     = planner.types;
  prepared->value_map = std::move(planner.value_map);

  duckdb::unique_ptr<duckdb::LogicalOperator> plan = std::move(planner.plan);
  if (context.config.enable_optimizer) {
    duckdb::Optimizer optimizer(*planner.binder, context);
    plan = optimizer.Optimize(std::move(plan));
  }
  plan->ResolveOperatorTypes();
  duckdb::ColumnBindingResolver resolver;
  duckdb::ColumnBindingResolver::Verify(*plan);
  resolver.VisitOperator(*plan);
  return {std::move(plan), std::move(prepared)};
}

//! Starts far above any window id a test process will reach, so a synthetic query can never
//! collide with a genuine execution window's registration (the registry rejects duplicates).
sirius::query_id_t next_test_query_id()
{
  static std::atomic<std::uint32_t> counter{1'000'000};
  return sirius::make_query_id(counter.fetch_add(1, std::memory_order_relaxed));
}

}  // namespace

scoped_test_query::scoped_test_query(duckdb::ClientContext& context)
  : ctx_(context.registered_state->Get<duckdb::SiriusContext>("sirius_state")),
    query_id_(next_test_query_id())
{
  if (usable()) { ctx_->get_data_repository_registry().create_for_query(query_id_); }
}

scoped_test_query::~scoped_test_query()
{
  if (!usable()) { return; }
  try {
    ctx_->get_data_repository_registry().erase(query_id_);
  } catch (...) {  // best-effort: never throw out of a test-scaffolding destructor
  }
}

bool scoped_test_query::usable() const noexcept { return ctx_ && ctx_->is_initialized(); }

void with_conversion_result(
  duckdb::Connection& con,
  const std::string& query,
  const std::function<void(pipeline::pipeline_conversion_result&)>& consume)
{
  auto& context = *con.context;

  // The optimizer's catalog reads and the GPU-native seq_scan ingestible construction require
  // an active transaction (production inherits one from the bind callsite). Hold it across
  // plan generation AND conversion — the ingestible is built eagerly — then roll back, since
  // the path is read-only.
  con.BeginTransaction();
  try {
    duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan;
    {
      optimizer_disable_guard guard(context);
      logical_plan = extract_logical_plan_sirius_order(context, query).logical_plan;
    }

    sirius::planner::sirius_physical_plan_generator physical_planner(context);
    auto sirius_plan = physical_planner.create_plan(std::move(logical_plan));

    // Apply fusion before conversion; this path has no RESULT_COLLECTOR wrapper.
    sirius::planner::sirius_physical_plan_generator::mark_fusable_merge_pipelines(context,
                                                                                  *sirius_plan);

    auto sirius_ctx_ptr = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    if (!sirius_ctx_ptr) {
      throw std::runtime_error(
        "[convert_query_to_dump] SiriusContext not registered on the connection");
    }
    const auto& op_params = sirius_ctx_ptr->get_config().get_operator_params();

    std::vector<int> active_gpu_ids;
    for (auto const* space : sirius_ctx_ptr->get_memory_manager().get_memory_spaces_for_tier(
           cucascade::memory::Tier::GPU)) {
      if (space != nullptr) { active_gpu_ids.push_back(space->get_device_id()); }
    }
    std::sort(active_gpu_ids.begin(), active_gpu_ids.end());
    active_gpu_ids.erase(std::unique(active_gpu_ids.begin(), active_gpu_ids.end()),
                         active_gpu_ids.end());

    // Null telemetry context: no engine, but derive planning width from the same configured GPU
    // spaces as production so multi-visible-GPU hosts do not inflate single-GPU test plans.
    pipeline::pipeline_build_context build_ctx(
      /*telemetry_context=*/nullptr,
      duckdb::Settings::Get<duckdb::PreserveInsertionOrderSetting>(context),
      std::move(active_gpu_ids));

    pipeline::sirius_pipeline_build_state state;
    auto root_pipeline =
      duckdb::make_shared_ptr<pipeline::sirius_meta_pipeline>(build_ctx, state, nullptr);
    root_pipeline->build(*sirius_plan);
    root_pipeline->ready();

    pipeline::sirius_pipeline_converter converter(build_ctx, op_params);
    auto result = converter.convert(*root_pipeline);

    // Consume *here*, while the plan tree and pipelines are in scope: the result's pipelines
    // reference operators owned by the plan tree, so a result that escaped to the caller
    // would read dangling pointers.
    consume(result);
    con.Rollback();
  } catch (...) {
    con.Rollback();
    throw;
  }
}

void with_initialized_engine(duckdb::Connection& con,
                             const std::string& query,
                             const std::function<void(sirius_engine&)>& consume)
{
  auto& context = *con.context;

  // Stands in for the execution window this helper never opens: registers this plan's
  // repository manager and drops it (with its repositories) on the way out.
  scoped_test_query test_query(context);

  con.BeginTransaction();
  try {
    extracted_plan extracted;
    {
      optimizer_disable_guard guard(context);
      extracted = extract_logical_plan_sirius_order(context, query);
    }

    sirius::planner::sirius_physical_plan_generator physical_planner(context);
    auto sirius_plan = physical_planner.create_plan(std::move(extracted.logical_plan));
    auto prepared    = duckdb::make_shared_ptr<sirius_prepared_statement_data>(
      std::move(extracted.prepared), std::move(sirius_plan));
    auto collector =
      duckdb::make_uniq_base<op::sirius_physical_result_collector,
                             op::sirius_physical_materialized_collector>(*prepared, context);

    sirius_interface iface(context);
    sirius_engine engine(context, iface, test_query.query_id());
    engine.initialize(std::move(collector));
    consume(engine);

    con.Rollback();
  } catch (...) {
    con.Rollback();
    throw;
  }
}

void with_initialized_streaming_fragment(
  duckdb::Connection& con,
  const std::string& query,
  std::vector<std::shared_ptr<cucascade::shared_data_repository>> output_repos,
  std::optional<op::partition_spec> spec,
  const std::function<void(sirius_engine&, op::sirius_physical_streaming_sink&)>& consume)
{
  auto& context = *con.context;

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw sirius::invalid_input_exception(
      "with_initialized_streaming_fragment: Sirius is not registered on this connection");
  }

  // Without a partition spec the plan is rooted in a single-destination sink: no repository would
  // index an empty vector below, and extra ones would be silently dropped rather than wired to
  // anything. With a spec, every repository becomes a destination.
  if (output_repos.empty() || (!spec.has_value() && output_repos.size() != 1)) {
    throw sirius::invalid_input_exception(
      "with_initialized_streaming_fragment: expected " +
      std::string(spec.has_value() ? "at least 1" : "exactly 1") + " output repository, got " +
      std::to_string(output_repos.size()));
  }

  con.BeginTransaction();
  try {
    // The real execution window, not scoped_test_query: consume() may execute(), and only a
    // window begin points the task creator at this connection. Sink output repositories are
    // never registered with the window's repository manager, so they survive finish().
    duckdb::SiriusContext::StandaloneQueryScope window(
      *sirius_ctx, context, "streaming_fragment_test");

    extracted_plan extracted;
    {
      optimizer_disable_guard guard(context);
      extracted = extract_logical_plan_sirius_order(context, query);
    }

    sirius::planner::sirius_physical_plan_generator physical_planner(context);
    auto sirius_plan = physical_planner.create_plan(std::move(extracted.logical_plan));

    // A STREAMING_SINK is a normal unary operator, unlike the RESULT_COLLECTOR, which keeps its
    // child outside `children[]` and needs special descent in the plan generator. Attaching the
    // subtree as children[0] is what keeps that special-casing unnecessary.
    auto types       = sirius_plan->types;
    auto cardinality = sirius_plan->estimated_cardinality;
    duckdb::unique_ptr<op::sirius_physical_streaming_sink> sink;
    if (spec.has_value()) {
      sink = duckdb::make_uniq<op::sirius_physical_streaming_sink>(
        std::move(types), cardinality, std::move(output_repos), std::move(*spec));
    } else {
      sink = duckdb::make_uniq<op::sirius_physical_streaming_sink>(
        std::move(types), cardinality, output_repos[0]);
    }
    sink->children.push_back(std::move(sirius_plan));

    sirius_interface iface(context);
    sirius_engine engine(context, iface, window.query_id());
    // The fragment owns the plan; the engine borrows it. initialize() would take ownership and
    // destroy the sink with the engine, leaving nothing to pull from afterwards.
    engine.initialize_internal(*sink);
    consume(engine, *sink);
    window.finish();

    con.Rollback();
  } catch (...) {
    con.Rollback();
    throw;
  }
}

std::string convert_query_to_dump(duckdb::Connection& con, const std::string& query)
{
  std::string dump;
  with_conversion_result(con, query, [&](pipeline::pipeline_conversion_result& result) {
    dump = pipeline::dump_pipeline_conversion_result(result);
  });
  return dump;
}

std::string convert_query_to_raw_schedule(duckdb::Connection& con, const std::string& query)
{
  std::string dump;
  with_conversion_result(con, query, [&](pipeline::pipeline_conversion_result& result) {
    dump = pipeline::dump_pipeline_schedule_raw(result);
  });
  return dump;
}

std::filesystem::path tpch_queries_dir()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test/tpch_performance/tpch_queries/orig";
#else
  return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
         "test/tpch_performance/tpch_queries/orig";
#endif
}

std::string read_tpch_query_file(int q)
{
  auto path = tpch_queries_dir() / ("q" + std::to_string(q) + ".sql");
  std::ifstream in(path);
  if (!in.good()) {
    throw std::runtime_error("[read_tpch_query_file] failed to open " + path.string());
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace sirius::test
