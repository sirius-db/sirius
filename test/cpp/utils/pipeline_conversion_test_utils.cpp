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
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_context.hpp"

#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/settings.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

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

//! Parse + plan + optimize + resolve a SQL query, mirroring the sirius-specific order of
//! `SiriusTableFunctionData::ExtractPlan`: `ResolveOperatorTypes` BEFORE `ColumnBindingResolver`.
//! DuckDB's `Connection::ExtractPlan` uses the reverse order, which trips sirius plan
//! generation with an "inequal types" binder error on some queries.
duckdb::unique_ptr<duckdb::LogicalOperator> extract_logical_plan_sirius_order(
  duckdb::ClientContext& context, const std::string& query)
{
  duckdb::Parser parser(context.GetParserOptions());
  parser.ParseQuery(query);

  duckdb::Planner planner(context);
  planner.CreatePlan(std::move(parser.statements[0]));
  D_ASSERT(planner.plan);

  duckdb::unique_ptr<duckdb::LogicalOperator> plan = std::move(planner.plan);
  if (context.config.enable_optimizer) {
    duckdb::Optimizer optimizer(*planner.binder, context);
    plan = optimizer.Optimize(std::move(plan));
  }
  plan->ResolveOperatorTypes();
  duckdb::ColumnBindingResolver resolver;
  duckdb::ColumnBindingResolver::Verify(*plan);
  resolver.VisitOperator(*plan);
  return plan;
}

}  // namespace

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
      logical_plan = extract_logical_plan_sirius_order(context, query);
    }

    sirius::planner::sirius_physical_plan_generator physical_planner(context);
    auto sirius_plan = physical_planner.create_plan(std::move(logical_plan));

    auto sirius_ctx_ptr = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    if (!sirius_ctx_ptr) {
      throw std::runtime_error(
        "[convert_query_to_dump] SiriusContext not registered on the connection");
    }
    const auto& op_params = sirius_ctx_ptr->get_config().get_operator_params();

    // Null telemetry context: no engine, per the pipeline_build_context ctor contract.
    pipeline::pipeline_build_context build_ctx(
      /*telemetry_context=*/nullptr,
      duckdb::Settings::Get<duckdb::PreserveInsertionOrderSetting>(context),
      static_cast<int>(sirius_ctx_ptr->get_config().get_hw_topology().gpus.size()));

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
