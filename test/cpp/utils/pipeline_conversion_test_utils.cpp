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

namespace sirius::test {

namespace {

//! RAII shim that mirrors `SiriusTableFunctionData::PrepareConnection` /
//! `CleanupConnection` (src/sirius_extension.cpp:142-162). Disables the
//! optimizers Sirius normally disables when running `gpu_execution`, then
//! restores the prior settings on destruction (and on exceptions).
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
    disabled.insert(duckdb::OptimizerType::STATISTICS_PROPAGATION);
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

//! Parse + plan + optimize + resolve a SQL query into a `LogicalOperator`, mirroring
//! `SiriusTableFunctionData::ExtractPlan` (src/sirius_extension.cpp:164-197) including the
//! sirius-specific order (`ResolveOperatorTypes` BEFORE `ColumnBindingResolver`).
//!
//! `Connection::ExtractPlan` is similar but uses the DuckDB-canonical order (resolver
//! first, then ResolveOperatorTypes), which leaves the plan in a state that sirius's
//! `sirius_physical_plan_generator::create_plan` re-resolves and trips on for some queries
//! ("inequal types" binder error). Reproducing the sirius order here keeps the path
//! byte-for-byte identical to production's GPUExecutionBind flow.
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

std::string convert_query_to_dump(duckdb::Connection& con, const std::string& query)
{
  auto& context = *con.context;

  // DuckDB's Optimizer reads catalog state, which requires an active transaction. The
  // production path inherits one from the TableFunction bind callsite; tests must open one
  // explicitly. Rollback because the conversion path is read-only.
  con.BeginTransaction();
  duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan;
  try {
    optimizer_disable_guard guard(context);
    logical_plan = extract_logical_plan_sirius_order(context, query);
  } catch (...) {
    con.Rollback();
    throw;
  }
  con.Rollback();

  sirius::planner::sirius_physical_plan_generator physical_planner(context);
  auto sirius_plan = physical_planner.create_plan(std::move(logical_plan));

  auto sirius_ctx_ptr = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx_ptr) {
    throw std::runtime_error(
      "[convert_query_to_dump] SiriusContext not registered on the connection");
  }
  const auto& op_params = sirius_ctx_ptr->get_config().get_operator_params();

  pipeline::pipeline_build_context build_ctx;
  build_ctx.preserve_insertion_order =
    duckdb::Settings::Get<duckdb::PreserveInsertionOrderSetting>(context);
  build_ctx.num_gpus = static_cast<int>(sirius_ctx_ptr->get_config().get_hw_topology().gpus.size());

  pipeline::sirius_pipeline_build_state state;
  auto root_pipeline =
    duckdb::make_shared_ptr<pipeline::sirius_meta_pipeline>(build_ctx, state, nullptr);
  root_pipeline->build(*sirius_plan);
  root_pipeline->ready();

  // Iceberg metadata cache: under flag ON the plan generator owns its own cache and the
  // converter's pointer is ignored on the tree-based path. Under flag OFF the converter
  // would read this cache to construct iceberg scans — TPC-H has no iceberg, so passing
  // an empty (but non-null) map keeps the legacy lookup site happy.
  static const std::unordered_map<std::string, std::shared_ptr<const op::scan::IcebergDeleteData>>
    kEmptyIcebergCache;
  pipeline::sirius_pipeline_converter converter(build_ctx, op_params, &kEmptyIcebergCache);
  auto result = converter.convert(*root_pipeline);

  // Dump *here* while `sirius_plan`, `root_pipeline`, `result` are all in scope. The result's
  // pipelines reference operators in the plan tree; if we returned the result and dumped at
  // the caller, the plan tree would already be destroyed and the dump would read dangling
  // pointers. Legacy (flag OFF) partially survives that hazard because its converter-inserted
  // operators (PARTITION, CONCAT, MERGE_*) are owned by result.inserted_operators_, but the
  // tree path (flag ON) has no inserted_operators_ at all — everything dangles.
  return pipeline::dump_pipeline_conversion_result(result);
}

}  // namespace sirius::test
