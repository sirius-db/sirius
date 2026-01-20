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

#include "op/sirius_physical_result_collector.hpp"

#include "duckdb/main/config.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include "gpu_context.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "log/logging.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "utils.hpp"

namespace sirius {
namespace op {

class sirius_materialized_collector_global_state : public duckdb::GlobalSinkState {
 public:
  duckdb::mutex glock;
  duckdb::shared_ptr<duckdb::ClientContext> context;
};

sirius_physical_result_collector::sirius_physical_result_collector(
  duckdb::SiriusPreparedStatementData& data)
  : sirius_physical_operator(
      duckdb::PhysicalOperatorType::RESULT_COLLECTOR, {duckdb::LogicalType::BOOLEAN}, 0),
    statement_type(data.prepared->statement_type),
    properties(data.prepared->properties),
    plan(*data.sirius_physical_plan),
    names(data.prepared->names)
{
  this->types = data.prepared->types;
}

duckdb::vector<duckdb::const_reference<sirius_physical_operator>>
sirius_physical_result_collector::get_children() const
{
  return {plan};
}

void sirius_physical_result_collector::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // operator is a sink, build a pipeline
  sink_state.reset();

  D_ASSERT(children.empty());

  // single operator: the operator becomes the data source of the current pipeline
  auto& state = meta_pipeline.get_state();
  state.set_pipeline_source(current, *this);

  // we create a new pipeline starting from the child
  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(plan);
}

sirius_physical_materialized_collector::sirius_physical_materialized_collector(
  duckdb::SiriusPreparedStatementData& data)
  : sirius_physical_result_collector(data),
    result_collection(duckdb::make_uniq<duckdb::GPUResultCollection>())
{
}

duckdb::unique_ptr<duckdb::GlobalSinkState>
sirius_physical_materialized_collector::get_global_sink_state(duckdb::ClientContext& context) const
{
  auto state     = duckdb::make_uniq<sirius_materialized_collector_global_state>();
  state->context = context.shared_from_this();
  return std::move(state);
}

duckdb::unique_ptr<duckdb::QueryResult> sirius_physical_materialized_collector::get_result(
  duckdb::GlobalSinkState& state)
{
  auto& gstate = state.Cast<sirius_materialized_collector_global_state>();
  if (!gstate.context)
    throw duckdb::InvalidInputException(
      "No context set in sirius_materialized_collector_global_state");
  auto prop   = gstate.context->GetClientProperties();
  auto result = duckdb::make_uniq<duckdb::GPUQueryResult>(
    statement_type, properties, names, types, prop, std::move(result_collection));
  return std::move(result);
}

}  // namespace op
}  // namespace sirius
