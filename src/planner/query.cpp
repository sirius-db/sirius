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

#include "planner/query.hpp"

#include "sirius_engine.hpp"
#include "sirius_interface.hpp"
#include "telemetry/telemetry_context.hpp"

namespace sirius::planner {

query::query(duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> pipelines,
             const quent::Context& context,
             telemetry::query_telemetry_info telemetry_info)
  : _plan_id(uuid::now_v7()), _pipelines(std::move(pipelines))
{
  build_indices();
  telemetry::emit_plan_telemetry(context, _pipelines, _plan_id, telemetry_info);
}

void query::build_indices()
{
  for (const auto& pipeline : _pipelines) {
    for (auto& op : pipeline->get_operators()) {
      op.get().set_pipeline(pipeline);
    }
    // Get the source operator (first operator in the pipeline)
    auto source = pipeline->get_source();
    if (source) {
      // Add to operator-to-pipeline map
      _operator_to_pipeline[source.get()] = pipeline;

      // If it's a scan-like source, add to scan operators vector. GPU_VALUES
      // must be included: task_scheduler::start_query() schedules the first
      // scan operator, which is the only kickoff a VALUES-only plan gets.
      if (source->type == op::SiriusPhysicalOperatorType::GPU_SCAN ||
          source->type == op::SiriusPhysicalOperatorType::GPU_VALUES) {
        _scan_operators.push_back(source.get());
      }
    }

    // Also add sink and intermediate operators to the map
    auto sink = pipeline->get_sink();
    if (sink) { _operator_to_pipeline[sink.get()] = pipeline; }

    for (auto& op_ref : pipeline->get_operators()) {
      _operator_to_pipeline[&op_ref.get()] = pipeline;
    }
  }
}

const duckdb::vector<op::sirius_physical_operator*>& query::get_scan_operators() const
{
  return _scan_operators;
}

duckdb::shared_ptr<pipeline::sirius_pipeline> query::get_pipeline(op::sirius_physical_operator* op)
{
  auto it = _operator_to_pipeline.find(op);
  if (it != _operator_to_pipeline.end()) { return it->second; }
  return nullptr;
}

const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& query::get_pipelines() const
{
  return _pipelines;
}

}  // namespace sirius::planner
