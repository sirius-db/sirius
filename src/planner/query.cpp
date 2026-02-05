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

namespace sirius::planner {

query::query(sirius_pipeline_hashmap pipeline_hashmap)
  : _pipeline_hashmap(std::move(pipeline_hashmap))
{
  build_indices();
}

void query::build_indices()
{
  for (auto& pipeline : _pipeline_hashmap._vec) {
    for (auto& op : pipeline->get_operators()) {
      op.get().set_pipeline(pipeline);
    }
    // Get the source operator (first operator in the pipeline)
    auto source = pipeline->get_source();
    if (source) {
      // Add to operator-to-pipeline map
      _operator_to_pipeline[source.get()] = pipeline;

      // If it's a table scan, add to scan operators vector
      if (source->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
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
  return _pipeline_hashmap._vec;
}

sirius_pipeline_hashmap& query::get_pipeline_hashmap() { return _pipeline_hashmap; }

}  // namespace sirius::planner
