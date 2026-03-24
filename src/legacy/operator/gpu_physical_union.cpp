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

#include "operator/gpu_physical_union.hpp"

#include "gpu_meta_pipeline.hpp"
#include "gpu_pipeline.hpp"

namespace duckdb {

GPUPhysicalUnion::GPUPhysicalUnion(vector<LogicalType> types,
                                   unique_ptr<GPUPhysicalOperator> top,
                                   unique_ptr<GPUPhysicalOperator> bottom,
                                   idx_t estimated_cardinality)
  : GPUPhysicalOperator(PhysicalOperatorType::UNION, std::move(types), estimated_cardinality)
{
  children.push_back(std::move(top));
  children.push_back(std::move(bottom));
}

GPUPhysicalUnion::~GPUPhysicalUnion() {}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void GPUPhysicalUnion::BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline)
{
  op_state.reset();
  sink_state.reset();

  // Check if order matters based on sink properties
  auto sink          = meta_pipeline.GetSink();
  bool order_matters = false;
  if (current.IsOrderDependent()) { order_matters = true; }
  if (sink) {
    if (sink->SinkOrderDependent()) { order_matters = true; }
    if (!sink->ParallelSink()) { order_matters = true; }
  }

  // Create a union pipeline that shares the same sink as 'current'
  auto& union_pipeline = meta_pipeline.CreateUnionPipeline(current, order_matters);

  // Build the first child into the current pipeline
  children[0]->BuildPipelines(current, meta_pipeline);

  if (order_matters) {
    // Union pipeline comes after all pipelines created by building current
    meta_pipeline.AddDependenciesFrom(union_pipeline, union_pipeline, false);
  }

  // Build the second child into the union pipeline
  children[1]->BuildPipelines(union_pipeline, meta_pipeline);

  // Assign proper batch index to the union pipeline
  meta_pipeline.AssignNextBatchIndex(union_pipeline);
}

vector<const_reference<GPUPhysicalOperator>> GPUPhysicalUnion::GetSources() const
{
  vector<const_reference<GPUPhysicalOperator>> result;
  for (auto& child : children) {
    auto child_sources = child->GetSources();
    result.insert(result.end(), child_sources.begin(), child_sources.end());
  }
  return result;
}

}  // namespace duckdb
