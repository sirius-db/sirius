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

#include "op/sirius_physical_cpu_source.hpp"

#include "config.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"

namespace sirius::op {

sirius_physical_cpu_source::sirius_physical_cpu_source(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection> collection)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::CPU_SOURCE, std::move(types), estimated_cardinality),
    collection(std::move(collection)),
    produce_single_row(false)
{
}

sirius_physical_cpu_source::sirius_physical_cpu_source(duckdb::vector<sirius::logical_type> types,
                                                       duckdb::idx_t estimated_cardinality,
                                                       bool produce_single_row)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::CPU_SOURCE, std::move(types), estimated_cardinality),
    collection(nullptr),
    produce_single_row(produce_single_row)
{
}

void sirius_physical_cpu_source::build_pipelines(pipeline::sirius_pipeline& current,
                                                 pipeline::sirius_meta_pipeline& meta_pipeline)
{
  if (!duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {
    sirius_physical_operator::build_pipelines(current, meta_pipeline);
    return;
  }
  // CPU_SOURCE is the sink of its own child_meta. Under the new protocol,
  // create_child_meta_pipeline pre-populates [*this] in the new child_meta's
  // operators[] (via `is_ready`), so post-reverse operators=[*this] with
  // source=sink=*this. CPU_SOURCE has no children, so no recursion.
  D_ASSERT(children.empty());
  meta_pipeline.create_child_meta_pipeline(current, *this);
}

}  // namespace sirius::op
