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

// #include "duckdb/parallel/meta_pipeline.hpp"
// #include "duckdb/parallel/pipeline.hpp"

#include "op/sirius_physical_cte.hpp"

#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <nvtx3/nvtx3.hpp>

namespace sirius {
namespace op {

sirius_physical_cte::sirius_physical_cte(std::string ctename,
                                         std::size_t table_index,
                                         duckdb::vector<sirius::logical_type> types,
                                         duckdb::unique_ptr<sirius_physical_operator> top,
                                         duckdb::unique_ptr<sirius_physical_operator> bottom,
                                         std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::CTE, std::move(types), estimated_cardinality),
    table_index(table_index),
    ctename(std::move(ctename))
{
  children.push_back(std::move(top));
  children.push_back(std::move(bottom));
}

sirius_physical_cte::~sirius_physical_cte() {}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_cte::build_pipelines(pipeline::sirius_pipeline& current,
                                          pipeline::sirius_meta_pipeline& meta_pipeline)
{
  D_ASSERT(children.size() == 2);

  auto& state = meta_pipeline.get_state();

  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(*children[0]);

  for (auto& cte_scan : cte_scans) {
    state.cte_dependencies.insert(duckdb::make_pair(
      cte_scan,
      duckdb::reference<pipeline::sirius_pipeline>(*child_meta_pipeline.get_base_pipeline())));
  }

  children[1]->build_pipelines(current, meta_pipeline);
}

duckdb::vector<duckdb::const_reference<sirius_physical_operator>> sirius_physical_cte::get_sources()
  const
{
  return children[1]->get_sources();
}

std::unique_ptr<operator_data> sirius_physical_cte::execute(const operator_data& input_data,
                                                            rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_cte::execute"};
  auto& input = dynamic_cast<const read_only_pipelineable_operator_data&>(input_data);
  auto ro_vec =
    const_cast<read_only_pipelineable_operator_data&>(input).release_read_only_batches();
  std::vector<std::shared_ptr<::cucascade::data_batch>> idle_batches;
  idle_batches.reserve(ro_vec.size());
  for (auto& ro : ro_vec) {
    idle_batches.push_back(::cucascade::data_batch::to_idle(std::move(ro)));
  }
  return std::make_unique<pipelineable_operator_data>(std::move(idle_batches));
}

}  // namespace op
}  // namespace sirius
