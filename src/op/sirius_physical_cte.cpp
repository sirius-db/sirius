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

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/operator/set/physical_cte.hpp"
#include "duckdb/parallel/event.hpp"
// #include "duckdb/parallel/meta_pipeline.hpp"
// #include "duckdb/parallel/pipeline.hpp"

#include "log/logging.hpp"
#include "op/sirius_physical_cte.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

namespace sirius {
namespace op {

sirius_physical_cte::sirius_physical_cte(std::string ctename,
                                         duckdb::idx_t table_index,
                                         duckdb::vector<duckdb::LogicalType> types,
                                         duckdb::unique_ptr<sirius_physical_operator> top,
                                         duckdb::unique_ptr<sirius_physical_operator> bottom,
                                         duckdb::idx_t estimated_cardinality)
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
  op_state.reset();
  sink_state.reset();

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
  return std::make_unique<operator_data>(input_data);
}

}  // namespace op
}  // namespace sirius
