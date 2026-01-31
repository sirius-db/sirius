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

#include "op/sirius_physical_order.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"
#include "operator/gpu_materialize.hpp"

namespace sirius {
namespace op {

sirius_physical_order::sirius_physical_order(duckdb::vector<duckdb::LogicalType> types,
                                             duckdb::vector<duckdb::BoundOrderByNode> orders,
                                             duckdb::vector<duckdb::idx_t> projections_p,
                                             duckdb::idx_t estimated_cardinality,
                                             bool is_index_sort_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::ORDER_BY, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    projections(std::move(projections_p)),
    is_index_sort(is_index_sort_p)
{
  // sort_result = duckdb::make_shared_ptr<GPUIntermediateRelation>(projections.size());
  // for (int col = 0; col < projections.size(); col++) {
  //   sort_result->columns[col] = nullptr;
  // }
}

}  // namespace op
}  // namespace sirius
