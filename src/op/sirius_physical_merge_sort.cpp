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

#include "op/sirius_physical_merge_sort.hpp"

#include "sirius/exception.hpp"

namespace sirius {
namespace op {

sirius_physical_merge_sort::sirius_physical_merge_sort(sirius_physical_order* order_by)
  : sirius_physical_merge_sort(
      order_by->types,                // copied by value
      copy_orders(order_by->orders),  // deep copy (contains unique_ptr<Expression>)
      order_by->projections,          // copied by value
      order_by->estimated_cardinality,
      order_by->is_index_sort,
      order_by->comparison)
{
  child_op    = order_by;
  chunk_bytes = order_by->chunk_bytes;
}

sirius_physical_merge_sort::sirius_physical_merge_sort(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  duckdb::vector<std::size_t> projections_p,
  std::size_t estimated_cardinality,
  bool is_index_sort_p,
  std::shared_ptr<const sort_comparison_config> comparison_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::MERGE_SORT, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    comparison(comparison_p ? std::move(comparison_p)
                            : std::make_shared<const sort_comparison_config>(this->orders,
                                                                             this->types.size())),
    projections(std::move(projections_p)),
    is_index_sort(is_index_sort_p)
{
}

}  // namespace op
}  // namespace sirius
