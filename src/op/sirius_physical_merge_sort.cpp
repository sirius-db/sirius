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
      order_by->comparison)
{
  chunk_bytes = order_by->chunk_bytes;
}

sirius_physical_merge_sort::sirius_physical_merge_sort(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  duckdb::vector<std::size_t> projections_p,
  std::size_t estimated_cardinality,
  std::shared_ptr<const sort_comparison_config> comparison_p)
  : sirius_physical_sort_stage(
      SiriusPhysicalOperatorType::MERGE_SORT,
      types,
      estimated_cardinality,
      comparison_p ? std::move(comparison_p)
                   : std::make_shared<const sort_comparison_config>(orders, types.size())),
    orders(std::move(orders)),
    projections(std::move(projections_p))
{
}

}  // namespace op
}  // namespace sirius
