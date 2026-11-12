/*
 * Copyright 2026, Sirius Contributors.
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

#include "planner/sirius_plan_sort_exchange.hpp"

#include "op/sirius_physical_merge_sort.hpp"
#include "op/sirius_physical_sort_partition.hpp"
#include "op/sirius_physical_sort_sample.hpp"

#include <stdexcept>
#include <utility>

namespace sirius::planner {

sort_exchange_options sort_exchange_options::from(operator_params const& params)
{
  return sort_exchange_options{params.sort_sample_bytes,
                               params.max_sort_partition_bytes,
                               params.max_sort_partition_memory_fraction,
                               std::nullopt};
}

duckdb::unique_ptr<op::sirius_physical_operator> build_sort_exchange(
  duckdb::unique_ptr<op::sirius_physical_order> order, sort_exchange_options const& options)
{
  if (!order || order->children.empty()) {
    throw std::invalid_argument("Sort exchange requires an ORDER_BY with a child input");
  }
  auto* order_ptr = order.get();

  auto original_projections = order_ptr->projections;
  auto const& child_types   = order_ptr->children[0]->types;

  duckdb::vector<std::size_t> identity_proj;
  identity_proj.reserve(child_types.size());
  for (std::size_t col_idx = 0; col_idx < child_types.size(); col_idx++) {
    identity_proj.push_back(col_idx);
  }
  order_ptr->projections = std::move(identity_proj);
  order_ptr->types       = child_types;

  auto sample =
    duckdb::make_uniq<op::sirius_physical_sort_sample>(order_ptr,
                                                       options.sort_sample_bytes,
                                                       options.max_partition_bytes,
                                                       options.max_partition_memory_fraction);
  if (options.boundary_key_count) { sample->set_boundary_key_count(*options.boundary_key_count); }
  auto* sample_ptr = sample.get();
  sample->children.push_back(std::move(order));

  auto partition = duckdb::make_uniq<op::sirius_physical_sort_partition>(order_ptr);
  partition->set_sample_op(sample_ptr);
  partition->children.push_back(std::move(sample));

  auto merge = duckdb::make_uniq<op::sirius_physical_merge_sort>(order_ptr);

  bool is_identity = (original_projections.size() == order_ptr->types.size());
  if (is_identity) {
    for (std::size_t i = 0; i < original_projections.size(); i++) {
      if (original_projections[i] != i) {
        is_identity = false;
        break;
      }
    }
  }
  if (!is_identity) {
    duckdb::vector<logical_type> output_types;
    output_types.reserve(original_projections.size());
    for (auto idx : original_projections) {
      output_types.push_back(order_ptr->types[idx]);
    }
    merge->set_final_projections(std::move(original_projections), std::move(output_types));
  }

  merge->children.push_back(std::move(partition));
  return merge;
}

}  // namespace sirius::planner
