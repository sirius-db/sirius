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

#pragma once

#include "duckdb/planner/bound_query_node.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"

#include <mutex>

namespace sirius {
namespace op {

class sirius_physical_merge_sort : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::MERGE_SORT;

 public:
  sirius_physical_merge_sort(sirius_physical_order* order_by);

  sirius_physical_merge_sort(duckdb::vector<sirius::logical_type> types,
                             duckdb::vector<duckdb::BoundOrderByNode> orders,
                             duckdb::vector<std::size_t> projections_p,
                             std::size_t estimated_cardinality,
                             bool is_index_sort_p = false);

  //! Input data
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  duckdb::vector<std::size_t> projections;
  bool is_index_sort;

  sirius_physical_operator* child_op;
  sirius_physical_operator* get_child_op() const { return child_op; }

 public:
  // Source interface
  bool is_source() const override { return true; }

  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }

 public:
  // Sink interface
  bool is_sink() const override { return true; }
  bool sink_order_dependent() const override { return false; }

 public:
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;
  std::string params_to_string() const override { return orders_to_string(orders); }

  //! Set the final output projection (applied after merge, to remove sort-key-only columns)
  void set_final_projections(duckdb::vector<std::size_t> proj,
                             duckdb::vector<sirius::logical_type> output_types)
  {
    _final_projections = std::move(proj);
    types              = std::move(output_types);
  }

 private:
  //! Final projection to apply after merge (empty = no extra projection)
  duckdb::vector<std::size_t> _final_projections;
  //! Guards concurrent calls to get_next_task_input_data().
  std::mutex _drain_mutex;
  //! Tracks which partition to drain next (one task per partition).
  size_t _current_partition_index{0};
};

}  // namespace op
}  // namespace sirius
