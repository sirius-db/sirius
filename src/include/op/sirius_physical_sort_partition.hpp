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

namespace sirius {
namespace op {

class sirius_physical_sort_sample;

class sirius_physical_sort_partition : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::SORT_PARTITION;

 public:
  sirius_physical_sort_partition(sirius_physical_order* order_by);

  sirius_physical_sort_partition(duckdb::vector<duckdb::LogicalType> types,
                                 duckdb::vector<duckdb::BoundOrderByNode> orders,
                                 duckdb::vector<duckdb::idx_t> projections_p,
                                 duckdb::idx_t estimated_cardinality);

  //! Order specification (copied from ORDER_BY)
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  duckdb::vector<duckdb::idx_t> projections;

 public:
  // Source interface
  bool is_source() const override { return true; }

  duckdb::OrderPreservationType source_order() const override
  {
    return duckdb::OrderPreservationType::FIXED_ORDER;
  }

 public:
  // Sink interface
  bool is_sink() const override { return true; }
  bool sink_order_dependent() const override { return false; }

 public:
  std::unique_ptr<operator_data> execute(
    const operator_data& input_data,
    rmm::cuda_stream_view stream = cudf::get_default_stream()) override;

  //! Set the sample operator to read partition boundaries from
  void set_sample_op(sirius_physical_sort_sample* sample) { _sample_op = sample; }

  //! Get the sample operator
  sirius_physical_sort_sample* get_sample_op() const { return _sample_op; }

 private:
  sirius_physical_sort_sample* _sample_op = nullptr;
};

}  // namespace op
}  // namespace sirius
