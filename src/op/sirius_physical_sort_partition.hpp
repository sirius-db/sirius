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

// Each slice retains its range index even when other ranges are empty.
class sort_partition_output : public pipelineable_operator_data {
 public:
  sort_partition_output(std::vector<std::shared_ptr<cucascade::data_batch>> batches,
                        std::vector<std::size_t> partitions)
    : pipelineable_operator_data(std::move(batches)), partitions(std::move(partitions))
  {
  }
  std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
    std::vector<std::shared_ptr<cucascade::data_batch>> batches) const override
  {
    auto result = std::make_unique<sort_partition_output>(std::move(batches), partitions);
    result->adopt_row_identity(*this);
    return result;
  }
  std::vector<std::size_t> partitions;
};

class sirius_physical_sort_sample;

class sirius_physical_sort_partition : public sirius_physical_operator {
 public:
  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }

  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::SORT_PARTITION;

 public:
  sirius_physical_sort_partition(sirius_physical_order* order_by);

  sirius_physical_sort_partition(duckdb::vector<sirius::logical_type> types,
                                 duckdb::vector<duckdb::BoundOrderByNode> orders,
                                 duckdb::vector<std::size_t> projections_p,
                                 std::size_t estimated_cardinality);

  //! Order specification (copied from ORDER_BY)
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  duckdb::vector<std::size_t> projections;

 public:
  // Source interface
  bool is_source() const override { return true; }

 public:
  // Sink interface
  bool is_sink() const override { return true; }

 public:
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  //! Set the sample operator to read partition boundaries from
  void set_sample_op(sirius_physical_sort_sample* sample) { _sample_op = sample; }

  //! Get the sample operator
  sirius_physical_sort_sample* get_sample_op() const { return _sample_op; }

  void sink(const operator_data& output, ::cuda::stream_ref stream) override;
  void on_finalize_operator() override;

 private:
  sirius_physical_sort_sample* _sample_op = nullptr;
  std::once_flag _initialize_partitions;
};

}  // namespace op
}  // namespace sirius
