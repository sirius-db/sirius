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

namespace sirius {
namespace op {

// Helper to deep copy BoundOrderByNode vector (contains unique_ptr<Expression>)
inline duckdb::vector<duckdb::BoundOrderByNode> copy_orders(
  const duckdb::vector<duckdb::BoundOrderByNode>& src)
{
  duckdb::vector<duckdb::BoundOrderByNode> result;
  result.reserve(src.size());
  for (const auto& order : src) {
    result.push_back(order.Copy());
  }
  return result;
}

//! Render sort keys as one line ("#0 ASC, #1 DESC") for params_to_string.
inline std::string orders_to_string(const duckdb::vector<duckdb::BoundOrderByNode>& orders)
{
  std::string result;
  for (const auto& order : orders) {
    if (!result.empty()) { result += ", "; }
    result += order.ToString();
  }
  return result;
}

class sirius_physical_order : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::ORDER_BY;

  sirius_physical_order(duckdb::vector<sirius::logical_type> types,
                        duckdb::vector<duckdb::BoundOrderByNode> orders,
                        duckdb::vector<std::size_t> projections_p,
                        std::size_t estimated_cardinality,
                        bool is_index_sort_p = false);

  //! Input data
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  duckdb::vector<std::size_t> projections;
  bool is_index_sort;

  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }
  bool sink_order_dependent() const override { return false; }

  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;
  std::string params_to_string() const override { return orders_to_string(orders); }
};

}  // namespace op
}  // namespace sirius
