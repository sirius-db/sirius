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

class sirius_physical_order : public sirius_physical_operator {
 public:
  static constexpr const duckdb::PhysicalOperatorType TYPE = duckdb::PhysicalOperatorType::ORDER_BY;

 public:
  sirius_physical_order(duckdb::vector<duckdb::LogicalType> types,
                        duckdb::vector<duckdb::BoundOrderByNode> orders,
                        duckdb::vector<duckdb::idx_t> projections_p,
                        duckdb::idx_t estimated_cardinality,
                        bool is_index_sort_p = false);

  //! Input data
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  duckdb::vector<duckdb::idx_t> projections;
  bool is_index_sort;

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
};

}  // namespace op
}  // namespace sirius
