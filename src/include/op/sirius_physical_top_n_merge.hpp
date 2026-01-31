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
#include "op/sirius_physical_top_n.hpp"

namespace duckdb {
struct DynamicFilterData;
}  // namespace duckdb

namespace sirius {
namespace op {

//! Represents a physical ordering of the data. Note that this will not change
//! the data but only add a selection vector.
class sirius_physical_top_n_merge : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::MERGE_TOP_N;

 public:
  sirius_physical_top_n_merge(sirius_physical_top_n* top_n);

  sirius_physical_top_n_merge(duckdb::vector<duckdb::LogicalType> types_p,
                              duckdb::vector<duckdb::BoundOrderByNode> orders,
                              duckdb::idx_t limit,
                              duckdb::idx_t offset,
                              duckdb::shared_ptr<duckdb::DynamicFilterData> dynamic_filter,
                              duckdb::idx_t estimated_cardinality);

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  duckdb::idx_t limit;
  duckdb::idx_t offset;
  //! Dynamic table filter (if any)
  duckdb::shared_ptr<duckdb::DynamicFilterData> dynamic_filter;

  sirius_physical_operator* child_op;
  sirius_physical_operator* get_child_op() const { return child_op; }

 public:
  bool is_source() const override { return true; }
  duckdb::OrderPreservationType source_order() const override
  {
    return duckdb::OrderPreservationType::FIXED_ORDER;
  }

 public:
  bool is_sink() const override { return true; }

  std::vector<std::shared_ptr<cucascade::data_batch>> execute(
    const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches) override;
};

}  // namespace op
}  // namespace sirius
