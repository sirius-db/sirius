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

#include "duckdb/common/enums/tuple_data_layout_enums.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/execution/operator/aggregate/distinct_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"

namespace sirius {
namespace op {

class sirius_physical_ungrouped_aggregate_merge : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::MERGE_AGGREGATE;

 public:
  sirius_physical_ungrouped_aggregate_merge(
    sirius_physical_ungrouped_aggregate* ungrouped_aggregate);

  sirius_physical_ungrouped_aggregate_merge(
    duckdb::vector<duckdb::LogicalType> types,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list,
    duckdb::idx_t estimated_cardinality,
    duckdb::TupleDataValidityType distinct_validity);

  //! The aggregates that have to be computed
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;
  duckdb::unique_ptr<duckdb::DistinctAggregateData> distinct_data;
  duckdb::unique_ptr<duckdb::DistinctAggregateCollectionInfo> distinct_collection_info;

  sirius_physical_operator* child_op;
  sirius_physical_operator* get_child_op() const { return child_op; }

  bool is_source() const override { return true; }

 public:
  bool is_sink() const override { return true; }

  std::vector<std::shared_ptr<cucascade::data_batch>> execute(
    const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches) override;
};

}  // namespace op
}  // namespace sirius
