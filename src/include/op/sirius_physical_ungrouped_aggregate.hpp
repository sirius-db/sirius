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
#include "expression/ast/node.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace sirius {
namespace op {

class sirius_physical_ungrouped_aggregate : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE;

 public:
  sirius_physical_ungrouped_aggregate(
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list,
    std::size_t estimated_cardinality,
    duckdb::TupleDataValidityType distinct_validity);

  //! The aggregates that have to be computed
  duckdb::vector<std::unique_ptr<sirius::ast::node>> aggregates;

  bool is_source() const override { return true; }

  // Returns the source (scan) batch ID that produced the local aggregate output with
  // `output_batch_id`. Used by the merge operator to sort partial results in scan order
  // before selecting NTH_ELEMENT(0) for the first() aggregate.
  uint64_t get_source_batch_id(uint64_t output_batch_id) const;

 public:
  bool is_sink() const override { return true; }
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

 private:
  // Maps output_batch_id -> source (scan) batch_id.  Populated by execute() and
  // read by sirius_physical_ungrouped_aggregate_merge to restore scan order.
  mutable std::mutex source_id_mutex_;
  std::unordered_map<uint64_t, uint64_t> output_to_source_id_;
};

}  // namespace op
}  // namespace sirius
