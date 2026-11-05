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
#include "op/sirius_physical_ungrouped_aggregate.hpp"

#include <atomic>
#include <memory>

namespace sirius {
namespace op {

/**
 * @brief Finalize one ungrouped aggregate result after its entire input finishes.
 *
 * A single task merges all partials, or produces the SQL empty-input row when no batches arrived.
 * Scheduling claims are atomic; the inherited mutex serializes input handout. Pending finalization
 * keeps @c all_ports_empty() false until the descriptor is handed to the task creator, whose
 * pipeline lock protects the transition to task accounting.
 */
class sirius_physical_ungrouped_aggregate_merge : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::MERGE_AGGREGATE;

 public:
  sirius_physical_ungrouped_aggregate_merge(
    sirius_physical_ungrouped_aggregate* ungrouped_aggregate);

  sirius_physical_ungrouped_aggregate_merge(
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list,
    std::size_t estimated_cardinality,
    duckdb::TupleDataValidityType distinct_validity);

  //! The aggregates that have to be computed
  duckdb::vector<std::unique_ptr<sirius::ast::node>> aggregates;

  sirius_physical_operator* child_op = nullptr;
  sirius_physical_operator* get_child_op() const { return child_op; }

  bool is_source() const override { return true; }

  std::optional<task_creation_hint> get_next_task_hint() override;
  [[nodiscard]] bool all_ports_empty() override;
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

 public:
  bool is_sink() const override { return true; }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

 private:
  [[nodiscard]] std::size_t empty_input_size_estimate() const;

  std::atomic<bool> _task_claimed{false};
  std::atomic<bool> _input_handed_out{false};
};

}  // namespace op
}  // namespace sirius
