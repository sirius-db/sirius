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

#include <cucascade/memory/memory_space.hpp>

#include <memory>

namespace sirius {
namespace op {

/**
 * @brief Marker input for the synthetic zero-input identity task
 *.
 *
 * Handed out by sirius_physical_ungrouped_aggregate_merge::get_next_task_input_data when the
 * upstream pipeline finished without ever producing a partial batch. Carries no data batches;
 * merge::execute recognizes it and constructs the aggregate identity row directly, so the row
 * flows through one completely normal gpu_pipeline_task.
 */
class ungrouped_aggregate_zero_input_data : public operator_data {
 public:
  // Source-style input (scan_operator_input pattern): no batches to lock, only capture the
  // task's reserved memory space so execute() has a target space for the identity output.
  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              rmm::cuda_stream_view /*stream*/) override
  {
    gpu_memory_space = const_cast<::cucascade::memory::memory_space*>(requested_memory_space);
  }

  // Nominal non-zero estimate: a 0-byte input basis degenerates the task's memory
  // reservation sizing (gpu_pipeline_task::get_estimated_reservation_size_info).
  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    return std::size_t{1} << 20;
  }

  ::cucascade::memory::memory_space* gpu_memory_space = nullptr;
};

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

  sirius_physical_operator* child_op;
  sirius_physical_operator* get_child_op() const { return child_op; }

  bool is_source() const override { return true; }

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  // Zero-input identity overrides — hard contract
  //: both read _saw_input /
  // _zero_input_task_created under the same operator `lock`; all_ports_empty() reports
  // non-empty while the identity task still needs creating (blocks premature pipeline
  // finish), and get_next_task_hint() returns READY{this} in that state so the task is
  // actually hinted. Any unlocked read reopens the "hint not yet issued but pipeline
  // finished" race.
  std::optional<task_creation_hint> get_next_task_hint() override;
  [[nodiscard]] bool all_ports_empty() override;

 public:
  bool is_sink() const override { return true; }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

 private:
  //! Caller must hold `lock`.
  bool needs_zero_input_task_locked();

  //! Set (under `lock`) when a partial batch is popped from the FULL port — exactly at
  //! the pop, nowhere else; suppresses
  //! the identity task (no double emit).
  bool _saw_input{false};
  //! Set (under `lock`) once the zero-input marker has been handed out: at most one
  //! identity task per query.
  bool _zero_input_task_created{false};
};

}  // namespace op
}  // namespace sirius
