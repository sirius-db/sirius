/*
 * Copyright 2026, Sirius Contributors.
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

#include "op/sirius_physical_operator.hpp"
#include "vss/vector_join.hpp"

#include <atomic>

namespace sirius::op {

/**
 * @brief The input handed to one sirius_physical_vector_join::execute() call.
 *
 * VECTOR_JOIN is a source, so there are no upstream batches to pass in. This just
 * carries an estimated size (so the scheduler can reserve GPU memory) and once
 * granted, the memory space where execute() builds the output table.
 *
 * Not a pipelineable_operator_data on purpose: the task creator on reading an empty
 * pipelineable input as "no data" would skip making a task for this source.
 */
class vector_join_input : public operator_data {
 public:
  explicit vector_join_input(std::size_t estimated_bytes) : _estimated_bytes(estimated_bytes) {}

  [[nodiscard]] operator_data_type get_type() const override { return operator_data_type::BASE; }

  /// prepare_for_processing() saves a pointer to the GPU memory the task is allowed to use.
  /// execute() reads that pointer via get_gpu_memory_space() and builds its output table there.
  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              rmm::cuda_stream_view /*stream*/) override
  {
    _gpu_memory_space = const_cast<::cucascade::memory::memory_space*>(requested_memory_space);
  }

  /// Feeds the memory reservation system so the scheduler knows how much GPU memory the task needs
  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    return _estimated_bytes;
  }

  /// Memory space captured by prepare_for_processing where execute() builds its output table
  [[nodiscard]] ::cucascade::memory::memory_space* get_gpu_memory_space() const
  {
    return _gpu_memory_space;
  }

 private:
  ::cucascade::memory::memory_space* _gpu_memory_space = nullptr;
  std::size_t _estimated_bytes;
};

/**
 * @brief GPU source operator for a k-nearest-neighbor vector join.
 *
 * Joins a right table against a left table by vector similarity. Dedup is the
 * special case where left == right. Assumes both pinned tables fit on the
 * device for now.
 */
class sirius_physical_vector_join : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::VECTOR_JOIN;

  sirius_physical_vector_join(duckdb::vector<sirius::logical_type> types,
                              duckdb::idx_t estimated_cardinality,
                              sirius::vss::vector_join_request request);

  [[nodiscard]] const sirius::vss::vector_join_request& request() const { return _request; }

  // -----------------------------
  // Source interface
  // -----------------------------
  /// tells the pipeline this is a leaf that produces data
  bool is_source() const override { return true; }

  /// scheduler asks any work? → returns READY once (make one task), nullopt after
  std::optional<task_creation_hint> get_next_task_hint() override;
  /// returns true once the single input has been handed out, so the scheduler stops asking
  [[nodiscard]] bool all_ports_empty() override;
  /// builds the one vector_join_input parcel for that task; returns nullptr after the first
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  // -----------------------------
  // Execution
  // -----------------------------
  /// does the actual GPU work and returns the output batches
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  /// tells the reservation system how much GPU memory the task will peak at, when there's no
  /// past run to estimate from
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  std::string params_to_string() const override;

 private:
  [[nodiscard]] std::size_t estimated_source_bytes() const;

  //! Resolved corpus/probe identity + tuning knobs, carried from SiriusVectorJoinBind.
  sirius::vss::vector_join_request _request;
  std::atomic<bool> _task_scheduled{false};    // gates the next task hint
  std::atomic<bool> _input_handed_out{false};  // gates the all ports empty
};

}  // namespace sirius::op
