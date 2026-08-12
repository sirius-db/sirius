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

#include <cudf/column/column_view.hpp>

#include <cstdint>
#include <mutex>
#include <vector>

namespace sirius::scan_manager {
class sirius_scan_manager;
}  // namespace sirius::scan_manager

namespace sirius::op {

/**
 * @brief The input handed to one sirius_physical_vector_join_select::execute() call.
 *
 * VECTOR_JOIN_SELECT (search stage) is a source: it Cartesian-walks the left×right
 * pinned batch grid and emits one task per (left batch, right batch) pair. This
 * parcel names that pair by index into the operator's snapshotted batch views,
 * carries the estimated size (so the scheduler can reserve GPU memory), and once
 * granted, the memory space where execute() builds the partial output.
 *
 * Not a pipelineable_operator_data on purpose: the task creator reading an empty
 * pipelineable input as "no data" would skip making a task for this source.
 */
class vector_join_input : public operator_data {
 public:
  vector_join_input(std::size_t left_idx, std::size_t right_idx, std::size_t estimated_bytes)
    : _left_idx(left_idx), _right_idx(right_idx), _estimated_bytes(estimated_bytes)
  {
  }

  [[nodiscard]] operator_data_type get_type() const override { return operator_data_type::BASE; }

  /// prepare_for_processing() saves a pointer to the GPU memory the task may use.
  /// execute() reads it via get_gpu_memory_space() and builds its output there.
  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              rmm::cuda_stream_view /*stream*/) override
  {
    _gpu_memory_space = const_cast<::cucascade::memory::memory_space*>(requested_memory_space);
  }

  /// Feeds the reservation system so the scheduler knows how much GPU memory this task needs.
  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    return _estimated_bytes;
  }

  [[nodiscard]] ::cucascade::memory::memory_space* get_gpu_memory_space() const
  {
    return _gpu_memory_space;
  }

  /// Index of this task's left batch (also the output partition, all of a left
  /// batch's per-right-batch partials share it, so the merge stage groups them).
  [[nodiscard]] std::size_t left_idx() const { return _left_idx; }
  /// Index of this task's right batch.
  [[nodiscard]] std::size_t right_idx() const { return _right_idx; }

 private:
  std::size_t _left_idx;
  std::size_t _right_idx;
  ::cucascade::memory::memory_space* _gpu_memory_space = nullptr;
  std::size_t _estimated_bytes;
};

/**
 * @brief GPU source operator for a k-nearest-neighbor vector join (search stage).
 *
 * Reads two pinned tables directly and emits one task per (left batch, right
 * batch) pair, self-sourced from the pinned cache rather than scan-fed ports.
 * Each task runs a brute-force knn of the left batch against one right batch
 * and emits that batch's partial top-k, with neighbor ids already shifted into
 * the global right-table row space.
 *
 * This is the first of three stages: the partials still need a per-left-batch
 * merge (reduce across right batches) and output materialization downstream.
 * Dedup is the special case where left == right. Assumes both pinned tables fit
 * on the device for now.
 */
class sirius_physical_vector_join_select : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::VECTOR_JOIN_SELECT;

  sirius_physical_vector_join_select(duckdb::vector<sirius::logical_type> types,
                                     duckdb::idx_t estimated_cardinality,
                                     sirius::vss::vector_join_request request,
                                     sirius::scan_manager::sirius_scan_manager* scan_manager);

  [[nodiscard]] const sirius::vss::vector_join_request& request() const { return _request; }

  // -----------------------------
  // Source interface
  // -----------------------------
  /// Tells the pipeline this is a leaf that produces data.
  bool is_source() const override { return true; }

  /// READY once (kicking off the drain of all pairs), nullopt after / when empty.
  std::optional<task_creation_hint> get_next_task_hint() override;
  /// True once every (left, right) pair has been handed out.
  [[nodiscard]] bool all_ports_empty() override;
  /// Builds the vector_join_input for the next pair; nullptr once the grid is walked.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  // -----------------------------
  // Execution
  // -----------------------------
  /// Runs brute_force_knn on this task's pair and returns the partial top-k batch.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  /// Routes each partial to the merge stage's partition = its left batch index,
  /// so a left batch's per-right-batch partials group together (mirrors PARTITION).
  void sink(const operator_data& output_data, rmm::cuda_stream_view stream) override;

  /// Peak GPU memory estimate for the reservation when there's no run history.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  std::string params_to_string() const override;

 private:
  /// Resolve both pinned tables and snapshot their per-batch vector-column views
  /// plus each right batch's global row offset. Idempotent and caller holds _op_mutex.
  void ensure_initialized_locked();

  /// Rough peak-bytes for one pair's task (partial output + brute-force scratch).
  [[nodiscard]] std::size_t per_pair_estimate(std::size_t left_idx) const;

  //! Resolved corpus/probe identity + tuning knobs, carried from SiriusVectorJoinBind.
  sirius::vss::vector_join_request _request;
  //! Scan manager the pinned left/right tables are resolved against (query-lived).
  sirius::scan_manager::sirius_scan_manager* _scan_manager;

  std::mutex _op_mutex;  // guards lazy init and the walk cursor
  bool _initialized{false};
  bool _hint_returned{false};
  //! Per-batch vector-column views (zero-copy over the pinned cache), left/right.
  std::vector<cudf::column_view> _left_views;
  std::vector<cudf::column_view> _right_views;
  //! Global row offset of each right batch (prefix sum of right batch row counts).
  std::vector<std::int64_t> _right_offsets;
  std::size_t _num_pairs{0};
  std::size_t _next_pair{0};
};

}  // namespace sirius::op
