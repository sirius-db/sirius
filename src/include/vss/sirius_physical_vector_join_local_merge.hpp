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

#include "op/sirius_physical_partition_consumer_operator.hpp"

#include <cstdint>
#include <mutex>

namespace sirius::op {

/**
 * @brief Merge stage of the vector join: reduces the selection stage's per-right-batch
 *        partials into each left row's global top-k.
 *
 * Blocking operator (sink of the selection pipeline, source of the next): the
 * selection stage tags each partial with its left batch index as the partition,
 * so all of a left batch's partials (one per right batch) land in one partition.
 * This drains one partition per task and cuVS-`knn_merge_parts` its partials into
 * the per-row top-k, following the MERGE_SORT per-partition-drain pattern.
 *
 * Output preserves the partition (left batch index) so the materialize stage can
 * gather each left row's columns. Assumes right batches held >= k rows, so every
 * partial has k columns (the ragged-batch case is a TODO).
 */
class sirius_physical_vector_join_local_merge : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::VECTOR_JOIN_LOCAL_MERGE;

  sirius_physical_vector_join_local_merge(duckdb::vector<sirius::logical_type> types,
                               duckdb::idx_t estimated_cardinality,
                               std::int64_t k);

  // Source + sink: blocking reduction between the selection and materialize pipelines.
  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }
  bool sink_order_dependent() const override { return false; }

  /// Drains all partials of one partition (one left batch) per call.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  /// knn_merge_parts the drained partials into the left batch's per-row top-k.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  /// Routes the merged result to the materialize stage's partition = its left
  /// batch index, so materialize can gather that batch's left columns.
  void sink(const operator_data& output_data, rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  std::string params_to_string() const override;

 private:
  std::int64_t _k;
  std::mutex _drain_mutex;             // guards concurrent get_next_task_input_data()
  std::size_t _current_partition_index{0};  // next partition (left batch) to drain
};

}  // namespace sirius::op
