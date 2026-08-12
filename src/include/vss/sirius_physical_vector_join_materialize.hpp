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
#include "vss/vector_join.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace sirius::scan_manager {
class sirius_scan_manager;
}  // namespace sirius::scan_manager

namespace sirius::op {

/**
 * @brief Materialize stage of the vector join: turns the merge's per-left-batch
 *        top-k id/distance lists into the output rows and streams them to DuckDB.
 *
 * The merge tags each per-row top-k result with its left batch index as the
 * partition. This drains one partition (left batch Ri) per task and builds the
 * TVF rows `[left_output_cols…, right_output_cols…, score]`:
 *   - left cols: `cudf::repeat` batch Ri's output columns k times (each left row
 *     is repeated for its k neighbors),
 *   - right cols: `cudf::gather` the (once-concatenated) right output columns by
 *     the global neighbor id,
 *   - score: the distance, mapped to similarity for cosine+similarity output.
 *
 * Output is a plain `pipelineable_operator_data` — the result collector streams
 * it to the host. Refinement (exact distances) happens upstream in the selection
 * stage, so materialize never touches the vectors. Per-row top-k only for now.
 */
class sirius_physical_vector_join_materialize : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::VECTOR_JOIN_MATERIALIZE;

  sirius_physical_vector_join_materialize(duckdb::vector<sirius::logical_type> types,
                                          duckdb::idx_t estimated_cardinality,
                                          sirius::vss::vector_join_request request,
                                          sirius::scan_manager::sirius_scan_manager* scan_manager);

  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }
  bool sink_order_dependent() const override { return false; }

  /// Drains all merge outputs of one partition (one left batch) per call.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  /// Gathers the left/right output columns for one left batch's top-k and emits the final rows.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  std::string params_to_string() const override;

 private:
  /// Resolve both pinned tables, snapshot the left output columns per batch, and
  /// concatenate the right output columns once (indexed by global right id).
  /// Idempotent; needs a stream + memory space, so it runs on first execute().
  void ensure_initialized(rmm::cuda_stream_view stream, ::cucascade::memory::memory_space& space);

  sirius::vss::vector_join_request _request;
  sirius::scan_manager::sirius_scan_manager* _scan_manager;

  std::mutex _drain_mutex;                  // guards get_next_task_input_data()
  std::size_t _current_partition_index{0};  // next partition (left batch) to drain

  std::mutex _init_mutex;  // guards the one-time init below
  bool _initialized{false};
  //! Left output columns as zero-copy views, indexed [output_col][batch].
  std::vector<std::vector<cudf::column_view>> _left_output_cols;
  //! Right output columns concatenated across batches; row i == global right id i.
  std::unique_ptr<cudf::table> _right_output_concat;
};

}  // namespace sirius::op
