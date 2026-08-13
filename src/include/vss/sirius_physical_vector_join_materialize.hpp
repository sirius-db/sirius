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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cstddef>
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
 * reduce_local already prepended this partition's repeated left output columns.
 * This stage drains one partition (left batch) per task and finishes the TVF rows:
 *   - left cols: passed through from the input,
 *   - right cols: on the fast path, col0 is the right table's id value, no gather.
 *                 On the payload path, gather right output columns by col0 (the right table's
 *                 global row number).
 *   - score: the distance/similarity.
 *
 * Output is a plain pipelineable_operator_data, the result collector streams
 * it to the host.
 */
class sirius_physical_vector_join_materialize : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::VECTOR_JOIN_MATERIALIZE;

  sirius_physical_vector_join_materialize(duckdb::vector<sirius::logical_type> types,
                                          duckdb::idx_t estimated_cardinality,
                                          sirius::vss::vector_join_request request,
                                          sirius::scan_manager::sirius_scan_manager* scan_manager,
                                          bool is_fast_path);

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
  /// Payload path only: snapshot the right output columns' per-batch views plus
  /// each batch's global row offset, so a row can be gathered from the batch that
  /// owns it. Idempotent. The fast path gathers nothing, so it skips this.
  void ensure_initialized();

  /// Payload path only: gather the right output columns for col0.
  [[nodiscard]] std::unique_ptr<cudf::table> gather_right_by_batch(
    cudf::column_view const& col0_partitioned,
    std::vector<cudf::size_type> const& part_offsets,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  sirius::vss::vector_join_request _request;
  sirius::scan_manager::sirius_scan_manager* _scan_manager;
  bool _is_fast_path{false};

  std::mutex _drain_mutex;                  // guards get_next_task_input_data()
  std::size_t _current_partition_index{0};  // next partition (left batch) to drain

  std::mutex _init_mutex;  // guards the one-time snapshot below
  bool _initialized{false};
  //! Payload path only: right output columns as zero-copy per-batch views,
  //  indexed [output_col][batch]. Unused on the fast path.
  std::vector<std::vector<cudf::column_view>> _right_output_views;
  //! Payload path only: global row offset (prefix sum) of each right batch.
  std::vector<std::int64_t> _right_offsets;
};

}  // namespace sirius::op
