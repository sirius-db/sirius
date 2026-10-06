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
#include "vss/pinned_column.hpp"
#include "vss/vector_join.hpp"
#include "vss/vector_join_materialized_side.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_reservation.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace sirius::scan_manager {
class sirius_scan_manager;
}  // namespace sirius::scan_manager
namespace duckdb {
class SiriusContext;
}  // namespace duckdb
namespace sirius::scan_manager {
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

  /// @param build_side  The corpus row order the fold numbered its neighbour ids against.
  ///                    Non-null exactly when the join runs a build phase; this operator
  ///                    resolves ids against that same list rather than re-deriving one, which
  ///                    is what keeps position i meaning the same row to both stages.
  sirius_physical_vector_join_materialize(
    duckdb::vector<sirius::logical_type> types,
    duckdb::idx_t estimated_cardinality,
    sirius::vss::vector_join_request request,
    sirius::scan_manager::sirius_scan_manager* scan_manager,
    std::shared_ptr<sirius::vss::materialized_side_buffer> build_side = nullptr,
    std::shared_ptr<sirius::vss::materialized_side_buffer> probe_side = nullptr,
    duckdb::SiriusContext* sirius_ctx                                 = nullptr);

  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }
  bool sink_order_dependent() const override { return false; }

  /// Drains all merge outputs of one partition (one left batch) per call.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  /// Gathers the left/right output columns for one left batch's top-k and emits the final rows.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  std::string params_to_string() const override;

 private:
  /// Resolve both pinned tables, snapshot the left output columns per batch, and
  /// concatenate the right output columns once (indexed by global right id).
  /// Idempotent; needs a stream + memory space, so it runs on first execute().
  void ensure_initialized(rmm::cuda_stream_view stream, ::cucascade::memory::memory_space& space);

  /// The corpus output columns concatenated across the build side's snapshot, in the order the
  /// fold numbered against. Build path only.
  std::vector<std::unique_ptr<cudf::column>> build_side_output_columns(
    std::size_t num_output_columns,
    rmm::cuda_stream_view stream,
    ::cucascade::memory::memory_space& space);

  /// The corpus output columns at @p neighbors (pin rows), from only the pin chunks those rows fall
  /// in: an answer of n x k rows reads n x k chunks at most, not the whole column.
  /// One piece of a left batch's pairs [left row, right id, distance] -> the TVF schema.
  std::unique_ptr<cudf::table> materialize_piece(std::size_t partition_idx,
                                                 cudf::table_view pairs,
                                                 cucascade::memory::memory_space& space,
                                                 rmm::cuda_stream_view stream);

  std::unique_ptr<cudf::table> gather_right_from_pin(cudf::column_view neighbors,
                                                     rmm::cuda_stream_view stream,
                                                     ::cucascade::memory::memory_space& space);

  /// The same from a HOST-tier pin, read in place on the host for a few scattered rows: staging
  /// copies whole chunks, and a small answer over a large corpus touches every chunk. Null when
  /// it does not apply -- too many rows, or a column that is not fixed-width or has nulls.
  std::unique_ptr<cudf::table> gather_right_on_host(cudf::column_view neighbors,
                                                    rmm::cuda_stream_view stream,
                                                    ::cucascade::memory::memory_space& space);

  /// The probe output columns as per-batch views in the probe side's snapshot order, which is
  /// the order the join stage numbered its output partitions by. Probe-scan path only.
  std::vector<std::vector<cudf::column_view>> probe_side_output_views(
    std::size_t num_output_columns,
    rmm::cuda_stream_view stream,
    ::cucascade::memory::memory_space& space);

  sirius::vss::vector_join_request _request;
  sirius::scan_manager::sirius_scan_manager* _scan_manager;
  /// Shared corpus row order; null on the pinned path, where the pinned entry plays that role.
  std::shared_ptr<sirius::vss::materialized_side_buffer> _build_side;
  /// Shared probe row order; null on the pinned path.
  std::shared_ptr<sirius::vss::materialized_side_buffer> _probe_side;

  std::mutex _drain_mutex;
  duckdb::SiriusContext* _sirius_ctx{nullptr};  // for the host memory space a disk piece comes back to                  // guards get_next_task_input_data()
  std::size_t _current_partition_index{0};  // next partition (left batch) to drain

  std::mutex _init_mutex;  // guards the one-time init below
  bool _initialized{false};
  //! Left output columns as zero-copy views, indexed [output_col][batch].
  std::vector<std::vector<cudf::column_view>> _left_output_cols;
  //! Keeps staged left copies alive when the pin is HOST-tier; empty for GPU-tier pins.
  std::vector<vss::staged_pinned_column> _staged_left;
  //! Probe-scan path: the borrows and re-staged copies backing _left_output_cols. Unlike the
  //! corpus columns, which are concatenated once and then owned outright, the left views point
  //! into these for the operator's whole life, so they are held rather than released per call.
  std::vector<::cucascade::read_only_data_batch> _probe_readers;
  std::vector<std::shared_ptr<::cucascade::data_batch>> _probe_restaged;
  std::vector<std::shared_ptr<::cucascade::memory::reservation>> _probe_reservations;
  //! Build path: right output columns concatenated across batches; row i == global right id i.
  std::unique_ptr<cudf::table> _right_output_concat;
  //! Pinned corpus: the pin, and where each of its chunks ends in pin row space.
  std::shared_ptr<const sirius::scan_manager::pinned_entry> _right_pin;
  std::vector<std::int64_t> _right_chunk_ends;
};

}  // namespace sirius::op
