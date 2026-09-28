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

#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "sirius_config.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace sirius {

namespace pipeline {
class sirius_pipeline;
class sirius_meta_pipeline;
}  // namespace pipeline

namespace op {

class sirius_physical_vector_threshold_join : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::VECTOR_THRESHOLD_JOIN;

 public:
  sirius_physical_vector_threshold_join(
    duckdb::LogicalOperator& op,
    duckdb::unique_ptr<sirius_physical_operator> left,
    duckdb::unique_ptr<sirius_physical_operator> right,
    std::size_t left_vector_col_idx,
    std::size_t right_vector_col_idx,
    float cutoff,
    std::string metric,
    std::int64_t dim,
    duckdb::JoinType join_type,
    std::size_t estimated_cardinality,
    uint64_t batch_bytes = sirius::config::DEFAULT_CONCAT_BATCH_BYTES);

  //! Column index of the FLOAT[dim] vector column within the left (probe) child's output.
  std::size_t left_vector_col_idx;
  //! Column index of the FLOAT[dim] vector column within the right (build) child's output.
  std::size_t right_vector_col_idx;
  //! Distance cutoff in the metric's own units (cosine similarity is pre-flipped to 1 - eps).
  float cutoff;
  //! Distance metric: "l2" or "cosine".
  std::string metric;
  //! Fixed vector dimensionality (from the ARRAY<FLOAT> logical type).
  std::int64_t dim;
  //! The join type (INNER for milestone 1).
  duckdb::JoinType join_type;
  //! Byte budget for each emitted output batch.
  uint64_t batch_bytes;
  //! Which left-child columns this join carries into its output.
  duckdb::vector<std::size_t> left_output_col_idxs;
  //! Which right-child columns this join carries into its output.
  duckdb::vector<std::size_t> right_output_col_idxs;

  //! Ask the join to emit the per-pair distance as one extra trailing FLOAT column.
  void enable_distance_output(bool as_similarity);

  //! Ask the join to skip all output-column gathers and emit only a narrow row-count carrier. Set
  //! by the aggregate planner when the sole consumer is an ungrouped count_star that reads just the
  //! row count. Shrinks the declared output schema to a single TINYINT column.
  void set_output_row_count_only();

 protected:
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

 public:
  //! Always a source as every join emits output.
  bool is_source() const override { return true; }

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //! Right-side (build) concat feeds the "build" port; the left side feeds "default".
  [[nodiscard]] std::string_view input_port_for(
    sirius_physical_operator const& producer) const override;

  //! Streams both sides through per-batch-pair threshold joins on a single partition; never
  //! hash-partitions, broadcasts, or enters build-probe.
  partition_strategy get_partition_strategy(const partition_sizing_input& in) override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

 protected:
  std::mutex batches_to_processed_mutex;
  //! Batch-id lists per partition, populated once on the first dispatch.
  std::vector<std::vector<uint64_t>> left_batch_ids;
  std::vector<std::vector<uint64_t>> right_batch_ids;
  //! Set when the id lists above are populated.
  bool ids_initialized_ = false;
  //! Cursor over the (partition, left batch, right batch) pairs
  std::size_t cursor_partition_ = 0;
  std::size_t cursor_left_      = 0;
  std::size_t cursor_right_     = 0;

  //! Set by enable_distance_output(): emit the pairwise distance as a trailing FLOAT column.
  bool emit_distance_ = false;
  //! When emitting the distance, output `1 - distance` instead of the raw metric distance.
  bool emit_distance_as_similarity_ = false;
  //! when set it skips every output-column gather and emit a 1-column carrier.
  bool output_row_count_only_ = false;
};

}  // namespace op
}  // namespace sirius
