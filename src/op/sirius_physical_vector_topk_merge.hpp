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
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "sirius_config.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace sirius {
namespace op {

class sirius_physical_vector_topk_join;

//! Merge stage of the exact per-row top-k vector join. Each task takes one left batch: its
//! forwarded left columns plus one partial per right batch from VECTOR_TOPK_JOIN, merges the
//! partials into each left row's overall top-k with cuVS knn_merge_parts, and emits
//! [left cols..., right cols..., ranking value (FLOAT)].
class sirius_physical_vector_topk_merge : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::VECTOR_TOPK_MERGE;

  sirius_physical_vector_topk_merge(
    const sirius_physical_vector_topk_join& join,
    uint64_t batch_bytes = sirius::config::DEFAULT_CONCAT_BATCH_BYTES);

  std::int64_t k;
  //! Ranked by cosine similarity: report `1 - cosine distance`.
  bool is_similarity;
  duckdb::JoinType join_type;
  //! Byte budget for each emitted output batch.
  uint64_t batch_bytes;

  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }

  //! Drains one left batch's partitions (its partials and its left columns) per call.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

 private:
  std::mutex drain_mutex_;
  std::size_t next_left_ordinal_ = 0;
  //! Right child's column types, for NULL-padding when the right side is empty.
  duckdb::vector<sirius::logical_type> right_types_;
};

}  // namespace op
}  // namespace sirius
