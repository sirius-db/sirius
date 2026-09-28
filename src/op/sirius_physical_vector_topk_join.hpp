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

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius {

namespace pipeline {
class sirius_pipeline;
class sirius_meta_pipeline;
}  // namespace pipeline

namespace op {

//! One (left batch, right batch) pair for the top-k select stage. The right batch is absent when
//! the right table is empty.
class topk_pair_input : public pipelineable_operator_data {
 public:
  topk_pair_input(std::vector<std::shared_ptr<::cucascade::data_batch>> batches,
                  std::size_t left_ordinal,
                  bool emit_left_part)
    : pipelineable_operator_data(std::move(batches)),
      left_ordinal(left_ordinal),
      emit_left_part(emit_left_part)
  {
  }

  //! Which left batch this is; the merge collects everything for one left batch together.
  std::size_t left_ordinal;
  //! Whether this pair also forwards the left batch's columns (once per left batch).
  bool emit_left_part;
};

//! Select-stage output: each batch goes to its own merge partition.
class topk_partial_output : public pipelineable_operator_data {
 public:
  topk_partial_output(std::vector<std::shared_ptr<::cucascade::data_batch>> batches,
                      std::vector<std::size_t> partitions)
    : pipelineable_operator_data(std::move(batches)), partitions(std::move(partitions))
  {
  }

  std::vector<std::size_t> partitions;
};

//! Merge partitions for left batch i: its per-right-batch partials go to 2i and its left columns
//! to 2i + 1, so the merge can tell the two apart without inspecting schemas.
inline std::size_t topk_partials_partition(std::size_t left_ordinal) { return 2 * left_ordinal; }
inline std::size_t topk_left_partition(std::size_t left_ordinal) { return 2 * left_ordinal + 1; }

//! Select stage of the exact per-row top-k vector join. For each (left batch, right batch) pair it
//! finds every left row's k nearest rows within that right batch and emits a partial
//! [right cols..., distance] of exactly n_left * k rows (padded with NULL rows at +inf distance
//! when the right batch has fewer than k rows). The first pair of each left batch also forwards
//! the left batch's columns. VECTOR_TOPK_MERGE combines the partials into the final top-k.
class sirius_physical_vector_topk_join : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::VECTOR_TOPK_JOIN;

 public:
  sirius_physical_vector_topk_join(duckdb::unique_ptr<sirius_physical_operator> left,
                                   duckdb::unique_ptr<sirius_physical_operator> right,
                                   std::size_t left_vector_col_idx,
                                   std::size_t right_vector_col_idx,
                                   std::int64_t k,
                                   std::string metric,
                                   bool is_similarity,
                                   std::int64_t dim,
                                   duckdb::JoinType join_type,
                                   std::size_t estimated_cardinality);

  //! Column index of the FLOAT[dim] vector column within the left (probe) child's output.
  std::size_t left_vector_col_idx;
  //! Column index of the FLOAT[dim] vector column within the right (build) child's output.
  std::size_t right_vector_col_idx;
  //! Neighbors kept per left row.
  std::int64_t k;
  //! Distance metric: "l2" or "cosine".
  std::string metric;
  //! Ranked by cosine similarity: the merge reports `1 - cosine distance`.
  bool is_similarity;
  //! Fixed vector dimensionality (from the ARRAY<FLOAT> logical type).
  std::int64_t dim;
  //! INNER or LEFT; LEFT only differs when the right side is empty.
  duckdb::JoinType join_type;

 protected:
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

 public:
  bool is_source() const override { return true; }
  //! Always feeds VECTOR_TOPK_MERGE through a repository.
  bool is_sink() const override { return true; }
  //! Emits two schemas (partials and forwarded left columns), so skip the output-schema check.
  [[nodiscard]] bool declared_output_schema_is_runtime_schema() const noexcept override
  {
    return false;
  }

  //! Enumerates every (left batch, right batch) pair, one per task.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //! Right-side (build) concat feeds the "build" port; the left side feeds "default".
  [[nodiscard]] std::string_view input_port_for(
    sirius_physical_operator const& producer) const override;

  //! Pairs are independent; the merge does the cross-batch combine, so never partition.
  partition_strategy get_partition_strategy(const partition_sizing_input& in) override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  //! Routes each output batch to the merge partition chosen in execute().
  void sink(const operator_data& output_data, ::cuda::stream_ref stream) override;

 protected:
  std::mutex batches_to_processed_mutex;
  //! Batch-id lists per partition, populated once on the first dispatch.
  std::vector<std::vector<uint64_t>> left_batch_ids;
  std::vector<std::vector<uint64_t>> right_batch_ids;
  bool ids_initialized_ = false;
  //! Cursor over (partition, left batch, right batch).
  std::size_t cursor_partition_ = 0;
  std::size_t cursor_left_      = 0;
  std::size_t cursor_right_     = 0;
  //! Left batches numbered across partitions, in dispatch order.
  std::size_t next_left_ordinal_ = 0;
  std::unordered_map<uint64_t, std::size_t> left_ordinal_;
};

}  // namespace op
}  // namespace sirius
