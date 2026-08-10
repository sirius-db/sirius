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

#include "exec/batch_stream.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cudf/types.hpp>

#include <cucascade/data/data_repository.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <vector>

namespace sirius::op {

/// How a partitioned sink routes rows across its output streams. Destination nodes are the
/// wrapper's routing table — the sink stays oblivious.
enum class partition_mode {
  hash,       ///< GPU-hash-partition by key_columns; key_columns must be non-empty.
  broadcast,  ///< Replicate every batch to all N outputs; key_columns must be empty.
};

struct partition_spec {
  partition_mode mode = partition_mode::hash;

  /// Hashed to pick a destination. Must be non-empty for hash mode; must be empty for broadcast.
  std::vector<int> key_columns;

  /// Per-key cast before hashing (e.g. INT32 vs INT64). Empty = hash as-is. Ignored in broadcast.
  std::vector<cudf::data_type> key_cast_types;
};

/// Fragment terminal: push pipeline output into one batch_stream per destination.
/// No on_data — consumer blocks in wait(); no channel-level backpressure.
class sirius_physical_streaming_sink : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::STREAMING_SINK;

  /// The pipeline feeding this sink is its only sender.
  static constexpr exec::sender_id_t PIPELINE_SENDER = 0;

  /// Single-destination sink (N = 1): identity push, no partitioning.
  /// @throws sirius::invalid_input_exception when `output_repository` is null.
  sirius_physical_streaming_sink(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::shared_ptr<cucascade::shared_data_repository> output_repository);

  /// N destinations: GPU-hash-partition each batch; slice i → output_repositories[i].
  /// @throws sirius::invalid_input_exception on empty/null repos, N>1 with no keys, cast/key
  ///         mismatch, or out-of-range key column.
  sirius_physical_streaming_sink(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::vector<std::shared_ptr<cucascade::shared_data_repository>> output_repositories,
    partition_spec spec);

  bool is_sink() const override { return true; }

  //! Append into operators[] (base sink path skips this). Membership required for finalize/EOS;
  //! lands at operators.back() after is_ready() reverses.
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  // -----------------------------------------------------------------------
  // Producer side (engine)
  // -----------------------------------------------------------------------

  /// Pass-through so sink() sees the batches; base returns empty.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  /// N=1: native push. N>1: hash-partition; skip empty slices. Refused push = silent drop.
  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

  /// 0 if N==1; ~2× when partitioned (hash_partition holds reorder + slices).
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  // -----------------------------------------------------------------------
  // Consumer side (external wrapper / session, never an engine task)
  // -----------------------------------------------------------------------

  [[nodiscard]] std::size_t num_output_streams() const { return _outputs.size(); }

  /// @return nullopt means nothing now, not EOS — use drained(index).
  /// @throws sirius::invalid_input_exception when `index` is out of range.
  /// @throws pending producer error ahead of queued batches (S4).
  std::optional<std::shared_ptr<cucascade::data_batch>> pull(std::size_t index = 0);

  /// Block until HAS_DATA or ended (S5 with pull: re-check after wake). External threads only.
  /// @throws sirius::invalid_input_exception when `index` is out of range.
  void wait(std::size_t index = 0);

  /// Clean end only (S3). Poisoned streams end by rethrow from pull().
  /// @throws sirius::invalid_input_exception when `index` is out of range.
  [[nodiscard]] bool drained(std::size_t index = 0) const;

  /// @throws sirius::invalid_input_exception when `index` is out of range.
  [[nodiscard]] exec::batch_stream::availability availability(std::size_t index = 0) const;

  /// Poison every output stream (S2 / P1–P4). First failure wins. Sink has no on_data — wait()
  /// wakes via the stream CV.
  void fail_output(std::exception_ptr error);

 protected:
  /// Pipeline completion = sender-set EOS (PIPELINE_SENDER).
  void on_finalize_operator() override;

  /// @throws sirius::invalid_input_exception when `index` is out of range.
  void validate_index(std::size_t index) const;

  /// One stream per destination; positional with the caller's repository list.
  std::vector<std::shared_ptr<exec::batch_stream>> _outputs;

  /// Empty key_columns is valid only for N==1 (native push, no hash_partition).
  partition_spec _spec;
};

}  // namespace sirius::op
