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

/// How a partitioned sink routes rows across its output streams.
///
/// Only the *function* lives here. Which compute node each partition ships to is the wrapper's
/// routing table, and N is `output_repositories.size()` — the sink stays oblivious to both.
struct partition_spec {
  /// Column indices hashed to pick a destination. Must be non-empty for a partitioned sink.
  std::vector<int> key_columns;

  /// Per-key cast applied before hashing, so keys that differ only in representation (INT32 vs
  /// INT64) still land together. Empty means "hash every key as-is".
  std::vector<cudf::data_type> key_cast_types;
};

/// Terminal operator of a streaming fragment: every batch its pipeline produces is pushed into
/// one `exec::batch_stream` per destination, where an external consumer pulls it.
///
/// Batches are exposed natively — in whatever tier they currently sit — so nothing is
/// materialized or converted to Arrow on the way out, and a queued batch stays spillable by the
/// downgrade executor until it is pulled.
///
/// Unlike the source, the sink arms no re-arm waker: its consumer is an external thread blocking
/// in `wait()`, not an engine task that needs re-nominating.
class sirius_physical_streaming_sink : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::STREAMING_SINK;

  /// The pipeline feeding this sink is its one and only sender.
  static constexpr exec::sender_id_t PIPELINE_SENDER = 0;

  /// Single-destination sink: one output stream, no partitioning. The N = 1 case of the
  /// constructor below.
  ///
  /// @throws sirius::invalid_input_exception when `output_repository` is null.
  sirius_physical_streaming_sink(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::shared_ptr<cucascade::shared_data_repository> output_repository);

  /// Partitioned sink: N output streams, one per destination, each with its own repository.
  /// Every batch is GPU-hash-partitioned by `spec` and slice *i* is pushed into
  /// `output_repositories[i]`.
  ///
  /// @throws sirius::invalid_input_exception when `output_repositories` is empty or contains a
  ///         null entry, or when N > 1 and `spec.key_columns` is empty (nothing to route by).
  sirius_physical_streaming_sink(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::vector<std::shared_ptr<cucascade::shared_data_repository>> output_repositories,
    partition_spec spec);

  bool is_sink() const override { return true; }

  //! The sink is appended to `current` (landing at operators[0] after the reverse) and set as
  //! the sink of its own child meta pipeline. The base sink path skips the append, which leaves
  //! the terminal operator out of the root pipeline and the source with nothing driving it.
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  // -----------------------------------------------------------------------
  // Producer side (engine)
  // -----------------------------------------------------------------------

  /// Pass-through. The executor runs every operator in the chain, terminal sink included, and
  /// feeds the chain's result back into sink(). The base implementation returns an empty batch
  /// list, so without this override the sink receives nothing and the pipeline "succeeds" with
  /// an empty output stream.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  /// Publish this task's output batches.
  ///
  /// With one destination the batches go into the output stream natively — no Arrow, no GPU
  /// upgrade, no copy. With N > 1 each batch is GPU-hash-partitioned by the partition spec and
  /// slice *i* is pushed into `_outputs[i]`; empty slices are skipped. A push after
  /// end-of-stream is refused by the stream rather than landing behind a consumer that already
  /// saw EOS.
  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

  /// 0 with one destination — the push allocates nothing on top of the input this task already
  /// materialized — and 2× the input bytes when partitioning, because hash_partition() holds a
  /// full reordered copy and the per-partition copies at the same time. The caller maxes this
  /// across the pipeline's operators, so an inflated single-destination answer would raise every
  /// cold-start reservation.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  // -----------------------------------------------------------------------
  // Consumer side (external wrapper / session, never an engine task)
  // -----------------------------------------------------------------------

  /// Number of output streams this sink exposes. One per destination.
  [[nodiscard]] std::size_t num_output_streams() const { return _outputs.size(); }

  /// Non-blocking pull of the next batch of output stream `index`.
  /// @return nullopt when nothing is available right now — which is *not* the same as EOS; ask
  ///         `drained(index)` to tell the two apart.
  /// @throws sirius::invalid_input_exception when `index` is out of range.
  std::optional<std::shared_ptr<cucascade::data_batch>> pull(std::size_t index = 0);

  /// Block until output stream `index` has a batch to pull or the stream has ended.
  /// External threads only.
  void wait(std::size_t index = 0);

  /// True when the pipeline has finished AND output stream `index` is empty.
  [[nodiscard]] bool drained(std::size_t index = 0) const;

  /// Non-blocking classification of output stream `index`: HAS_DATA / WAITING / END_OF_STREAM.
  [[nodiscard]] exec::batch_stream::availability availability(std::size_t index = 0) const;

  /// The query died: poison every output stream. External consumers unblock from wait() and
  /// collect the rethrow from pull() — never a clean end. First failure wins.
  void fail_output(std::exception_ptr error);

 protected:
  /// The pipeline finishing is this stream's end-of-stream: it is the single expected sender.
  void on_finalize_operator() override;

  /// @throws sirius::invalid_input_exception when `index` is out of range.
  void validate_index(std::size_t index) const;

  /// One stream per destination; positional with the caller's repository list.
  std::vector<std::shared_ptr<exec::batch_stream>> _outputs;

  /// Routing spec. Empty `key_columns` is valid only when `_outputs.size() == 1`, in which
  /// case `sink()` short-circuits to a single native push without invoking the hash-partition
  /// kernel at all — the spec is irrelevant, and there is nothing to route.
  partition_spec _spec;
};

}  // namespace sirius::op
