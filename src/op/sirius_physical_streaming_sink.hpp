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

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
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

  partition_spec() = default;

  /// Hash-mode shorthand: partition_spec({col}, {cast}) without naming the mode.
  partition_spec(std::vector<int> keys, std::vector<cudf::data_type> casts = {})
    : key_columns(std::move(keys)), key_cast_types(std::move(casts))
  {
  }
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
                                         ::cuda::stream_ref stream) override;

  /// N=1: native push. N>1: hash-partition; skip empty slices. Refused push = silent drop.
  /// Hash mode partitions each input batch in row pieces of at most the max piece bytes,
  /// all of them before publishing any, so an OOM publishes nothing and can be retried.
  void sink(const operator_data& input_data, ::cuda::stream_ref stream) override;

  /// Hash mode publishes only once every piece is partitioned.
  [[nodiscard]] bool sink_publishes_on_success_only() const override
  {
    return _outputs.size() > 1 && _spec.mode == partition_mode::hash;
  }

  /// The largest piece of an input batch hash_partition sees at once: it peaks at about three
  /// times its input (the input, a reordered copy, the per-destination slices).
  static constexpr std::size_t default_max_piece_bytes = std::size_t{256} << 20;

  /// The piece size for a GPU space of @p pool_bytes: 1/192 of it, within [32 MiB,
  /// default_max_piece_bytes], so four pipeline threads' peaks stay near 6% of the pool on a
  /// 32 GB GPU as on a 189 GB one. 0 (an unknown size) gives default_max_piece_bytes.
  [[nodiscard]] static std::size_t piece_bytes_for_pool(std::size_t pool_bytes) noexcept
  {
    if (pool_bytes == 0) { return default_max_piece_bytes; }
    return std::clamp<std::size_t>(
      pool_bytes / 192, std::size_t{32} << 20, default_max_piece_bytes);
  }

  /// Overrides piece_bytes_for_pool (tests).
  void set_max_piece_bytes(std::size_t bytes)
  {
    _max_piece_bytes = std::max<std::size_t>(1, bytes);
  }

  /// Hash mode: each destination's slices are held until they add up to @p bytes, then
  /// published as one batch, so an exchange ships and spills fewer, larger batches. 0 (the
  /// default, unless SIRIUS_CN_EXCHANGE_BATCH_BYTES says otherwise) publishes every slice.
  void set_combine_bytes(std::size_t bytes) { _combine_bytes = bytes; }

  /// The row ranges [begin, end) an input of @p rows rows and @p bytes bytes is partitioned in,
  /// each at most @p max_piece bytes (one range when it already fits, or on one row).
  [[nodiscard]] static std::vector<std::pair<std::size_t, std::size_t>> partition_pieces(
    std::size_t rows, std::size_t bytes, std::size_t max_piece);

  /// 0 if N==1; when partitioned, the slices of the whole input plus hash_partition's reorder of
  /// one piece (~2x for an input of one piece).
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

  /// The stream behind output `index`, for a consumer that drains it from its own thread while
  /// the fragment runs.
  /// @throws sirius::invalid_input_exception when `index` is out of range.
  [[nodiscard]] std::shared_ptr<exec::batch_stream> output_stream(std::size_t index) const;

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

  /// 0 follows the GPU space's size (piece_bytes_for_pool).
  std::size_t _max_piece_bytes = 0;

 private:
  /// Publishes @p batches to output @p index, as one batch when there are several and the
  /// GPU has room to combine them, else one by one.
  void publish_combined(std::size_t index,
                        std::vector<std::shared_ptr<cucascade::data_batch>> batches,
                        ::cuda::stream_ref stream);

  std::size_t _combine_bytes;
  std::mutex _pending_mutex;
  /// Per output, slices held until they reach _combine_bytes, and their bytes.
  std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> _pending;
  std::vector<std::size_t> _pending_bytes;
};

}  // namespace sirius::op
