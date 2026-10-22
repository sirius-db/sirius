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

#include "exchange/transport.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace sirius::exchange {
class exchange_service;
}

namespace sirius::op {

/// How an EXCHANGE_SINK routes rows to ranks.
struct exchange_spec {
  enum class mode {
    hash,       ///< By a hash of key_columns.
    range,      ///< By key_columns against range boundaries, set before the first batch. Each
                ///< input batch must be sorted on key_columns, as ORDER_BY leaves it; each
                ///< slice stays sorted.
    broadcast,  ///< Every row to every rank.
    gather,     ///< Every row to rank `target`.
  };

  /// Distinct from PARTITION's seed, which partitions each rank's share afterwards.
  static constexpr std::uint32_t default_hash_seed = 0x9e3779b9u;

  mode routing = mode::hash;
  /// hash and range: the columns routed on.
  std::vector<int> key_columns;
  /// hash: per-key cast before hashing, as PARTITION's; empty = hash as-is.
  std::vector<cudf::data_type> key_cast_types;
  std::uint32_t hash_seed = default_hash_seed;
  /// range: the order of each key column.
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
  /// gather: the receiving rank.
  exchange::rank_t target = 0;
};

/**
 * @brief Sends its pipeline's output to the ranks of an exchange: the sending half of an
 * exchange, paired with an EXCHANGE_SOURCE of the same stream on every rank.
 *
 * Each batch is split by the spec and each slice sent to its rank; this rank's own slice is
 * handed over without a copy. When the pipeline finishes, every rank that may receive is told the
 * stream has ended: all ranks, or for a gather only the target.
 *
 * In a plan it is the child of its EXCHANGE_SOURCE, so it ends the pipeline of its producer
 * subtree, and that pipeline feeds the source's (dataless) port.
 */
class sirius_physical_exchange_sink : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::EXCHANGE_SINK;

  /// @throws sirius::invalid_input_exception on a spec that does not fit @p types.
  sirius_physical_exchange_sink(duckdb::vector<sirius::logical_type> types,
                                std::size_t estimated_cardinality,
                                std::shared_ptr<exchange::exchange_service> service,
                                exchange::stream_id stream,
                                exchange_spec spec);

  bool is_sink() const override { return true; }

  [[nodiscard]] exchange::stream_id stream() const noexcept { return _stream; }
  [[nodiscard]] exchange_spec const& spec() const noexcept { return _spec; }

  /// range: the world size - 1 boundary rows, over key_columns only, ascending in the spec's
  /// order; rank i gets rows from boundary i - 1 (inclusive) to boundary i (exclusive).
  void set_range_boundaries(std::shared_ptr<cudf::table const> boundaries);

  /// Splits each batch into its ranks' slices, as routed_operator_data. Done here rather than in
  /// sink(): an OOM here reschedules the task, while one in sink() fails the query, since sink()
  /// may already have sent earlier slices.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  /// Sends each slice to its rank.
  void sink(const operator_data& input_data, ::cuda::stream_ref stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

 protected:
  /// Ends the stream to every rank that may receive it.
  void on_finalize_operator() override;

 private:
  /// The slices of @p batch, one per rank; empty for a rank that gets none of it.
  std::vector<std::shared_ptr<cucascade::data_batch>> route(
    cucascade::read_only_data_batch const& batch,
    std::shared_ptr<cucascade::data_batch> const& handle,
    ::cuda::stream_ref stream);

  std::shared_ptr<exchange::exchange_service> _service;
  exchange::stream_id _stream;
  exchange_spec _spec;
  std::mutex _boundaries_mutex;
  std::shared_ptr<cudf::table const> _boundaries;
};

}  // namespace sirius::op
