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

#include "op/sirius_physical_operator.hpp"
#include "op/window/cumulative.hpp"
#include "op/window/framed.hpp"
#include "op/window/ranking.hpp"
#include "op/window/whole_partition.hpp"

#include <cstdint>
#include <variant>
#include <vector>

namespace sirius::op {

/// The input a window needs, and so the exchange the planner puts under it.
enum class window_exchange : std::uint8_t {
  /// Complete hash buckets from PARTITION → CONCAT, unsorted.
  hash,
  /// Complete groups from MERGE_SORT, sorted on the partition keys followed by the order keys.
  sorted,
};

/**
 * @brief Pipelineable adapter that evaluates window functions over complete groups of partitions.
 *
 * A window evaluates one of: aggregates over the whole partition, ranking functions over the
 * window ORDER BY, running aggregates over the window ORDER BY (the frame `ROWS BETWEEN UNBOUNDED
 * PRECEDING AND CURRENT ROW`), or aggregates over other frames of the window ORDER BY (framed
 * mode). The child must publish batches that each hold every row of the partitions they contain.
 *
 * Whole-partition, ranking and running windows read hash buckets: PARTITION and CONCAT do the
 * routing and apply the FULL barrier, and each batch is one bucket.
 *
 * A framed window reads the ranges of a distributed sort on the partition keys followed by the
 * order keys. Range boundaries are cut on the partition keys only, so a partition is never split
 * between ranges, and each batch is one merged range, already sorted. MERGE_SORT must publish a
 * range as a single batch: a range divided between batches would split a partition. A range can
 * be windowed as soon as MERGE_SORT publishes it, so the input port is a PIPELINE barrier. Range
 * sizing is only as good as the sort's sample and the estimated cardinality, and a range that
 * does not fit fails in MERGE_SORT before the window runs.
 *
 * A task evaluates one batch, a bucket or a merged range, as one cuDF table on one GPU. A batch
 * holds every row of each partition it contains, so a single partition is never split. Limits on
 * a batch:
 * - Its row count must stay below the `cudf::size_type` limit (2^31 - 1). CONCAT rejects a bucket
 *   above it. A merged range is one cuDF table, so it is under the limit when it arrives.
 * - Its input and results must fit in the GPU's memory. `execute` rejects a batch whose known
 *   lower bound exceeds the memory space's maximum.
 *
 * A partition larger than either limit is not handled by this operator.
 */
class sirius_physical_window : public sirius_physical_operator {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::WINDOW;

  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    // Neither hash buckets nor merged ranges, which are windowed as they arrive, preserve the
    // input's global sequence.
    return sirius::OrderPreservationType::NO_ORDER;
  }

  /// A window of aggregates over the whole partition, evaluated by @p plan. The plan's group
  /// columns are the partition columns.
  sirius_physical_window(duckdb::vector<logical_type> output_types,
                         duckdb::vector<logical_type> prepared_types,
                         std::size_t input_width,
                         whole_partition_plan plan,
                         std::size_t estimated_cardinality);

  /// A window of ranking functions over the window ORDER BY.
  sirius_physical_window(duckdb::vector<logical_type> output_types,
                         duckdb::vector<logical_type> prepared_types,
                         std::size_t input_width,
                         std::vector<cudf::size_type> partition_columns,
                         window_order order,
                         std::vector<ranking_function> ranking_functions,
                         std::size_t estimated_cardinality);

  /// A window of running aggregates over the window ORDER BY.
  sirius_physical_window(duckdb::vector<logical_type> output_types,
                         duckdb::vector<logical_type> prepared_types,
                         std::size_t input_width,
                         std::vector<cudf::size_type> partition_columns,
                         window_order order,
                         std::vector<window_aggregate> cumulative_aggregates,
                         std::size_t estimated_cardinality);

  /// A window of framed aggregates over input sorted on the partition and order keys, evaluated by
  /// @p plan. The plan's group columns are the partition columns and must not be empty.
  sirius_physical_window(duckdb::vector<logical_type> output_types,
                         duckdb::vector<logical_type> prepared_types,
                         std::size_t input_width,
                         framed_window_plan plan,
                         std::size_t estimated_cardinality);

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(operator_data const& input,
                                         ::cuda::stream_ref stream) override;
  std::size_t no_history_peak_memory_estimate(input_stats const& stats) const override;

  /// The indices of the partition key columns in the prepared input.
  std::vector<cudf::size_type> const& partition_columns() const { return _partition_columns; }

  /// The exchange this window's input must come from.
  [[nodiscard]] window_exchange input_exchange() const;

  /// The window ORDER BY. Empty for a window of aggregates over the whole partition.
  [[nodiscard]] window_order const& order() const;

  /// A sorted window takes merged ranges as soon as they are published.
  [[nodiscard]] MemoryBarrierType input_barrier_for(
    sirius_physical_operator const& producer) const override;

 private:
  struct ranking_mode {
    window_order order;
    std::vector<ranking_function> functions;
  };

  struct cumulative_mode {
    window_order order;
    std::vector<window_aggregate> aggregates;
  };

  /// Cold-start estimate of the memory a task needs for a bucket.
  std::size_t workspace_bytes(std::size_t rows, std::size_t input_bytes) const;
  duckdb::vector<logical_type> _prepared_types;
  std::size_t _input_width;
  std::vector<cudf::size_type> _partition_columns;
  std::variant<whole_partition_plan, ranking_mode, cumulative_mode, framed_window_plan> _mode;
};

}  // namespace sirius::op
