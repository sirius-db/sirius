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
#include "op/sirius_physical_sort_sample.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace sirius::exchange {
class exchange_service;
}

namespace sirius::op {

/**
 * @brief SORT_SAMPLE on one rank of a range exchange: boundaries from every rank's sample.
 *
 * Every rank shares an evenly spaced summary of its sample, and merges all of them in rank order,
 * so every rank holds the same global sample. From it each derives the same world size minus one
 * boundaries for the range exchange after this operator, and its own partition boundaries from
 * the part of the sample inside its range. Every rank's boundary task joins the exchange, one
 * with no input included.
 *
 * distribute_plan() puts it in place of the plan's SORT_SAMPLE (from()). It keeps the base's TYPE,
 * so the plan and the task creator treat it as a SORT_SAMPLE.
 */
class sirius_physical_distributed_sort_sample final : public sirius_physical_sort_sample {
 public:
  struct distributed_sampling {
    std::shared_ptr<exchange::exchange_service> service;
    /// The all-gather's stream id, the same on every rank.
    exchange::stream_id stream{0};
    /// Where gathered summaries are merged.
    cucascade::memory::memory_space* gpu{nullptr};
    /// Receives the range exchange's boundaries.
    std::function<void(std::shared_ptr<cudf::table const>)> set_exchange_boundaries;
  };

  //! Rows of each rank's sample summary, at most, unless its partitions need more.
  static constexpr std::size_t distributed_summary_rows = 1024;
  //! Sample rows each of a rank's partitions gets from the global sample.
  static constexpr std::size_t summary_rows_per_partition = 32;

  sirius_physical_distributed_sort_sample(
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<duckdb::BoundOrderByNode> orders,
    std::size_t estimated_cardinality,
    distributed_sampling sampling,
    uint64_t sort_sample_bytes           = config::DEFAULT_SORT_SAMPLE_BYTES,
    uint64_t max_partition_bytes         = 0,
    double max_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION);

  /**
   * @brief Takes the place of a planned SORT_SAMPLE: its order, settings, child and physical
   * types move or copy over. @p local is left without orders or children, for its slot to drop.
   *
   * The caller re-points whatever held @p local, e.g. its SORT_PARTITION's sample.
   */
  static duckdb::unique_ptr<sirius_physical_distributed_sort_sample> from(
    sirius_physical_sort_sample& local, distributed_sampling sampling);

  //! The boundary task is still owed, input or not, until it has been claimed.
  [[nodiscard]] bool all_ports_empty() override;
  //! The boundary task runs with no input, since every rank's all-gather waits for it.
  //! get_next_task_input_data() has already claimed it (SCHEDULED) when the task creator asks.
  [[nodiscard]] bool runs_without_input() const override
  {
    return boundary_state() != BoundaryState::DONE;
  }

 protected:
  //! Shares @p sample_keys (sorted, possibly empty), then sets the range exchange's boundaries
  //! and returns this rank's, from the global sample; this rank keeps up to @p num_parts
  //! partitions.
  boundaries make_boundaries(cudf::table_view sample_keys,
                             std::size_t sample_bytes,
                             std::size_t num_parts,
                             cucascade::memory::memory_space* space,
                             ::cuda::stream_ref stream) override;

 private:
  distributed_sampling _sampling;
  //! The all-gather's result, kept for an OOM retry of the boundary task, since the peers will
  //! not gather on that stream again. Only one boundary task runs at a time.
  std::optional<std::vector<std::shared_ptr<::cucascade::data_batch>>> _gathered;
  //! This rank has entered the all-gather.
  bool _gather_entered = false;
};

}  // namespace sirius::op
