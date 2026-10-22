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

#include "exchange/transport.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"
#include "sirius_config.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace sirius::exchange {
class exchange_service;
}

namespace sirius {
namespace op {

class sirius_physical_sort_sample : public sirius_physical_operator {
 public:
  enum class BoundaryState : uint8_t { NOT_DONE, SCHEDULED, DONE };

  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::SORT_SAMPLE;

  sirius_physical_sort_sample(
    sirius_physical_order* order_by,
    uint64_t sort_sample_bytes           = config::DEFAULT_SORT_SAMPLE_BYTES,
    uint64_t max_partition_bytes         = 0,
    double max_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION);

  sirius_physical_sort_sample(
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<duckdb::BoundOrderByNode> orders,
    std::size_t estimated_cardinality,
    uint64_t sort_sample_bytes           = config::DEFAULT_SORT_SAMPLE_BYTES,
    uint64_t max_partition_bytes         = 0,
    double max_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION);

  //! Order specification (copied from ORDER_BY) — determines which columns to sample
  duckdb::vector<duckdb::BoundOrderByNode> orders;

  /**
   * @brief Sampling across the ranks of a range exchange.
   *
   * Every rank shares an evenly spaced summary of its sample, and merges all of them in rank
   * order, so every rank holds the same global sample. From it each derives the same world size
   * - 1 boundaries for the range exchange after this operator, and its own partition boundaries
   * from the part of the sample inside its range. Every rank's boundary task joins the exchange,
   * one with no input included.
   */
  struct distributed_sampling {
    std::shared_ptr<exchange::exchange_service> service;
    /// The all-gather's stream id, the same on every rank.
    exchange::stream_id stream{0};
    /// Where gathered summaries are merged.
    cucascade::memory::memory_space* gpu{nullptr};
    /// Receives the range exchange's boundaries.
    std::function<void(std::shared_ptr<cudf::table const>)> set_exchange_boundaries;
  };
  void set_distributed(distributed_sampling sampling) { _distributed = std::move(sampling); }

  //! Rows of each rank's sample summary shared in distributed mode.
  static constexpr std::size_t distributed_summary_rows = 1024;

 public:
  bool is_source() const override { return true; }
  // SORT_SAMPLE is not a pipeline sink — it runs as an intermediate operator in the
  // SORT_PARTITION pipeline, so the tree-based build matches the legacy 3-pipeline shape.
  bool is_sink() const override { return false; }
  bool sink_order_dependent() const override { return false; }

  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  //! Override to wait for enough sample bytes before returning READY
  std::optional<task_creation_hint> get_next_task_hint() override;

  //! Override to pull sample bytes before boundary computation, then one batch at a time
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //! Distributed: the boundary task is still owed, input or not, until it has been claimed.
  [[nodiscard]] bool all_ports_empty() override;
  //! get_next_task_input_data() has already claimed the task (SCHEDULED) when this is asked.
  [[nodiscard]] bool runs_without_input() const override
  {
    return _distributed.has_value() &&
           _boundary_state.load(std::memory_order_acquire) != BoundaryState::DONE;
  }

  //! Get the computed partition boundaries (P-1 rows, sort key columns only)
  const cudf::table& get_partition_boundaries() const { return *_partition_boundaries; }

  //! Get the computed number of partitions
  size_t get_num_partitions() const { return _num_partitions; }

  //! Whether boundaries have been computed
  bool boundaries_computed() const
  {
    return _boundary_state.load(std::memory_order_acquire) == BoundaryState::DONE;
  }

  //! Release the partition boundaries table to free GPU memory
  void clear_partition_boundaries() { _partition_boundaries.reset(); }

 private:
  //! Distributed mode: shares @p sample_keys (sorted, possibly empty), then sets both sets of
  //! boundaries from the global sample; this rank keeps up to @p num_parts partitions.
  void compute_distributed_boundaries(cudf::table_view sample_keys,
                                      std::size_t num_parts,
                                      std::vector<cudf::order> const& column_order,
                                      std::vector<cudf::null_order> const& null_precedence,
                                      ::cuda::stream_ref stream);

  std::optional<distributed_sampling> _distributed;
  //! Partition boundary rows (P-1 rows containing sort key column values)
  std::unique_ptr<cudf::table> _partition_boundaries;

  //! Number of partitions computed from the sample
  size_t _num_partitions = 1;

  //! Boundary computation lifecycle: NOT_DONE → SCHEDULED (input claimed) → DONE.
  //! get_next_task_input_data() transitions to SCHEDULED under the task-creation lock;
  //! execute() completes the transition to DONE. On OOM the state stays SCHEDULED so the
  //! rescheduled task retries while task_creator does not spawn a duplicate boundary task.
  std::atomic<BoundaryState> _boundary_state{BoundaryState::NOT_DONE};

  //! Target bytes to accumulate for the boundary sample (from sirius_config operator_params)
  uint64_t _sort_sample_bytes = config::DEFAULT_SORT_SAMPLE_BYTES;

  //! Override for max partition bytes (0 = use GPU memory fraction)
  size_t _max_partition_bytes_override = 0;

  //! Fraction of available GPU memory per partition (from sirius_config operator_params)
  double _max_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION;
};

}  // namespace op
}  // namespace sirius
