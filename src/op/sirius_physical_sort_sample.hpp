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

#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sort_comparison_config.hpp"
#include "sirius_config.hpp"

#include <cudf/table/table.hpp>

#include <cucascade/data/data_repository_manager.hpp>

#include <atomic>

namespace sirius {
namespace op {

class sort_sample_input : public pipelineable_operator_data {
 public:
  explicit sort_sample_input(std::vector<std::shared_ptr<cucascade::data_batch>> batches)
    : pipelineable_operator_data(std::move(batches))
  {
  }
  std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
    std::vector<std::shared_ptr<cucascade::data_batch>> batches) const override
  {
    auto result = std::make_unique<sort_sample_input>(std::move(batches));
    result->adopt_row_identity(*this);
    return result;
  }
};

// Boundary locks travel with the sample output until partitioning has completed.
class sort_sample_output : public pipelineable_operator_data {
 public:
  sort_sample_output(std::vector<std::shared_ptr<cucascade::data_batch>> batches,
                     std::optional<cucascade::read_only_data_batch> boundaries = std::nullopt)
    : pipelineable_operator_data(std::move(batches)), boundaries(std::move(boundaries))
  {
  }
  std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
    std::vector<std::shared_ptr<cucascade::data_batch>> batches) const override
  {
    auto result = std::make_unique<sort_sample_output>(std::move(batches), boundaries);
    result->adopt_row_identity(*this);
    return result;
  }
  std::optional<cucascade::read_only_data_batch> boundaries;
};

class sirius_physical_sort_sample : public sirius_physical_operator {
 public:
  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }

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
    double max_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION,
    std::shared_ptr<const sort_comparison_config> comparison_p = nullptr,
    uint64_t chunk_bytes                                       = config::DEFAULT_SORT_CHUNK_BYTES);

  //! Order specification (copied from ORDER_BY) — determines which columns to sample
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  std::shared_ptr<const sort_comparison_config> comparison;

 public:
  bool is_source() const override { return true; }
  // SORT_SAMPLE is not a pipeline sink — it runs as an intermediate operator in the
  // SORT_PARTITION pipeline, so the tree-based build matches the legacy 3-pipeline shape.
  bool is_sink() const override { return false; }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  //! Override to wait for enough sample bytes before returning READY
  std::optional<task_creation_hint> get_next_task_hint() override;

  //! Override to pull sample bytes before boundary computation, then one batch at a time
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  void register_boundaries(cucascade::shared_data_repository_manager& manager);

  //! Get the computed number of partitions
  size_t get_num_partitions() const { return _num_partitions; }

  //! Target bytes of one range partition (the merge unit); valid once boundaries are computed
  size_t get_partition_bytes() const { return _partition_bytes; }

  //! Whether boundaries have been computed
  bool boundaries_computed() const
  {
    return _boundary_state.load(std::memory_order_acquire) == BoundaryState::DONE;
  }

  void clear_partition_boundaries();

 private:
  cucascade::shared_data_repository* _boundary_repo = nullptr;
  std::shared_ptr<cucascade::data_batch> _partition_boundaries;

  //! Number of partitions computed from the sample
  size_t _num_partitions = 1;

  //! Target bytes per range partition, set with _num_partitions
  size_t _partition_bytes = 0;

  //! Output chunk target of the sort; the floor of the partition target
  uint64_t _chunk_bytes = config::DEFAULT_SORT_CHUNK_BYTES;

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
