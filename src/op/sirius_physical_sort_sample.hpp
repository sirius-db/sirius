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
#include "sirius_config.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <atomic>
#include <memory>
#include <optional>
#include <vector>

namespace sirius {
namespace op {

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
    double max_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION);

  //! Order specification (copied from ORDER_BY) — determines which columns to sample
  duckdb::vector<duckdb::BoundOrderByNode> orders;

  //! Target bytes to accumulate for the boundary sample.
  [[nodiscard]] uint64_t sort_sample_bytes() const noexcept { return _sort_sample_bytes; }
  //! Max partition bytes, or 0 to size partitions from the GPU memory fraction.
  [[nodiscard]] size_t max_partition_bytes_override() const noexcept
  {
    return _max_partition_bytes_override;
  }
  //! Fraction of available GPU memory per partition.
  [[nodiscard]] double max_partition_memory_fraction() const noexcept
  {
    return _max_partition_memory_fraction;
  }

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

 protected:
  //! Partition boundaries, as make_boundaries() derives them.
  struct boundaries {
    std::size_t num_partitions = 1;
    //! num_partitions - 1 rows of the sort key columns; null for one partition.
    std::unique_ptr<cudf::table> table;
  };

  //! The boundaries for @p num_parts partitions from the sorted @p sample_keys (no columns and a
  //! null @p space with no input); the default picks evenly spaced rows. Runs again on a retry.
  virtual boundaries make_boundaries(cudf::table_view sample_keys,
                                     std::size_t sample_bytes,
                                     std::size_t num_parts,
                                     cucascade::memory::memory_space* space,
                                     ::cuda::stream_ref stream);

  /**
   * @brief Moves @p other's children here, and copies its planning state that no constructor
   * argument carries (physical types, alternate physical types, owning delim join).
   *
   * For a sample built to replace @p other in a plan. A new planning field of the operator goes
   * here, so no replacement drops it.
   */
  void take_planning_state_from(sirius_physical_sort_sample& other);

  [[nodiscard]] BoundaryState boundary_state() const noexcept
  {
    return _boundary_state.load(std::memory_order_acquire);
  }

  //! A device INT32 column holding @p indices.
  static std::unique_ptr<cudf::column> index_column(std::vector<int32_t> const& indices,
                                                    ::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr);

  //! Rows i * rows / parts of @p sorted for i in [1, parts), offset by @p first: parts - 1
  //! evenly spaced boundaries.
  static std::unique_ptr<cudf::table> evenly_spaced(cudf::table_view sorted,
                                                    cudf::size_type first,
                                                    cudf::size_type rows,
                                                    std::size_t parts,
                                                    ::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr);

 private:
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
