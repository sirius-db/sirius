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

#include <cudf/table/table.hpp>

#include <atomic>

namespace sirius {
namespace op {

class sirius_physical_sort_sample : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::SORT_SAMPLE;

  static constexpr duckdb::idx_t DEFAULT_NUM_SAMPLE_BATCHES = 5;

  //! Maximum fraction of available GPU memory per partition (33%)
  static constexpr double MAX_PARTITION_MEMORY_FRACTION = 0.33;

  sirius_physical_sort_sample(sirius_physical_order* order_by);

  sirius_physical_sort_sample(duckdb::vector<duckdb::LogicalType> types,
                              duckdb::vector<duckdb::BoundOrderByNode> orders,
                              duckdb::idx_t estimated_cardinality,
                              duckdb::idx_t num_sample_batches = DEFAULT_NUM_SAMPLE_BATCHES);

  //! Order specification (copied from ORDER_BY) — determines which columns to sample
  duckdb::vector<duckdb::BoundOrderByNode> orders;

  //! Number of batches to sample before computing partition boundaries
  duckdb::idx_t num_sample_batches;

 public:
  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }
  bool sink_order_dependent() const override { return false; }

  duckdb::OrderPreservationType source_order() const override
  {
    return duckdb::OrderPreservationType::FIXED_ORDER;
  }

  operator_data execute(const operator_data& input_data,
                        rmm::cuda_stream_view stream = cudf::get_default_stream()) override;

  //! Override to wait for N batches before returning READY
  std::optional<task_creation_hint> get_next_task_hint() override;

  //! Get the computed partition boundaries (P-1 rows, sort key columns only)
  const cudf::table& get_partition_boundaries() const { return *_partition_boundaries; }

  //! Get the computed number of partitions
  size_t get_num_partitions() const { return _num_partitions; }

  //! Whether boundaries have been computed
  bool boundaries_computed() const { return _boundaries_computed.load(); }

  //! Override the maximum bytes per partition (0 = use default GPU memory-based calculation)
  void set_max_partition_bytes(size_t bytes) { _max_partition_bytes_override = bytes; }

 private:
  //! Partition boundary rows (P-1 rows containing sort key column values)
  std::unique_ptr<cudf::table> _partition_boundaries;

  //! Number of partitions computed from the sample
  size_t _num_partitions = 1;

  //! Whether partition boundaries have been computed
  std::atomic<bool> _boundaries_computed{false};

  //! Override for max partition bytes (0 = use default GPU memory-based calculation)
  size_t _max_partition_bytes_override = 0;
};

}  // namespace op
}  // namespace sirius
