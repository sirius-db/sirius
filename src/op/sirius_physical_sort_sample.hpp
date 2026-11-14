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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_buffer.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
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

  ~sirius_physical_sort_sample() override;

  //! Order specification (copied from ORDER_BY) — determines which columns to sample
  duckdb::vector<duckdb::BoundOrderByNode> orders;

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

  /**
   * @brief The partition boundaries (P-1 rows, sort key columns only) as a table on the caller's
   * GPU
   *
   * The boundaries are computed once, on the GPU that ran the boundary task, and kept in host
   * memory. Each GPU that partitions batches gets its own device copy, uploaded on first use and
   * complete (its stream synchronised) before any caller sees it, so the returned view can be read
   * by kernels on @p space 's device from any stream. The view stays valid until
   * clear_partition_boundaries().
   *
   * Must be called with @p space 's device current, after boundaries_computed() returned true and
   * only when get_num_partitions() > 1.
   *
   * @param space  GPU memory space of the batch being partitioned; the copy is allocated from it
   * @param stream Stream used for the one-time upload
   */
  cudf::table_view get_partition_boundaries(cucascade::memory::memory_space& space,
                                            ::cuda::stream_ref stream);

  //! Number of boundary rows (get_num_partitions() - 1, or 0 for a single partition)
  size_t num_boundary_rows() const { return _host_boundaries ? _host_boundaries->num_rows : 0; }

  //! Get the computed number of partitions
  size_t get_num_partitions() const { return _num_partitions; }

  //! Whether boundaries have been computed
  bool boundaries_computed() const
  {
    return _boundary_state.load(std::memory_order_acquire) == BoundaryState::DONE;
  }

  //! Release the host copy of the partition boundaries and every per-GPU copy
  void clear_partition_boundaries();

 private:
  //! Host copy of the boundary rows in cudf::pack layout, written once before DONE is published
  struct host_boundaries {
    std::vector<uint8_t> metadata;
    std::vector<uint8_t> data;
    size_t num_rows = 0;
  };

  //! One GPU's copy of the boundary rows; `view` points into `data`
  struct device_boundaries {
    rmm::device_buffer data;
    cudf::table_view view;
  };

  //! Partition boundary rows (P-1 rows containing sort key column values), immutable once
  //! _boundary_state is DONE and until clear_partition_boundaries()
  std::unique_ptr<host_boundaries> _host_boundaries;

  //! Per-GPU copies keyed by device id, created lazily under _device_boundaries_mutex
  std::mutex _device_boundaries_mutex;
  std::map<int, std::unique_ptr<device_boundaries>> _device_boundaries;

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
