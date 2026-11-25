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

#include "op/cross_rank_sort_boundaries.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sort_key_sample.hpp"
#include "sirius_config.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace sirius {
namespace op {

class sirius_physical_sort_sample : public sirius_physical_operator {
 public:
  struct range_sample_stats {
    bool complete_input               = false;
    std::size_t sample_batches        = 0;
    std::size_t sample_rows           = 0;
    std::size_t sample_key_rows       = 0;
    std::size_t sample_stride         = 1;
    std::size_t sample_bytes          = 0;
    std::size_t estimated_total_bytes = 0;
    std::string_view est_source       = "planner";
    std::size_t target_range_bytes    = 0;
    std::size_t num_ranges            = 1;
    // Unset if counting failed with an error the task would retry on.
    std::optional<std::size_t> repeated_boundaries = 0;
  };

  [[nodiscard]] range_sample_stats const& get_range_sample_stats() const noexcept
  {
    return _range_stats;
  }
  [[nodiscard]] std::uint64_t get_input_batch_count() const noexcept
  {
    return _input_batches.load(std::memory_order_relaxed);
  }
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

  //! Input bytes to see before the boundaries are chosen; 0 waits for all the input. Input that
  //! ends first is used whole.
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

  /**
   * @brief The key rows the boundaries are picked from.
   *
   * The SORT_KEY_SAMPLE that ends the producing pipeline adds the key rows of every sorted batch
   * before the batch reaches this operator's input port. Nothing is kept once the boundaries are
   * chosen.
   */
  [[nodiscard]] std::shared_ptr<sort_key_sample> const& key_sample() const noexcept
  {
    return _key_sample;
  }

  //! The input whose key rows have been added so far.
  [[nodiscard]] sort_key_sample::totals sampled_input() const { return _key_sample->seen(); }

  //! Cap on the key rows kept for the boundaries. Set while planning, before key_sample() is
  //! given to the SORT_KEY_SAMPLE.
  //! @throws std::logic_error once key_sample() is shared, a batch was added or the boundary task
  //! was claimed.
  void set_max_sample_key_rows(std::size_t rows)
  {
    if (boundary_state() != BoundaryState::NOT_DONE || _key_sample.use_count() != 1 ||
        _key_sample->seen().batches != 0) {
      throw std::logic_error(
        "[sirius_physical_sort_sample::set_max_sample_key_rows] the plan cannot change during "
        "execution");
    }
    _key_sample = std::make_shared<sort_key_sample>(rows);
  }

  //! READY for the boundary task once the input has ended, or once key rows of sort_sample_bytes()
  //! of it were added when that is not 0; afterwards as the base class.
  std::optional<task_creation_hint> get_next_task_hint() override;

  //! One batch for the boundary task, which reads its sample from key_sample(), then one batch at
  //! a time.
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

  //! Boundaries come from @p hook instead of this operator's sample alone. Set while planning.
  //! @throws std::logic_error once the boundary task has been claimed.
  void set_cross_rank_boundaries(std::shared_ptr<cross_rank_sort_boundaries> hook)
  {
    if (boundary_state() != BoundaryState::NOT_DONE) {
      throw std::logic_error(
        "[sirius_physical_sort_sample::set_cross_rank_boundaries] the plan cannot change during "
        "execution");
    }
    _cross_rank_boundaries = std::move(hook);
  }
  //! The hook set by set_cross_rank_boundaries(), or null when this operator's sample stands alone.
  [[nodiscard]] cross_rank_sort_boundaries* cross_rank_boundaries() const noexcept
  {
    return _cross_rank_boundaries.get();
  }

  //! With a hook the boundary task is still owed, input or not, until it has been claimed.
  [[nodiscard]] bool all_ports_empty() override;
  //! With a hook the boundary task runs with no input, since the other processes wait for it.
  //! get_next_task_input_data() has already claimed it (SCHEDULED) when the task creator asks.
  [[nodiscard]] bool runs_without_input() const override
  {
    return _cross_rank_boundaries != nullptr && boundary_state() != BoundaryState::DONE;
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
  //! Evenly spaced boundaries for @p num_parts partitions from this operator's own sorted
  //! @p sample_keys, with at most one partition per key row.
  sort_boundaries local_boundaries(cudf::table_view sample_keys,
                                   std::size_t num_parts,
                                   cucascade::memory::memory_space* space,
                                   ::cuda::stream_ref stream);

  //! Positions and order of the sort key columns, from @ref orders.
  struct key_order {
    std::vector<int> columns;
    std::vector<cudf::order> column_order;
    std::vector<cudf::null_order> null_precedence;
  };
  //! @throws not_implemented_exception for a key that is not a column reference.
  [[nodiscard]] key_order sort_key_order() const;

  //! The input the ranges are sized from.
  struct sample_summary {
    std::size_t batches    = 0;  //!< Batches the key rows come from.
    std::size_t rows       = 0;  //!< Rows of those batches.
    std::size_t bytes      = 0;  //!< Bytes of those batches.
    bool upstream_finished = false;
    bool complete_input    = false;
    //! Projected bytes of the whole input, while upstream is still running.
    std::optional<std::size_t> projected_bytes;
  };
  //! Whether the boundary task may start on @p input: the input has ended, or a batch waits and
  //! key rows of sort_sample_bytes() of input were added. At 0 it waits for the end of the input.
  [[nodiscard]] bool boundary_input_ready(port const& input) const;

  struct range_sizing {
    std::size_t num_parts             = 1;
    std::size_t target_range_bytes    = 0;
    std::size_t estimated_total_bytes = 0;
    std::string_view est_source       = "observed";
  };
  //! The number of ranges for the input @p summary describes.
  [[nodiscard]] range_sizing size_ranges(sample_summary const& summary,
                                         cucascade::memory::memory_space& space,
                                         ::cuda::stream_ref stream) const;

  //! Ends the boundary task with one range, or with the hook's answer to an empty sample.
  void set_boundaries_without_sample(::cuda::stream_ref stream);
  //! Sets the boundaries from @p key_tables, each sorted and taken at @p stride.
  void set_boundaries(sample_summary const& summary,
                      range_sizing const& sizing,
                      key_order const& keys,
                      std::vector<cudf::table_view> const& key_tables,
                      std::size_t stride,
                      cucascade::memory::memory_space& space,
                      ::cuda::stream_ref stream);

  [[nodiscard]] BoundaryState boundary_state() const noexcept
  {
    return _boundary_state.load(std::memory_order_acquire);
  }

  //! See key_sample().
  std::shared_ptr<sort_key_sample> _key_sample = std::make_shared<sort_key_sample>();

  //! Partition boundary rows (P-1 rows containing sort key column values)
  std::unique_ptr<cudf::table> _partition_boundaries;

  //! Number of partitions computed from the sample
  size_t _num_partitions = 1;
  range_sample_stats _range_stats;
  std::atomic<std::uint64_t> _input_batches{0};

  //! Boundary computation lifecycle: NOT_DONE → SCHEDULED (input claimed) → DONE.
  //! get_next_task_input_data() transitions to SCHEDULED under the task-creation lock;
  //! execute() completes the transition to DONE. On OOM the state stays SCHEDULED so the
  //! rescheduled task retries while task_creator does not spawn a duplicate boundary task.
  std::atomic<BoundaryState> _boundary_state{BoundaryState::NOT_DONE};

  //! See sort_sample_bytes() (from sirius_config operator_params)
  uint64_t _sort_sample_bytes = config::DEFAULT_SORT_SAMPLE_BYTES;

  //! Override for max partition bytes (0 = use GPU memory fraction)
  size_t _max_partition_bytes_override = 0;

  //! Fraction of available GPU memory per partition (from sirius_config operator_params)
  double _max_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION;

  //! See set_cross_rank_boundaries().
  std::shared_ptr<cross_rank_sort_boundaries> _cross_rank_boundaries;
};

}  // namespace op
}  // namespace sirius
