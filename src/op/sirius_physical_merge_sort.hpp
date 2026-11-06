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

#include "op/sirius_physical_order.hpp"
#include "op/sort_comparison_config.hpp"

#include <mutex>

namespace sirius::op {

// Shared scheduling and completion mechanics. Each stage has a FULL input barrier.
class sirius_physical_sort_stage : public sirius_physical_operator {
 public:
  using batches = std::vector<std::shared_ptr<cucascade::data_batch>>;
  struct task_data : pipelineable_operator_data {
    std::size_t partition;
    task_data(batches values,
              std::size_t partition,
              std::optional<std::size_t> position = std::nullopt);
    std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
      batches values) const override;
  };

  sirius_physical_sort_stage(SiriusPhysicalOperatorType type,
                             duckdb::vector<sirius::logical_type> types,
                             std::size_t cardinality,
                             std::shared_ptr<const sort_comparison_config> comparison);
  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }
  MemoryBarrierType input_barrier_for(sirius_physical_operator const&) const override
  {
    return MemoryBarrierType::FULL;
  }
  sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }
  bool all_ports_empty() override;
  std::optional<task_creation_hint> get_next_task_hint() override;
  std::unique_ptr<operator_data> get_next_task_input_data() override = 0;
  void on_task_output_published(operator_data const& output) override;
  std::size_t no_history_peak_memory_estimate(input_stats const& stats) const override;

  std::shared_ptr<const sort_comparison_config> comparison;
  std::size_t chunk_bytes{config::DEFAULT_SORT_CHUNK_BYTES};

 protected:
  std::size_t _partition{0};
  std::size_t _outstanding{0};
};

// Input: individually sorted batches in each range partition.
// Output: normalized/coalesced runs at frozen run positions within that partition.
class sirius_physical_sort_normalize : public sirius_physical_sort_stage {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::SORT_NORMALIZE;
  explicit sirius_physical_sort_normalize(sirius_physical_order* order);
  void sink(operator_data const& output, ::cuda::stream_ref stream) override;
  [[nodiscard]] std::unique_ptr<operator_data> get_next_task_input_data() override;
  [[nodiscard]] std::unique_ptr<operator_data> execute(operator_data const&,
                                                       ::cuda::stream_ref) override;
};

// Input: frozen runs. Output: (run, cut-table) pairs at positions (2*i, 2*i+1).
// Cut tables have one INT32 column containing internal boundaries, no endpoints.
// Both kinds of batch are ordinary spillable repository entries. Even an empty
// cut table is significant: it describes a single tile.
class sirius_physical_sort_cuts : public sirius_physical_sort_stage {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::SORT_CUTS;
  explicit sirius_physical_sort_cuts(sirius_physical_order* order);
  void sink(operator_data const& output, ::cuda::stream_ref stream) override;
  [[nodiscard]] std::unique_ptr<operator_data> get_next_task_input_data() override;
  [[nodiscard]] bool declared_output_schema_is_runtime_schema() const noexcept override
  {
    return false;
  }
  [[nodiscard]] std::unique_ptr<operator_data> execute(operator_data const&,
                                                       ::cuda::stream_ref) override;
};

// Input: one matched run/cuts pair per task. Output: owning pieces partitioned by
// global tile number, positioned by run ordinal. The partition cursor accumulates
// tile counts under FULL input closure.
class sirius_physical_sort_extract : public sirius_physical_sort_stage {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::SORT_EXTRACT;
  struct input_data : pipelineable_operator_data {
    std::size_t run, first_tile;
    input_data(batches values, std::size_t run, std::size_t first_tile);
    [[nodiscard]] std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
      batches values) const override;
  };
  struct output_data : pipelineable_operator_data {
    std::size_t run;
    std::vector<std::size_t> tiles;
    output_data(batches values, std::size_t run, std::vector<std::size_t> tiles);
    [[nodiscard]] std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
      batches values) const override;
  };
  explicit sirius_physical_sort_extract(sirius_physical_order* order);
  [[nodiscard]] std::unique_ptr<operator_data> get_next_task_input_data() override;
  [[nodiscard]] std::unique_ptr<operator_data> execute(operator_data const&,
                                                       ::cuda::stream_ref) override;
  void sink(operator_data const& output, ::cuda::stream_ref stream) override;

 private:
  std::size_t _run{0};
  std::size_t _run_count{0};
  std::size_t _first_tile{0};
  std::size_t _tile_count{0};
};

// Input: all pieces of one tile. Output: one sorted batch at global position tile.
// Empty tiles leave position gaps; batch IDs and task completion order are irrelevant.
class sirius_physical_merge_sort : public sirius_physical_sort_stage {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::MERGE_SORT;
  [[nodiscard]] std::unique_ptr<operator_data> get_next_task_input_data() override;
  sirius_physical_merge_sort(sirius_physical_order* order_by);
  sirius_physical_merge_sort(duckdb::vector<sirius::logical_type> types,
                             duckdb::vector<duckdb::BoundOrderByNode> orders,
                             duckdb::vector<std::size_t> projections,
                             std::size_t estimated_cardinality,
                             std::shared_ptr<const sort_comparison_config> comparison = nullptr);
  sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }
  [[nodiscard]] std::unique_ptr<operator_data> execute(operator_data const&,
                                                       ::cuda::stream_ref) override;
  void sink(operator_data const& output, ::cuda::stream_ref stream) override;
  void set_final_projections(duckdb::vector<std::size_t> projection,
                             duckdb::vector<sirius::logical_type> output_types)
  {
    _final_projections = std::move(projection);
    types              = std::move(output_types);
  }

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  duckdb::vector<std::size_t> projections;

 private:
  duckdb::vector<std::size_t> _final_projections;
};
}  // namespace sirius::op
