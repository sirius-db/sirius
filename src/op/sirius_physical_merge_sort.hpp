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

#include <cucascade/data/data_repository_manager.hpp>

#include <mutex>
#include <set>

namespace sirius::op {

// Input: individually sorted contributions in each range partition. Each range is a
// refinement node with an interval of output positions. A node whose decoded bytes fit
// chunk_bytes (from metadata only) is merged by one task and published at the start of
// its interval. An oversized node is refined in place, through registered repositories
// owned by this operator:
//
//   ready -> sampling (bounded contribution groups) -> plan (one splitter task)
//         -> cutting (bounded contribution groups) -> children ready
//
// Children subdivide the parent's interval, so output order is independent of task
// completion order and positions may have gaps. Every task is one bounded step; the
// state machine advances only when a task's output is published.
class sirius_physical_merge_sort : public sirius_physical_operator {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::MERGE_SORT;
  using batches              = std::vector<std::shared_ptr<cucascade::data_batch>>;

  enum class step : uint8_t { merge, sample, plan, cut };

  // `ordinals` identify each payload batch within its node (merge/sample/cut). For a
  // cut, the node's shared splitter table is the final batch. `parameter` is the
  // sample stride (sample), splitter step (plan) or first child node (cut).
  struct work : pipelineable_operator_data {
    step kind;
    std::size_t node;
    std::vector<uint64_t> ordinals;
    uint64_t parameter;
    work(batches values,
         step kind,
         std::size_t node,
         std::vector<uint64_t> ordinals,
         uint64_t parameter,
         std::optional<std::size_t> position = std::nullopt);
    [[nodiscard]] std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
      batches values) const override;
  };

  struct piece {
    std::size_t child;
    uint64_t ordinal, rows, bytes;
  };

  // Batches align with `ordinals` (sample) or `pieces` (cut). A plan result holds the
  // splitter table, and `ordinals` holds its row count; a merge result holds one
  // positioned output batch.
  struct result : pipelineable_operator_data {
    step kind;
    std::size_t node;
    std::vector<uint64_t> ordinals;
    std::vector<piece> pieces;
    result(batches values,
           step kind,
           std::size_t node,
           std::vector<uint64_t> ordinals,
           std::vector<piece> pieces,
           std::optional<std::size_t> position = std::nullopt);
    [[nodiscard]] std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
      batches values) const override;
  };

  [[nodiscard]] std::unique_ptr<operator_data> get_next_task_input_data() override;
  std::optional<task_creation_hint> get_next_task_hint() override;
  bool all_ports_empty() override;
  void on_task_output_published(operator_data const& output) override;
  void register_refinement(cucascade::shared_data_repository_manager& manager);
  sirius_physical_merge_sort(sirius_physical_order* order_by);
  sirius_physical_merge_sort(duckdb::vector<sirius::logical_type> types,
                             duckdb::vector<duckdb::BoundOrderByNode> orders,
                             duckdb::vector<std::size_t> projections,
                             std::size_t estimated_cardinality,
                             std::shared_ptr<const sort_comparison_config> comparison = nullptr);
  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }
  MemoryBarrierType input_barrier_for(sirius_physical_operator const&) const override
  {
    return MemoryBarrierType::FULL;
  }
  sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }
  std::size_t no_history_peak_memory_estimate(input_stats const& stats) const override;
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
  std::shared_ptr<const sort_comparison_config> comparison;
  // Decoded-byte target for a merged output chunk; larger nodes are refined.
  std::size_t chunk_bytes{config::DEFAULT_SORT_CHUNK_BYTES};

 private:
  struct contribution {
    uint64_t ordinal, rows, bytes;
  };
  // collecting: children whose producers are still cutting; not yet actionable.
  enum class phase : uint8_t {
    collecting,
    ready,
    merging,
    sampling,
    plan_ready,
    planning,
    cutting,
    done
  };
  struct node {
    std::size_t lo, hi;  // output positions [lo, hi)
    std::vector<contribution> parts;
    uint64_t rows{0}, bytes{0};
    phase state{phase::collecting};
    std::size_t cursor{0}, outstanding{0};
    uint64_t stride{0}, sample_step{0};
    std::size_t first_child{0}, child_count{0};
  };

  void ingest();
  [[nodiscard]] bool fits(node const& n) const;
  [[nodiscard]] std::optional<std::size_t> actionable() const;
  [[nodiscard]] std::unique_ptr<work> claim(std::size_t id);
  [[nodiscard]] std::unique_ptr<work> claim_group(std::size_t id, step kind, bool pop);
  void retire(std::size_t id);

  std::unique_ptr<operator_data> execute_merge(work const&, ::cuda::stream_ref);
  std::unique_ptr<operator_data> execute_sample(work const&, ::cuda::stream_ref);
  std::unique_ptr<operator_data> execute_plan(work const&, ::cuda::stream_ref);
  std::unique_ptr<operator_data> execute_cut(work const&, ::cuda::stream_ref);

  duckdb::vector<std::size_t> _final_projections;
  // Payload contributions by node, positioned by ordinal; samples likewise; one splitter
  // table per node at position zero. All are registered, so queued state is spillable.
  cucascade::shared_data_repository* _parts     = nullptr;
  cucascade::shared_data_repository* _samples   = nullptr;
  cucascade::shared_data_repository* _splitters = nullptr;
  std::vector<node> _nodes;
  // Actionable or in-flight nodes ordered by output position: leftmost work first.
  std::set<std::pair<std::size_t, std::size_t>> _active;
  bool _ingested{false};
};
}  // namespace sirius::op
