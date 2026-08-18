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

#include "cudf/aggregation.hpp"
#include "cudf/types.hpp"
#include "duckdb/execution/operator/aggregate/distinct_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/radix_partitioned_hashtable.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/storage/data_table.hpp"
#include "expression/ast/node.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/aggregate/clustered_merge_bypass.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"

#include <memory>
#include <numeric>
#include <optional>

namespace sirius {
namespace planner {
class sirius_physical_plan_generator;
}  // namespace planner
namespace op {

class sirius_physical_grouped_aggregate_merge : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::MERGE_GROUP_BY;

 public:
  sirius_physical_grouped_aggregate_merge(
    sirius_physical_grouped_aggregate* grouped_aggregate,
    uint64_t hash_partition_bytes = config::DEFAULT_HASH_PARTITION_BYTES);

  sirius_physical_grouped_aggregate_merge(
    duckdb::vector<sirius::logical_type> types,
    std::vector<int> group_idx,
    std::vector<cudf::aggregation::Kind> cudf_aggregates,
    std::vector<int> cudf_aggregate_idx,
    std::vector<std::vector<int>> cudf_aggregate_struct_col_indices,
    std::vector<AggregateSlot> aggregate_slots,
    bool has_avg,
    bool has_count_distinct,
    std::size_t estimated_cardinality);

  sirius_physical_grouped_aggregate_merge(
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> groups,
    std::size_t estimated_cardinality);

  sirius_physical_grouped_aggregate_merge(
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> groups,
    duckdb::vector<duckdb::GroupingSet> grouping_sets,
    duckdb::vector<duckdb::unsafe_vector<std::size_t>> grouping_functions,
    std::size_t estimated_cardinality,
    duckdb::TupleDataValidityType group_validity,
    duckdb::TupleDataValidityType distinct_validity);

  //! The grouping sets
  duckdb::GroupedAggregateData grouped_aggregate_data;

  duckdb::vector<duckdb::GroupingSet> grouping_sets;
  //! The radix partitioned hash tables (one per grouping set)
  duckdb::vector<duckdb::HashAggregateGroupingData> groupings;
  duckdb::unique_ptr<duckdb::DistinctAggregateCollectionInfo> distinct_collection_info;
  //! A recreation of the input chunk, with nulls for everything that isn't a group
  duckdb::vector<sirius::logical_type> input_group_types;

  // Filters given to sink and friends
  duckdb::unsafe_vector<std::size_t> non_distinct_filter;
  duckdb::unsafe_vector<std::size_t> distinct_filter;

  sirius_physical_operator* child_op;
  sirius_physical_operator* get_child_op() const { return child_op; }

  // Grouped aggregatge definitions for cudf compute
  std::vector<int> group_idx;
  std::vector<cudf::aggregation::Kind> cudf_aggregates;
  std::vector<int> cudf_aggregate_idx;
  std::vector<std::vector<int>> cudf_aggregate_struct_col_indices;

  // AVG and COUNT DISTINCT decomposition metadata
  std::vector<AggregateSlot> aggregate_slots;
  bool has_avg            = false;
  bool has_count_distinct = false;

  std::size_t current_partition_index = 0;

 public:
  std::vector<int> get_output_grouping_indices() const
  {
    std::vector<int> indices(group_idx.size());
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
  }

  // Source interface
  bool is_source() const override { return true; }

  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }

  // Sink interface
  bool is_sink() const override { return true; }

  bool sink_order_dependent() const override { return false; }

  //! Whether this merge joins its downstream pipeline.
  [[nodiscard]] bool fuse_into_parent() const noexcept { return _fuse_into_parent; }

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //! Decide the partition count for the upstream PARTITION operator that feeds this merge
  partition_strategy get_partition_strategy(const partition_sizing_input& in) override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  // --- Clustered merge bypass (see op/aggregate/clustered_merge_bypass.hpp for the proof) ---

  //! Configure the bypass knobs. Stamped from operator_params at plan time; the operator
  //! default is OFF so unstamped construction paths keep the normal merge.
  void set_clustered_bypass_params(bool enabled, double max_overlap_fraction) noexcept
  {
    _clustered_bypass_enabled              = enabled;
    _clustered_bypass_max_overlap_fraction = max_overlap_fraction;
  }

  //! The predicate of the FILTER directly above this merge (non-owning; the filter operator
  //! outlives the plan). Stamped by the plan generator when the merge's tree parent is a FILTER;
  //! unit tests may stamp it directly. Required for the bypass: without a downstream filter to
  //! push into the partials, skipping the merge would not remove any work.
  void set_clustered_bypass_filter(const sirius::ast::node* filter_expression) noexcept
  {
    _bypass_filter_expression = filter_expression;
  }

  //! Static eligibility (knob + plan shape) checked before spending GPU time on the range proof:
  //! single GPU, a stamped downstream filter, no AVG / COUNT(DISTINCT) post-processing (their
  //! partial schema differs from the merge output schema, so the filter predicate could not be
  //! evaluated on partial rows), no grouping sets, and only merge combines that are identities
  //! on singletons (SUM/MIN/MAX/COUNT — required by the bypass proof).
  [[nodiscard]] bool clustered_bypass_wanted() const;

  //! Run the runtime range proof over the partial batches waiting on the upstream PARTITION's
  //! port and arm the bypass when it succeeds. Called by the PARTITION at sizing time, before
  //! get_partition_strategy. Returns whether the bypass was armed.
  bool try_plan_clustered_bypass(
    const std::vector<std::shared_ptr<::cucascade::data_batch>>& batches);

  //! Whether the bypass is armed (for tests / logging). Locks the operator mutex (same lock
  //! the arming write takes), hence non-const.
  [[nodiscard]] bool clustered_bypass_armed();

 private:
  friend class sirius::planner::sirius_physical_plan_generator;
  void set_fuse_into_parent(bool fuse) noexcept { _fuse_into_parent = fuse; }

  bool _fuse_into_parent = false;

  //! Clustered merge bypass state. The plan is written once at partition-sizing time (under
  //! `lock`) and read by the single merge task that sizing produces.
  bool _clustered_bypass_enabled                     = false;
  double _clustered_bypass_max_overlap_fraction      = 0.0;
  const sirius::ast::node* _bypass_filter_expression = nullptr;
  std::optional<clustered_bypass::plan> _bypass_plan;
};

}  // namespace op
}  // namespace sirius
