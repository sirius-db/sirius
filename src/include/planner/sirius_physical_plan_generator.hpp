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

#include "duckdb/catalog/dependency_list.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "op/dynamic_filter_identity.hpp"
#include "op/sirius_physical_operator.hpp"
#include "planner/dynamic_filter_candidate_cache.hpp"

#include <memory>
#include <string>
#include <unordered_map>

namespace sirius::op {
class sirius_physical_table_scan;
}  // namespace sirius::op

namespace duckdb {
class ClientContext;
class GPUContext;
class ColumnDataCollection;
class DynamicTableFilterSet;
class LogicalOperator;
class LogicalAggregate;
class LogicalColumnDataGet;
class LogicalComparisonJoin;
class LogicalDelimGet;
class LogicalDummyScan;
class LogicalEmptyResult;
class LogicalExpressionGet;
class LogicalFilter;
class LogicalGet;
class LogicalLimit;
class LogicalOrder;
class LogicalTopN;
class LogicalProjection;
class LogicalMaterializedCTE;
class LogicalCTERef;
}  // namespace duckdb

namespace sirius::op {
class sirius_dynamic_filter_set;
}

namespace sirius::planner {

//! The physical plan generator generates a physical execution plan from a
//! logical query plan
class sirius_physical_plan_generator {
 public:
  explicit sirius_physical_plan_generator(duckdb::ClientContext& context);
  ~sirius_physical_plan_generator();

  duckdb::LogicalDependencyList dependencies;
  //! Recursive CTEs require at least one ChunkScan, referencing the working_table.
  //! This data structure is used to establish it.
  duckdb::unordered_map<std::size_t, duckdb::shared_ptr<duckdb::ColumnDataCollection>>
    recursive_cte_tables;
  //! Used to reference the recurring tables
  duckdb::unordered_map<std::size_t, duckdb::shared_ptr<duckdb::ColumnDataCollection>>
    recurring_cte_tables;
  //! Materialized CTE ids must be collected.
  duckdb::unordered_map<
    std::size_t,
    duckdb::vector<duckdb::const_reference<sirius::op::sirius_physical_operator>>>
    materialized_ctes;
  // duckdb::unordered_map<std::size_t, duckdb::shared_ptr<duckdb::GPUIntermediateRelation>>
  // gpu_recursive_cte_tables;

 private:
  /// Dynamic-filter planning is one generator-owned transaction:
  ///
  /// ```text
  /// DuckDB logical tree
  ///          |
  ///          | capture exact join-node set now
  ///          | (C1b will also capture source evidence here)
  ///          v
  /// ColumnBindingResolver rewrites the tree
  ///          |
  ///          | extract one Sirius-owned candidate per join
  ///          v
  /// recursive physical planning
  ///          |
  ///          +--> scan (consumer) and producer find the same channel object
  ///          +--> producer receives a sidecar builder and strong planning IDs
  ///          v
  /// pipeline conversion (the legacy publication path stays authoritative until C1a-2b)
  /// ```
  ///
  /// @brief Map from duckdb::DynamicTableFilterSet pointers to sirius_dynamic_filter_set channels.
  /// DuckDB pairs a producer (join) and a consumer (scan) by planting the same
  /// DynamicTableFilterSet pointer on both sides, so we key the channels by these pointers.
  std::unordered_map<duckdb::DynamicTableFilterSet const*,
                     std::shared_ptr<sirius::op::sirius_dynamic_filter_set>>
    dynamic_filter_channels;

  /// @brief One immutable dynamic-filter candidate per comparison join. Consumed by
  /// plan_comparison_join() and, later, C3 route discovery via candidate_for().
  dynamic_filter_candidate_cache candidate_cache;

  /// @brief The minting authority for dynamic-filter identity in this plan:
  ///
  /// - Publication-plan IDs (one per producing join)
  /// - Target IDs (one per producer-to-scan edge)
  /// - Channel IDs (one per shared scan channel)
  ///
  /// C3's route registry must receive this same allocator through the generator handoff.
  sirius::op::dynamic_filter_identity_allocator dynamic_filter_id_allocator;

  /// @brief One publication-plan ID per producing logical join, no matter how many targets it
  /// adds or how many times it is looked at. Keyed by logical node address, like the candidate
  /// cache (stable through create_plan()).
  std::unordered_map<duckdb::LogicalOperator const*, sirius::op::dynamic_filter_publication_plan_id>
    dynamic_filter_publication_ids;

  /// @brief One channel ID per shared scan channel: producers that feed the same scan reuse the
  /// same channel ID (but keep distinct target IDs). Keyed by the same preserved DuckDB pointer
  /// that keys @ref dynamic_filter_channels.
  std::unordered_map<duckdb::DynamicTableFilterSet const*, sirius::op::dynamic_filter_channel_id>
    dynamic_filter_channel_ids;

 public:
  /// @brief Look up or create the dynamic-filter channel keyed by @p key.
  ///
  /// Returns nullptr when @p key is null or dynamic-filter pushdown is disabled. Repeated calls
  /// with the same non-null key return the same channel.
  [[nodiscard]] std::shared_ptr<sirius::op::sirius_dynamic_filter_set>
  get_or_create_dynamic_filter_channel(duckdb::DynamicTableFilterSet const* key);

  /// @brief How many distinct dynamic-filter channels this planning transaction has created.
  [[nodiscard]] std::size_t dynamic_filter_channel_count() const noexcept
  {
    return dynamic_filter_channels.size();
  }

  //! Creates a plan from the logical operator. This involves resolving column bindings and
  //! generating physical operator nodes. A generator owns one planning transaction, so this
  //! entrypoint may be called exactly once per generator instance.
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(
    duckdb::unique_ptr<duckdb::LogicalOperator> logical);

  //! Whether or not we can (or should) use a batch-index based operator for executing the given
  //! sink
  static bool use_batch_index(duckdb::ClientContext& context,
                              sirius::op::sirius_physical_operator& plan);
  //! Whether or not we should preserve insertion order for executing the given sink
  static bool preserve_insertion_order(duckdb::ClientContext& context,
                                       sirius::op::sirius_physical_operator& plan);
  //! The order preservation type of the given operator decided by recursively looking at its
  //! children
  static sirius::OrderPreservationType order_preservation_recursive(
    sirius::op::sirius_physical_operator& op);

 protected:
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalOperator& op);

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(
    duckdb::LogicalAggregate& op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalAnyJoin
  // &op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(
    duckdb::LogicalColumnDataGet& op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(
    duckdb::LogicalComparisonJoin& op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // create_plan(duckdb::LogicalCopyDatabase &op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalCreate
  // &op); duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // create_plan(duckdb::LogicalCreateTable &op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalCreateIndex
  // &op); duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // create_plan(duckdb::LogicalCreateSecret &op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // create_plan(duckdb::LogicalCrossProduct &op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalDelete
  // &op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalDelimGet& op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalDistinct
  // &op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(
    duckdb::LogicalDummyScan& expr);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(
    duckdb::LogicalEmptyResult& op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(
    duckdb::LogicalExpressionGet& op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalExport
  // &op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalFilter& op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalGet& op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalLimit& op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalOrder& op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalTopN& op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // create_plan(duckdb::LogicalPositionalJoin &op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(
    duckdb::LogicalProjection& op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalInsert
  // &op); duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // create_plan(duckdb::LogicalCopyToFile &op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalExplain
  // &op); duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // create_plan(duckdb::LogicalSetOperation &op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalUpdate
  // &op); duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // create_plan(duckdb::LogicalPrepare &expr);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalWindow
  // &expr); duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // create_plan(duckdb::LogicalExecute &op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalPragma
  // &op); duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // create_plan(duckdb::LogicalSample &op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalSet &op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalReset &op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalSimple
  // &op); duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // create_plan(duckdb::LogicalVacuum &op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalUnnest
  // &op); duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // create_plan(duckdb::LogicalRecursiveCTE &op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(
    duckdb::LogicalMaterializedCTE& op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalCTERef& op);
  // duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(duckdb::LogicalPivot &op);

  // duckdb::unique_ptr<sirius::op::sirius_physical_operator>
  // plan_asof_join(duckdb::LogicalComparisonJoin &op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan_comparison_join(
    duckdb::LogicalComparisonJoin& op);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan_delim_join(
    duckdb::LogicalComparisonJoin& op);

  // private:
  bool preserve_insertion_order(sirius::op::sirius_physical_operator& plan);
  // bool use_batch_index(sirius::op::sirius_physical_operator &plan);
 public:
  std::size_t delim_index = 0;

 public:
  duckdb::ClientContext& context;
  // duckdb::GPUContext& gpu_context;

 public:
  //! Recursive post-pass that derives each operator's `_parent_op` from the final tree's
  //! `children[]`, run after all tree rewrites have settled so the pointer cannot drift.
  //! Public so the engine can re-run it after the RESULT_COLLECTOR wrap is added around the
  //! plan post-generation — otherwise the wrapped child's `_parent_op` would stay null and
  //! break the tree-parent-driven wiring.
  static void set_parent_ops(sirius::op::sirius_physical_operator& op,
                             sirius::op::sirius_physical_operator* parent);

 private:
  //! Walk the plan tree and insert the GPU pipeline operators (PARTITION, CONCAT, sort chain,
  //! merge operators, scan companions, GPU_VALUES) so the tree carries the full execution
  //! structure before the pipeline converter runs.
  void insert_gpu_pipeline_operators(
    duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan);
};
}  // namespace sirius::planner
