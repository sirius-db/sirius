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
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/logical_tokens.hpp"
#include "op/sirius_physical_operator.hpp"

namespace duckdb {
class ClientContext;
class GPUContext;
class ColumnDataCollection;
}  // namespace duckdb

namespace sirius::planner {

//! The physical plan generator generates a physical execution plan from a
//! logical query plan
class sirius_physical_plan_generator {
 public:
  explicit sirius_physical_plan_generator(duckdb::ClientContext& context,
                                          duckdb::GPUContext& gpu_context);
  ~sirius_physical_plan_generator();

  duckdb::LogicalDependencyList dependencies;
  //! Recursive CTEs require at least one ChunkScan, referencing the working_table.
  //! This data structure is used to establish it.
  duckdb::unordered_map<duckdb::idx_t, duckdb::shared_ptr<duckdb::ColumnDataCollection>>
    recursive_cte_tables;
  //! Used to reference the recurring tables
  duckdb::unordered_map<duckdb::idx_t, duckdb::shared_ptr<duckdb::ColumnDataCollection>>
    recurring_cte_tables;
  //! Materialized CTE ids must be collected.
  duckdb::unordered_map<
    duckdb::idx_t,
    duckdb::vector<duckdb::const_reference<sirius::op::sirius_physical_operator>>>
    materialized_ctes;
  // duckdb::unordered_map<duckdb::idx_t, duckdb::shared_ptr<duckdb::GPUIntermediateRelation>>
  // gpu_recursive_cte_tables;

 public:
  //! Creates a plan from the logical operator. This involves resolving column bindings and
  //! generating physical operator nodes.
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
  static duckdb::OrderPreservationType order_preservation_recursive(
    sirius::op::sirius_physical_operator& op);

  static bool has_equality(duckdb::vector<duckdb::JoinCondition>& conds,
                           duckdb::idx_t& range_count);

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
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> extract_aggregate_expressions(
    duckdb::unique_ptr<sirius::op::sirius_physical_operator> child,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& expressions,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups,
    duckdb::optional_ptr<duckdb::vector<duckdb::GroupingSet>> grouping_sets);

  // private:
  bool preserve_insertion_order(sirius::op::sirius_physical_operator& plan);
  // bool use_batch_index(sirius::op::sirius_physical_operator &plan);
 public:
  duckdb::idx_t delim_index = 0;

 public:
  duckdb::ClientContext& context;
  duckdb::GPUContext& gpu_context;
};
}  // namespace sirius::planner
