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
#include "op/sirius_physical_operator.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::op {
class sirius_physical_table_scan;
}  // namespace sirius::op

namespace duckdb {
class ClientContext;
class Expression;
class FunctionData;
class LogicalType;
class Value;
class GPUContext;
class ColumnDataCollection;
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
class LogicalSetOperation;
class LogicalMaterializedCTE;
class LogicalCTERef;
}  // namespace duckdb

namespace sirius::planner {

/// Resolved parquet file set identifying a parquet-family scan
/// ("parquet_scan" / "read_parquet" / "sirius_read_parquet"), derived exactly
/// as the scan's ingestible_table_info derives it — so a plan-time cache probe
/// and the prepare-time cache match see the same identity. Returns empty when
/// the identity cannot be resolved (non-parquet function, missing bind data,
/// empty file list, or a missing/NULL sirius_read_parquet URI parameter);
/// callers treat empty as "no identity", never as an error.
[[nodiscard]] std::vector<std::string> resolve_parquet_scan_file_paths(
  std::string_view function_name,
  duckdb::FunctionData const* bind_data,
  duckdb::vector<duckdb::Value> const& parameters);

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

  //! Builds the STREAMING_SOURCE a `sirius_stream_source(id)` read stands for, wired to the
  //! repository and expected sender set the fragment declared for that id on this connection.
  //! Records the built operator back into the catalog so the fragment can register it with its
  //! stream_session once the plan tree owns it.
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_streaming_source_plan(
    duckdb::LogicalGet& op);
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
  // &op);
  //! `UNION ALL` only; the builder rejects distinct UNION. EXCEPT / INTERSECT share the same
  //! DuckDB node but keep their own throwing case in the dispatch switch, so they never arrive.
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_plan(
    duckdb::LogicalSetOperation& op);
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

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> try_plan_dense_count_join(
    duckdb::LogicalAggregate& op);

  // Sirius reads and projects nested (STRUCT/LIST/MAP) columns but cannot operate
  // on them yet: WHERE / GROUP BY / JOIN ON over a nested column must fail with a
  // clear error naming the column instead of crashing or returning wrong results.
  // @p operation names the context, e.g. "a filter predicate".
  static void reject_nested_column_operation(duckdb::Expression const& expr,
                                             std::string_view operation);

  // Same check when only a column type + name are available (e.g. a predicate
  // pushed into LogicalGet::table_filters).
  static void reject_nested_column_type(duckdb::LogicalType const& type,
                                        std::string_view column_name,
                                        std::string_view operation);

  // private:
  bool preserve_insertion_order(sirius::op::sirius_physical_operator& plan);
  // bool use_batch_index(sirius::op::sirius_physical_operator &plan);
 public:
  std::size_t delim_index = 0;

 public:
  duckdb::ClientContext& context;
  // duckdb::GPUContext& gpu_context;

 public:
  //! Recursive post-pass that derives each operator's `_parent_op` from the final tree after
  //! rewrites finish. The engine calls it again after adding the RESULT_COLLECTOR wrapper to
  //! update the wrapped child's parent for tree-parent wiring.
  static void set_parent_ops(sirius::op::sirius_physical_operator& op,
                             sirius::op::sirius_physical_operator* parent);

  //! Mark eligible merge operators for downstream pipeline fusion.
  static void mark_fusable_merge_pipelines(duckdb::ClientContext& context,
                                           sirius::op::sirius_physical_operator& op);

 private:
  static void mark_fusable_merge_pipelines(sirius::op::sirius_physical_operator& op,
                                           bool fusion_enabled);

 public:
  //! Walk the plan tree and insert the GPU pipeline operators (PARTITION, CONCAT, sort chain,
  //! merge operators, scan companions, GPU_VALUES) so the tree carries the full execution
  //! structure before the pipeline converter runs. Public so wrap-contract tests can drive the
  //! rewrite over hand-built operator trees and assert the wrapper shapes and physical-sidecar
  //! copies directly.
  void insert_gpu_pipeline_operators(
    duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan);
};
}  // namespace sirius::planner
