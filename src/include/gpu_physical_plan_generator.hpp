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

#include "duckdb/common/common.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/logical_tokens.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/catalog/dependency_list.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "gpu_physical_operator.hpp"

namespace duckdb {
class ClientContext;
class GPUContext;
class ColumnDataCollection;

//! The physical plan generator generates a physical execution plan from a
//! logical query plan
class GPUPhysicalPlanGenerator {
public:
	explicit GPUPhysicalPlanGenerator(ClientContext &context, GPUContext& gpu_context);
	~GPUPhysicalPlanGenerator();

	LogicalDependencyList dependencies;
	//! Recursive CTEs require at least one ChunkScan, referencing the working_table.
	//! This data structure is used to establish it.
	unordered_map<idx_t, shared_ptr<ColumnDataCollection>> recursive_cte_tables;
	//! Materialized CTE ids must be collected.
	unordered_map<idx_t, vector<const_reference<GPUPhysicalOperator>>> materialized_ctes;
	unordered_map<idx_t, shared_ptr<GPUIntermediateRelation>> gpu_recursive_cte_tables;
public:
	//! Creates a plan from the logical operator. This involves resolving column bindings and generating physical
	//! operator nodes.
	unique_ptr<GPUPhysicalOperator> CreatePlan(unique_ptr<LogicalOperator> logical);

	//! Whether or not we can (or should) use a batch-index based operator for executing the given sink
	static bool UseBatchIndex(ClientContext &context, GPUPhysicalOperator &plan);
	//! Whether or not we should preserve insertion order for executing the given sink
	static bool PreserveInsertionOrder(ClientContext &context, GPUPhysicalOperator &plan);

	static bool HasEquality(vector<JoinCondition> &conds, idx_t &range_count);

protected:
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalOperator &op);

	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalAggregate &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalAnyJoin &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalColumnDataGet &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalComparisonJoin &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalCopyDatabase &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalCreate &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalCreateTable &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalCreateIndex &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalCreateSecret &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalCrossProduct &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalDelete &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalDelimGet &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalDistinct &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalDummyScan &expr);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalEmptyResult &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalExpressionGet &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalExport &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalFilter &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalGet &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalLimit &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalOrder &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalTopN &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalPositionalJoin &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalProjection &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalInsert &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalCopyToFile &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalExplain &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalSetOperation &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalUpdate &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalPrepare &expr);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalWindow &expr);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalExecute &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalPragma &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalSample &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalSet &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalReset &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalSimple &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalVacuum &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalUnnest &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalRecursiveCTE &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalMaterializedCTE &op);
	unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalCTERef &op);
	// unique_ptr<GPUPhysicalOperator> CreatePlan(LogicalPivot &op);

	// unique_ptr<GPUPhysicalOperator> PlanAsOfJoin(LogicalComparisonJoin &op);
	unique_ptr<GPUPhysicalOperator> PlanComparisonJoin(LogicalComparisonJoin &op);
	unique_ptr<GPUPhysicalOperator> PlanDelimJoin(LogicalComparisonJoin &op);
	unique_ptr<GPUPhysicalOperator> ExtractAggregateExpressions(unique_ptr<GPUPhysicalOperator> child,
	                                                         vector<unique_ptr<Expression>> &expressions,
	                                                         vector<unique_ptr<Expression>> &groups);

// private:
	// bool PreserveInsertionOrder(GPUPhysicalOperator &plan);
	// bool UseBatchIndex(GPUPhysicalOperator &plan);
public:
	idx_t delim_index = 0;

public:
	ClientContext &context;
	GPUContext& gpu_context;
};
} // namespace duckdb