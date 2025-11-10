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

#include "gpu_physical_operator.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/operator/join/physical_join.hpp"
#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/execution/join_hashtable.hpp"
#include "duckdb/execution/operator/join/perfect_hash_join_executor.hpp"
#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "gpu_columns.hpp"

namespace duckdb {

template <typename T>
void nestedLoopJoin(T** left_data, T** right_data, uint64_t* &row_ids_left, uint64_t* &row_ids_right, uint64_t* &count, uint64_t left_size, uint64_t right_size, int* condition_mode, int num_keys);

//! PhysicalNestedLoopJoin represents a nested loop join between two tables
class GPUPhysicalNestedLoopJoin : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::NESTED_LOOP_JOIN;

public:
	GPUPhysicalNestedLoopJoin(LogicalOperator &op, unique_ptr<GPUPhysicalOperator> left, unique_ptr<GPUPhysicalOperator> right,
	                       vector<JoinCondition> cond, JoinType join_type, idx_t estimated_cardinality);

	vector<JoinCondition> conditions;
	//! The types of the join keys
	vector<LogicalType> condition_types;
	//! The type of the join
	JoinType join_type;

	//! The indices for getting the payload columns
	vector<idx_t> payload_column_idxs;
	//! The types of the payload columns
	vector<LogicalType> payload_types;

	//! Positions of the RHS columns that need to output
	vector<idx_t> rhs_output_columns;
	//! The types of the output
	vector<LogicalType> rhs_output_types;

	//! Duplicate eliminated types; only used for delim_joins (i.e. correlated subqueries)
	vector<LogicalType> delim_types;

    shared_ptr<GPUIntermediateRelation> right_temp_data;

protected:
	// CachingOperator Interface
    OperatorResultType Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const override;

    static void BuildJoinPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline, GPUPhysicalOperator &op, bool build_rhs = true);
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline);

public:
	// Source interface
    SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

	bool IsSource() const override {
		return PropagatesBuildSide(join_type);
	}
	bool ParallelSource() const override {
		return true;
	}

public:
	// Sink Interface
    SinkResultType Sink(GPUIntermediateRelation &input_relation) const override;

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return true;
	}

	static bool IsSupported(const vector<JoinCondition> &conditions, JoinType join_type);

public:
	//! Returns a list of the types of the join conditions
	vector<LogicalType> GetJoinTypes() const;

private:
	// // resolve joins that output max N elements (SEMI, ANTI, MARK)
	void ResolveSimpleJoin(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const;
	// // resolve joins that can potentially output N*M elements (INNER, LEFT, FULL)
	OperatorResultType ResolveComplexJoin(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const;
};

} // namespace duckdb
