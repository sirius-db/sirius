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

    GPUIntermediateRelation *right_temp_data;

public:
	// Operator Interface
	// unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;

	// bool ParallelOperator() const override {
	// 	return true;
	// }

protected:
	// CachingOperator Interface
	// OperatorResultType ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	//                                    GlobalOperatorState &gstate, OperatorState &state) const override;

    OperatorResultType Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const override;

    static void BuildJoinPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline, GPUPhysicalOperator &op, bool build_rhs = true);
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline);

public:
	// Source interface
	// unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	// unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	//                                                  GlobalSourceState &gstate) const override;
	// SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override;
    SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

	bool IsSource() const override {
		return PropagatesBuildSide(join_type);
	}
	bool ParallelSource() const override {
		return true;
	}

public:
	// Sink Interface
	// unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	// unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	// SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
    SinkResultType Sink(GPUIntermediateRelation &input_relation) const override;
	// SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	// SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	//                           OperatorSinkFinalizeInput &input) const override;

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
	// // // resolve joins that output max N elements (SEMI, ANTI, MARK)
	// void ResolveSimpleJoin(ExecutionContext &context, GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation, OperatorState &state) const;
	// // // resolve joins that can potentially output N*M elements (INNER, LEFT, FULL)
	// OperatorResultType ResolveComplexJoin(ExecutionContext &context, GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation,
	//                                       OperatorState &state) const;

	// // resolve joins that output max N elements (SEMI, ANTI, MARK)
	void ResolveSimpleJoin(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const;
	// // resolve joins that can potentially output N*M elements (INNER, LEFT, FULL)
	OperatorResultType ResolveComplexJoin(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const;
};

} // namespace duckdb
