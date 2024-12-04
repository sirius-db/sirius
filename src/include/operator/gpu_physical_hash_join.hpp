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

// template <typename T>
// void buildHashTableOri(uint64_t *keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int mode);

// template <typename T>
// void probeHashTableOri(uint64_t *keys, unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids_left, uint64_t* &row_ids_right, uint64_t* &count, uint64_t N, int mode);

void probeHashTable(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids_left, uint64_t* &row_ids_right, uint64_t* &count, 
			uint64_t N, int* condition_mode, int num_keys, bool is_right);

void probeHashTableRightSemiAnti(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int* condition_mode, int num_keys);

void probeHashTableSingleMatch(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids_left, uint64_t* &row_ids_right, 
            uint64_t* &count, uint64_t N, int* condition_mode, int num_keys, int join_mode);

void probeHashTableMark(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint8_t* &output, uint64_t N, int* condition_mode, int num_keys);

void buildHashTable(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int* condition_mode, int num_keys, bool is_right);

void scanHashTableRight(unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids, uint64_t* &count, int join_mode, int num_keys);

class GPUPhysicalHashJoin : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::HASH_JOIN;
	
public:
						   
	GPUPhysicalHashJoin(LogicalOperator &op, unique_ptr<GPUPhysicalOperator> left, unique_ptr<GPUPhysicalOperator> right,
	                 vector<JoinCondition> cond, JoinType join_type, const vector<idx_t> &left_projection_map,
	                 const vector<idx_t> &right_projection_map, vector<LogicalType> delim_types,
	                 idx_t estimated_cardinality);
	GPUPhysicalHashJoin(LogicalOperator &op, unique_ptr<GPUPhysicalOperator> left, unique_ptr<GPUPhysicalOperator> right,
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

	// OperatorResultType Execute(ExecutionContext &context, GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation,
	// 									GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorResultType Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const override;

	static void BuildJoinPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline, GPUPhysicalOperator &op, bool build_rhs = true);
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline);

protected:
	// CachingOperator Interface
	// OperatorResultType ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	//                                    GlobalOperatorState &gstate, OperatorState &state) const override;

	// Source interface
	// unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	// unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	//                                                  GlobalSourceState &gstate) const override;
	// SourceResultType GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation, OperatorSourceInput &input) const override;
	SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

	// double GetProgress(ClientContext &context, GlobalSourceState &gstate) const override;

	//! Becomes a source when it is an external join
	bool IsSource() const override {
		return true;
	}

	bool ParallelSource() const override {
		return true;
	}

public:
	// Sink Interface
	// unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	// unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	// SinkResultType Sink(ExecutionContext &context, GPUIntermediateRelation &input_relation, OperatorSinkInput &input) const override;
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

	mutable unsigned long long* gpu_hash_table;
	mutable uint64_t ht_len;

	GPUIntermediateRelation* hash_table_result;

};
} // namespace duckdb