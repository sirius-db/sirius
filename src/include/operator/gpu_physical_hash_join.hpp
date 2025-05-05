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
#include "utils.hpp"
#include "cudf_utils.hpp"
#include "gpu_columns.hpp"

namespace duckdb {

void cudf_probe(GPUColumn **probe_keys, cudf::hash_join* hash_table, int num_keys, uint64_t*& row_ids_left, uint64_t*& row_ids_right, uint64_t*& count);

void cudf_build(GPUColumn **build_keys, cudf::hash_join*& hash_table, int num_keys);

void cudf_mixed_join(GPUColumn** probe_columns, GPUColumn** build_columns, const vector<JoinCondition>& conditions, JoinType join_type, uint64_t*& row_ids_left, uint64_t*& row_ids_right, uint64_t*& count);

void probeHashTable(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids_left, uint64_t* &row_ids_right, uint64_t* &count, 
			uint64_t N, int* condition_mode, int num_keys, bool is_right);

void probeHashTableRightSemiAnti(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int* condition_mode, int num_keys);

void probeHashTableSingleMatch(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids_left, uint64_t* &row_ids_right, 
            uint64_t* &count, uint64_t N, int* condition_mode, int num_keys, int join_mode);

void probeHashTableRightSemiAntiSingleMatch(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int* condition_mode, int num_keys);

void probeHashTableMark(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint8_t* &output, uint64_t N, int* condition_mode, int num_keys);

void buildHashTable(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int* condition_mode, int num_keys, bool is_right);

void scanHashTableRight(unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids, uint64_t* &count, int join_mode, int num_keys);

class GPUPhysicalHashJoin : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::HASH_JOIN;

	struct JoinProjectionColumns {
		vector<idx_t> col_idxs;
		vector<LogicalType> col_types;
	};
public:
						   
	GPUPhysicalHashJoin(LogicalOperator &op, unique_ptr<GPUPhysicalOperator> left, unique_ptr<GPUPhysicalOperator> right,
	                 vector<JoinCondition> cond, JoinType join_type, const vector<idx_t> &left_projection_map,
	                 const vector<idx_t> &right_projection_map, vector<LogicalType> delim_types,
	                 idx_t estimated_cardinality, unique_ptr<JoinFilterPushdownInfo> pushdown_info);
	GPUPhysicalHashJoin(LogicalOperator &op, unique_ptr<GPUPhysicalOperator> left, unique_ptr<GPUPhysicalOperator> right,
	                 vector<JoinCondition> cond, JoinType join_type, idx_t estimated_cardinality);

	vector<JoinCondition> conditions;
	//! Scans where we should push generated filters into (if any)
	unique_ptr<JoinFilterPushdownInfo> filter_pushdown;

	//! Initialize HT for this operator
	void InitializeHashTable(ClientContext &context) const;

	//! The types of the join keys
	vector<LogicalType> condition_types;
	//! The type of the join
	JoinType join_type;

	//! The indices/types of the payload columns
	JoinProjectionColumns payload_columns;
	//! The indices/types of the lhs columns that need to be output
	JoinProjectionColumns lhs_output_columns;
	//! The indices/types of the rhs columns that need to be output
	JoinProjectionColumns rhs_output_columns;

	//! Duplicate eliminated types; only used for delim_joins (i.e. correlated subqueries)
	vector<LogicalType> delim_types;

	mutable bool unique_build_keys = false;

	mutable bool unique_probe_keys = false;

	mutable cudf::hash_join* cudf_hash_table;

	// OperatorResultType Execute(ExecutionContext &context, GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation,
	// 									GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorResultType Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const override;

	static void BuildJoinPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline, GPUPhysicalOperator &op, bool build_rhs = true);
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline);

	//! Join Keys statistics (optional)
	vector<unique_ptr<BaseStatistics>> join_stats;
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