#pragma once

#include "gpu_physical_operator.hpp"
#include "duckdb/execution/operator/aggregate/distinct_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/radix_partitioned_hashtable.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"

namespace duckdb {

template <typename T, typename V>
void groupedAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template <typename T>
void groupedWithoutAggregate(uint8_t **keys, uint64_t* count, uint64_t N, uint64_t num_keys);

template <typename T, typename V>
void groupedDistinctAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* distinct_mode);

template <typename V>
void groupedStringAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template <typename V>
void groupedStringAggregateV2(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template <typename V>
void groupedStringAggregateV3(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template<typename T>
void combineColumns(T* a, T* b, T* c, uint64_t N_a, uint64_t N_b);

void combineStrings(uint8_t* a, uint8_t* b, uint8_t* c, 
        uint64_t* offset_a, uint64_t* offset_b, uint64_t* offset_c, 
        uint64_t num_bytes_a, uint64_t num_bytes_b, uint64_t N_a, uint64_t N_b);

class ClientContext;

class GPUPhysicalGroupedAggregate : public GPUPhysicalOperator {
public:

	GPUPhysicalGroupedAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions, idx_t estimated_cardinality);

	GPUPhysicalGroupedAggregate(ClientContext &context, vector<LogicalType> types, vector<unique_ptr<Expression>> expressions,
	                      vector<unique_ptr<Expression>> groups, idx_t estimated_cardinality);

    GPUPhysicalGroupedAggregate(ClientContext &context, vector<LogicalType> types, vector<unique_ptr<Expression>> expressions,
	                      vector<unique_ptr<Expression>> groups, vector<GroupingSet> grouping_sets,
	                      vector<unsafe_vector<idx_t>> grouping_functions, idx_t estimated_cardinality);

	//! The grouping sets
	GroupedAggregateData grouped_aggregate_data;

	vector<GroupingSet> grouping_sets;
	//! The radix partitioned hash tables (one per grouping set)
	vector<HashAggregateGroupingData> groupings;
	unique_ptr<DistinctAggregateCollectionInfo> distinct_collection_info;
	//! A recreation of the input chunk, with nulls for everything that isnt a group
	vector<LogicalType> input_group_types;

	// Filters given to Sink and friends
	unsafe_vector<idx_t> non_distinct_filter;
	unsafe_vector<idx_t> distinct_filter;

	unordered_map<Expression *, size_t> filter_indexes;

	GPUIntermediateRelation* group_by_result;

public:
	// // Source interface
	// unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	// unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	//                                                  GlobalSourceState &gstate) const override;
	// SourceResultType GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation, OperatorSourceInput &input) const override;
	SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

	// double GetProgress(ClientContext &context, GlobalSourceState &gstate) const override;

	//Source interface
	bool IsSource() const override {
		return true;
	}
	bool ParallelSource() const override {
		return true;
	}

	OrderPreservationType SourceOrder() const override {
		return OrderPreservationType::NO_ORDER;
	}

public:
	// Sink interface
	// SinkResultType Sink(ExecutionContext &context, GPUIntermediateRelation& input_relation, OperatorSinkInput &input) const override;
	SinkResultType Sink(GPUIntermediateRelation &input_relation) const override;
	// SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	// SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	//                           OperatorSinkFinalizeInput &input) const override;
	// SinkFinalizeType FinalizeInternal(Pipeline &pipeline, Event &event, ClientContext &context, GlobalSinkState &gstate,
	//                                   bool check_distinct) const;

	// unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	// unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	// Sink interface
	bool IsSink() const override {
		return true;
	}

	bool ParallelSink() const override {
		return true;
	}

	bool SinkOrderDependent() const override {
		return false;
	}

// public:
// 	string ParamsToString() const override;
// 	//! Toggle multi-scan capability on a hash table, which prevents the scan of the aggregate from being destructive
// 	//! If this is not toggled the GetData method will destroy the hash table as it is scanning it
// 	static void SetMultiScan(GlobalSinkState &state);

// private:
// 	//! When we only have distinct aggregates, we can delay adding groups to the main ht
// 	bool CanSkipRegularSink() const;

// 	//! Finalize the distinct aggregates
// 	SinkFinalizeType FinalizeDistinct(Pipeline &pipeline, Event &event, ClientContext &context,
// 	                                  GlobalSinkState &gstate) const;
// 	//! Combine the distinct aggregates
// 	void CombineDistinct(ExecutionContext &context, OperatorSinkCombineInput &input) const;
// 	//! Sink the distinct aggregates for a single grouping
	// void SinkDistinctGrouping(ExecutionContext &context, GPUIntermediateRelation &input_relation, OperatorSinkInput &input,
	//                           idx_t grouping_idx) const;
	void SinkDistinctGrouping(GPUIntermediateRelation &input_relation, idx_t grouping_idx) const;
// 	//! Sink the distinct aggregates
	// void SinkDistinct(ExecutionContext &context, GPUIntermediateRelation& input_relation, OperatorSinkInput &input) const;
	void SinkDistinct(GPUIntermediateRelation &input_relation) const;
// 	//! Create groups in the main ht for groups that would otherwise get filtered out completely
// 	SinkResultType SinkGroupsOnly(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate,
// 	                              DataChunk &input) const;
};
} // namespace duckdb