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
#include "duckdb/execution/operator/aggregate/distinct_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/radix_partitioned_hashtable.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"

namespace duckdb {

uint64_t* createFixedSizeOffsets(size_t record_size, uint64_t num_rows);

void cudf_groupby(vector<shared_ptr<GPUColumn>>& keys, vector<shared_ptr<GPUColumn>>& aggregate_keys, uint64_t num_keys, uint64_t num_aggregates, AggregationType* agg_mode);

void cudf_duplicate_elimination(vector<shared_ptr<GPUColumn>>& keys, uint64_t num_keys);

template<typename T>
void combineColumns(T* a, T* b, T*& c, uint64_t N_a, uint64_t N_b);

void combineStrings(uint8_t* a, uint8_t* b, uint8_t*& c, 
        uint64_t* offset_a, uint64_t* offset_b, uint64_t*& offset_c, 
        uint64_t num_bytes_a, uint64_t num_bytes_b, uint64_t N_a, uint64_t N_b);

void combineMasks(cudf::bitmask_type* a, cudf::bitmask_type* b, cudf::bitmask_type*& c, uint64_t N_a, uint64_t N_b);

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

	shared_ptr<GPUIntermediateRelation> group_by_result;

public:
	// Source interface
	SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

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
	SinkResultType Sink(GPUIntermediateRelation &input_relation) const override;

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

private:
	static bool CheckGroupKeyTypesForSiriusImpl(const vector<shared_ptr<GPUColumn>> &columns);
};
} // namespace duckdb