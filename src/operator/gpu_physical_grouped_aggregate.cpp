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

#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_materialize.hpp"
#include "log/logging.hpp"
#include "utils.hpp"

namespace duckdb {

template <typename T>
shared_ptr<GPUColumn>
ResolveTypeCombineColumns(shared_ptr<GPUColumn> column1, shared_ptr<GPUColumn> column2, GPUBufferManager* gpuBufferManager) {
	T* combine;
	cudf::bitmask_type* combine_mask;
	T* a = reinterpret_cast<T*> (column1->data_wrapper.data);
	T* b = reinterpret_cast<T*> (column2->data_wrapper.data);
	combineColumns<T>(a, b, combine, column1->column_length, column2->column_length);
	combineMasks(column1->data_wrapper.validity_mask, column2->data_wrapper.validity_mask,
				combine_mask, column1->column_length, column2->column_length);
	shared_ptr<GPUColumn> result = make_shared_ptr<GPUColumn>(column1->column_length + column2->column_length, column1->data_wrapper.type, reinterpret_cast<uint8_t*>(combine), combine_mask);
	if (column1->is_unique && column2->is_unique) {
		result->is_unique = true;
	}
	return result;
}

shared_ptr<GPUColumn>
ResolveTypeCombineStrings(shared_ptr<GPUColumn> column1, shared_ptr<GPUColumn> column2, GPUBufferManager* gpuBufferManager) {
	uint8_t* combine;
	uint64_t* offset_combine;
	cudf::bitmask_type* combine_mask;
	uint8_t* a = column1->data_wrapper.data;
	uint8_t* b = column2->data_wrapper.data;
	uint64_t* offset_a = column1->data_wrapper.offset;
	uint64_t* offset_b = column2->data_wrapper.offset;
	uint64_t num_bytes_a = column1->data_wrapper.num_bytes;
	uint64_t num_bytes_b = column2->data_wrapper.num_bytes;

	combineStrings(a, b, combine, offset_a, offset_b, offset_combine, num_bytes_a, num_bytes_b, column1->column_length, column2->column_length);
	combineMasks(column1->data_wrapper.validity_mask, column2->data_wrapper.validity_mask,
				combine_mask, column1->column_length, column2->column_length);
	shared_ptr<GPUColumn> result = make_shared_ptr<GPUColumn>(column1->column_length + column2->column_length, GPUColumnType(GPUColumnTypeId::VARCHAR), combine, offset_combine, num_bytes_a + num_bytes_b, true, combine_mask);
	if (column1->is_unique && column2->is_unique) {
		result->is_unique = true;
	}
	return result;	
}

shared_ptr<GPUColumn>
CombineColumns(shared_ptr<GPUColumn> column1, shared_ptr<GPUColumn> column2, GPUBufferManager* gpuBufferManager) {
    switch(column1->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT32:
				return ResolveTypeCombineColumns<int32_t>(column1, column2, gpuBufferManager);
      case GPUColumnTypeId::INT64:
				return ResolveTypeCombineColumns<uint64_t>(column1, column2, gpuBufferManager);
      case GPUColumnTypeId::FLOAT64:
				return ResolveTypeCombineColumns<double>(column1, column2, gpuBufferManager);
      case GPUColumnTypeId::VARCHAR:
				return ResolveTypeCombineStrings(column1, column2, gpuBufferManager);
	  default:
        throw NotImplementedException("Unsupported sirius column type in `CombineColumns: %d",
																			static_cast<int>(column1->data_wrapper.type.id()));
    }
}

void
HandleGroupByAggregateCuDF(vector<shared_ptr<GPUColumn>> &group_by_keys, vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates, int num_group_keys) {

	AggregationType* agg_mode = gpuBufferManager->customCudaHostAlloc<AggregationType>(aggregates.size());
	SIRIUS_LOG_DEBUG("Handling group by aggregate expression");
	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
		if (expr.IsDistinct()) {
			if (expr.function.name.compare("count") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
				agg_mode[agg_idx] = AggregationType::COUNT_DISTINCT;
			} else {
				SIRIUS_LOG_DEBUG("Grouped aggregate function (distinct)  not supported: {}", expr.function.name);
				throw NotImplementedException("Grouped aggregate function (distinct) not supported: %s", expr.function.name);
			}
		} else {
			if (expr.function.name.compare("count") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr && aggregate_keys[agg_idx]->column_length == 0) {
				agg_mode[agg_idx] = AggregationType::COUNT;
			} else if (expr.function.name.compare("sum") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr && aggregate_keys[agg_idx]->column_length == 0) {
				agg_mode[agg_idx] = AggregationType::SUM;
			} else if (expr.function.name.compare("sum") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
				agg_mode[agg_idx] = AggregationType::SUM;
			} else if (expr.function.name.compare("sum_no_overflow") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr && aggregate_keys[agg_idx]->column_length == 0) {
				agg_mode[agg_idx] = AggregationType::SUM;
			} else if (expr.function.name.compare("sum_no_overflow") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
				agg_mode[agg_idx] = AggregationType::SUM;
				if (aggregate_keys[agg_idx]->data_wrapper.type.id() == GPUColumnTypeId::INT32) {
					SIRIUS_LOG_DEBUG("Converting INT32 to INT64 for sum_no_overflow");
					uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(aggregate_keys[agg_idx]->column_length, 0, 0);
					convertInt32ToInt64(aggregate_keys[agg_idx]->data_wrapper.data, reinterpret_cast<uint8_t*>(temp), aggregate_keys[agg_idx]->column_length);
					aggregate_keys[agg_idx]->data_wrapper.data = reinterpret_cast<uint8_t*>(temp);
					aggregate_keys[agg_idx]->data_wrapper.type = GPUColumnType(GPUColumnTypeId::INT64);
					aggregate_keys[agg_idx]->data_wrapper.num_bytes = aggregate_keys[agg_idx]->data_wrapper.num_bytes * 2;
				}
			} else if (expr.function.name.compare("avg") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
				agg_mode[agg_idx] = AggregationType::AVERAGE;
			} else if (expr.function.name.compare("max") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
				agg_mode[agg_idx] = AggregationType::MAX;
			} else if (expr.function.name.compare("min") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
				agg_mode[agg_idx] = AggregationType::MIN;
			} else if (expr.function.name.compare("count_star") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr) {
				agg_mode[agg_idx] = AggregationType::COUNT_STAR;
			} else if (expr.function.name.compare("count") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
				agg_mode[agg_idx] = AggregationType::COUNT;
			} else {
				SIRIUS_LOG_DEBUG("Grouped aggregate function (not distinct) not supported: {}", expr.function.name);
				throw NotImplementedException("Grouped aggregate (not distinct) function not supported: %s", expr.function.name);
			}
		}
	}
	
	cudf_groupby(group_by_keys, aggregate_keys, num_group_keys, aggregates.size(), agg_mode);
}

void
HandleDistinctGroupByCuDF(vector<shared_ptr<GPUColumn>> &group_by_keys, vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, DistinctAggregateCollectionInfo &distinct_info, int num_group_keys) {
	AggregationType* distinct_mode = gpuBufferManager->customCudaHostAlloc<AggregationType>(distinct_info.indices.size());

	for (int idx = 0; idx < distinct_info.indices.size(); idx++) {
		auto distinct_idx = distinct_info.indices[idx];
		auto& expr = distinct_info.aggregates[distinct_idx]->Cast<BoundAggregateExpression>();
		if (expr.function.name.compare("count") == 0 && aggregate_keys[idx]->data_wrapper.data != nullptr) {
			distinct_mode[idx] = AggregationType::COUNT_DISTINCT;
		} else if (aggregate_keys[idx]->data_wrapper.data == nullptr) {
			throw NotImplementedException("Count distinct with null column not supported yet");		
		} else {
			throw NotImplementedException("Aggregate function not supported");
		}
	}

	cudf_groupby(group_by_keys, aggregate_keys, num_group_keys, distinct_info.indices.size(), distinct_mode);
}

void
HandleDuplicateEliminationCuDF(vector<shared_ptr<GPUColumn>> &group_by_keys, GPUBufferManager* gpuBufferManager, int num_group_keys) {
	cudf_duplicate_elimination(group_by_keys, num_group_keys);
}

static vector<LogicalType> CreateGroupChunkTypes(vector<unique_ptr<Expression>> &groups) {
	set<idx_t> group_indices;

	if (groups.empty()) {
		return {};
	}

	for (auto &group : groups) {
		D_ASSERT(group->type == ExpressionType::BOUND_REF);
		auto &bound_ref = group->Cast<BoundReferenceExpression>();
		group_indices.insert(bound_ref.index);
	}
	idx_t highest_index = *group_indices.rbegin();
	vector<LogicalType> types(highest_index + 1, LogicalType::SQLNULL);
	for (auto &group : groups) {
		auto &bound_ref = group->Cast<BoundReferenceExpression>();
		types[bound_ref.index] = bound_ref.return_type;
	}
	return types;
}

GPUPhysicalGroupedAggregate::GPUPhysicalGroupedAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions, idx_t estimated_cardinality)
    : GPUPhysicalGroupedAggregate(context, std::move(types), std::move(expressions), {}, estimated_cardinality) {


}

GPUPhysicalGroupedAggregate::GPUPhysicalGroupedAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions,
                                             vector<unique_ptr<Expression>> groups_p, idx_t estimated_cardinality)
    : GPUPhysicalGroupedAggregate(context, std::move(types), std::move(expressions), std::move(groups_p), {}, {},
                            estimated_cardinality) {
}

//expressions is the list of aggregates to be computed. Each aggregates has a bound_ref expression to a column
//groups_p is the list of group by columns. Each group by column is a bound_ref expression to a column
//grouping_sets_p is the list of grouping set. Each grouping set is a set of indexes to the group by columns. Seems like DuckDB group the groupby columns into several sets and for every grouping set there is one radix_table
//grouping_functions_p is a list of indexes to the groupby expressions (groups_p) for each grouping_sets. The first level of the vector is the grouping set and the second level is the indexes to the groupby expression for that set.
GPUPhysicalGroupedAggregate::GPUPhysicalGroupedAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions,
                                             vector<unique_ptr<Expression>> groups_p,
                                             vector<GroupingSet> grouping_sets_p,
                                             vector<unsafe_vector<idx_t>> grouping_functions_p,
                                             idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::HASH_GROUP_BY, std::move(types), estimated_cardinality),
      grouping_sets(std::move(grouping_sets_p)) {

		// get a list of all aggregates to be computed
		const idx_t group_count = groups_p.size();
		if (grouping_sets.empty()) {
			GroupingSet set;
			for (idx_t i = 0; i < group_count; i++) {
				set.insert(i);
			}
			grouping_sets.push_back(std::move(set));
		}
		input_group_types = CreateGroupChunkTypes(groups_p);

		grouped_aggregate_data.InitializeGroupby(std::move(groups_p), std::move(expressions),
												std::move(grouping_functions_p));

		auto &aggregates = grouped_aggregate_data.aggregates;
		// filter_indexes must be pre-built, not lazily instantiated in parallel...
		// Because everything that lives in this class should be read-only at execution time
		idx_t aggregate_input_idx = 0;
		for (idx_t i = 0; i < aggregates.size(); i++) {
			auto &aggregate = aggregates[i];
			auto &aggr = aggregate->Cast<BoundAggregateExpression>();
			aggregate_input_idx += aggr.children.size();
			if (aggr.aggr_type == AggregateType::DISTINCT) {
				distinct_filter.push_back(i);
			} else if (aggr.aggr_type == AggregateType::NON_DISTINCT) {
				non_distinct_filter.push_back(i);
			} else { // LCOV_EXCL_START
				throw NotImplementedException("AggregateType not implemented in PhysicalHashAggregate");
			} // LCOV_EXCL_STOP
		}

		for (idx_t i = 0; i < aggregates.size(); i++) {
			auto &aggregate = aggregates[i];
			auto &aggr = aggregate->Cast<BoundAggregateExpression>();
			if (aggr.filter) {
				auto &bound_ref_expr = aggr.filter->Cast<BoundReferenceExpression>();
				if (!filter_indexes.count(aggr.filter.get())) {
					// Replace the bound reference expression's index with the corresponding index of the payload chunk
					//TODO: Still not quite sure why duckdb replace the index
					filter_indexes[aggr.filter.get()] = bound_ref_expr.index;
					bound_ref_expr.index = aggregate_input_idx;
				}
				aggregate_input_idx++;
			}
		}

		distinct_collection_info = DistinctAggregateCollectionInfo::Create(grouped_aggregate_data.aggregates);

		for (idx_t i = 0; i < grouping_sets.size(); i++) {
			groupings.emplace_back(grouping_sets[i], grouped_aggregate_data, distinct_collection_info);
		}

	//The output of groupby is ordered as the grouping columns first followed by the aggregate columns
	//See RadixHTLocalSourceState::Scan for more details
	idx_t total_output_columns = 0;
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		total_output_columns++;
	}
	total_output_columns += grouped_aggregate_data.GroupCount();
	group_by_result = make_shared_ptr<GPUIntermediateRelation>(total_output_columns);
}

SinkResultType
GPUPhysicalGroupedAggregate::Sink(GPUIntermediateRelation& input_relation) const {
  	SIRIUS_LOG_DEBUG("Perform groupby and aggregation");

	auto start = std::chrono::high_resolution_clock::now();

	// if (distinct_collection_info) {
	// 	SinkDistinct(input_relation);
	// 	return SinkResultType::FINISHED;
	// }

	uint64_t column_size = 0;
	for (int i = 0; i < input_relation.columns.size(); i++) {
		if (input_relation.columns[i] != nullptr) {
			if (input_relation.columns[i]->row_ids != nullptr) {
				column_size = input_relation.columns[i]->row_id_count;
			} else if (input_relation.columns[i]->data_wrapper.data != nullptr) {
				column_size = input_relation.columns[i]->column_length;
			}
			break;
		} else {
			throw NotImplementedException("Input relation is null");
		}
	}

	// DataChunk &aggregate_input_chunk = local_state.aggregate_input_chunk;
	auto &aggregates = grouped_aggregate_data.aggregates;
	idx_t aggregate_input_idx = 0;

	if (groupings.size() > 1) throw NotImplementedException("Multiple groupings not supported yet");

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	uint64_t num_group_keys = grouped_aggregate_data.groups.size();
	vector<shared_ptr<GPUColumn>> group_by_column(grouped_aggregate_data.groups.size());
	vector<shared_ptr<GPUColumn>> aggregate_column(aggregates.size());
	for (int i = 0; i < grouped_aggregate_data.groups.size(); i++) {
		group_by_column[i] = nullptr;
	}
	for (int i = 0; i < aggregates.size(); i++) {
		aggregate_column[i] = nullptr;
	}

	// Reading groupby columns based on the grouping set
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping = groupings[i];
		int idx = 0;
		for (auto &group_idx : grouping_sets[i]) {
			// Retrieve the expression containing the index in the input chunk
			auto &group = grouped_aggregate_data.groups[group_idx];
			D_ASSERT(group->type == ExpressionType::BOUND_REF);
			auto &bound_ref_expr = group->Cast<BoundReferenceExpression>();
			SIRIUS_LOG_DEBUG("Passing input column index {} to group by column index {}", bound_ref_expr.index, idx);
			group_by_column[idx] = HandleMaterializeExpression(input_relation.columns[bound_ref_expr.index], gpuBufferManager);
			idx++;
		}
	}

	int aggr_idx = 0;
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		SIRIUS_LOG_DEBUG("Aggregate type: {}", aggr.function.name);
		if (aggr.filter) throw NotImplementedException("Aggregates with filters not supported yet");
		if (aggr.children.size() > 1) throw NotImplementedException("Aggregates with multiple children not supported yet");
		for (auto &child_expr : aggr.children) {
			D_ASSERT(child_expr->type == ExpressionType::BOUND_REF);
			auto &bound_ref_expr = child_expr->Cast<BoundReferenceExpression>();
			SIRIUS_LOG_DEBUG("Passing input column index {} to aggregate column index {}", bound_ref_expr.index, aggr_idx);
			aggregate_column[aggr_idx] = HandleMaterializeExpression(input_relation.columns[bound_ref_expr.index], gpuBufferManager);
		}
		aggr_idx++;
	}

	aggr_idx = 0;
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		if (aggr.children.size() == 0) {
			//we have a count(*) aggregate
			SIRIUS_LOG_DEBUG("Passing * aggregate to index {} in aggregation result", aggr_idx);
			aggregate_column[aggr_idx] = make_shared_ptr<GPUColumn>(column_size, GPUColumnType(GPUColumnTypeId::INT64), nullptr, nullptr);
		}
		if (aggr.filter) {
			throw NotImplementedException("Filter not supported yet");
			auto it = filter_indexes.find(aggr.filter.get());
			D_ASSERT(it != filter_indexes.end());
			SIRIUS_LOG_DEBUG("Reading aggregation filter from index {}", it->second);
			input_relation.checkLateMaterialization(it->second);
		}
		aggr_idx++;
	}

	// Execute only if input is not empty
	if (column_size > 0) {
		// bool can_use_sirius_impl = CheckGroupKeyTypesForSiriusImpl(group_by_column);
		if (aggregates.size() == 0) {
			// if (can_use_sirius_impl) {
			// 	HandleDuplicateElimination(group_by_column, gpuBufferManager, num_group_keys);
			// } else {
				// HandleGroupByAggregateCuDF(group_by_column, aggregate_column, gpuBufferManager, aggregates, num_group_keys);
			// }
			if (group_by_column[0]->column_length > INT32_MAX) {
				throw NotImplementedException("Group by column length or aggregate column length is too large for CuDF");
			} else {
				HandleDuplicateEliminationCuDF(group_by_column, gpuBufferManager, num_group_keys);
			}
		} else {
			if (group_by_column[0]->column_length > INT32_MAX || aggregate_column[0]->column_length > INT32_MAX) {
				throw NotImplementedException("Group by column length or aggregate column length is too large for CuDF");
			} else {
				HandleGroupByAggregateCuDF(group_by_column, aggregate_column, gpuBufferManager, aggregates, num_group_keys);
			}
		}
	}
	
	// Reading groupby columns based on the grouping set
	for (idx_t i = 0; i < groupings.size(); i++) {
		for (int idx = 0; idx < grouping_sets[i].size(); idx++) {
			//TODO: has to fix this for columns with partially NULL values
			if (group_by_result->columns[idx] == nullptr) {
				SIRIUS_LOG_DEBUG("Passing group by column {} to group by result column {}", idx, idx);
				group_by_result->columns[idx] = group_by_column[idx];
				group_by_result->columns[idx]->row_ids = nullptr;
				group_by_result->columns[idx]->row_id_count = 0;
			} else if (group_by_result->columns[idx] != nullptr) {
				if (group_by_column[idx]->data_wrapper.data != nullptr && group_by_result->columns[idx]->data_wrapper.data != nullptr) {
					SIRIUS_LOG_DEBUG("Combining group by column {} with group by result column {}", idx, idx);
					group_by_result->columns[idx] = CombineColumns(group_by_result->columns[idx], group_by_column[idx], gpuBufferManager);
				} else if (group_by_column[idx]->data_wrapper.data != nullptr && group_by_result->columns[idx]->data_wrapper.data == nullptr) {
					SIRIUS_LOG_DEBUG("Passing group by column {} to group by result column {}", idx, idx);
					group_by_result->columns[idx] = group_by_column[idx];
					group_by_result->columns[idx]->row_ids = nullptr;
					group_by_result->columns[idx]->row_id_count = 0;
				} else {
					SIRIUS_LOG_DEBUG("Group by column {} is null, skipping", idx);
				}
			}

		}
	}

	for (int aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		//TODO: has to fix this for columns with partially NULL values
		if (group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] == nullptr) {
			SIRIUS_LOG_DEBUG("Passing aggregate column {} to group by result column {}", aggr_idx, grouped_aggregate_data.groups.size() + aggr_idx);
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] = aggregate_column[aggr_idx];
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx]->row_ids = nullptr;
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx]->row_id_count = 0;
		} else if (group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] != nullptr) {
			if (aggregate_column[aggr_idx]->data_wrapper.data != nullptr && group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx]->data_wrapper.data != nullptr) {
				SIRIUS_LOG_DEBUG("Combining aggregate column {} with group by result column {}", aggr_idx, grouped_aggregate_data.groups.size() + aggr_idx);
				group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] = CombineColumns(group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx], aggregate_column[aggr_idx], gpuBufferManager);
			} else if (aggregate_column[aggr_idx]->data_wrapper.data != nullptr && group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx]->data_wrapper.data == nullptr) {
				SIRIUS_LOG_DEBUG("Passing aggregate column {} to group by result column {}", aggr_idx, grouped_aggregate_data.groups.size() + aggr_idx);
				group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] = aggregate_column[aggr_idx];
				group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx]->row_ids = nullptr;
				group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx]->row_id_count = 0;
			} else {
				SIRIUS_LOG_DEBUG("Aggregate column {} is null, skipping", aggr_idx);
			}
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Group Aggregate Sink time: {:.2f} ms", duration.count()/1000.0);
	
  	return SinkResultType::FINISHED;
}

SourceResultType
GPUPhysicalGroupedAggregate::GetData(GPUIntermediateRelation &output_relation) const {
	if (groupings.size() > 1) throw NotImplementedException("Multiple groupings not supported yet");

	for (int col = 0; col < group_by_result->columns.size(); col++) {
		SIRIUS_LOG_DEBUG("Writing group by result to column {}", col);
		// output_relation.columns[col] = group_by_result->columns[col];
		bool old_unique = group_by_result->columns[col]->is_unique;
		if (group_by_result->columns[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
			// output_relation.columns[col] = make_shared_ptr<GPUColumn>(
			// 	group_by_result->columns[col]->column_length, group_by_result->columns[col]->data_wrapper.type,
			// 	group_by_result->columns[col]->data_wrapper.data, group_by_result->columns[col]->data_wrapper.offset,
			// 	group_by_result->columns[col]->data_wrapper.num_bytes, true, group_by_result->columns[col]->data_wrapper.validity_mask);
			output_relation.columns[col] = make_shared_ptr<GPUColumn>(group_by_result->columns[col]);
		} else {
			// output_relation.columns[col] = make_shared_ptr<GPUColumn>(
			// 	group_by_result->columns[col]->column_length, group_by_result->columns[col]->data_wrapper.type,
			// 	group_by_result->columns[col]->data_wrapper.data, nullptr, group_by_result->columns[col]->data_wrapper.num_bytes,
			// 	false, group_by_result->columns[col]->data_wrapper.validity_mask);
			output_relation.columns[col] = make_shared_ptr<GPUColumn>(group_by_result->columns[col]);
		}
		output_relation.columns[col]->is_unique = old_unique;
	}

	return SourceResultType::FINISHED;
}

bool GPUPhysicalGroupedAggregate::CheckGroupKeyTypesForSiriusImpl(const vector<shared_ptr<GPUColumn>> &columns) {
	for (const auto& column: columns) {
		if (column->data_wrapper.type.id() != GPUColumnTypeId::INT64) {
			return false;
		}
	}
	return true;
}

} // namespace duckdb