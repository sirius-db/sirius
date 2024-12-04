#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_materialize.hpp"

namespace duckdb {

template <typename T>
GPUColumn*
ResolveTypeCombineColumns(GPUColumn* column1, GPUColumn* column2, GPUBufferManager* gpuBufferManager) {
	T* combine = gpuBufferManager->customCudaMalloc<T>(column1->column_length + column2->column_length, 0, 0);
	T* a = reinterpret_cast<T*> (column1->data_wrapper.data);
	T* b = reinterpret_cast<T*> (column2->data_wrapper.data);
	combineColumns<T>(a, b, combine, column1->column_length, column2->column_length);
	return new GPUColumn(column1->column_length + column2->column_length, column1->data_wrapper.type, reinterpret_cast<uint8_t*>(combine));
}

GPUColumn*
CombineColumns(GPUColumn* column1, GPUColumn* column2, GPUBufferManager* gpuBufferManager) {
    switch(column1->data_wrapper.type) {
      case ColumnType::INT64:
		return ResolveTypeCombineColumns<uint64_t>(column1, column2, gpuBufferManager);
		break;
      case ColumnType::FLOAT64:
		return ResolveTypeCombineColumns<double>(column1, column2, gpuBufferManager);
		break;
      default:
        throw NotImplementedException("Unsupported column type");
    }
}

template <typename T, typename V>
void
ResolveTypeGroupByAggregateExpression(GPUColumn** &group_by_keys, GPUColumn** &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates, int num_group_keys) {
	uint64_t count[1];
	count[0] = 0;
	uint8_t** group_by_data = new uint8_t*[num_group_keys];
	uint8_t** aggregate_data = new uint8_t*[aggregates.size()];

	for (int group = 0; group < num_group_keys; group++) {
		group_by_data[group] = (group_by_keys[group]->data_wrapper.data);
	}
	size_t size = group_by_keys[0]->column_length;

	int* agg_mode = new int[aggregates.size()];

	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
		if (expr.function.name.compare("count") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr) {
			agg_mode[agg_idx] = 5;
			aggregate_data[agg_idx] = nullptr;
		} else if (expr.function.name.compare("sum") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr) {
			agg_mode[agg_idx] = 5;
			aggregate_data[agg_idx] = nullptr;
		} else if (expr.function.name.compare("sum") == 0) {
			agg_mode[agg_idx] = 0;
			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);
		} else if (expr.function.name.compare("avg") == 0) {
			if (aggregate_keys[agg_idx]->data_wrapper.type != ColumnType::FLOAT64) throw NotImplementedException("Column type is supposed to be double");
			agg_mode[agg_idx] = 1;
			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);
		} else if (expr.function.name.compare("max") == 0) {
			agg_mode[agg_idx] = 2;
			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);
		} else if (expr.function.name.compare("min") == 0) {
			agg_mode[agg_idx] = 3;
			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);
		} else if (expr.function.name.compare("count_star") == 0) {
			agg_mode[agg_idx] = 4;
			aggregate_data[agg_idx] = nullptr;
		} else if (expr.function.name.compare("count") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
			agg_mode[agg_idx] = 4;
			aggregate_data[agg_idx] = nullptr;
		} else {
			throw NotImplementedException("Aggregate function not supported");
		}
	}

	groupedAggregate<T, V>(group_by_data, aggregate_data, count, size, num_group_keys, aggregates.size(), agg_mode);

	// Reading groupby columns based on the grouping set
	for (idx_t group = 0; group < num_group_keys; group++) {
		group_by_keys[group] = new GPUColumn(count[0], group_by_keys[group]->data_wrapper.type, reinterpret_cast<uint8_t*>(group_by_data[group]));
	}

	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
		if (expr.function.name.compare("count_star") == 0 || expr.function.name.compare("count") == 0) {
			aggregate_keys[agg_idx] = new GPUColumn(count[0], ColumnType::INT64, reinterpret_cast<uint8_t*>(aggregate_data[agg_idx]));
		} else {
			aggregate_keys[agg_idx] = new GPUColumn(count[0], aggregate_keys[agg_idx]->data_wrapper.type, reinterpret_cast<uint8_t*>(aggregate_data[agg_idx]));
		}
	}
}

void
HandleGroupByAggregateExpression(GPUColumn** &group_by_keys, GPUColumn** &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates, int num_group_keys) {
    switch(group_by_keys[0]->data_wrapper.type) {
      case ColumnType::INT64:
	  	if (aggregate_keys[0]->data_wrapper.type == ColumnType::INT64) {
			ResolveTypeGroupByAggregateExpression<uint64_t, uint64_t>(group_by_keys, aggregate_keys, gpuBufferManager, aggregates, num_group_keys);
		} else if (aggregate_keys[0]->data_wrapper.type == ColumnType::FLOAT64) {
			ResolveTypeGroupByAggregateExpression<uint64_t, double>(group_by_keys, aggregate_keys, gpuBufferManager, aggregates, num_group_keys);
		} else throw NotImplementedException("Unsupported column type");
		break;
      case ColumnType::FLOAT64:
	  	throw NotImplementedException("Unsupported column type");
      default:
        throw NotImplementedException("Unsupported column type");
    }
}

template <typename T>
void ResolveTypeDuplicateElimination(GPUColumn** &group_by_keys, GPUBufferManager* gpuBufferManager, int num_group_keys) {
	uint64_t count[1];
	count[0] = 0;
	uint8_t** group_by_data = new uint8_t*[num_group_keys];

	for (int group = 0; group < num_group_keys; group++) {
		group_by_data[group] = (group_by_keys[group]->data_wrapper.data);
	}
	size_t size = group_by_keys[0]->column_length;

	groupedWithoutAggregate<T>(group_by_data, count, size, num_group_keys);

	// Reading groupby columns based on the grouping set
	for (idx_t group = 0; group < num_group_keys; group++) {
		group_by_keys[group] = new GPUColumn(count[0], group_by_keys[group]->data_wrapper.type, reinterpret_cast<uint8_t*>(group_by_data[group]));
	}
}

void HandleDuplicateElimination(GPUColumn** &group_by_keys, GPUBufferManager* gpuBufferManager, int num_group_keys) {
    switch(group_by_keys[0]->data_wrapper.type) {
      case ColumnType::INT64:
	  	ResolveTypeDuplicateElimination<uint64_t>(group_by_keys, gpuBufferManager, num_group_keys);
		break;
      case ColumnType::FLOAT64:
	  	throw NotImplementedException("Unsupported column type");
      default:
        throw NotImplementedException("Unsupported column type");
    }
}

template <typename T, typename V>
void ResolveTypeDistinctGroupBy(GPUColumn** &group_by_keys, GPUColumn** &aggregate_keys, GPUBufferManager* gpuBufferManager, DistinctAggregateCollectionInfo &distinct_info, int num_group_keys) {
	uint64_t count[1];
	count[0] = 0;
	uint8_t** group_by_data = new uint8_t*[num_group_keys];
	uint8_t** distinct_aggregate_data = new uint8_t*[distinct_info.indices.size()];

	for (int group = 0; group < num_group_keys; group++) {
		group_by_data[group] = (group_by_keys[group]->data_wrapper.data);
	}
	size_t size = group_by_keys[0]->column_length;

	int* distinct_mode = new int[distinct_info.indices.size()];

	for (int idx = 0; idx < distinct_info.indices.size(); idx++) {
		auto distinct_idx = distinct_info.indices[idx];
		auto& expr = distinct_info.aggregates[distinct_idx]->Cast<BoundAggregateExpression>();
		if (expr.function.name.compare("count") == 0 && aggregate_keys[idx]->data_wrapper.data != nullptr) {
			distinct_mode[idx] = 0;
			distinct_aggregate_data[idx] = aggregate_keys[idx]->data_wrapper.data;
		} else if (aggregate_keys[idx]->data_wrapper.data == nullptr) {
			throw NotImplementedException("Count distinct with null column not supported yet");		
		} else {
			throw NotImplementedException("Aggregate function not supported");
		}
	}

	groupedDistinctAggregate<uint64_t, uint64_t>(group_by_data, distinct_aggregate_data, count, size, num_group_keys, distinct_info.indices.size(), distinct_mode);

	// Reading groupby columns based on the grouping set
	for (idx_t group = 0; group < num_group_keys; group++) {
		group_by_keys[group] = new GPUColumn(count[0], group_by_keys[group]->data_wrapper.type, reinterpret_cast<uint8_t*>(group_by_data[group]));
	}

	for (int idx = 0; idx < distinct_info.indices.size(); idx++) {
		auto distinct_idx = distinct_info.indices[idx];
		auto& expr = distinct_info.aggregates[distinct_idx]->Cast<BoundAggregateExpression>();
		if (expr.function.name.compare("count") == 0) {
			aggregate_keys[idx] = new GPUColumn(count[0], ColumnType::INT64, reinterpret_cast<uint8_t*>(distinct_aggregate_data[idx]));
		}
	}

}

void HandleDistinctGroupBy(GPUColumn** &group_by_keys, GPUColumn** &aggregate_keys, GPUBufferManager* gpuBufferManager, DistinctAggregateCollectionInfo &distinct_info, int num_group_keys) {
    switch(group_by_keys[0]->data_wrapper.type) {
      case ColumnType::INT64: {
	  	if (aggregate_keys[0]->data_wrapper.type == ColumnType::INT64) {
			ResolveTypeDistinctGroupBy<uint64_t, uint64_t>(group_by_keys, aggregate_keys, gpuBufferManager, distinct_info, num_group_keys);
		} else throw NotImplementedException("Unsupported column type");
		break;
	  } case ColumnType::FLOAT64:
	  	throw NotImplementedException("Unsupported column type");
      default:
        throw NotImplementedException("Unsupported column type");
    }
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
	group_by_result = new GPUIntermediateRelation(total_output_columns);
}

// SinkResultType
// GPUPhysicalGroupedAggregate::Sink(ExecutionContext &context, GPUIntermediateRelation& input_relation, OperatorSinkInput &input) const {

SinkResultType
GPUPhysicalGroupedAggregate::Sink(GPUIntermediateRelation& input_relation) const {
  	printf("Perform groupby and aggregation\n");

	if (distinct_collection_info) {
		SinkDistinct(input_relation);
		return SinkResultType::FINISHED;
	}

	// DataChunk &aggregate_input_chunk = local_state.aggregate_input_chunk;
	auto &aggregates = grouped_aggregate_data.aggregates;
	idx_t aggregate_input_idx = 0;

	if (groupings.size() > 1) throw NotImplementedException("Multiple groupings not supported yet");

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	// uint64_t** group_keys = new uint64_t*[grouped_aggregate_data.groups.size()];
	uint64_t num_group_keys = grouped_aggregate_data.groups.size();
	// double** aggregate_vals = new double*[aggregates.size()];
	GPUColumn** group_by_column = new GPUColumn*[grouped_aggregate_data.groups.size()];
	GPUColumn** aggregate_column = new GPUColumn*[aggregates.size()];
	for (int i = 0; i < grouped_aggregate_data.groups.size(); i++) {
		group_by_column[i] = nullptr;
	}
	for (int i = 0; i < aggregates.size(); i++) {
		aggregate_column[i] = nullptr;
	}
	int aggr_idx = 0;
	int size;

	// Reading groupby columns based on the grouping set
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping = groupings[i];
		int idx = 0;
		for (auto &group_idx : grouping_sets[i]) {
			// Retrieve the expression containing the index in the input chunk
			auto &group = grouped_aggregate_data.groups[group_idx];
			D_ASSERT(group->type == ExpressionType::BOUND_REF);
			auto &bound_ref_expr = group->Cast<BoundReferenceExpression>();
			printf("Reading groupby columns from index %d and passing it to index %d in groupby result\n", bound_ref_expr.index, bound_ref_expr.index);
			// input_relation.checkLateMaterialization(bound_ref_expr.index);
			// group_by_result->columns[idx] = input_relation.columns[bound_ref_expr.index];
			// idx++;

			// if (input_relation.checkLateMaterialization(bound_ref_expr.index)) {
			// 	uint64_t* temp = reinterpret_cast<uint64_t*> (input_relation.columns[bound_ref_expr.index]->data_wrapper.data);
			// 	uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (input_relation.columns[bound_ref_expr.index]->row_ids);
			// 	group_keys[idx] = gpuBufferManager->customCudaMalloc<uint64_t>(input_relation.columns[bound_ref_expr.index]->row_id_count, 0, 0);
			// 	materializeExpression<uint64_t>(temp, group_keys[idx], row_ids_input, input_relation.columns[bound_ref_expr.index]->row_id_count);
			// 	size = input_relation.columns[bound_ref_expr.index]->row_id_count;
			// } else {
			// 	group_keys[idx] = reinterpret_cast<uint64_t*> (input_relation.columns[bound_ref_expr.index]->data_wrapper.data);
			// 	size = input_relation.columns[bound_ref_expr.index]->column_length;
			// }
			group_by_column[idx] = HandleMaterializeExpression(input_relation.columns[bound_ref_expr.index], bound_ref_expr, gpuBufferManager);
			idx++;
		}
	}

	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		printf("Aggregate type: %s\n", aggr.function.name.c_str());
		if (aggr.children.size() > 1) throw NotImplementedException("Aggregates with multiple children not supported yet");
		for (auto &child_expr : aggr.children) {
			D_ASSERT(child_expr->type == ExpressionType::BOUND_REF);
			auto &bound_ref_expr = child_expr->Cast<BoundReferenceExpression>();
			printf("Reading aggregation column from index %d and passing it to index %d in groupby result\n", bound_ref_expr.index, grouped_aggregate_data.groups.size() + aggr_idx);
			aggregate_column[aggr_idx] = HandleMaterializeExpression(input_relation.columns[bound_ref_expr.index], bound_ref_expr, gpuBufferManager);
		}
		//here we probably have count(*) or sum(*) or something like that
		if (aggr.children.size() == 0) {
			// throw NotImplementedException("Aggregate without children not supported yet");
			printf("Passing * aggregate to index %d in groupby result\n", grouped_aggregate_data.groups.size() + aggr_idx);
			aggregate_column[aggr_idx] = new GPUColumn(0, ColumnType::INT64, nullptr);
		}
		aggr_idx++;
	}
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		if (aggr.filter) {
			throw NotImplementedException("Filter not supported yet");
			auto it = filter_indexes.find(aggr.filter.get());
			D_ASSERT(it != filter_indexes.end());
			printf("Reading aggregation filter from index %d\n", it->second);
			input_relation.checkLateMaterialization(it->second);
		}
	}

	uint64_t count[1];
	if (aggregates.size() == 0) {
		HandleDuplicateElimination(group_by_column, gpuBufferManager, num_group_keys);
	} else {
		HandleGroupByAggregateExpression(group_by_column, aggregate_column, gpuBufferManager, aggregates, num_group_keys);
	}
	
	// Reading groupby columns based on the grouping set
	for (idx_t i = 0; i < groupings.size(); i++) {
		for (int idx = 0; idx < grouping_sets[i].size(); idx++) {
			//TODO: has to fix this for columns with partially NULL values
			// group_by_result->columns[idx] = new GPUColumn(count[0], ColumnType::INT64, reinterpret_cast<uint8_t*>(group_keys[idx]));
			if (group_by_result->columns[idx] == nullptr && group_by_column[idx]->column_length > 0 && group_by_column[idx]->data_wrapper.data != nullptr) {
				group_by_result->columns[idx] = group_by_column[idx];
				group_by_result->columns[idx]->row_ids = nullptr;
				group_by_result->columns[idx]->row_id_count = 0;
			} else if (group_by_result->columns[idx] != nullptr && group_by_column[idx]->column_length > 0 && group_by_column[idx]->data_wrapper.data != nullptr) {
				// have to combine groupby from different meta pipelines
				// printf("%ld\n", group_by_column[idx]->column_length);
				group_by_result->columns[idx] = CombineColumns(group_by_result->columns[idx], group_by_column[idx], gpuBufferManager);
			}

		}
	}

	for (int aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		// group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] = new GPUColumn(count[0], ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(aggregate_vals[aggr_idx]));
		//TODO: has to fix this for columns with partially NULL values
		if (group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] == nullptr && aggregate_column[aggr_idx]->column_length > 0 && aggregate_column[aggr_idx]->data_wrapper.data != nullptr) {
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] = aggregate_column[aggr_idx];
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx]->row_ids = nullptr;
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx]->row_id_count = 0;
		} else if (group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] != nullptr && aggregate_column[aggr_idx]->column_length > 0 && aggregate_column[aggr_idx]->data_wrapper.data != nullptr) {
			// have to combine groupby from different meta pipelines
			// printf("%ld\n", aggregate_column[aggr_idx]->column_length);
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] = CombineColumns(group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx], aggregate_column[aggr_idx], gpuBufferManager);
		}
	}

  	return SinkResultType::FINISHED;
}

// SourceResultType
// GPUPhysicalGroupedAggregate::GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation, OperatorSourceInput &input) const {
SourceResultType
GPUPhysicalGroupedAggregate::GetData(GPUIntermediateRelation &output_relation) const {
//   printf("group by result size %d\n", group_by_result->columns.size());
	if (groupings.size() > 1) throw NotImplementedException("Multiple groupings not supported yet");

	for (int col = 0; col < group_by_result->columns.size(); col++) {
		printf("Writing group by result to column %d\n", col);
		// output_relation.columns[col] = group_by_result->columns[col];
		output_relation.columns[col] = new GPUColumn(group_by_result->columns[col]->column_length, group_by_result->columns[col]->data_wrapper.type, group_by_result->columns[col]->data_wrapper.data);
	}

  	return SourceResultType::FINISHED;
}

// void 
// GPUPhysicalGroupedAggregate::SinkDistinct(ExecutionContext &context, GPUIntermediateRelation& input_relation, OperatorSinkInput &input) const {
void
GPUPhysicalGroupedAggregate::SinkDistinct(GPUIntermediateRelation& input_relation) const {
	// throw NotImplementedException("Distinct not supported yet");
	if (groupings.size() > 1) throw NotImplementedException("Multiple groupings not supported yet");
	for (idx_t i = 0; i < groupings.size(); i++) {
		SinkDistinctGrouping(input_relation, i);
	}
}

// void
// GPUPhysicalGroupedAggregate:: SinkDistinctGrouping(ExecutionContext &context, GPUIntermediateRelation &input_relation, OperatorSinkInput &input,
	                        //   idx_t grouping_idx) const {
void
GPUPhysicalGroupedAggregate::SinkDistinctGrouping(GPUIntermediateRelation& input_relation, idx_t grouping_idx) const {
	auto &distinct_info = *distinct_collection_info;

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	uint64_t num_group_keys = grouped_aggregate_data.groups.size();
	GPUColumn** group_by_column = new GPUColumn*[grouped_aggregate_data.groups.size()];
	GPUColumn** distinct_aggregate_columns = new GPUColumn*[distinct_info.indices.size()];

	for (int i = 0; i < grouped_aggregate_data.groups.size(); i++) {
		group_by_column[i] = nullptr;
	}
	for (int i = 0; i < distinct_info.indices.size(); i++) {
		distinct_aggregate_columns[i] = nullptr;
	}

	for (idx_t group_idx = 0; group_idx < grouped_aggregate_data.groups.size(); group_idx++) {
		auto &group = grouped_aggregate_data.groups[group_idx];
		auto &bound_ref = group->Cast<BoundReferenceExpression>();
		printf("Reading groupby columns from index %d and passing it to index %d in groupby result\n", bound_ref.index, group_idx);
		// input_relation.checkLateMaterialization(bound_ref.index);
		// group_by_result->columns[group_idx] = input_relation.columns[bound_ref.index];
		group_by_column[group_idx] = HandleMaterializeExpression(input_relation.columns[bound_ref.index], bound_ref, gpuBufferManager);
	}

	int aggr_idx = 0;
	for (idx_t &idx : distinct_info.indices) {
		auto &aggregate = grouped_aggregate_data.aggregates[idx]->Cast<BoundAggregateExpression>();
		printf("Processing distinct aggregate %s\n", aggregate.function.name.c_str());
		// throw NotImplementedException("Distinct not supported yet");

		D_ASSERT(distinct_info.table_map.count(idx));

		if (aggregate.filter) {
			throw NotImplementedException("Filter not supported yet");
			auto it = filter_indexes.find(aggregate.filter.get());
      		printf("Reading filter columns from index %d\n", it->second);

			for (idx_t group_idx = 0; group_idx < grouped_aggregate_data.groups.size(); group_idx++) {
				auto &group = grouped_aggregate_data.groups[group_idx];
				auto &bound_ref = group->Cast<BoundReferenceExpression>();
				printf("Reading groupby columns from index %d and passing it to index %d in groupby result\n", bound_ref.index, group_idx);
				input_relation.checkLateMaterialization(bound_ref.index);
				group_by_result->columns[group_idx] = input_relation.columns[bound_ref.index];
			}
			for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
				auto &child = aggregate.children[child_idx];
				auto &bound_ref = child->Cast<BoundReferenceExpression>();
				printf("Reading aggregation column from index %d and passing it to index %d in groupby result\n", bound_ref.index, grouped_aggregate_data.groups.size() + idx);
				input_relation.checkLateMaterialization(bound_ref.index);
				group_by_result->columns[grouped_aggregate_data.groups.size() + idx] = input_relation.columns[bound_ref.index];
			}
		} else {

			if (aggregate.children.size() > 1) throw NotImplementedException("Aggregates with multiple children not supported yet");
			for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
				auto &child = aggregate.children[child_idx];
				auto &bound_ref = child->Cast<BoundReferenceExpression>();
				printf("Reading aggregation column from index %d and passing it to index %d in groupby result\n", bound_ref.index, grouped_aggregate_data.groups.size() + idx);
				// input_relation.checkLateMaterialization(bound_ref.index);
				// group_by_result->columns[grouped_aggregate_data.groups.size() + idx] = input_relation.columns[bound_ref.index];
				distinct_aggregate_columns[aggr_idx] = HandleMaterializeExpression(input_relation.columns[bound_ref.index], bound_ref, gpuBufferManager);
			}
			aggr_idx++;
		}
	}

	uint64_t count[1];
	HandleDistinctGroupBy(group_by_column, distinct_aggregate_columns, gpuBufferManager, distinct_info, num_group_keys);

	// Reading groupby columns based on the grouping set
	for (int idx = 0; idx < grouped_aggregate_data.groups.size(); idx++) {
		//TODO: has to fix this for columns with partially NULL values
		if (group_by_result->columns[idx] == nullptr && group_by_column[idx]->column_length > 0 && group_by_column[idx]->data_wrapper.data != nullptr) {
			group_by_result->columns[idx] = group_by_column[idx];
			group_by_result->columns[idx]->row_ids = nullptr;
			group_by_result->columns[idx]->row_id_count = 0;
		} else if (group_by_result->columns[idx] != nullptr && group_by_column[idx]->column_length > 0 && group_by_column[idx]->data_wrapper.data != nullptr) {
			group_by_result->columns[idx] = CombineColumns(group_by_result->columns[idx], group_by_column[idx], gpuBufferManager);
		}
	}

	for (int aggr_idx = 0; aggr_idx < distinct_info.indices.size(); aggr_idx++) {
		//TODO: has to fix this for columns with partially NULL values
		//TODO: has to fix this for group by where there would be both distinct and non distinct aggregates at the same time
		if (group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] == nullptr && distinct_aggregate_columns[aggr_idx]->column_length > 0 && distinct_aggregate_columns[aggr_idx]->data_wrapper.data != nullptr) {
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] = distinct_aggregate_columns[aggr_idx];
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx]->row_ids = nullptr;
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx]->row_id_count = 0;
		} else if (group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] != nullptr && distinct_aggregate_columns[aggr_idx]->column_length > 0 && distinct_aggregate_columns[aggr_idx]->data_wrapper.data != nullptr) {
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] = CombineColumns(group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx], distinct_aggregate_columns[aggr_idx], gpuBufferManager);
		}
	}

}

} // namespace duckdb