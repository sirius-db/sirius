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
	T* a = reinterpret_cast<T*> (column1->data_wrapper.data);
	T* b = reinterpret_cast<T*> (column2->data_wrapper.data);
	combineColumns<T>(a, b, combine, column1->column_length, column2->column_length);
	shared_ptr<GPUColumn> result = make_shared_ptr<GPUColumn>(column1->column_length + column2->column_length, column1->data_wrapper.type, reinterpret_cast<uint8_t*>(combine));
	if (column1->is_unique && column2->is_unique) {
		result->is_unique = true;
	}
	return result;
}

shared_ptr<GPUColumn>
ResolveTypeCombineStrings(shared_ptr<GPUColumn> column1, shared_ptr<GPUColumn> column2, GPUBufferManager* gpuBufferManager) {
	uint8_t* combine;
	uint64_t* offset_combine;
	uint8_t* a = column1->data_wrapper.data;
	uint8_t* b = column2->data_wrapper.data;
	uint64_t* offset_a = column1->data_wrapper.offset;
	uint64_t* offset_b = column2->data_wrapper.offset;
	uint64_t num_bytes_a = column1->data_wrapper.num_bytes;
	uint64_t num_bytes_b = column2->data_wrapper.num_bytes;

	combineStrings(a, b, combine, offset_a, offset_b, offset_combine, num_bytes_a, num_bytes_b, column1->column_length, column2->column_length);
	shared_ptr<GPUColumn> result = make_shared_ptr<GPUColumn>(column1->column_length + column2->column_length, GPUColumnType(GPUColumnTypeId::VARCHAR), combine, offset_combine, num_bytes_a + num_bytes_b, true);
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

template <typename T, typename V>
void
ResolveTypeGroupByAggregateExpression(vector<shared_ptr<GPUColumn>> &group_by_keys, vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates, int num_group_keys) {
	uint64_t count[1];
	count[0] = 0;
	uint8_t** group_by_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_group_keys);
	uint8_t** aggregate_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(aggregates.size());

	for (int group = 0; group < num_group_keys; group++) {
		if (group_by_keys[group]->data_wrapper.data == nullptr && group_by_keys[group]->column_length != 0) {
			throw NotImplementedException("Group by column is null");
		}
		group_by_data[group] = (group_by_keys[group]->data_wrapper.data);
	}
	size_t size = group_by_keys[0]->column_length;

	int* agg_mode = gpuBufferManager->customCudaHostAlloc<int>(aggregates.size());

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
			if (aggregate_keys[agg_idx]->data_wrapper.type.id() != GPUColumnTypeId::FLOAT64) throw NotImplementedException("Column type is supposed to be double");
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

	// groupedAggregate<T, V>(group_by_data, aggregate_data, count, size, num_group_keys, aggregates.size(), agg_mode);
	hashGroupedAggregate<T, V>(group_by_data, aggregate_data, count, size, num_group_keys, aggregates.size(), agg_mode);

	// Reading groupby columns based on the grouping set
	for (idx_t group = 0; group < num_group_keys; group++) {
		bool old_unique = group_by_keys[group]->is_unique;
		group_by_keys[group] = make_shared_ptr<GPUColumn>(count[0], group_by_keys[group]->data_wrapper.type, reinterpret_cast<uint8_t*>(group_by_data[group]));
		group_by_keys[group]->is_unique = old_unique;
	}

	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
		if (expr.function.name.compare("count_star") == 0 || expr.function.name.compare("count") == 0) {
			aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(count[0], GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(aggregate_data[agg_idx]));
		} else {
			aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(count[0], aggregate_keys[agg_idx]->data_wrapper.type, reinterpret_cast<uint8_t*>(aggregate_data[agg_idx]));
		}
	}
}

template <typename V>
void
ResolveTypeGroupByString(vector<shared_ptr<GPUColumn>> &group_by_keys, vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates, int num_group_keys) {
	uint64_t count[1];
	count[0] = 0;
	uint8_t** group_by_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_group_keys);
	uint64_t** offset_data = gpuBufferManager->customCudaHostAlloc<uint64_t*>(num_group_keys);
	uint64_t* num_bytes = gpuBufferManager->customCudaHostAlloc<uint64_t>(num_group_keys);
	uint8_t** aggregate_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(aggregates.size());

	size_t size = group_by_keys[0]->column_length;
	uint64_t num_rows = static_cast<uint64_t>(size);
	for (int group = 0; group < num_group_keys; group++) {
		group_by_data[group] = (group_by_keys[group]->data_wrapper.data);
		if (group_by_keys[group]->data_wrapper.data == nullptr && group_by_keys[group]->column_length != 0) {
			throw NotImplementedException("Group by column is null");
		}

		if (group_by_keys[group]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
			offset_data[group] = (group_by_keys[group]->data_wrapper.offset);
		} else {
			size_t column_size = group_by_keys[group]->data_wrapper.getColumnTypeSize();
			offset_data[group] = createFixedSizeOffsets(column_size, num_rows);
		}
	}

	int* agg_mode = gpuBufferManager->customCudaHostAlloc<int>(aggregates.size());
	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
		SIRIUS_LOG_DEBUG("Aggregate function name {}", expr.function.name);
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
			if (aggregate_keys[agg_idx]->data_wrapper.type.id() != GPUColumnTypeId::FLOAT64) throw NotImplementedException("Column type is supposed to be double");
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

		// Create a buffer for this column if its not already specified
		if(aggregate_data[agg_idx] == nullptr) {
			aggregate_data[agg_idx] = reinterpret_cast<uint8_t*>(gpuBufferManager->customCudaMalloc<V>(size, 0, 0));
		}
		SIRIUS_LOG_DEBUG("Aggregate function name {} got agg_mode of {}", expr.function.name, agg_mode[agg_idx]);
	}

	groupedStringAggregate<V>(group_by_data, aggregate_data, offset_data, num_bytes, count, num_rows, num_group_keys, aggregates.size(), agg_mode);
	// optimizedGroupedStringAggregate<V>(group_by_data, aggregate_data, offset_data, num_bytes, count, num_rows, num_group_keys, aggregates.size(), agg_mode);

	// Reading groupby columns based on the grouping set
	for (idx_t group = 0; group < num_group_keys; group++) {
		bool old_unique = group_by_keys[group]->is_unique;
		if (group_by_keys[group]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
			if (offset_data[group] == nullptr && count[0] > 0) throw NotImplementedException("Offset data is null");
			group_by_keys[group] = make_shared_ptr<GPUColumn>(count[0], group_by_keys[group]->data_wrapper.type, reinterpret_cast<uint8_t*>(group_by_data[group]), reinterpret_cast<uint64_t*>(offset_data[group]), num_bytes[group], true);
		} else {
			group_by_keys[group] = make_shared_ptr<GPUColumn>(count[0], group_by_keys[group]->data_wrapper.type, reinterpret_cast<uint8_t*>(group_by_data[group]));
		}
		group_by_keys[group]->is_unique = old_unique;
	}

	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
		if (expr.function.name.compare("count_star") == 0 || expr.function.name.compare("count") == 0) {
			aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(count[0], GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(aggregate_data[agg_idx]));
		} else {
			aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(count[0], aggregate_keys[agg_idx]->data_wrapper.type, reinterpret_cast<uint8_t*>(aggregate_data[agg_idx]));
		}
	}
}

void
HandleGroupByAggregateExpression(vector<shared_ptr<GPUColumn>> &group_by_keys, vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates, int num_group_keys) {
	bool string_groupby = false;
	for (int i = 0; i < num_group_keys; i++) {
		if (group_by_keys[i]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
			string_groupby = true;
		}
	}

	GPUColumnType aggregate_type;
	if (aggregates.size() == 1) {
		aggregate_type = aggregate_keys[0]->data_wrapper.type;
	} else {
		//check if all the aggregate functions are of the same type
		bool same_type = true;
		GPUColumnType prev_type;
		for (int i = 0; i < aggregates.size(); i++) {
			if (aggregates[i]->Cast<BoundAggregateExpression>().function.name.compare("count") != 0 && 
						aggregates[i]->Cast<BoundAggregateExpression>().function.name.compare("count_star") != 0) {
				prev_type = aggregate_keys[i]->data_wrapper.type;
				break;
			}
		}
		for (int i = 0; i < aggregates.size(); i++) {
			if (aggregates[i]->Cast<BoundAggregateExpression>().function.name.compare("count") != 0 && 
						aggregates[i]->Cast<BoundAggregateExpression>().function.name.compare("count_star") != 0) {
				aggregate_type = aggregate_keys[i]->data_wrapper.type;
				if (aggregate_type.id() != prev_type.id()) {
					throw NotImplementedException("All aggregate functions must be of the same type");
				}
				prev_type = aggregate_type;
			}
		}
	}

	if (string_groupby) {
		if (aggregate_type.id() == GPUColumnTypeId::INT64) {
			ResolveTypeGroupByString<uint64_t>(group_by_keys, aggregate_keys, gpuBufferManager, aggregates, num_group_keys);
		} else if (aggregate_type.id() == GPUColumnTypeId::FLOAT64) {
			ResolveTypeGroupByString<double>(group_by_keys, aggregate_keys, gpuBufferManager, aggregates, num_group_keys);
		} else {
			throw NotImplementedException("Unsupported sirius column type in `HandleGroupByAggregateExpression`: {}",
																		static_cast<int>(aggregate_type.id()));
		}
	} else {
		//check if all the group by keys are all integers
		for (int i = 0; i < num_group_keys; i++) {
			if (group_by_keys[i]->data_wrapper.type.id() != GPUColumnTypeId::INT64) {
				throw NotImplementedException("Group by column is not an integer in `HandleGroupByAggregateExpression`");
			}
		}
		switch(group_by_keys[0]->data_wrapper.type.id()) {
		case GPUColumnTypeId::INT64:
			if (aggregate_type.id() == GPUColumnTypeId::INT64) {
				ResolveTypeGroupByAggregateExpression<uint64_t, uint64_t>(group_by_keys, aggregate_keys, gpuBufferManager, aggregates, num_group_keys);
			} else if (aggregate_type.id() == GPUColumnTypeId::FLOAT64) {
				ResolveTypeGroupByAggregateExpression<uint64_t, double>(group_by_keys, aggregate_keys, gpuBufferManager, aggregates, num_group_keys);
			} else throw NotImplementedException("Unsupported sirius column type in `HandleGroupByAggregateExpression`: {}",
																					 static_cast<int>(aggregate_type.id()));
			break;
		case GPUColumnTypeId::FLOAT64:
		default:
			throw NotImplementedException("Unsupported sirius column type in `HandleGroupByAggregateExpression`: {}",
																		static_cast<int>(group_by_keys[0]->data_wrapper.type.id()));
		}
	}
}

void
HandleGroupByAggregateCuDF(vector<shared_ptr<GPUColumn>> &group_by_keys, vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates, int num_group_keys) {

	AggregationType* agg_mode = gpuBufferManager->customCudaHostAlloc<AggregationType>(aggregates.size());
	SIRIUS_LOG_DEBUG("Handling group by aggregate expression");
	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
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
			SIRIUS_LOG_DEBUG("Aggregate function not supported: {}", expr.function.name);
			throw NotImplementedException("Aggregate function not supported");
		}
	}
	
	cudf_groupby(group_by_keys, aggregate_keys, num_group_keys, aggregates.size(), agg_mode);
}

template <typename T>
void ResolveTypeDuplicateElimination(vector<shared_ptr<GPUColumn>> &group_by_keys, GPUBufferManager* gpuBufferManager, int num_group_keys) {
	uint64_t count[1];
	count[0] = 0;
	uint8_t** group_by_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_group_keys);

	for (int group = 0; group < num_group_keys; group++) {
		group_by_data[group] = (group_by_keys[group]->data_wrapper.data);
	}
	size_t size = group_by_keys[0]->column_length;

	groupedWithoutAggregate<T>(group_by_data, count, size, num_group_keys);

	// Reading groupby columns based on the grouping set
	for (idx_t group = 0; group < num_group_keys; group++) {
		bool old_unique = group_by_keys[group]->is_unique;
		group_by_keys[group] = make_shared_ptr<GPUColumn>(count[0], group_by_keys[group]->data_wrapper.type, reinterpret_cast<uint8_t*>(group_by_data[group]));
		group_by_keys[group]->is_unique = old_unique;
	}
}

void HandleDuplicateElimination(vector<shared_ptr<GPUColumn>> &group_by_keys, GPUBufferManager* gpuBufferManager, int num_group_keys) {
	//check if all the group by keys are all integers
	for (int i = 0; i < num_group_keys; i++) {
		if (group_by_keys[i]->data_wrapper.type.id() != GPUColumnTypeId::INT64) {
			throw NotImplementedException("Group by column is not an integer in `HandleDuplicateElimination`");
		}
	}
    switch(group_by_keys[0]->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT64:
	  	ResolveTypeDuplicateElimination<uint64_t>(group_by_keys, gpuBufferManager, num_group_keys);
		break;
      case GPUColumnTypeId::FLOAT64:
      default:
        throw NotImplementedException("Unsupported sirius column type in `HandleDuplicateElimination`: %d",
																			static_cast<int>(group_by_keys[0]->data_wrapper.type.id()));
    }
}

template <typename T, typename V>
void ResolveTypeDistinctGroupBy(vector<shared_ptr<GPUColumn>> &group_by_keys, vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, DistinctAggregateCollectionInfo &distinct_info, int num_group_keys) {
	uint64_t count[1];
	count[0] = 0;
	uint8_t** group_by_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_group_keys);
	uint8_t** distinct_aggregate_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(distinct_info.indices.size());

	for (int group = 0; group < num_group_keys; group++) {
		group_by_data[group] = (group_by_keys[group]->data_wrapper.data);
	}
	size_t size = group_by_keys[0]->column_length;

	int* distinct_mode = gpuBufferManager->customCudaHostAlloc<int>(distinct_info.indices.size());

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
		bool old_unique = group_by_keys[group]->is_unique;
		group_by_keys[group] = make_shared_ptr<GPUColumn>(count[0], group_by_keys[group]->data_wrapper.type, reinterpret_cast<uint8_t*>(group_by_data[group]));
		group_by_keys[group]->is_unique = old_unique;
	}

	for (int idx = 0; idx < distinct_info.indices.size(); idx++) {
		auto distinct_idx = distinct_info.indices[idx];
		auto& expr = distinct_info.aggregates[distinct_idx]->Cast<BoundAggregateExpression>();
		if (expr.function.name.compare("count") == 0) {
			aggregate_keys[idx] = make_shared_ptr<GPUColumn>(count[0], GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(distinct_aggregate_data[idx]));
		}
	}

}

//TODO: Distinct Aggregate currently does not support string type
void HandleDistinctGroupBy(vector<shared_ptr<GPUColumn>> &group_by_keys, vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, DistinctAggregateCollectionInfo &distinct_info, int num_group_keys) {
	//check if all the group by keys are all integers
	SIRIUS_LOG_DEBUG("Handling distinct group by");
	for (int i = 0; i < num_group_keys; i++) {
		if (group_by_keys[i]->data_wrapper.type.id() != GPUColumnTypeId::INT64) {
			throw NotImplementedException("Group by column is not an integer in `HandleDistinctGroupBy`");
		}
	}
    switch(group_by_keys[0]->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT64: {
	  	if (aggregate_keys[0]->data_wrapper.type.id() == GPUColumnTypeId::INT64) {
			ResolveTypeDistinctGroupBy<uint64_t, uint64_t>(group_by_keys, aggregate_keys, gpuBufferManager, distinct_info, num_group_keys);
		} else throw NotImplementedException("Unsupported sirius column type in `HandleDistinctGroupBy`: %d",
																				 static_cast<int>(aggregate_keys[0]->data_wrapper.type.id()));
		break;
	  } case GPUColumnTypeId::FLOAT64:
      default:
        throw NotImplementedException("Unsupported sirius column type in `HandleDistinctGroupBy`: %d",
																			static_cast<int>(group_by_keys[0]->data_wrapper.type.id()));
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
	group_by_result = make_shared_ptr<GPUIntermediateRelation>(total_output_columns);
}

SinkResultType
GPUPhysicalGroupedAggregate::Sink(GPUIntermediateRelation& input_relation) const {
  	SIRIUS_LOG_DEBUG("Perform groupby and aggregation");

	auto start = std::chrono::high_resolution_clock::now();

	if (distinct_collection_info) {
		SinkDistinct(input_relation);
		return SinkResultType::FINISHED;
	}

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
			group_by_column[idx] = HandleMaterializeExpression(input_relation.columns[bound_ref_expr.index], bound_ref_expr, gpuBufferManager);
			idx++;
		}
	}

	int aggr_idx = 0;
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		SIRIUS_LOG_DEBUG("Aggregate type: {}", aggr.function.name);
		if (aggr.children.size() > 1) throw NotImplementedException("Aggregates with multiple children not supported yet");
		for (auto &child_expr : aggr.children) {
			D_ASSERT(child_expr->type == ExpressionType::BOUND_REF);
			auto &bound_ref_expr = child_expr->Cast<BoundReferenceExpression>();
			SIRIUS_LOG_DEBUG("Passing input column index {} to aggregate column index {}", bound_ref_expr.index, aggr_idx);
			aggregate_column[aggr_idx] = HandleMaterializeExpression(input_relation.columns[bound_ref_expr.index], bound_ref_expr, gpuBufferManager);
		}
		aggr_idx++;
	}

	aggr_idx = 0;
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		if (aggr.children.size() == 0) {
			//we have a count(*) aggregate
			SIRIUS_LOG_DEBUG("Passing * aggregate to index {} in aggregation result", aggr_idx);
			aggregate_column[aggr_idx] = make_shared_ptr<GPUColumn>(column_size, GPUColumnType(GPUColumnTypeId::INT64), nullptr);
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

	bool can_use_sirius_impl = CheckGroupKeyTypesForSiriusImpl(group_by_column);
	uint64_t count[1];
	if (aggregates.size() == 0) {
		if (can_use_sirius_impl) {
			HandleDuplicateElimination(group_by_column, gpuBufferManager, num_group_keys);
		} else {
			HandleGroupByAggregateCuDF(group_by_column, aggregate_column, gpuBufferManager, aggregates, num_group_keys);
		}
	} else {
		bool string_cudf_supported = true;
		// for (int col = 0; col < num_group_keys; col++) {
		// 	// if types is VARCHAR, check the number of bytes
		// 	if (group_by_column[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
		// 		if (group_by_column[col]->data_wrapper.num_bytes > INT32_MAX) {
		// 			string_cudf_supported = false;
		// 		}
		// 	}
		// }
		if (group_by_column[0]->column_length > INT32_MAX || aggregate_column[0]->column_length > INT32_MAX || !string_cudf_supported) {
			HandleGroupByAggregateExpression(group_by_column, aggregate_column, gpuBufferManager, aggregates, num_group_keys);
		} else {
			HandleGroupByAggregateCuDF(group_by_column, aggregate_column, gpuBufferManager, aggregates, num_group_keys);
			// HandleGroupByAggregateExpression(group_by_column, aggregate_column, gpuBufferManager, aggregates, num_group_keys);
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
			output_relation.columns[col] = make_shared_ptr<GPUColumn>(group_by_result->columns[col]->column_length, group_by_result->columns[col]->data_wrapper.type, group_by_result->columns[col]->data_wrapper.data,
					group_by_result->columns[col]->data_wrapper.offset, group_by_result->columns[col]->data_wrapper.num_bytes, true);
		} else {
			output_relation.columns[col] = make_shared_ptr<GPUColumn>(group_by_result->columns[col]->column_length, group_by_result->columns[col]->data_wrapper.type, group_by_result->columns[col]->data_wrapper.data);
		}
		output_relation.columns[col]->is_unique = old_unique;
	}
  	return SourceResultType::FINISHED;
}

void
GPUPhysicalGroupedAggregate::SinkDistinct(GPUIntermediateRelation& input_relation) const {
	if (groupings.size() > 1) throw NotImplementedException("Multiple groupings not supported yet");
	for (idx_t i = 0; i < groupings.size(); i++) {
		SinkDistinctGrouping(input_relation, i);
	}
}

void
GPUPhysicalGroupedAggregate::SinkDistinctGrouping(GPUIntermediateRelation& input_relation, idx_t grouping_idx) const {
	auto &distinct_info = *distinct_collection_info;

	SIRIUS_LOG_DEBUG("Perform groupby and aggregation distinct");

	auto start = std::chrono::high_resolution_clock::now();

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	uint64_t num_group_keys = grouped_aggregate_data.groups.size();
	vector<shared_ptr<GPUColumn>> group_by_column(grouped_aggregate_data.groups.size());
	vector<shared_ptr<GPUColumn>> distinct_aggregate_columns(distinct_info.indices.size());

	for (int i = 0; i < grouped_aggregate_data.groups.size(); i++) {
		group_by_column[i] = nullptr;
	}
	for (int i = 0; i < distinct_info.indices.size(); i++) {
		distinct_aggregate_columns[i] = nullptr;
	}

	for (idx_t group_idx = 0; group_idx < grouped_aggregate_data.groups.size(); group_idx++) {
		auto &group = grouped_aggregate_data.groups[group_idx];
		auto &bound_ref = group->Cast<BoundReferenceExpression>();
		group_by_column[group_idx] = HandleMaterializeExpression(input_relation.columns[bound_ref.index], bound_ref, gpuBufferManager);
	}

	int aggr_idx = 0;
	for (idx_t &idx : distinct_info.indices) {
		auto &aggregate = grouped_aggregate_data.aggregates[idx]->Cast<BoundAggregateExpression>();
		SIRIUS_LOG_DEBUG("Processing distinct aggregate {}", aggregate.function.name);
		// throw NotImplementedException("Distinct not supported yet");

		D_ASSERT(distinct_info.table_map.count(idx));

		if (aggregate.filter) {
			throw NotImplementedException("Filter not supported yet");
			auto it = filter_indexes.find(aggregate.filter.get());
      		SIRIUS_LOG_DEBUG("Reading filter columns from index {}", it->second);

			for (idx_t group_idx = 0; group_idx < grouped_aggregate_data.groups.size(); group_idx++) {
				auto &group = grouped_aggregate_data.groups[group_idx];
				auto &bound_ref = group->Cast<BoundReferenceExpression>();
				SIRIUS_LOG_DEBUG("Reading groupby columns from index {} and passing it to index {} in groupby result", bound_ref.index, group_idx);
				input_relation.checkLateMaterialization(bound_ref.index);
				group_by_result->columns[group_idx] = input_relation.columns[bound_ref.index];
			}
			for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
				auto &child = aggregate.children[child_idx];
				auto &bound_ref = child->Cast<BoundReferenceExpression>();
				SIRIUS_LOG_DEBUG("Reading aggregation column from index {} and passing it to index {} in groupby result", bound_ref.index, grouped_aggregate_data.groups.size() + idx);
				input_relation.checkLateMaterialization(bound_ref.index);
				group_by_result->columns[grouped_aggregate_data.groups.size() + idx] = input_relation.columns[bound_ref.index];
			}
		} else {

			if (aggregate.children.size() > 1) throw NotImplementedException("Aggregates with multiple children not supported yet");
			for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
				auto &child = aggregate.children[child_idx];
				auto &bound_ref = child->Cast<BoundReferenceExpression>();
				SIRIUS_LOG_DEBUG("Reading aggregation column from index {} and passing it to index {} in distinct aggregatec column", bound_ref.index, aggr_idx);
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
			SIRIUS_LOG_DEBUG("Passing group by column {} to group by result column {}", idx, idx);
			group_by_result->columns[idx] = group_by_column[idx];
			group_by_result->columns[idx]->row_ids = nullptr;
			group_by_result->columns[idx]->row_id_count = 0;
		} else if (group_by_result->columns[idx] != nullptr && group_by_column[idx]->column_length > 0 && group_by_column[idx]->data_wrapper.data != nullptr) {
			SIRIUS_LOG_DEBUG("Combining group by column {} with group by result column {}", idx, idx);
			group_by_result->columns[idx] = CombineColumns(group_by_result->columns[idx], group_by_column[idx], gpuBufferManager);
		}
	}

	for (int aggr_idx = 0; aggr_idx < distinct_info.indices.size(); aggr_idx++) {
		//TODO: has to fix this for columns with partially NULL values
		//TODO: has to fix this for group by where there would be both distinct and non distinct aggregates at the same time
		if (group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] == nullptr && distinct_aggregate_columns[aggr_idx]->column_length > 0 && distinct_aggregate_columns[aggr_idx]->data_wrapper.data != nullptr) {
			SIRIUS_LOG_DEBUG("Passing aggregate column {} to group by result column {}", grouped_aggregate_data.groups.size() + aggr_idx, grouped_aggregate_data.groups.size() + aggr_idx);
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] = distinct_aggregate_columns[aggr_idx];
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx]->row_ids = nullptr;
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx]->row_id_count = 0;
		} else if (group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] != nullptr && distinct_aggregate_columns[aggr_idx]->column_length > 0 && distinct_aggregate_columns[aggr_idx]->data_wrapper.data != nullptr) {
			SIRIUS_LOG_DEBUG("Combining aggregate column {} with group by result column {}", grouped_aggregate_data.groups.size() + aggr_idx, grouped_aggregate_data.groups.size() + aggr_idx);
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] = CombineColumns(group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx], distinct_aggregate_columns[aggr_idx], gpuBufferManager);
		}
	}

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Group Aggregate Distinct Sink time: {:.2f} ms", duration.count()/1000.0);
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