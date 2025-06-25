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

#include "operator/gpu_physical_ungrouped_aggregate.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_materialize.hpp"
#include "log/logging.hpp"

namespace duckdb {

// template <typename T>
// void
// ResolveTypeAggregateExpression(vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates) {
// 	uint8_t** aggregate_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(aggregates.size());
// 	uint8_t** result = gpuBufferManager->customCudaHostAlloc<uint8_t*>(aggregates.size());
// 	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
// 		result[agg_idx] = nullptr;
// 	}

// 	size_t size = aggregate_keys[0]->column_length;

// 	int* agg_mode = gpuBufferManager->customCudaHostAlloc<int>(aggregates.size());

// 	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
// 		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
// 		if (expr.function.name.compare("count") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr) {
// 			agg_mode[agg_idx] = 5;
// 			aggregate_data[agg_idx] = nullptr;
// 		} else if (expr.function.name.compare("sum") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr) {
// 			agg_mode[agg_idx] = 5;
// 			aggregate_data[agg_idx] = nullptr;
// 		} else if (expr.function.name.compare("sum") == 0) {
// 			agg_mode[agg_idx] = 0;
// 			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);
// 		} else if (expr.function.name.compare("avg") == 0) {
// 			if (aggregate_keys[agg_idx]->data_wrapper.type.id() != GPUColumnTypeId::FLOAT64) throw NotImplementedException("Column type is supposed to be double");
// 			agg_mode[agg_idx] = 1;
// 			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);
// 		} else if (expr.function.name.compare("max") == 0) {
// 			agg_mode[agg_idx] = 2;
// 			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);
// 		} else if (expr.function.name.compare("min") == 0) {
// 			agg_mode[agg_idx] = 3;
// 			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);
// 		} else if (expr.function.name.compare("count_star") == 0) {
// 			agg_mode[agg_idx] = 4;
// 			aggregate_data[agg_idx] = nullptr;
// 		} else if (expr.function.name.compare("count") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
// 			agg_mode[agg_idx] = 4;
// 			aggregate_data[agg_idx] = nullptr;
// 		} else if (expr.function.name.compare("first") == 0) {
// 			agg_mode[agg_idx] = 6;
// 			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);;
// 		} else {
// 			throw NotImplementedException("Aggregate function not supported");
// 		}
// 	}

// 	ungroupedAggregate<T>(aggregate_data, result, size, agg_mode, aggregates.size());

// 	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
// 		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
// 		if (expr.function.name.compare("count_star") == 0 || expr.function.name.compare("count") == 0) {
// 			aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(result[agg_idx]));
// 		} else if (size == 0){
// 			aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(result[agg_idx]));
// 		} else { 
// 			aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(1, aggregate_keys[agg_idx]->data_wrapper.type, reinterpret_cast<uint8_t*>(result[agg_idx]));
// 		}
// 	}
// }

// void
// HandleAggregateExpression(vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates) {
// 	//check if all the aggregate functions are of the same type
// 	bool same_type = true;
// 	GPUColumnType prev_type;
// 	for (int i = 0; i < aggregates.size(); i++) {
// 		if (aggregates[i]->Cast<BoundAggregateExpression>().function.name.compare("count") != 0 && 
// 					aggregates[i]->Cast<BoundAggregateExpression>().function.name.compare("count_star") != 0) {
// 			prev_type = aggregate_keys[i]->data_wrapper.type;
// 			break;
// 		}
// 	}
// 	for (int i = 0; i < aggregates.size(); i++) {
// 		if (aggregates[i]->Cast<BoundAggregateExpression>().function.name.compare("count") != 0 && 
// 					aggregates[i]->Cast<BoundAggregateExpression>().function.name.compare("count_star") != 0) {
// 			const GPUColumnType& aggregate_type = aggregate_keys[i]->data_wrapper.type;
// 			if (aggregate_type.id() != prev_type.id()) {
// 				throw NotImplementedException("All aggregate functions must be of the same type");
// 			}
// 			prev_type = aggregate_type;
// 		}
// 	}

//     switch(aggregate_keys[0]->data_wrapper.type.id()) {
//       case GPUColumnTypeId::INT64:
// 		ResolveTypeAggregateExpression<uint64_t>(aggregate_keys, gpuBufferManager, aggregates);
// 		break;
//       case GPUColumnTypeId::FLOAT64:
// 	  	ResolveTypeAggregateExpression<double>(aggregate_keys, gpuBufferManager, aggregates);
// 		break;
//       default:
//         throw NotImplementedException("Unsupported sirius column type in `HandleAggregateExpression`: %d",
// 																			static_cast<int>(aggregate_keys[0]->data_wrapper.type.id()));
//     }
// }

void
HandleAggregateExpressionCuDF(vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates) {
	AggregationType* agg_mode = gpuBufferManager->customCudaHostAlloc<AggregationType>(aggregates.size());
	SIRIUS_LOG_DEBUG("Handling ungrouped aggregate expression");
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
		} else if (expr.function.name.compare("first") == 0) {
			agg_mode[agg_idx] = AggregationType::FIRST;
		} else {
			SIRIUS_LOG_DEBUG("Aggregate function not supported: {}", expr.function.name);
			throw NotImplementedException("Aggregate function not supported");
		}
	}

	cudf_aggregate(aggregate_keys, aggregates.size(), agg_mode);
}

GPUPhysicalUngroupedAggregate::GPUPhysicalUngroupedAggregate(vector<LogicalType> types,
                                                       vector<unique_ptr<Expression>> expressions,
                                                       idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::UNGROUPED_AGGREGATE, std::move(types), estimated_cardinality),
      aggregates(std::move(expressions)) {

	distinct_collection_info = DistinctAggregateCollectionInfo::Create(aggregates);
	aggregation_result = make_shared_ptr<GPUIntermediateRelation>(aggregates.size());
	if (!distinct_collection_info) {
		return;
	}
	distinct_data = make_uniq<DistinctAggregateData>(*distinct_collection_info);

}

SinkResultType 
GPUPhysicalUngroupedAggregate::Sink(GPUIntermediateRelation &input_relation) const {
	SIRIUS_LOG_DEBUG("Performing ungrouped aggregation");
	auto start = std::chrono::high_resolution_clock::now();

	if (distinct_data) {
		SinkDistinct(input_relation);
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

	idx_t payload_idx = 0;
	idx_t next_payload_idx = 0;
	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	vector<shared_ptr<GPUColumn>> aggregate_column(aggregates.size());
	for (int aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		aggregate_column[aggr_idx] = nullptr;
	}

	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		D_ASSERT(aggregates[aggr_idx]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
		auto &aggregate = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();

		payload_idx = next_payload_idx;
		next_payload_idx = payload_idx + aggregate.children.size();

		if (aggregate.IsDistinct()) {
			continue;
		}

		if (aggregate.filter) {
			auto &bound_ref_expr = aggregate.filter->Cast<BoundReferenceExpression>();
			SIRIUS_LOG_DEBUG("Reading filter column from index {}", bound_ref_expr.index);
		}

		idx_t payload_cnt = 0;

		SIRIUS_LOG_DEBUG("Aggregate type: {}", aggregate.function.name);
		if (aggregate.children.size() > 1) throw NotImplementedException("Aggregates with multiple children not supported yet");
		for (idx_t i = 0; i < aggregate.children.size(); ++i) {
			for (auto &child_expr : aggregate.children) {
				D_ASSERT(child_expr->type == ExpressionType::BOUND_REF);
				SIRIUS_LOG_DEBUG("Reading aggregation column from index {} and passing it to index {} in aggregation result", payload_idx + payload_cnt, aggr_idx);
				aggregate_column[aggr_idx] = HandleMaterializeExpression(input_relation.columns[payload_idx + payload_cnt], gpuBufferManager);
				payload_cnt++;
			}
		}
	}

	for (int aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggregate = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();
		//here we probably have count(*) or sum(*) or something like that
		if (aggregate.children.size() == 0) {
			SIRIUS_LOG_DEBUG("Passing * aggregate to index {} in aggregation result", aggr_idx);
			aggregate_column[aggr_idx] = make_shared_ptr<GPUColumn>(column_size, GPUColumnType(GPUColumnTypeId::INT64), nullptr);
		}
	}

	bool string_cudf_supported = true;
	for (int col = 0; col < aggregates.size(); col++) {
		// if types is VARCHAR, check the number of bytes
		if (aggregate_column[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
			throw NotImplementedException("String column not supported");
		}
	}
	if (aggregate_column[0]->column_length > INT32_MAX) {
		throw NotImplementedException("Column length greater than INT32_MAX is not supported");
	} else {
		HandleAggregateExpressionCuDF(aggregate_column, gpuBufferManager, aggregates);
	}

	for (int aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		//TODO: has to fix this for columns with partially NULL values
		if (aggregation_result->columns[aggr_idx] == nullptr) {
			SIRIUS_LOG_DEBUG("Passing aggregate column {} to aggregation result column {}", aggr_idx, aggr_idx);
			aggregation_result->columns[aggr_idx] = aggregate_column[aggr_idx];
			aggregation_result->columns[aggr_idx]->row_ids = nullptr;
			aggregation_result->columns[aggr_idx]->row_id_count = 0;
		} else if (aggregation_result->columns[aggr_idx] != nullptr) {
			if (aggregate_column[aggr_idx]->data_wrapper.data != nullptr && aggregation_result->columns[aggr_idx]->data_wrapper.data != nullptr) {
				throw NotImplementedException("Combine not implemented yet for ungrouped aggregate");
			} else if (aggregate_column[aggr_idx]->data_wrapper.data != nullptr && aggregation_result->columns[aggr_idx]->data_wrapper.data == nullptr) {
				SIRIUS_LOG_DEBUG("Passing aggregate column {} to aggregation result column {}", aggr_idx, aggr_idx);
				aggregation_result->columns[aggr_idx] = aggregate_column[aggr_idx];
				aggregation_result->columns[aggr_idx]->row_ids = nullptr;
				aggregation_result->columns[aggr_idx]->row_id_count = 0;
			} else {
				SIRIUS_LOG_DEBUG("Aggregate column {} is null, skipping", aggr_idx);
			}
		}
	}

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Ungrouped aggregate Sink time: {:.2f} ms", duration.count()/1000.0);
  	return SinkResultType::FINISHED;
}
  
SourceResultType
GPUPhysicalUngroupedAggregate::GetData(GPUIntermediateRelation &output_relation) const {
	auto start = std::chrono::high_resolution_clock::now();
	for (int col = 0; col < aggregation_result->columns.size(); col++) {
		SIRIUS_LOG_DEBUG("Writing aggregation result to column {}", col);
		output_relation.columns[col] = make_shared_ptr<GPUColumn>(aggregation_result->columns[col]->column_length, aggregation_result->columns[col]->data_wrapper.type, aggregation_result->columns[col]->data_wrapper.data);
	}

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Ungrouped aggregate GetData time: {:.2f} ms", duration.count()/1000.0);
	return SourceResultType::FINISHED;
}

void
GPUPhysicalUngroupedAggregate::SinkDistinct(GPUIntermediateRelation &input_relation) const {
	auto &distinct_info = *distinct_collection_info;
	auto &distinct_indices = distinct_info.Indices();
	auto &distinct_filter = distinct_info.Indices();

	for (auto &idx : distinct_indices) {
		auto &aggregate = aggregates[idx]->Cast<BoundAggregateExpression>();

		D_ASSERT(distinct_info.table_map.count(idx));

		if (aggregate.filter) {
			auto &bound_ref_expr = aggregate.filter->Cast<BoundReferenceExpression>();
			SIRIUS_LOG_DEBUG("Reading filter column from index {}", bound_ref_expr.index);

			for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
				auto &child = aggregate.children[child_idx];
				auto &bound_ref = child->Cast<BoundReferenceExpression>();
				SIRIUS_LOG_DEBUG("Reading aggregation column from index {} and passing it to index {} in groupby result", bound_ref.index, bound_ref.index);
				input_relation.checkLateMaterialization(bound_ref.index);
				aggregation_result->columns[bound_ref.index] = input_relation.columns[bound_ref.index];
			}
		} else {
			for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
				auto &child = aggregate.children[child_idx];
				auto &bound_ref = child->Cast<BoundReferenceExpression>();
				SIRIUS_LOG_DEBUG("Reading aggregation column from index {} and passing it to index {} in groupby result", bound_ref.index, bound_ref.index);
				input_relation.checkLateMaterialization(bound_ref.index);
				aggregation_result->columns[bound_ref.index] = input_relation.columns[bound_ref.index];
			}
		}
	}

}

} // namespace duckdb