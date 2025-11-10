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

void
HandleAggregateExpressionCuDF(vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates) {
	AggregationType* agg_mode = gpuBufferManager->customCudaHostAlloc<AggregationType>(aggregates.size());
	SIRIUS_LOG_DEBUG("Handling ungrouped aggregate expression");
	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
		if (expr.IsDistinct()) {
			if (expr.function.name.compare("count") == 0) {
				agg_mode[agg_idx] = AggregationType::COUNT_DISTINCT;
			} else {
				SIRIUS_LOG_DEBUG("Aggregate function (distinct)  not supported: {}", expr.function.name);
				throw NotImplementedException("Aggregate function (distinct) not supported");
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
			} else if (expr.function.name.compare("first") == 0) {
				agg_mode[agg_idx] = AggregationType::FIRST;
			} else {
				SIRIUS_LOG_DEBUG("Aggregate function (not distinct) not supported: {}", expr.function.name);
				throw NotImplementedException("Aggregate function (not distinct) not supported");
			}
		}
	}

	cudf_aggregate(aggregate_keys, aggregates.size(), agg_mode);

	// Duckdb requires count(distinct) returns int64
	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		if (agg_mode[agg_idx] == AggregationType::COUNT_DISTINCT &&
				aggregate_keys[agg_idx]->data_wrapper.type.id() != GPUColumnTypeId::INT64) {
			auto from_cudf_column_view = aggregate_keys[agg_idx]->convertToCudfColumn();
			auto to_cudf_type = cudf::data_type(cudf::type_id::INT64);
			auto to_cudf_column = cudf::cast(from_cudf_column_view,
																			 to_cudf_type,
																		 	 rmm::cuda_stream_default,
																	  	 GPUBufferManager::GetInstance().mr);
			aggregate_keys[agg_idx]->setFromCudfColumn(*to_cudf_column, false, nullptr, 0, gpuBufferManager);
		}
	}
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
	vector<shared_ptr<GPUColumn>> aggregate_column(aggregates.size());
	for (int aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		aggregate_column[aggr_idx] = nullptr;
	}

	if (distinct_data) {
		MaterializeDistinctInput(input_relation, aggregate_column);
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
			aggregate_column[aggr_idx] = make_shared_ptr<GPUColumn>(column_size, GPUColumnType(GPUColumnTypeId::INT64), nullptr, nullptr);
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
		// output_relation.columns[col] = make_shared_ptr<GPUColumn>(aggregation_result->columns[col]->column_length, aggregation_result->columns[col]->data_wrapper.type, aggregation_result->columns[col]->data_wrapper.data);
		output_relation.columns[col] = make_shared_ptr<GPUColumn>(aggregation_result->columns[col]);
	}

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Ungrouped aggregate GetData time: {:.2f} ms", duration.count()/1000.0);
	return SourceResultType::FINISHED;
}

void
GPUPhysicalUngroupedAggregate::MaterializeDistinctInput(
		GPUIntermediateRelation &input_relation, vector<shared_ptr<GPUColumn>>& aggregate_column) const {
	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	auto &distinct_info = *distinct_collection_info;
	auto &distinct_indices = distinct_info.Indices();
	auto &distinct_filter = distinct_info.Indices();

	for (auto &idx : distinct_indices) {
		auto &aggregate = aggregates[idx]->Cast<BoundAggregateExpression>();

		D_ASSERT(distinct_info.table_map.count(idx));

		if (aggregate.filter) {
			auto &bound_ref_expr = aggregate.filter->Cast<BoundReferenceExpression>();
			SIRIUS_LOG_DEBUG("Reading filter column from index {}", bound_ref_expr.index);
		}

		for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
			auto &child = aggregate.children[child_idx];
			auto &bound_ref = child->Cast<BoundReferenceExpression>();
			SIRIUS_LOG_DEBUG("Reading aggregation column from index {} and passing it to index {} in groupby result", bound_ref.index, bound_ref.index);
			aggregate_column[bound_ref.index] = HandleMaterializeExpression(input_relation.columns[bound_ref.index], gpuBufferManager);
		}
	}

}

} // namespace duckdb