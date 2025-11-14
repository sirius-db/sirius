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

#include "operator/gpu_physical_order.hpp"
#include "operator/gpu_materialize.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_materialize.hpp"
#include "log/logging.hpp"

namespace duckdb {

void
HandleOrderBy(vector<shared_ptr<GPUColumn>> &order_by_keys, vector<shared_ptr<GPUColumn>> &projection_columns, const vector<BoundOrderByNode> &orders, uint64_t num_projections) {
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	OrderByType* order_by_type = gpuBufferManager->customCudaHostAlloc<OrderByType>(orders.size());
	for (int order_idx = 0; order_idx < orders.size(); order_idx++) {
		if (orders[order_idx].type == OrderType::ASCENDING) {
			order_by_type[order_idx] = OrderByType::ASCENDING;
		} else {
			order_by_type[order_idx] = OrderByType::DESCENDING;
		}
	}
	
	cudf_orderby(order_by_keys, projection_columns, orders.size(), num_projections, order_by_type);
}

GPUPhysicalOrder::GPUPhysicalOrder(vector<LogicalType> types, vector<BoundOrderByNode> orders, vector<idx_t> projections_p,
                             idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::ORDER_BY, std::move(types), estimated_cardinality),
      orders(std::move(orders)), projections(std::move(projections_p)) {

    sort_result = make_shared_ptr<GPUIntermediateRelation>(projections.size());
    for (int col = 0; col < projections.size(); col++) {
      sort_result->columns[col] = nullptr;
    }
}

SourceResultType
GPUPhysicalOrder::GetData(GPUIntermediateRelation &output_relation) const {
    auto start = std::chrono::high_resolution_clock::now();
    for (int col = 0; col < sort_result->columns.size(); col++) {
      SIRIUS_LOG_DEBUG("Writing order by result to column {}", col);
      output_relation.columns[col] = sort_result->columns[col];
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_DEBUG("Order by GetData time: {:.2f} ms", duration.count()/1000.0);
    return SourceResultType::FINISHED;
}

SinkResultType 
GPUPhysicalOrder::Sink(GPUIntermediateRelation &input_relation) const {

    auto start = std::chrono::high_resolution_clock::now();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    vector<shared_ptr<GPUColumn>> order_by_keys(orders.size());
    vector<shared_ptr<GPUColumn>> projection_columns(projections.size());
    
    for (int projection_idx = 0; projection_idx < projections.size(); projection_idx++) {
        auto input_idx = projections[projection_idx];
        projection_columns[projection_idx] = HandleMaterializeExpression(input_relation.columns[input_idx], gpuBufferManager);
        input_relation.columns[input_idx] = projection_columns[projection_idx];
    }
    
    for (int order_idx = 0; order_idx < orders.size(); order_idx++) {
      auto& expr = *orders[order_idx].expression;
      if (expr.expression_class != ExpressionClass::BOUND_REF) {
        throw NotImplementedException("Order by expression not supported");
      }
      auto input_idx = expr.Cast<BoundReferenceExpression>().index;
      order_by_keys[order_idx] = HandleMaterializeExpression(input_relation.columns[input_idx], gpuBufferManager);
    }


    if (order_by_keys[0]->column_length > INT32_MAX ) {
      throw NotImplementedException("Order by with column length greater than INT32_MAX is not supported");
    }

    HandleOrderBy(order_by_keys, projection_columns, orders, projections.size());

    for (int col = 0; col < projections.size(); col++) {
      if (sort_result->columns[col] == nullptr || sort_result->columns[col]->column_length == 0 ||
          sort_result->columns[col]->data_wrapper.data == nullptr) {
        sort_result->columns[col] = projection_columns[col];
        sort_result->columns[col]->row_ids = nullptr;
        sort_result->columns[col]->row_id_count = 0;
      } else if (sort_result->columns[col] != nullptr && projection_columns[col]->column_length > 0 && projection_columns[col]->data_wrapper.data != nullptr) {
        throw NotImplementedException("Order by with partially NULL values is not supported");
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_DEBUG("Order by Sink time: {:.2f} ms", duration.count()/1000.0);
    return SinkResultType::FINISHED;
}


} // namespace duckdb