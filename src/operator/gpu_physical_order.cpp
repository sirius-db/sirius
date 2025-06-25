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

// void ResolveOrderByString(vector<shared_ptr<GPUColumn>> &sort_columns, int* sort_orders, int num_cols) {
//   GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
//   uint8_t** col_keys = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_cols);
//   uint64_t** col_offsets = gpuBufferManager->customCudaHostAlloc<uint64_t*>(num_cols);
//   uint64_t* col_num_bytes = gpuBufferManager->customCudaHostAlloc<uint64_t>(num_cols);

//   for(int i = 0; i < num_cols; i++) {
//     shared_ptr<GPUColumn> curr_column = sort_columns[i];
//     col_keys[i] = curr_column->data_wrapper.data;
//     col_offsets[i] = curr_column->data_wrapper.offset;
    
//     SIRIUS_LOG_DEBUG("ResolveOrderByString: For idx {} got num bytes of {}", i, curr_column->data_wrapper.num_bytes);
//   }
//   uint64_t num_rows = static_cast<uint64_t>(sort_columns[0]->column_length);

//   // Sort the results
//   orderByString(col_keys, col_offsets, sort_orders, col_num_bytes, num_rows, num_cols);

//   // Write the results back
//   for(int i = 0; i < num_cols; i++) {
//     shared_ptr<GPUColumn> curr_column = sort_columns[i];
//     curr_column->data_wrapper.data = col_keys[i];
//     curr_column->data_wrapper.offset = col_offsets[i];
//     curr_column->data_wrapper.num_bytes = col_num_bytes[i];

//     SIRIUS_LOG_DEBUG("ResolveOrderByString: Wrote num bytes of {} for idx {}", col_num_bytes[i], i);
//   }
// }

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
    // shared_ptr<GPUColumn>* sort_columns = new shared_ptr<GPUColumn>[orders.size()];
    // int* sort_orders = new int[orders.size()];
    // int idx = 0;
    // bool string_sort = false;
    // for (auto &order : orders) {
    //   // key_types.push_back(order.expression->return_type);
    //   // key_executor.AddExpression(*order.expression);
    //   auto& expr = *order.expression;
    //   expr.Print();
    //   if (expr.expression_class != ExpressionClass::BOUND_REF) {
    //     throw NotImplementedException("Order by expression not supported");
    //   }

    //   // Record the column to sort on
    //   auto &bound_ref_expr = expr.Cast<BoundReferenceExpression>();
    //   auto input_idx = bound_ref_expr.index;
    //   SIRIUS_LOG_DEBUG("Reading order by keys from index {}", input_idx);
    //   sort_columns[idx] = HandleMaterializeExpression(
    //     input_relation.columns[input_idx], bound_ref_expr, gpuBufferManager
    //   );
    //   if (sort_columns[idx]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
    //     string_sort = true;
    //   }

    //   // Record the sort method
    //   auto sort_method = order.type;
    //   int sort_type = 0;
    //   if(sort_method == OrderType::DESCENDING) {
    //     sort_type = 1;
    //   }
    //   sort_orders[idx] = sort_type;
    //   SIRIUS_LOG_DEBUG(
    //     "Order By got sort column: Col Length - {}, Size - {}, Bytes - {}, Sort Order - {}\n", 
    //     (int) sort_columns[idx]->column_length, (int) sort_columns[idx]->data_wrapper.size,  
    //     (int) sort_columns[idx]->data_wrapper.num_bytes, sort_orders[idx]
    //   );

    //   idx++;
    // }

    // // Now actually perform the order by
    // if(string_sort) {
    //   ResolveOrderByString(sort_columns, sort_orders, orders.size());
    // } else {
    //   throw NotImplementedException("Non String Order By not yet supported");
    // }

    // // Copy the sorted columns back into the input relationship
    // int sort_cols_idx = 0;
    // for (auto &order : orders) {
    //   auto& expr = *order.expression;
    //   auto &bound_ref_expr = expr.Cast<BoundReferenceExpression>();
    //   auto input_idx = bound_ref_expr.index;
    //   input_relation.columns[input_idx] = sort_columns[sort_cols_idx];

    //   sort_cols_idx += 1;
    // }

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

    // for (int col = 0; col < projections.size(); col++) {
    //   // if types is VARCHAR, check the number of bytes
    //   if (projection_columns[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
    //     if (projection_columns[col]->data_wrapper.num_bytes > INT32_MAX) {
    //       throw NotImplementedException("String column size greater than INT32_MAX is not supported");
    //     }
    //   }
    // }
    HandleOrderBy(order_by_keys, projection_columns, orders, projections.size());

    for (int col = 0; col < projections.size(); col++) {
      if (sort_result->columns[col] == nullptr && projection_columns[col]->column_length > 0 && projection_columns[col]->data_wrapper.data != nullptr) {
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