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

#include "duckdb/common/assert.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"

#include "gpu_physical_top_n.hpp"
#include "gpu_physical_order.hpp"
#include "gpu_materialize.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

namespace duckdb {

GPUPhysicalTopN::GPUPhysicalTopN(vector<LogicalType> types_p, vector<BoundOrderByNode> orders, idx_t limit, idx_t offset,
                           shared_ptr<DynamicFilterData> dynamic_filter_p, idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::TOP_N, std::move(types_p), estimated_cardinality), orders(std::move(orders)),
      limit(limit), offset(offset), dynamic_filter(std::move(dynamic_filter_p)) {
    sort_result = make_shared_ptr<GPUIntermediateRelation>(types.size());
    for (int col = 0; col < types.size(); col++) {
    	sort_result->columns[col] = nullptr;
    }
}

GPUPhysicalTopN::~GPUPhysicalTopN() {
}


void
HandleTopN(vector<shared_ptr<GPUColumn>> &order_by_keys, vector<shared_ptr<GPUColumn>> &projection_columns, const vector<BoundOrderByNode> &orders, uint64_t num_projections) {
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

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
// SinkResultType PhysicalTopN::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
SinkResultType GPUPhysicalTopN::Sink(GPUIntermediateRelation& input_relation) const {
	auto start = std::chrono::high_resolution_clock::now();
    // throw NotImplementedException("Top N Sink not implemented");
    if (dynamic_filter) {
				// `dynamic_filter` is currently not leveraged
        SIRIUS_LOG_WARN("`dynamic_filter` is currently not leveraged in `GPUPhysicalTopN`");
    }
    if (offset > 0) {
        throw NotImplementedException("Top N Sink with offset not implemented");
    }


	vector<shared_ptr<GPUColumn>> order_by_keys(orders.size());
	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

  	vector<shared_ptr<GPUColumn>> projection_columns(types.size());
  
	for (int projection_idx = 0; projection_idx < types.size(); projection_idx++) {
		auto input_idx = projection_idx;
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
  
	for (int col = 0; col < types.size(); col++) {
		// if types is VARCHAR, check the number of bytes
		if (projection_columns[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
			if (projection_columns[col]->data_wrapper.num_bytes > INT32_MAX) {
				throw NotImplementedException("String column size greater than INT32_MAX is not supported");
			}
		}
	}
  	HandleTopN(order_by_keys, projection_columns, orders, types.size());

	for (int col = 0; col < types.size(); col++) {
		if (sort_result->columns[col] == nullptr && projection_columns[col]->column_length > 0 && projection_columns[col]->data_wrapper.data != nullptr) {
			sort_result->columns[col] = projection_columns[col];
			sort_result->columns[col]->row_ids = nullptr;
			sort_result->columns[col]->row_id_count = 0;
		} else if (sort_result->columns[col] != nullptr && projection_columns[col]->column_length > 0 && projection_columns[col]->data_wrapper.data != nullptr) {
			throw NotImplementedException("Order by with partially NULL values is not supported");
		}
	}
    
	// append to the local sink state
	// auto &gstate = input.global_state.Cast<TopNGlobalState>();
	// auto &sink = input.local_state.Cast<TopNLocalState>();
	// sink.heap.Sink(chunk, &gstate.boundary_value);
	// sink.heap.Reduce();
	// return SinkResultType::NEED_MORE_INPUT;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Top N Sink time: {:.2f} ms", duration.count()/1000.0);
	return SinkResultType::FINISHED;
}

// //===--------------------------------------------------------------------===//
// // Combine
// //===--------------------------------------------------------------------===//
// SinkCombineResultType PhysicalTopN::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
// 	auto &gstate = input.global_state.Cast<TopNGlobalState>();
// 	auto &lstate = input.local_state.Cast<TopNLocalState>();

// 	// scan the local top N and append it to the global heap
// 	lock_guard<mutex> glock(gstate.lock);
// 	gstate.heap.Combine(lstate.heap);

// 	return SinkCombineResultType::FINISHED;
// }

// //===--------------------------------------------------------------------===//
// // Finalize
// //===--------------------------------------------------------------------===//
// SinkFinalizeType PhysicalTopN::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
//                                         OperatorSinkFinalizeInput &input) const {
// 	auto &gstate = input.global_state.Cast<TopNGlobalState>();
// 	// global finalize: compute the final top N
// 	gstate.heap.Finalize();
// 	return SinkFinalizeType::READY;
// }

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
// class TopNOperatorState : public GlobalSourceState {
// public:
// 	TopNScanState state;
// 	bool initialized = false;
// };

// unique_ptr<GlobalSourceState> PhysicalTopN::GetGlobalSourceState(ClientContext &context) const {
// 	return make_uniq<TopNOperatorState>();
// }

// SourceResultType PhysicalTopN::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
SourceResultType GPUPhysicalTopN::GetData(GPUIntermediateRelation& output_relation) const {
	auto start = std::chrono::high_resolution_clock::now();
	if (limit == 0) {
		return SourceResultType::FINISHED;
	}

	for (int col = 0; col < sort_result->columns.size(); col++) {
		SIRIUS_LOG_DEBUG("Writing top n result to column {}", col);
    	auto limit_const = min(limit, sort_result->columns[col]->column_length);
    	output_relation.columns[col] = make_shared_ptr<GPUColumn>(limit_const, sort_result->columns[col]->data_wrapper.type, sort_result->columns[col]->data_wrapper.data,
                          sort_result->columns[col]->data_wrapper.offset, sort_result->columns[col]->data_wrapper.num_bytes, sort_result->columns[col]->data_wrapper.is_string_data);
    	output_relation.columns[col]->is_unique = sort_result->columns[col]->is_unique;
		if (limit_const > 0 && output_relation.columns[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
			Allocator& allocator = Allocator::DefaultAllocator();
			uint64_t* new_num_bytes = reinterpret_cast<uint64_t*>(allocator.AllocateData(sizeof(uint64_t)));
			callCudaMemcpyDeviceToHost<uint64_t>(new_num_bytes, sort_result->columns[col]->data_wrapper.offset + limit_const, 1, 0);
			output_relation.columns[col]->data_wrapper.num_bytes = new_num_bytes[0];
		}
	}


	// auto &state = input.global_state.Cast<TopNOperatorState>();
	// auto &gstate = sink_state->Cast<TopNGlobalState>();

	// if (!state.initialized) {
	// 	gstate.heap.InitializeScan(state.state, true);
	// 	state.initialized = true;
	// }
	// gstate.heap.Scan(state.state, chunk);

	// return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Top N GetData time: {:.2f} ms", duration.count()/1000.0);
    return SourceResultType::FINISHED;
}

} // namespace duckdb
