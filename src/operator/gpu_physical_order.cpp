#include "operator/gpu_physical_order.hpp"
#include "operator/gpu_materialize.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

namespace duckdb {

  //call gpu_processing("select n_name, count(*) from nation group by n_name order by n_name");
  //call gpu_processing("select n_name, count(*) from nation group by n_name");
  //call gpu_processing("select n_nationkey, count(*) from nation group by n_nationkey order by n_nationkey");

void
HandleOrderBy(GPUColumn** &order_by_keys, GPUColumn** &projection_columns, const vector<BoundOrderByNode> &orders, uint64_t num_projections) {
	OrderByType* order_by_type = new OrderByType[orders.size()];
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

    sort_result = new GPUIntermediateRelation(projections.size());
}
  
SourceResultType
GPUPhysicalOrder::GetData(GPUIntermediateRelation &output_relation) const {
  for (int col = 0; col < sort_result->columns.size(); col++) {
    printf("Writing order by result to column %d\n", col);
    output_relation.columns[col] = sort_result->columns[col];
  }

  return SourceResultType::FINISHED;
}

SinkResultType 
GPUPhysicalOrder::Sink(GPUIntermediateRelation &input_relation) const {
    // printf("types size %ld\n", types.size());
    // printf("orders size %ld\n", orders.size());
    // printf("order by index %ld\n", orders[0].expression->Cast<BoundReferenceExpression>().index);
    // printf("projections size %ld\n", projections.size());
  // throw NotImplementedException("Order by is not implemented");
  // printf("Currently order by is not doing anything since it's always after group by\n");

  GPUColumn** order_by_keys = new GPUColumn*[orders.size()];
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

  for (int order_idx = 0; order_idx < orders.size(); order_idx++) {
    auto& expr = *orders[order_idx].expression;
    if (expr.expression_class != ExpressionClass::BOUND_REF) {
      throw NotImplementedException("Order by expression not supported");
    }
    auto input_idx = expr.Cast<BoundReferenceExpression>().index;
    order_by_keys[order_idx] = HandleMaterializeExpression(input_relation.columns[input_idx], expr.Cast<BoundReferenceExpression>(), gpuBufferManager);
  }

  GPUColumn** projection_columns = new GPUColumn*[projections.size()];
  
  for (int projection_idx = 0; projection_idx < projections.size(); projection_idx++) {
    auto input_idx = projections[projection_idx];
    // if projection is not in order by keys, then we need to materialize it
    // otherwise we will copy it
    // bool found_projection = false;
    // for (int order_idx = 0; order_idx < orders.size(); order_idx++) {
    //   if (orders[order_idx].expression->Cast<BoundReferenceExpression>().index == input_idx) { 
    //     if (input_relation.columns[input_idx]->data_wrapper.type == ColumnType::VARCHAR) {
    //       uint64_t* temp_offset = gpuBufferManager->customCudaMalloc<uint64_t>(input_relation.columns[input_idx]->column_length, 0, false);
    //       uint8_t* temp_column = gpuBufferManager->customCudaMalloc<uint8_t>(input_relation.columns[input_idx]->data_wrapper.num_bytes, 0, false);
    //       callCudaMemcpyDeviceToDevice<uint64_t>(temp_offset, input_relation.columns[input_idx]->data_wrapper.offset, input_relation.columns[input_idx]->column_length, 0);
    //       callCudaMemcpyDeviceToDevice<uint8_t>(temp_column, input_relation.columns[input_idx]->data_wrapper.data, input_relation.columns[input_idx]->data_wrapper.num_bytes, 0);
    //       projection_columns[projection_idx] = new GPUColumn(input_relation.columns[input_idx]->column_length, input_relation.columns[input_idx]->data_wrapper.type, temp_column, temp_offset, input_relation.columns[input_idx]->data_wrapper.num_bytes, true);
    //     } else {
    //       uint8_t* temp_column = gpuBufferManager->customCudaMalloc<uint8_t>(input_relation.columns[input_idx]->data_wrapper.num_bytes, 0, false);
    //       callCudaMemcpyDeviceToDevice<uint8_t>(temp_column, input_relation.columns[input_idx]->data_wrapper.data, input_relation.columns[input_idx]->data_wrapper.num_bytes, 0);
    //       projection_columns[projection_idx] = new GPUColumn(input_relation.columns[input_idx]->column_length, input_relation.columns[input_idx]->data_wrapper.type, temp_column);
    //     }
    //     found_projection = true;
    //     break;
    //   }
    // }

    // if (!found_projection) {
      auto expr = BoundReferenceExpression(LogicalType::ANY, input_idx);
      projection_columns[projection_idx] = HandleMaterializeExpression(input_relation.columns[input_idx], expr, gpuBufferManager);
    // }
  }


  HandleOrderBy(order_by_keys, projection_columns, orders, projections.size());

  for (int col = 0; col < projections.size(); col++) {
    sort_result->columns[col] = projection_columns[col];
  }

  return SinkResultType::FINISHED;
}


} // namespace duckdb