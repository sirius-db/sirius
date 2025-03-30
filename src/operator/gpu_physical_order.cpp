#include "operator/gpu_physical_order.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_materialize.hpp"

namespace duckdb {

GPUPhysicalOrder::GPUPhysicalOrder(vector<LogicalType> types, vector<BoundOrderByNode> orders, vector<idx_t> projections,
                             idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::ORDER_BY, std::move(types), estimated_cardinality),
      orders(std::move(orders)), projections(std::move(projections)) {
          
    this->sort_result = new GPUIntermediateRelation(projections.size());

}

// SourceResultType
// GPUPhysicalOrder::GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation, OperatorSourceInput &input) const {
  
SourceResultType
GPUPhysicalOrder::GetData(GPUIntermediateRelation &output_relation) const {
  for (int col = 0; col < sort_result->columns.size(); col++) {
    printf("Writing order by result to column %ld\n", col);
    output_relation.columns[col] = sort_result->columns[col];
  }

  return SourceResultType::FINISHED;
}

void ResolveOrderByString(GPUColumn** sort_columns, int* sort_orders, int num_cols) {
  uint8_t** col_keys = new uint8_t*[num_cols];
	uint64_t** col_offsets = new uint64_t*[num_cols];
  uint64_t* col_num_bytes = new uint64_t[num_cols];

  for(int i = 0; i < num_cols; i++) {
    GPUColumn* curr_column = sort_columns[i];
    col_keys[i] = curr_column->data_wrapper.data;
    col_offsets[i] = curr_column->data_wrapper.offset;
    
    std::cout << "ResolveOrderByString: For idx " << i << " got num bytes of " << curr_column->data_wrapper.num_bytes << std::endl;
  }
  uint64_t num_rows = static_cast<uint64_t>(sort_columns[0]->column_length);

  // Sort the results
  orderByString(col_keys, col_offsets, sort_orders, col_num_bytes, num_rows, num_cols);

  // Write the results back
  for(int i = 0; i < num_cols; i++) {
    GPUColumn* curr_column = sort_columns[i];
    curr_column->data_wrapper.data = col_keys[i];
    curr_column->data_wrapper.offset = col_offsets[i];
    curr_column->data_wrapper.num_bytes = col_num_bytes[i];

    std::cout << "ResolveOrderByString: Wrote num bytes of " << col_num_bytes[i] << " for idx " << i << std::endl;
  }
}

SinkResultType 
GPUPhysicalOrder::Sink(GPUIntermediateRelation &input_relation) const {
  printf("Currently order by is not doing anything since it's always after group by\n");
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  GPUColumn** sort_columns = new GPUColumn*[orders.size()];
  int* sort_orders = new int[orders.size()];
  int idx = 0;
  bool string_sort = false;
  for (auto &order : orders) {
    // key_types.push_back(order.expression->return_type);
    // key_executor.AddExpression(*order.expression);
    auto& expr = *order.expression;
    expr.Print();
    if (expr.expression_class != ExpressionClass::BOUND_REF) {
      throw NotImplementedException("Order by expression not supported");
    }

    // Record the column to sort on
    auto &bound_ref_expr = expr.Cast<BoundReferenceExpression>();
    auto input_idx = bound_ref_expr.index;
    printf("Reading order by keys from index %ld\n", input_idx);
    sort_columns[idx] = HandleMaterializeExpression(
      input_relation.columns[input_idx], bound_ref_expr, gpuBufferManager
    );
    if (sort_columns[idx]->data_wrapper.type == ColumnType::VARCHAR) {
      string_sort = true;
    }

    // Record the sort method
    auto sort_method = order.type;
    int sort_type = 0;
    if(sort_method == OrderType::DESCENDING) {
      sort_type = 1;
    }
    sort_orders[idx] = sort_type;
    printf(
      "Order By got sort column: Col Length - %d, Size - %d, Bytes - %d, Sort Order - %d\n", 
      (int) sort_columns[idx]->column_length, (int) sort_columns[idx]->data_wrapper.size,  
      (int) sort_columns[idx]->data_wrapper.num_bytes, sort_orders[idx]
    );

    idx++;
  }

  printf("Sorting the keys\n");
  if(string_sort) {
    ResolveOrderByString(sort_columns, sort_orders, orders.size());
  } else {
    throw NotImplementedException("Non String Order By not yet supported");
  }

  // Copy the sorted columns back into the input relationship
  int sort_cols_idx = 0;
  for (auto &order : orders) {
    auto& expr = *order.expression;
    auto &bound_ref_expr = expr.Cast<BoundReferenceExpression>();
    auto input_idx = bound_ref_expr.index;
    input_relation.columns[input_idx] = sort_columns[sort_cols_idx];

    sort_cols_idx += 1;
  }

  sort_result->columns.resize(projections.size());
  std::cout << "Writing result to relation with " << sort_result->columns.size() << " cols for " << projections.size() << " projections" << std::endl;
  for (auto &projection : projections) {
    printf("Sinking order by projections from index %ld\n", projection);
    sort_result->columns[projection] = input_relation.columns[projection];
  }

  std::cout << "Returning sink result finished" << std::endl;
  return SinkResultType::FINISHED;
}


} // namespace duckdb