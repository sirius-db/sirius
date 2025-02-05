#include "operator/gpu_physical_order.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

namespace duckdb {

GPUPhysicalOrder::GPUPhysicalOrder(vector<LogicalType> types, vector<BoundOrderByNode> orders, vector<idx_t> projections,
                             idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::ORDER_BY, std::move(types), estimated_cardinality),
      orders(std::move(orders)), projections(std::move(projections)) {

    sort_result = new GPUIntermediateRelation(projections.size());
}
  
SourceResultType
GPUPhysicalOrder::GetData(GPUIntermediateRelation &output_relation) const {
  for (int col = 0; col < sort_result->columns.size(); col++) {
    printf("Writing order by result to column %ld\n", col);
    output_relation.columns[col] = sort_result->columns[col];
  }

  return SourceResultType::FINISHED;
}

SinkResultType 
GPUPhysicalOrder::Sink(GPUIntermediateRelation &input_relation) const {
  printf("Currently order by is not doing anything since it's always after group by\n");
  for (auto &order : orders) {
    auto& expr = *order.expression;
    expr.Print();
    if (expr.expression_class != ExpressionClass::BOUND_REF) {
      throw NotImplementedException("Order by expression not supported");
    }
    auto input_idx = expr.Cast<BoundReferenceExpression>().index;
    printf("Reading order by keys from index %ld\n", input_idx);
    input_relation.checkLateMaterialization(input_idx);
  }

  printf("Sorting the keys\n");

  for (auto &projection : projections) {
    printf("Sinking order by projections from index %ld\n", projection);
    input_relation.checkLateMaterialization(projection);
    sort_result->columns[projection] = input_relation.columns[projection];
  }

  return SinkResultType::FINISHED;
}


} // namespace duckdb