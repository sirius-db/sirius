#include "operator/gpu_physical_table_scan.hpp"

namespace duckdb {

GPUPhysicalTableScan::GPUPhysicalTableScan(vector<LogicalType> types, TableFunction function_p,
                                     unique_ptr<FunctionData> bind_data_p, vector<LogicalType> returned_types_p,
                                     vector<column_t> column_ids_p, vector<idx_t> projection_ids_p,
                                     vector<string> names_p, unique_ptr<TableFilterSet> table_filters_p,
                                     idx_t estimated_cardinality, ExtraOperatorInfo extra_info)
    : GPUPhysicalOperator(PhysicalOperatorType::TABLE_SCAN, std::move(types), estimated_cardinality),
      function(std::move(function_p)), bind_data(std::move(bind_data_p)), returned_types(std::move(returned_types_p)),
      column_ids(std::move(column_ids_p)), projection_ids(std::move(projection_ids_p)), names(std::move(names_p)),
      table_filters(std::move(table_filters_p)), extra_info(extra_info) {
}

SourceResultType
GPUPhysicalTableScan::GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation, OperatorSourceInput &input) const {
  if (output_relation.columns.size() != GetTypes().size()) throw InvalidInputException("Mismatched column count");

  auto table_name = function.to_string(bind_data.get()); //we get it from ParamsToString();

  printf("Table Scanning\n");
  //Find table name in the buffer manager
  //If there is a filter: apply filter, and write to output_relation (late materialized)
  //If there is no filter: write to output_relation (late materialized)
    
  return SourceResultType::FINISHED;
}

} // namespace duckdb