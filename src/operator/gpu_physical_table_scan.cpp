#include "operator/gpu_physical_table_scan.hpp"
#include "gpu_buffer_manager.hpp"

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

// SourceResultType
// GPUPhysicalTableScan::GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation, OperatorSourceInput &input) const {

SourceResultType
GPUPhysicalTableScan::GetData(GPUIntermediateRelation &output_relation) const {
  if (output_relation.columns.size() != GetTypes().size()) throw InvalidInputException("Mismatched column count");

  auto table_name = function.to_string(bind_data.get()); //we get it from ParamsToString();

  printf("Table Scanning %s\n", table_name.c_str());
  //Find table name in the buffer manager
  auto gpuBufferManager = &(GPUBufferManager::GetInstance());
  auto it = gpuBufferManager->tables.find(table_name);
  GPUIntermediateRelation* table;
  //If there is a filter: apply filter, and write to output_relation (late materialized)
    if (it != gpuBufferManager->tables.end()) {
        // Key found, print the value
        table = it->second;
        for (int col = 0; col < column_ids.size(); col++) {
            auto column_it = find(table->column_names.begin(), table->column_names.end(), names[col]);
            if (column_it == table->column_names.end()) {
                throw InvalidInputException("Column not found");
            }
            auto column_name = table->column_names[column_ids[col]];
            printf("column found %s\n", column_name.c_str());
            if (column_name != names[col]) {
                throw InvalidInputException("Column name mismatch");
            }
        }
    } else {
        // table not found
        throw InvalidInputException("Table not found");
    }

    if (table_filters) {
      for (auto &f : table_filters->filters) {
        auto &column_index = f.first;
        auto &filter = f.second;
        if (column_index < names.size()) {
          printf("Reading filter column from index %ld\n", column_index);
        }
      }
    }

    int index = 0;
    for (auto projection_id : projection_ids) {
        printf("Reading column index (late materialized) %ld and passing it to index in output relation %ld\n", projection_id, index);
        printf("Writing row IDs to output relation in index %ld\n", index);
        output_relation.columns[index] = table->columns[projection_id];
        output_relation.columns[index]->row_ids = new uint64_t[1];
        index++;
    }
  //If there is no filter: write to output_relation (late materialized)
    
  return SourceResultType::FINISHED;
}

} // namespace duckdb