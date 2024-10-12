#include "operator/gpu_physical_table_scan.hpp"
#include "gpu_buffer_manager.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"

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
        for (int i = 0; i < table->column_names.size(); i++) {
            printf("Cached Column name: %s\n", table->column_names[i].c_str());
        }
        for (int col = 0; col < column_ids.size(); col++) {
            printf("Finding column %s\n", names[column_ids[col]].c_str());
            auto column_it = find(table->column_names.begin(), table->column_names.end(), names[column_ids[col]]);
            if (column_it == table->column_names.end()) {
                throw InvalidInputException("Column not found");
            } 
            auto column_name = table->column_names[column_ids[col]];
            printf("column found %s\n", column_name.c_str());
            if (column_name != names[column_ids[col]]) {
                throw InvalidInputException("Column name mismatch");
            }
        }
    } else {
        // table not found
        throw InvalidInputException("Table not found");
    }

    uint64_t* row_ids = nullptr;
    uint64_t* count = nullptr;
    if (table_filters) {
      for (auto &f : table_filters->filters) {
        auto &column_index = f.first;
        auto &filter = f.second;
        if (column_index < names.size()) {
          printf("Reading filter column from index %ld\n", column_ids[column_index]);
          printf("filter type %d\n", filter->filter_type);
          if (filter->filter_type == TableFilterType::CONJUNCTION_AND) {
            auto filter_pointer = filter.get();
            auto &filter_conjunction_and = filter_pointer->Cast<ConjunctionAndFilter>();
            for (auto &filter_inside : filter_conjunction_and.child_filters) {
              if (filter_inside->filter_type == TableFilterType::CONSTANT_COMPARISON) {
                printf("Reading constant comparison filter\n");
                auto filter_constant = filter_inside->Cast<ConstantFilter>();
                size_t size = table->columns[column_ids[column_index]]->column_length;
                count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                uint64_t* a = reinterpret_cast<uint64_t*> (table->columns[column_ids[column_index]]->data_wrapper.data);
                uint64_t b = filter_constant.constant.GetValue<uint64_t>();
                //TODO: we have to handle the compare_mode here
                comparisonConstantExpression<uint64_t>(a, b, row_ids, count, (uint64_t) size, 1);
              } else if (filter_inside->filter_type == TableFilterType::IS_NOT_NULL) {
                continue;
              } else {
                throw NotImplementedException("Filter type not supported");
              }
            }
          } else {
            throw NotImplementedException("Filter type not supported");
          }
        }
      }
    }
    int index = 0;
    // projection id means that from this set of column ids that are being scanned, which index of column ids are getting projected out
    for (auto projection_id : projection_ids) {
        printf("Reading column index (late materialized) %ld and passing it to index in output relation %ld\n", column_ids[projection_id], projection_id);
        printf("Writing row IDs to output relation in index %ld\n", index);
        output_relation.columns[index] = table->columns[column_ids[projection_id]];
        // output_relation.columns[index]->row_ids = new uint64_t[1];
        output_relation.length = table->length;
        if (row_ids) {
          printf("Row IDs are not null\n");
          output_relation.columns[index]->row_ids = row_ids; 
        }
        if (count) {
          output_relation.columns[index]->row_id_count = count[0];
          printf("Count is %ld\n", count[0]);
        }
        // printf("%s %d %d\n", output_relation.columns[index]->name.c_str(), output_relation.columns[index]->column_length, output_relation.length);
        index++;
    }
    // for (auto col : table->columns) {
    //   printf("hey table relation column size %d column name %s column type %d\n", col->column_length, col->name.c_str());
    // }
    // for (auto col : output_relation.columns) {
    //   printf("hey source relation column size %d column name %s column type %d\n", col->column_length, col->name.c_str());
    // }
    
    return SourceResultType::FINISHED;
}

} // namespace duckdb