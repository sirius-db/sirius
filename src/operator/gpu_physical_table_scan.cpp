#include "operator/gpu_physical_table_scan.hpp"
#include "gpu_buffer_manager.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"

#include <algorithm>
#include <string>

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


template <typename T>
void ResolveTypeComparisonConstantExpression (GPUColumn* column, uint64_t* &count, uint64_t* & row_ids, ConstantFilter filter_constant, ExpressionType expression_type) {
    T* a = reinterpret_cast<T*> (column->data_wrapper.data);
    T b = filter_constant.constant.GetValue<T>();
    T c = 0;
    size_t size = column->column_length;
    switch (expression_type) {
      case ExpressionType::COMPARE_EQUAL:
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 0);
        break;
      case ExpressionType::COMPARE_NOTEQUAL:
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 1);
        break;
      case ExpressionType::COMPARE_GREATERTHAN:
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 2);
        break;
      case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 3);
        break;
      case ExpressionType::COMPARE_LESSTHAN:
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 4);
        break;
      case ExpressionType::COMPARE_LESSTHANOREQUALTO:
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 5);
        break;
      default:
        throw NotImplementedException("Comparison type not supported");
    }
}

void HandleComparisonConstantExpression(GPUColumn* column, uint64_t* &count, uint64_t* &row_ids, ConstantFilter filter_constant, ExpressionType expression_type) {
    switch(column->data_wrapper.type) {
      case ColumnType::INT32:
        ResolveTypeComparisonConstantExpression<int>(column, count, row_ids, filter_constant, expression_type);
        break;
      case ColumnType::INT64:
        ResolveTypeComparisonConstantExpression<uint64_t>(column, count, row_ids, filter_constant, expression_type);
        break;
      case ColumnType::FLOAT32:
        ResolveTypeComparisonConstantExpression<float>(column, count, row_ids, filter_constant, expression_type);
        break;
      case ColumnType::FLOAT64:
        ResolveTypeComparisonConstantExpression<double>(column, count, row_ids, filter_constant, expression_type);
        break;
      default:
        throw NotImplementedException("HandleComparisonConstantExpression Unsupported column type");
    }
}

template <typename T>
void ResolveTypeBetweenExpression (GPUColumn* column, uint64_t* &count, uint64_t* & row_ids, ConstantFilter filter_constant1, ConstantFilter filter_constant2) {
    T* a = reinterpret_cast<T*> (column->data_wrapper.data);
    T b = filter_constant1.constant.GetValue<T>();
    T c = filter_constant2.constant.GetValue<T>();
    size_t size = column->column_length;
    comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 6);
}

void HandleBetweenExpression(GPUColumn* column, uint64_t* &count, uint64_t* &row_ids, ConstantFilter filter_constant1, ConstantFilter filter_constant2) {
    switch(column->data_wrapper.type) {
      case ColumnType::INT32:
        ResolveTypeBetweenExpression<int>(column, count, row_ids, filter_constant1, filter_constant2);
        break;
      case ColumnType::INT64:
        ResolveTypeBetweenExpression<uint64_t>(column, count, row_ids, filter_constant1, filter_constant2);
        break;
      case ColumnType::FLOAT32:
        ResolveTypeBetweenExpression<float>(column, count, row_ids, filter_constant1, filter_constant2);
        break;
      case ColumnType::FLOAT64:
        ResolveTypeBetweenExpression<double>(column, count, row_ids, filter_constant1, filter_constant2);
        break;
      default:
        throw NotImplementedException("HandleBetweenExpression Unsupported column type");
    }
}

SourceResultType
GPUPhysicalTableScan::GetData(GPUIntermediateRelation &output_relation) const {
  if (output_relation.columns.size() != GetTypes().size()) throw InvalidInputException("Mismatched column count");

  auto table_name = function.to_string(bind_data.get()); //we get it from ParamsToString();
  auto gpuBufferManager = &(GPUBufferManager::GetInstance());

  printf("Table Scanning %s\n", table_name.c_str());
  std::cout << "Existing buffer manager tables searching for table " << table_name << ": ";
  for (const auto& pair : gpuBufferManager->tables) {
    std::cout << pair.first << " ";
  }
  std::cout << std::endl;

  //Find table name in the buffer manager
  std::string upper_table_name = table_name;
  std::transform(upper_table_name.begin(), upper_table_name.end(), upper_table_name.begin(), ::toupper);
  std::cout << "Searching for table " << upper_table_name << std::endl;

  auto it = gpuBufferManager->tables.find(upper_table_name);
  GPUIntermediateRelation* table;

  //If there is a filter: apply filter, and write to output_relation (late materialized)
    if (it != gpuBufferManager->tables.end()) {
        // Key found, print the value
        table = it->second;
        for (int i = 0; i < table->column_names.size(); i++) {
            printf("Cached Column name: %s\n", table->column_names[i].c_str());
        }
        for (int col = 0; col < column_ids.size(); col++) {
            std::string search_col = names[column_ids[col]];
            std::string upper_col_name = search_col;
            std::transform(upper_col_name.begin(), upper_col_name.end(), upper_col_name.begin(), ::toupper);
            printf("Finding column %s\n", upper_col_name.c_str());

            auto column_it = find(table->column_names.begin(), table->column_names.end(), upper_col_name);
            if (column_it == table->column_names.end()) {
                throw InvalidInputException("Column not found");
            } 

            auto column_name = table->column_names[column_ids[col]];
            printf("Column found %s\n", column_name.c_str());
            if (column_name != upper_col_name) {
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
          // printf("Reading filter column from index %ld\n", column_ids[column_index]);
          // printf("filter type %d\n", filter->filter_type);
          if (filter->filter_type == TableFilterType::CONJUNCTION_AND) {
            auto filter_pointer = filter.get();
            auto &filter_conjunction_and = filter_pointer->Cast<ConjunctionAndFilter>();
            if (filter_conjunction_and.child_filters.size() == 3) {
              if (filter_conjunction_and.child_filters[0]->filter_type == TableFilterType::CONSTANT_COMPARISON && filter_conjunction_and.child_filters[1]->filter_type == TableFilterType::CONSTANT_COMPARISON && filter_conjunction_and.child_filters[2]->filter_type == TableFilterType::IS_NOT_NULL) {
                auto filter_constant1 = filter_conjunction_and.child_filters[0]->Cast<ConstantFilter>();
                auto filter_constant2 = filter_conjunction_and.child_filters[1]->Cast<ConstantFilter>();
                ExpressionType expression_type1 = filter_constant1.comparison_type;
                ExpressionType expression_type2 = filter_constant2.comparison_type;
                count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                if (filter_constant1.comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO && filter_constant2.comparison_type == ExpressionType::COMPARE_LESSTHANOREQUALTO) {
                    HandleBetweenExpression(table->columns[column_ids[column_index]], count, row_ids, filter_constant1, filter_constant2);
                } else {
                    throw NotImplementedException("Between expression not supported");
                }
                printf("Count %ld\n", count[0]);
              }
            } else {
              for (auto &filter_inside : filter_conjunction_and.child_filters) {
                if (filter_inside->filter_type == TableFilterType::CONSTANT_COMPARISON) {
                  // printf("Reading constant comparison filter\n");
                  auto filter_constant = filter_inside->Cast<ConstantFilter>();
                  ExpressionType expression_type = filter_constant.comparison_type;
                  size_t size = table->columns[column_ids[column_index]]->column_length;
                  count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                  HandleComparisonConstantExpression(table->columns[column_ids[column_index]], count, row_ids, filter_constant, expression_type);
                  printf("Count %ld\n", count[0]);
                  if (count[0] == 0) throw NotImplementedException("No match found");
                } else if (filter_inside->filter_type == TableFilterType::IS_NOT_NULL) {
                  continue;
                } else {
                  throw NotImplementedException("Filter type not supported");
                }
              }
            }
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
          output_relation.columns[index]->row_ids = row_ids; 
        }
        if (count) {
          output_relation.columns[index]->row_id_count = count[0];
        }
        index++;
    }
    
    return SourceResultType::FINISHED;
}

} // namespace duckdb