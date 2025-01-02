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
        throw NotImplementedException("Unsupported column type");
    }
}

template <typename T>
void ResolveTypeBetweenExpression (GPUColumn* column, uint64_t* &count, uint64_t* & row_ids, ConstantFilter filter_constant1, ConstantFilter filter_constant2) {
    T* a = reinterpret_cast<T*> (column->data_wrapper.data);
    T b = filter_constant1.constant.GetValue<T>();
    T c = filter_constant2.constant.GetValue<T>();
    size_t size = column->column_length;
    // Determine operation tyoe
    bool is_lower_inclusive = filter_constant1.comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO;
    bool is_upper_inclusive = filter_constant2.comparison_type == ExpressionType::COMPARE_LESSTHANOREQUALTO;
    int op_mode;
    if(is_lower_inclusive && is_upper_inclusive) {
      op_mode = 6;
    } else if(is_lower_inclusive && !is_upper_inclusive) {
      op_mode = 8;
    } else if(!is_lower_inclusive && is_upper_inclusive) {
      op_mode = 9;
    } else {
      op_mode = 10;
    }
    // printf("Op mode %d\n", op_mode);
    comparisonConstantExpression<T>(a, b, c, row_ids, count, size, op_mode);
}

void ResolveStringBetweenExpression(GPUColumn* string_column, uint64_t* &count, uint64_t* & row_ids, ConstantFilter filter_constant1, ConstantFilter filter_constant2) {
  // Read the in the string column
  DataWrapper str_data_wrapper = string_column->data_wrapper;
  uint64_t num_chars = str_data_wrapper.num_bytes;
  char* d_char_data = reinterpret_cast<char*>(str_data_wrapper.data);
  uint64_t num_strings = string_column->column_length;
  uint64_t* d_str_indices = str_data_wrapper.offset;
  std::cout << "Resolve String between got column with values: Num Chars - " << num_chars << ", Num Strings - " << num_strings << std::endl;
  // Get the between values
  std::string lower_string = filter_constant1.constant.ToString();
  std::string upper_string = filter_constant2.constant.ToString();
  bool is_lower_inclusive = filter_constant1.comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO;
  bool is_upper_inclusive = filter_constant1.comparison_type == ExpressionType::COMPARE_LESSTHANOREQUALTO;
  comparisonStringExpression(d_char_data, num_chars, d_str_indices, num_strings, lower_string, upper_string, is_lower_inclusive, is_upper_inclusive, row_ids, count);
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
      case ColumnType::VARCHAR:
        ResolveStringBetweenExpression(column, count, row_ids, filter_constant1, filter_constant2);
        break;
      default:
        throw NotImplementedException("Unsupported column type");
    }
}

template <typename T>
GPUColumn* 
ResolveTypeMaterializeExpression(GPUColumn* column, GPUBufferManager* gpuBufferManager) {
    // GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    size_t size;
    T* a = nullptr;
    if (column->row_ids != nullptr) {
        T* temp = reinterpret_cast<T*> (column->data_wrapper.data);
        uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (column->row_ids);
        a = gpuBufferManager->customCudaMalloc<T>(column->row_id_count, 0, 0).data_;
        materializeExpression<T>(temp, a, row_ids_input, column->row_id_count);
        size = column->row_id_count;
    } else {
        a = reinterpret_cast<T*> (column->data_wrapper.data);
        size = column->column_length;
    }
    GPUColumn* result = new GPUColumn(size, column->data_wrapper.type, reinterpret_cast<uint8_t*>(a));
    return result;
}

GPUColumn* ResolveStringMateralizeExpression(GPUColumn* column, GPUBufferManager* gpuBufferManager) {
  // Column is already materalized so just return it
  size_t num_rows;
  uint8_t* result = nullptr;
  uint64_t* result_offset = nullptr;
  uint64_t* new_num_bytes;
  if(column->row_ids != nullptr) {
    // Materalize the string column
    uint8_t* data = column->data_wrapper.data;
    uint64_t* offset = column->data_wrapper.offset;
    uint64_t* row_ids = column->row_ids;
    num_rows = column->row_id_count;
    materializeString(data, offset, result, result_offset, row_ids, new_num_bytes, num_rows);
  } else {
    result = column->data_wrapper.data;
    result_offset = column->data_wrapper.offset;
    num_rows = column->column_length;
    new_num_bytes = new uint64_t[1];
    new_num_bytes[0] = column->data_wrapper.num_bytes;
  }
  GPUColumn* result_column = new GPUColumn(num_rows, ColumnType::VARCHAR, reinterpret_cast<uint8_t*>(result), result_offset, new_num_bytes[0], true);
  return result_column;
}

GPUColumn* 
HandleMaterializeExpression(GPUColumn* column, GPUBufferManager* gpuBufferManager) {
    switch(column->data_wrapper.type) {
        case ColumnType::INT32:
            return ResolveTypeMaterializeExpression<int>(column, gpuBufferManager);
        case ColumnType::INT64:
            return ResolveTypeMaterializeExpression<uint64_t>(column, gpuBufferManager);
        case ColumnType::FLOAT32:
            return ResolveTypeMaterializeExpression<float>(column, gpuBufferManager);
        case ColumnType::FLOAT64:
            return ResolveTypeMaterializeExpression<double>(column, gpuBufferManager);
        case ColumnType::BOOLEAN:
            return ResolveTypeMaterializeExpression<uint8_t>(column, gpuBufferManager);
        case ColumnType::VARCHAR:
            return ResolveStringMateralizeExpression(column, gpuBufferManager);
        default:
            throw NotImplementedException("Unsupported column type");
    }
}

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
            printf("Column found %s\n", column_name.c_str());
            if (column_name != names[column_ids[col]]) {
                throw InvalidInputException("Column name mismatch");
            }
        }
    } else {
        // table not found
        throw InvalidInputException("Table not found");
    }

    uint64_t* row_ids = nullptr;
    uint64_t* prev_row_ids = nullptr;
    uint64_t* count = nullptr;
    uint64_t prev_row_ids_count = 0;
    if (table_filters) {
      for (auto &f : table_filters->filters) {
        auto &column_index = f.first;
        auto &filter = f.second;
        if (column_index < names.size()) {
          printf("Reading filter column from index %ld\n", column_ids[column_index]);
          // printf("filter type %d\n", filter->filter_type);
          if (filter->filter_type == TableFilterType::CONJUNCTION_AND) {
            auto filter_pointer = filter.get();
            auto &filter_conjunction_and = filter_pointer->Cast<ConjunctionAndFilter>();
            if (filter_conjunction_and.child_filters.size() == 3) {
              printf("This is between filter\n");
              if (filter_conjunction_and.child_filters[0]->filter_type == TableFilterType::CONSTANT_COMPARISON && filter_conjunction_and.child_filters[1]->filter_type == TableFilterType::CONSTANT_COMPARISON && filter_conjunction_and.child_filters[2]->filter_type == TableFilterType::IS_NOT_NULL) {
                auto filter_constant1 = filter_conjunction_and.child_filters[0]->Cast<ConstantFilter>();
                auto filter_constant2 = filter_conjunction_and.child_filters[1]->Cast<ConstantFilter>();
                ExpressionType expression_type1 = filter_constant1.comparison_type;
                ExpressionType expression_type2 = filter_constant2.comparison_type;
                count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0).data_;

                if (prev_row_ids) {
                  printf("The previous row ids count is %ld\n", prev_row_ids_count);
                  table->columns[column_ids[column_index]]->row_ids = prev_row_ids;
                  table->columns[column_ids[column_index]]->row_id_count = prev_row_ids_count;
                }
                GPUColumn* materialized_column = HandleMaterializeExpression(table->columns[column_ids[column_index]], gpuBufferManager);
                table->columns[column_ids[column_index]]->row_ids = nullptr;
                table->columns[column_ids[column_index]]->row_id_count = 0;
                bool is_first_greater = filter_constant1.comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO || filter_constant1.comparison_type == ExpressionType::COMPARE_GREATERTHAN;
                bool is_second_greater = filter_constant2.comparison_type == ExpressionType::COMPARE_LESSTHANOREQUALTO || filter_constant2.comparison_type == ExpressionType::COMPARE_LESSTHAN;
                if (is_first_greater && is_second_greater) {
                  HandleBetweenExpression(materialized_column, count, row_ids, filter_constant1, filter_constant2);
                // if (filter_constant1.comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO && filter_constant2.comparison_type == ExpressionType::COMPARE_LESSTHANOREQUALTO) {
                    // HandleBetweenExpression(materialized_column, count, row_ids, filter_constant1, filter_constant2);
                } else {
                    throw NotImplementedException("Between expression not supported");
                }
                printf("Count %ld\n", count[0]);
                if (count[0] == 0) throw NotImplementedException("No match found");
              }
            } else {
              for (auto &filter_inside : filter_conjunction_and.child_filters) {
                if (filter_inside->filter_type == TableFilterType::CONSTANT_COMPARISON) {
                  printf("Reading constant comparison filter\n");
                  auto filter_constant = filter_inside->Cast<ConstantFilter>();
                  ExpressionType expression_type = filter_constant.comparison_type;
                  size_t size = table->columns[column_ids[column_index]]->column_length;
                  count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0).data_;

                  if (prev_row_ids) {
                    printf("The previous row ids count is %ld\n", prev_row_ids_count);
                    table->columns[column_ids[column_index]]->row_ids = prev_row_ids;
                    table->columns[column_ids[column_index]]->row_id_count = prev_row_ids_count;
                  }
                  GPUColumn* materialized_column = HandleMaterializeExpression(table->columns[column_ids[column_index]], gpuBufferManager);
                  table->columns[column_ids[column_index]]->row_ids = nullptr;
                  table->columns[column_ids[column_index]]->row_id_count = 0;
                  HandleComparisonConstantExpression(materialized_column, count, row_ids, filter_constant, expression_type);
                  printf("Count %ld\n", count[0]);
                  if (count[0] == 0) throw NotImplementedException("No match found");
                } else if (filter_inside->filter_type == TableFilterType::IS_NOT_NULL) {
                  continue;
                } else {
                  throw NotImplementedException("Filter type not supported");
                }
              }
            }
          } else {
            throw NotImplementedException("Filter aside from conjunction and not supported");
          }

          if (prev_row_ids) {
            uint64_t* new_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(count[0], 0, 0).data_;
            materializeExpression<uint64_t>(prev_row_ids, new_row_ids, row_ids, count[0]);
            prev_row_ids = new_row_ids;
            prev_row_ids_count = count[0];
          } else {
            prev_row_ids = row_ids;
            prev_row_ids_count = count[0];
          }
        }
      }
    }
    int index = 0;
    // projection id means that from this set of column ids that are being scanned, which index of column ids are getting projected out
    for (auto projection_id : projection_ids) {
        printf("Reading column index (late materialized) %ld and passing it to index in output relation %ld\n", column_ids[projection_id], projection_id);
        printf("Writing row IDs to output relation in index %ld\n", index);
        // output_relation.columns[index] = new GPUColumn(table->columns[column_ids[projection_id]]->column_length, table->columns[column_ids[projection_id]]->data_wrapper.type, table->columns[column_ids[projection_id]]->data_wrapper.data);
        // output_relation.columns[index]->data_wrapper.offset = table->columns[column_ids[projection_id]]->data_wrapper.offset;
        // output_relation.columns[index]->data_wrapper.num_bytes = table->columns[column_ids[projection_id]]->data_wrapper.num_bytes;
        output_relation.columns[index] = new GPUColumn(table->columns[column_ids[projection_id]]->column_length, table->columns[column_ids[projection_id]]->data_wrapper.type, table->columns[column_ids[projection_id]]->data_wrapper.data,
                        table->columns[column_ids[projection_id]]->data_wrapper.offset, table->columns[column_ids[projection_id]]->data_wrapper.num_bytes, table->columns[column_ids[projection_id]]->data_wrapper.is_string_data);
        if (row_ids) {
          output_relation.columns[index]->row_ids = prev_row_ids; 
        }
        if (count) {
          output_relation.columns[index]->row_id_count = prev_row_ids_count;
        }
        index++;
    }
    
    return SourceResultType::FINISHED;
}

} // namespace duckdb