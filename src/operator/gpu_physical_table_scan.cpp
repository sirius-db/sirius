#include "operator/gpu_physical_table_scan.hpp"
#include "gpu_buffer_manager.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "gpu_columns.hpp"

namespace duckdb {
  
GPUPhysicalTableScan::GPUPhysicalTableScan(vector<LogicalType> types, TableFunction function_p,
    unique_ptr<FunctionData> bind_data_p, vector<LogicalType> returned_types_p,
    vector<ColumnIndex> column_ids_p, vector<idx_t> projection_ids_p,
    vector<string> names_p, unique_ptr<TableFilterSet> table_filters_p,
    idx_t estimated_cardinality, ExtraOperatorInfo extra_info,
    vector<Value> parameters_p)
        : GPUPhysicalOperator(PhysicalOperatorType::TABLE_SCAN, std::move(types), estimated_cardinality),
        function(std::move(function_p)), bind_data(std::move(bind_data_p)), returned_types(std::move(returned_types_p)),
        column_ids(std::move(column_ids_p)), projection_ids(std::move(projection_ids_p)), names(std::move(names_p)),
        table_filters(std::move(table_filters_p)), extra_info(extra_info), parameters(std::move(parameters_p)) {

    column_size = new uint64_t[column_ids.size()];
    for (int col = 0; col < column_ids.size(); col++) {
      column_size[col] = 0;
      scanned_types.push_back(returned_types[column_ids[col].GetPrimaryIndex()]);
      scanned_ids.push_back(col);
    }
    fake_table_filters = make_uniq<TableFilterSet>();
    already_cached = new bool[column_ids.size()];
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

void ResolveStringExpression(GPUColumn* string_column, uint64_t* &count, uint64_t* & row_ids, ConstantFilter filter_constant, ExpressionType expression_type) {
    // Read the in the string column
    DataWrapper str_data_wrapper = string_column->data_wrapper;
    uint64_t num_chars = str_data_wrapper.num_bytes;
    char* d_char_data = reinterpret_cast<char*>(str_data_wrapper.data);
    uint64_t num_strings = string_column->column_length;
    uint64_t* d_str_indices = str_data_wrapper.offset;
    // Get the between values
    std::string compare_string = filter_constant.constant.ToString();

    switch (expression_type) {
      case ExpressionType::COMPARE_EQUAL:
        comparisonStringExpression(d_char_data, num_chars, d_str_indices, num_strings, compare_string, 0, row_ids, count);
        break;
      case ExpressionType::COMPARE_NOTEQUAL:
        comparisonStringExpression(d_char_data, num_chars, d_str_indices, num_strings, compare_string, 1, row_ids, count);
        break;
      case ExpressionType::COMPARE_GREATERTHAN:
        comparisonStringExpression(d_char_data, num_chars, d_str_indices, num_strings, compare_string, 2, row_ids, count);
        break;
      case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
        comparisonStringExpression(d_char_data, num_chars, d_str_indices, num_strings, compare_string, 3, row_ids, count);
        break;
      case ExpressionType::COMPARE_LESSTHAN:
        comparisonStringExpression(d_char_data, num_chars, d_str_indices, num_strings, compare_string, 4, row_ids, count);
        break;
      case ExpressionType::COMPARE_LESSTHANOREQUALTO:
        comparisonStringExpression(d_char_data, num_chars, d_str_indices, num_strings, compare_string, 5, row_ids, count);
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
      case ColumnType::VARCHAR:
        ResolveStringExpression(column, count, row_ids, filter_constant, expression_type);
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
    // Get the between values
    std::string lower_string = filter_constant1.constant.ToString();
    std::string upper_string = filter_constant2.constant.ToString();
    bool is_lower_inclusive = filter_constant1.comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO;
    bool is_upper_inclusive = filter_constant1.comparison_type == ExpressionType::COMPARE_LESSTHANOREQUALTO;
    comparisonStringBetweenExpression(d_char_data, num_chars, d_str_indices, num_strings, lower_string, upper_string, is_lower_inclusive, is_upper_inclusive, row_ids, count);
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
        a = gpuBufferManager->customCudaMalloc<T>(column->row_id_count, 0, 0);
        materializeExpression<T>(temp, a, row_ids_input, column->row_id_count);
        size = column->row_id_count;
    } else {
        a = reinterpret_cast<T*> (column->data_wrapper.data);
        size = column->column_length;
    }
    GPUColumn* result = new GPUColumn(size, column->data_wrapper.type, reinterpret_cast<uint8_t*>(a));
    result->is_unique = column->is_unique;
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
  result_column->is_unique = column->is_unique;
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


void HandleArbitraryConstantExpression(GPUColumn** column, uint64_t* &count, uint64_t* &row_ids, ConstantFilter** filter_constant, int num_expr) {
  
  uint8_t** col = new uint8_t*[num_expr];
  uint64_t** offset = new uint64_t*[num_expr];
  uint64_t* constant_offset = new uint64_t[num_expr + 1];
  CompareType* compare_mode = new CompareType[num_expr];
  ScanDataType* data_type = new ScanDataType[num_expr];

  int total_bytes = 0;
  for (int expr = 0; expr < num_expr; expr++) {
    // printf("%d\n", filter_constant[expr]->comparison_type);
    switch(filter_constant[expr]->comparison_type) {
      case ExpressionType::COMPARE_EQUAL: {
        compare_mode[expr] = EQUAL;
        break;
      } case ExpressionType::COMPARE_NOTEQUAL: {
        compare_mode[expr] = NOTEQUAL;
        break;
      } case ExpressionType::COMPARE_GREATERTHAN: {
        compare_mode[expr] = GREATERTHAN;
        break;
      } case ExpressionType::COMPARE_GREATERTHANOREQUALTO: {
        compare_mode[expr] = GREATERTHANOREQUALTO;
        break;
      } case ExpressionType::COMPARE_LESSTHAN: {
        compare_mode[expr] = LESSTHAN;
        break;
      } case ExpressionType::COMPARE_LESSTHANOREQUALTO: {
        compare_mode[expr] = LESSTHANOREQUALTO;
        break;
      } default: {
        throw NotImplementedException("Unsupported comparison type");
      }
    }

    switch(column[expr]->data_wrapper.type) {
      case ColumnType::INT32: {
        total_bytes += sizeof(int);
        data_type[expr] = INT32;
        break;
      } case ColumnType::INT64: {
        total_bytes += sizeof(uint64_t);
        data_type[expr] = INT64;
        break;
      } case ColumnType::FLOAT32: {
        total_bytes += sizeof(float);
        data_type[expr] = FLOAT32;
        break;
      } case ColumnType::FLOAT64: {
        total_bytes += sizeof(double);
        data_type[expr] = FLOAT64;
        break;
      } case ColumnType::VARCHAR: {
        std::string lower_string = filter_constant[expr]->constant.ToString();
        total_bytes += lower_string.size();
        data_type[expr] = VARCHAR;
        break;
      } default: {
        throw NotImplementedException("Unsupported column type");
      }
    }
  }

  uint8_t* constant_compare = new uint8_t[total_bytes];

  uint64_t init_offset = 0;
  for (int expr = 0; expr < num_expr; expr++) {
    col[expr] = column[expr]->data_wrapper.data;
    offset[expr] = column[expr]->data_wrapper.offset;

    switch(column[expr]->data_wrapper.type) {
      case ColumnType::INT32: {
        int temp = filter_constant[expr]->constant.GetValue<int>();
        memcpy(constant_compare + init_offset, &temp, sizeof(int));
        constant_offset[expr] = init_offset;
        init_offset += sizeof(int);
        break;
      } case ColumnType::INT64: {
        uint64_t temp = filter_constant[expr]->constant.GetValue<uint64_t>();
        memcpy(constant_compare + init_offset, &temp, sizeof(uint64_t));
        constant_offset[expr] = init_offset;
        init_offset += sizeof(uint64_t);
        break;
      } case ColumnType::FLOAT32: {
        float temp = filter_constant[expr]->constant.GetValue<float>();
        memcpy(constant_compare + init_offset, &temp, sizeof(float));
        constant_offset[expr] = init_offset;
        init_offset += sizeof(float);
        break;
      } case ColumnType::FLOAT64: {
        double temp = filter_constant[expr]->constant.GetValue<double>();
        memcpy(constant_compare + init_offset, &temp, sizeof(double));
        constant_offset[expr] = init_offset;
        init_offset += sizeof(double);
        break;
      } case ColumnType::VARCHAR: {
        std::string lower_string = filter_constant[expr]->constant.ToString();
        memcpy(constant_compare + init_offset, lower_string.data(), lower_string.size());
        constant_offset[expr] = init_offset;
        init_offset += lower_string.size();
        break;
      } default: {
        throw NotImplementedException("Unsupported column type");
      }
    }
  }
  constant_offset[num_expr] = init_offset;
  
  uint64_t N = column[0]->column_length;
  tableScanExpression(col, offset, constant_compare, constant_offset, data_type, row_ids, count, N, compare_mode, num_expr);
}

class TableScanGlobalSourceState : public GlobalSourceState {
public:
	TableScanGlobalSourceState(ClientContext &context, const GPUPhysicalTableScan &op) {
    if (op.function.init_global) {
			auto filters = table_filters ? *table_filters : GetTableFilters(op);
			TableFunctionInitInput input(op.bind_data.get(), op.column_ids, op.scanned_ids, filters,
			                             op.extra_info.sample_options);

			global_state = op.function.init_global(context, input);
			if (global_state) {
				max_threads = global_state->MaxThreads();
			}
		} else {
			max_threads = 1;
		}
		if (op.function.in_out_function) {
      throw NotImplementedException("In-out function not supported");
		}
	}

	idx_t max_threads = 0;
	unique_ptr<GlobalTableFunctionState> global_state;
	bool in_out_final = false;
	DataChunk input_chunk;
	unique_ptr<TableFilterSet> table_filters;

	optional_ptr<TableFilterSet> GetTableFilters(const GPUPhysicalTableScan &op) const {
		return table_filters ? table_filters.get() : op.fake_table_filters.get();
	}
	idx_t MaxThreads() override {
		return max_threads;
	}
};

class TableScanLocalSourceState : public LocalSourceState {
public:
	TableScanLocalSourceState(ExecutionContext &context, TableScanGlobalSourceState &gstate,
	                          const GPUPhysicalTableScan &op) {
		if (op.function.init_local) {
			TableFunctionInitInput input(op.bind_data.get(), op.column_ids, op.scanned_ids,
			                             gstate.GetTableFilters(op), op.extra_info.sample_options);
			local_state = op.function.init_local(context, input, gstate.global_state.get());
		}
	}

	unique_ptr<LocalTableFunctionState> local_state;
};

unique_ptr<LocalSourceState> GPUPhysicalTableScan::GetLocalSourceState(ExecutionContext &context,
                                                                    GlobalSourceState &gstate) const {
	return make_uniq<TableScanLocalSourceState>(context, gstate.Cast<TableScanGlobalSourceState>(), *this);
}

unique_ptr<GlobalSourceState> GPUPhysicalTableScan::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<TableScanGlobalSourceState>(context, *this);
}

SourceResultType
GPUPhysicalTableScan::GetDataDuckDB(ExecutionContext &exec_context) {
    D_ASSERT(!column_ids.empty());
    auto gpuBufferManager = &(GPUBufferManager::GetInstance());

    TableFunctionToStringInput input(function, bind_data.get());
    auto to_string_result = function.to_string(input);
    string table_name;
    for (const auto &it : to_string_result) {
      if (it.first.compare("Table") == 0) {
        table_name = it.second;
        break;
      }
    }

    GPUIntermediateRelation* table;
    auto &catalog_table = Catalog::GetCatalog(exec_context.client, INVALID_CATALOG);

    bool all_cached = true;
    for (int col = 0; col < column_ids.size(); col++) {
        already_cached[col] = gpuBufferManager->checkIfColumnCached(table_name, names[column_ids[col].GetPrimaryIndex()]);
        if (!already_cached[col]) {
          all_cached = false;
        } 
    }

    if (all_cached) {
      return SourceResultType::FINISHED;
    }

    collection = make_uniq<ColumnDataCollection>(Allocator::Get(exec_context.client), scanned_types);

    // initialize execution context with pipeline = nullptr
    auto g_state = GetGlobalSourceState(exec_context.client);
    auto l_state = GetLocalSourceState(exec_context, *g_state);

    auto &l_state_scan = l_state->Cast<TableScanLocalSourceState>();
    auto &g_state_scan = g_state->Cast<TableScanGlobalSourceState>();

    TableFunctionInput data(bind_data.get(), l_state_scan.local_state.get(), g_state_scan.global_state.get());

    if (function.function) {
      bool has_more_output = true;

      do {
        auto chunk = make_uniq<DataChunk>();
        chunk->Initialize(Allocator::Get(exec_context.client), scanned_types);
        function.function(exec_context.client, data, *chunk);
        has_more_output = chunk->size() > 0;
        // get the size of each column in the chunk
        for (int col = 0; col < column_ids.size(); col++) {
          // if (!already_cached[col]) {
            if (chunk->data[col].GetType() == LogicalType::VARCHAR) {
              Vector string_vector = chunk->data[col];
              string_vector.Flatten(chunk->size());
              for (int row = 0; row < chunk->size(); row++) {
                std::string curr_string = string_vector.GetValue(row).ToString();
                column_size[col] += curr_string.length();
              }
            } else {
              column_size[col] += chunk->data[col].GetAllocationSize(chunk->size());
            }
          // }
        }
        collection->Append(*chunk);
      } while (has_more_output);

      printf("Collection size %d\n", collection->Count());

      uint64_t total_size = 0;
      for (int col = 0; col < column_ids.size(); col++) {
        if (!already_cached[col]) {
          total_size += column_size[col];
        }
      }

      if (gpuBufferManager->gpuCachingPointer[0] + total_size >= gpuBufferManager->cache_size_per_gpu) {
        if (total_size > gpuBufferManager->cache_size_per_gpu) {
          throw InvalidInputException("Total size of columns to be cached is greater than the cache size");
        }
        gpuBufferManager->ResetCache();
        for (int col = 0; col < column_ids.size(); col++) {
          already_cached[col] = false;
          gpuBufferManager->createTableAndColumnInGPU(catalog_table, exec_context.client, table_name, names[column_ids[col].GetPrimaryIndex()]);
        }
      } else {
          for (int col = 0; col < column_ids.size(); col++) {
              if (!already_cached[col]) {
                gpuBufferManager->createTableAndColumnInGPU(catalog_table, exec_context.client, table_name, names[column_ids[col].GetPrimaryIndex()]);
              } 
          }
      }

      ScanDataDuckDB(gpuBufferManager, table_name);
      return SourceResultType::FINISHED;
    } else {
      throw NotImplementedException("Table in-out function not supported");
    }
}

void
GPUPhysicalTableScan::ScanDataDuckDB(GPUBufferManager* gpuBufferManager, string table_name) const{
    if (function.function) {
      bool has_more_output = true;
      // allocate size in gpu buffer manager cpu processing region
      uint8_t** ptr = new uint8_t*[scanned_types.size()];
      uint8_t** d_ptr = new uint8_t*[scanned_types.size()];
      uint8_t** tmp_ptr = new uint8_t*[scanned_types.size()];
      uint64_t** offset_ptr = new uint64_t*[scanned_types.size()];
      uint64_t** d_offset_ptr = new uint64_t*[scanned_types.size()];

      for (int col = 0; col < scanned_types.size(); col++) {
        if (!already_cached[col]) {
          ptr[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(column_size[col]);
          d_ptr[col] = gpuBufferManager->customCudaMalloc<uint8_t>(column_size[col], 0, 1);
          if (scanned_types[col] == LogicalType::VARCHAR) {
            offset_ptr[col] = gpuBufferManager->customCudaHostAlloc<uint64_t>(collection->Count() + 1);
            d_offset_ptr[col] = gpuBufferManager->customCudaMalloc<uint64_t>(collection->Count() + 1, 0, 1);
            offset_ptr[col][0] = 0;
          }
          tmp_ptr[col] = ptr[col];
        }
      } 
      bool scan_initialized = false;
      ColumnDataScanState scan_state;
      uint64_t start_idx = 0;

      do {
        auto result = make_uniq<DataChunk>();
        collection->InitializeScanChunk(*result);
        if (!scan_initialized) {
          collection->InitializeScan(scan_state, ColumnDataScanProperties::DISALLOW_ZERO_COPY);
          scan_initialized = true;
        }
        collection->Scan(scan_state, *result);
        for (int col = 0; col < result->ColumnCount(); col++) {
          if (!already_cached[col]) {
            if (result->data[col].GetType() == LogicalType::VARCHAR) {
              Vector string_vector = result->data[col];
              string_vector.Flatten(result->size());
              for (int row = 0; row < result->size(); row++) {
                std::string curr_string = string_vector.GetValue(row).ToString();
                memcpy(tmp_ptr[col], curr_string.data(), curr_string.length());
                offset_ptr[col][start_idx + row + 1] = offset_ptr[col][start_idx + row] + curr_string.length();
                tmp_ptr[col] += curr_string.length();
              }
            } else {
              memcpy(tmp_ptr[col], result->data[col].GetData(), result->data[col].GetAllocationSize(result->size()));
              tmp_ptr[col] += result->data[col].GetAllocationSize(result->size());
            }
          }
        }
        start_idx += result->size();
        has_more_output = result->size() > 0;
      } while (has_more_output);


      for (int col = 0; col < column_ids.size(); col++) {
        if (!already_cached[col]) {
            if (scanned_types[col] == LogicalType::VARCHAR) {
              if (column_size[col] != offset_ptr[col][collection->Count()]) {
                throw InvalidInputException("Column size mismatch");
              }
              callCudaMemcpyHostToDevice<uint8_t>(d_ptr[col], ptr[col], column_size[col], 0);
              callCudaMemcpyHostToDevice<uint64_t>(d_offset_ptr[col], offset_ptr[col], collection->Count() + 1, 0);
            } else {
              callCudaMemcpyHostToDevice<uint8_t>(d_ptr[col], ptr[col], column_size[col], 0);
            }
        }
      }

      for (int col = 0; col < column_ids.size(); col++) {
        if (!already_cached[col]) {
            auto up_column_name = names[column_ids[col].GetPrimaryIndex()];
            auto up_table_name = table_name;
            transform(up_table_name.begin(), up_table_name.end(), up_table_name.begin(), ::toupper);
            transform(up_column_name.begin(), up_column_name.end(), up_column_name.begin(), ::toupper);
            auto column_it = find(gpuBufferManager->tables[up_table_name]->column_names.begin(), gpuBufferManager->tables[up_table_name]->column_names.end(), up_column_name);
            if (column_it == gpuBufferManager->tables[up_table_name]->column_names.end()) {
                throw InvalidInputException("Column not found");
            }
            int column_idx = column_it - gpuBufferManager->tables[up_table_name]->column_names.begin();
            ColumnType column_type = convertLogicalTypeToColumnType(scanned_types[col]);
            gpuBufferManager->tables[up_table_name]->columns[column_idx]->column_length = collection->Count();
            gpuBufferManager->tables[up_table_name]->length = collection->Count();
            if (scanned_types[col] == LogicalType::VARCHAR) {
              gpuBufferManager->tables[up_table_name]->columns[column_idx]->data_wrapper = DataWrapper(column_type, d_ptr[col], d_offset_ptr[col], collection->Count(), column_size[col], true);
            } else {
              gpuBufferManager->tables[up_table_name]->columns[column_idx]->data_wrapper = DataWrapper(column_type, d_ptr[col], collection->Count());
            }
            printf("Column %s cached in GPU at index %d\n", up_column_name.c_str(), column_idx);
        }
      }
    } else {
      throw NotImplementedException("Table in-out function not supported");
    }
}

SourceResultType
GPUPhysicalTableScan::GetData(GPUIntermediateRelation &output_relation) const {
  auto start = std::chrono::high_resolution_clock::now();
  if (output_relation.columns.size() != GetTypes().size()) throw InvalidInputException("Mismatched column count");

  printf("Getting data\n");
  TableFunctionToStringInput input(function, bind_data.get());
  auto to_string_result = function.to_string(input);
  string table_name;
  for (const auto &it : to_string_result) {
    if (it.first.compare("Table") == 0) {
      table_name = it.second;
      break;
    }
  }

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
            printf("Finding column %s\n", names[column_ids[col].GetPrimaryIndex()].c_str());
            auto column_it = find(table->column_names.begin(), table->column_names.end(), names[column_ids[col].GetPrimaryIndex()]);
            if (column_it == table->column_names.end()) {
                throw InvalidInputException("Column not found");
            }
            auto column_name = table->column_names[column_ids[col].GetPrimaryIndex()];
            printf("Column found %s\n", column_name.c_str());
            if (column_name != names[column_ids[col].GetPrimaryIndex()]) {
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

      int num_expr = 0;
      for (auto &f : table_filters->filters) {
        auto &column_index = f.first;
        auto &filter = f.second;
        table->columns[column_ids[column_index].GetPrimaryIndex()]->row_ids = nullptr;
        table->columns[column_ids[column_index].GetPrimaryIndex()]->row_id_count = 0;

        if (filter->filter_type == TableFilterType::OPTIONAL_FILTER) {
          continue;
        }

        if (column_index < names.size()) {

          if (filter->filter_type == TableFilterType::CONJUNCTION_AND) {
            auto filter_pointer = filter.get();
            auto &filter_conjunction_and = filter_pointer->Cast<ConjunctionAndFilter>();
            for (int expr = 0; expr < filter_conjunction_and.child_filters.size(); expr++) {
                auto& filter_inside = filter_conjunction_and.child_filters[expr];
                if (filter_inside->filter_type == TableFilterType::CONSTANT_COMPARISON) {
                  num_expr++;
                } else if (filter_inside->filter_type == TableFilterType::IS_NOT_NULL) {
                  continue;
                } else {
                  throw NotImplementedException("Filter type not supported");
                }
            }
          } else {
            // count how many filters in table_filters->filters that are not optional filters
            if (filter->filter_type == TableFilterType::CONSTANT_COMPARISON) {
              num_expr++;
            } else {
              throw NotImplementedException("Filter aside from constant comparison not supported");
            }
          }

        }
      }

      ConstantFilter** filter_constants = new ConstantFilter*[num_expr];
      GPUColumn** expression_columns = new GPUColumn*[num_expr];

      int expr_idx = 0;
      for (auto &f : table_filters->filters) {
        auto &column_index = f.first;
        auto &filter = f.second;

        if (filter->filter_type == TableFilterType::OPTIONAL_FILTER) {
          continue;
        }

        if (column_index < names.size()) {
          printf("Reading filter column from index %ld\n", column_ids[column_index].GetPrimaryIndex());

          if (filter->filter_type == TableFilterType::CONJUNCTION_AND) {
            auto filter_pointer = filter.get();
            auto &filter_conjunction_and = filter_pointer->Cast<ConjunctionAndFilter>();

            for (int expr = 0; expr < filter_conjunction_and.child_filters.size(); expr++) {
                auto& filter_inside = filter_conjunction_and.child_filters[expr];
                if (filter_inside->filter_type == TableFilterType::CONSTANT_COMPARISON) {
                  printf("Reading constant comparison filter\n");
                  filter_constants[expr_idx] = &(filter_inside->Cast<ConstantFilter>());
                  expression_columns[expr_idx] = table->columns[column_ids[column_index].GetPrimaryIndex()];
                  expr_idx++;
                } else if (filter_inside->filter_type == TableFilterType::IS_NOT_NULL) {
                  continue;
                } else {
                  throw NotImplementedException("Filter type not supported");
                }
            }

          } else {
            // count how many filters in table_filters->filters
            if (filter->filter_type == TableFilterType::CONSTANT_COMPARISON) {
              filter_constants[expr_idx] = &(filter->Cast<ConstantFilter>());
              expression_columns[expr_idx] = table->columns[column_ids[column_index].GetPrimaryIndex()];
              expr_idx++;
            } else {
              throw NotImplementedException("Filter aside from conjunction and not supported");
            }
          }

        }
      }

      printf("Num expr %d\n", num_expr);
      if (num_expr > 0) {
        count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
        HandleArbitraryConstantExpression(expression_columns, count, row_ids, filter_constants, num_expr);
        if (count[0] == 0) throw NotImplementedException("No match found");
      }
    }
    int index = 0;
    // projection id means that from this set of column ids that are being scanned, which index of column ids are getting projected out
    if (function.filter_prune) {
      for (auto projection_id : projection_ids) {
          printf("Reading column index (late materialized) %ld and passing it to index in output relation %ld\n", column_ids[projection_id].GetPrimaryIndex(), index);
          printf("Writing row IDs to output relation in index %ld\n", index);
          output_relation.columns[index] = new GPUColumn(table->columns[column_ids[projection_id].GetPrimaryIndex()]->column_length, table->columns[column_ids[projection_id].GetPrimaryIndex()]->data_wrapper.type, table->columns[column_ids[projection_id].GetPrimaryIndex()]->data_wrapper.data,
                          table->columns[column_ids[projection_id].GetPrimaryIndex()]->data_wrapper.offset, table->columns[column_ids[projection_id].GetPrimaryIndex()]->data_wrapper.num_bytes, table->columns[column_ids[projection_id].GetPrimaryIndex()]->data_wrapper.is_string_data);
          output_relation.columns[index]->is_unique = table->columns[column_ids[projection_id].GetPrimaryIndex()]->is_unique;
          if (row_ids) {
            output_relation.columns[index]->row_ids = row_ids; 
          }
          if (count) {
            output_relation.columns[index]->row_id_count = count[0];
          }
          index++;
      }

      if (projection_ids.size() == 0) {
        printf("Projection ids size is 0 so we are projecting all columns\n");
        for (auto column_id : column_ids) {
            printf("Reading column index (late materialized) %ld and passing it to index in output relation %ld\n", column_id.GetPrimaryIndex(), index);
            printf("Writing row IDs to output relation in index %ld\n", index);
            output_relation.columns[index] = new GPUColumn(table->columns[column_id.GetPrimaryIndex()]->column_length, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.type, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.data,
                            table->columns[column_id.GetPrimaryIndex()]->data_wrapper.offset, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.num_bytes, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.is_string_data);
            output_relation.columns[index]->is_unique = table->columns[column_id.GetPrimaryIndex()]->is_unique;
            if (row_ids) {
              output_relation.columns[index]->row_ids = row_ids; 
            }
            if (count) {
              output_relation.columns[index]->row_id_count = count[0];
            }
            index++;
        }
      }
    } else {
      //THIS IS FOR INDEX_SCAN
      for (auto column_id : column_ids) {
          printf("Reading column index (late materialized) %ld and passing it to index in output relation %ld\n", column_id.GetPrimaryIndex(), index);
          printf("Writing row IDs to output relation in index %ld\n", index);
          output_relation.columns[index] = new GPUColumn(table->columns[column_id.GetPrimaryIndex()]->column_length, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.type, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.data,
                          table->columns[column_id.GetPrimaryIndex()]->data_wrapper.offset, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.num_bytes, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.is_string_data);
          output_relation.columns[index]->is_unique = table->columns[column_id.GetPrimaryIndex()]->is_unique;
          if (row_ids) {
            output_relation.columns[index]->row_ids = row_ids; 
          }
          if (count) {
            output_relation.columns[index]->row_id_count = count[0];
          }
          index++;
      }
    }
  
    //measure time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Table Scan time: %.2f ms\n", duration.count()/1000.0);
    return SourceResultType::FINISHED;
}

} // namespace duckdb