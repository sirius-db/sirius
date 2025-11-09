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

#include "config.hpp"
#include "operator/gpu_physical_table_scan.hpp"
#include "gpu_buffer_manager.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "gpu_columns.hpp"
#include "gpu_materialize.hpp"
#include "utils.hpp"
#include "log/logging.hpp"

namespace duckdb {

uint64_t GetChunkDataByteSize(LogicalType type, idx_t cardinality) {
		auto physical_size = GetTypeIdSize(type.InternalType());
		return cardinality * physical_size;
}
  
GPUPhysicalTableScan::GPUPhysicalTableScan(vector<LogicalType> types, TableFunction function_p,
    unique_ptr<FunctionData> bind_data_p, vector<LogicalType> returned_types_p,
    vector<ColumnIndex> column_ids_p, vector<idx_t> projection_ids_p,
    vector<string> names_p, unique_ptr<TableFilterSet> table_filters_p,
    idx_t estimated_cardinality, ExtraOperatorInfo extra_info,
    vector<Value> parameters_p)
        : GPUPhysicalOperator(PhysicalOperatorType::TABLE_SCAN, std::move(types), estimated_cardinality),
        function(std::move(function_p)), bind_data(std::move(bind_data_p)), returned_types(std::move(returned_types_p)),
        column_ids(std::move(column_ids_p)), projection_ids(std::move(projection_ids_p)), names(std::move(names_p)),
        table_filters(std::move(table_filters_p)), extra_info(extra_info), parameters(std::move(parameters_p)),
        gen_row_id_column(column_ids.back().GetPrimaryIndex() == DConstants::INVALID_INDEX) {
    
    auto num_cols = column_ids.size() - gen_row_id_column;
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    column_size = gpuBufferManager->customCudaHostAlloc<uint64_t>(column_ids.size());
    mask_size = gpuBufferManager->customCudaHostAlloc<uint64_t>(column_ids.size());
    for (int col = 0; col < num_cols; col++) {
      column_size[col] = 0;
      mask_size[col] = 0;
      scanned_types.push_back(returned_types[column_ids[col].GetPrimaryIndex()]);
      scanned_ids.push_back(col);
    }

    if(num_cols == 0) { // Ensure that scanned_types and ids are properly initialized 
      scanned_types.push_back(LogicalType(LogicalTypeId::UBIGINT));
    } 

    fake_table_filters = make_uniq<TableFilterSet>();
    already_cached = gpuBufferManager->customCudaHostAlloc<bool>(column_ids.size());
    if (Config::USE_OPT_TABLE_SCAN) {
      num_rows = 0;
      cuda_streams.resize(Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS);
      for (int i = 0; i < Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS; i++) {
        cudaStreamCreate(&cuda_streams[i]);
      }
    }
    SIRIUS_LOG_DEBUG("Table scan column ids: {}", column_ids.size());
}


template <typename T>
void ResolveTypeComparisonConstantExpression (shared_ptr<GPUColumn> column, uint64_t* &count, uint64_t* & row_ids, ConstantFilter filter_constant, ExpressionType expression_type) {
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

void ResolveStringExpression(shared_ptr<GPUColumn> string_column, uint64_t* &count, uint64_t* & row_ids, ConstantFilter filter_constant, ExpressionType expression_type) {
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

void HandleComparisonConstantExpression(shared_ptr<GPUColumn> column, uint64_t* &count, uint64_t* &row_ids, ConstantFilter filter_constant, ExpressionType expression_type) {
    switch(column->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT32:
        ResolveTypeComparisonConstantExpression<int>(column, count, row_ids, filter_constant, expression_type);
        break;
      case GPUColumnTypeId::INT64:
        ResolveTypeComparisonConstantExpression<uint64_t>(column, count, row_ids, filter_constant, expression_type);
        break;
      case GPUColumnTypeId::FLOAT32:
        ResolveTypeComparisonConstantExpression<float>(column, count, row_ids, filter_constant, expression_type);
        break;
      case GPUColumnTypeId::FLOAT64:
        ResolveTypeComparisonConstantExpression<double>(column, count, row_ids, filter_constant, expression_type);
        break;
      case GPUColumnTypeId::VARCHAR:
        ResolveStringExpression(column, count, row_ids, filter_constant, expression_type);
        break;
      default:
        throw NotImplementedException("Unsupported sirius column type in `HandleComparisonConstantExpression`: %d",
                                      static_cast<int>(column->data_wrapper.type.id()));
    }
}

template <typename T>
void ResolveTypeBetweenExpression (shared_ptr<GPUColumn> column, uint64_t* &count, uint64_t* & row_ids, ConstantFilter filter_constant1, ConstantFilter filter_constant2) {
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
    comparisonConstantExpression<T>(a, b, c, row_ids, count, size, op_mode);
}

void ResolveStringBetweenExpression(shared_ptr<GPUColumn> string_column, uint64_t* &count, uint64_t* & row_ids, ConstantFilter filter_constant1, ConstantFilter filter_constant2) {
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

void HandleBetweenExpression(shared_ptr<GPUColumn> column, uint64_t* &count, uint64_t* &row_ids, ConstantFilter filter_constant1, ConstantFilter filter_constant2) {
    switch(column->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT32:
        ResolveTypeBetweenExpression<int>(column, count, row_ids, filter_constant1, filter_constant2);
        break;
      case GPUColumnTypeId::INT64:
        ResolveTypeBetweenExpression<uint64_t>(column, count, row_ids, filter_constant1, filter_constant2);
        break;
      case GPUColumnTypeId::FLOAT32:
        ResolveTypeBetweenExpression<float>(column, count, row_ids, filter_constant1, filter_constant2);
        break;
      case GPUColumnTypeId::FLOAT64:
        ResolveTypeBetweenExpression<double>(column, count, row_ids, filter_constant1, filter_constant2);
        break;
      case GPUColumnTypeId::VARCHAR:
        ResolveStringBetweenExpression(column, count, row_ids, filter_constant1, filter_constant2);
        break;
      default:
        throw NotImplementedException("Unsupported sirius column type in `HandleBetweenExpression`: %d",
                                      static_cast<int>(column->data_wrapper.type.id()));
    }
}

void HandleArbitraryConstantExpression(vector<shared_ptr<GPUColumn>> &column, uint64_t* &count, uint64_t* &row_ids, ConstantFilter** &filter_constant, int num_expr) {
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  uint8_t** col = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_expr);
  uint32_t** bitmask = gpuBufferManager->customCudaHostAlloc<uint32_t*>(num_expr);
  uint64_t** offset = gpuBufferManager->customCudaHostAlloc<uint64_t*>(num_expr);
  uint64_t* constant_offset = gpuBufferManager->customCudaHostAlloc<uint64_t>(num_expr + 1);
  CompareType* compare_mode = gpuBufferManager->customCudaHostAlloc<CompareType>(num_expr);
  ScanDataType* data_type = gpuBufferManager->customCudaHostAlloc<ScanDataType>(num_expr);

  uint64_t total_bytes = 0;
  for (int expr = 0; expr < num_expr; expr++) {
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
      } case ExpressionType::OPERATOR_IS_NULL: {
        compare_mode[expr] = IS_NULL;
        break;
      } case ExpressionType::OPERATOR_IS_NOT_NULL: {
        compare_mode[expr] = IS_NOT_NULL;
        break;
      } default: {
        throw NotImplementedException("Unsupported comparison type");
      }
    }

    if (filter_constant[expr]->constant.IsNull()) {
      total_bytes += sizeof(int);
      data_type[expr] = ScanDataType::SQLNULL;
      continue;
    }

    switch(column[expr]->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT16: {
        total_bytes += sizeof(int16_t);
        data_type[expr] = INT16;
        break;
      } case GPUColumnTypeId::INT32: {
        total_bytes += sizeof(int);
        data_type[expr] = INT32;
        break;
      } case GPUColumnTypeId::INT64: {
        total_bytes += sizeof(int64_t);
        data_type[expr] = INT64;
        break;
      } case GPUColumnTypeId::FLOAT32: {
        total_bytes += sizeof(float);
        data_type[expr] = FLOAT32;
        break;
      } case GPUColumnTypeId::FLOAT64: {
        total_bytes += sizeof(double);
        data_type[expr] = FLOAT64;
        break;
      } case GPUColumnTypeId::DATE: {
        total_bytes += sizeof(int);
        data_type[expr] = DATE;
        break;
      } case GPUColumnTypeId::VARCHAR: {
        std::string lower_string = filter_constant[expr]->constant.ToString();
        total_bytes += lower_string.size();
        data_type[expr] = VARCHAR;
        break;
      } case GPUColumnTypeId::DECIMAL: {
        switch (column[expr]->data_wrapper.getColumnTypeSize()) {
          case sizeof(int32_t): {
            total_bytes += sizeof(int32_t);
            data_type[expr] = DECIMAL32;
            break;
          }
          case sizeof(int64_t): {
            total_bytes += sizeof(int64_t);
            data_type[expr] = DECIMAL64;
            break;
          }
          throw NotImplementedException("Unsupported sirius DECIMAL column type size in `HandleArbitraryConstantExpression`: %zu",
                                        column[expr]->data_wrapper.getColumnTypeSize());
        }
        break;
      } default: {
        throw NotImplementedException("Unsupported sirius column type in `HandleArbitraryConstantExpression`: %d",
                                      static_cast<int>(column[expr]->data_wrapper.type.id()));
      }
    }
  }

  uint8_t* constant_compare = gpuBufferManager->customCudaHostAlloc<uint8_t>(total_bytes);

  uint64_t init_offset = 0;
  for (int expr = 0; expr < num_expr; expr++) {
    col[expr] = column[expr]->data_wrapper.data;
    offset[expr] = column[expr]->data_wrapper.offset;
    bitmask[expr] = column[expr]->data_wrapper.validity_mask;

    if (filter_constant[expr]->constant.IsNull()) {
      int temp = 0xFFFFFFFF; // Use a sentinel value to indicate NULL
      memcpy(constant_compare + init_offset, &temp, sizeof(int));
      constant_offset[expr] = init_offset;
      init_offset += sizeof(int);
      continue;
    }

    switch(column[expr]->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT16: {
        int temp = filter_constant[expr]->constant.GetValue<int16_t>();
        memcpy(constant_compare + init_offset, &temp, sizeof(int16_t));
        constant_offset[expr] = init_offset;
        init_offset += sizeof(int16_t);
        break;
      }
      case GPUColumnTypeId::INT32:
      case GPUColumnTypeId::DATE: {
        int temp = filter_constant[expr]->constant.GetValue<int>();
        memcpy(constant_compare + init_offset, &temp, sizeof(int));
        constant_offset[expr] = init_offset;
        init_offset += sizeof(int);
        break;
      } case GPUColumnTypeId::INT64: {
        int64_t temp = filter_constant[expr]->constant.GetValue<int64_t>();
        memcpy(constant_compare + init_offset, &temp, sizeof(int64_t));
        constant_offset[expr] = init_offset;
        init_offset += sizeof(int64_t);
        break;
      } case GPUColumnTypeId::FLOAT32: {
        float temp = filter_constant[expr]->constant.GetValue<float>();
        memcpy(constant_compare + init_offset, &temp, sizeof(float));
        constant_offset[expr] = init_offset;
        init_offset += sizeof(float);
        break;
      } case GPUColumnTypeId::FLOAT64: {
        double temp = filter_constant[expr]->constant.GetValue<double>();
        memcpy(constant_compare + init_offset, &temp, sizeof(double));
        constant_offset[expr] = init_offset;
        init_offset += sizeof(double);
        break;
      } case GPUColumnTypeId::VARCHAR: {
        std::string lower_string = filter_constant[expr]->constant.ToString();
        memcpy(constant_compare + init_offset, lower_string.data(), lower_string.size());
        constant_offset[expr] = init_offset;
        init_offset += lower_string.size();
        break;
      } case GPUColumnTypeId::DECIMAL: {
        switch (column[expr]->data_wrapper.getColumnTypeSize()) {
          case sizeof(int32_t): {
            // `GetValue()` cannot work since it will convert to double first, same for below
            int32_t temp = filter_constant[expr]->constant.GetValueUnsafe<int32_t>();
            memcpy(constant_compare + init_offset, &temp, sizeof(int32_t));
            constant_offset[expr] = init_offset;
            init_offset += sizeof(int32_t);
            break;
          }
          case sizeof(int64_t): {
            int64_t temp = filter_constant[expr]->constant.GetValueUnsafe<int64_t>();
            memcpy(constant_compare + init_offset, &temp, sizeof(int64_t));
            constant_offset[expr] = init_offset;
            init_offset += sizeof(int64_t);
            break;
          }
          throw NotImplementedException("Unsupported sirius DECIMAL column type size in `HandleArbitraryConstantExpression`: %zu",
                                        column[expr]->data_wrapper.getColumnTypeSize());
        }
        break;
      } default: {
        throw NotImplementedException("Unsupported sirius column type in `HandleArbitraryConstantExpression`: %d",
                                      column[expr]->data_wrapper.type.id());
      }
    }
  }
  constant_offset[num_expr] = init_offset;
  
  uint64_t N = column[0]->column_length;
  tableScanExpression(col, offset, bitmask, constant_compare, constant_offset, data_type, row_ids, count, N, compare_mode, num_expr);
}

class GPUTableScanGlobalSourceState : public GlobalSourceState {
public:
	GPUTableScanGlobalSourceState(ClientContext &context, const GPUPhysicalTableScan &op) {
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

  // The followings are used in `TableScanCoalesceTask`
  void InitForTableScanCoalesceTask(const GPUPhysicalTableScan& op, uint8_t** mask_ptr_p) {
    offset_info_aligned.row_offset = 0;
    offset_info_aligned.column_data_offsets.resize(op.column_ids.size(), 0);
    offset_info_unaligned.row_offset = op.num_rows;
    offset_info_unaligned.column_data_offsets.assign(op.column_size, op.column_size + op.column_ids.size());
    mask_ptr = mask_ptr_p;
    unaligned_mask_byte_pos = op.num_rows / 8;
    unaligned_mask_in_byte_pos = op.num_rows % 8;
    if (unaligned_mask_in_byte_pos == 0) {
      --unaligned_mask_byte_pos;
      unaligned_mask_in_byte_pos += 8;
    }
  }

  void NextChunkOffsetsAligned(uint64_t chunk_rows, const vector<uint64_t>& chunk_column_sizes,
                               uint64_t* out_row_offset, vector<uint64_t>& out_column_data_offsets) {
    auto& offset_info = offset_info_aligned;
    std::unique_lock lock(offset_info.mutex);
    *out_row_offset = offset_info.row_offset;
    out_column_data_offsets = offset_info.column_data_offsets;
    offset_info.row_offset += chunk_rows;
    for (size_t i = 0; i < chunk_column_sizes.size(); ++i) {
      offset_info.column_data_offsets[i] += chunk_column_sizes[i];
    }
  }

  inline void AssignBits(uint8_t from, int from_pos, uint8_t* to, int to_pos, int n) {
    uint8_t mask = (1u << n) - 1;
    uint8_t bits = (from >> from_pos) & mask;
    *to &= ~(mask << to_pos);
    *to |= (bits << to_pos);
  }

  void NextChunkOffsetsUnaligned(uint64_t chunk_rows, const vector<uint64_t>& chunk_column_sizes,
                                 uint64_t* out_row_offset, vector<uint64_t>& out_column_data_offsets,
                                 const vector<uint8_t>& chunk_unaligned_mask_bytes) {
    auto& offset_info = offset_info_unaligned;
    std::unique_lock lock(offset_info.mutex);
    offset_info.row_offset -= chunk_rows;
    for (size_t i = 0; i < chunk_column_sizes.size(); ++i) {
      offset_info.column_data_offsets[i] -= chunk_column_sizes[i];
    }
    *out_row_offset = offset_info.row_offset;
    out_column_data_offsets = offset_info.column_data_offsets;

    // Need to compact the unaligned mask byte per column
    if (unaligned_mask_in_byte_pos >= chunk_rows) {
      for (size_t i = 0; i < chunk_unaligned_mask_bytes.size(); ++i) {
        if (mask_ptr[i] == nullptr) {
          continue;
        }
        AssignBits(chunk_unaligned_mask_bytes[i], 0, mask_ptr[i] + unaligned_mask_byte_pos,
                   unaligned_mask_in_byte_pos - chunk_rows, chunk_rows);
      }
      unaligned_mask_in_byte_pos -= chunk_rows;
      if (unaligned_mask_in_byte_pos == 0) {
        --unaligned_mask_byte_pos;
        unaligned_mask_in_byte_pos += 8;
      }
    } else {
      int n1 = unaligned_mask_in_byte_pos, n2 = chunk_rows - n1;
      for (size_t i = 0; i < chunk_unaligned_mask_bytes.size(); ++i) {
        if (mask_ptr[i] == nullptr) {
          continue;
        }
        AssignBits(chunk_unaligned_mask_bytes[i], n2, mask_ptr[i] + unaligned_mask_byte_pos, 0, n1);
        AssignBits(chunk_unaligned_mask_bytes[i], 0, mask_ptr[i] + unaligned_mask_byte_pos - 1, 8 - n2, n2);
      }
      --unaligned_mask_byte_pos;
      unaligned_mask_in_byte_pos = 8 - n2;
    }
  }

  // For both rows which are null mask aligned and unaligned
  struct {
    std::mutex mutex;
    uint64_t row_offset;
    vector<uint64_t> column_data_offsets;
  } offset_info_aligned, offset_info_unaligned;

  // For compacting null mask bytes of unaligned portion per column. We write starting from last bit
  // since the unaligned portion is written from the end.
  uint8_t** mask_ptr;
  uint64_t unaligned_mask_byte_pos;
  int unaligned_mask_in_byte_pos;
};

class GPUTableScanLocalSourceState : public LocalSourceState {
public:
	GPUTableScanLocalSourceState(ExecutionContext &context, GPUTableScanGlobalSourceState &gstate,
	                          const GPUPhysicalTableScan &op) {
		if (op.function.init_local) {
			TableFunctionInitInput input(op.bind_data.get(), op.column_ids, op.scanned_ids,
			                             gstate.GetTableFilters(op), op.extra_info.sample_options);
			local_state = op.function.init_local(context, input, gstate.global_state.get());
		}
    num_rows = 0;
    column_size.resize(op.column_ids.size(), 0);
	}

	unique_ptr<LocalTableFunctionState> local_state;

  // Used in `TableScanGetSizeTask`
  uint64_t num_rows;
  vector<uint64_t> column_size;
};

unique_ptr<LocalSourceState> GPUPhysicalTableScan::GetLocalSourceState(ExecutionContext &context,
                                                                    GlobalSourceState &gstate) const {
	return make_uniq<GPUTableScanLocalSourceState>(context, gstate.Cast<GPUTableScanGlobalSourceState>(), *this);
}

unique_ptr<GlobalSourceState> GPUPhysicalTableScan::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<GPUTableScanGlobalSourceState>(context, *this);
}

class TableScanGetSizeTask : public BaseExecutorTask {
public:
	TableScanGetSizeTask(TaskExecutor &executor, int task_id_p, TableFunction& function_p, ExecutionContext& context_p,
                       GPUPhysicalTableScan& op_p, GlobalSourceState* g_state_p, LocalSourceState* l_state_p)
	    : BaseExecutorTask(executor), task_id(task_id_p), function(function_p), context(context_p),
        op(op_p), g_state(g_state_p), l_state(l_state_p) {}

	void ExecuteTask() override {
    auto num_cols = op.column_ids.size() - op.gen_row_id_column;
    auto &g_state_scan = g_state->Cast<GPUTableScanGlobalSourceState>();
    auto &l_state_scan = l_state->Cast<GPUTableScanLocalSourceState>();
    TableFunctionInput data(op.bind_data.get(), l_state_scan.local_state.get(), g_state_scan.global_state.get());
    auto chunk = make_uniq<DataChunk>();
    chunk->Initialize(Allocator::Get(context.client), op.scanned_types);

    while (true) {
      // Get next chunk
      function.function(context.client, data, *chunk);
      if (chunk->size() == 0) {
        break;
      }
      // Get column size and mask size of each column in the chunk
      l_state_scan.num_rows += chunk->size();      
      for (int col = 0; col < num_cols; col++) {
        auto& vec = chunk->data[col];
        if (vec.GetType().id() == LogicalTypeId::VARCHAR) {
          vec.Flatten(chunk->size());
          auto duckdb_strings = reinterpret_cast<string_t*>(vec.GetData());
          auto &validity = FlatVector::Validity(vec);
          for (int row = 0; row < chunk->size(); row++) {
            if (validity.RowIsValid(row)) {
              l_state_scan.column_size[col] += duckdb_strings[row].GetSize();
            }
          }
        } else {
          l_state_scan.column_size[col] += GetChunkDataByteSize(op.scanned_types[col], chunk->size());
        }
      }

      // Clear the chunk
      chunk->Reset();
    }
	}

private:
  int task_id;
  TableFunction& function;
  ExecutionContext& context;
  GPUPhysicalTableScan& op;
  GlobalSourceState* g_state;
  LocalSourceState* l_state;
};

class TableScanCoalesceTask : public BaseExecutorTask {
public:
	TableScanCoalesceTask(TaskExecutor &executor, int task_id_p, TableFunction& function_p, ExecutionContext& context_p,
                        GPUPhysicalTableScan& op_p, GlobalSourceState* g_state_p, LocalSourceState* l_state_p,
                        uint8_t** data_ptr_p, uint8_t** mask_ptr_p, uint64_t** offset_ptr_p,
                        int64_t* duckdb_storage_row_ids_ptr_p)
	    : BaseExecutorTask(executor), task_id(task_id_p), function(function_p), context(context_p),
        op(op_p), g_state(g_state_p), l_state(l_state_p), data_ptr(data_ptr_p), mask_ptr(mask_ptr_p),
        offset_ptr(offset_ptr_p), duckdb_storage_row_ids_ptr(duckdb_storage_row_ids_ptr_p) {}

	void ExecuteTask() override {
    auto &g_state_scan = g_state->Cast<GPUTableScanGlobalSourceState>();
    auto &l_state_scan = l_state->Cast<GPUTableScanLocalSourceState>();
    TableFunctionInput data(op.bind_data.get(), l_state_scan.local_state.get(), g_state_scan.global_state.get());
    auto chunk = make_uniq<DataChunk>();
    chunk->Initialize(Allocator::Get(context.client), op.scanned_types);
    uint64_t row_offset_aligned, row_offset_unaligned;
    vector<uint64_t> column_data_offsets_aligned, column_data_offsets_unaligned;
    vector<uint64_t> chunk_column_sizes_aligned, chunk_column_sizes_unaligned;
    vector<uint8_t> unaligned_mask_bytes;
    unaligned_mask_bytes.resize(op.orig_column_ids.size() - op.gen_row_id_column);
    while (true) {
      // Get next chunk
      function.function(context.client, data, *chunk);
      if (chunk->size() == 0) {
        break;
      }
      // Compute chunk column size and get write offset, here we treat null mask aligned rows and
      // unaligned rows separately.
      uint64_t num_rows_unaligned = chunk->size() % 8;
      uint64_t num_rows_aligned = chunk->size() - num_rows_unaligned;
      chunk_column_sizes_aligned.assign(op.column_ids.size(), 0);
      chunk_column_sizes_unaligned.assign(op.column_ids.size(), 0);
      for (int col = 0; col < op.orig_column_ids.size() - op.gen_row_id_column; col++) {
        if (!op.already_cached[col]) {
          auto& vec = chunk->data[col];
          vec.Flatten(chunk->size());
          if (vec.GetType().id() == LogicalTypeId::VARCHAR) {
            auto duckdb_strings = reinterpret_cast<string_t*>(vec.GetData());
            auto &validity = FlatVector::Validity(vec);
            for (int row = 0; row < num_rows_aligned; row++) {
              if (validity.RowIsValid(row)) {
                chunk_column_sizes_aligned[col] += duckdb_strings[row].GetSize();
              }
            }
            for (int row = num_rows_aligned; row < chunk->size(); row++) {
              if (validity.RowIsValid(row)) {
                chunk_column_sizes_unaligned[col] += duckdb_strings[row].GetSize();
              }
            }
          } else {
            chunk_column_sizes_aligned[col] = GetChunkDataByteSize(vec.GetType(), num_rows_aligned);
            chunk_column_sizes_unaligned[col] = GetChunkDataByteSize(vec.GetType(), num_rows_unaligned);
          }
        }
      }
      if (num_rows_aligned > 0) {
        g_state_scan.NextChunkOffsetsAligned(num_rows_aligned, chunk_column_sizes_aligned,
                                             &row_offset_aligned, column_data_offsets_aligned);
      }
      if (num_rows_unaligned > 0) {
        for (int col = 0; col < op.orig_column_ids.size() - op.gen_row_id_column; col++) {
          if (!op.already_cached[col]) {
            auto &validity = FlatVector::Validity(chunk->data[col]);
            unaligned_mask_bytes[col] = (validity.GetData() == nullptr) ? 0xff
              : reinterpret_cast<uint8_t*>(validity.GetData())[num_rows_aligned / 8];
          }
        }
        g_state_scan.NextChunkOffsetsUnaligned(num_rows_unaligned, chunk_column_sizes_unaligned,
                                               &row_offset_unaligned, column_data_offsets_unaligned,
                                               unaligned_mask_bytes);
      }
      // Copy data to pinned cpu memory. We write rows which are null mask aligned from the beginning of the pinned
      // memory buffer, and write rows which are null mask unaligned from the end of the pinned memory buffer.
      for (int col = 0; col < op.orig_column_ids.size() - op.gen_row_id_column; col++) {
        if (!op.already_cached[col]) {
          auto& vec = chunk->data[col];
          // For data
          auto &validity = FlatVector::Validity(vec);
          if (vec.GetType().id() == LogicalTypeId::VARCHAR) {
            auto duckdb_strings = reinterpret_cast<string_t*>(vec.GetData());
            // For rows with null mask aligned
            uint64_t chunk_data_offset = 0;
            for (int row = 0; row < num_rows_aligned; row++) {
              if (validity.RowIsValid(row)) {
                uint64_t len = duckdb_strings[row].GetSize();
                offset_ptr[col][row_offset_aligned + row + 1] = len;
                uint64_t write_offset = column_data_offsets_aligned[col] + chunk_data_offset;
                memcpy(data_ptr[col] + write_offset, duckdb_strings[row].GetData(), len);
                chunk_data_offset += len;
              } else {
                offset_ptr[col][row_offset_aligned + row + 1] = 0;
              }
            }
            // For rows with null mask unaligned
            chunk_data_offset = 0;
            for (int row = num_rows_aligned; row < chunk->size(); row++) {
              if (validity.RowIsValid(row)) {
                uint64_t len = duckdb_strings[row].GetSize();
                offset_ptr[col][row_offset_unaligned + (row - num_rows_aligned) + 1] = len;
                uint64_t write_offset = column_data_offsets_unaligned[col] + chunk_data_offset;
                memcpy(data_ptr[col] + write_offset, duckdb_strings[row].GetData(), len);
                chunk_data_offset += len;
              } else {
                offset_ptr[col][row_offset_unaligned + (row - num_rows_aligned) + 1] = 0;
              }
            }
          } else {
            auto typeIdSize = GetTypeIdSize(vec.GetType().InternalType());
            // For rows with null mask aligned
            if (num_rows_aligned > 0) {
              uint64_t write_offset = row_offset_aligned * typeIdSize;
              memcpy(data_ptr[col] + write_offset, vec.GetData(), chunk_column_sizes_aligned[col]);
            }
            // For rows with null mask unaligned
            if (num_rows_unaligned > 0) {
              uint64_t write_offset = row_offset_unaligned * typeIdSize;
              uint64_t read_offset = num_rows_aligned * typeIdSize;
              memcpy(data_ptr[col] + write_offset, vec.GetData() + read_offset, chunk_column_sizes_unaligned[col]);
            }
          }
          // Write aligned null mask bytes. Unaligned null mask bytes are handled in `NextChunkOffsetsUnaligned()`.
          if (num_rows_aligned > 0) {
            if (validity.GetData() == nullptr) {
              memset(mask_ptr[col] + row_offset_aligned / 8, 0xff, num_rows_aligned / 8);
            } else {
              memcpy(mask_ptr[col] + row_offset_aligned / 8, validity.GetData(), num_rows_aligned / 8);
            }
          }
        }
      }
      // Also copy row id if required
      if (duckdb_storage_row_ids_ptr != nullptr) {
        int col = op.scanned_ids.size() - 1;
        auto& vec = chunk->data[col];
        vec.Flatten(chunk->size());
        // For rows with null mask aligned
        if (num_rows_aligned > 0) {
          memcpy(duckdb_storage_row_ids_ptr + row_offset_aligned, vec.GetData(), num_rows_aligned * sizeof(int64_t));
        }
        // For rows with null mask unaligned
        if (num_rows_unaligned > 0) {
          memcpy(duckdb_storage_row_ids_ptr + row_offset_unaligned, vec.GetData() + num_rows_aligned * sizeof(int64_t),
                 num_rows_unaligned * sizeof(int64_t));
        }
      }
      // Clear the chunk
      chunk->Reset();
    }
	}

private:
  int task_id;
  TableFunction& function;
  ExecutionContext& context;
  GPUPhysicalTableScan& op;
  GlobalSourceState* g_state;
  LocalSourceState* l_state;
  uint8_t** data_ptr;
  uint8_t** mask_ptr;
  uint64_t** offset_ptr;
  int64_t* duckdb_storage_row_ids_ptr;
};

SourceResultType
GPUPhysicalTableScan::GetDataDuckDBOpt(ExecutionContext &exec_context) {
  // Check if columns are all cached
  SIRIUS_LOG_DEBUG("GPUPhysicalTableScan GetDataDuckDBOpt invoked");
  D_ASSERT(!column_ids.empty());
  auto gpuBufferManager = &(GPUBufferManager::GetInstance());

  SIRIUS_LOG_DEBUG("Reading data from duckdb storage");

  TableFunctionToStringInput input(function, bind_data.get());
  auto to_string_result = function.to_string(input);
  string table_name;
  for (const auto &it : to_string_result) {
    if (it.first.compare("Table") == 0) {
      table_name = it.second;
      break;
    }
  }

  // Get cached column info
  for(int i = 0; i < column_ids.size(); i++) { 
    SIRIUS_LOG_DEBUG("Scan Idx {} has column id of {}", i, column_ids[i].GetPrimaryIndex());
  } 

  auto num_columns = column_ids.size() - gen_row_id_column;
  bool all_cached = num_columns > 0;
  SIRIUS_LOG_DEBUG("Checking if all of the {}/{} columns of table {} are cached", num_columns, column_ids.size(), table_name);

  for (int col = 0; col < num_columns; col++) {
    already_cached[col] = gpuBufferManager->checkIfColumnCached(table_name, names[column_ids[col].GetPrimaryIndex()]);
    if (!already_cached[col]) {
      all_cached = false;
      uncached_scan_column_ids.push_back(column_ids[col].GetPrimaryIndex());
    } 
  }

  if (all_cached) {
    SIRIUS_LOG_DEBUG("Early terminating from GetDataDuckdbOpt because all cols are cached");
    return SourceResultType::FINISHED;
  }

  if (function.function) {
    // Read data and converting data from duckdb
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the first scan to get column size and mask size. Here we perform two scans streamingly
    // since storing all scanned chunks together in memory is extremely slow.
    int num_threads = 1; // TaskScheduler::GetScheduler(exec_context.client).NumberOfThreads();
    TaskExecutor executor(exec_context.client);
    auto g_state = GetGlobalSourceState(exec_context.client);
    vector<unique_ptr<LocalSourceState>> l_states;
    l_states.resize(num_threads);
    SIRIUS_LOG_DEBUG("GetDataDuckdbOpt: Starting Scheduling of Tasks with l_states of size {} and uncached of size {}", l_states.size(), uncached_scan_column_ids.size());
    for (int i = 0; i < num_threads; ++i) {
      l_states[i] = GetLocalSourceState(exec_context, *g_state);
      auto task = make_uniq<TableScanGetSizeTask>(executor, i, function, exec_context, *this,
                                                  g_state.get(), l_states[i].get());
      executor.ScheduleTask(std::move(task));
    }
    SIRIUS_LOG_DEBUG("GetDataDuckdbOpt: Finished Scheduling of Tasks");

    executor.WorkOnTasks();
    SIRIUS_LOG_DEBUG("GetDataDuckdbOpt: Tasks finished executing");

    for (int i = 0; i < num_threads; ++i) {
      auto &l_state_scan = l_states[i]->Cast<GPUTableScanLocalSourceState>();
      num_rows += l_state_scan.num_rows;
      for (int col = 0; col < column_ids.size() - gen_row_id_column; col++) {
        column_size[col] += l_state_scan.column_size[col];
      }
    }
    SIRIUS_LOG_DEBUG("GetDataDuckdbOpt: Gathered Results From Tasks with num rows of {}", num_rows);

    std::fill(mask_size, mask_size + num_columns, getMaskBytesSize(num_rows));
    uint64_t total_size = 0;
    for (int col = 0; col < column_ids.size() - gen_row_id_column; col++) {
      if (!already_cached[col]) {
        total_size += column_size[col];
        total_size += mask_size[col];
        if (scanned_types[col].id() == LogicalTypeId::VARCHAR) {
          // Add size of offsets
          total_size += sizeof(uint64_t) * (num_rows + 1);
        }
      }
    }

    // Create table/columns in gpu and check OOM
    SIRIUS_LOG_DEBUG("GetDataDuckDBOpt creating {} columns for table {} with total cols of {}", num_columns, table_name, column_ids.size());

    auto &catalog_table = Catalog::GetCatalog(exec_context.client, INVALID_CATALOG);
    if (gpuBufferManager->gpuCachingPointer[0] + gpuBufferManager->cpuCachingPointer[0] + total_size >=
        gpuBufferManager->cache_size_per_gpu) {
      if (total_size > gpuBufferManager->cache_size_per_gpu) {
        throw InvalidInputException("Total size of columns (%lu) to be cached is greater than the cache size (%lu)",
                                    total_size, gpuBufferManager->cache_size_per_gpu);
      }
      gpuBufferManager->ResetCache();
      uncached_scan_column_ids.clear();
      for (int col = 0; col < num_columns; col++) {
        already_cached[col] = false;
        uncached_scan_column_ids.push_back(column_ids[col].GetPrimaryIndex());
        gpuBufferManager->createTableAndColumnInGPU(catalog_table, exec_context.client, table_name, names[column_ids[col].GetPrimaryIndex()]);
      }
    } else {
      for (int col = 0; col < num_columns; col++) {
        if (!already_cached[col]) {
          gpuBufferManager->createTableAndColumnInGPU(catalog_table, exec_context.client, table_name, names[column_ids[col].GetPrimaryIndex()]);
        } 
      }
    }

    SIRIUS_LOG_DEBUG("GetDataDuckDBOpt finished caching all necessary columns");

    // Perform the second scan to copy data to gpu
    if(num_columns > 0) ScanDataDuckDBOpt(exec_context, gpuBufferManager, table_name);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_DEBUG("Reading data and converting data from duckdb time: {:.2f} ms", duration.count()/1000.0);
    return SourceResultType::FINISHED;
  } else {
    throw NotImplementedException("Table in-out function not supported");
  }

  SIRIUS_LOG_DEBUG("GetDataDuckDBOpt updated num rows to {}", num_rows);
}

void GPUPhysicalTableScan::ScanDataDuckDBOpt(
  ExecutionContext &exec_context, GPUBufferManager* gpuBufferManager, string table_name) {

  SIRIUS_LOG_DEBUG("GPUPhysicalTableScan ScanDataDuckDBOpt invoked");  
  if (function.function) {
    // Perform the second scan to coalesce duckdb chunk data to continuous pinned memory data,
    // `offset_ptr` represents string length before copied to gpu, after which we do prefix sum
    uint8_t** data_ptr = gpuBufferManager->customCudaHostAlloc<uint8_t*>(scanned_types.size());
    uint8_t** mask_ptr = gpuBufferManager->customCudaHostAlloc<uint8_t*>(scanned_types.size());
    uint64_t** offset_ptr = gpuBufferManager->customCudaHostAlloc<uint64_t*>(scanned_types.size());
    std::fill(data_ptr, data_ptr + scanned_types.size(), nullptr);
    std::fill(mask_ptr, mask_ptr + scanned_types.size(), nullptr);
    std::fill(offset_ptr, offset_ptr + scanned_types.size(), nullptr);
    for (int col = 0; col < column_ids.size() - gen_row_id_column; col++) {
      if (!already_cached[col]) {
        data_ptr[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(column_size[col]);
        mask_ptr[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(mask_size[col]);
        if (scanned_types[col].id() == LogicalTypeId::VARCHAR) {
          offset_ptr[col] = gpuBufferManager->customCudaHostAlloc<uint64_t>(num_rows + 1);
          offset_ptr[col][0] = 0;
        }
      }
    }

    // Get table in gpu buffer manager
    auto up_table_name = table_name;
    transform(up_table_name.begin(), up_table_name.end(), up_table_name.begin(), ::toupper);
    auto& table = gpuBufferManager->tables[up_table_name];

    // Since we are doing parallel scan in optimized table scan and store it into gpu cache, we need to keep
    // row orders of different columns consistent (e.g., one scan caches col_0 and another scan caches col_1).
    // Now we achieve this by aligning the row order of scanned columns according to duckdb storage row ids.
    bool scan_duckdb_storage_row_ids = num_rows > 1 && uncached_scan_column_ids.size() < table->column_count;
    int64_t* duckdb_storage_row_ids_ptr = nullptr;
    // Need to save the original column info, since we need to modify them to let duckdb produce row_id column,
    // but after the scan finishes we still need to use the original column info.
    orig_column_ids = column_ids;
    orig_scanned_ids = scanned_ids;
    orig_scanned_types = scanned_types;
    if (scan_duckdb_storage_row_ids) {
      duckdb_storage_row_ids_ptr = gpuBufferManager->customCudaHostAlloc<int64_t>(num_rows);
      // Need to modify column info used during duckdb scan to let it produce row_id column
      if (column_ids.back().GetPrimaryIndex() != COLUMN_IDENTIFIER_ROW_ID) {
        column_ids.push_back(ColumnIndex(COLUMN_IDENTIFIER_ROW_ID));
      }
      scanned_ids.push_back(scanned_ids.size());
      scanned_types.push_back(LogicalType::BIGINT);
    }

    int num_threads = TaskScheduler::GetScheduler(exec_context.client).NumberOfThreads();
    TaskExecutor executor(exec_context.client);
    auto g_state = GetGlobalSourceState(exec_context.client);
    g_state->Cast<GPUTableScanGlobalSourceState>().InitForTableScanCoalesceTask(*this, mask_ptr);
    vector<unique_ptr<LocalSourceState>> l_states;
    l_states.resize(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      l_states[i] = GetLocalSourceState(exec_context, *g_state);
      auto task = make_uniq<TableScanCoalesceTask>(executor, i, function, exec_context, *this,
                                                   g_state.get(), l_states[i].get(),
                                                   data_ptr, mask_ptr, offset_ptr,
                                                   duckdb_storage_row_ids_ptr);
      executor.ScheduleTask(std::move(task));
    }
    executor.WorkOnTasks();

    // Restore the modified column info
    if (scan_duckdb_storage_row_ids) {
      column_ids = orig_column_ids;
      scanned_ids = orig_scanned_ids;
      scanned_types = orig_scanned_types;
    }
  
    // Copy data from pinned cpu memory to gpu
    uint8_t** d_data_ptr = gpuBufferManager->customCudaHostAlloc<uint8_t*>(scanned_types.size());
    uint8_t** d_mask_ptr = gpuBufferManager->customCudaHostAlloc<uint8_t*>(scanned_types.size());
    uint64_t** d_offset_ptr = gpuBufferManager->customCudaHostAlloc<uint64_t*>(scanned_types.size());
    std::fill(d_data_ptr, d_data_ptr + scanned_types.size(), nullptr);
    std::fill(d_mask_ptr, d_mask_ptr + scanned_types.size(), nullptr);
    std::fill(d_offset_ptr, d_offset_ptr + scanned_types.size(), nullptr);
    for (int col = 0; col < scanned_types.size(); col++) {
      if (!already_cached[col]) {
        d_data_ptr[col] = gpuBufferManager->customCudaMalloc<uint8_t>(column_size[col], 0, 1);
        d_mask_ptr[col] = gpuBufferManager->customCudaMalloc<uint8_t>(mask_size[col], 0, 1);
        if (scanned_types[col].id() == LogicalTypeId::VARCHAR) {
          d_offset_ptr[col] = gpuBufferManager->customCudaMalloc<uint64_t>(num_rows + 1, 0, 1);
        }
      }
    }

    uint64_t num_cuda_memcpy = 0;
    for (int col = 0; col < column_ids.size() - gen_row_id_column; col++) {
      if (!already_cached[col]) {
        // For data
        uint64_t write_offset = 0;
        while (write_offset < column_size[col]) {
          uint64_t write_len = std::min(Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE, column_size[col] - write_offset);
          cudaMemcpyAsync(d_data_ptr[col] + write_offset, data_ptr[col] + write_offset,
                          write_len, cudaMemcpyHostToDevice,
                          cuda_streams[num_cuda_memcpy++ % Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS]);
          write_offset += write_len;
        }
        // For mask
        write_offset = 0;
        while (write_offset < mask_size[col]) {
          uint64_t write_len = std::min(Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE, mask_size[col] - write_offset);
          cudaMemcpyAsync(d_mask_ptr[col] + write_offset, mask_ptr[col] + write_offset,
                          write_len, cudaMemcpyHostToDevice,
                          cuda_streams[num_cuda_memcpy++ % Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS]);
          write_offset += write_len;
        }
        if (scanned_types[col].id() == LogicalTypeId::VARCHAR) {
          // For offsets
          write_offset = 0;
          while (write_offset < sizeof(uint64_t) * (num_rows + 1)) {
            uint64_t write_len = std::min(Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE,
                                          sizeof(uint64_t) * (num_rows + 1) - write_offset);
            cudaMemcpyAsync(reinterpret_cast<uint8_t*>(d_offset_ptr[col]) + write_offset,
                            reinterpret_cast<uint8_t*>(offset_ptr[col]) + write_offset,
                            write_len, cudaMemcpyHostToDevice,
                            cuda_streams[num_cuda_memcpy++ % Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS]);
            write_offset += write_len;
          }
        }
      }
    }
    // For duckdb storage row ids
    int64_t* d_duckdb_storage_row_ids_ptr = nullptr;
    if (scan_duckdb_storage_row_ids) {
      d_duckdb_storage_row_ids_ptr = gpuBufferManager->customCudaMalloc<int64_t>(num_rows, 0, 0);
      uint64_t write_offset = 0;
      while (write_offset < sizeof(int64_t) * num_rows) {
        uint64_t write_len = std::min(Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE,
                                      sizeof(int64_t) * num_rows - write_offset);
        cudaMemcpyAsync(reinterpret_cast<uint8_t*>(d_duckdb_storage_row_ids_ptr) + write_offset,
                        reinterpret_cast<uint8_t*>(duckdb_storage_row_ids_ptr) + write_offset,
                        write_len, cudaMemcpyHostToDevice,
                        cuda_streams[num_cuda_memcpy++ % Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS]);
        write_offset += write_len;
      }
    }
    cudaDeviceSynchronize();

    // Perform prefix sum on gpu to form string offsets
    int num_prefix_sum = 0;
    for (int col = 0; col < column_ids.size() - gen_row_id_column; col++) {
      if (!already_cached[col]) {
        if (scanned_types[col].id() == LogicalTypeId::VARCHAR) {
          callCubPrefixSum(d_offset_ptr[col], d_offset_ptr[col], num_rows + 1, true,
                           cuda_streams[num_prefix_sum++ % Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS],
                           [&](size_t size) {
            return gpuBufferManager->customCudaMalloc<uint8_t>(size, 0, 0);
          });
        }
      }
    }
    cudaDeviceSynchronize();

    for (int col = 0; col < column_ids.size() - gen_row_id_column; col++) {
      if (!already_cached[col]) {
        auto up_column_name = names[column_ids[col].GetPrimaryIndex()];
        transform(up_column_name.begin(), up_column_name.end(), up_column_name.begin(), ::toupper);
        auto column_it = find(table->column_names.begin(), table->column_names.end(), up_column_name);
        if (column_it == table->column_names.end()) {
          throw InvalidInputException("Column not found: %s.%s", up_table_name, up_column_name);
        }
        int column_idx = column_it - table->column_names.begin();
        GPUColumnType column_type = convertLogicalTypeToColumnType(scanned_types[col]);
        table->columns[column_idx]->column_length = num_rows;
        cudf::bitmask_type* validity_mask = reinterpret_cast<cudf::bitmask_type*>(d_mask_ptr[col]);
        if (scanned_types[col] == LogicalType::VARCHAR) {
          table->columns[column_idx]->data_wrapper = DataWrapper(column_type, d_data_ptr[col], d_offset_ptr[col],
                                                                 num_rows, column_size[col], true, validity_mask);
        } else {
          table->columns[column_idx]->data_wrapper = DataWrapper(column_type, d_data_ptr[col], num_rows, validity_mask);
        }
        SIRIUS_LOG_DEBUG("Column {}.{} cached in GPU at index {}", up_table_name, up_column_name, column_idx);
      }
    }

    // Align the row order of scanned columns according to duckdb storage row ids if required
    if (scan_duckdb_storage_row_ids) {
      // Get output indices of reordering based on duckdb storage row ids
      uint64_t* reorder_row_ids_out_indices = gpuBufferManager->customCudaMalloc<uint64_t>(num_rows, 0, 0);
      reorderRowIds(d_duckdb_storage_row_ids_ptr, reorder_row_ids_out_indices, num_rows);
      // Reorder each scanned column
      for (int col: uncached_scan_column_ids) {
        // Materialize column
        table->columns[col]->row_ids = reorder_row_ids_out_indices;
        table->columns[col]->row_id_count = num_rows;
        auto reordered_column = HandleMaterializeExpression(table->columns[col], gpuBufferManager);
        // Copy materialized column (in processing region) to cached column (caching region)
        num_cuda_memcpy = 0;
        uint64_t write_offset = 0;
        while (write_offset < table->columns[col]->data_wrapper.num_bytes) {
          uint64_t write_len = std::min(Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE,
                                        table->columns[col]->data_wrapper.num_bytes - write_offset);
          cudaMemcpyAsync(table->columns[col]->data_wrapper.data + write_offset,
                          reordered_column->data_wrapper.data + write_offset,
                          write_len, cudaMemcpyDeviceToDevice,
                          cuda_streams[num_cuda_memcpy++ % Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS]);
          write_offset += write_len;
        }
        if (table->columns[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
          write_offset = 0;
          while (write_offset < sizeof(uint64_t) * (num_rows + 1)) {
            uint64_t write_len = std::min(Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE,
                                          sizeof(uint64_t) * (num_rows + 1) - write_offset);
            cudaMemcpyAsync(reinterpret_cast<uint8_t*>(table->columns[col]->data_wrapper.offset) + write_offset,
                            reinterpret_cast<uint8_t*>(reordered_column->data_wrapper.offset) + write_offset,
                            write_len, cudaMemcpyDeviceToDevice,
                            cuda_streams[num_cuda_memcpy++ % Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS]);
            write_offset += write_len;
          }
        }
        if (table->columns[col]->data_wrapper.validity_mask != nullptr) {
          write_offset = 0;
          while (write_offset < table->columns[col]->data_wrapper.mask_bytes) {
            uint64_t write_len = std::min(Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE,
                                          table->columns[col]->data_wrapper.mask_bytes - write_offset);
            cudaMemcpyAsync(reinterpret_cast<uint8_t*>(table->columns[col]->data_wrapper.validity_mask) + write_offset,
                            reinterpret_cast<uint8_t*>(reordered_column->data_wrapper.validity_mask) + write_offset,
                            write_len, cudaMemcpyDeviceToDevice,
                            cuda_streams[num_cuda_memcpy++ % Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS]);
            write_offset += write_len;
          }
        }
        cudaDeviceSynchronize();
        // Reset column's `row_ids` and `row_id_count` and free data of `reordered_column` in processing region
        table->columns[col]->row_ids = nullptr;
        table->columns[col]->row_id_count = 0;
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(reordered_column->data_wrapper.data), 0);
        if (table->columns[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
          gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(reordered_column->data_wrapper.offset), 0);
        }
      }
      // Free `reorder_row_ids_out_indices` and `d_duckdb_storage_row_ids_ptr`
      gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(reorder_row_ids_out_indices), 0);
      gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_duckdb_storage_row_ids_ptr), 0);
    }
  } else {
    throw NotImplementedException("Table in-out function not supported");
  }
}

SourceResultType
GPUPhysicalTableScan::GetDataDuckDB(ExecutionContext &exec_context) {
  SIRIUS_LOG_DEBUG("GPUPhysicalTableScan GetDataDuckDB invoked");

  // Use optimized scan if required
  if (Config::USE_OPT_TABLE_SCAN) {
    return GetDataDuckDBOpt(exec_context);
  }

    D_ASSERT(!column_ids.empty());
    auto gpuBufferManager = &(GPUBufferManager::GetInstance());

    SIRIUS_LOG_DEBUG("Reading data from duckdb storage");

    TableFunctionToStringInput input(function, bind_data.get());
    auto to_string_result = function.to_string(input);
    string table_name;
    for (const auto &it : to_string_result) {
      if (it.first.compare("Table") == 0) {
        table_name = it.second;
        break;
      }
    }

    shared_ptr<GPUIntermediateRelation> table;
    auto &catalog_table = Catalog::GetCatalog(exec_context.client, INVALID_CATALOG);

    bool all_cached = true;
    for (int col = 0; col < column_ids.size() - gen_row_id_column; col++) {
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

    auto &l_state_scan = l_state->Cast<GPUTableScanLocalSourceState>();
    auto &g_state_scan = g_state->Cast<GPUTableScanGlobalSourceState>();

    TableFunctionInput data(bind_data.get(), l_state_scan.local_state.get(), g_state_scan.global_state.get());

    if (function.function) {
      auto start = std::chrono::high_resolution_clock::now();
      bool has_more_output = true;

      do {
        auto chunk = make_uniq<DataChunk>();
        chunk->Initialize(Allocator::Get(exec_context.client), scanned_types);
        function.function(exec_context.client, data, *chunk);
        has_more_output = chunk->size() > 0;
        // get the size of each column in the chunk
        for (int col = 0; col < column_ids.size() - gen_row_id_column; col++) {
            Vector vec = chunk->data[col];
            vec.Flatten(chunk->size());
            if (vec.GetType() == LogicalType::VARCHAR) {
              for (int row = 0; row < chunk->size(); row++) {
                std::string curr_string = vec.GetValue(row).ToString();
                column_size[col] += curr_string.length();
              }
            } else {
              column_size[col] += GetChunkDataByteSize(scanned_types[col], chunk->size());
            }
            ValidityMask validity_mask = FlatVector::Validity(vec);
            uint64_t validity_mask_size = validity_mask.ValidityMaskSize(chunk->size());
            mask_size[col] += validity_mask_size;
        }
        collection->Append(*chunk);
      } while (has_more_output);

      uint64_t total_size = 0;
      for (int col = 0; col < column_ids.size(); col++) {
        //round up to nearest 64 byte to adjust with libcudf valdity mask
        mask_size[col] = 64 * ((mask_size[col] + 63) / 64);
        if (!already_cached[col]) {
          total_size += column_size[col];
          total_size += mask_size[col];
        }
      }

      auto num_columns = column_ids.size() - gen_row_id_column;
      SIRIUS_LOG_DEBUG("GetDataDuckDB creating {} columns for table {} with total cols of {}", num_columns, table_name, column_ids.size());

      if (gpuBufferManager->gpuCachingPointer[0] + gpuBufferManager->cpuCachingPointer[0] + total_size >= gpuBufferManager->cache_size_per_gpu) {
        if (total_size > gpuBufferManager->cache_size_per_gpu) {
          throw InvalidInputException("Total size of columns to be cached is greater than the cache size");
        }
        gpuBufferManager->ResetCache();
        for (int col = 0; col < num_columns; col++) {
          already_cached[col] = false;
          gpuBufferManager->createTableAndColumnInGPU(catalog_table, exec_context.client, table_name, names[column_ids[col].GetPrimaryIndex()]);
        }
      } else {
          for (int col = 0; col < num_columns; col++) {
              if (!already_cached[col]) {
                gpuBufferManager->createTableAndColumnInGPU(catalog_table, exec_context.client, table_name, names[column_ids[col].GetPrimaryIndex()]);
              } 
          }
      }

      ScanDataDuckDB(gpuBufferManager, table_name);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      SIRIUS_LOG_DEBUG("Reading data and converting data from duckdb time: {:.2f} ms", duration.count()/1000.0);
      return SourceResultType::FINISHED;
    } else {
      throw NotImplementedException("Table in-out function not supported");
    }
}

void
GPUPhysicalTableScan::ScanDataDuckDB(GPUBufferManager* gpuBufferManager, string table_name) const{
    SIRIUS_LOG_DEBUG("GPUPhysicalTableScan ScanDataDuckDB invoked");

    if (function.function) {
      bool has_more_output = true;
      // allocate size in gpu buffer manager cpu processing region
      uint8_t** ptr = gpuBufferManager->customCudaHostAlloc<uint8_t*>(scanned_types.size());
      uint8_t** d_ptr = gpuBufferManager->customCudaHostAlloc<uint8_t*>(scanned_types.size());
      uint8_t** tmp_ptr = gpuBufferManager->customCudaHostAlloc<uint8_t*>(scanned_types.size());
      uint8_t** d_mask_ptr = gpuBufferManager->customCudaHostAlloc<uint8_t*>(scanned_types.size());
      uint8_t** mask_ptr = gpuBufferManager->customCudaHostAlloc<uint8_t*>(scanned_types.size());
      uint8_t** tmp_mask_ptr = gpuBufferManager->customCudaHostAlloc<uint8_t*>(scanned_types.size());
      uint64_t** offset_ptr = gpuBufferManager->customCudaHostAlloc<uint64_t*>(scanned_types.size());
      uint64_t** d_offset_ptr = gpuBufferManager->customCudaHostAlloc<uint64_t*>(scanned_types.size());

      for (int col = 0; col < scanned_types.size(); col++) {
        if (!already_cached[col]) {
          ptr[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(column_size[col]);
          d_ptr[col] = gpuBufferManager->customCudaMalloc<uint8_t>(column_size[col], 0, 1);
          mask_ptr[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(mask_size[col]);
          memset(mask_ptr[col], 0, mask_size[col] * sizeof(uint8_t));
          d_mask_ptr[col] = gpuBufferManager->customCudaMalloc<uint8_t>(mask_size[col], 0, 1);
          if (scanned_types[col] == LogicalType::VARCHAR) {
            offset_ptr[col] = gpuBufferManager->customCudaHostAlloc<uint64_t>(collection->Count() + 1);
            d_offset_ptr[col] = gpuBufferManager->customCudaMalloc<uint64_t>(collection->Count() + 1, 0, 1);
            offset_ptr[col][0] = 0;
          }
          tmp_ptr[col] = ptr[col];
          tmp_mask_ptr[col] = mask_ptr[col];
        }
      } 
      bool scan_initialized = false;
      ColumnDataScanState scan_state;
      uint64_t start_idx = 0;

      SIRIUS_LOG_DEBUG("Converting duckdb format to Sirius format (this will take a while)");
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
            Vector vec = result->data[col];
            vec.Flatten(result->size());
            if (vec.GetType() == LogicalType::VARCHAR) {
              for (int row = 0; row < result->size(); row++) {
                std::string curr_string = vec.GetValue(row).ToString();
                memcpy(tmp_ptr[col], curr_string.data(), curr_string.length());
                offset_ptr[col][start_idx + row + 1] = offset_ptr[col][start_idx + row] + curr_string.length();
                tmp_ptr[col] += curr_string.length();
              }
            } else {
              memcpy(tmp_ptr[col], vec.GetData(), GetChunkDataByteSize(scanned_types[col], result->size()));
              tmp_ptr[col] += GetChunkDataByteSize(scanned_types[col], result->size());
            }
            ValidityMask validity_mask = FlatVector::Validity(vec);
            uint64_t validity_mask_size = validity_mask.ValidityMaskSize(result->size());
            //TODO: We currently assume that the validity mask is always stored. There can be an optimization where we don't store the validity mask if all valuesa are valid
            if (validity_mask.GetData() == nullptr) {
              memset(tmp_mask_ptr[col], 0xff, validity_mask_size * sizeof(uint8_t));
            } else {
              memcpy(tmp_mask_ptr[col], reinterpret_cast<uint8_t*>(validity_mask.GetData()), validity_mask_size * sizeof(uint8_t));
            }
            tmp_mask_ptr[col] += validity_mask_size;
          }
        }
        start_idx += result->size();
        has_more_output = result->size() > 0;
      } while (has_more_output);


      for (int col = 0; col < column_ids.size() - gen_row_id_column; col++) {
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
            callCudaMemcpyHostToDevice<uint8_t>(d_mask_ptr[col], mask_ptr[col], mask_size[col], 0);
        }
      }

      for (int col = 0; col < column_ids.size() - gen_row_id_column; col++) {
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
            GPUColumnType column_type = convertLogicalTypeToColumnType(scanned_types[col]);
            gpuBufferManager->tables[up_table_name]->columns[column_idx]->column_length = collection->Count();
            cudf::bitmask_type* validity_mask = reinterpret_cast<cudf::bitmask_type*>(d_mask_ptr[col]);
            if (scanned_types[col] == LogicalType::VARCHAR) {
              gpuBufferManager->tables[up_table_name]->columns[column_idx]->data_wrapper = DataWrapper(column_type, d_ptr[col], d_offset_ptr[col], collection->Count(), column_size[col], true, validity_mask);
            } else {
              gpuBufferManager->tables[up_table_name]->columns[column_idx]->data_wrapper = DataWrapper(column_type, d_ptr[col], collection->Count(), validity_mask);
            }
            SIRIUS_LOG_DEBUG("Column {} cached in GPU at index {}", up_column_name, column_idx);
        }
      }
    } else {
      throw NotImplementedException("Table in-out function not supported");
    }
}

SourceResultType
GPUPhysicalTableScan::GetData(GPUIntermediateRelation &output_relation) const {
  SIRIUS_LOG_DEBUG("GPUPhysicalTableScan GetData invoked");
  auto start = std::chrono::high_resolution_clock::now();
  if (output_relation.columns.size() != GetTypes().size()) throw InvalidInputException("Mismatched column count");

  TableFunctionToStringInput input(function, bind_data.get());
  auto to_string_result = function.to_string(input);
  string table_name;
  for (const auto &it : to_string_result) {
    if (it.first.compare("Table") == 0) {
      table_name = it.second;
      break;
    }
  }

  //Find table name in the buffer manager only if we need to load actual table columns
  auto num_cols = column_ids.size() - gen_row_id_column;

  auto gpuBufferManager = &(GPUBufferManager::GetInstance());
  transform(table_name.begin(), table_name.end(), table_name.begin(), ::toupper);
  
  shared_ptr<GPUIntermediateRelation> table;
  if(num_cols > 0) { 
    SIRIUS_LOG_DEBUG("Table Scanning {}", table_name);
    auto it = gpuBufferManager->tables.find(table_name);
    if (it != gpuBufferManager->tables.end()) {
        // Key found, print the value
        table = it->second;
        for (int i = 0; i < table->column_names.size(); i++) {
            SIRIUS_LOG_DEBUG("Cached Column name: {}", table->column_names[i]);
        }
        for (int col = 0; col < num_cols; col++) {
            auto column_to_find = names[column_ids[col].GetPrimaryIndex()];
            transform(column_to_find.begin(), column_to_find.end(), column_to_find.begin(), ::toupper);
            SIRIUS_LOG_DEBUG("Finding column {}", column_to_find);
            auto column_it = find(table->column_names.begin(), table->column_names.end(), column_to_find);
            if (column_it == table->column_names.end()) {
                throw InvalidInputException("Column not found");
            }
            auto column_name = table->column_names[column_ids[col].GetPrimaryIndex()];
            SIRIUS_LOG_DEBUG("Column found {}", column_name);
        }
    } else {
        // table not found
        throw InvalidInputException("Table not found");
    }
  }
  SIRIUS_LOG_DEBUG("Finished checking GPU Buffer Manager with num cols of {}", num_cols);

  //If there is a filter: apply filter, and write to output_relation (late materialized)
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
                } else if (filter_inside->filter_type == TableFilterType::IS_NOT_NULL ||
                           filter_inside->filter_type == TableFilterType::OPTIONAL_FILTER) {
                  continue;
                } else {
                  throw NotImplementedException("Filter type not supported: %d", static_cast<int>(filter_inside->filter_type));
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

      ConstantFilter** filter_constants = gpuBufferManager->customCudaHostAlloc<ConstantFilter*>(num_expr);
      vector<shared_ptr<GPUColumn>> expression_columns(num_expr);

      int expr_idx = 0;
      for (auto &f : table_filters->filters) {
        auto &column_index = f.first;
        auto &filter = f.second;

        if (filter->filter_type == TableFilterType::OPTIONAL_FILTER) {
          continue;
        }

        if (column_index < names.size()) {
          SIRIUS_LOG_DEBUG("Reading filter column from index {}", column_ids[column_index].GetPrimaryIndex());

          if (filter->filter_type == TableFilterType::CONJUNCTION_AND) {
            auto filter_pointer = filter.get();
            auto &filter_conjunction_and = filter_pointer->Cast<ConjunctionAndFilter>();

            for (int expr = 0; expr < filter_conjunction_and.child_filters.size(); expr++) {
                auto& filter_inside = filter_conjunction_and.child_filters[expr];
                if (filter_inside->filter_type == TableFilterType::CONSTANT_COMPARISON) {
                  SIRIUS_LOG_DEBUG("Reading constant comparison filter");
                  filter_constants[expr_idx] = &(filter_inside->Cast<ConstantFilter>());
                  expression_columns[expr_idx] = table->columns[column_ids[column_index].GetPrimaryIndex()];
                  expr_idx++;
                } else if (filter_inside->filter_type == TableFilterType::IS_NOT_NULL ||
                           filter_inside->filter_type == TableFilterType::OPTIONAL_FILTER) {
                  continue;
                } else {
                  throw NotImplementedException("Filter type not supported: %d", static_cast<int>(filter_inside->filter_type));
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

      if (num_expr > 0) {
        HandleArbitraryConstantExpression(expression_columns, count, row_ids, filter_constants, num_expr);
        // if (count[0] == 0) throw NotImplementedException("No match found");
      }
    }
    SIRIUS_LOG_DEBUG("Finished processing table filters");

    int index = 0;
    // projection id means that from this set of column ids that are being scanned, which index of column ids are getting projected out
    if (function.filter_prune) {
      SIRIUS_LOG_DEBUG("Running filter prune with args {}, {}, {}", projection_ids.size(), column_ids.size(), gen_row_id_column);
      if (projection_ids.size() == 0) {
        SIRIUS_LOG_DEBUG("Projection ids size is 0 so we are projecting all columns");
        for (int col = 0; col < column_ids.size() - gen_row_id_column; col++) {
            auto column_id = column_ids[col];
            SIRIUS_LOG_DEBUG("Reading column index (late materialized) {} and passing it to index in output relation {}", column_id.GetPrimaryIndex(), index);
            SIRIUS_LOG_DEBUG("Writing row IDs to output relation in index {}", index);
            // output_relation.columns[index] = make_shared_ptr<GPUColumn>(table->columns[column_id.GetPrimaryIndex()]->column_length, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.type, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.data,
            //                 table->columns[column_id.GetPrimaryIndex()]->data_wrapper.offset, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.num_bytes, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.is_string_data,
            //                 table->columns[column_id.GetPrimaryIndex()]->data_wrapper.validity_mask);
            // output_relation.columns[index]->is_unique = table->columns[column_id.GetPrimaryIndex()]->is_unique;
            output_relation.columns[index] = make_shared_ptr<GPUColumn>(table->columns[column_id.GetPrimaryIndex()]);
            if (row_ids) {
              output_relation.columns[index]->row_ids = row_ids; 
            }
            if (count) {
              output_relation.columns[index]->row_id_count = count[0];
            }
            index++;
        }
      } else { 
          for (int col = 0; col < projection_ids.size() - gen_row_id_column; ++col) {
            auto projection_id = projection_ids[col];
            SIRIUS_LOG_DEBUG("Reading column index (late materialized) {} and passing it to index in output relation {}", column_ids[projection_id].GetPrimaryIndex(), index);
            SIRIUS_LOG_DEBUG("Writing row IDs to output relation in index {}", index);
            // output_relation.columns[index] = make_shared_ptr<GPUColumn>(table->columns[column_ids[projection_id].GetPrimaryIndex()]->column_length, table->columns[column_ids[projection_id].GetPrimaryIndex()]->data_wrapper.type, table->columns[column_ids[projection_id].GetPrimaryIndex()]->data_wrapper.data,
            //                 table->columns[column_ids[projection_id].GetPrimaryIndex()]->data_wrapper.offset, table->columns[column_ids[projection_id].GetPrimaryIndex()]->data_wrapper.num_bytes, table->columns[column_ids[projection_id].GetPrimaryIndex()]->data_wrapper.is_string_data,
            //                 table->columns[column_ids[projection_id].GetPrimaryIndex()]->data_wrapper.validity_mask);
            // output_relation.columns[index]->is_unique = table->columns[column_ids[projection_id].GetPrimaryIndex()]->is_unique;
            output_relation.columns[index] = make_shared_ptr<GPUColumn>(table->columns[column_ids[projection_id].GetPrimaryIndex()]);
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
      SIRIUS_LOG_DEBUG("Running Index Scan with args {}, {}", column_ids.size(), gen_row_id_column);
      for (int col = 0; col < column_ids.size() - gen_row_id_column; col++) {
          auto column_id = column_ids[col];
          SIRIUS_LOG_DEBUG("Reading column index (late materialized) {} and passing it to index in output relation {}", column_id.GetPrimaryIndex(), index);
          SIRIUS_LOG_DEBUG("Writing row IDs to output relation in index {}", index);
          // output_relation.columns[index] = make_shared_ptr<GPUColumn>(table->columns[column_id.GetPrimaryIndex()]->column_length, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.type, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.data,
          //                 table->columns[column_id.GetPrimaryIndex()]->data_wrapper.offset, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.num_bytes, table->columns[column_id.GetPrimaryIndex()]->data_wrapper.is_string_data,
          //                 table->columns[column_id.GetPrimaryIndex()]->data_wrapper.validity_mask);
          // output_relation.columns[index]->is_unique = table->columns[column_id.GetPrimaryIndex()]->is_unique;
          output_relation.columns[index] = make_shared_ptr<GPUColumn>(table->columns[column_id.GetPrimaryIndex()]);
          if (row_ids) {
            output_relation.columns[index]->row_ids = row_ids; 
          }
          if (count) {
            output_relation.columns[index]->row_id_count = count[0];
          }
          index++;
      }
    }

    // Create row id column if required
    SIRIUS_LOG_DEBUG("Running with gen row id of {} and output relationship having {} cols", gen_row_id_column, output_relation.columns.size());
    if (gen_row_id_column) {
      auto num_out_rows = 0;
      if (output_relation.column_count < 2) { // In this case fallback to using the number of rows record by the GetDataDuckDBOpt
        num_out_rows = num_rows;
      } else { 
        num_out_rows = output_relation.columns[0]->column_length;
      }
      
      auto data = gpuBufferManager->customCudaMalloc<uint8_t>(num_out_rows * sizeof(int64_t), 0, 0);
      createRowIdColumn(data, num_out_rows);
      output_relation.columns.back() = make_shared_ptr<GPUColumn>(
        num_out_rows, GPUColumnType(GPUColumnTypeId::INT64), data, nullptr, num_out_rows * sizeof(int64_t), false, nullptr);
      output_relation.columns.back()->is_unique = true;
      if (row_ids) {
        output_relation.columns.back()->row_ids = row_ids; 
      }
      if (count) {
        output_relation.columns.back()->row_id_count = count[0];
      }
    }

    //measure time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_DEBUG("Table Scan time: {:.2f} ms", duration.count()/1000.0);
    return SourceResultType::FINISHED;
}

} // namespace duckdb