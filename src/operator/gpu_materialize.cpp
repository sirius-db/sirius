#include "operator/gpu_materialize.hpp"
#include "log/logging.hpp"

namespace duckdb {

template <typename T>
shared_ptr<GPUColumn> 
ResolveTypeMaterializeExpression(shared_ptr<GPUColumn> column, GPUBufferManager* gpuBufferManager) {
    size_t size;
    T* a;
    if (column->data_wrapper.data == nullptr || column->column_length == 0) {
        return make_shared_ptr<GPUColumn>(column->column_length, column->data_wrapper.type, nullptr);
    }
    if (column->row_ids != nullptr) {
        T* temp = reinterpret_cast<T*> (column->data_wrapper.data);
        uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (column->row_ids);
        materializeExpression<T>(temp, a, row_ids_input, column->row_id_count);
        size = column->row_id_count;
    } else {
        a = reinterpret_cast<T*> (column->data_wrapper.data);
        size = column->column_length;
    }
    shared_ptr<GPUColumn> result = make_shared_ptr<GPUColumn>(size, column->data_wrapper.type, reinterpret_cast<uint8_t*>(a));
    result->is_unique = column->is_unique;
    return result;
}

shared_ptr<GPUColumn> 
ResolveTypeMaterializeString(shared_ptr<GPUColumn> column, GPUBufferManager* gpuBufferManager) {
    size_t size;
    uint8_t* a;
    uint64_t* result_offset; 
    uint64_t* new_num_bytes;
    if (column->data_wrapper.data == nullptr || column->column_length == 0) {
        return make_shared_ptr<GPUColumn>(column->column_length, column->data_wrapper.type, nullptr, nullptr, column->data_wrapper.num_bytes, column->data_wrapper.is_string_data);
    }
    if (column->row_ids != nullptr) {
		// Late materalize the input relationship
		uint8_t* data = column->data_wrapper.data;
		uint64_t* offset = column->data_wrapper.offset;
		uint64_t* row_ids = column->row_ids;
		size = column->row_id_count;
		materializeString(data, offset, a, result_offset, row_ids, new_num_bytes, size);
    } else {
        a = column->data_wrapper.data;
        result_offset = column->data_wrapper.offset;
        size = column->column_length;
        new_num_bytes = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
        new_num_bytes[0] = column->data_wrapper.num_bytes;
    }
    shared_ptr<GPUColumn> result = make_shared_ptr<GPUColumn>(size, column->data_wrapper.type, a, result_offset, new_num_bytes[0], column->data_wrapper.is_string_data);
    result->is_unique = column->is_unique;
    return result;
}

shared_ptr<GPUColumn> 
HandleMaterializeExpression(shared_ptr<GPUColumn> column, GPUBufferManager* gpuBufferManager) {
    switch(column->data_wrapper.type.id()) {
        case GPUColumnTypeId::INT32:
        case GPUColumnTypeId::DATE:
            return ResolveTypeMaterializeExpression<int>(column, gpuBufferManager);
        case GPUColumnTypeId::INT64:
            return ResolveTypeMaterializeExpression<uint64_t>(column, gpuBufferManager);
        case GPUColumnTypeId::FLOAT32:
            return ResolveTypeMaterializeExpression<float>(column, gpuBufferManager);
        case GPUColumnTypeId::FLOAT64:
            return ResolveTypeMaterializeExpression<double>(column, gpuBufferManager);
        case GPUColumnTypeId::BOOLEAN:
            return ResolveTypeMaterializeExpression<uint8_t>(column, gpuBufferManager);
        case GPUColumnTypeId::VARCHAR:
            return ResolveTypeMaterializeString(column, gpuBufferManager);
        case GPUColumnTypeId::DECIMAL: {
            switch (column->data_wrapper.getColumnTypeSize()) {
                case sizeof(int32_t):
                    return ResolveTypeMaterializeExpression<int32_t>(column, gpuBufferManager);
                case sizeof(int64_t):
                    return ResolveTypeMaterializeExpression<int64_t>(column, gpuBufferManager);
                case sizeof(__int128_t):
                    return ResolveTypeMaterializeExpression<__int128_t>(column, gpuBufferManager);
                default:
                    throw NotImplementedException("Unsupported sirius DECIMAL column type size in `HandleMaterializeExpression`: %zu",
                                                  column->data_wrapper.getColumnTypeSize());
            }
        }
        default:
            throw NotImplementedException("Unsupported sirius column type in `HandleMaterializeExpression`: %d",
                                          static_cast<int>(column->data_wrapper.type.id()));
    }
}

void
HandleMaterializeRowIDs(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager, bool maintain_unique) {
    vector<uint64_t*> new_row_ids;
    vector<uint64_t*> prev_row_ids;
    for (int i = 0; i < input_relation.columns.size(); i++) {
        SIRIUS_LOG_DEBUG("Materializing column idx {} in input relation to idx {} in output relation", i, i);
        if (count == 0) {
            output_relation.columns[i] = make_shared_ptr<GPUColumn>(0, input_relation.columns[i]->data_wrapper.type, nullptr,
                        nullptr, 0, input_relation.columns[i]->data_wrapper.is_string_data);
            output_relation.columns[i]->row_id_count = 0;
            if (maintain_unique) {
                output_relation.columns[i]->is_unique = input_relation.columns[i]->is_unique;
            } else {
                output_relation.columns[i]->is_unique = false;
            }
            continue;
        }
        output_relation.columns[i] = make_shared_ptr<GPUColumn>(input_relation.columns[i]->column_length, input_relation.columns[i]->data_wrapper.type, input_relation.columns[i]->data_wrapper.data,
                        input_relation.columns[i]->data_wrapper.offset, input_relation.columns[i]->data_wrapper.num_bytes, input_relation.columns[i]->data_wrapper.is_string_data);
        if (maintain_unique) {
            output_relation.columns[i]->is_unique = input_relation.columns[i]->is_unique;
        } else {
            output_relation.columns[i]->is_unique = false;
        }
        if (row_ids) {
            if (input_relation.columns[i]->row_ids == nullptr) {
                output_relation.columns[i]->row_ids = row_ids;
            } else {
                auto it = find(prev_row_ids.begin(), prev_row_ids.end(), input_relation.columns[i]->row_ids);
                if (it != prev_row_ids.end()) {
                    auto idx = it - prev_row_ids.begin();
                    output_relation.columns[i]->row_ids = new_row_ids[idx];
                } else {
                    uint64_t* temp_prev_row_ids = reinterpret_cast<uint64_t*> (input_relation.columns[i]->row_ids);
                    uint64_t* temp_new_row_ids;
                    materializeExpression<uint64_t>(temp_prev_row_ids, temp_new_row_ids, row_ids, count);
                    output_relation.columns[i]->row_ids = temp_new_row_ids;
                    new_row_ids.push_back(temp_new_row_ids);
                    prev_row_ids.push_back(temp_prev_row_ids);
                }
            }
        }
        output_relation.columns[i]->row_id_count = count;
    }
}

void
HandleMaterializeRowIDsRHS(GPUIntermediateRelation& hash_table_result, GPUIntermediateRelation& output_relation, 
    vector<column_t> rhs_output_columns, size_t offset, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager, bool maintain_unique) {
    vector<uint64_t*> new_row_ids;
    vector<uint64_t*> prev_row_ids;
    for (idx_t i = 0; i < rhs_output_columns.size(); i++) {
        const auto rhs_col = rhs_output_columns[i];
        if (count == 0) {
            output_relation.columns[offset + i] = make_shared_ptr<GPUColumn>(0, hash_table_result.columns[rhs_col]->data_wrapper.type, nullptr,
                        nullptr, 0, hash_table_result.columns[rhs_col]->data_wrapper.is_string_data);
            output_relation.columns[offset + i]->row_id_count = 0;
            if (maintain_unique) {
                output_relation.columns[offset + i]->is_unique = hash_table_result.columns[rhs_col]->is_unique;
            } else {
                output_relation.columns[offset + i]->is_unique = false;
            }
            continue;
        }
        output_relation.columns[offset + i] = make_shared_ptr<GPUColumn>(hash_table_result.columns[rhs_col]->column_length, hash_table_result.columns[rhs_col]->data_wrapper.type, hash_table_result.columns[rhs_col]->data_wrapper.data,
                        hash_table_result.columns[rhs_col]->data_wrapper.offset, hash_table_result.columns[rhs_col]->data_wrapper.num_bytes, hash_table_result.columns[rhs_col]->data_wrapper.is_string_data);
        if (maintain_unique) {
            output_relation.columns[offset + i]->is_unique = hash_table_result.columns[rhs_col]->is_unique;
        } else {
            output_relation.columns[offset + i]->is_unique = false;
        }
        if (row_ids) {
            if (hash_table_result.columns[rhs_col]->row_ids == nullptr) {
                output_relation.columns[offset + i]->row_ids = row_ids;
            } else {
                auto it = find(prev_row_ids.begin(), prev_row_ids.end(), hash_table_result.columns[rhs_col]->row_ids);
                if (it != prev_row_ids.end()) {
                    auto idx = it - prev_row_ids.begin();
                    output_relation.columns[offset + i]->row_ids = new_row_ids[idx];
                } else {
                    uint64_t* temp_prev_row_ids = reinterpret_cast<uint64_t*> (hash_table_result.columns[rhs_col]->row_ids);
                    uint64_t* temp_new_row_ids;
                    materializeExpression<uint64_t>(temp_prev_row_ids, temp_new_row_ids, row_ids, count);
                    output_relation.columns[offset + i]->row_ids = temp_new_row_ids;
                    new_row_ids.push_back(temp_new_row_ids);
                    prev_row_ids.push_back(temp_prev_row_ids);
                }
            }
        }
        output_relation.columns[offset + i]->row_id_count = count;
    }
}

void
HandleMaterializeRowIDsLHS(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, 
    vector<column_t> lhs_output_columns, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager, bool maintain_unique) {
    vector<uint64_t*> new_row_ids;
    vector<uint64_t*> prev_row_ids;
    for (idx_t i = 0; i < lhs_output_columns.size(); i++) {
        const auto lhs_col = lhs_output_columns[i];
        SIRIUS_LOG_DEBUG("Materializing column idx {} from input relation to idx {} in output relation", lhs_col, i);
        if (count == 0) {
            output_relation.columns[i] = make_shared_ptr<GPUColumn>(0, input_relation.columns[lhs_col]->data_wrapper.type, nullptr,
                nullptr, 0, input_relation.columns[lhs_col]->data_wrapper.is_string_data);
            output_relation.columns[i]->row_id_count = 0;
            if (maintain_unique) {
                output_relation.columns[i]->is_unique = input_relation.columns[lhs_col]->is_unique;
            } else {
                output_relation.columns[i]->is_unique = false;
            }
            continue;
        }
        output_relation.columns[i] = make_shared_ptr<GPUColumn>(input_relation.columns[lhs_col]->column_length, input_relation.columns[lhs_col]->data_wrapper.type, input_relation.columns[lhs_col]->data_wrapper.data,
                input_relation.columns[lhs_col]->data_wrapper.offset, input_relation.columns[lhs_col]->data_wrapper.num_bytes, input_relation.columns[lhs_col]->data_wrapper.is_string_data);
        if (maintain_unique) {
            output_relation.columns[i]->is_unique = input_relation.columns[lhs_col]->is_unique;
        } else {
            output_relation.columns[i]->is_unique = false;
        }
        if (row_ids) {
            if (input_relation.columns[lhs_col]->row_ids == nullptr) {
                output_relation.columns[i]->row_ids = row_ids;
            } else {
                auto it = find(prev_row_ids.begin(), prev_row_ids.end(), input_relation.columns[lhs_col]->row_ids);
                if (it != prev_row_ids.end()) {
                    auto idx = it - prev_row_ids.begin();
                    output_relation.columns[i]->row_ids = new_row_ids[idx];
                } else {
                    uint64_t* temp_prev_row_ids = reinterpret_cast<uint64_t*> (input_relation.columns[lhs_col]->row_ids);
                    uint64_t* temp_new_row_ids;
                    materializeExpression<uint64_t>(temp_prev_row_ids, temp_new_row_ids, row_ids, count);
                    output_relation.columns[i]->row_ids = temp_new_row_ids;
                    new_row_ids.push_back(temp_new_row_ids);
                    prev_row_ids.push_back(temp_prev_row_ids);
                }
            }
        }
        output_relation.columns[i]->row_id_count = count;
    }
}

} // namespace duckdb