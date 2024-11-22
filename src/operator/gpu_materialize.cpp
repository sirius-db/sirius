#include "operator/gpu_materialize.hpp"

namespace duckdb {

template <typename T>
GPUColumn* 
ResolveTypeMaterializeExpression(GPUColumn* column, BoundReferenceExpression& bound_ref, GPUBufferManager* gpuBufferManager) {
    size_t size;
    T* a;
    if (column->data_wrapper.data == nullptr) {
        return new GPUColumn(column->column_length, column->data_wrapper.type, nullptr);
    }
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
    return result;
}

GPUColumn* 
HandleMaterializeExpression(GPUColumn* column, BoundReferenceExpression& bound_ref, GPUBufferManager* gpuBufferManager) {
    switch(column->data_wrapper.type) {
        case ColumnType::INT32:
            return ResolveTypeMaterializeExpression<int>(column, bound_ref, gpuBufferManager);
        case ColumnType::INT64:
            return ResolveTypeMaterializeExpression<uint64_t>(column, bound_ref, gpuBufferManager);
        case ColumnType::FLOAT32:
            return ResolveTypeMaterializeExpression<float>(column, bound_ref, gpuBufferManager);
        case ColumnType::FLOAT64:
            return ResolveTypeMaterializeExpression<double>(column, bound_ref, gpuBufferManager);
        case ColumnType::BOOLEAN:
            return ResolveTypeMaterializeExpression<uint8_t>(column, bound_ref, gpuBufferManager);
        default:
            throw NotImplementedException("Unsupported column type");
    }
}

// GPUColumn* 
// HandleMaterializeRowIDs(GPUColumn* in_column, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager) {
//     GPUColumn* out_column;
//     if (row_ids == nullptr) throw NotImplementedException("Row ids cannot be null");
//     if (in_column->row_ids == nullptr) {
//         out_column = new GPUColumn(in_column->column_length, in_column->data_wrapper.type, in_column->data_wrapper.data);
//         out_column->row_ids = row_ids;
//         out_column->row_id_count = count;
//     } else {
//         out_column = new GPUColumn(in_column->column_length, in_column->data_wrapper.type, in_column->data_wrapper.data);
//         uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (in_column->row_ids);
//         uint64_t* new_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(count, 0, 0);
//         materializeExpression<uint64_t>(row_ids_input, new_row_ids, row_ids, count);
//         out_column->row_ids = new_row_ids;
//         out_column->row_id_count = count;
//     }
//     return out_column;
// }

void
HandleMaterializeRowIDs(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager) {
    vector<uint64_t*> new_row_ids;
    vector<uint64_t*> prev_row_ids;
    for (int i = 0; i < input_relation.columns.size(); i++) {
        printf("Passing column idx %d in input relation to idx %d in output relation\n", i, i);
        if (count == 0) {
            output_relation.columns[i] = new GPUColumn(0, input_relation.columns[i]->data_wrapper.type, input_relation.columns[i]->data_wrapper.data);
            output_relation.columns[i]->row_id_count = 0;
            continue;
        }
        output_relation.columns[i] = new GPUColumn(input_relation.columns[i]->column_length, input_relation.columns[i]->data_wrapper.type, input_relation.columns[i]->data_wrapper.data);
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
                    uint64_t* temp_new_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(count, 0, 0);
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
HandleMaterializeRowIDsRHS(GPUIntermediateRelation& hash_table_result, GPUIntermediateRelation& output_relation, vector<idx_t> rhs_output_columns, size_t offset, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager) {
    vector<uint64_t*> new_row_ids;
    vector<uint64_t*> prev_row_ids;
    for (idx_t i = 0; i < rhs_output_columns.size(); i++) {
        const auto output_col_idx = rhs_output_columns[i];
        printf("Passing column idx %d from RHS (late materialized) to idx %d in output relation\n", i, offset + output_col_idx);
        if (count == 0) {
            output_relation.columns[offset + output_col_idx] = new GPUColumn(0, hash_table_result.columns[i]->data_wrapper.type, hash_table_result.columns[i]->data_wrapper.data);
            output_relation.columns[offset + output_col_idx]->row_id_count = 0;
            continue;
        }
        output_relation.columns[offset + output_col_idx] = new GPUColumn(hash_table_result.columns[i]->column_length, hash_table_result.columns[i]->data_wrapper.type, hash_table_result.columns[i]->data_wrapper.data);
        if (row_ids) {
            if (hash_table_result.columns[i]->row_ids == nullptr) {
                output_relation.columns[offset + output_col_idx]->row_ids = row_ids;
            } else {
                auto it = find(prev_row_ids.begin(), prev_row_ids.end(), hash_table_result.columns[i]->row_ids);
                if (it != prev_row_ids.end()) {
                    auto idx = it - prev_row_ids.begin();
                    output_relation.columns[offset + output_col_idx]->row_ids = new_row_ids[idx];
                } else {
                    uint64_t* temp_prev_row_ids = reinterpret_cast<uint64_t*> (hash_table_result.columns[i]->row_ids);
                    uint64_t* temp_new_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(count, 0, 0);
                    materializeExpression<uint64_t>(temp_prev_row_ids, temp_new_row_ids, row_ids, count);
                    output_relation.columns[offset + output_col_idx]->row_ids = temp_new_row_ids;
                    new_row_ids.push_back(temp_new_row_ids);
                    prev_row_ids.push_back(temp_prev_row_ids);
                }
            }
        }
        output_relation.columns[offset + output_col_idx]->row_id_count = count;
    }
}

} // namespace duckdb