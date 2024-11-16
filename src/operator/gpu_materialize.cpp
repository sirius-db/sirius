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

GPUColumn* 
HandleMaterializeRowIDs(GPUColumn* in_column, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager) {
    GPUColumn* out_column = in_column;
    if (row_ids) {
        if (in_column->row_ids == nullptr) {
            out_column->row_ids = row_ids;
            out_column->row_id_count = count;
        } else {
            uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (in_column->row_ids);
            uint64_t* new_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(count, 0, 0);
            materializeExpression<uint64_t>(row_ids_input, new_row_ids, row_ids, count);
            out_column->row_ids = new_row_ids;
            out_column->row_id_count = count;
            printf("row id count %ld\n", count);
        }
    }
    return out_column;
}

} // namespace duckdb