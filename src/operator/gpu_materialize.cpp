#include "operator/gpu_materialize.hpp"

namespace duckdb {

template <typename T>
GPUColumn* 
ResolveTypeMaterializeExpression(GPUColumn* column, BoundReferenceExpression& bound_ref, GPUBufferManager* gpuBufferManager) {
    size_t size;
    T* a;
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
        default:
            throw NotImplementedException("Unsupported column type");
    }
}

} // namespace duckdb