#include "gpu_columns.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

DataWrapper::DataWrapper(ColumnType _type, uint8_t* _data, size_t _size) : data(_data), size(_size) {
    type = _type;
    num_bytes = size * getColumnTypeSize();
    is_string_data = false;
};

DataWrapper::DataWrapper(ColumnType _type, uint8_t* _data, uint64_t* _offset, size_t _size, size_t _num_bytes, bool _is_string_data) : 
    data(_data), size(_size), type(_type), offset(_offset), num_bytes(_num_bytes), is_string_data(_is_string_data) {};

size_t 
DataWrapper::getColumnTypeSize() {
    switch (type) {
        case ColumnType::INT32:
            return sizeof(int);
        case ColumnType::INT64:
            return sizeof(uint64_t);
        case ColumnType::FLOAT32:
            return sizeof(float);
        case ColumnType::FLOAT64:
            return sizeof(double);
        case ColumnType::BOOLEAN:
            return sizeof(uint8_t);
        case ColumnType::VARCHAR:
            return 128;
    }
    return 0;
}

GPUColumn::GPUColumn(size_t _column_length, ColumnType type, uint8_t* data) {
    column_length = _column_length;
    data_wrapper = DataWrapper(type, data, _column_length);
    row_ids = nullptr;
    data_wrapper.offset = nullptr;
    data_wrapper.num_bytes = column_length * data_wrapper.getColumnTypeSize();
    is_unique = false;
}

GPUColumn::GPUColumn(size_t _column_length, ColumnType type, uint8_t* data, uint64_t* offset, size_t num_bytes, bool is_string_data) {
    column_length = _column_length;
    data_wrapper = DataWrapper(type, data, offset, _column_length, num_bytes, is_string_data);
    row_ids = nullptr;
    if (is_string_data) {
        data_wrapper.num_bytes = num_bytes;
    } else {
        data_wrapper.num_bytes = column_length * data_wrapper.getColumnTypeSize();
    }
    is_unique = false;
}

GPUColumn::GPUColumn(GPUColumn& other) {
    data_wrapper = other.data_wrapper;
    row_ids = other.row_ids;
    row_id_count = other.row_id_count;
    column_length = other.column_length;
    is_unique = other.is_unique;
}

cudf::column_view
GPUColumn::convertToCudfColumn() {
    cudf::size_type size = column_length;
    if (data_wrapper.type == ColumnType::INT64) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::UINT64), size, reinterpret_cast<void*>(data_wrapper.data), nullptr, 0);
        return column;
    } else if (data_wrapper.type == ColumnType::INT32) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::INT32), size, reinterpret_cast<void*>(data_wrapper.data), nullptr, 0);
        return column;
    } else if (data_wrapper.type == ColumnType::FLOAT32) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::FLOAT32), size, reinterpret_cast<void*>(data_wrapper.data), nullptr, 0);
        return column;
    } else if (data_wrapper.type == ColumnType::FLOAT64) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::FLOAT64), size, reinterpret_cast<void*>(data_wrapper.data), nullptr, 0);
        return column;
    } else if (data_wrapper.type == ColumnType::BOOLEAN) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::BOOL8), size, reinterpret_cast<void*>(data_wrapper.data), nullptr, 0);
        return column;
    } else if (data_wrapper.type == ColumnType::VARCHAR) {

        //convert offset to int32
        int32_t* new_offset = convertSiriusOffsetToCudfOffset();

        auto offsets_col = cudf::column_view(
            cudf::data_type{cudf::type_id::INT32},
            size + 1,
            reinterpret_cast<void*>(new_offset),
            nullptr,
            0
        );

        std::vector<cudf::column_view> children;
        children.push_back(offsets_col);

        // Build string column
        auto str_col = cudf::column_view(
            cudf::data_type{cudf::type_id::STRING},
            size,
            reinterpret_cast<void*>(data_wrapper.data),    // No top-level data buffer
            nullptr,    // Optional null mask
            0,                       // Null count
            0,                       // Offset
            std::move(children)
        );
        return str_col;
    }
}

void
GPUColumn::setFromCudfColumn(cudf::column& cudf_column, bool _is_unique, int32_t* _row_ids, uint64_t _row_id_count, GPUBufferManager* gpuBufferManager) {
    cudf::data_type col_type = cudf_column.type();
    cudf::size_type col_size = cudf_column.size();
    cudf::column::contents cont = cudf_column.release();
    // rmm_owned_buffer = std::move(cont.data);
    gpuBufferManager->rmm_stored_buffers.push_back(std::move(cont.data));

    data_wrapper.data = reinterpret_cast<uint8_t*>(gpuBufferManager->rmm_stored_buffers.back()->data());
    data_wrapper.size = col_size;
    column_length = data_wrapper.size;
    is_unique = _is_unique;
    //add data to allocation table in gpu buffer manager
    gpuBufferManager->allocation_table[0][reinterpret_cast<void*>(data_wrapper.data)] = column_length;

    if (col_type == cudf::data_type(cudf::type_id::STRING)) {
        cudf::column::contents child_cont = cont.children[0]->release();
        data_wrapper.is_string_data = true;
        data_wrapper.type = ColumnType::VARCHAR;
        int32_t* temp_offset = reinterpret_cast<int32_t*>(child_cont.data->data());
        convertCudfOffsetToSiriusOffset(temp_offset);
        //copy data from offset to num_bytes
        uint64_t* temp_num_bytes = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
        callCudaMemcpyDeviceToHost<uint64_t>(temp_num_bytes, data_wrapper.offset + column_length, 1, 0);
        data_wrapper.num_bytes = temp_num_bytes[0];
    } else if (col_type == cudf::data_type(cudf::type_id::UINT64)) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = ColumnType::INT64;
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    } else if (col_type == cudf::data_type(cudf::type_id::INT32)) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = ColumnType::INT32;
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    } else if (col_type == cudf::data_type(cudf::type_id::FLOAT32)) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = ColumnType::FLOAT32;
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    } else if (col_type == cudf::data_type(cudf::type_id::FLOAT64)) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = ColumnType::FLOAT64;
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    } else if (col_type == cudf::data_type(cudf::type_id::BOOL8)) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = ColumnType::BOOLEAN;
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    }

    if (_row_ids != nullptr) {
        convertCudfRowIdsToSiriusRowIds(_row_ids);
        row_id_count = _row_id_count;
    } else {
        row_ids = nullptr;
        row_id_count = 0;
    }
}

void
GPUColumn::setFromCudfScalar(cudf::scalar& cudf_scalar, GPUBufferManager* gpuBufferManager) {
    cudf::data_type scalar_type = cudf_scalar.type();
    if (scalar_type == cudf::data_type(cudf::type_id::UINT64)) {
        auto& typed_scalar = static_cast<cudf::numeric_scalar<uint64_t>&>(cudf_scalar);
        data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(sizeof(uint64_t), 0, 0);
        callCudaMemcpyDeviceToDevice<uint8_t>(data_wrapper.data, reinterpret_cast<uint8_t*>(typed_scalar.data()), sizeof(uint64_t), 0);
        data_wrapper.type = ColumnType::INT64;
        data_wrapper.num_bytes = sizeof(uint64_t);
    } else if (scalar_type == cudf::data_type(cudf::type_id::INT32)) {
        auto& typed_scalar = static_cast<cudf::numeric_scalar<int32_t>&>(cudf_scalar);
        data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(sizeof(int32_t), 0, 0);
        callCudaMemcpyDeviceToDevice<uint8_t>(data_wrapper.data, reinterpret_cast<uint8_t*>(typed_scalar.data()), sizeof(int32_t), 0);
        data_wrapper.type = ColumnType::INT32;
        data_wrapper.num_bytes = sizeof(int32_t);
    } else if (scalar_type == cudf::data_type(cudf::type_id::FLOAT32)) {
        auto& typed_scalar = static_cast<cudf::numeric_scalar<float>&>(cudf_scalar);
        data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(sizeof(float), 0, 0);
        callCudaMemcpyDeviceToDevice<uint8_t>(data_wrapper.data, reinterpret_cast<uint8_t*>(typed_scalar.data()), sizeof(float), 0);
        data_wrapper.type = ColumnType::FLOAT32;
        data_wrapper.num_bytes = sizeof(float);
    } else if (scalar_type == cudf::data_type(cudf::type_id::FLOAT64)) {
        auto& typed_scalar = static_cast<cudf::numeric_scalar<double>&>(cudf_scalar);
        data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(sizeof(double), 0, 0);
        callCudaMemcpyDeviceToDevice<uint8_t>(data_wrapper.data, reinterpret_cast<uint8_t*>(typed_scalar.data()), sizeof(double), 0);
        data_wrapper.type = ColumnType::FLOAT64;
        data_wrapper.num_bytes = sizeof(double);
    } else if (scalar_type == cudf::data_type(cudf::type_id::BOOL8)) {
        auto& typed_scalar = static_cast<cudf::numeric_scalar<bool>&>(cudf_scalar);
        data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(sizeof(uint8_t), 0, 0);
        callCudaMemcpyDeviceToDevice<uint8_t>(data_wrapper.data, reinterpret_cast<uint8_t*>(typed_scalar.data()), sizeof(uint8_t), 0);
        data_wrapper.type = ColumnType::BOOLEAN;
        data_wrapper.num_bytes = sizeof(uint8_t);
    }

    data_wrapper.size = 1;
    column_length = 1;
    data_wrapper.offset = nullptr;
    data_wrapper.is_string_data = false;
    row_ids = nullptr;
    row_id_count = 0;

}

int32_t*
GPUColumn::convertSiriusOffsetToCudfOffset() {
    return convertUInt64ToInt32(data_wrapper.offset, column_length + 1);
}

int32_t*
GPUColumn::convertSiriusRowIdsToCudfRowIds() {
    return convertUInt64ToInt32(row_ids, row_id_count);
}

void
GPUColumn::convertCudfRowIdsToSiriusRowIds(int32_t* cudf_row_ids) {
    row_ids = convertInt32ToUInt64(cudf_row_ids, row_id_count);
}

void
GPUColumn::convertCudfOffsetToSiriusOffset(int32_t* cudf_offset) {
    data_wrapper.offset = convertInt32ToUInt64(cudf_offset, column_length + 1);
}

GPUIntermediateRelation::GPUIntermediateRelation(size_t column_count) :
        column_count(column_count) {
    column_names.resize(column_count);
    columns.resize(column_count);
    for (int i = 0; i < column_count; i++) {
        columns[i] = nullptr;
    }
}

bool
GPUIntermediateRelation::checkLateMaterialization(size_t idx) {
    printf("Checking if column idx %ld needs to be materialized from column size %d\n", idx, columns.size());
    if (columns[idx] == nullptr) {
        printf("Column idx %ld is null\n", idx);
        return false;
    }

    if (columns[idx]->row_ids == nullptr) {
        printf("Column idx %d already materialized\n", idx);
    } else {
        printf("Column idx %d needs to be materialized\n", idx);
    }
    return columns[idx]->row_ids != nullptr;
}

int*
GPUColumn::GetDataInt32() {
    return reinterpret_cast<int*>(data_wrapper.data);
}

uint64_t*
GPUColumn::GetDataUInt64() {
    return reinterpret_cast<uint64_t*>(data_wrapper.data);
}

float* 
GPUColumn::GetDataFloat32() {
    return reinterpret_cast<float*>(data_wrapper.data);
}

double*
GPUColumn::GetDataFloat64() {
    return reinterpret_cast<double*>(data_wrapper.data);
}

uint8_t*
GPUColumn::GetDataBoolean() {
    return reinterpret_cast<uint8_t*>(data_wrapper.data);
}

char*
GPUColumn::GetDataVarChar() {
    return reinterpret_cast<char*>(data_wrapper.data);
}

uint8_t* 
GPUColumn::GetData() {
    switch (data_wrapper.type) {
        case ColumnType::INT32:
            return reinterpret_cast<uint8_t*>(GetDataInt32());
        case ColumnType::INT64:
            return reinterpret_cast<uint8_t*>(GetDataUInt64());
        case ColumnType::FLOAT32:
            return reinterpret_cast<uint8_t*>(GetDataFloat32());
        case ColumnType::FLOAT64:
            return reinterpret_cast<uint8_t*>(GetDataFloat64());
        case ColumnType::BOOLEAN:
            return reinterpret_cast<uint8_t*>(GetDataBoolean());
        case ColumnType::VARCHAR:
            return reinterpret_cast<uint8_t*>(GetDataVarChar());
        default:
            return nullptr;
    }
}

} // namespace duckdb