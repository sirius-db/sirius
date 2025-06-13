#include "gpu_columns.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"
#include "duckdb/common/types/decimal.hpp"

namespace duckdb {

size_t GPUDecimalTypeInfo::GetDecimalTypeSize() const {
    if (width_ <= Decimal::MAX_WIDTH_INT16) {
        return sizeof(int16_t);
    } else if (width_ <= Decimal::MAX_WIDTH_INT32) {
        return sizeof(int32_t);
    } else if (width_ <= Decimal::MAX_WIDTH_INT64) {
        return sizeof(int64_t);
    } else if (width_ <= Decimal::MAX_WIDTH_INT128) {
        return sizeof(__int128_t);
    } else {
        throw InternalException("Decimal has a width of %d which is bigger than the maximum supported width of %d",
                                width_, DecimalType::MaxWidth());
    }
}

DataWrapper::DataWrapper(GPUColumnType _type, uint8_t* _data, size_t _size) : data(_data), size(_size) {
    type = _type;
    num_bytes = size * getColumnTypeSize();
    is_string_data = false;
};

DataWrapper::DataWrapper(GPUColumnType _type, uint8_t* _data, uint64_t* _offset, size_t _size, size_t _num_bytes, bool _is_string_data) : 
    data(_data), size(_size), type(_type), offset(_offset), num_bytes(_num_bytes), is_string_data(_is_string_data) {};

size_t 
DataWrapper::getColumnTypeSize() const {
    switch (type.id()) {
        case GPUColumnTypeId::INT32:
        case GPUColumnTypeId::DATE:
            return sizeof(int);
        case GPUColumnTypeId::INT64:
            return sizeof(uint64_t);
        case GPUColumnTypeId::INT128:
            return sizeof(__uint128_t);
        case GPUColumnTypeId::FLOAT32:
            return sizeof(float);
        case GPUColumnTypeId::FLOAT64:
            return sizeof(double);
        case GPUColumnTypeId::BOOLEAN:
            return sizeof(uint8_t);
        case GPUColumnTypeId::VARCHAR:
            return 128;
        case GPUColumnTypeId::DECIMAL: {
            GPUDecimalTypeInfo* decimal_type_info = type.GetDecimalTypeInfo();
            if (decimal_type_info == nullptr) {
                throw InternalException("`decimal_type_info` not set for DECIMAL type in `getColumnTypeSize`");
            }
            return decimal_type_info->GetDecimalTypeSize();
        }
        default:
            throw duckdb::InternalException("Unsupported sirius column type in `getColumnTypeSize()`: %d",
                                            static_cast<int>(type.id()));
    }
}

GPUColumn::GPUColumn(size_t _column_length, GPUColumnType type, uint8_t* data) {
    column_length = _column_length;
    data_wrapper = DataWrapper(type, data, _column_length);
    row_ids = nullptr;
    data_wrapper.offset = nullptr;
    data_wrapper.num_bytes = column_length * data_wrapper.getColumnTypeSize();
    is_unique = false;
}

GPUColumn::GPUColumn(size_t _column_length, GPUColumnType type, uint8_t* data, uint64_t* offset, size_t num_bytes, bool is_string_data) {
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
    SIRIUS_LOG_DEBUG("Converting GPUColumn to cuDF column");
    cudf::size_type size = column_length;
    if (data_wrapper.type.id() == GPUColumnTypeId::INT64) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::UINT64), size, reinterpret_cast<void*>(data_wrapper.data), nullptr, 0);
        return column;
    } else if (data_wrapper.type.id() == GPUColumnTypeId::INT32) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::INT32), size, reinterpret_cast<void*>(data_wrapper.data), nullptr, 0);
        return column;
    } else if (data_wrapper.type.id() == GPUColumnTypeId::FLOAT32) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::FLOAT32), size, reinterpret_cast<void*>(data_wrapper.data), nullptr, 0);
        return column;
    } else if (data_wrapper.type.id() == GPUColumnTypeId::FLOAT64) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::FLOAT64), size, reinterpret_cast<void*>(data_wrapper.data), nullptr, 0);
        return column;
    } else if (data_wrapper.type.id() == GPUColumnTypeId::BOOLEAN) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::BOOL8), size, reinterpret_cast<void*>(data_wrapper.data), nullptr, 0);
        return column;
    } else if (data_wrapper.type.id() == GPUColumnTypeId::DATE) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::TIMESTAMP_DAYS), size, reinterpret_cast<void*>(data_wrapper.data), nullptr, 0);
        return column;
    } else if (data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
        //convert offset to int32
        // int32_t* new_offset = convertSiriusOffsetToCudfOffset();

        auto offsets_col = cudf::column_view(
            cudf::data_type{cudf::type_id::INT64},
            size + 1,
            reinterpret_cast<void*>(data_wrapper.offset),
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
    } else if (data_wrapper.type.id() == GPUColumnTypeId::DECIMAL) {
        cudf::data_type cudf_type;
        switch (data_wrapper.getColumnTypeSize()) {
            case sizeof(int32_t): {
                // cudf decimal type uses negative scale, same for below
                cudf_type = cudf::data_type(cudf::type_id::DECIMAL32, -data_wrapper.type.GetDecimalTypeInfo()->scale_);
                break;
            }
            case sizeof(int64_t): {
                cudf_type = cudf::data_type(cudf::type_id::DECIMAL64, -data_wrapper.type.GetDecimalTypeInfo()->scale_);
                break;
            }
            default:
                throw duckdb::InternalException("Unsupported sirius DECIMAL column type size in `convertToCudfColumn()`: %zu",
                                                data_wrapper.getColumnTypeSize());
        }
        return cudf::column_view(cudf_type, size, reinterpret_cast<void*>(data_wrapper.data), nullptr, 0);
    }
    throw duckdb::InternalException("Unsupported sirius column type in `convertToCudfColumn()`: %d", data_wrapper.type.id());
}

void
GPUColumn::setFromCudfColumn(cudf::column& cudf_column, bool _is_unique, int32_t* _row_ids, uint64_t _row_id_count, GPUBufferManager* gpuBufferManager) {
    SIRIUS_LOG_DEBUG("Set a GPUColumn from cudf::column");
    cudf::data_type col_type = cudf_column.type();
    cudf::size_type col_size = cudf_column.size();
    cudf::column::contents cont = cudf_column.release();
    gpuBufferManager->rmm_stored_buffers.push_back(std::move(cont.data));

    data_wrapper.data = reinterpret_cast<uint8_t*>(gpuBufferManager->rmm_stored_buffers.back()->data());
    data_wrapper.size = col_size;
    column_length = data_wrapper.size;
    is_unique = _is_unique;
    //add data to allocation table in gpu buffer manager
    if (col_type == cudf::data_type(cudf::type_id::STRING)) {
        if (cont.children[0]->type().id() == cudf::type_id::INT32) {
            cudf::column::contents child_cont = cont.children[0]->release();
            gpuBufferManager->rmm_stored_buffers.push_back(std::move(child_cont.data));
            data_wrapper.is_string_data = true;
            data_wrapper.type = GPUColumnType(GPUColumnTypeId::VARCHAR);
            int32_t* temp_offset = reinterpret_cast<int32_t*>(gpuBufferManager->rmm_stored_buffers.back()->data());
            convertCudfOffsetToSiriusOffset(temp_offset);
            //copy data from offset to num_bytes
            uint64_t* temp_num_bytes = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
            callCudaMemcpyDeviceToHost<uint64_t>(temp_num_bytes, data_wrapper.offset + column_length, 1, 0);
            data_wrapper.num_bytes = temp_num_bytes[0];
        } else if (cont.children[0]->type().id() == cudf::type_id::INT64) {
            cudf::column::contents child_cont = cont.children[0]->release();
            gpuBufferManager->rmm_stored_buffers.push_back(std::move(child_cont.data));
            data_wrapper.is_string_data = true;
            data_wrapper.type = GPUColumnType(GPUColumnTypeId::VARCHAR);
            data_wrapper.offset = reinterpret_cast<uint64_t*>(gpuBufferManager->rmm_stored_buffers.back()->data());
            //copy data from offset to num_bytes
            uint64_t* temp_num_bytes = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
            callCudaMemcpyDeviceToHost<uint64_t>(temp_num_bytes, data_wrapper.offset + column_length, 1, 0);
            data_wrapper.num_bytes = temp_num_bytes[0];
        }
    } else if (col_type == cudf::data_type(cudf::type_id::UINT64)) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::INT64);
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    } else if (col_type == cudf::data_type(cudf::type_id::INT32)) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::INT32);
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    } else if (col_type == cudf::data_type(cudf::type_id::FLOAT32)) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::FLOAT32);
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    } else if (col_type == cudf::data_type(cudf::type_id::FLOAT64)) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::FLOAT64);
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    } else if (col_type == cudf::data_type(cudf::type_id::BOOL8)) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::BOOLEAN);
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    } else if (col_type == cudf::data_type(cudf::type_id::TIMESTAMP_DAYS)) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::DATE);
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    } else if (col_type.id() == cudf::type_id::DECIMAL32) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::DECIMAL);
        // cudf decimal type uses negative scale, same for below
        data_wrapper.type.SetDecimalTypeInfo(Decimal::MAX_WIDTH_INT32, -col_type.scale());
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    } else if (col_type.id() == cudf::type_id::DECIMAL64) {
        data_wrapper.is_string_data = false;
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::DECIMAL);
        data_wrapper.type.SetDecimalTypeInfo(Decimal::MAX_WIDTH_INT64, -col_type.scale());
        data_wrapper.num_bytes = col_size * data_wrapper.getColumnTypeSize();
        data_wrapper.offset = nullptr;
    } else {
        throw NotImplementedException("Unsupported cudf data type in `setFromCudfColumn`: %d",
                                      static_cast<int>(col_type.id()));
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
    SIRIUS_LOG_DEBUG("Set a GPUColumn from cudf::scalar");
    cudf::data_type scalar_type = cudf_scalar.type();
    if (scalar_type == cudf::data_type(cudf::type_id::UINT64)) {
        auto& typed_scalar = static_cast<cudf::numeric_scalar<uint64_t>&>(cudf_scalar);
        data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(sizeof(uint64_t), 0, 0);
        callCudaMemcpyDeviceToDevice<uint8_t>(data_wrapper.data, reinterpret_cast<uint8_t*>(typed_scalar.data()), sizeof(uint64_t), 0);
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::INT64);
        data_wrapper.num_bytes = sizeof(uint64_t);
    } else if (scalar_type == cudf::data_type(cudf::type_id::INT32)) {
        auto& typed_scalar = static_cast<cudf::numeric_scalar<int32_t>&>(cudf_scalar);
        data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(sizeof(int32_t), 0, 0);
        callCudaMemcpyDeviceToDevice<uint8_t>(data_wrapper.data, reinterpret_cast<uint8_t*>(typed_scalar.data()), sizeof(int32_t), 0);
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::INT32);
        data_wrapper.num_bytes = sizeof(int32_t);
    } else if (scalar_type == cudf::data_type(cudf::type_id::FLOAT32)) {
        auto& typed_scalar = static_cast<cudf::numeric_scalar<float>&>(cudf_scalar);
        data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(sizeof(float), 0, 0);
        callCudaMemcpyDeviceToDevice<uint8_t>(data_wrapper.data, reinterpret_cast<uint8_t*>(typed_scalar.data()), sizeof(float), 0);
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::FLOAT32);
        data_wrapper.num_bytes = sizeof(float);
    } else if (scalar_type == cudf::data_type(cudf::type_id::FLOAT64)) {
        auto& typed_scalar = static_cast<cudf::numeric_scalar<double>&>(cudf_scalar);
        data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(sizeof(double), 0, 0);
        callCudaMemcpyDeviceToDevice<uint8_t>(data_wrapper.data, reinterpret_cast<uint8_t*>(typed_scalar.data()), sizeof(double), 0);
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::FLOAT64);
        data_wrapper.num_bytes = sizeof(double);
    } else if (scalar_type == cudf::data_type(cudf::type_id::BOOL8)) {
        auto& typed_scalar = static_cast<cudf::numeric_scalar<bool>&>(cudf_scalar);
        data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(sizeof(uint8_t), 0, 0);
        callCudaMemcpyDeviceToDevice<uint8_t>(data_wrapper.data, reinterpret_cast<uint8_t*>(typed_scalar.data()), sizeof(uint8_t), 0);
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::BOOLEAN);
        data_wrapper.num_bytes = sizeof(uint8_t);
    } else if (scalar_type.id() == cudf::type_id::DECIMAL32){
        auto& typed_scalar = static_cast<cudf::fixed_point_scalar<numeric::decimal32>&>(cudf_scalar);
        data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(sizeof(int32_t), 0, 0);
        callCudaMemcpyDeviceToDevice<uint8_t>(data_wrapper.data, reinterpret_cast<uint8_t*>(typed_scalar.data()), sizeof(int32_t), 0);
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::DECIMAL);
         // cudf decimal type uses negative scale, same for below
        data_wrapper.type.SetDecimalTypeInfo(Decimal::MAX_WIDTH_INT32, -typed_scalar.type().scale());
        data_wrapper.num_bytes = sizeof(int32_t);
    } else if (scalar_type.id() == cudf::type_id::DECIMAL64){
        auto& typed_scalar = static_cast<cudf::fixed_point_scalar<numeric::decimal64>&>(cudf_scalar);
        data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(sizeof(int64_t), 0, 0);
        callCudaMemcpyDeviceToDevice<uint8_t>(data_wrapper.data, reinterpret_cast<uint8_t*>(typed_scalar.data()), sizeof(int64_t), 0);
        data_wrapper.type = GPUColumnType(GPUColumnTypeId::DECIMAL);
        data_wrapper.type.SetDecimalTypeInfo(Decimal::MAX_WIDTH_INT64, -typed_scalar.type().scale());
        data_wrapper.num_bytes = sizeof(int64_t);
    } else {
        throw NotImplementedException("Unsupported cudf data type in `setFromCudfScalar`: %d",
                                      static_cast<int>(scalar_type.id()));
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
    SIRIUS_LOG_DEBUG("Checking if column idx {} needs to be materialized from column size {}", idx, columns.size());
    if (columns[idx] == nullptr) {
        SIRIUS_LOG_DEBUG("Column idx {} is null", idx);
        return false;
    }

    if (columns[idx]->row_ids == nullptr) {
        SIRIUS_LOG_DEBUG("Column idx {} already materialized", idx);
    } else {
        SIRIUS_LOG_DEBUG("Column idx {} needs to be materialized", idx);
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
    switch (data_wrapper.type.id()) {
        case GPUColumnTypeId::INT32:
            return reinterpret_cast<uint8_t*>(GetDataInt32());
        case GPUColumnTypeId::INT64:
            return reinterpret_cast<uint8_t*>(GetDataUInt64());
        case GPUColumnTypeId::FLOAT32:
            return reinterpret_cast<uint8_t*>(GetDataFloat32());
        case GPUColumnTypeId::FLOAT64:
            return reinterpret_cast<uint8_t*>(GetDataFloat64());
        case GPUColumnTypeId::BOOLEAN:
            return reinterpret_cast<uint8_t*>(GetDataBoolean());
        case GPUColumnTypeId::VARCHAR:
            return reinterpret_cast<uint8_t*>(GetDataVarChar());
        default:
            return nullptr;
    }
}

} // namespace duckdb