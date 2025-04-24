#include "gpu_columns.hpp"

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
    // if (data == nullptr) isNull = 1;
    // else isNull = 0;
    is_unique = false;
}

GPUColumn::GPUColumn(string _name, size_t _column_length, ColumnType type, uint8_t* data) {
    name = _name;
    column_length = _column_length;
    data_wrapper = DataWrapper(type, data, _column_length);
    row_ids = nullptr;
    data_wrapper.offset = nullptr;
    data_wrapper.num_bytes = column_length * data_wrapper.getColumnTypeSize();
    // if (data == nullptr) isNull = 1;
    // else isNull = 0;
    is_unique = false;
}

GPUColumn::GPUColumn(string _name, size_t _column_length, ColumnType type, uint8_t* data, uint64_t* offset, size_t num_bytes, bool is_string_data) {
    name = _name;
    column_length = _column_length;
    data_wrapper = DataWrapper(type, data, offset, _column_length, num_bytes, is_string_data);
    row_ids = nullptr;
    // isString = is_string_data;
    if (is_string_data) {
        data_wrapper.num_bytes = num_bytes;
    } else {
        data_wrapper.num_bytes = column_length * data_wrapper.getColumnTypeSize();
    }
    is_unique = false;
}

GPUColumn::GPUColumn(size_t _column_length, ColumnType type, uint8_t* data, uint64_t* offset, size_t num_bytes, bool is_string_data) {
    column_length = _column_length;
    data_wrapper = DataWrapper(type, data, offset, _column_length, num_bytes, is_string_data);
    row_ids = nullptr;
    // isString = is_string_data;
    if (is_string_data) {
        data_wrapper.num_bytes = num_bytes;
    } else {
        data_wrapper.num_bytes = column_length * data_wrapper.getColumnTypeSize();
    }
    is_unique = false;
}

GPUColumn::GPUColumn(const GPUColumn& other) {
    name = other.name;
    data_wrapper = other.data_wrapper;
    row_ids = other.row_ids;
    row_id_count = other.row_id_count;
    column_length = other.column_length;
    // isString = other.isString;
    is_unique = other.is_unique;
}

cudf::column_view
GPUColumn::convertToCudfColumn() {
    if (data_wrapper.type == ColumnType::INT64) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::UINT64), column_length, data_wrapper.data, nullptr, 0);
        return column;
    } else if (data_wrapper.type == ColumnType::INT32) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::INT32), column_length, data_wrapper.data, nullptr, 0);
        return column;
    } else if (data_wrapper.type == ColumnType::FLOAT32) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::FLOAT32), column_length, data_wrapper.data, nullptr, 0);
        return column;
    } else if (data_wrapper.type == ColumnType::FLOAT64) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::FLOAT64), column_length, data_wrapper.data, nullptr, 0);
        return column;
    } else if (data_wrapper.type == ColumnType::BOOLEAN) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::BOOL8), column_length, data_wrapper.data, nullptr, 0);
        return column;
    } else if (data_wrapper.type == ColumnType::VARCHAR) {

        //convert offset to int32
        int32_t* new_offset = convertSiriusOffsetToCudfOffset();

        auto offsets_col = cudf::column_view(
            cudf::data_type{cudf::type_id::INT32},
            column_length + 1,
            new_offset,
            nullptr,
            0
        );

        std::vector<cudf::column_view> children;
        children.push_back(offsets_col);

        // Build string column
        auto str_col = cudf::column_view(
            cudf::data_type{cudf::type_id::STRING},
            column_length,
            (void*) data_wrapper.data,    // No top-level data buffer
            nullptr,    // Optional null mask
            0,                       // Null count
            0,                       // Offset
            std::move(children)
        );
        return str_col;
    }
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