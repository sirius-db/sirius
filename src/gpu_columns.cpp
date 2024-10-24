#include "gpu_columns.hpp"

namespace duckdb {

ColumnType convertLogicalTypetoColumnType(LogicalType type) {
    ColumnType column_type;
    switch (type.id()) {
        case LogicalTypeId::INTEGER:
            column_type = ColumnType::INT32;
            break;
        case LogicalTypeId::BIGINT:
            column_type = ColumnType::INT64;
            break;
        case LogicalTypeId::FLOAT:
            column_type = ColumnType::FLOAT32;
            break;
        case LogicalTypeId::DOUBLE:
            column_type = ColumnType::FLOAT64;
            break;
        case LogicalTypeId::VARCHAR:
            column_type = ColumnType::VARCHAR;
            break;
        default:
            column_type = ColumnType::INT32;
            break;
    }
    return column_type;
}

DataWrapper::DataWrapper(ColumnType type, uint8_t* data, size_t size) : data(data), size(size), type(type) {};

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
        case ColumnType::VARCHAR:
            return 128;
    }
    return 0;
}

GPUColumn::GPUColumn(size_t _column_length, ColumnType type, uint8_t* data) {
    column_length = _column_length;
    data_wrapper = DataWrapper(type, data, _column_length);
    row_ids = nullptr;
}

GPUColumn::GPUColumn(string _name, size_t _column_length, ColumnType type, uint8_t* data) {
    name = _name;
    column_length = _column_length;
    data_wrapper = DataWrapper(type, data, _column_length);
    row_ids = nullptr;
}

// DataChunk
// GPUColumn::ConvertToChunk() {
//     name = _name;
//     column_length = _column_length;
//     data_wrapper = DataWrapper(type, data, _column_length);
//     row_ids = nullptr;
// }


// GPUIntermediateRelation::GPUIntermediateRelation(size_t length, size_t column_count) :
//         length(length), column_count(column_count) {
//     column_names.resize(column_count);
//     columns.resize(column_count);
//     for (int i = 0; i < column_count; i++) {
//         columns[i] = nullptr;
//     }
// }

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
    // if (columns[idx]) {
    //     printf("Im here\n");
    //     if (columns[idx]->row_ids != nullptr) {
    //         printf("Im even more here\n");
    //     }
    // }
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
        case ColumnType::VARCHAR:
            return reinterpret_cast<uint8_t*>(GetDataVarChar());
        default:
            return nullptr;
    }
}

} // namespace duckdb