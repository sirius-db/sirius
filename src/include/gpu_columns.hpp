#pragma once

#include "helper/common.h"
#include "duckdb/common/types.hpp"

using namespace std;

namespace duckdb {

enum class ColumnType {
    INT32 = 0,
    INT64,
    FLOAT32,
    FLOAT64,
    BOOLEAN,
    VARCHAR
};

inline ColumnType convertLogicalTypeToColumnType(LogicalType type) {
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
        case LogicalTypeId::BOOLEAN:
            column_type = ColumnType::BOOLEAN;
            break;
        case LogicalTypeId::VARCHAR:
            column_type = ColumnType::VARCHAR;
            break;
        default:
            throw InvalidInputException("Unsupported column type");
            break;
    }
    return column_type;
}

class DataWrapper {
public:
    DataWrapper() = default; // Add default constructor
    DataWrapper(ColumnType type, uint8_t* data, size_t size);
    DataWrapper(ColumnType type, uint8_t* data, uint64_t* offset, size_t size, size_t num_bytes, bool is_string_data);
	ColumnType type;
	uint8_t* data;
    size_t size;
    uint64_t* offset{nullptr};
    size_t num_bytes;
    size_t getColumnTypeSize();
    bool is_string_data{false};
};

class GPUColumn {
public:
    GPUColumn(string name, size_t column_length, ColumnType type, uint8_t* data);
    GPUColumn(size_t column_length, ColumnType type, uint8_t* data);
    GPUColumn(string _name, size_t _column_length, ColumnType type, uint8_t* data, uint64_t* offset, size_t num_bytes, bool is_string_data);
    GPUColumn(size_t _column_length, ColumnType type, uint8_t* data, uint64_t* offset, size_t num_bytes, bool is_string_data);
    GPUColumn(const GPUColumn& other);
    ~GPUColumn(){};
    int* GetDataInt32();
    uint64_t* GetDataUInt64();
    float* GetDataFloat32();
    double* GetDataFloat64();
    char* GetDataVarChar();
    uint8_t* GetDataBoolean();
    uint64_t* GetRowIds();
    uint8_t* GetData();

    string name;
    DataWrapper data_wrapper;
    uint64_t* row_ids;
    size_t row_id_count;
    size_t column_length;
    // bool isNull;
    bool is_unique;
    // bool isString{false};
};

class GPUIntermediateRelation {
public:
    // GPUIntermediateRelation(size_t length, size_t column_count);
    GPUIntermediateRelation(size_t column_count);
    ~GPUIntermediateRelation(){};
    bool checkLateMaterialization(size_t idx);

    string names;
	vector<string> column_names;
    vector<GPUColumn*> columns;
    // map<string, GPUColumn*> columns;
    size_t length;
    size_t column_count;
};


} // namespace duckdb   