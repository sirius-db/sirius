#pragma once

#include "helper/common.h"
#include "duckdb/common/types.hpp"

using namespace std;

namespace duckdb {

enum class ColumnType {
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    BOOLEAN,
    VARCHAR
};

ColumnType convertLogicalTypetoColumnType(LogicalType type);

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