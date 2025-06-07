#pragma once

#include "helper/common.h"
#include "duckdb/common/types.hpp"
#include "cudf_utils.hpp"
using namespace std;

namespace duckdb {

class GPUBufferManager;

int32_t* convertUInt64ToInt32(uint64_t* data, size_t N);
uint64_t* convertInt32ToUInt64(int32_t* data, size_t N);
template <typename T> void callCudaMemcpyHostToDevice(T* dest, T* src, size_t size, int gpu);
template <typename T> void callCudaMemcpyDeviceToHost(T* dest, T* src, size_t size, int gpu);
template <typename T> void callCudaMemcpyDeviceToDevice(T* dest, T* src, size_t size, int gpu);

enum class ColumnType {
    INT32 = 0,
    INT64,
    FLOAT32,
    FLOAT64,
    BOOLEAN,
    VARCHAR,
    INT128
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
    size_t size; // number of rows in the column (currently equals to column_length)
    uint64_t* offset{nullptr};
    size_t num_bytes; // number of bytes in the column
    size_t getColumnTypeSize();
    bool is_string_data{false};
};

class GPUColumn {
public:
    GPUColumn(size_t column_length, ColumnType type, uint8_t* data);
    GPUColumn(size_t _column_length, ColumnType type, uint8_t* data, uint64_t* offset, size_t num_bytes, bool is_string_data);
    GPUColumn(GPUColumn& other);
    ~GPUColumn(){};
    int* GetDataInt32();
    uint64_t* GetDataUInt64();
    float* GetDataFloat32();
    double* GetDataFloat64();
    char* GetDataVarChar();
    uint8_t* GetDataBoolean();
    uint64_t* GetRowIds();
    uint8_t* GetData();

    DataWrapper data_wrapper;
    uint64_t* row_ids;
    size_t row_id_count; // number of rows in the row_ids array
    size_t column_length; // number of rows in the column (currently equals to column_length)
    bool is_unique; // indicator whether the column has unique values

    cudf::column_view convertToCudfColumn();
    int32_t* convertSiriusOffsetToCudfOffset(); // convert the offset of GPUColumn to the offset of the cudf column
    int32_t* convertSiriusRowIdsToCudfRowIds(); // convert the row_ids of the GPUColumn to the row_ids of the cudf column
    void convertCudfRowIdsToSiriusRowIds(int32_t* cudf_row_ids); // convert the row_ids of the cudf column to the row_ids of the GPUColumn
    void convertCudfOffsetToSiriusOffset(int32_t* cudf_offset); // convert the offset of the cudf column to the offset of the GPUColumn
    void setFromCudfColumn(cudf::column& cudf_column, bool _is_unique, int32_t* _row_ids, uint64_t _row_id_count, GPUBufferManager* gpuBufferManager);
    void setFromCudfScalar(cudf::scalar& cudf_scalar, GPUBufferManager* gpuBufferManager); // set the GPUColumn from the cudf scalar
};

class GPUIntermediateRelation {
public:
    GPUIntermediateRelation(size_t column_count);
    ~GPUIntermediateRelation(){};
    bool checkLateMaterialization(size_t idx);

    string names;
	vector<string> column_names;
    vector<shared_ptr<GPUColumn>> columns;
    size_t column_count;
};



} // namespace duckdb   