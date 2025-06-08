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

enum class GPUColumnTypeId {
    INVALID = 0,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    BOOLEAN,
    DATE,
    VARCHAR,
    INT128
};

struct GPUColumnType {
public:
    GPUColumnType() : id_(GPUColumnTypeId::INVALID) {}
    explicit GPUColumnType(GPUColumnTypeId id) : id_(id) {}
    ~GPUColumnType() = default;

    inline GPUColumnTypeId id() const {
        return id_;
    }

private:
    GPUColumnTypeId id_;
};

inline GPUColumnType convertLogicalTypeToColumnType(LogicalType type) {
    switch (type.id()) {
        case LogicalTypeId::INTEGER:
            return GPUColumnType(GPUColumnTypeId::INT32);
        case LogicalTypeId::BIGINT:
            return GPUColumnType(GPUColumnTypeId::INT64);
        case LogicalTypeId::FLOAT:
            return GPUColumnType(GPUColumnTypeId::FLOAT32);
        case LogicalTypeId::DOUBLE:
            return GPUColumnType(GPUColumnTypeId::FLOAT64);
        case LogicalTypeId::BOOLEAN:
            return GPUColumnType(GPUColumnTypeId::BOOLEAN);
        case LogicalTypeId::DATE:
            return GPUColumnType(GPUColumnTypeId::DATE);
        case LogicalTypeId::VARCHAR:
            return GPUColumnType(GPUColumnTypeId::VARCHAR);
        default:
            throw InvalidInputException("Unsupported duckdb column type in `convertLogicalTypeToColumnType`: %d",
                                        static_cast<int>(type.id()));
    }
}

class DataWrapper {
public:
    DataWrapper() = default; // Add default constructor
    DataWrapper(GPUColumnType type, uint8_t* data, size_t size);
    DataWrapper(GPUColumnType type, uint8_t* data, uint64_t* offset, size_t size, size_t num_bytes, bool is_string_data);
	GPUColumnType type;
	uint8_t* data;
    size_t size; // number of rows in the column (currently equals to column_length)
    uint64_t* offset{nullptr};
    size_t num_bytes; // number of bytes in the column
    size_t getColumnTypeSize();
    bool is_string_data{false};
};

class GPUColumn {
public:
    GPUColumn(size_t column_length, GPUColumnType type, uint8_t* data);
    GPUColumn(size_t _column_length, GPUColumnType type, uint8_t* data, uint64_t* offset, size_t num_bytes, bool is_string_data);
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