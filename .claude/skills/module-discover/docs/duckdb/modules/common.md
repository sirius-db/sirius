# common

**Status**: USED
**Path**: `duckdb/src/include/duckdb/common/`
**Headers we include**:
- `duckdb/common/assert.hpp`
- `duckdb/common/enums/expression_type.hpp`
- `duckdb/common/enums/logical_operator_type.hpp`
- `duckdb/common/enums/physical_operator_type.hpp`
- `duckdb/common/helper.hpp`
- `duckdb/common/types.hpp`
- `duckdb/common/types/column/column_data_collection.hpp`
- `duckdb/common/types/data_chunk.hpp`
- `duckdb/common/types/decimal.hpp`
- `duckdb/common/types/validity_mask.hpp`
- `duckdb/common/types/value.hpp`
- `duckdb/common/types/vector.hpp`
- `duckdb/common/vector_size.hpp`
- `duckdb/common/multi_file/multi_file_states.hpp`

## Summary

The `common` module is the largest DuckDB module and provides foundational types used throughout the system: data chunks, vectors, type system, values, enums, exceptions, and utility functions. Sirius uses it extensively for data exchange between CPU (DuckDB) and GPU (cuDF), type mapping, and expression type identification.

## API Reference

### DataChunk

**Header**: `duckdb/common/types/data_chunk.hpp`
**Signature**:
```cpp
class DataChunk {
public:
    vector<Vector> data;
    idx_t size() const;
    idx_t ColumnCount() const;
    void SetCardinality(idx_t count);
    void SetCardinality(const DataChunk &other);
    void Initialize(Allocator &allocator, const vector<LogicalType> &types, idx_t capacity = STANDARD_VECTOR_SIZE);
    void Reset();
    void Flatten();
    void Reference(DataChunk &chunk);
};
```

**Description**: A columnar data batch — the fundamental unit of data exchange in DuckDB. Each `DataChunk` contains a set of `Vector` columns with a shared row count. Default capacity is `STANDARD_VECTOR_SIZE` (2048).

**Our usage**:
- `src/op/scan/duckdb_scan_task.cpp` — Receives scanned data from DuckDB as DataChunks, converts to GPU format
- `src/op/result/host_table_chunk_reader.cpp` — Converts GPU results back to DataChunks for DuckDB
- `src/operator/gpu_physical_table_scan.cpp` — Scans data into DataChunks
- `src/sirius_extension.cpp` — Used in table function execute callbacks

### Vector

**Header**: `duckdb/common/types/vector.hpp`
**Signature**:
```cpp
class Vector {
public:
    Vector(LogicalType type, idx_t capacity = STANDARD_VECTOR_SIZE);
    VectorType GetVectorType() const;
    LogicalType &GetType();
    data_ptr_t GetData();
    void SetVectorType(VectorType vector_type);
    void Reference(const Value &value);
    void Flatten(idx_t count);
};
```

**Description**: A single column of data. Can be flat (contiguous), constant, dictionary-encoded, or sequence. Sirius primarily works with flat vectors.

**Our usage**:
- `src/op/result/host_table_chunk_reader.cpp` — Access raw vector data for GPU→CPU transfer
- `src/gpu_columns.cpp` — Read vector data to build GPU columns

### FlatVector

**Header**: `duckdb/common/types/vector.hpp`
**Signature**:
```cpp
struct FlatVector {
    static data_ptr_t GetData(Vector &vector);
    static const ValidityMask &Validity(const Vector &vector);
    static void SetValidity(Vector &vector, ValidityMask &new_mask);
    static bool IsNull(const Vector &vector, idx_t idx);
    static void SetNull(Vector &vector, idx_t idx, bool is_null);
};

struct StringVector {
    static void AddBuffer(Vector &vector, buffer_ptr<VectorBuffer> buffer);
};
```

**Description**: Static helper for accessing flat (contiguous) vector data. Primary interface for reading/writing raw column data.

**Our usage**:
- `src/op/result/host_table_chunk_reader.cpp` — `FlatVector::GetData()` to get raw pointers, `FlatVector::Validity()` for null masks
- `src/gpu_columns.cpp` — Read raw data from vectors
- `test/cpp/memory/test_host_table_utils.cpp` — Test vector data access

### ValidityMask

**Header**: `duckdb/common/types/validity_mask.hpp`
**Signature**:
```cpp
class ValidityMask {
public:
    validity_t *GetData() const;
    bool RowIsValid(idx_t row_idx) const;
    void SetInvalid(idx_t row_idx);
    bool AllValid() const;
    idx_t CountValid(idx_t count) const;
    static constexpr idx_t BITS_PER_VALUE = sizeof(validity_t) * 8;  // 64
    static constexpr idx_t STANDARD_MASK_SIZE = STANDARD_VECTOR_SIZE / BITS_PER_VALUE;
};
```

**Description**: Bitmask tracking NULL values in a vector. Each bit represents one row (1 = valid, 0 = NULL).

**Our usage**:
- `src/op/result/host_table_chunk_reader.cpp` — Convert between DuckDB validity masks and cuDF null bitmasks
- `test/cpp/memory/test_host_table_utils.cpp` — Test null handling

### LogicalType / LogicalTypeId

**Header**: `duckdb/common/types.hpp`
**Signature**:
```cpp
enum class LogicalTypeId : uint8_t {
    BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT, UTINYINT, USMALLINT, UINTEGER, UBIGINT,
    FLOAT, DOUBLE, DATE, TIMESTAMP, TIMESTAMP_NS, TIMESTAMP_MS, TIMESTAMP_SEC,
    VARCHAR, BLOB, DECIMAL, HUGEINT, INTERVAL, STRUCT, LIST, MAP, ...
};

class LogicalType {
public:
    LogicalTypeId id() const;
    PhysicalType InternalType() const;
    bool operator==(const LogicalType &rhs) const;
    static LogicalType INTEGER, BIGINT, FLOAT, DOUBLE, VARCHAR, BOOLEAN, DATE, TIMESTAMP;
};

struct DecimalType {
    static uint8_t GetWidth(const LogicalType &type);
    static uint8_t GetScale(const LogicalType &type);
};
```

**Description**: DuckDB's type system. `LogicalTypeId` identifies the SQL type, `LogicalType` wraps it with extra info (e.g., decimal precision). Sirius maps these to cuDF data types.

**Our usage**:
- `src/gpu_columns.cpp` — Map DuckDB types to cuDF types
- `src/expression_executor/gpu_expression_translator.cpp` — Determine GPU operation types from expression types
- `src/planner/sirius_physical_plan_generator.cpp` — Check supported types for GPU execution
- `src/fallback.cpp` — Check type support to decide CPU vs GPU execution

### Value

**Header**: `duckdb/common/types/value.hpp`
**Signature**:
```cpp
class Value {
public:
    LogicalType type() const;
    static Value BOOLEAN(bool value);
    static Value INTEGER(int32_t value);
    static Value BIGINT(int64_t value);
    static Value UBIGINT(uint64_t value);
    static Value MinimumValue(const LogicalType &type);
    static Value MaximumValue(const LogicalType &type);
    template <class T> T GetValue() const;
    bool IsNull() const;
};

struct BooleanValue { static bool Get(const Value &value); };
struct StringValue { static const string &Get(const Value &value); };
struct IntegerValue { static int32_t Get(const Value &value); };
struct UBigIntValue { static uint64_t Get(const Value &value); };
```

**Description**: Type-safe variant holding a single DuckDB value. Used for constants, configuration values, and statistics.

**Our usage**:
- `src/sirius_extension.cpp` — Extract function arguments (e.g., SQL query string, memory sizes)
- `src/expression_executor/gpu_expression_translator.cpp` — Extract constant values for GPU expressions
- `src/operator/gpu_physical_table_scan.cpp` — Build filter constants

### ColumnDataCollection

**Header**: `duckdb/common/types/column/column_data_collection.hpp`
**Signature**:
```cpp
class ColumnDataCollection {
public:
    ColumnDataCollection(Allocator &allocator, vector<LogicalType> types);
    void Append(DataChunk &chunk);
    idx_t Count() const;
    const vector<LogicalType> &Types() const;
    void InitializeScan(ColumnDataScanState &state) const;
    void Scan(ColumnDataScanState &state, DataChunk &result) const;
};
```

**Description**: In-memory columnar storage that can hold arbitrary amounts of data (unlike DataChunk which is limited to STANDARD_VECTOR_SIZE). Used for materialization.

**Our usage**:
- `src/plan/gpu_plan_recursive_cte.cpp` — CTE materialization
- `src/operator/gpu_physical_table_scan.cpp` — Collect scanned data

### ExpressionType / PhysicalOperatorType / LogicalOperatorType (Enums)

**Header**: `duckdb/common/enums/expression_type.hpp`, `duckdb/common/enums/physical_operator_type.hpp`, `duckdb/common/enums/logical_operator_type.hpp`

**Key values used**:
```cpp
// Expression types
ExpressionType::COMPARE_EQUAL, COMPARE_NOTEQUAL, COMPARE_LESSTHAN, COMPARE_GREATERTHAN,
COMPARE_LESSTHANOREQUALTO, COMPARE_GREATERTHANOREQUALTO, COMPARE_DISTINCT_FROM,
COMPARE_NOT_DISTINCT_FROM, COMPARE_IN, COMPARE_NOT_IN,
CONJUNCTION_AND, CONJUNCTION_OR,
OPERATOR_COALESCE, OPERATOR_NOT, OPERATOR_IS_NULL, OPERATOR_IS_NOT_NULL, OPERATOR_TRY,
VALUE_CONSTANT, BOUND_REF, BOUND_FUNCTION, BOUND_AGGREGATE, BOUND_CAST

// Physical operator types
PhysicalOperatorType::FILTER, HASH_JOIN, HASH_GROUP_BY, UNGROUPED_AGGREGATE,
ORDER_BY, TOP_N, TABLE_SCAN, PROJECTION, NESTED_LOOP_JOIN, DELIM_JOIN, ...

// Logical operator types
LogicalOperatorType::LOGICAL_GET, LOGICAL_FILTER, LOGICAL_AGGREGATE_AND_GROUP_BY,
LOGICAL_ORDER_BY, LOGICAL_TOP_N, LOGICAL_LIMIT, LOGICAL_WINDOW, LOGICAL_CTE, ...
```

**Our usage**:
- `src/expression_executor/gpu_expression_translator.cpp` — Switch on expression types to dispatch GPU evaluation
- `src/gpu_physical_plan_generator.cpp` — Switch on physical/logical operator types to create GPU operators
- `src/planner/sirius_physical_plan_generator.cpp` — Same as above for new Sirius code path
- `src/fallback.cpp` — Check operator types for fallback decisions

### Exceptions

**Headers**: `duckdb/common/exception.hpp` and sub-headers
```cpp
class InvalidInputException : public Exception { ... };
class BinderException : public Exception { ... };
class NotImplementedException : public Exception { ... };
class InternalException : public Exception { ... };
```

**Our usage**:
- Throughout codebase for error handling when GPU operations fail or encounter unsupported features

### Utility Functions

**Header**: `duckdb/common/helper.hpp`
```cpp
template <class T, class... ARGS> unique_ptr<T> make_uniq(ARGS&&... args);
template <class T, class... ARGS> shared_ptr<T> make_shared_ptr(ARGS&&... args);
template <class SRC, class TGT> TGT &Cast(SRC &source);
```

**Our usage**:
- Throughout codebase — `make_uniq<>()` for creating unique pointers, `Cast<>()` for safe downcasting

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `ArrowWrapper` | `common/arrow/` | Arrow format interop |
| `Serializer/Deserializer` | `common/serializer/` | Binary serialization |
| `RowOperations` | `common/row_operations/` | Tuple-format row operations |
| `ProgressBar` | `common/progress_bar/` | Query progress tracking |
| `VectorOperations` | `common/vector_operations/` | Vectorized operations (mostly; `Cast()` is used) |
| `Crypto` | `common/crypto/` | Hashing and encryption utilities |
