# Legacy Mode Data Structures

This document describes the core data structures used by Sirius Legacy Mode to represent columnar data on the GPU.

## Table of Contents

1. [Overview](#overview)
2. [GPUColumnType](#gpucolumntype)
3. [DataWrapper](#datawrapper)
4. [GPUColumn](#gpucolumn)
5. [GPUIntermediateRelation](#gpuintermediaterelation)
6. [Late Materialization](#late-materialization)
7. [cuDF Integration](#cudf-integration)
8. [Memory Layout](#memory-layout)
9. [Next Steps](#next-steps)

---

## Overview

Legacy Mode represents columnar data using three primary structures:

1. **GPUColumnType**: Type metadata (INT32, VARCHAR, etc.)
2. **DataWrapper**: Raw GPU memory pointers + validity masks
3. **GPUColumn**: Complete column with data, row IDs, and metadata
4. **GPUIntermediateRelation**: Collection of columns (like a table)

These structures enable:

- **Columnar storage**: Efficient GPU processing
- **Lazy materialization**: Deferred row gathering for filter chains
- **NULL support**: Validity masks for nullable columns
- **cuDF interop**: Easy conversion to/from cuDF tables

**File**: `src/include/gpu_columns.hpp`

---

## GPUColumnType

**GPUColumnType** represents the data type of a column, mapping DuckDB's `LogicalType` to GPU-friendly types.

### Type Enumeration

**File**: `src/include/gpu_columns.hpp:58-74`

```cpp
enum class GPUColumnTypeId {
    INVALID = 0,
    INT16,              // SMALLINT
    INT32,              // INTEGER
    INT64,              // BIGINT
    FLOAT32,            // FLOAT
    FLOAT64,            // DOUBLE
    BOOLEAN,            // BOOLEAN
    DATE,               // DATE (days since epoch)
    TIMESTAMP_SEC,      // TIMESTAMP (seconds)
    TIMESTAMP_MS,       // TIMESTAMP (milliseconds)
    TIMESTAMP_US,       // TIMESTAMP (microseconds)
    TIMESTAMP_NS,       // TIMESTAMP (nanoseconds)
    VARCHAR,            // Variable-length string
    INT128,             // HUGEINT (128-bit integer)
    DECIMAL             // Fixed-point decimal
};
```

### GPUColumnType Class

```cpp
struct GPUColumnType {
public:
    GPUColumnType() : id_(GPUColumnTypeId::INVALID) {}
    explicit GPUColumnType(GPUColumnTypeId id) : id_(id) {}

    inline GPUColumnTypeId id() const { return id_; }

    // For DECIMAL types
    inline GPUDecimalTypeInfo* GetDecimalTypeInfo() const {
        return decimal_type_info_.get();
    }

    void SetDecimalTypeInfo(uint8_t width, uint8_t scale) {
        decimal_type_info_ = make_shared_ptr<GPUDecimalTypeInfo>(width, scale);
    }

private:
    GPUColumnTypeId id_;
    shared_ptr<GPUDecimalTypeInfo> decimal_type_info_;  // Only for DECIMAL
};
```

### Type Conversions

**DuckDB LogicalType → GPUColumnType:**

```cpp
GPUColumnType convertLogicalTypeToColumnType(LogicalType type) {
    switch (type.id()) {
        case LogicalTypeId::INTEGER:
            return GPUColumnType(GPUColumnTypeId::INT32);
        case LogicalTypeId::VARCHAR:
            return GPUColumnType(GPUColumnTypeId::VARCHAR);
        case LogicalTypeId::DECIMAL:
            GPUColumnType column_type(GPUColumnTypeId::DECIMAL);
            column_type.SetDecimalTypeInfo(
                DecimalType::GetWidth(type),
                DecimalType::GetScale(type));
            return column_type;
        // ... other types
    }
}
```

**GPUColumnType → DuckDB LogicalType:**

```cpp
LogicalType convertColumnTypeToLogicalType(const GPUColumnType& type) {
    switch (type.id()) {
        case GPUColumnTypeId::INT32:
            return LogicalType::INTEGER;
        case GPUColumnTypeId::VARCHAR:
            return LogicalType::VARCHAR;
        case GPUColumnTypeId::DECIMAL:
            auto* info = type.GetDecimalTypeInfo();
            return LogicalType::DECIMAL(info->width_, info->scale_);
        // ... other types
    }
}
```

---

## DataWrapper

**DataWrapper** encapsulates raw GPU memory pointers for a column's data.

### Structure

**File**: `src/include/gpu_columns.hpp:167-188`

```cpp
class DataWrapper {
public:
    // Fixed-width types (INT32, FLOAT64, etc.)
    DataWrapper(GPUColumnType type,
                uint8_t* data,
                size_t size,
                cudf::bitmask_type* validity_mask);

    // Variable-width types (VARCHAR)
    DataWrapper(GPUColumnType type,
                uint8_t* data,
                uint64_t* offset,       // String start offsets
                size_t size,            // Number of strings
                size_t num_bytes,       // Total string bytes
                bool is_string_data,
                cudf::bitmask_type* validity_mask);

    GPUColumnType type;                 // Column type
    uint8_t* data;                      // GPU data pointer
    size_t size;                        // Number of rows
    uint64_t* offset;                   // String offsets (VARCHAR only)
    size_t num_bytes;                   // String data bytes (VARCHAR only)
    bool is_string_data;                // Whether this is a VARCHAR column
    cudf::bitmask_type* validity_mask;  // NULL mask (1 bit per row)
    size_t mask_bytes;                  // Size of validity mask

    size_t getColumnTypeSize() const;   // Returns sizeof(type)
};
```

### Fixed-Width vs Variable-Width

**Fixed-Width Types (INT32, FLOAT64, DATE, etc.):**

```
data: [42, 17, 99, 123, 5, ...]  (contiguous array)
validity_mask: [1, 1, 0, 1, 1, ...]  (1 = valid, 0 = NULL)
```

**Variable-Width Types (VARCHAR):**

```
data: ['H', 'e', 'l', 'l', 'o', 'W', 'o', 'r', 'l', 'd', ...]  (all characters concatenated)
offset: [0, 5, 10, ...]  (start position of each string)
validity_mask: [1, 1, ...]
```

**Example VARCHAR Column:**

```
Strings: ["Hello", "World", "GPU"]

data:     ['H', 'e', 'l', 'l', 'o', 'W', 'o', 'r', 'l', 'd', 'G', 'P', 'U']
          ^ 0                    ^ 5                    ^ 10       ^ 13
offset:   [0, 5, 10, 13]
size:     3  (3 strings)
num_bytes: 13  (13 characters)
```

### Validity Masks

**NULL values are represented using a bitmask:**

```cpp
// Allocate validity mask (1 bit per row, rounded to 64-byte alignment)
size_t mask_bytes = getMaskBytesSize(column_length);
cudf::bitmask_type* mask = createNullMask(
    column_length,
    cudf::mask_state::ALL_VALID  // Start with all valid
);

// Set row 5 to NULL
cudf::set_bit_unsafe(mask, 5, false);
```

**Mask Layout (32-bit bitmask units):**

```
Rows:  [0-31]  [32-63]  [64-95]  [96-127]  ...
Mask:  [0xFFFFFFFF]  [0xFFFFFFFF]  [0xFFFFFFFF]  [0xFFFFFFFF]  ...
       (all valid)
```

---

## GPUColumn

**GPUColumn** represents a complete column with data, metadata, and optional row indirections.

### Structure

**File**: `src/include/gpu_columns.hpp:190-250`

```cpp
class GPUColumn {
public:
    // Fixed-width constructor
    GPUColumn(size_t column_length,
              GPUColumnType type,
              uint8_t* data,
              cudf::bitmask_type* validity_mask);

    // Variable-width constructor
    GPUColumn(size_t column_length,
              GPUColumnType type,
              uint8_t* data,
              uint64_t* offset,
              size_t num_bytes,
              bool is_string_data,
              cudf::bitmask_type* validity_mask);

    // Copy constructor (shallow copy, shares data pointers)
    GPUColumn(shared_ptr<GPUColumn> other);

    ~GPUColumn();

    // Data access methods
    int* GetDataInt32();
    uint64_t* GetDataUInt64();
    float* GetDataFloat32();
    double* GetDataFloat64();
    char* GetDataVarChar();
    uint8_t* GetDataBoolean();
    uint64_t* GetRowIds();
    uint8_t* GetData();

    // Core fields
    DataWrapper data_wrapper;       // Raw GPU memory
    uint64_t* row_ids;              // Row indirection (for late materialization)
    size_t row_id_count;            // Number of row IDs
    size_t column_length;           // Number of rows
    bool is_unique;                 // Whether column has unique values

    // CPU cache metadata
    uint8_t* segment_start_ptr;     // Start of CPU cache segment
    int segment_id;                 // CPU cache segment ID

    // cuDF interop
    cudf::column_view convertToCudfColumn();
    void setFromCudfColumn(cudf::column& cudf_column, ...);
    void setFromCudfScalar(cudf::scalar& cudf_scalar, ...);

    // Memory size
    size_t getTotalColumnSize();    // Returns total bytes (data + mask + offsets)
};
```

### Memory Ownership

**GPUColumn is typically wrapped in `shared_ptr` for automatic memory management:**

```cpp
// Create column with GPU-allocated data
auto column = make_shared_ptr<GPUColumn>(
    row_count,
    GPUColumnType(GPUColumnTypeId::INT32),
    gpu_data_ptr,        // Allocated via GPUBufferManager
    validity_mask        // Allocated via createNullMask
);

// Column is automatically freed when last shared_ptr is destroyed
// Destructor calls GPUBufferManager::customCudaFree() for all pointers
```

### Late Materialization

**GPUColumn supports lazy row gathering via `row_ids`:**

**Example: Filter Chain**

```sql
SELECT * FROM users WHERE age > 25 AND salary > 50000;
```

**Without Late Materialization:**

```
Filter 1 (age > 25):
  Input:  [1000 rows × 10 columns]
  Output: [300 rows × 10 columns]  → Full gather (expensive!)

Filter 2 (salary > 50000):
  Input:  [300 rows × 10 columns]
  Output: [150 rows × 10 columns]  → Full gather (expensive!)
```

**With Late Materialization:**

```
Filter 1 (age > 25):
  Input:  [1000 rows × 10 columns]
  Output: [300 row_ids]  → Only compute row IDs (cheap!)
  row_ids = [5, 12, 17, 23, ...]

Filter 2 (salary > 50000):
  Input:  [300 row_ids × 10 columns]
  Output: [150 row_ids]  → Only filter row IDs (cheap!)
  row_ids = [12, 23, 45, ...]

Final Materialization:
  Gather 150 rows × 10 columns using row_ids [12, 23, 45, ...]
```

**Checking for Late Materialization:**

```cpp
bool GPUIntermediateRelation::checkLateMaterialization(size_t col_idx) {
    auto& column = columns[col_idx];
    return (column->row_ids != nullptr && column->row_id_count > 0);
}
```

**Materializing Rows:**

```cpp
// If column has row_ids, gather actual data
if (checkLateMaterialization(col_idx)) {
    T* materialized_data;
    materializeExpression<T>(
        column->data_wrapper.data,  // Source data
        materialized_data,          // Output buffer
        column->row_ids,            // Row indices
        column->row_id_count,       // Number of rows
        column->data_wrapper.validity_mask,  // Source mask
        out_mask                    // Output mask
    );

    // Replace column with materialized data
    column->data_wrapper.data = reinterpret_cast<uint8_t*>(materialized_data);
    column->row_ids = nullptr;
    column->row_id_count = 0;
}
```

---

## GPUIntermediateRelation

**GPUIntermediateRelation** represents a collection of columns (analogous to a table or batch).

### Structure

**File**: `src/include/gpu_columns.hpp:252-262`

```cpp
class GPUIntermediateRelation {
public:
    GPUIntermediateRelation(size_t column_count);
    ~GPUIntermediateRelation();

    bool checkLateMaterialization(size_t col_idx);

    string names;                               // Relation name (optional)
    vector<string> column_names;                // Column names
    vector<shared_ptr<GPUColumn>> columns;      // Columns
    size_t column_count;                        // Number of columns
};
```

### Usage Example

**Creating a Relation:**

```cpp
// Allocate relation with 3 columns
GPUIntermediateRelation relation(3);

// Add columns
relation.columns.push_back(make_shared_ptr<GPUColumn>(...));  // Column 0: id (INT32)
relation.columns.push_back(make_shared_ptr<GPUColumn>(...));  // Column 1: name (VARCHAR)
relation.columns.push_back(make_shared_ptr<GPUColumn>(...));  // Column 2: age (INT32)

// Set column names
relation.column_names = {"id", "name", "age"};
```

**Operator Data Flow:**

```cpp
OperatorResultType GPUPhysicalFilter::Execute(
    GPUIntermediateRelation& input_relation,
    GPUIntermediateRelation& output_relation) const {

    // Input: 1000 rows × 3 columns
    SIRIUS_LOG_DEBUG("Input: {} rows, {} columns",
                    input_relation.columns[0]->column_length,
                    input_relation.column_count);

    // Apply filter: age > 25
    // ... filtering logic ...

    // Output: 300 rows × 3 columns
    SIRIUS_LOG_DEBUG("Output: {} rows, {} columns",
                    output_relation.columns[0]->column_length,
                    output_relation.column_count);

    return OperatorResultType::FINISHED;
}
```

---

## Late Materialization

### Motivation

**Problem:** Filter chains gather data multiple times.

```sql
SELECT * FROM users WHERE age > 25 AND salary > 50000 AND department = 'Engineering';
```

**Without Late Materialization:**

```
TABLE_SCAN:  1M rows → GPU
FILTER (age):  1M → 300K rows (gather all 10 columns)  ← expensive!
FILTER (salary):  300K → 150K rows (gather all 10 columns)  ← expensive!
FILTER (dept):  150K → 50K rows (gather all 10 columns)  ← expensive!
RESULT: 50K rows
```

**With Late Materialization:**

```
TABLE_SCAN:  1M rows → GPU
FILTER (age):  1M → 300K row_ids (no gather)  ← cheap!
FILTER (salary):  300K → 150K row_ids (no gather)  ← cheap!
FILTER (dept):  150K → 50K row_ids (no gather)  ← cheap!
Materialize:  Gather 50K rows × 10 columns once  ← only one gather!
RESULT: 50K rows
```

**Speedup:** 3x fewer memory operations, ~2x overall speedup.

### Implementation

**Filter Operator (Produces row_ids):**

```cpp
OperatorResultType GPUPhysicalFilter::Execute(
    GPUIntermediateRelation& input_relation,
    GPUIntermediateRelation& output_relation) const {

    // Evaluate predicate → selection vector
    uint64_t* selection_vector;
    uint64_t selected_count;
    EvaluateFilterExpression(input_relation, selection_vector, selected_count);

    // Create output columns with row_ids (no gathering yet)
    for (size_t col = 0; col < input_relation.columns.size(); col++) {
        auto& input_col = input_relation.columns[col];

        // Create column pointing to original data + row_ids
        auto output_col = make_shared_ptr<GPUColumn>(input_col);
        output_col->row_ids = selection_vector;
        output_col->row_id_count = selected_count;

        output_relation.columns.push_back(output_col);
    }

    return OperatorResultType::FINISHED;
}
```

**Result Collector (Materializes row_ids):**

```cpp
void GPUPhysicalResultCollector::FinalMaterialize(
    GPUIntermediateRelation& input_relation) {

    for (size_t col = 0; col < input_relation.columns.size(); col++) {
        if (input_relation.checkLateMaterialization(col)) {
            auto& column = input_relation.columns[col];

            // Gather rows using row_ids
            T* materialized_data = gpuBufferManager->customCudaMalloc<T>(column->row_id_count);
            materializeExpression<T>(
                column->data_wrapper.data,
                materialized_data,
                column->row_ids,
                column->row_id_count,
                column->data_wrapper.validity_mask,
                out_mask
            );

            // Replace with materialized data
            column->data_wrapper.data = reinterpret_cast<uint8_t*>(materialized_data);
            column->row_ids = nullptr;
        }
    }
}
```

---

## cuDF Integration

### GPUColumn ↔ cuDF Column Conversions

**GPUColumn → cuDF Column:**

```cpp
cudf::column_view GPUColumn::convertToCudfColumn() {
    if (data_wrapper.is_string_data) {
        // Convert VARCHAR to cuDF strings_column_view
        auto offsets_col = cudf::column_view(
            cudf::data_type{cudf::type_id::INT64},
            column_length + 1,
            reinterpret_cast<const void*>(data_wrapper.offset)
        );

        auto chars_col = cudf::column_view(
            cudf::data_type{cudf::type_id::INT8},
            data_wrapper.num_bytes,
            reinterpret_cast<const void*>(data_wrapper.data)
        );

        return cudf::strings_column_view(offsets_col, chars_col);
    } else {
        // Fixed-width type
        cudf::data_type cudf_type = ConvertGPUTypeToCudfType(data_wrapper.type);
        return cudf::column_view(
            cudf_type,
            column_length,
            reinterpret_cast<const void*>(data_wrapper.data),
            reinterpret_cast<const cudf::bitmask_type*>(data_wrapper.validity_mask),
            cudf::UNKNOWN_NULL_COUNT
        );
    }
}
```

**cuDF Column → GPUColumn:**

```cpp
void GPUColumn::setFromCudfColumn(cudf::column& cudf_column,
                                  bool _is_unique,
                                  int32_t* _row_ids,
                                  uint64_t _row_id_count,
                                  GPUBufferManager* gpuBufferManager) {
    // Extract data pointer
    data_wrapper.data = const_cast<uint8_t*>(
        reinterpret_cast<const uint8_t*>(cudf_column.view().data<uint8_t>())
    );

    // Extract validity mask
    if (cudf_column.nullable()) {
        data_wrapper.validity_mask = const_cast<cudf::bitmask_type*>(
            cudf_column.view().null_mask()
        );
    }

    // Convert offsets for VARCHAR
    if (cudf_column.type().id() == cudf::type_id::STRING) {
        convertCudfOffsetToSiriusOffset(
            const_cast<int32_t*>(cudf_column.child(0).view().data<int32_t>())
        );
    }

    // Set row IDs if provided
    if (_row_ids) {
        convertCudfRowIdsToSiriusRowIds(_row_ids);
    }
}
```

### Offset Conversions

**cuDF uses INT32 offsets, Sirius uses UINT64:**

```cpp
int32_t* GPUColumn::convertSiriusOffsetToCudfOffset() {
    // Sirius: uint64_t offsets
    // cuDF:   int32_t offsets
    auto* cudf_offset = gpuBufferManager->customCudaMalloc<int32_t>(column_length + 1);

    // Convert on GPU
    ConvertUInt64ToInt32Kernel<<<blocks, threads>>>(
        data_wrapper.offset,
        cudf_offset,
        column_length + 1
    );

    return cudf_offset;
}

void GPUColumn::convertCudfOffsetToSiriusOffset(int32_t* cudf_offset) {
    // cuDF:   int32_t offsets
    // Sirius: uint64_t offsets
    auto* sirius_offset = gpuBufferManager->customCudaMalloc<uint64_t>(column_length + 1);

    // Convert on GPU
    ConvertInt32ToUInt64Kernel<<<blocks, threads>>>(
        cudf_offset,
        sirius_offset,
        column_length + 1
    );

    data_wrapper.offset = sirius_offset;
}
```

---

## Memory Layout

### Fixed-Width Column (INT32)

```
GPUColumn {
    column_length: 1000
    row_id_count: 0
    row_ids: nullptr
    data_wrapper: {
        type: INT32
        size: 1000
        data: ┌─────────────────────────────────┐
              │ [42, 17, 99, 123, 5, 78, ...]  │ GPU memory (4000 bytes)
              └─────────────────────────────────┘
        validity_mask: ┌──────────────────┐
                       │ [0xFFFFFFFF, ...] │ GPU memory (128 bytes, 1 bit per row)
                       └──────────────────┘
    }
}
```

### Variable-Width Column (VARCHAR)

```
GPUColumn {
    column_length: 3
    row_id_count: 0
    row_ids: nullptr
    data_wrapper: {
        type: VARCHAR
        size: 3
        num_bytes: 13
        is_string_data: true
        data: ┌───────────────────────────────────────┐
              │ ['H','e','l','l','o','W','o','r',... │ GPU memory (13 bytes)
              └───────────────────────────────────────┘
        offset: ┌─────────────┐
                │ [0, 5, 10, 13] │ GPU memory (32 bytes, 4 offsets)
                └─────────────┘
        validity_mask: ┌──────────┐
                       │ [0xFFFFFFFF] │ GPU memory (4 bytes)
                       └──────────┘
    }
}
```

### Column with Late Materialization

```
GPUColumn {
    column_length: 1000
    row_id_count: 150
    row_ids: ┌───────────────────────────┐
             │ [12, 23, 45, 67, 89, ...] │ GPU memory (1200 bytes, 150 indices)
             └───────────────────────────┘
    data_wrapper: {
        type: INT32
        size: 1000
        data: ┌─────────────────────────────────┐
              │ [42, 17, 99, 123, 5, 78, ...]  │ GPU memory (4000 bytes, NOT yet gathered)
              └─────────────────────────────────┘
    }
}
```

**Note:** The `data` pointer still points to original 1000-row array, but only 150 rows (specified by `row_ids`) are valid.

---

## Next Steps

**Related Documentation:**

- **[Memory Management](memory-management.md)**: How GPU memory is allocated for these structures
- **[Operators](operators.md)**: How operators manipulate GPUColumn and GPUIntermediateRelation
- **[Pipeline Execution](pipeline-execution.md)**: How data flows through pipelines
- **[Expression Executor](../05-core-components/expression-executor.md)**: How expressions operate on GPUColumn

**Comparison:**

- **[New Mode Operators](../04-new-mode/operators.md)**: Compare with data_batch in New Mode
- **[Execution Modes](../02-architecture/execution-modes.md)**: Understand trade-offs

**For Developers:**

- **[Adding Operators](../07-development/adding-operators.md)**: Working with GPUColumn in custom operators
- **[Debugging](../07-development/debugging.md)**: Debugging data structure issues
