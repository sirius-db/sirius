# API Reference

This document provides a reference for key classes, methods, and interfaces in Sirius.

## Table of Contents

1. [Overview](#overview)
2. [New Mode API](#new-mode-api)
3. [Legacy Mode API](#legacy-mode-api)
4. [cucascade Integration](#cucascade-integration)
5. [Configuration API](#configuration-api)
6. [Expression Evaluation API](#expression-evaluation-api)
7. [Memory Management API](#memory-management-api)
8. [Logging API](#logging-api)
9. [Next Steps](#next-steps)

---

## Overview

This reference covers the most commonly used APIs in Sirius development:

- **New Mode operators** (`sirius_physical_operator`)
- **Legacy Mode operators** (`GPUPhysicalOperator`)
- **Data structures** (`data_batch`, `GPUColumn`)
- **Configuration** (`sirius_config`, `SiriusContext`)
- **Memory management** (`sirius_memory_reservation_manager`)
- **Logging** (`SIRIUS_LOG_*` macros)

For complete API documentation, see the header files in `src/include/`.

---

## New Mode API

### sirius_physical_operator

**Base class for all New Mode operators.**

**Header**: `src/include/op/sirius_physical_operator.hpp`

```cpp
class sirius_physical_operator {
public:
    // Constructor
    sirius_physical_operator(
        SiriusPhysicalOperatorType type,
        duckdb::vector<duckdb::LogicalType> types,
        duckdb::idx_t estimated_cardinality);

    // Operator metadata
    SiriusPhysicalOperatorType type;                      // Operator type
    duckdb::vector<duckdb::unique_ptr<sirius_physical_operator>> children;
    duckdb::vector<duckdb::LogicalType> types;           // Output types
    duckdb::idx_t estimated_cardinality;                 // Row count estimate
    size_t operator_id;                                  // Unique ID

    // Core methods
    virtual std::string get_name() const;
    virtual std::string to_string() const;
    void print() const;

    // Intermediate operators: Execute method
    virtual std::vector<std::shared_ptr<cucascade::data_batch>> execute(
        const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
        rmm::cuda_stream_view stream);

    // Sink operators: Accumulate data
    virtual void sink(
        std::shared_ptr<cucascade::data_batch> batch,
        rmm::cuda_stream_view stream);

    // Source operators: Produce tasks
    virtual task_creation_hint get_next_task_hint();
    virtual std::shared_ptr<cucascade::data_batch> get_next_task_input_batch(
        rmm::cuda_stream_view stream);

    // Type checks
    virtual bool is_source() const { return false; }
    virtual bool is_sink() const { return false; }

    // Pipeline building
    virtual void build_pipelines(
        pipeline::sirius_pipeline& current,
        pipeline::sirius_meta_pipeline& meta_pipeline);
};
```

**Usage Example:**

```cpp
class sirius_physical_filter : public sirius_physical_operator {
public:
    sirius_physical_filter(
        duckdb::vector<duckdb::LogicalType> types,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions,
        duckdb::idx_t estimated_cardinality)
      : sirius_physical_operator(
            SiriusPhysicalOperatorType::FILTER,
            std::move(types),
            estimated_cardinality)
    {
        // Initialize filter expression
    }

    std::vector<std::shared_ptr<cucascade::data_batch>> execute(
        const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
        rmm::cuda_stream_view stream) override
    {
        // Filter implementation
        std::vector<std::shared_ptr<cucascade::data_batch>> output;
        for (auto const& batch : input_batches) {
            auto filtered = apply_filter(batch, stream);
            output.push_back(filtered);
        }
        return output;
    }
};
```

### sirius_pipeline

**Represents a pipeline in New Mode.**

**Header**: `src/include/pipeline/sirius_pipeline.hpp`

```cpp
class sirius_pipeline {
public:
    // Constructor
    sirius_pipeline(sirius::engine& engine);

    // Pipeline components
    sirius_physical_operator* get_source() const;
    sirius_physical_operator* get_sink() const;
    std::vector<sirius_physical_operator*> get_operators() const;

    // Execution
    void execute();
    void reset();

    // Dependencies
    void add_dependency(std::shared_ptr<sirius_pipeline> pipeline);
    std::vector<std::shared_ptr<sirius_pipeline>> get_dependencies() const;

    // Metadata
    size_t get_id() const;
    bool is_ready() const;
};
```

### task_creation_hint

**Hint for dynamic task creation.**

**Header**: `src/include/op/sirius_physical_operator.hpp`

```cpp
enum class TaskCreationHint {
    WAITING_FOR_INPUT_DATA,  // No tasks available yet
    READY,                   // Task ready to execute
    NO_MORE_TASKS            // Operator finished
};

struct task_creation_hint {
    TaskCreationHint hint;
    sirius_physical_operator* producer;  // Operator producing input
};
```

**Usage:**

```cpp
task_creation_hint sirius_physical_aggregate::get_next_task_hint() {
    if (!finalized_) {
        // Still accumulating data
        return {TaskCreationHint::WAITING_FOR_INPUT_DATA, nullptr};
    } else if (has_output_) {
        // Ready to emit aggregated result
        return {TaskCreationHint::READY, nullptr};
    } else {
        // All done
        return {TaskCreationHint::NO_MORE_TASKS, nullptr};
    }
}
```

---

## Legacy Mode API

### GPUPhysicalOperator

**Base class for all Legacy Mode operators.**

**Header**: `src/include/gpu_physical_operator.hpp`

```cpp
class GPUPhysicalOperator {
public:
    // Constructor
    GPUPhysicalOperator(
        PhysicalOperatorType type,
        vector<LogicalType> types,
        idx_t estimated_cardinality);

    // Operator metadata
    PhysicalOperatorType type;                           // Operator type
    vector<unique_ptr<GPUPhysicalOperator>> children;
    vector<LogicalType> types;                           // Output types
    idx_t estimated_cardinality;                         // Row count estimate

    // State
    unique_ptr<GlobalSinkState> sink_state;
    unique_ptr<GlobalOperatorState> op_state;

    // Core methods
    virtual string GetName() const;
    const vector<LogicalType>& GetTypes() const;

    // Source operators
    virtual SourceResultType GetData(
        GPUIntermediateRelation& output_relation) const;
    virtual bool IsSource() const { return false; }

    // Intermediate operators
    virtual OperatorResultType Execute(
        GPUIntermediateRelation& input_relation,
        GPUIntermediateRelation& output_relation) const;

    // Sink operators
    virtual SinkResultType Sink(
        GPUIntermediateRelation& input_relation) const;
    virtual SinkFinalizeType CombineFinalize(
        vector<shared_ptr<GPUIntermediateRelation>>& input,
        GPUIntermediateRelation& output) const;
    virtual bool IsSink() const { return false; }

    // Pipeline building
    virtual void BuildPipelines(
        GPUPipeline& current,
        GPUMetaPipeline& meta_pipeline);
};
```

### GPUColumn

**Columnar data structure in Legacy Mode.**

**Header**: `src/include/gpu_columns.hpp`

```cpp
class GPUColumn {
public:
    // Constructor (fixed-width)
    GPUColumn(
        size_t column_length,
        GPUColumnType type,
        uint8_t* data,
        cudf::bitmask_type* validity_mask);

    // Constructor (variable-width, e.g., VARCHAR)
    GPUColumn(
        size_t column_length,
        GPUColumnType type,
        uint8_t* data,
        uint64_t* offset,
        size_t num_bytes,
        bool is_string_data,
        cudf::bitmask_type* validity_mask);

    // Data access
    uint8_t* GetData();
    uint64_t* GetRowIds();
    template<typename T> T* GetDataAs();

    // Metadata
    size_t column_length;           // Number of rows
    size_t row_id_count;            // For late materialization
    uint64_t* row_ids;              // Row indices
    bool is_unique;                 // Unique values?

    DataWrapper data_wrapper;       // Raw GPU memory

    // cuDF interop
    cudf::column_view convertToCudfColumn();
    void setFromCudfColumn(cudf::column& cudf_column, ...);

    // Memory size
    size_t getTotalColumnSize();
};
```

### GPUIntermediateRelation

**Collection of columns (analogous to a table).**

**Header**: `src/include/gpu_columns.hpp`

```cpp
class GPUIntermediateRelation {
public:
    // Constructor
    GPUIntermediateRelation(size_t column_count);

    // Data
    vector<shared_ptr<GPUColumn>> columns;
    vector<string> column_names;
    size_t column_count;

    // Late materialization check
    bool checkLateMaterialization(size_t col_idx);
};
```

---

## cucascade Integration

### data_batch

**Core data unit in New Mode.**

**Header**: `<cucascade/data/data_batch.hpp>`

```cpp
namespace cucascade {

class data_batch {
public:
    // Factory methods
    static std::shared_ptr<data_batch> from_cudf_table(
        std::unique_ptr<cudf::table> table,
        rmm::cuda_stream_view stream);

    // Accessors
    size_t get_row_count() const;
    size_t get_column_count() const;
    std::shared_ptr<cudf::column> get_column(size_t index) const;

    // Conversion
    std::unique_ptr<cudf::table> to_cudf_table() const;

    // Memory
    size_t get_memory_usage() const;
};

}  // namespace cucascade
```

**Usage:**

```cpp
// Create from cuDF table
auto cudf_table = create_cudf_table();
auto batch = cucascade::data_batch::from_cudf_table(
    std::move(cudf_table), stream);

// Access data
size_t rows = batch->get_row_count();
auto column = batch->get_column(0);

// Convert back to cuDF
auto cudf_table_again = batch->to_cudf_table();
```

### shared_data_repository

**Inter-pipeline data storage.**

**Header**: `<cucascade/data/data_repository.hpp>`

```cpp
namespace cucascade {

class shared_data_repository {
public:
    // Constructor
    shared_data_repository(
        const std::string& name,
        memory_reservation_manager& memory_mgr);

    // Push data (producer)
    void push_data_batch(
        std::shared_ptr<data_batch> batch,
        rmm::cuda_stream_view stream);

    // Pull data (consumer)
    std::shared_ptr<data_batch> pull_batch(
        rmm::cuda_stream_view stream);

    // Status
    bool has_data() const;
    bool is_complete() const;
    void mark_complete();

    // Metadata
    std::string get_name() const;
    size_t get_batch_count() const;
};

}  // namespace cucascade
```

---

## Configuration API

### sirius_config

**Global configuration (singleton).**

**Header**: `src/include/sirius_config.hpp`

```cpp
class sirius_config {
public:
    // Singleton access
    static sirius_config& instance();

    // Configuration getters
    size_t get_gpu_memory_limit() const;
    size_t get_num_threads() const;
    std::string get_log_level() const;
    bool get_enable_fallback() const;

    // Configuration setters
    void set_gpu_memory_limit(size_t bytes);
    void set_num_threads(size_t count);
    void set_log_level(const std::string& level);
    void set_enable_fallback(bool enable);

    // Load from file
    void load_from_file(const std::string& path);
};
```

**Usage:**

```cpp
// Get configuration
auto& config = sirius_config::instance();
size_t mem_limit = config.get_gpu_memory_limit();

// Set configuration
config.set_num_threads(8);
config.set_log_level("DEBUG");
```

### SiriusContext

**Per-connection context.**

**Header**: `src/include/sirius_context.hpp`

```cpp
class SiriusContext {
public:
    // Constructor
    SiriusContext(duckdb::ClientContext& context);

    // Configuration
    void set_config_value(const std::string& key, const std::string& value);
    std::string get_config_value(const std::string& key) const;

    // Execution mode
    enum class ExecutionMode { LEGACY, NEW };
    void set_execution_mode(ExecutionMode mode);
    ExecutionMode get_execution_mode() const;

    // Memory management
    memory_reservation_manager& get_memory_manager();

    // DuckDB context
    duckdb::ClientContext& get_client_context();
};
```

---

## Expression Evaluation API

### GpuExpressionExecutor

**Evaluates SQL expressions on GPU.**

**Header**: `src/include/expression_executor/gpu_expression_executor.hpp`

```cpp
namespace sirius {

class GpuExpressionExecutor {
public:
    // Constructor
    GpuExpressionExecutor(duckdb::Expression& expression);

    // Filter rows (returns filtered batch)
    std::shared_ptr<cucascade::data_batch> select(
        std::shared_ptr<cucascade::data_batch> input,
        rmm::cuda_stream_view stream);

    // Evaluate expression (returns new column)
    std::shared_ptr<cudf::column> evaluate(
        std::shared_ptr<cucascade::data_batch> input,
        rmm::cuda_stream_view stream);

    // Legacy Mode API
    void Select(
        GPUIntermediateRelation& input,
        GPUIntermediateRelation& output);

private:
    duckdb::Expression& expression_;
};

}  // namespace sirius
```

**Usage:**

```cpp
// Create executor for "age > 25"
auto expression = parse_expression("age > 25");
sirius::GpuExpressionExecutor executor(*expression);

// Filter rows
auto input_batch = ...;
auto filtered_batch = executor.select(input_batch, stream);

// Rows with age > 25 remain
```

---

## Memory Management API

### sirius_memory_reservation_manager

**Multi-tier memory management.**

**Header**: `src/include/memory/sirius_memory_reservation_manager.hpp`

```cpp
class sirius_memory_reservation_manager {
public:
    // Constructor
    sirius_memory_reservation_manager(
        size_t gpu_limit,
        size_t host_limit,
        size_t disk_limit);

    // Reserve memory
    bool reserve_gpu_memory(size_t bytes);
    bool reserve_host_memory(size_t bytes);
    bool reserve_disk_memory(size_t bytes);

    // Release memory
    void release_gpu_memory(size_t bytes);
    void release_host_memory(size_t bytes);
    void release_disk_memory(size_t bytes);

    // Queries
    size_t get_available_gpu_memory() const;
    size_t get_available_host_memory() const;
    size_t get_available_disk_memory() const;

    // Downgrade (spill to lower tier)
    void downgrade_data(
        std::shared_ptr<cucascade::data_batch> batch,
        MemorySpace from,
        MemorySpace to);
};
```

**Usage:**

```cpp
auto& mem_mgr = context.get_memory_manager();

// Reserve 1 GB of GPU memory
if (mem_mgr.reserve_gpu_memory(1ULL * 1024 * 1024 * 1024)) {
    // Allocate GPU data
    auto batch = ...;

    // ... use batch ...

    // Release reservation
    mem_mgr.release_gpu_memory(1ULL * 1024 * 1024 * 1024);
} else {
    // GPU memory exhausted, fallback or downgrade
}
```

---

## Logging API

### Logging Macros

**Header**: `src/include/log/logging.hpp`

```cpp
// Log levels
SIRIUS_LOG_TRACE("Trace message: {}", value);
SIRIUS_LOG_DEBUG("Debug message: {}", value);
SIRIUS_LOG_INFO("Info message: {}", value);
SIRIUS_LOG_WARN("Warning message: {}", value);
SIRIUS_LOG_ERROR("Error message: {}", value);
SIRIUS_LOG_CRITICAL("Critical error: {}", value);
```

**Format String:**

Uses **fmt** library syntax:

```cpp
SIRIUS_LOG_DEBUG("Processing {} rows, {} columns", row_count, col_count);
SIRIUS_LOG_DEBUG("Operator: {} (ID: {})", op->get_name(), op->get_operator_id());
SIRIUS_LOG_DEBUG("Memory usage: {:.2f} GB", bytes / 1e9);
```

### Log Level Configuration

**Set log level:**

```cpp
#include "log/logging.hpp"

// Set to DEBUG
sirius::log::set_log_level(sirius::log::LogLevel::DEBUG);

// Set to INFO
sirius::log::set_log_level(sirius::log::LogLevel::INFO);

// Disable logging
sirius::log::set_log_level(sirius::log::LogLevel::OFF);
```

**Via SQL:**

```sql
SET sirius_log_level = 'DEBUG';
SET sirius_log_level = 'INFO';
SET sirius_log_level = 'OFF';
```

**Via Environment Variable:**

```bash
export SIRIUS_LOG_LEVEL=DEBUG
./duckdb
```

---

## Next Steps

**Related Documentation:**

- **[Glossary](glossary.md)**: Terms and definitions
- **[File Index](file-index.md)**: Complete file listing
- **[Config Options](config-options.md)**: All configuration parameters

**Development:**

- **[Adding Operators](../07-development/adding-operators.md)**: Implement new operators
- **[Debugging](../07-development/debugging.md)**: Debug with logging and tools
- **[Testing Guide](../07-development/testing-guide.md)**: Write tests

**Architecture:**

- **[New Mode Overview](../04-new-mode/overview.md)**: Understand New Mode
- **[Legacy Mode Overview](../03-legacy-mode/overview.md)**: Understand Legacy Mode
- **[Core Components](../05-core-components/)**: Deep dives into subsystems
