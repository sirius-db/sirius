# Debugging Sirius

This guide provides practical techniques for debugging Sirius queries, operators, and GPU execution.

## Table of Contents

1. [Overview](#overview)
2. [Logging System](#logging-system)
3. [Debugging Query Execution](#debugging-query-execution)
4. [Debugging Operators](#debugging-operators)
5. [GPU Debugging](#gpu-debugging)
6. [Memory Debugging](#memory-debugging)
7. [Common Issues](#common-issues)
8. [Tools and Utilities](#tools-and-utilities)
9. [Next Steps](#next-steps)

---

## Overview

Debugging Sirius involves several layers:

1. **SQL Layer**: Query parsing, binding, and logical planning
2. **Planning Layer**: Physical plan generation and operator conversion
3. **Execution Layer**: Pipeline execution and task scheduling
4. **Operator Layer**: Individual operator execution
5. **GPU Layer**: CUDA kernels, cuDF operations, and memory management

This guide covers techniques for each layer.

---

## Logging System

Sirius uses a custom logging system based on **spdlog** with configurable log levels.

### Log Levels

```cpp
enum class LogLevel {
    TRACE,      // Most verbose
    DEBUG,      // Detailed debugging information
    INFO,       // General information
    WARN,       // Warnings
    ERROR,      // Errors
    CRITICAL,   // Critical errors
    OFF         // Disable logging
};
```

### Configuration

**Set log level via environment variable:**

```bash
export SIRIUS_LOG_LEVEL=DEBUG
./duckdb
```

**Set log level in SQL:**

```sql
SET sirius_log_level = 'DEBUG';
```

**Set log level in code:**

```cpp
#include "log/logging.hpp"

sirius::log::set_log_level(sirius::log::LogLevel::DEBUG);
```

### Logging Macros

**File**: `src/include/log/logging.hpp`

```cpp
// Log at different levels
SIRIUS_LOG_TRACE("Very detailed trace: {}", value);
SIRIUS_LOG_DEBUG("Debug info: {}", value);
SIRIUS_LOG_INFO("General info: {}", value);
SIRIUS_LOG_WARN("Warning: {}", value);
SIRIUS_LOG_ERROR("Error: {}", value);
SIRIUS_LOG_CRITICAL("Critical error: {}", value);

// Format string (uses fmt library)
SIRIUS_LOG_DEBUG("Processing {} rows, {} columns", row_count, col_count);
SIRIUS_LOG_DEBUG("Operator: {} (ID: {})", op->get_name(), op->get_operator_id());
```

### Example Usage in Operator

```cpp
std::vector<std::shared_ptr<cucascade::data_batch>>
sirius_physical_filter::execute(
    const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
    rmm::cuda_stream_view stream)
{
    SIRIUS_LOG_DEBUG("FILTER: Executing with {} input batches", input_batches.size());

    std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;

    for (size_t i = 0; i < input_batches.size(); i++) {
        auto const& batch = input_batches[i];
        if (!batch) {
            SIRIUS_LOG_WARN("FILTER: Batch {} is null, skipping", i);
            continue;
        }

        SIRIUS_LOG_DEBUG("FILTER: Processing batch {} ({} rows)", i, batch->get_row_count());

        auto filtered = gpu_expression_executor.select(batch, stream);

        if (filtered) {
            SIRIUS_LOG_DEBUG("FILTER: Output {} rows (selectivity: {:.2f}%)",
                            filtered->get_row_count(),
                            100.0 * filtered->get_row_count() / batch->get_row_count());
            output_batches.push_back(filtered);
        }
    }

    SIRIUS_LOG_DEBUG("FILTER: Produced {} output batches", output_batches.size());
    return output_batches;
}
```

**Output:**

```
[DEBUG] FILTER: Executing with 5 input batches
[DEBUG] FILTER: Processing batch 0 (1000 rows)
[DEBUG] FILTER: Output 300 rows (selectivity: 30.00%)
[DEBUG] FILTER: Processing batch 1 (1000 rows)
[DEBUG] FILTER: Output 350 rows (selectivity: 35.00%)
...
[DEBUG] FILTER: Produced 5 output batches
```

---

## Debugging Query Execution

### 1. Enable Query Logging

**SQL:**

```sql
SET sirius_log_level = 'DEBUG';

SELECT * FROM gpu_execution('SELECT * FROM users WHERE age > 25');
```

**Output:**

```
[DEBUG] Query: SELECT * FROM users WHERE age > 25
[DEBUG] Logical Plan:
[DEBUG]   RESULT_COLLECTOR
[DEBUG]     FILTER (age > 25)
[DEBUG]       TABLE_SCAN (users)
[DEBUG] Physical Plan:
[DEBUG]   sirius_physical_result_collector
[DEBUG]     sirius_physical_filter (age > 25)
[DEBUG]       sirius_physical_table_scan (users)
[DEBUG] Building pipelines...
[DEBUG] Pipeline #1: TABLE_SCAN → FILTER → RESULT_COLLECTOR
[DEBUG] Executing pipeline #1...
[DEBUG] TABLE_SCAN: Reading 1000 rows
[DEBUG] FILTER: Output 300 rows (selectivity: 30.00%)
[DEBUG] RESULT_COLLECTOR: Accumulated 300 rows
[DEBUG] Query completed: 300 rows returned
```

### 2. Print Physical Plan

**Add to your operator or executor:**

```cpp
void print_operator_tree(sirius_physical_operator* op, int depth = 0) {
    std::string indent(depth * 2, ' ');
    SIRIUS_LOG_INFO("{}{}[{}] (ID: {}, Card: {})",
                   indent,
                   op->get_name(),
                   SiriusPhysicalOperatorTypeToString(op->type),
                   op->get_operator_id(),
                   op->estimated_cardinality);

    for (auto& child : op->children) {
        print_operator_tree(child.get(), depth + 1);
    }
}

// Usage
print_operator_tree(physical_plan.get());
```

**Output:**

```
[INFO] sirius_physical_result_collector[RESULT_COLLECTOR] (ID: 0, Card: 300)
[INFO]   sirius_physical_filter[FILTER] (ID: 1, Card: 300)
[INFO]     sirius_physical_table_scan[TABLE_SCAN] (ID: 2, Card: 1000)
```

### 3. Trace Pipeline Execution

**Enable pipeline tracing:**

```cpp
#define TRACE_PIPELINE_EXECUTION 1

void sirius_engine::execute() {
    for (auto& pipeline : pipelines_) {
#ifdef TRACE_PIPELINE_EXECUTION
        SIRIUS_LOG_DEBUG("Executing pipeline #{} (source: {}, sink: {})",
                        pipeline->get_id(),
                        pipeline->get_source()->get_name(),
                        pipeline->get_sink() ? pipeline->get_sink()->get_name() : "none");
#endif

        pipeline->execute();
    }
}
```

### 4. Step-by-Step Debugging with GDB

**Build with debug symbols:**

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

**Run with GDB:**

```bash
gdb --args ./build/release/duckdb
```

**GDB Commands:**

```gdb
# Set breakpoint in operator
break sirius_physical_filter::execute

# Run query
run
> SELECT * FROM gpu_execution('SELECT * FROM users WHERE age > 25');

# Step through execution
next      # Step over
step      # Step into
continue  # Continue execution

# Print variables
print input_batches.size()
print batch->get_row_count()

# Backtrace
bt        # Show call stack

# Watch variable
watch row_count
```

---

## Debugging Operators

### 1. Add Detailed Logging

**Before:**

```cpp
auto result = process_data(input);
```

**After:**

```cpp
SIRIUS_LOG_DEBUG("Processing input: {} rows, {} columns",
                input->get_row_count(), input->get_column_count());

auto result = process_data(input);

SIRIUS_LOG_DEBUG("Output: {} rows, {} columns",
                result->get_row_count(), result->get_column_count());

// Verify output
D_ASSERT(result->get_row_count() <= input->get_row_count());
D_ASSERT(result->get_column_count() == input->get_column_count());
```

### 2. Inspect Data Batches

**Print batch summary:**

```cpp
void print_batch_summary(std::shared_ptr<cucascade::data_batch> batch) {
    if (!batch) {
        SIRIUS_LOG_DEBUG("Batch: nullptr");
        return;
    }

    SIRIUS_LOG_DEBUG("Batch: {} rows, {} columns",
                    batch->get_row_count(),
                    batch->get_column_count());

    for (size_t i = 0; i < batch->get_column_count(); i++) {
        auto col = batch->get_column(i);
        SIRIUS_LOG_DEBUG("  Column {}: type={}, size={} bytes",
                        i,
                        col->type().id(),
                        col->size());
    }
}
```

**Print batch data (small batches only!):**

```cpp
void print_batch_data(std::shared_ptr<cucascade::data_batch> batch, size_t max_rows = 10) {
    auto cudf_table = batch->to_cudf_table();

    // Transfer to host for printing
    std::vector<std::string> rows;
    for (size_t row = 0; row < std::min(batch->get_row_count(), max_rows); row++) {
        std::ostringstream oss;
        for (size_t col = 0; col < batch->get_column_count(); col++) {
            // Extract value (simplified, actual implementation more complex)
            oss << cudf_table->get_column(col).element<int32_t>(row) << " ";
        }
        rows.push_back(oss.str());
    }

    SIRIUS_LOG_DEBUG("Batch data (first {} rows):", max_rows);
    for (const auto& row : rows) {
        SIRIUS_LOG_DEBUG("  {}", row);
    }
}
```

### 3. Validate Invariants

**Add assertions:**

```cpp
std::shared_ptr<cucascade::data_batch> sirius_physical_join::probe_hash_table(
    std::shared_ptr<cucascade::data_batch> probe_batch)
{
    D_ASSERT(hash_table_ != nullptr);  // Hash table must be built
    D_ASSERT(probe_batch != nullptr);  // Probe batch must be valid
    D_ASSERT(probe_batch->get_row_count() > 0);  // Must have rows

    auto result = perform_probe(probe_batch);

    // Validate result
    D_ASSERT(result != nullptr);
    D_ASSERT(result->get_column_count() ==
             probe_batch->get_column_count() + build_batch_->get_column_count());

    return result;
}
```

### 4. Compare with DuckDB CPU Execution

**Run same query on CPU:**

```sql
-- GPU execution
SELECT * FROM gpu_execution('SELECT * FROM users WHERE age > 25');

-- CPU execution (for comparison)
SELECT * FROM users WHERE age > 25;
```

**Compare results:**

```bash
# Save GPU result
echo "SELECT * FROM gpu_execution('SELECT * FROM users WHERE age > 25');" | \
    ./duckdb | tee gpu_result.txt

# Save CPU result
echo "SELECT * FROM users WHERE age > 25;" | \
    ./duckdb | tee cpu_result.txt

# Compare
diff gpu_result.txt cpu_result.txt
```

---

## GPU Debugging

### 1. Check for CUDA Errors

**Enable synchronous error checking:**

```cpp
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            SIRIUS_LOG_ERROR("CUDA error: {} at {}:{}", \
                           cudaGetErrorString(err), \
                           __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)

// Usage
CHECK_CUDA_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
```

**Synchronize streams for debugging:**

```cpp
// Force synchronization after each GPU operation
cudaStreamSynchronize(stream);
CHECK_CUDA_ERROR(cudaGetLastError());
```

### 2. Use CUDA-MEMCHECK

**Run with cuda-memcheck:**

```bash
cuda-memcheck ./build/release/duckdb
```

**Options:**

```bash
# Check for memory errors
cuda-memcheck --tool memcheck ./duckdb

# Check for race conditions
cuda-memcheck --tool racecheck ./duckdb

# Check for initialization errors
cuda-memcheck --tool initcheck ./duckdb

# Check for synchronization errors
cuda-memcheck --tool synccheck ./duckdb
```

### 3. Use Nsight Compute for Profiling

**Profile a specific query:**

```bash
ncu --target-processes all \
    --kernel-name "your_kernel_name" \
    ./duckdb -c "SELECT * FROM gpu_execution('...');"
```

**Profile metrics:**

```bash
ncu --metrics gpu__time_duration,dram__throughput \
    ./duckdb -c "SELECT * FROM gpu_execution('...');"
```

### 4. Debug cuDF Operations

**Enable cuDF logging:**

```cpp
// Before cuDF operations
cudf::logger().set_level(spdlog::level::debug);

// Your cuDF operation
auto result = cudf::filter(table, predicate, stream);

// Check for errors
cudaStreamSynchronize(stream);
if (cudaGetLastError() != cudaSuccess) {
    SIRIUS_LOG_ERROR("cuDF operation failed");
}
```

**Print cuDF table:**

```cpp
void print_cudf_table(cudf::table_view table, size_t max_rows = 10) {
    // Transfer to host
    std::vector<std::unique_ptr<cudf::column>> host_cols;
    for (size_t i = 0; i < table.num_columns(); i++) {
        host_cols.push_back(std::make_unique<cudf::column>(
            table.column(i), rmm::cuda_stream_default));
    }

    // Print (simplified)
    for (size_t row = 0; row < std::min(table.num_rows(), max_rows); row++) {
        std::ostringstream oss;
        for (size_t col = 0; col < table.num_columns(); col++) {
            // Extract and print value
            // (actual implementation depends on column type)
        }
        SIRIUS_LOG_DEBUG("{}", oss.str());
    }
}
```

---

## Memory Debugging

### 1. Track GPU Memory Usage

**Query GPU memory:**

```cpp
void print_gpu_memory_usage() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    size_t used_bytes = total_bytes - free_bytes;

    SIRIUS_LOG_INFO("GPU Memory: {:.2f} GB used / {:.2f} GB total ({:.1f}%)",
                   used_bytes / 1e9,
                   total_bytes / 1e9,
                   100.0 * used_bytes / total_bytes);
}

// Call periodically
print_gpu_memory_usage();
```

### 2. Track RMM Pool Usage

**Query RMM statistics:**

```cpp
void print_rmm_stats(rmm::mr::pool_memory_resource* mr) {
    SIRIUS_LOG_INFO("RMM Pool:");
    SIRIUS_LOG_INFO("  Allocated: {:.2f} GB", mr->get_allocated_size() / 1e9);
    SIRIUS_LOG_INFO("  Free: {:.2f} GB", mr->get_free_size() / 1e9);
}
```

### 3. Detect Memory Leaks

**Use CUDA-MEMCHECK:**

```bash
cuda-memcheck --leak-check full ./duckdb
```

**Use RMM Logging:**

```cpp
// Enable RMM logging
rmm::logger().set_level(spdlog::level::debug);

// All allocations/deallocations will be logged
auto* ptr = mr->allocate(size);
// ... use ptr ...
mr->deallocate(ptr, size);
```

### 4. Debug Out-of-Memory Errors

**Catch OOM exceptions:**

```cpp
try {
    auto batch = process_large_batch(input);
} catch (const rmm::out_of_memory& e) {
    SIRIUS_LOG_ERROR("Out of GPU memory: {}", e.what());
    print_gpu_memory_usage();
    print_rmm_stats(mr);

    // Try recovery or fail gracefully
    throw;
}
```

---

## Common Issues

### Issue 1: Incorrect Results

**Symptoms:**

- Query returns wrong rows or values
- Results differ from DuckDB CPU execution

**Debugging Steps:**

1. **Enable DEBUG logging:**

   ```sql
   SET sirius_log_level = 'DEBUG';
   ```

2. **Compare with CPU:**

   ```sql
   -- GPU (check result)
   SELECT * FROM gpu_execution('SELECT * FROM t WHERE x > 10');

   -- CPU (expected result)
   SELECT * FROM t WHERE x > 10;
   ```

3. **Print intermediate results:**

   ```cpp
   SIRIUS_LOG_DEBUG("Filter input: {} rows", input->get_row_count());
   // Print first few rows
   print_batch_data(input, 5);

   auto output = apply_filter(input);

   SIRIUS_LOG_DEBUG("Filter output: {} rows", output->get_row_count());
   print_batch_data(output, 5);
   ```

4. **Check expression evaluation:**

   ```cpp
   SIRIUS_LOG_DEBUG("Evaluating expression: {}", expression->ToString());
   auto result = gpu_expression_executor.select(batch, stream);
   SIRIUS_LOG_DEBUG("Expression result: {} rows selected", result->get_row_count());
   ```

### Issue 2: Performance Regression

**Symptoms:**

- Query is slower than expected
- Slower than DuckDB CPU execution

**Debugging Steps:**

1. **Profile with Nsight Compute:**

   ```bash
   ncu --target-processes all ./duckdb -c "SELECT * FROM gpu_execution('...');"
   ```

2. **Check memory transfers:**

   ```cpp
   auto start = std::chrono::high_resolution_clock::now();
   cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
   auto end = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
   SIRIUS_LOG_DEBUG("Transfer time: {} ms ({:.2f} GB/s)",
                   duration.count(),
                   (size / 1e9) / (duration.count() / 1000.0));
   ```

3. **Check batch sizes:**

   ```cpp
   SIRIUS_LOG_DEBUG("Batch size: {} rows ({:.2f} MB)",
                   batch->get_row_count(),
                   batch->get_memory_usage() / 1e6);
   ```

### Issue 3: GPU Errors

**Symptoms:**

- CUDA error messages
- Segmentation faults
- Kernel launch failures

**Debugging Steps:**

1. **Enable synchronous error checking:**

   ```cpp
   // After each GPU operation
   cudaStreamSynchronize(stream);
   CHECK_CUDA_ERROR(cudaGetLastError());
   ```

2. **Run with cuda-memcheck:**

   ```bash
   cuda-memcheck ./duckdb
   ```

3. **Check for NULL pointers:**

   ```cpp
   D_ASSERT(batch != nullptr);
   D_ASSERT(batch->get_row_count() > 0);
   D_ASSERT(batch->get_column(0) != nullptr);
   ```

---

## Tools and Utilities

### 1. Nsight Systems (System-wide Profiling)

**Capture trace:**

```bash
nsys profile --trace=cuda,nvtx \
             --output=sirius_trace \
             ./duckdb -c "SELECT * FROM gpu_execution('...');"
```

**View trace:**

```bash
nsys-ui sirius_trace.nsys-rep
```

### 2. Nsight Compute (Kernel Profiling)

**Profile specific kernel:**

```bash
ncu --target-processes all \
    --kernel-name "my_kernel" \
    --metrics all \
    --set full \
    ./duckdb -c "SELECT * FROM gpu_execution('...');"
```

### 3. nvidia-smi (GPU Monitoring)

**Monitor GPU usage:**

```bash
watch -n 1 nvidia-smi
```

**Log GPU stats:**

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.free \
           --format=csv \
           --loop=1 \
           > gpu_stats.csv
```

### 4. GDB with CUDA Support

**Install cuda-gdb:**

```bash
# Already included with CUDA toolkit
cuda-gdb --args ./duckdb
```

**CUDA-specific commands:**

```gdb
# Show CUDA threads
info cuda threads

# Show CUDA blocks
info cuda blocks

# Switch to CUDA thread
cuda thread 0

# Show CUDA kernel
info cuda kernels
```

---

## Next Steps

**Related Documentation:**

- **[Building and Testing](building-and-testing.md)**: Setup development environment
- **[Adding Operators](adding-operators.md)**: Implement and debug new operators
- **[Testing Guide](testing-guide.md)**: Write effective tests

**Tools:**

- **Nsight Systems**: https://developer.nvidia.com/nsight-systems
- **Nsight Compute**: https://developer.nvidia.com/nsight-compute
- **cuda-gdb**: https://docs.nvidia.com/cuda/cuda-gdb/

**Performance:**

- **[Performance Tips](../appendices/performance-tips.md)**: Optimize query performance
