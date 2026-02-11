# Legacy Mode Memory Management

This document explains how Sirius Legacy Mode manages GPU and CPU memory using the **GPUBufferManager** singleton.

## Table of Contents

1. [Overview](#overview)
2. [GPUBufferManager](#gpubuffermanager)
3. [Memory Regions](#memory-regions)
4. [Allocation Strategies](#allocation-strategies)
5. [Caching System](#caching-system)
6. [RMM Integration](#rmm-integration)
7. [Memory Lifecycle](#memory-lifecycle)
8. [Performance Considerations](#performance-considerations)
9. [Next Steps](#next-steps)

---

## Overview

Legacy Mode uses a **singleton GPUBufferManager** to handle all GPU and CPU memory allocations. This centralized manager provides:

- **Pre-allocated memory pools**: Reduces allocation overhead
- **Caching layer**: Reuses frequently accessed data (e.g., dimension tables)
- **CPU fallback**: Uses pinned CPU memory when GPU memory is exhausted
- **RMM integration**: Leverages RAPIDS Memory Manager for pooling

**Key Characteristics:**

- **Manual memory management**: Operators explicitly allocate and free memory
- **No automatic spilling**: Unlike New Mode, Legacy Mode doesn't automatically spill to disk
- **Reference counting**: Uses `shared_ptr<GPUColumn>` for automatic cleanup
- **Batch-scoped lifetimes**: Memory is typically freed at pipeline completion

**File**: `src/gpu_buffer_manager.cpp`

---

## GPUBufferManager

The **GPUBufferManager** is a singleton that manages three types of memory regions:

1. **GPU Processing Memory**: Active computation workspace (uses RMM pool)
2. **GPU Cache Memory**: Stores frequently accessed data
3. **CPU Pinned Memory**: Fallback for GPU-resident data and CPU processing

### Class Structure

**File**: `src/include/gpu_buffer_manager.hpp`

```cpp
class GPUBufferManager {
public:
    // Singleton access
    static GPUBufferManager& GetInstance();

    // GPU device memory allocation (processing)
    template <typename T>
    T* customCudaMalloc(size_t size, int gpu = 0, bool caching = false);

    // CPU pinned memory allocation
    template <typename T>
    T* customCudaHostAlloc(size_t size);

    // Memory freeing
    template <typename T>
    void customCudaFree(T* ptr, int gpu = 0, bool caching = false);
    template <typename T>
    void customCudaHostFree(T* ptr);

private:
    // Memory regions
    uint8_t** gpuProcessing;        // Processing memory per GPU
    uint8_t** gpuCache;             // Cache memory per GPU
    uint8_t** cpuCache;             // CPU cache (overflow from GPU)
    uint8_t* cpuProcessing;         // CPU processing (pinned)

    // Allocation pointers (bump allocator)
    size_t* gpuProcessingPointer;   // Current position in processing pool
    size_t* gpuCachingPointer;      // Current position in cache pool
    size_t* cpuCachingPointer;      // Current position in CPU cache
    size_t cpuProcessingPointer;    // Current position in CPU processing

    // Pool sizes
    size_t cache_size_per_gpu;      // Cache pool size
    size_t processing_size_per_gpu; // Processing pool size
    size_t processing_size_per_cpu; // CPU processing pool size

    // RMM integration
    rmm::mr::cuda_memory_resource* cuda_mr;
    rmm::mr::pool_memory_resource* mr;

    // Allocation tracking
    vector<unordered_map<void*, size_t>> allocation_table;
    vector<unordered_map<void*, size_t>> locked_allocation_table;
};
```

### Initialization

**File**: `src/gpu_buffer_manager.cpp:145-200`

```cpp
GPUBufferManager::GPUBufferManager(size_t cache_size_per_gpu,
                                   size_t processing_size_per_gpu,
                                   size_t processing_size_per_cpu)
  : cache_size_per_gpu(cache_size_per_gpu),
    processing_size_per_gpu(processing_size_per_gpu),
    processing_size_per_cpu(processing_size_per_cpu) {

    SIRIUS_LOG_INFO("Initializing GPU buffer manager: "
                   "GPU Cache Size - {}, GPU Processing Size - {}, CPU Processing Size - {}",
                   cache_size_per_gpu, processing_size_per_gpu, processing_size_per_cpu);

    // Step 1: Initialize RMM pool for processing memory
    cuda_mr = new rmm::mr::cuda_memory_resource();
    mr = new rmm::mr::pool_memory_resource(
        cuda_mr, processing_size_per_gpu, processing_size_per_cpu);
    cudf::set_current_device_resource(mr);

    // Step 2: Allocate cache memory
    for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
        size_t free_gpu_mem_size = getFreeGPUMemorySize(gpu) * 0.99;

        if (free_gpu_mem_size >= cache_size_per_gpu) {
            // All cache fits in GPU
            gpuCache[gpu] = callCudaMalloc<uint8_t>(cache_size_per_gpu, gpu);
            cpuCache[gpu] = nullptr;
            available_gpu_cache_size[gpu] = cache_size_per_gpu;
            SIRIUS_LOG_INFO("Allocated cache size {} in GPU 0", cache_size_per_gpu);
        } else {
            // Hybrid GPU + CPU cache
            gpuCache[gpu] = callCudaMalloc<uint8_t>(free_gpu_mem_size, gpu);
            cpuCache[gpu] = allocatePinnedCPUMemory(cache_size_per_gpu - free_gpu_mem_size);
            available_gpu_cache_size[gpu] = free_gpu_mem_size;
            SIRIUS_LOG_INFO("Allocated hybrid cache: {} in GPU, {} in CPU",
                           free_gpu_mem_size, cache_size_per_gpu - free_gpu_mem_size);
        }

        gpuProcessingPointer[gpu] = 0;
        gpuCachingPointer[gpu] = 0;
        cpuCachingPointer[gpu] = 0;
    }

    // Step 3: Allocate CPU processing memory (pinned or pageable)
    cpuProcessing = Config::USE_PIN_MEM_FOR_CPU_PROCESSING
                      ? allocatePinnedCPUMemory(processing_size_per_cpu)
                      : allocatePageableCPUMemory(processing_size_per_cpu);
    cpuProcessingPointer = 0;
}
```

**Default Sizes (configurable):**

- **GPU Cache**: ~10-20% of GPU memory (e.g., 4 GB on 32 GB GPU)
- **GPU Processing**: ~60-70% of GPU memory (e.g., 20 GB on 32 GB GPU)
- **CPU Processing**: ~10-20 GB of pinned host memory

---

## Memory Regions

### 1. GPU Processing Memory

**Purpose**: Active workspace for computations (filters, joins, aggregations).

**Characteristics:**

- **Managed by RMM**: Uses `rmm::mr::pool_memory_resource` for fast allocation
- **Largest pool**: Typically 60-70% of GPU memory
- **Short-lived allocations**: Memory freed at operator/pipeline completion
- **Not cached**: Data doesn't persist across queries

**Usage Example:**

```cpp
// Allocate GPU memory for filter output
auto* filtered_data = gpuBufferManager->customCudaMalloc<int32_t>(row_count);

// ... perform filtering on GPU ...

// Free after use
gpuBufferManager->customCudaFree(filtered_data);
```

### 2. GPU Cache Memory

**Purpose**: Stores frequently accessed data (e.g., dimension tables for joins).

**Characteristics:**

- **Long-lived allocations**: Persists across queries
- **Smaller pool**: Typically 10-20% of GPU memory
- **Manual caching**: Operators must explicitly mark data as cacheable
- **Hybrid CPU fallback**: Uses pinned CPU memory if GPU cache is full

**Usage Example:**

```cpp
// Check if table is cached
if (!gpuBufferManager->isCached(table_id)) {
    // Load to cache (caching = true)
    auto* cached_data = gpuBufferManager->customCudaMalloc<int32_t>(
        row_count, gpu, /*caching=*/true);

    // Transfer data to cache
    cudaMemcpy(cached_data, source_data, size, cudaMemcpyHostToDevice);

    // Register in cache
    gpuBufferManager->registerCache(table_id, cached_data, size);
}
```

**Cache Eviction:**

- **Currently manual**: No automatic LRU eviction
- **Future enhancement**: Automatic eviction based on access patterns

### 3. CPU Pinned Memory

**Purpose**: Fast data transfer between CPU and GPU.

**Characteristics:**

- **Pinned (page-locked)**: Allocated via `cudaMallocHost()`
- **Fast PCIe transfers**: ~10 GB/s (vs ~5 GB/s for pageable memory)
- **Two regions**:
  - **CPU Cache**: Overflow from GPU cache
  - **CPU Processing**: Workspace for CPU-side operations

**Usage Example:**

```cpp
// Allocate pinned host memory for transfer
auto* host_buffer = gpuBufferManager->customCudaHostAlloc<int32_t>(row_count);

// Transfer GPU → Host (fast)
cudaMemcpy(host_buffer, device_buffer, size, cudaMemcpyDeviceToHost);

// Convert to DuckDB format
ConvertToDuckDBChunk(host_buffer, duckdb_chunk);

// Free host memory
gpuBufferManager->customCudaHostFree(host_buffer);
```

---

## Allocation Strategies

### Bump Allocator (Processing & Cache)

For both processing and cache regions, GPUBufferManager uses a **bump pointer allocator**:

```cpp
template <typename T>
T* GPUBufferManager::customCudaMalloc(size_t size, int gpu, bool caching) {
    size_t byte_size = size * sizeof(T);

    if (caching) {
        // Allocate from cache region
        T* ptr = reinterpret_cast<T*>(gpuCache[gpu] + gpuCachingPointer[gpu]);
        gpuCachingPointer[gpu] += byte_size;

        if (gpuCachingPointer[gpu] > cache_size_per_gpu) {
            throw OutOfMemoryException("GPU cache exhausted");
        }

        return ptr;
    } else {
        // Allocate from processing region (via RMM)
        void* ptr = nullptr;
        RMM_CUDA_TRY(mr->allocate(byte_size));
        allocation_table[gpu][ptr] = byte_size;
        return static_cast<T*>(ptr);
    }
}
```

**Advantages:**

- **Fast allocation**: O(1) pointer increment
- **No fragmentation**: Sequential allocation pattern
- **Predictable**: No need for complex memory management

**Disadvantages:**

- **No reuse**: Memory not reclaimed until pool reset
- **Wasted space**: Deleted allocations leave holes
- **Fixed size**: Pool size must be pre-allocated

### RMM Pool Allocator (Processing Only)

The GPU processing region uses **RAPIDS Memory Manager (RMM)** with a pool allocator:

```cpp
// RMM pool wraps cudaMalloc with intelligent pooling
rmm::mr::cuda_memory_resource* cuda_mr = new rmm::mr::cuda_memory_resource();
rmm::mr::pool_memory_resource* mr = new rmm::mr::pool_memory_resource(
    cuda_mr, processing_size_per_gpu, processing_size_per_cpu);

// Set as default for cuDF operations
cudf::set_current_device_resource(mr);
```

**Advantages:**

- **Reuses freed memory**: Maintains free list
- **Integrates with cuDF**: All cuDF operations use this pool
- **Coalesces allocations**: Reduces fragmentation
- **Fast for repeated patterns**: Typical database workload

---

## Caching System

### Cached Data Lifecycle

1. **First Access**: Data is loaded into GPU/CPU cache
2. **Subsequent Accesses**: Cache hit, no data transfer needed
3. **Cache Invalidation**: Manual deletion when table updates

### Example: Dimension Table Caching

**Scenario:** Join fact table (1B rows) with dimension table (1M rows, fits in cache).

**Query 1: Initial Load**

```sql
SELECT f.*, d.name FROM facts f JOIN dim d ON f.dim_id = d.id;
```

```
TABLE_SCAN (dim)
    ↓
Load to GPU cache (10 ms)
    ↓
Build hash table (50 ms)
    ↓
TABLE_SCAN (facts) + Probe (1000 ms)
    ↓
RESULT_COLLECTOR

Total: ~1060 ms
```

**Query 2: Cache Hit**

```sql
SELECT f.*, d.region FROM facts f JOIN dim d ON f.dim_id = d.id;
```

```
Hash table already in GPU cache
    ↓
TABLE_SCAN (facts) + Probe (1000 ms)
    ↓
RESULT_COLLECTOR

Total: ~1000 ms (6% speedup)
```

### Cache Management API

```cpp
// Check if cached
bool GPUBufferManager::isCached(string table_id);

// Add to cache
void GPUBufferManager::registerCache(string table_id, void* ptr, size_t size);

// Remove from cache
void GPUBufferManager::invalidateCache(string table_id);

// Clear all cache
void GPUBufferManager::clearCache();
```

---

## RMM Integration

### RAPIDS Memory Manager (RMM)

**RMM** is a CUDA memory manager optimized for data analytics workloads. Sirius Legacy Mode integrates RMM for GPU processing memory:

**Benefits:**

1. **Memory Pooling**: Reuses freed allocations
2. **cuDF Integration**: All cuDF operations use the same pool
3. **Suballocators**: Supports different allocation strategies
4. **Tracking**: Logs all allocations for debugging

**Configuration:**

```cpp
// Create pool with 20 GB initial size, 30 GB max
rmm::mr::pool_memory_resource* mr = new rmm::mr::pool_memory_resource(
    cuda_mr,
    /*initial_pool_size=*/20ULL * 1024 * 1024 * 1024,  // 20 GB
    /*maximum_pool_size=*/30ULL * 1024 * 1024 * 1024   // 30 GB
);
```

**Allocation via RMM:**

```cpp
// All cuDF operations automatically use RMM pool
auto cudf_table = cudf::io::read_parquet(...);  // Uses RMM pool internally

// Manual allocation
void* ptr = mr->allocate(size);  // Fast if size recently freed
mr->deallocate(ptr, size);       // Returns to pool
```

### Memory Tracking

RMM tracks all allocations for debugging:

```cpp
// Get current memory usage
size_t bytes_allocated = mr->get_allocated_size();
size_t bytes_free = mr->get_free_size();

SIRIUS_LOG_DEBUG("RMM Pool: {} bytes allocated, {} bytes free",
                bytes_allocated, bytes_free);
```

---

## Memory Lifecycle

### Typical Allocation Lifecycle

**Example: Filter Operator**

```cpp
OperatorResultType GPUPhysicalFilter::Execute(
    GPUIntermediateRelation& input_relation,
    GPUIntermediateRelation& output_relation) const {

    // Step 1: Allocate selection vector
    auto* selection_vector = gpuBufferManager->customCudaMalloc<uint64_t>(
        input_relation.row_count);

    // Step 2: Evaluate filter (GPU kernel)
    EvaluateFilterExpression(input_relation, selection_vector);

    // Step 3: Allocate output columns
    for (size_t col = 0; col < input_relation.columns.size(); col++) {
        auto& input_col = input_relation.columns[col];
        size_t output_size = CountSelectedRows(selection_vector);

        auto* output_data = gpuBufferManager->customCudaMalloc<int32_t>(output_size);

        // Step 4: Gather selected rows
        GatherRows(input_col->data, output_data, selection_vector, output_size);

        // Step 5: Wrap in GPUColumn (shared_ptr)
        auto output_col = make_shared_ptr<GPUColumn>(
            output_size, input_col->data_wrapper.type, output_data, nullptr);

        output_relation.columns.push_back(output_col);
    }

    // Step 6: Free selection vector
    gpuBufferManager->customCudaFree(selection_vector);

    // Note: output_data is owned by GPUColumn and freed when shared_ptr goes out of scope

    return OperatorResultType::FINISHED;
}
```

### Memory Ownership

**GPUColumn uses `shared_ptr` for automatic cleanup:**

```cpp
class GPUColumn {
public:
    DataWrapper data_wrapper;  // Contains raw pointer to GPU memory
    ~GPUColumn() {
        // Automatically frees GPU memory when refcount reaches 0
        if (data_wrapper.data) {
            gpuBufferManager->customCudaFree(data_wrapper.data);
        }
    }
};

// Usage
auto column = make_shared_ptr<GPUColumn>(...);  // Allocates GPU memory
// ... use column ...
// column automatically freed when last reference destroyed
```

---

## Performance Considerations

### Memory Transfer Bottlenecks

**PCIe Bandwidth Limits:**

| Transfer Type | Bandwidth | Use Case |
|---------------|-----------|----------|
| **Pageable CPU → GPU** | ~5 GB/s | Initial table scan (slow) |
| **Pinned CPU → GPU** | ~10 GB/s | Optimized table scan |
| **GPU → GPU** | ~900 GB/s | Operator data flow (fast) |
| **GPU → CPU (result)** | ~10 GB/s | Result collection |

**Optimization Tips:**

1. **Use pinned memory** for CPU ↔ GPU transfers (`cudaMallocHost`)
2. **Batch transfers**: Combine small transfers into larger ones
3. **Overlap transfer + compute**: Use CUDA streams
4. **Cache frequently accessed data**: Avoid repeated transfers

### Memory Pooling Benefits

**Without Pooling (direct cudaMalloc):**

```
cudaMalloc(10 MB)   → 500 μs  (kernel launch overhead)
cudaMalloc(10 MB)   → 500 μs
cudaMalloc(10 MB)   → 500 μs
Total: 1500 μs
```

**With Pooling (RMM):**

```
First allocation:    → 500 μs  (actual cudaMalloc)
Subsequent (reused): → 1 μs    (pool lookup)
Subsequent (reused): → 1 μs
Total: 502 μs (3x faster)
```

### Cache Hit Ratios

**Example Workload: TPC-H Query 5**

```
Dimension Tables (cached):
  - nation (25 rows, 1 KB) → 100% cache hit
  - region (5 rows, 0.5 KB) → 100% cache hit
  - supplier (1M rows, 50 MB) → 100% cache hit

Fact Tables (not cached):
  - lineitem (600M rows, 60 GB) → 0% cache hit (too large)
  - orders (150M rows, 15 GB) → 0% cache hit

Cache Hit Ratio: 3/5 tables, ~6% speedup from caching
```

### Memory Exhaustion Handling

**Current Behavior:**

- **GPU OOM**: Throws `OutOfMemoryException`, query fails
- **No automatic spilling**: Unlike New Mode, Legacy Mode doesn't spill to disk

**User Options:**

1. **Reduce batch size**: Scan fewer rows per batch
2. **Increase GPU memory**: Use larger GPU or multi-GPU
3. **Use New Mode**: Automatic spilling to disk (see [New Mode](../04-new-mode/overview.md))

**Example Configuration:**

```sql
-- Reduce memory pressure by limiting batch size
SET sirius_config.max_batch_size = 10000;  -- Default: 100000

-- Increase GPU memory limit (if available)
SET sirius_config.gpu_memory_limit = '40GB';  -- Default: auto-detect
```

---

## Next Steps

**Related Documentation:**

- **[Data Structures](data-structures.md)**: GPUColumn and GPUIntermediateRelation internals
- **[Operators](operators.md)**: How operators allocate and use memory
- **[Pipeline Execution](pipeline-execution.md)**: Memory flow through pipelines
- **[New Mode Memory Management](../05-core-components/memory-management.md)**: Compare with multi-tier system

**Comparison:**

- **[Execution Modes](../02-architecture/execution-modes.md)**: Legacy vs New memory management trade-offs

**For Developers:**

- **[Debugging](../07-development/debugging.md)**: Debugging memory issues
- **[Performance Tips](../appendices/performance-tips.md)**: Memory optimization strategies
