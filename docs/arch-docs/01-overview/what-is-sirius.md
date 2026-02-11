# What is Sirius?

## Introduction

Sirius is a **GPU-native SQL execution engine** designed to accelerate analytical query processing by leveraging the massive parallelism of modern GPUs. It integrates seamlessly with DuckDB, a popular in-process analytical database, allowing users to transparently offload query execution to the GPU.

## The Problem Sirius Solves

Traditional CPU-based database systems face performance bottlenecks when processing large-scale analytical queries:

- **Limited parallelism**: CPUs typically have 8-64 cores
- **Memory bandwidth**: CPU memory bandwidth is often the bottleneck for data-intensive operations
- **SIMD limitations**: CPU SIMD instructions provide limited vectorization

GPUs offer a compelling solution:

- **Massive parallelism**: Modern GPUs have thousands of cores
- **High memory bandwidth**: GPU memory bandwidth is 10-20x higher than CPU
- **Native vectorization**: GPU architecture is designed for data parallelism

However, leveraging GPUs for SQL processing requires specialized execution engines. **Sirius bridges this gap** by providing a complete GPU-native SQL execution infrastructure that plugs into DuckDB.

## How Sirius Works

### High-Level Architecture

```
User Application
       ↓
   DuckDB
       ↓
Sirius Extension ──→ GPU Execution
       ↓                    ↓
   Results   ←──────────────┘
```

Sirius operates as a **DuckDB extension**, exposing table functions that execute SQL queries on the GPU:

```sql
-- Legacy mode
SELECT * FROM gpu_processing('SELECT * FROM large_table WHERE x > 1000');

-- New mode (recommended)
SELECT * FROM gpu_execution('SELECT * FROM large_table WHERE x > 1000');
```

### Key Components

1. **Query Parser & Planner**: Translates SQL queries into physical execution plans
2. **GPU Operators**: Implements relational operators (scan, filter, join, aggregate) on GPU
3. **Pipeline Executor**: Orchestrates parallel execution of operator pipelines
4. **Memory Manager**: Manages data movement across GPU/HOST/DISK tiers
5. **Result Collector**: Materializes GPU results back to DuckDB format

## Core Technologies

Sirius is built on top of several key technologies:

### DuckDB
- **Role**: SQL parsing, planning, and result management
- **Integration**: Sirius extends DuckDB through the extension API
- **File**: `src/sirius_extension.cpp` contains the integration layer

### CUDA
- **Role**: GPU kernel execution and memory management
- **Usage**: All GPU operators compile to CUDA kernels
- **Version**: Requires CUDA 11.x or higher

### cuDF
- **Role**: GPU-accelerated DataFrame library (similar to Pandas)
- **Usage**: Core data structure for column operations
- **Integration**: Most operators use cuDF's high-level APIs

### RMM (RAPIDS Memory Manager)
- **Role**: GPU memory allocation and pooling
- **Usage**: Manages all GPU memory allocations
- **Integration**: Custom memory resources for multi-tier storage

### Cucascade
- **Role**: Task scheduling and data repository management
- **Usage**: New mode exclusively (not in legacy mode)
- **Purpose**: Efficient inter-pipeline data flow and memory management

## Two Execution Modes

Sirius supports two execution modes, reflecting its evolution:

### Legacy Mode (`gpu_processing`)
- **Status**: Older, maintenance mode
- **Entry**: `gpu_processing()` table function
- **Operators**: `GPUPhysicalOperator` base class
- **Memory**: `GPUBufferManager` singleton
- **Use Case**: Existing queries and backwards compatibility

### New Mode (`gpu_execution`)
- **Status**: Modern, actively developed
- **Entry**: `gpu_execution()` table function
- **Operators**: `sirius_physical_operator` base class
- **Memory**: Cucascade data repositories with multi-tier storage
- **Use Case**: New features and performance improvements

**Recommendation**: Use new mode for all new development. Legacy mode is maintained for compatibility but receives fewer updates.

## When to Use Sirius

### Ideal Use Cases

✅ **Large-scale analytical queries**
- TPC-H, TPC-DS style workloads
- Queries scanning millions/billions of rows
- Complex aggregations and joins

✅ **Column-oriented operations**
- Filters on numeric/date columns
- Aggregations (SUM, COUNT, AVG)
- Sort operations

✅ **GPU-friendly data types**
- Numeric types (INT, BIGINT, FLOAT, DOUBLE)
- Fixed-width types (DATE, TIMESTAMP)
- String operations (with caveats)

### Not Ideal For

❌ **Small queries (< 100K rows)**
- GPU kernel launch overhead dominates
- CPU execution is faster

❌ **Highly random access patterns**
- GPUs excel at sequential/strided access
- Random lookups perform poorly

❌ **Complex nested data structures**
- Nested JSON, deeply nested arrays
- GPU support is limited

❌ **Frequent small updates**
- Sirius is optimized for read-heavy OLAP
- Not designed for OLTP workloads

## Performance Characteristics

### What Makes Sirius Fast

1. **Parallel Execution**: Thousands of GPU threads process data simultaneously
2. **High Bandwidth**: GPU memory bandwidth (>1 TB/s) vs CPU (<100 GB/s)
3. **Vectorized Operations**: cuDF provides optimized kernels for columnar data
4. **Pipeline Fusion**: Multiple operators execute in fused pipelines reducing materialization
5. **Memory Tiers**: Intelligent spilling to host/disk prevents OOM errors

### Typical Speedups

Based on TPC-H benchmarks:

- **Simple filters/scans**: 2-5x speedup
- **Aggregations**: 3-10x speedup
- **Joins (hash joins)**: 5-20x speedup
- **Complex multi-join queries**: 10-50x speedup

> **Note**: Speedups depend on data size, query complexity, GPU model, and data transfer overhead.

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with compute capability 7.0+ (Volta or newer)
  - Recommended: RTX 3090, A100, H100
  - Minimum: GTX 1080, RTX 2060
- **GPU Memory**: 8GB minimum, 16GB+ recommended
- **CPU Memory**: 32GB+ for staging data

### Software
- **OS**: Linux (Ubuntu 20.04+, RHEL 8+)
- **CUDA**: 11.5 or higher
- **GCC**: 9.x or 10.x (for CUDA compatibility)
- **CMake**: 3.20+

### Dependencies
- DuckDB (embedded)
- cuDF and RAPIDS ecosystem
- RMM (RAPIDS Memory Manager)
- Cucascade (included as submodule)

## Sirius vs Other GPU Databases

| Feature | Sirius | BlazingSQL | OmniSci/HeavyDB |
|---------|--------|------------|------------------|
| Integration | DuckDB extension | Standalone | Standalone |
| SQL Support | Full DuckDB SQL | Partial | Full SQL |
| GPU Library | cuDF | cuDF | Custom kernels |
| Open Source | Yes | Yes (discontinued) | Partial |
| In-Process | Yes | No | No |
| Development | Active | Discontinued | Active (commercial) |

**Key Differentiators**:
- **DuckDB Integration**: Seamless integration means no data migration
- **In-Process**: No client-server overhead, embedded in applications
- **cuDF-based**: Leverages battle-tested RAPIDS ecosystem
- **Flexible**: Can fall back to CPU execution for unsupported operations

## Architecture Preview

Here's a simplified view of query execution in Sirius:

```
┌─────────────────────────────────────────────────────────────┐
│ User Query: SELECT SUM(price) FROM orders WHERE qty > 10    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ DuckDB: Parse → Logical Plan → Bind                         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Sirius Physical Planner: Generate GPU operator tree         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Pipeline Builder: Create execution pipelines                │
│ Pipeline 1: SCAN → FILTER → HASH_AGGREGATE                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ GPU Execution: Run pipelines on CUDA streams                │
│ - Allocate GPU memory                                        │
│ - Execute operators in parallel                              │
│ - Handle memory spilling if needed                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Result Collector: Transfer results to CPU                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ DuckDB: Return results to user                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts to Understand

Before diving deeper, familiarize yourself with these concepts:

1. **Physical Operators**: The building blocks of query execution (scan, filter, join, aggregate)
2. **Pipelines**: Chains of operators that execute together without materializing intermediate results
3. **Data Batches**: Column-oriented data structures (via cuDF) that flow through operators
4. **Memory Tiers**: GPU → Host → Disk hierarchy for managing larger-than-memory datasets
5. **Task Executors**: Thread pools that orchestrate parallel pipeline execution

For detailed explanations, see [Key Concepts](key-concepts.md).

## Example: Simple Query Execution

Let's trace a simple query through Sirius:

```sql
SELECT * FROM gpu_execution('
    SELECT customer_id, SUM(total_price) as total
    FROM orders
    WHERE order_date >= ''2024-01-01''
    GROUP BY customer_id
');
```

**Execution Flow**:

1. **Parse**: DuckDB parses the inner query
2. **Plan**: Sirius planner creates operator tree:
   ```
   RESULT_COLLECTOR
        ↓
   HASH_GROUP_BY (customer_id)
        ↓
   FILTER (order_date >= '2024-01-01')
        ↓
   TABLE_SCAN (orders)
   ```
3. **Pipeline**: Single pipeline: SCAN → FILTER → AGGREGATE → RESULT
4. **Execute**:
   - Allocate GPU memory for orders table
   - Scan rows in parallel on GPU
   - Apply filter predicate (vectorized)
   - Hash aggregate by customer_id
   - Transfer results to CPU
5. **Return**: DuckDB returns results to user

## Getting Started

Ready to dive deeper? Here are your next steps:

1. **Quick Start**: [Quick Start Guide](quick-start.md) - Build and run Sirius
2. **Key Concepts**: [Key Concepts](key-concepts.md) - Essential terminology
3. **Architecture**: [System Overview](../02-architecture/system-overview.md) - Detailed architecture
4. **Development**: [Building and Testing](../07-development/building-and-testing.md) - Development setup

## For Beginners

> **New to GPU computing?** Here are some background resources:
> - [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - NVIDIA official docs
> - [cuDF Documentation](https://docs.rapids.ai/api/cudf/stable/) - RAPIDS DataFrame library
> - [DuckDB Documentation](https://duckdb.org/docs/) - DuckDB SQL engine
> - [Column-Oriented Databases](https://en.wikipedia.org/wiki/Column-oriented_DBMS) - Background on columnar storage

## Summary

Sirius is a GPU-accelerated SQL execution engine that:
- Integrates seamlessly with DuckDB
- Leverages CUDA, cuDF, and RMM for GPU execution
- Provides massive speedups for analytical queries
- Supports two execution modes (legacy and modern)
- Handles larger-than-memory datasets with multi-tier storage

**Next**: Learn the essential concepts in [Key Concepts](key-concepts.md).
