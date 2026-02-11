# Key Concepts

This document defines the essential concepts and terminology used throughout Sirius. Refer back to this page when encountering unfamiliar terms.

## Table of Contents
- [Execution Model](#execution-model)
- [Data Structures](#data-structures)
- [Memory Management](#memory-management)
- [Operators](#operators)
- [Planning](#planning)
- [Threading and Parallelism](#threading-and-parallelism)

---

## Execution Model

### Pipeline
A **pipeline** is a chain of operators that execute together without materializing intermediate results. Pipelines maximize performance by:
- Reducing memory pressure (no intermediate materialization)
- Enabling operator fusion
- Allowing parallel execution across data partitions

**Example**: `SCAN → FILTER → PROJECT` forms a single pipeline where data flows directly from one operator to the next.

**Files**:
- Legacy: `src/include/gpu_pipeline.hpp`
- New: `src/include/pipeline/sirius_pipeline.hpp`

### Meta Pipeline
A **meta pipeline** represents a collection of pipelines with dependencies. It manages:
- Pipeline ordering (DAG of dependencies)
- Resource allocation
- Parallel execution scheduling

Think of it as the "execution plan coordinator" that ensures pipelines execute in the correct order.

**Files**:
- Legacy: `src/include/gpu_meta_pipeline.hpp`
- New: `src/include/pipeline/sirius_meta_pipeline.hpp`

### Task
A **task** is a unit of work that executes on a thread pool. In new mode, tasks are dynamically created based on:
- Data availability (hint: `READY` or `WAITING_FOR_INPUT_DATA`)
- Resource availability (GPU memory, CUDA streams)
- Pipeline dependencies

**File**: `src/include/pipeline/sirius_pipeline_itask.hpp`

### Source, Operator, Sink
Operators in a pipeline have specific roles:

- **Source**: Produces data (e.g., TABLE_SCAN, DUCKDB_SCAN)
  - No input, generates output batches
  - Entry point of a pipeline

- **Operator**: Transforms data (e.g., FILTER, PROJECTION, JOIN probe side)
  - Takes input batches, produces output batches
  - Executes in the middle of a pipeline

- **Sink**: Accumulates data (e.g., HASH_JOIN build side, AGGREGATE, RESULT_COLLECTOR)
  - Takes input batches, builds internal state
  - Typically ends a pipeline (may start a new one downstream)

**Example Pipeline**:
```
[Source] TABLE_SCAN → [Operator] FILTER → [Sink] HASH_AGGREGATE
```

---

## Data Structures

### Data Batch (New Mode)
A **`cucascade::data_batch`** is the fundamental data unit in new mode. It represents:
- A column-oriented batch of rows (typically backed by cuDF)
- Metadata (schema, row count, memory location)
- Ownership semantics (shared_ptr for efficient passing)

**Key Properties**:
- Immutable once created (no in-place modification)
- Can reside in GPU, HOST, or DISK memory
- Passed between operators via shared pointers

**File**: `cucascade/include/cucascade/data/data_batch.hpp`

### GPUIntermediateRelation (Legacy Mode)
A **`GPUIntermediateRelation`** is the legacy equivalent of data_batch. It contains:
- Vector of `GPUColumn` objects
- Row count
- Schema information

**File**: `src/include/gpu_columns.hpp`

### GPUColumn (Legacy Mode)
A **`GPUColumn`** represents a single column of data on the GPU:
- Typed data pointer (via cuDF column)
- Null mask (for handling NULL values)
- Data type information

**File**: `src/include/gpu_columns.hpp`

### Data Repository (New Mode)
A **`cucascade::data_repository`** is a staging area for inter-pipeline communication:
- Producers push data batches
- Consumers pull data batches
- Supports multi-tier storage (GPU → HOST → DISK)
- Thread-safe for concurrent access

**Key Distinction**: Unlike pipes/queues, repositories can store batches persistently, enabling complex execution patterns like joins where one side needs to be fully materialized before probing.

**File**: `cucascade/include/cucascade/data/data_repository.hpp`

---

## Memory Management

### Memory Tiers
Sirius manages data across three memory tiers:

1. **GPU Memory** (fastest, smallest)
   - Allocated via RMM device_memory_resource
   - Typical size: 8-80 GB
   - Used for active processing

2. **Host Memory** (medium speed, larger)
   - Allocated via RMM host_memory_resource
   - Pinned memory for efficient GPU transfers
   - Typical size: 32-512 GB

3. **Disk Storage** (slowest, largest)
   - Fallback for spilling
   - Unlimited size (constrained by disk)

**File**: `src/include/memory/sirius_memory_reservation_manager.hpp`

### Memory Reservation
Before allocating GPU memory, operators **reserve** memory to ensure availability:
- `reserve(size)` - Request memory allocation
- If reservation fails, trigger **downgrade** (spill to host/disk)
- Prevents OOM crashes

**Downgrade Process**:
```
GPU full → Identify eviction candidates → Copy to Host → Free GPU memory
Host full → Copy to Disk → Free Host memory
```

**File**: `src/include/memory/sirius_memory_reservation_manager.hpp`

### RMM (RAPIDS Memory Manager)
**RMM** provides:
- Custom CUDA memory allocators
- Memory pooling (reduces allocation overhead)
- Multi-tier memory resources
- Statistics and profiling

Sirius configures RMM with custom memory resources to enable spilling.

---

## Operators

### Physical Operator
A **physical operator** is an execution node in the physical query plan. It implements:
- `execute()` - Process input batches and produce output (new mode)
- `Execute()` - Transform input relation to output (legacy mode)
- `sink()` - Accumulate input data into state (new mode)
- `Sink()` - Accumulate input relation (legacy mode)

**Base Classes**:
- Legacy: `GPUPhysicalOperator` (`src/include/gpu_physical_operator.hpp`)
- New: `sirius_physical_operator` (`src/include/op/sirius_physical_operator.hpp`)

### Operator Types
Common operator types:

**Scans**:
- **TABLE_SCAN**: Read from DuckDB table (legacy)
- **DUCKDB_SCAN**: Read from DuckDB using native API (new)
- **DUMMY_SCAN**: Test/placeholder operator

**Filters and Projections**:
- **FILTER**: Apply WHERE clause predicates
- **PROJECTION**: Select/compute columns

**Aggregates**:
- **UNGROUPED_AGGREGATE**: SUM/COUNT/AVG without GROUP BY
- **HASH_GROUP_BY**: Grouped aggregation with hash table

**Joins**:
- **HASH_JOIN**: Equi-join using hash table (build + probe)
- **NESTED_LOOP_JOIN**: Cartesian product or non-equi joins

**Sorting**:
- **ORDER_BY**: Sort result set
- **TOP_N**: Partial sort (LIMIT optimization)

**Output**:
- **RESULT_COLLECTOR**: Materialize final results to DuckDB format

### Operator Ports (New Mode)
**Ports** enable inter-pipeline communication:
- **Input Ports**: Receive data from upstream pipelines (via data repositories)
- **Output Ports**: Send data to downstream pipelines (via data repositories)

Example:
```
Pipeline 1: SCAN → FILTER → HASH_JOIN (build)
                                ↓ (output port)
                          data_repository
                                ↓ (input port)
Pipeline 2: HASH_JOIN (probe) → PROJECT → RESULT
```

**File**: Defined in individual operator headers, e.g., `src/include/op/sirius_physical_hash_join.hpp`

---

## Planning

### Logical Plan
A **logical plan** represents what to compute, not how:
- Produced by DuckDB's planner
- Uses logical operators (LogicalFilter, LogicalAggregate, etc.)
- Database-agnostic

**Example**:
```
LogicalProjection [customer_id, total]
    ↓
LogicalAggregate [GROUP BY customer_id, SUM(price)]
    ↓
LogicalFilter [date > '2024-01-01']
    ↓
LogicalGet [orders table]
```

### Physical Plan
A **physical plan** represents how to execute:
- Produced by Sirius's physical planner
- Uses physical operators (TABLE_SCAN, FILTER, HASH_AGGREGATE)
- GPU-specific optimizations applied

**Example** (corresponding to logical plan above):
```
RESULT_COLLECTOR
    ↓
HASH_GROUP_BY [customer_id]
    ↓
FILTER [date > '2024-01-01']
    ↓
TABLE_SCAN [orders]
```

**Files**:
- Legacy: `src/gpu_physical_plan_generator.cpp`
- New: `src/planner/sirius_physical_plan_generator.cpp`

### Pipeline Breaking
Some operators require **pipeline breaks** (materialize intermediate results):

**Pipeline Breakers**:
- **Hash Joins**: Build side must complete before probe side starts
- **Aggregates**: All input must be seen before producing output
- **Sorts**: All data must be present to sort

**Non-Breaking Operators**:
- Filters, projections, limits (streaming)

---

## Threading and Parallelism

### Task Executor
A **task executor** is a thread pool that processes tasks:

**Types** (new mode):
1. **pipeline_executor**: Executes GPU pipeline tasks
2. **task_creator**: Converts plans to tasks
3. **downgrade_executor**: Handles memory spilling
4. **duckdb_scan_executor**: Offloads DuckDB scans to CPU

Each executor has:
- Configurable thread pool size
- Work-stealing queue
- CUDA stream pool (for GPU executors)

**File**: `src/include/parallel/task_executor.hpp`

### CUDA Stream
A **CUDA stream** is a sequence of GPU operations:
- Operations in same stream execute sequentially
- Operations in different streams execute concurrently
- Sirius uses stream pool for parallel pipeline execution

**Key Benefit**: Multiple pipelines can execute on GPU simultaneously using different streams.

### Task Creation Hints
In new mode, operators provide **hints** about task readiness:

- **`READY`**: Task can execute now (input data available)
- **`WAITING_FOR_INPUT_DATA`**: Task blocked on upstream pipeline

Hint system enables **dynamic task scheduling**:
```cpp
task_creation_hint get_next_task_hint() {
    if (input_port->has_data()) {
        return {TaskCreationHint::READY, nullptr};
    }
    return {TaskCreationHint::WAITING_FOR_INPUT_DATA, upstream_producer};
}
```

**File**: `src/include/op/sirius_physical_operator.hpp` (lines 53-60)

---

## Configuration

### Sirius Config
**`sirius_config`** holds global configuration:
- Thread pool sizes
- Memory limits (GPU/HOST/DISK)
- Logging levels
- Hardware topology (GPU count, NUMA nodes)

Loaded from config file at startup.

**File**: `src/sirius_config.cpp`

### Sirius Context
**`SiriusContext`** is per-connection state:
- Associated with a DuckDB ClientContext
- Stores runtime state (memory reservations, active pipelines)
- Thread-safe for concurrent queries

**File**: `src/sirius_context.cpp`

---

## Key Terminology Quick Reference

| Term | Definition | Mode |
|------|------------|------|
| **Pipeline** | Chain of operators executing together | Both |
| **Task** | Unit of work on thread pool | New |
| **Data Batch** | Column-oriented data chunk | New |
| **Data Repository** | Inter-pipeline staging area | New |
| **Port** | Connection point between pipelines | New |
| **Memory Tier** | GPU/HOST/DISK storage level | Both |
| **Downgrade** | Spill data to lower memory tier | New |
| **Task Executor** | Thread pool for parallel execution | New |
| **Physical Operator** | Execution node in plan | Both |
| **Source** | Data-producing operator | Both |
| **Sink** | Data-accumulating operator | Both |

---

## Advanced Concepts

### Operator Fusion
**Fusion** combines multiple operators into a single GPU kernel:
- Reduces memory traffic
- Improves cache locality
- Only possible for streaming operators (no materialization)

**Example**: SCAN + FILTER + PROJECT can fuse into single kernel that reads, filters, and projects in one pass.

### Batched Execution
Rather than processing entire dataset at once, Sirius processes data in **batches**:
- Batch size configurable (default ~100K rows)
- Enables pipelining
- Reduces memory footprint

### Expression Evaluation
**Expressions** (arithmetic, predicates) are evaluated using cuDF:
- Compiled to GPU kernels at runtime
- Vectorized execution across entire columns
- Null handling built-in

**File**: `src/expression/gpu_expression_executor.cpp`

---

## For Beginners

> **New to columnar databases?**
> - **Row-oriented**: Traditional databases store entire rows together
> - **Column-oriented**: Sirius stores each column separately
> - **Why?** Analytical queries often access few columns but many rows. Columnar layout improves cache locality and enables vectorized operations.

> **New to pipeline execution?**
> - Think of an assembly line: each operator is a station, data flows through
> - No intermediate storage between operators (unless pipeline break)
> - Multiple "lines" (pipelines) can run in parallel

---

## Next Steps

Now that you understand the key concepts:
1. Review [System Overview](../02-architecture/system-overview.md) to see how components fit together
2. Explore [Execution Modes](../02-architecture/execution-modes.md) to compare legacy vs new
3. Deep dive into [New Mode Overview](../04-new-mode/overview.md) for modern execution

For quick reference, bookmark the [Glossary](../08-reference/glossary.md).
