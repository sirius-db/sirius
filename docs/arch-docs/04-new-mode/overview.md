# New Mode Overview

This document provides a comprehensive overview of Sirius's New Mode execution engine, accessed via the `gpu_execution()` table function.

## Status and Development

**Current Status**: ✅ **Active Development** - Recommended for all new work

- ✅ Modern architecture with advanced features
- ✅ Actively maintained and improved
- ✅ Better performance (1.2-1.6x vs Legacy)
- ✅ Scales beyond GPU memory with multi-tier storage
- 🚀 All new features developed here

## What is New Mode?

New Mode is Sirius's modern GPU execution engine, redesigned from the ground up with:
- **Dynamic task-based execution** for better GPU utilization
- **Multi-tier memory management** (GPU → HOST → DISK) for large datasets
- **Cucascade integration** for efficient data flow
- **Asynchronous execution** with CUDA streams
- **Port-based pipeline communication** for flexibility

### Key Characteristics

| Aspect | Description |
|--------|-------------|
| **Entry Point** | `gpu_execution()` table function |
| **Operator Base** | `sirius_physical_operator` |
| **Data Structure** | `cucascade::data_batch` |
| **Memory Model** | Multi-tier with automatic spilling |
| **Pipeline Model** | Dynamic task creation with hints |
| **Execution** | Asynchronous with CUDA streams |

---

## When to Use New Mode

### Primary Use Cases

✅ **All New Development**
- Default choice for new queries and features
- Better performance and scalability

✅ **Large Datasets**
- Handles datasets larger than GPU memory
- Automatic spilling to HOST/DISK

✅ **Complex Queries**
- Better scheduling and parallelism
- Advanced operators (window functions, CTEs)

✅ **Production Workloads**
- More robust error handling
- Better resource management

### Advantages Over Legacy Mode

| Feature | Legacy Mode | New Mode |
|---------|-------------|----------|
| **Performance** | Baseline | 🚀 1.2-1.6x faster |
| **Memory Scalability** | GPU-limited | ✅ Unlimited (spilling) |
| **Task Scheduling** | Static | ✅ Dynamic with hints |
| **Parallelism** | Limited | ✅ Better concurrency |
| **Development** | Maintenance only | ✅ Active |

---

## High-Level Architecture

### Component Overview

```
┌───────────────────────────────────────────────────────────┐
│                   User Application                         │
└───────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────┐
│                      DuckDB                                │
└───────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────┐
│          gpu_execution() Table Function                    │
│          File: src/sirius_extension.cpp:353-452            │
└───────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────┐
│         Sirius Physical Plan Generator                     │
│         File: src/planner/sirius_physical_plan_generator.cpp│
└───────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────┐
│         sirius_physical_operator Tree                      │
│         File: src/include/op/sirius_physical_operator.hpp  │
└───────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────┐
│         Sirius Engine                                      │
│         File: src/sirius_engine.cpp                        │
└───────────────────────────────────────────────────────────┘
                           ↓
    ┌──────────────────────┴───────────────────────┐
    ↓                      ↓                        ↓
┌─────────┐        ┌──────────────┐        ┌──────────────┐
│Pipeline │        │ Task         │        │ Data         │
│Builder  │   →    │ Executors    │   ←→   │ Repositories │
└─────────┘        └──────────────┘        └──────────────┘
    ↓                      ↓                        ↓
┌───────────────────────────────────────────────────────────┐
│         CUDA / cuDF / Cucascade / GPU Hardware             │
└───────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Entry Point: gpu_execution()
- **File**: `src/sirius_extension.cpp:353-452`
- **Purpose**: Table function accepting SQL query string
- **Binding**: Parses query and generates Sirius physical plan
- **Execution**: Runs plan via sirius_engine

#### 2. Sirius Physical Plan Generator
- **File**: `src/planner/sirius_physical_plan_generator.cpp`
- **Purpose**: Convert logical plan to sirius_physical_operator tree
- **Output**: Tree of sirius_physical_operator objects
- **Features**: Type resolution, operator-specific planning

#### 3. sirius_physical_operator
- **File**: `src/include/op/sirius_physical_operator.hpp`
- **Purpose**: Base class for all operators
- **Methods**: `execute()`, `sink()`, `get_next_task_hint()`
- **Examples**: All operators in `src/op/` directory

#### 4. Sirius Engine
- **File**: `src/sirius_engine.cpp`
- **Purpose**: Main execution coordinator
- **Responsibilities**:
  - Initialize pipelines and repositories
  - Create and schedule tasks
  - Manage execution state

#### 5. Task Executors
- **File**: `src/include/parallel/task_executor.hpp`
- **Types**:
  - `pipeline_executor` - GPU task execution
  - `task_creator` - Dynamic task generation
  - `downgrade_executor` - Memory tier management
  - `duckdb_scan_executor` - CPU scans

#### 6. Cucascade Data Repositories
- **Files**: `cucascade/include/cucascade/data/`
- **Purpose**: Inter-pipeline data exchange
- **Features**: Multi-tier storage, thread-safe queues

---

## Execution Flow

### Step-by-Step Query Execution

```sql
SELECT category, SUM(price)
FROM products
WHERE price > 100
GROUP BY category
ORDER BY total DESC
```

#### Phase 1: Parse and Plan (DuckDB)
```
SQL String → Parser → Logical Plan

LogicalOrder [total DESC]
     ↓
LogicalProjection [category, SUM(price) as total]
     ↓
LogicalAggregate [GROUP BY category, SUM(price)]
     ↓
LogicalFilter [price > 100]
     ↓
LogicalGet [products]
```

#### Phase 2: Physical Planning (Sirius)
**File**: `src/planner/sirius_physical_plan_generator.cpp`

```
Logical Plan → Sirius Physical Planner → Physical Operator Tree

RESULT_COLLECTOR
     ↓
ORDER_BY [total DESC]
     ↓
HASH_GROUP_BY [category, SUM(price)]
     ↓
FILTER [price > 100]
     ↓
DUCKDB_SCAN [products]
```

#### Phase 3: Initialize Engine
**File**: `src/sirius_engine.cpp`

```
sirius_engine::initialize():
  1. Build pipelines from operator tree
  2. Create data repositories for inter-pipeline communication
  3. Establish port connections (input/output)
  4. Initialize task executors
  5. Prepare CUDA streams

Pipelines Created:
  Pipeline 1: SCAN → FILTER → HASH_GROUP_BY (sink)
  Pipeline 2: HASH_GROUP_BY (source) → ORDER_BY (sink)
  Pipeline 3: ORDER_BY (source) → RESULT_COLLECTOR

Data Repositories:
  Repository 1: Pipeline 1 → Pipeline 2 (aggregate results)
  Repository 2: Pipeline 2 → Pipeline 3 (sorted results)
```

#### Phase 4: Execute
**File**: `src/sirius_engine.cpp`

```
sirius_engine::execute():
  1. task_creator generates initial tasks
  2. pipeline_executor schedules tasks on CUDA streams
  3. Tasks execute when inputs ready (hint: READY)
  4. Results flow through repositories
  5. Dependent pipelines create tasks when data available

Task Execution Flow:
  Task 1.1: Scan batch 1 → Filter → Aggregate (partial)
  Task 1.2: Scan batch 2 → Filter → Aggregate (partial)
  ...
  Task 1.N: Finalize aggregate → push to Repository 1
           [Pipeline 1 complete, Repository 1 has data]
  Task 2.1: Pull from Repository 1 → Sort → push to Repository 2
           [Pipeline 2 complete, Repository 2 has data]
  Task 3.1: Pull from Repository 2 → Collect results
           [Pipeline 3 complete, return to DuckDB]
```

#### Phase 5: Result Collection
```
GPU Data → Result Collector → CPU DataChunk → DuckDB → User
```

---

## Key Innovations

### 1. Dynamic Task Creation with Hints

Operators provide hints about readiness:

```cpp
enum class TaskCreationHint {
    READY,                  // Task can execute now
    WAITING_FOR_INPUT_DATA  // Task blocked on input
};

task_creation_hint get_next_task_hint() override {
    if (input_port->has_data()) {
        return {TaskCreationHint::READY, nullptr};
    }
    return {TaskCreationHint::WAITING_FOR_INPUT_DATA, upstream_producer};
}
```

**Benefits**:
- No wasted task creation for blocked operators
- Better GPU utilization
- Automatic scheduling based on data availability

### 2. Port-Based Communication

Pipelines communicate via ports connected to data repositories:

```cpp
// Producer pipeline
output_port->push_data_batch(batch);

// Consumer pipeline
auto batch = input_port->pull_batch();
```

**Advantages**:
- Decoupled producers/consumers
- Thread-safe multi-producer/multi-consumer
- Automatic memory tier management

### 3. Multi-Tier Memory Management

Data automatically migrates across tiers:

```
┌─────────────────┐
│   GPU Memory    │  Active processing
│   (8-80 GB)     │
└─────────────────┘
        ↕ automatic downgrade/upgrade
┌─────────────────┐
│  HOST Memory    │  Staging area
│  (32-512 GB)    │
└─────────────────┘
        ↕ automatic spill/load
┌─────────────────┐
│  DISK Storage   │  Overflow
│  (Unlimited)    │
└─────────────────┘
```

**Benefits**:
- Handle datasets larger than GPU memory
- Automatic eviction and promotion
- No OOM errors (spill to disk)

### 4. Asynchronous Execution

Operations execute asynchronously on CUDA streams:

```cpp
auto output_batches = operator->execute(input_batches, cuda_stream);
// Don't wait - submit next task
```

**Advantages**:
- Overlap data transfer and computation
- Multiple pipelines execute concurrently
- Better GPU utilization

---

## Data Flow

### Cucascade data_batch Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│ Source Operator (e.g., DUCKDB_SCAN)                     │
│ Creates shared_ptr<data_batch>                          │
└─────────────────────────────────────────────────────────┘
                       ↓
         shared_ptr<data_batch> (cuDF table)
         • Columnar data
         • Memory location (GPU/HOST/DISK)
         • Reference counted
                       ↓
┌─────────────────────────────────────────────────────────┐
│ Processing Operator (e.g., FILTER)                      │
│ execute() → returns vector<shared_ptr<data_batch>>      │
└─────────────────────────────────────────────────────────┘
                       ↓
         shared_ptr<data_batch> (filtered)
                       ↓
┌─────────────────────────────────────────────────────────┐
│ Sink Operator (e.g., HASH_GROUP_BY)                     │
│ sink() → accumulates into internal state                │
└─────────────────────────────────────────────────────────┘
                       ↓
         Internal state (hash table)
                       ↓
┌─────────────────────────────────────────────────────────┐
│ Output via Port                                          │
│ push_data_batch() → to data_repository                  │
└─────────────────────────────────────────────────────────┘
                       ↓
         data_repository (multi-tier storage)
                       ↓
┌─────────────────────────────────────────────────────────┐
│ Consumer Pipeline                                        │
│ pull_batch() → from data_repository                     │
└─────────────────────────────────────────────────────────┘
```

### Inter-Pipeline Communication

```
Producer Pipeline              Data Repository           Consumer Pipeline
─────────────────             ──────────────────        ─────────────────
1. Execute operator
2. Create data_batch
3. output_port->push()  ──→   Stores in GPU/HOST/DISK
                                      │
                                      │ (automatic tier mgmt)
                                      │
                                      ↓
4. get_next_task_hint()         has_data() = true
   returns READY                      │
                                      ↓
5. pull_batch()         ←──────  Returns data_batch
6. Execute operator
```

---

## Pipeline Model

### sirius_pipeline Structure

**File**: `src/include/pipeline/sirius_pipeline.hpp`

```cpp
class sirius_pipeline {
public:
    // Operators in this pipeline
    vector<unique_ptr<sirius_physical_operator>> operators;

    // Communication ports
    vector<input_port> input_ports;
    vector<output_port> output_ports;

    // Dynamic task creation
    unique_ptr<sirius_pipeline_itask> create_next_task();
    task_creation_hint get_task_hint();

    // Execution
    void execute(cuda_stream_view stream);
};
```

### sirius_meta_pipeline (DAG)

**File**: `src/include/pipeline/sirius_meta_pipeline.hpp`

```cpp
class sirius_meta_pipeline {
public:
    // All pipelines
    vector<unique_ptr<sirius_pipeline>> pipelines;

    // Data repositories for communication
    vector<shared_ptr<data_repository>> repositories;

    // Port connections
    void connect_ports();

    // Initialize and execute
    void initialize();
    void execute();
};
```

### Pipeline Breaks

Same as Legacy Mode - certain operators require pipeline breaks:
- **Hash Joins**: Build before probe
- **Aggregates**: All input before output
- **Sorts**: Complete data for sorting

**Difference**: In New Mode, communication via repositories enables better parallelism.

---

## Task Execution Model

### Task Interface

**File**: `src/include/pipeline/sirius_pipeline_itask.hpp`

```cpp
class sirius_pipeline_itask {
public:
    // Execute task on CUDA stream
    virtual void compute_task(cuda_stream_view stream) = 0;

    // Publish results
    virtual void publish_output() = 0;

    // Check if task complete
    virtual bool is_complete() = 0;
};
```

### Task Creation Flow

```
1. task_creator calls pipeline->get_task_hint()
       ↓
2. If READY: pipeline->create_next_task()
       ↓
3. task_creator submits to pipeline_executor
       ↓
4. pipeline_executor assigns CUDA stream
       ↓
5. Task executes: compute_task(stream)
       ↓
6. Task publishes: publish_output()
       ↓
7. Dependent pipelines get READY hint
       ↓
8. Repeat for dependent pipelines
```

---

## Operator Categories

### Sources (Produce Data)
- **DUCKDB_SCAN**: Read from DuckDB tables (preferred)
- **TABLE_SCAN**: Legacy table scan
- **DUMMY_SCAN**: Testing placeholder

### Transforms (Process Data)
- **FILTER**: Apply WHERE predicates
- **PROJECTION**: Select/compute columns
- **LIMIT**: Restrict row count

### Aggregates (Accumulate Data)
- **UNGROUPED_AGGREGATE**: SUM/COUNT without grouping
- **HASH_GROUP_BY**: Grouped aggregation

### Joins (Combine Datasets)
- **HASH_JOIN**: Equi-join via hash table
- **NESTED_LOOP_JOIN**: Non-equi joins

### Sorts (Ordering)
- **ORDER_BY**: Full sort
- **TOP_N**: Partial sort with limit
- **MERGE_SORT**: Multi-way merge

### Partitioning
- **PARTITION**: Partition data by key
- **SORT_PARTITION**: Partition with ordering

### Output
- **RESULT_COLLECTOR**: Materialize to DuckDB format

---

## Configuration

### Engine Configuration

```ini
# Thread pool sizes
pipeline_executor_threads=4
task_creator_threads=2
downgrade_executor_threads=2
duckdb_scan_executor_threads=4

# Memory tiers (MB)
gpu_memory_limit=8192
host_memory_limit=32768
disk_memory_limit=-1  # Unlimited

# CUDA streams
cuda_streams_per_executor=1

# Logging
log_level=INFO
```

### Per-Pipeline Configuration

```cpp
// Configure data repository
repository->configure_tiers(
    gpu_limit_mb: 2048,
    host_limit_mb: 8192,
    disk_limit_mb: -1
);
```

---

## Performance Characteristics

### Advantages

✅ **Dynamic Scheduling**: Better GPU utilization via hint-based task creation
✅ **Asynchronous Execution**: Overlap computation and data transfer
✅ **Memory Scalability**: Handle datasets >> GPU memory
✅ **Better Parallelism**: Multiple pipelines execute concurrently

### Typical Performance Gains

Based on TPC-H SF10 (NVIDIA A100):

| Query Type | Speedup vs Legacy |
|------------|-------------------|
| Scan + Filter | 1.14x |
| Aggregation | 1.37x |
| Hash Join | 1.54x |
| Multi-Join | 1.64x |
| Complex (Q9, Q18) | 1.5-1.7x |

**Why Faster?**
1. Better task scheduling reduces idle time
2. Asynchronous execution overlaps operations
3. Multi-tier memory prevents OOM re-execution
4. Improved data locality via repositories

---

## Example: Complete Query Flow

```sql
SELECT * FROM gpu_execution('
    SELECT category, COUNT(*) as count
    FROM products
    WHERE price > 50
    GROUP BY category
    ORDER BY count DESC
');
```

**Execution Trace**:

```
1. GPUExecutionBind()
   ├─ Parse query
   ├─ Logical Plan: Get → Filter → Aggregate → Order
   └─ Physical Plan: DUCKDB_SCAN → FILTER → HASH_GROUP_BY → ORDER_BY → RESULT

2. sirius_engine::initialize()
   ├─ Pipeline 1: SCAN → FILTER → HASH_GROUP_BY (sink)
   ├─ Pipeline 2: HASH_GROUP_BY (source) → ORDER_BY (sink)
   ├─ Pipeline 3: ORDER_BY (source) → RESULT_COLLECTOR
   ├─ Repository 1: Pipeline 1 output → Pipeline 2 input
   └─ Repository 2: Pipeline 2 output → Pipeline 3 input

3. sirius_engine::execute()
   ├─ task_creator: create tasks for Pipeline 1 (hint: READY)
   │   └─ pipeline_executor: execute tasks on CUDA streams
   │       ├─ Task 1: Scan batch 1 → Filter → Aggregate (partial)
   │       ├─ Task 2: Scan batch 2 → Filter → Aggregate (partial)
   │       └─ Task N: Finalize → push to Repository 1
   │
   ├─ task_creator: create tasks for Pipeline 2 (hint: READY after Repository 1 has data)
   │   └─ pipeline_executor: execute tasks
   │       └─ Task: Pull from Repository 1 → Sort → push to Repository 2
   │
   └─ task_creator: create tasks for Pipeline 3 (hint: READY after Repository 2 has data)
       └─ pipeline_executor: execute tasks
           └─ Task: Pull from Repository 2 → Collect → return to DuckDB

4. Return Results
   └─ Convert cuDF → DuckDB DataChunk → User
```

---

## Key Files Reference

| Component | File | Description |
|-----------|------|-------------|
| Entry Point | `src/sirius_extension.cpp:353-452` | gpu_execution() implementation |
| Engine | `src/sirius_engine.cpp` | Main execution coordinator |
| Interface | `src/sirius_interface.cpp` | Query interface layer |
| Operator Base | `src/include/op/sirius_physical_operator.hpp` | Base class |
| Operators | `src/op/sirius_physical_*.cpp` | All operator implementations |
| Planner | `src/planner/sirius_physical_plan_generator.cpp` | Physical planning |
| Pipeline | `src/include/pipeline/sirius_pipeline.hpp` | Pipeline structure |
| Task | `src/include/pipeline/sirius_pipeline_itask.hpp` | Task interface |
| Executors | `src/include/parallel/task_executor.hpp` | Task executors |
| Memory | `src/include/memory/sirius_memory_reservation_manager.hpp` | Memory management |
| Cucascade | `cucascade/include/cucascade/` | Data repositories |

---

## Next Steps

Now that you understand New Mode at a high level:

1. **Entry Points**: [New Mode Entry Points](entry-points.md) - Deep dive into `gpu_execution()`
2. **Operators**: [New Mode Operators](operators.md) - sirius_physical_operator details
3. **Cucascade**: [Cucascade Integration](cucascade-integration.md) - Data repositories and memory
4. **Pipelines**: [Pipeline Execution](pipeline-execution.md) - Dynamic task model
5. **Task Creation**: [Task Creation](task-creation.md) - Hint-based scheduling
6. **Operator Guide**: [Operator Guide](operator-guide.md) - Comprehensive operator reference

For comparison with the older system, see [Legacy Mode Overview](../03-legacy-mode/overview.md).
