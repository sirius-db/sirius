# Legacy Mode Overview

This document provides an overview of Sirius's Legacy Mode execution engine, accessed via the `gpu_processing()` table function.

## Status and Maintenance

**Current Status**: Maintenance mode
- ✅ Stable and battle-tested
- ⚠️ No new features being added
- 🔧 Bug fixes and critical updates only
- 🚀 New Mode (`gpu_execution`) recommended for new development

## What is Legacy Mode?

Legacy Mode is Sirius's original GPU execution engine, developed before the introduction of cucascade and the modern task-based execution model. It provides GPU-accelerated SQL execution through a simpler, more straightforward architecture.

### Key Characteristics

| Aspect | Description |
|--------|-------------|
| **Entry Point** | `gpu_processing()` table function |
| **Operator Base** | `GPUPhysicalOperator` |
| **Data Structure** | `GPUIntermediateRelation` |
| **Memory Model** | `GPUBufferManager` singleton |
| **Pipeline Model** | Static pipeline construction |
| **Execution** | Synchronous operator execution |

## When to Use Legacy Mode

### Use Cases

✅ **Maintenance of Existing Queries**
- Queries already using `gpu_processing()`
- Legacy applications depending on specific behavior

✅ **Backwards Compatibility**
- Testing against old baseline
- Verifying behavior parity

✅ **Debugging and Comparison**
- Simpler codebase for understanding concepts
- Comparing performance against new mode

### When NOT to Use

❌ **New Development** - Use New Mode instead
❌ **Advanced Features** - Limited operator support
❌ **Large Datasets** - No automatic spilling
❌ **Complex Queries** - Less sophisticated scheduling

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
│          gpu_processing() Table Function                   │
│          File: src/sirius_extension.cpp:240-339            │
└───────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────┐
│         GPU Physical Plan Generator                        │
│         File: src/gpu_physical_plan_generator.cpp          │
└───────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────┐
│         GPUPhysicalOperator Tree                           │
│         File: src/include/gpu_physical_operator.hpp        │
└───────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────┐
│         GPU Executor                                       │
│         File: src/gpu_executor.cpp                         │
└───────────────────────────────────────────────────────────┘
                           ↓
         ┌─────────────────┴──────────────────┐
         ↓                                     ↓
┌──────────────────────┐          ┌──────────────────────┐
│   GPU Pipelines      │          │  GPUBufferManager    │
│   (Execution)        │   ←──→   │  (Memory)            │
└──────────────────────┘          └──────────────────────┘
         ↓                                     ↓
┌───────────────────────────────────────────────────────────┐
│            CUDA / cuDF / GPU Hardware                      │
└───────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Entry Point: gpu_processing()
- **File**: `src/sirius_extension.cpp:240-339`
- **Purpose**: Table function that accepts SQL query string
- **Binding**: Parses query and generates GPU physical plan
- **Execution**: Runs plan on GPU and returns results

#### 2. GPU Physical Plan Generator
- **File**: `src/gpu_physical_plan_generator.cpp`
- **Purpose**: Convert DuckDB logical plan to GPU physical operators
- **Output**: Tree of `GPUPhysicalOperator` objects

#### 3. GPUPhysicalOperator
- **File**: `src/include/gpu_physical_operator.hpp`
- **Purpose**: Base class for all GPU operators
- **Implements**: Source, Operator, and Sink interfaces
- **Examples**: TABLE_SCAN, FILTER, HASH_JOIN, AGGREGATE

#### 4. GPU Executor
- **File**: `src/gpu_executor.cpp`
- **Purpose**: Orchestrates pipeline execution
- **Responsibilities**:
  - Build pipelines from operator tree
  - Schedule pipeline execution (respecting dependencies)
  - Coordinate memory management

#### 5. GPUBufferManager
- **File**: `src/gpu_buffer_manager.cpp`
- **Purpose**: Centralized GPU memory management
- **Pattern**: Singleton
- **Functions**: Allocate/free GPU memory, caching

---

## Execution Flow

### Step-by-Step Query Execution

```sql
SELECT category, SUM(price)
FROM products
WHERE price > 100
GROUP BY category
```

#### Phase 1: Parse and Plan (DuckDB)
```
SQL String → Parser → Logical Plan
  LogicalAggregate [category, SUM(price)]
       ↓
  LogicalFilter [price > 100]
       ↓
  LogicalGet [products]
```

#### Phase 2: Physical Planning (Sirius)
**File**: `src/gpu_physical_plan_generator.cpp`
```
Logical Plan → GPU Physical Planner → Physical Operator Tree
  GPUPhysicalHashGroupBy [category]
       ↓
  GPUPhysicalFilter [price > 100]
       ↓
  GPUPhysicalTableScan [products]
```

#### Phase 3: Pipeline Construction
**File**: `src/gpu_executor.cpp`
```
Physical Plan → Build Pipelines → GPUMetaPipeline

Pipeline 1: SCAN → FILTER → HASH_AGGREGATE (sink)
Pipeline 2: HASH_AGGREGATE (source) → RESULT_COLLECTOR
```

#### Phase 4: Execution
```
For each pipeline in dependency order:
  1. Initialize operator states
  2. Source: GetData() → GPUIntermediateRelation
  3. Operators: Execute(input, output)
  4. Sink: Sink(input) → accumulate in state
  5. Finalize: CombineFinalize() → produce output
```

#### Phase 5: Result Collection
```
GPU Data → ResultCollector → CPU DataChunk → DuckDB → User
```

---

## Data Flow

### GPUIntermediateRelation Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│ Source Operator (e.g., TABLE_SCAN)                      │
│ GetData() → Creates GPUIntermediateRelation             │
└─────────────────────────────────────────────────────────┘
                       ↓
         GPUIntermediateRelation (batch of rows)
         • Vector of GPUColumn
         • Row count
         • Type information
                       ↓
┌─────────────────────────────────────────────────────────┐
│ Processing Operator (e.g., FILTER)                      │
│ Execute(input, output) → Transforms data                │
└─────────────────────────────────────────────────────────┘
                       ↓
         GPUIntermediateRelation (filtered rows)
                       ↓
┌─────────────────────────────────────────────────────────┐
│ Sink Operator (e.g., HASH_AGGREGATE)                    │
│ Sink(input) → Accumulates into internal state           │
└─────────────────────────────────────────────────────────┘
                       ↓
         Internal state (hash table, aggregate values)
                       ↓
┌─────────────────────────────────────────────────────────┐
│ Finalize                                                 │
│ CombineFinalize() → Produce final GPUIntermediateRelation│
└─────────────────────────────────────────────────────────┘
```

---

## Memory Management

### GPUBufferManager Architecture

```cpp
class GPUBufferManager {
private:
    static GPUBufferManager* instance;  // Singleton

public:
    // Allocate GPU memory
    template<typename T>
    T* customCudaMalloc(size_t count);

    // Free GPU memory
    template<typename T>
    void customCudaFree(T* ptr);
};
```

**Characteristics**:
- **Singleton Pattern**: Single global instance
- **Direct CUDA Calls**: Uses cudaMalloc/cudaFree
- **Simple Caching**: Minimal allocation caching
- **No Spilling**: Limited to available GPU memory

**Limitations**:
- Cannot handle datasets larger than GPU memory
- No automatic eviction or spilling
- Global state can cause contention

---

## Pipeline Model

### GPUPipeline Structure

**File**: `src/include/gpu_pipeline.hpp`

```cpp
class GPUPipeline {
public:
    // Source operator (produces data)
    unique_ptr<GPUPhysicalOperator> source;

    // Processing operators (transform data)
    vector<unique_ptr<GPUPhysicalOperator>> operators;

    // Sink operator (accumulates data)
    unique_ptr<GPUPhysicalOperator> sink;

    // Execute the entire pipeline
    void Execute(ExecutionContext& context);
};
```

### GPUMetaPipeline (DAG)

**File**: `src/include/gpu_meta_pipeline.hpp`

```cpp
class GPUMetaPipeline {
public:
    // All pipelines
    vector<unique_ptr<GPUPipeline>> pipelines;

    // Pipeline dependencies (DAG)
    vector<vector<idx_t>> dependencies;

    // Execute pipelines in topological order
    void Execute();
};
```

### Pipeline Breaks

Certain operators require **pipeline breaks** (materialize intermediate results):

1. **Hash Joins**: Build side must complete before probe
2. **Aggregates**: All input required before producing output
3. **Sorts**: Complete dataset needed for sorting

**Example**:
```
Pipeline 1: SCAN → FILTER → HASH_JOIN (build)
           [Pipeline Break - materialize hash table]
Pipeline 2: SCAN → FILTER → HASH_JOIN (probe) → RESULT
```

---

## Operator Categories

### Sources
- **TABLE_SCAN**: Read from DuckDB tables
- **COLUMN_DATA_SCAN**: Read from column data collection
- **DUMMY_SCAN**: Testing placeholder

### Transforms
- **FILTER**: Apply WHERE predicates
- **PROJECTION**: Select/compute columns
- **LIMIT**: Restrict row count

### Sinks
- **HASH_AGGREGATE**: Grouped aggregation (GROUP BY)
- **UNGROUPED_AGGREGATE**: Aggregation without grouping
- **HASH_JOIN**: Build side of hash join
- **RESULT_COLLECTOR**: Final result materialization

### Complex Operators
- **HASH_JOIN**: Two-phase (build + probe)
- **ORDER_BY**: Full sort
- **TOP_N**: Partial sort with limit

---

## Configuration

Legacy mode uses global configuration:

```cpp
// In DuckDB
SET gpu_memory_limit = 8192;  // MB
SET enable_gpu_caching = true;
```

**Key Settings**:
- GPU memory limit
- Enable/disable result caching
- Batch size for scanning
- Thread count for execution

---

## Comparison to New Mode

| Feature | Legacy Mode | New Mode |
|---------|-------------|----------|
| **Simplicity** | ✅ Simpler codebase | More complex |
| **Memory** | ❌ Single-tier only | ✅ Multi-tier spilling |
| **Scheduling** | ❌ Static pipelines | ✅ Dynamic tasks |
| **Parallelism** | ⚠️ Limited | ✅ Better concurrency |
| **Scalability** | ❌ GPU memory bound | ✅ Scales beyond GPU |
| **Performance** | Baseline | ✅ 1.2-1.6x faster |
| **Development** | ❌ Maintenance only | ✅ Active |

---

## Key Files Reference

| Component | File | Description |
|-----------|------|-------------|
| Entry Point | `src/sirius_extension.cpp:240-339` | gpu_processing() table function |
| Operator Base | `src/include/gpu_physical_operator.hpp` | GPUPhysicalOperator class |
| Executor | `src/gpu_executor.cpp` | Pipeline execution engine |
| Planner | `src/gpu_physical_plan_generator.cpp` | Logical to physical conversion |
| Memory | `src/gpu_buffer_manager.cpp` | GPUBufferManager singleton |
| Pipeline | `src/include/gpu_pipeline.hpp` | GPUPipeline structure |
| Operators | `src/operator/gpu_physical_*.cpp` | Individual operator implementations |

---

## Example: Complete Query Flow

```sql
SELECT * FROM gpu_processing('
    SELECT category, COUNT(*) as count
    FROM products
    WHERE price > 50
    GROUP BY category
');
```

**Execution Trace**:
```
1. GPUProcessingBind()
   ├─ Parse: "SELECT category, COUNT(*) FROM products WHERE..."
   ├─ Logical Plan: Get → Filter → Aggregate
   └─ Physical Plan: SCAN → FILTER → HASH_AGGREGATE

2. Build Pipelines
   ├─ Pipeline 1: SCAN → FILTER → HASH_AGGREGATE (sink)
   └─ Pipeline 2: HASH_AGGREGATE (source) → RESULT

3. Execute Pipeline 1
   ├─ SCAN.GetData() → GPUIntermediateRelation (batch 1)
   ├─ FILTER.Execute() → filtered rows
   ├─ HASH_AGGREGATE.Sink() → build hash table
   └─ Repeat for all batches

4. Finalize Pipeline 1
   └─ HASH_AGGREGATE.CombineFinalize() → aggregate results

5. Execute Pipeline 2
   ├─ HASH_AGGREGATE.GetData() → aggregated results
   └─ RESULT.Sink() → collect and transfer to CPU

6. Return Results
   └─ Convert to DuckDB DataChunk → return to user
```

---

## Next Steps

Now that you understand Legacy Mode at a high level:

1. **Entry Points**: [Legacy Mode Entry Points](entry-points.md) - Deep dive into `gpu_processing()`
2. **Operators**: [Legacy Mode Operators](operators.md) - All GPUPhysicalOperator implementations
3. **Pipelines**: [Pipeline Execution](pipeline-execution.md) - Detailed pipeline model
4. **Memory**: [Memory Management](memory-management.md) - GPUBufferManager internals
5. **Data Structures**: [Data Structures](data-structures.md) - GPUColumn and GPUIntermediateRelation

For modern development, see [New Mode Overview](../04-new-mode/overview.md).
