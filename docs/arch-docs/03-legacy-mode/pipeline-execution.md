# Legacy Mode Pipeline Execution

This document explains how Sirius Legacy Mode organizes operators into pipelines and executes them to process queries.

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Structure](#pipeline-structure)
3. [GPUMetaPipeline](#gpumetapipeline)
4. [Pipeline Building](#pipeline-building)
5. [Pipeline Execution](#pipeline-execution)
6. [Batch Processing](#batch-processing)
7. [Pipeline Dependencies](#pipeline-dependencies)
8. [Examples](#examples)
9. [Next Steps](#next-steps)

---

## Overview

In Legacy Mode, query execution is organized into **pipelines** — chains of operators that process data together. A pipeline consists of:

- **Source operator**: Generates initial data (e.g., TABLE_SCAN)
- **Intermediate operators**: Transform data in transit (e.g., FILTER, PROJECTION)
- **Sink operator**: Accumulates data across batches (e.g., HASH_JOIN build, RESULT_COLLECTOR)

Pipelines are constructed from the physical operator tree during initialization, then executed sequentially by the **GPUExecutor**.

**Key Characteristics:**

- **Pull-based execution**: Data flows from source → operators → sink
- **Single-threaded per pipeline**: Each pipeline executes on one thread
- **Pipeline-breaking operators**: Sinks (joins, aggregates) split pipelines
- **Dependency management**: GPUMetaPipeline orchestrates pipeline ordering

---

## Pipeline Structure

### GPUPipeline Class

**File**: `src/include/gpu_pipeline.hpp:64-158`

```cpp
class GPUPipeline : public enable_shared_from_this<GPUPipeline> {
public:
    GPUExecutor& executor;                                  // Execution context

    // Pipeline components
    optional_ptr<GPUPhysicalOperator> source;               // Data source
    vector<reference<GPUPhysicalOperator>> operators;       // Intermediate operators
    optional_ptr<GPUPhysicalOperator> sink;                 // Data sink

    // State
    unique_ptr<GlobalSourceState> source_state;             // Source state (e.g., scan position)
    bool ready;                                             // Whether pipeline is ready to execute
    atomic<bool> initialized;                               // Whether pipeline has been initialized

    // Dependencies
    vector<shared_ptr<GPUPipeline>> dependencies;           // Pipelines that must complete first
    vector<weak_ptr<GPUPipeline>> parents;                  // Pipelines that depend on this one

    // Methods
    void AddDependency(shared_ptr<GPUPipeline>& pipeline);  // Add dependency
    void Ready();                                           // Mark pipeline as ready
    void Schedule(shared_ptr<Event>& event);                // Schedule for execution
    vector<reference<GPUPhysicalOperator>> GetAllOperators(); // Get all operators (source + operators + sink)
};
```

### Pipeline Components

| Component | Type | Purpose | Example |
|-----------|------|---------|---------|
| **source** | `optional_ptr<GPUPhysicalOperator>` | Generates data | TABLE_SCAN, COLUMN_DATA_SCAN |
| **operators** | `vector<reference<GPUPhysicalOperator>>` | Transform data | FILTER, PROJECTION, ORDER_BY |
| **sink** | `optional_ptr<GPUPhysicalOperator>` | Accumulates data | HASH_JOIN, RESULT_COLLECTOR |
| **dependencies** | `vector<shared_ptr<GPUPipeline>>` | Pipelines that must finish first | Build side of join |
| **source_state** | `unique_ptr<GlobalSourceState>` | Source-specific state | Scan position, batch counter |

### Example Pipeline

**Query:**

```sql
SELECT name, age FROM users WHERE age > 25;
```

**Pipeline:**

```
[Source]     TABLE_SCAN (users)
    ↓
[Operators]  FILTER (age > 25)
    ↓
[Sink]       RESULT_COLLECTOR
```

**In Memory:**

```cpp
GPUPipeline {
    source: GPUPhysicalTableScan("users"),
    operators: [
        GPUPhysicalFilter("age > 25")
    ],
    sink: GPUPhysicalResultCollector(),
    dependencies: []  // No dependencies
}
```

---

## GPUMetaPipeline

A **GPUMetaPipeline** groups multiple pipelines that share the same sink operator. It manages:

- **Pipeline construction**: Creates child pipelines for complex operators (joins, unions)
- **Dependency tracking**: Ensures pipelines execute in correct order
- **Batch indexing**: Assigns unique batch IDs to prevent conflicts

**File**: `src/include/gpu_meta_pipeline.hpp:27-120`

```cpp
class GPUMetaPipeline : public enable_shared_from_this<GPUMetaPipeline> {
public:
    GPUMetaPipeline(GPUExecutor& gpu_executor,
                    GPUPipelineBuildState& state,
                    optional_ptr<GPUPhysicalOperator> sink);

    // Build the meta pipeline
    void Build(GPUPhysicalOperator& op);
    void Ready();

    // Create pipelines
    GPUPipeline& CreatePipeline();
    GPUPipeline& CreateUnionPipeline(GPUPipeline& current, bool order_matters);
    void CreateChildPipeline(GPUPipeline& current,
                            GPUPhysicalOperator& op,
                            GPUPipeline& last_pipeline);
    GPUMetaPipeline& CreateChildMetaPipeline(GPUPipeline& current, GPUPhysicalOperator& op);

private:
    GPUExecutor& executor;                                  // Execution context
    GPUPipelineBuildState& state;                          // Build state
    optional_ptr<GPUPhysicalOperator> sink;                // Shared sink
    vector<shared_ptr<GPUPipeline>> pipelines;             // All pipelines with this sink
    vector<shared_ptr<GPUMetaPipeline>> children;          // Child meta pipelines
    reference_map_t<GPUPipeline, vector<reference<GPUPipeline>>> dependencies; // Intra-meta dependencies
};
```

### Meta Pipeline Roles

1. **Pipeline Creation**:
   - Creates base pipeline for the main execution path
   - Creates union pipelines for UNION ALL
   - Creates child pipelines for complex operators

2. **Dependency Management**:
   - Tracks which pipelines must complete before others
   - Prevents race conditions on shared state
   - Ensures correct execution order

3. **Batch Indexing**:
   - Assigns unique batch indices to prevent conflicts
   - Increments by `BATCH_INCREMENT = 10000000000000` per pipeline
   - Allows tracking batch progress across pipelines

---

## Pipeline Building

Pipeline construction happens during `GPUExecutor::InitializeInternal()` via recursive `BuildPipelines()` calls on operators.

### Building Process

**File**: `src/gpu_executor.cpp:73-79`

```cpp
void GPUExecutor::Initialize(unique_ptr<GPUPhysicalOperator> plan) {
    SIRIUS_LOG_DEBUG("Initializing GPUExecutor");
    Reset();
    gpu_owned_plan = std::move(plan);
    InitializeInternal(*gpu_owned_plan);
}

void GPUExecutor::InitializeInternal(GPUPhysicalOperator& plan) {
    // Step 1: Create root meta pipeline
    auto& root_pipeline = CreatePipeline();
    auto root_meta_pipeline = make_shared<GPUMetaPipeline>(*this, state, nullptr);

    // Step 2: Build pipeline graph recursively
    root_meta_pipeline->Build(plan);

    // Step 3: Ready all pipelines
    root_meta_pipeline->Ready();

    // Step 4: Schedule pipelines
    SchedulePipelines();
}
```

### Operator BuildPipelines() Methods

Each operator implements `BuildPipelines()` to control pipeline construction:

**Intermediate Operator (Pass-Through):**

```cpp
void GPUPhysicalFilter::BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline) {
    // Add this operator to the current pipeline
    meta_pipeline.GetState().AddPipelineOperator(current, *this);

    // Continue building with children
    children[0]->BuildPipelines(current, meta_pipeline);
}
```

**Sink Operator (Pipeline Break):**

```cpp
void GPUPhysicalHashJoin::BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline) {
    // Step 1: This operator becomes the sink of the probe pipeline
    meta_pipeline.GetState().SetPipelineSink(current, this, 1);

    // Step 2: Build probe side (continues current pipeline)
    children[0]->BuildPipelines(current, meta_pipeline);

    // Step 3: Create child meta pipeline for build side
    auto& child_meta = meta_pipeline.CreateChildMetaPipeline(current, *this);

    // Step 4: Build build side (new pipeline)
    child_meta.Build(*children[1]);

    // Step 5: Add dependency: probe depends on build
    current.AddDependency(child_meta.GetBasePipeline());
}
```

### Pipeline Breaking Rules

Operators that **break pipelines** (create new pipeline with this as sink):

| Operator | Reason | Child Pipelines |
|----------|--------|-----------------|
| **HASH_JOIN** | Must build hash table before probing | Build side gets separate pipeline |
| **HASH_GROUP_BY** | Must accumulate all groups before output | Input gets separate pipeline |
| **UNGROUPED_AGGREGATE** | Must see all data before computing aggregate | Input gets separate pipeline |
| **RESULT_COLLECTOR** | Final sink, collects all results | Input continues current pipeline |
| **CTE** | Materializes result for reuse | Input gets separate pipeline |

Operators that **continue pipelines** (added to current pipeline):

- **FILTER**, **PROJECTION**: Transform data in-place
- **ORDER_BY**, **LIMIT**, **TOP_N**: Can process batches incrementally
- **NESTED_LOOP_JOIN**: Doesn't require global state

---

## Pipeline Execution

Once pipelines are built and ready, the **GPUExecutor** executes them sequentially.

### Execution Loop

**File**: `src/gpu_executor.cpp:81-200`

```cpp
void GPUExecutor::Execute() {
    SIRIUS_LOG_DEBUG("Total meta pipelines {}", scheduled.size());

    // Execute each pipeline in order
    for (const auto& pipeline : scheduled) {
        // Step 1: Allocate intermediate relations
        vector<shared_ptr<GPUIntermediateRelation>> intermediate_relations;
        intermediate_relations.reserve(pipeline->operators.size());

        for (idx_t i = 0; i < pipeline->operators.size(); i++) {
            auto& prev_operator = i == 0 ? *(pipeline->source) : pipeline->operators[i - 1].get();
            auto inter_rel = make_shared_ptr<GPUIntermediateRelation>(prev_operator.GetTypes().size());
            intermediate_relations.push_back(std::move(inter_rel));
        }

        // Allocate final relation
        auto& last_op = pipeline->operators.empty() ? *pipeline->source : pipeline->operators.back().get();
        auto final_relation = make_shared_ptr<GPUIntermediateRelation>(last_op.GetTypes().size());

        // Step 2: Get data from source
        auto& source_relation = pipeline->operators.empty() ? final_relation : intermediate_relations[0];
        pipeline->source->GetData(*source_relation);

        SIRIUS_LOG_DEBUG("Source: {} (type: {})",
                        pipeline->source->GetName(),
                        PhysicalOperatorToString(pipeline->source->type));

        // Step 3: Execute intermediate operators
        for (int current_idx = 1; current_idx <= pipeline->operators.size(); current_idx++) {
            auto& current_operator = pipeline->operators[current_idx - 1];
            auto& prev_relation = (current_idx == 1) ? source_relation : intermediate_relations[current_idx - 2];
            auto& current_relation = (current_idx == pipeline->operators.size())
                                       ? final_relation
                                       : intermediate_relations[current_idx];

            SIRIUS_LOG_DEBUG("Operator: {} (type: {})",
                            current_operator.get().GetName(),
                            PhysicalOperatorToString(current_operator.get().type));

            // Execute operator
            current_operator.get().Execute(*prev_relation, *current_relation);
        }

        // Step 4: Sink final data
        if (pipeline->sink) {
            SIRIUS_LOG_DEBUG("Sink: {} (type: {})",
                            pipeline->sink->GetName(),
                            PhysicalOperatorToString(pipeline->sink->type));

            pipeline->sink->Sink(*final_relation);
        }
    }

    // Step 5: Finalize sinks (e.g., hash table construction)
    for (const auto& pipeline : scheduled) {
        if (pipeline->sink && pipeline->sink->sink_state) {
            // Call CombineFinalize to merge accumulated data
            vector<shared_ptr<GPUIntermediateRelation>> accumulated;
            shared_ptr<GPUIntermediateRelation> output;
            pipeline->sink->CombineFinalize(accumulated, *output);
        }
    }
}
```

### Execution Phases

**Phase 1: Initialization**

- Allocate `GPUIntermediateRelation` objects for each operator output
- Initialize source state (e.g., scan position)

**Phase 2: Source Data**

- Call `source->GetData(output_relation)`
- Source reads data from DuckDB and transfers to GPU
- Populates `GPUIntermediateRelation` with GPU columns

**Phase 3: Operator Chain**

- For each intermediate operator in sequence:
  - Call `operator->Execute(input_relation, output_relation)`
  - Operator transforms input GPU data → output GPU data
  - Output becomes input for next operator

**Phase 4: Sink Data**

- Call `sink->Sink(final_relation)`
- Sink accumulates data into global state
- Returns `NEED_MORE_INPUT` to continue or `FINISHED`

**Phase 5: Finalize**

- After all data is sinked:
  - Call `sink->CombineFinalize(inputs, output)`
  - Merge accumulated data (e.g., build hash table)
  - Prepare for dependent pipelines

---

## Batch Processing

Legacy Mode processes data in **batches** rather than row-by-row. Each batch is represented as a `GPUIntermediateRelation`.

### Batch Size

Batch size is determined by:

1. **Source operator**: TABLE_SCAN reads DuckDB chunks (typically 1024-2048 rows)
2. **GPU memory**: Limited by available GPU RAM
3. **Cardinality estimates**: Planner estimates row counts

**Typical batch sizes:**

- **Small queries**: 1K - 10K rows per batch
- **Large scans**: 100K - 1M rows per batch (limited by GPU memory)
- **Post-filter**: Varies based on selectivity

### Batch Flow Example

**Query:**

```sql
SELECT name FROM users WHERE age > 25;
```

**Pipeline:**

```
TABLE_SCAN → FILTER → RESULT_COLLECTOR
```

**Batch Execution:**

```
Batch 1:
  TABLE_SCAN.GetData() → 1000 rows
  FILTER.Execute()     → 300 rows (30% selectivity)
  RESULT_COLLECTOR.Sink() → accumulate 300 rows

Batch 2:
  TABLE_SCAN.GetData() → 1000 rows
  FILTER.Execute()     → 350 rows (35% selectivity)
  RESULT_COLLECTOR.Sink() → accumulate 350 rows

Batch 3:
  TABLE_SCAN.GetData() → SOURCE_FINISHED
  RESULT_COLLECTOR.FinalMaterialize() → transfer 650 rows to CPU
```

---

## Pipeline Dependencies

Pipelines execute in topological order based on dependencies. Dependencies are added when:

1. **Join build side**: Probe pipeline depends on build pipeline
2. **CTE materialization**: CTE scan depends on CTE build pipeline
3. **Correlated subqueries**: Outer query depends on subquery pipeline

### Dependency Graph Example

**Query:**

```sql
SELECT o.order_id, c.name
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.total > 100;
```

**Pipeline Graph:**

```
Pipeline 1 (Build):
  TABLE_SCAN (customers)
      ↓
  HASH_JOIN (sink, build side)
  [builds hash table]

Pipeline 2 (Probe):
  TABLE_SCAN (orders)
      ↓
  FILTER (total > 100)
      ↓
  HASH_JOIN (execute, probe side)
  [depends on Pipeline 1]
      ↓
  RESULT_COLLECTOR (sink)
```

**Dependency:**

```cpp
pipeline2.dependencies.push_back(pipeline1);
```

**Execution Order:**

1. Execute Pipeline 1 completely → build hash table
2. Wait for Pipeline 1 to finish
3. Execute Pipeline 2 → probe hash table batch-by-batch

### Scheduling

The **GPUExecutor** schedules pipelines using a simple algorithm:

```cpp
void GPUExecutor::SchedulePipelines() {
    // Collect all pipelines
    vector<shared_ptr<GPUPipeline>> all_pipelines;
    root_meta_pipeline->GetPipelines(all_pipelines, true);

    // Schedule in dependency order
    for (auto& pipeline : all_pipelines) {
        if (pipeline->dependencies.empty()) {
            // No dependencies, can execute immediately
            scheduled.push_back(pipeline);
        } else {
            // Wait for dependencies to complete
            // Add to scheduled after dependencies finish
            ScheduleAfterDependencies(pipeline);
        }
    }
}
```

---

## Examples

### Example 1: Simple Filter Query

**Query:**

```sql
SELECT * FROM users WHERE age > 25;
```

**Physical Plan:**

```
RESULT_COLLECTOR
    ↓
FILTER (age > 25)
    ↓
TABLE_SCAN (users)
```

**Pipeline:**

```
GPUPipeline {
    source: TABLE_SCAN (users),
    operators: [FILTER (age > 25)],
    sink: RESULT_COLLECTOR,
    dependencies: []
}
```

**Execution:**

1. `TABLE_SCAN.GetData()` → Read batch from DuckDB, transfer to GPU
2. `FILTER.Execute()` → Evaluate `age > 25` on GPU, compact rows
3. `RESULT_COLLECTOR.Sink()` → Accumulate filtered rows
4. Repeat until `TABLE_SCAN` returns `SOURCE_FINISHED`
5. `RESULT_COLLECTOR.FinalMaterialize()` → Transfer to CPU, convert to DuckDB format

### Example 2: Join Query

**Query:**

```sql
SELECT o.order_id, c.name
FROM orders o
JOIN customers c ON o.customer_id = c.id;
```

**Physical Plan:**

```
RESULT_COLLECTOR
    ↓
HASH_JOIN (probe: orders, build: customers)
    ├─ TABLE_SCAN (orders)  [probe side]
    └─ TABLE_SCAN (customers)  [build side]
```

**Pipelines:**

```
Pipeline 1 (Build):
GPUPipeline {
    source: TABLE_SCAN (customers),
    operators: [],
    sink: HASH_JOIN (build side),
    dependencies: []
}

Pipeline 2 (Probe):
GPUPipeline {
    source: TABLE_SCAN (orders),
    operators: [HASH_JOIN (probe side)],
    sink: RESULT_COLLECTOR,
    dependencies: [Pipeline 1]
}
```

**Execution:**

**Phase 1: Build Hash Table**

1. `TABLE_SCAN(customers).GetData()` → Read all customer batches
2. `HASH_JOIN.Sink()` → Accumulate customer batches
3. `HASH_JOIN.CombineFinalize()` → Build hash table on GPU

**Phase 2: Probe Hash Table**

4. `TABLE_SCAN(orders).GetData()` → Read order batch
5. `HASH_JOIN.Execute()` → Probe hash table, gather matches
6. `RESULT_COLLECTOR.Sink()` → Accumulate joined results
7. Repeat steps 4-6 for all order batches
8. `RESULT_COLLECTOR.FinalMaterialize()` → Transfer to CPU

### Example 3: Aggregate Query

**Query:**

```sql
SELECT department, AVG(salary) FROM employees GROUP BY department;
```

**Physical Plan:**

```
RESULT_COLLECTOR
    ↓
HASH_GROUP_BY (group: department, agg: AVG(salary))
    ↓
TABLE_SCAN (employees)
```

**Pipelines:**

```
Pipeline 1 (Scan + Sink):
GPUPipeline {
    source: TABLE_SCAN (employees),
    operators: [],
    sink: HASH_GROUP_BY (sink phase),
    dependencies: []
}

Pipeline 2 (Source):
GPUPipeline {
    source: HASH_GROUP_BY (source phase),
    operators: [],
    sink: RESULT_COLLECTOR,
    dependencies: [Pipeline 1]
}
```

**Execution:**

**Phase 1: Accumulate Data**

1. `TABLE_SCAN.GetData()` → Read employee batches
2. `HASH_GROUP_BY.Sink()` → Accumulate all batches
3. `HASH_GROUP_BY.CombineFinalize()` → Perform groupby aggregation on GPU

**Phase 2: Output Results**

4. `HASH_GROUP_BY.GetData()` → Emit aggregated groups
5. `RESULT_COLLECTOR.Sink()` → Collect results
6. `RESULT_COLLECTOR.FinalMaterialize()` → Transfer to CPU

---

## Next Steps

**Related Documentation:**

- **[Operators](operators.md)**: Detailed operator implementations
- **[Memory Management](memory-management.md)**: GPUBufferManager and memory allocation
- **[Data Structures](data-structures.md)**: GPUIntermediateRelation and GPUColumn internals
- **[Architecture Diagram](architecture-diagram.md)**: Visual pipeline flow

**Comparison:**

- **[New Mode Pipeline Execution](../04-new-mode/pipeline-execution.md)**: Compare with task-based execution
- **[Execution Modes](../02-architecture/execution-modes.md)**: Understand trade-offs

**For Developers:**

- **[Debugging](../07-development/debugging.md)**: Debugging pipeline execution
- **[Testing Guide](../07-development/testing-guide.md)**: Writing pipeline tests
