# Glossary

Alphabetical reference of terms and concepts used throughout Sirius documentation.

---

## A

### Aggregate
An operator that computes summary statistics (SUM, COUNT, AVG, etc.) over groups of rows or entire datasets.

**Related**: HASH_GROUP_BY, UNGROUPED_AGGREGATE

### Asynchronous Execution
Execution model where operations run concurrently without blocking, using CUDA streams and thread pools.

**Context**: New Mode uses asynchronous execution extensively.

---

## B

### Batch
A columnar collection of rows processed together. Batch size affects GPU efficiency and memory usage.

**Related**: data_batch, GPUIntermediateRelation

### Bind Phase
First phase of table function execution where the query is analyzed, planned, and prepared.

**Related**: GPUProcessingBind, GPUExecutionBind

### Build Side
In a hash join, the side that constructs the hash table. Must complete before probe side starts.

**Related**: HASH_JOIN, Pipeline Break

---

## C

### Columnar Storage
Data layout where each column is stored separately, enabling efficient vectorized operations.

**Contrast**: Row-oriented storage

### CUDA Stream
Sequence of GPU operations that execute in order. Multiple streams enable concurrent GPU execution.

**Related**: pipeline_executor, Task Executor

### cuDF
GPU DataFrame library (part of RAPIDS) providing columnar data structures and operations.

**Related**: data_batch, Expression Evaluation

### Cucascade
Data repository and memory management framework used in New Mode for inter-pipeline communication and multi-tier storage.

**Files**: `cucascade/` directory

---

## D

### Data Batch
Columnar data unit in New Mode, backed by cuDF tables.

**Type**: `cucascade::data_batch`

**Related**: GPUIntermediateRelation (Legacy equivalent)

### Data Repository
Storage abstraction for inter-pipeline communication supporting multi-tier memory (GPU/HOST/DISK).

**Type**: `cucascade::data_repository`

**Related**: Port, Multi-Tier Memory

### Downgrade
Process of moving data from faster to slower memory tier (GPU → HOST → DISK) when memory pressure occurs.

**Related**: downgrade_executor, Memory Tier

### DuckDB
In-process analytical database that Sirius extends for GPU acceleration.

**Integration**: Sirius is a DuckDB extension

---

## E

### Execute Interface
Operator method that processes input data and produces output.

**Legacy**: `OperatorResultType Execute(GPUIntermediateRelation& in, GPUIntermediateRelation& out)`

**New**: `vector<shared_ptr<data_batch>> execute(vector<shared_ptr<data_batch>>& in, cuda_stream_view stream)`

### Expression
Computation (arithmetic, comparison, etc.) applied to column data.

**Examples**: `price * quantity`, `date >= '2024-01-01'`

**Related**: Expression Executor

---

## F

### Fallback
Mechanism to execute query on CPU (via DuckDB) when GPU execution fails.

**Config**: `ENABLE_FALLBACK`

### Filter
Operator that applies WHERE clause predicates to restrict rows.

**Type**: GPUPhysicalFilter (Legacy), sirius_physical_filter (New)

---

## G

### Global State
Operator state shared across all threads/tasks during execution.

**Types**: GlobalSourceState, GlobalOperatorState, GlobalSinkState

### GPUBufferManager
Singleton memory manager in Legacy Mode for GPU allocations.

**File**: `src/gpu_buffer_manager.cpp`

### GPUIntermediateRelation
Data structure in Legacy Mode representing a batch of rows.

**Components**: Vector of GPUColumn, row count, types

**New Mode Equivalent**: data_batch

### gpu_execution()
Table function for New Mode GPU execution.

**Entry**: `src/sirius_extension.cpp:353-452`

### gpu_processing()
Table function for Legacy Mode GPU execution.

**Entry**: `src/sirius_extension.cpp:240-339`

---

## H

### Hash Join
Join algorithm using hash table for efficient lookups.

**Phases**: Build (create hash table), Probe (lookup matches)

**Pipeline Break**: Required between build and probe

### Hash Table
Data structure for O(1) lookups used in joins and aggregates.

**Usage**: HASH_JOIN, HASH_GROUP_BY

### Hint
Signal from operator about task readiness.

**Values**: `READY`, `WAITING_FOR_INPUT_DATA`

**Related**: Task Creation, Dynamic Scheduling

---

## I

### In-Process
Database that runs within the application process, not as a separate server.

**Example**: DuckDB, SQLite

**Contrast**: Client-server databases (PostgreSQL, MySQL)

### Input Port
Connection point where pipeline receives data from upstream pipeline.

**Related**: Output Port, Data Repository

---

## J

### Join
Operator combining rows from two tables based on condition.

**Types**: HASH_JOIN, NESTED_LOOP_JOIN, MERGE_JOIN

---

## K

### Kernel
GPU function that executes in parallel across many threads.

**Context**: cuDF operators compile to CUDA kernels

---

## L

### Legacy Mode
Original Sirius execution engine using `gpu_processing()` table function.

**Status**: Maintenance mode

**Contrast**: New Mode

### Logical Plan
Database-agnostic representation of query semantics (what to compute, not how).

**Source**: DuckDB planner

**Next**: Physical Plan

---

## M

### Materialization
Storing intermediate results in memory rather than streaming through operators.

**Causes**: Pipeline breaks (joins, aggregates, sorts)

### Memory Reservation
Pre-allocating memory before use to prevent out-of-memory errors.

**Manager**: sirius_memory_reservation_manager

### Memory Tier
Level in memory hierarchy: GPU (fastest), HOST (medium), DISK (slowest).

**Related**: Downgrade, Upgrade, Multi-Tier Storage

### Meta Pipeline
Collection of pipelines with dependency relationships (DAG).

**Legacy**: GPUMetaPipeline

**New**: sirius_meta_pipeline

### Multi-Tier Storage
Memory management across GPU/HOST/DISK tiers with automatic spilling.

**Implementation**: Cucascade repositories

**Benefit**: Handle datasets larger than GPU memory

---

## N

### Nested Loop Join
Join algorithm with O(N*M) complexity, used for non-equi joins.

**Usage**: When hash join not applicable

### New Mode
Modern Sirius execution engine using `gpu_execution()` table function.

**Status**: Active development

**Contrast**: Legacy Mode

---

## O

### Operator
Execution unit implementing relational operation (scan, filter, join, etc.).

**Legacy**: GPUPhysicalOperator

**New**: sirius_physical_operator

### Output Port
Connection point where pipeline sends data to downstream pipeline.

**Related**: Input Port, Data Repository

---

## P

### Physical Plan
Executable representation of query with specific algorithms and operators.

**Generator**: sirius_physical_plan_generator

### Pipeline
Chain of operators executing together without materializing intermediate results.

**Legacy**: GPUPipeline

**New**: sirius_pipeline

### Pipeline Break
Point where intermediate results must be materialized before continuing.

**Causes**: Hash joins (build/probe), aggregates, sorts

### Pinned Memory
Host memory locked in RAM, enabling efficient GPU transfers.

**Related**: RMM, DMA

### Port
Connection between pipelines using data repositories.

**Types**: Input Port (receive), Output Port (send)

### Probe Side
In hash join, the side that looks up matches in the hash table.

**Related**: Build Side, HASH_JOIN

---

## Q

### Query Result
DuckDB structure containing query output.

**Type**: MaterializedQueryResult or StreamQueryResult

---

## R

### RAPIDS
NVIDIA library ecosystem for GPU-accelerated data science.

**Components**: cuDF (DataFrames), cuML (ML), cuGraph (graphs), RMM (memory)

### Result Collector
Operator that materializes final results and transfers to CPU.

**Type**: RESULT_COLLECTOR

### RMM (RAPIDS Memory Manager)
Memory allocation library providing pooling and multi-tier resources.

**Integration**: Used for all GPU allocations in New Mode

---

## S

### Scan
Operator that reads data from source (table, file, etc.).

**Types**: TABLE_SCAN, DUCKDB_SCAN, DUMMY_SCAN

### Sink
Operator that accumulates input data without immediately producing output.

**Examples**: Hash join build, aggregate, result collector

### Source
Operator that produces data without consuming input.

**Examples**: Table scan, column data scan

### Spilling
Moving data from faster to slower storage when memory full.

**Related**: Downgrade, Multi-Tier Storage

### Stream
See CUDA Stream

---

## T

### Table Function
DuckDB mechanism allowing custom query processing via function calls.

**Sirius Functions**: `gpu_processing()`, `gpu_execution()`

### Task
Unit of work executed by thread pool.

**Interface**: itask, sirius_pipeline_itask

### Task Creation Hint
Signal indicating whether task can execute or must wait for input.

**Values**: `READY`, `WAITING_FOR_INPUT_DATA`

### Task Executor
Thread pool that processes tasks.

**Types**: pipeline_executor, task_creator, downgrade_executor, duckdb_scan_executor

### Thread Pool
Set of worker threads that execute tasks concurrently.

**Related**: Task Executor

### TPC-H
Standard analytical database benchmark.

**Usage**: Sirius performance testing

---

## U

### Ungrouped Aggregate
Aggregate operation without GROUP BY clause.

**Example**: `SELECT COUNT(*), SUM(price) FROM orders`

**Type**: UNGROUPED_AGGREGATE

### Upgrade
Moving data from slower to faster memory tier (DISK → HOST → GPU).

**Related**: Downgrade, Memory Tier

---

## V

### Vectorized Execution
Processing entire columns at once using SIMD or GPU parallelism.

**Benefit**: Better cache utilization, higher throughput

**Context**: cuDF provides vectorized operations

---

## W

### Window Function
Computation over sliding window of rows.

**Examples**: ROW_NUMBER(), RANK(), LAG()

**Support**: Partial in New Mode

---

## X-Z

### YAML
Configuration file format used for Sirius settings.

**File**: sirius.cfg

---

## Acronyms

| Acronym | Full Name | Description |
|---------|-----------|-------------|
| **API** | Application Programming Interface | Programming interface |
| **AST** | Abstract Syntax Tree | Parsed query structure |
| **CUDA** | Compute Unified Device Architecture | NVIDIA GPU programming model |
| **DAG** | Directed Acyclic Graph | Pipeline dependency graph |
| **DMA** | Direct Memory Access | Hardware-accelerated memory transfer |
| **GDDR** | Graphics DDR | GPU memory type |
| **GPU** | Graphics Processing Unit | Parallel processor |
| **HBM** | High Bandwidth Memory | GPU memory type (A100, H100) |
| **NUMA** | Non-Uniform Memory Access | Multi-socket memory architecture |
| **OLAP** | Online Analytical Processing | Analytical workloads |
| **OLTP** | Online Transaction Processing | Transactional workloads |
| **OOM** | Out Of Memory | Memory exhaustion error |
| **PCIe** | Peripheral Component Interconnect Express | CPU-GPU bus |
| **RAII** | Resource Acquisition Is Initialization | C++ memory management pattern |
| **RAPIDS** | RAPIDS AI Processing Framework | NVIDIA data science libraries |
| **RMM** | RAPIDS Memory Manager | GPU memory management library |
| **SIMD** | Single Instruction Multiple Data | CPU vectorization |
| **SQL** | Structured Query Language | Database query language |

---

## See Also

- [API Reference](api-reference.md) - Class and method documentation
- [File Index](file-index.md) - Important files by category
- [Config Options](config-options.md) - Configuration parameters
