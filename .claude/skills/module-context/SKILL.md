---
name: module-context
description: Automatically identify which dependency library modules are relevant to a task and load their API documentation into context. Use PROACTIVELY before implementing features, fixing bugs, or writing new operators — analyzes the task description and loads cudf, rmm, duckdb, cucascade module docs to improve code quality. Trigger when the user asks to implement, add, fix, or modify GPU operators, pipeline components, memory management, joins, aggregations, sorting, expressions, or data I/O.
---

# Module Context Loader

You are a context-routing skill for the Sirius GPU SQL engine. Your job is to analyze a task description and load the relevant dependency module documentation so that the implementing agent has accurate API knowledge.

## Available Documentation

Module docs are pre-generated at `.claude/skills/module-discover/docs/`. Each library has:
- `README.md` — Module map with USED/UNUSED status and file-to-module mappings
- `modules/<name>.md` — Per-module API reference with signatures, descriptions, and usage examples

### Library Index

| Library | Namespace | Modules | Docs Path |
|---------|-----------|---------|-----------|
| **cudf** | `cudf::` | 25 modules (19 USED) | `docs/cudf/` |
| **rmm** | `rmm::`, `rmm::mr::` | 17 modules (8 USED) | `docs/rmm/` |
| **duckdb** | `duckdb::` | 13 modules (10 USED) | `docs/duckdb/` |
| **cucascade** | `cucascade::` | 3 modules (2 USED) | `docs/cucascade/` |
| **libkvikio** | `kvikio::` | 7 modules (0 USED) | `docs/libkvikio/` |

## Workflow

### Step 1: Analyze the Task

Read the user's task description and identify which **functional areas** it touches. Use this keyword-to-module mapping:

#### Join operations
**Keywords**: join, hash join, inner join, left join, right join, full join, semi join, anti join, nested loop, conditional join, equi-join, non-equi
**Modules to load**:
- `cudf/modules/join.md` — hash_join, conditional_join, mixed_join APIs
- `cudf/modules/ast.md` — AST expressions for conditional/mixed joins
- `cudf/modules/copying.md` — gather() to materialize join results
- `duckdb/modules/planner.md` — BoundExpression types for join conditions
- `duckdb/modules/execution.md` — PhysicalOperator for plan translation
- `cucascade/modules/data.md` — data_batch for pipeline I/O

#### Aggregation / Group By
**Keywords**: aggregate, group by, groupby, sum, count, min, max, avg, mean, distinct, reduce, having
**Modules to load**:
- `cudf/modules/aggregation.md` — groupby, reduce, aggregation factories
- `cudf/modules/stream_compaction.md` — drop_duplicates for DISTINCT
- `cudf/modules/dictionary.md` — dictionary encoding for merge optimization
- `duckdb/modules/function.md` — FunctionBinder for aggregate functions
- `cucascade/modules/data.md` — data_batch

#### Sorting / Order By / Top-N
**Keywords**: sort, order by, top-n, limit, merge sort, rank, partition
**Modules to load**:
- `cudf/modules/sorting.md` — sorted_order, merge, search bounds
- `cudf/modules/copying.md` — gather, slice, concatenate
- `cudf/modules/partitioning.md` — hash_partition
- `duckdb/modules/execution.md` — PhysicalOperator

#### Filter / Projection / Expressions
**Keywords**: filter, where, projection, expression, cast, comparison, like, regex, substring, between, case when, coalesce, in list
**Modules to load**:
- `cudf/modules/unary_binary.md` — binary_operation, cast, unary_operation
- `cudf/modules/scalar.md` — numeric_scalar, string_scalar for constants
- `cudf/modules/strings.md` — GPU string operations (like, contains, regex)
- `cudf/modules/datetime.md` — date/time extraction
- `duckdb/modules/planner.md` — BoundExpression hierarchy
- `duckdb/modules/common.md` — LogicalType, Value

#### Data I/O / Table Scan
**Keywords**: scan, parquet, read, datasource, I/O, file, hybrid scan, table scan
**Modules to load**:
- `cudf/modules/io.md` — parquet reader, datasource, hybrid_scan
- `cudf/modules/table.md` — table, table_view, column_view
- `cucascade/modules/data.md` — data_batch, data representations
- `cucascade/modules/memory.md` — memory spaces, host memory resources
- `duckdb/modules/execution.md` — scan task infrastructure

#### Memory Management
**Keywords**: memory, OOM, allocation, buffer, pool, reservation, spill, downgrade, evict, GPU memory, pinned memory
**Modules to load**:
- `rmm/modules/memory_resources.md` — device_memory_resource, pool_memory_resource
- `rmm/modules/resource_refs.md` — device_async_resource_ref
- `rmm/modules/device_containers.md` — device_buffer, device_uvector
- `rmm/modules/cuda_streams.md` — cuda_stream_view
- `rmm/modules/error_handling.md` — out_of_memory exception
- `cucascade/modules/memory.md` — reservation_manager, memory_space, tiered memory

#### Pipeline / Execution Engine
**Keywords**: pipeline, task, executor, stream, thread pool, scheduling, meta pipeline, sink, source
**Modules to load**:
- `cucascade/modules/data.md` — data_batch, data_repository
- `cucascade/modules/memory.md` — reservations, stream_pool
- `rmm/modules/cuda_streams.md` — cuda_stream_view
- `duckdb/modules/parallel.md` — ThreadContext, TaskScheduler
- `duckdb/modules/execution.md` — ExecutionContext

#### Type System / Data Types
**Keywords**: type, data type, decimal, varchar, string, date, timestamp, integer, logical type, type_id
**Modules to load**:
- `cudf/modules/types_core.md` — type_id, data_type, size_type
- `cudf/modules/fixed_point.md` — DECIMAL support
- `cudf/modules/table.md` — column_view, type accessors
- `duckdb/modules/common.md` — LogicalType, Value, PhysicalType

#### New Operator Implementation
**Keywords**: new operator, implement operator, add operator, physical operator
**Modules to load**:
- `duckdb/modules/execution.md` — PhysicalOperator base class
- `duckdb/modules/planner.md` — expression types for plan translation
- `cudf/modules/table.md` — table/column views
- `cudf/modules/copying.md` — gather, scatter, concatenate
- `rmm/modules/resource_refs.md` — device_async_resource_ref parameter pattern
- `rmm/modules/cuda_streams.md` — stream parameter pattern
- `cucascade/modules/data.md` — data_batch I/O pattern
- Load the specific cudf module for the operator's function (join, sort, aggregate, etc.)

#### Extension / Registration
**Keywords**: extension, register, table function, load, configuration, setting
**Modules to load**:
- `duckdb/modules/main.md` — ClientContext, Connection, DBConfig
- `duckdb/modules/function.md` — TableFunction, ScalarFunction
- `duckdb/modules/parser.md` — CreateTableFunctionInfo
- `duckdb/modules/catalog.md` — Catalog registration

### Step 2: Load Module Documentation

For each identified module:

1. Read the module's `.md` file from `.claude/skills/module-discover/docs/<library>/modules/<module>.md`
2. Extract the **API Reference** section (signatures + descriptions)
3. Extract the **Our Usage** examples (existing call sites in our codebase)

**Loading priority** (if context is limited):
1. APIs we already use (highest — patterns to follow)
2. APIs in included headers (medium — available and likely useful)
3. APIs available but unused (lowest — only if the task requires new functionality)

### Step 3: Present Context

Output a structured context block that the implementing agent can reference:

```markdown
## Relevant Library Context for: <task summary>

### Modules Loaded
- cudf/join — hash_join, conditional_join (for implementing the join operator)
- rmm/cuda_streams — cuda_stream_view (standard stream parameter)
- ...

### Key APIs

#### <API Name> (`<library>/<module>`)
<signature>
<brief description>
**Existing usage pattern**: `<file>:<line>` — <how we use it>

#### <Next API...>

### Patterns to Follow
- <Pattern observed from our existing code, e.g., "All operators take rmm::cuda_stream_view as parameter">
- <Pattern, e.g., "Join operators build hash table on smaller side, then gather results">
```

### Step 4: Flag Gaps

If the task requires functionality that:
- Exists in an UNUSED module → mention the module and suggest loading its docs
- Doesn't exist in any documented library → flag it explicitly
- Requires a version-specific API → note the version condition

## Guidelines

1. **Be selective.** Don't load every module. A typical task needs 3-6 modules. Loading too much dilutes the signal.
2. **Prioritize used modules.** Our existing usage patterns are the most valuable context — they show how APIs are actually integrated.
3. **Include the file-to-module mapping** from the README when relevant, so the implementer knows which existing files to look at.
4. **Cross-reference libraries.** Most tasks span multiple libraries (e.g., a join needs cudf/join + duckdb/planner + rmm/streams + cucascade/data).
5. **Surface version gotchas.** cudf has significant API differences between 25.04 and 26.04+. Always note when a loaded API has version-conditional behavior.
6. **Read the actual module docs.** Don't summarize from memory — read the `.md` files to get accurate signatures.
