# Super Sirius Documentation

Super Sirius is the new task-based GPU execution engine in Sirius. It uses `namespace sirius` and replaces the legacy `gpu_processing` path with a pipelined, multi-threaded architecture that partitions work across GPU and CPU thread pools.

With a Sirius config file (`~/.sirius/sirius.yaml`), GPU execution is **transparent** — users write plain SQL and supported queries automatically execute on the GPU. Unsupported queries silently fall back to CPU. The explicit `CALL gpu_execution('...')` function is still available but no longer required. Legacy `sirius.cfg` is still recognized for compatibility.

```sql
-- Just load the extension. If ~/.sirius/sirius.yaml exists, GPU is automatic.
LOAD 'sirius.duckdb_extension';

-- Plain SQL — transparently executed on GPU:
SELECT l_returnflag, SUM(l_quantity) FROM lineitem GROUP BY l_returnflag;
```

## How It Differs from Legacy Sirius

| Aspect | Legacy (`gpu_processing`) | Super Sirius |
|--------|---------------------------|-------------------------------|
| Namespace | `duckdb` | `sirius` |
| Entry point | `CALL gpu_processing(...)` | Plain SQL (transparent) or `CALL gpu_execution(...)` |
| Plan generator | `GPUPhysicalPlanGenerator` | `sirius_physical_plan_generator` |
| Operators | `GPUPhysicalOperator` in `src/operator/` | `sirius_physical_operator` in `src/op/` |
| Execution model | Single-threaded GPU executor | Multi-pipeline task-based execution |
| Memory management | `GPUBufferManager` | cuCascade tiered memory (GPU/Host/Disk) |

## Table of Contents

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture-overview.md) | Component diagram, thread model, ownership hierarchy, execution lifecycle |
| [Execution Flow](execution-flow.md) | End-to-end query trace with file:line references |
| [Physical Plan Generation](physical-plan-generation.md) | Logical-to-physical mapping, pipeline construction, splitting rules |
| [Operators](operators.md) | All physical operators: interface, GPU implementation, cuDF APIs |
| [Expression Executor](expression-executor.md) | gpu_expression_executor, GPU expression translator, cuDF AST |
| [Pipeline Execution](pipeline-execution.md) | GPU executor, task scheduling, completion, OOM handling, per-task-device contract under SCHED-RR |
| [Task Creator](task-creator.md) | Task creation: hint chain, per-operator scheduling behavior |
| [Scan](scan.md) | Scan subsystem: parquet scan, DuckDB scan, caching, prefetched data source |
| [Memory Management](memory-management.md) | cuCascade tiers, reservations, downgrade executor |
| [Data Management](data-management.md) | Data batches, repositories, ports, barrier semantics |
| [Configuration](configuration.md) | sirius_config, operator_params, SET variables |
| [Optimizations](optimizations.md) | Performance optimizations with PRs, code paths, configs |
| [Multi-GPU Architecture](multi-gpu-architecture.md) | How Sirius executes SQL across every GPU on a node — tiers, pin tables, SCHED-RR, cross-GPU transfers, downgrade, concurrency invariants |

## Suggested Reading Order

1. **Architecture Overview** — understand the component layout and thread model
2. **Execution Flow** — trace a query end-to-end through the system
3. **Physical Plan Generation** — how DuckDB logical plans become Sirius pipelines
4. **Operators** — what each operator does on the GPU
5. **Expression Executor** — how expressions are evaluated on GPU
6. **Pipeline Execution** — how tasks are dispatched and executed
7. **Task Creator** — how the system decides when to create tasks
8. **Scan** — how data enters the system from storage
9. **Memory Management** — GPU memory tiers, reservations, spilling
10. **Data Management** — data batch lifecycle and port wiring
11. **Configuration** — tuning knobs and runtime settings
12. **Optimizations** — performance improvements and their mechanisms

<!-- last-updated-commit: d6baaedc0d2bb07b27a00c65135513f3c23f0b37 -->
