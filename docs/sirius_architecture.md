# Sirius GPU Extension — Architecture & Mental Model

> Reference doc for understanding the codebase before optimizing ClickBench.
> Machine: Chameleon GH200 (original target) / Quadro RTX 6000 x86_64 (current: `cudf-25.12` branch)

---

## 1. What Sirius Is

Sirius is a **DuckDB extension** that intercepts SQL queries and executes them on the GPU using **libcudf (RAPIDS)** instead of DuckDB's CPU engine. It is not a fork of DuckDB — it builds on top of DuckDB v1.2.1 and adds two table functions:

```sql
-- Step 1: allocate GPU memory pools (once per session)
call gpu_buffer_init('19 GB', '19 GB', pinned_memory_size = '100 GB');

-- Step 2: run any SQL on GPU
call gpu_processing('SELECT COUNT(*) FROM hits');
```

If the GPU plan fails for any reason, it transparently falls back to DuckDB CPU execution.

---

## 2. Query Execution Flow

```
User: call gpu_processing("SELECT ...")
            │
            ▼
  sirius_extension.cpp
  GPUProcessingFunction()
            │
            ├─ Parse SQL via DuckDB planner
            │     → produces DuckDB LogicalOperator tree
            │
            ├─ SiriusInitPlanExtractor
            │     → extracts the logical tree
            │
            ├─ GPUPhysicalPlanGenerator (visitor pattern)
            │     → maps each LogicalOp → GPUPhysicalOp
            │     → throws NotImplementedException for unsupported ops
            │
            ├─ GPUContext::GPUExecuteQuery()
            │     │
            │     ├─ GPUExecutor::Initialize()
            │     │     → builds GPUMetaPipeline tree
            │     │     → creates GPUPipeline graph (with dependencies)
            │     │
            │     └─ Execute each GPUPipeline in order:
            │           source.GetData()        → GPU columns from DuckDB table
            │           operator_1.Execute()    → Filter (WHERE)
            │           operator_2.Execute()    → Projection (SELECT exprs)
            │           operator_3.Execute()    → GroupedAggregate (GROUP BY)
            │           ...
            │           sink.Sink()             → collects output
            │
            └─ GPUPhysicalResultCollector::GetResult()
                  → transfers GPU columns back to CPU
                  → wraps as DuckDB MaterializedQueryResult

On any exception: fallback to standard DuckDB CPU execution
```

**Key files:**

| File | Role |
|------|------|
| `src/sirius_extension.cpp` | Entry points, DuckDB registration, fallback logic |
| `src/gpu_physical_plan_generator.cpp` | Logical → GPU physical plan (visitor) |
| `src/gpu_executor.cpp` | Pipeline scheduling and execution |
| `src/gpu_context.cpp` | Query context, GPU/CPU bridge, error recovery |
| `src/gpu_pipeline.cpp` | Pipeline (source → ops → sink) |
| `src/gpu_meta_pipeline.cpp` | Grouped pipelines with shared sinks |
| `src/gpu_buffer_manager.cpp` | GPU/CPU memory pool management (RMM) |
| `src/gpu_columns.cpp` | Column representation (DuckDB ↔ GPU types) |
| `src/gpu_physical_operator.cpp` | Base class: source/operator/sink interfaces |

---

## 3. Memory Architecture

```
CPU                                          GPU
──────────────────────────────────────────────────────────
DuckDB heap (table chunks)
        │
        │  callCudaMemcpyHostToDevice()
        ▼                                    GPU Cache Pool (RMM)
Pinned Memory ──────────────────────────►  Cached columns
(page-locked,                              (frequently accessed)
 fast DMA)
        ▲                                    GPU Processing Pool (RMM)
        │                                    Intermediate operator results
        │  callCudaMemcpyDeviceToHost()      Join hash tables
        │                                    Sort buffers
CPU Result Buffer ◄─────────────────────    Final aggregated data
        │
DuckDB result
```

**GPUBufferManager** is a singleton with:
- **GPU Cache**: holds table data resident on GPU across queries
- **GPU Processing**: temporary workspace for operators
- **CPU Cache / CPU Processing**: overflow if GPU fills up
- Backed by **RMM pool allocator** wrapping `cuda_memory_resource`
- Pinned CPU memory (`allocatePinnedCPUMemory`) for fast DMA — enabled by default when data >= cache_size

---

## 4. GPU Operators — What's Implemented

### ✅ Fully GPU-Accelerated

| Operator | CUDA Implementation | Notes |
|----------|-------------------|-------|
| Table Scan | `GetDataDuckDB()` + memcpy | DuckDB reads from disk, transfers to GPU |
| Filter (WHERE) | `expression_executor/gpu_dispatch_select.cu` | Evaluates predicates on GPU |
| Projection (SELECT) | `expression_executor/gpu_dispatch_materialize.cu` | Expression evaluation |
| Hash Join (inner/left/right/semi/anti) | `cuda/operator/hash_join_*.cu` | Multi-key, CUB prefix scan |
| Nested Loop Join | `cuda/operator/nested_loop_join.cu` | For small tables |
| Grouped Aggregate (GROUP BY) | `cuda/cudf/cudf_groupby.cu` | libcudf groupby API |
| Ungrouped Aggregate | `cuda/cudf/cudf_aggregate.cu` | SUM/COUNT/AVG/MIN/MAX |
| Order By | `cuda/cudf/cudf_orderby.cu` | libcudf sort (stable, multi-col) |
| Top-N | custom CUDA kernel | Config: `use_custom_top_n` |
| Limit | `gpu_physical_limit` | Simple row slice |
| DISTINCT | `cuda/cudf/cudf_duplicate_elimination.cu` | libcudf unique |
| String Matching | `cuda/operator/strings_matching.cu` | LIKE, simple regex |
| Substring | `cuda/operator/substring.cu` | SUBSTRING() |
| CTE | `gpu_physical_cte` | Materialized CTEs only |
| Delim Join | `gpu_physical_delim_join` | Correlated subqueries |

### ❌ Not Implemented → CPU Fallback

- WINDOW functions
- UNNEST / CROSS PRODUCT
- UNION / EXCEPT / INTERSECT
- INSERT / UPDATE / DELETE
- ASOF join, POSITIONAL join
- RECURSIVE CTE, PIVOT
- `REGEXP_REPLACE` (detected in `fallback.cpp`, triggers fallback)

---

## 5. Expression Executor

The GPU expression executor lives in `src/expression_executor/` and handles:

```
WHERE clause  →  GpuExpressionExecutor::Select()   → selection vector (bitmask)
SELECT exprs  →  GpuExpressionExecutor::Execute()  → new GPU columns
```

**Supported expression types:**
- `BoundComparisonExpression`: `<`, `>`, `=`, `!=`, `<=`, `>=`
- `BoundConjunctionExpression`: `AND`, `OR`
- `BoundCaseExpression`: `CASE WHEN...THEN...END`
- `BoundBetweenExpression`: `BETWEEN x AND y`
- `BoundCastExpression`: type casting
- `BoundFunctionExpression`: built-in functions
- `BoundOperatorExpression`: arithmetic `+`, `-`, `*`, `/`
- `BoundConstantExpression`: literals
- `BoundReferenceExpression`: column references

**Key config:**
```sql
SET use_cudf_expr = true;  -- compile to libcudf column operations (default: true)
```
When enabled, expressions are compiled into libcudf column ops. Falls back to custom CUDA kernels if a specific expression isn't handled.

---

## 6. ClickBench Query Characteristics

The 43 ClickBench queries on the `hits` table (100M rows, 105 columns, web analytics data) exercise these patterns:

| Pattern | Example Queries | GPU Handling |
|---------|----------------|--------------|
| Simple COUNT/SUM/AVG | Q1–Q7 | Fully GPU (ungrouped agg) |
| GROUP BY + ORDER BY + LIMIT | Q8–Q18, Q25–Q28 | Fully GPU |
| Filter + GROUP BY (high selectivity) | Q36–Q43 | Fully GPU |
| VARCHAR GROUP BY (search phrases, URLs) | Q13–Q15, Q33–Q35 | **GPU fallback observed** |
| LIKE string patterns | Q20–Q23 | GPU (string matching) |
| `SELECT *` + ORDER BY (wide row) | Q24 | GPU but needs large cache (`80 GB`) |
| `REGEXP_REPLACE` | Q29 | **CPU fallback** (known, detected in fallback.cpp) |
| `SUM(col+N)` × 89 | Q30 | Many expression evaluations |
| CASE WHEN | Q40 | GPU (expression executor) |
| `DATE_TRUNC` + GROUP BY | Q43 | GPU |

**Notable Q24**: requires `80 GB` caching region — this is the `SELECT *` query that reads the entire 100M-row dataset into GPU memory. Special-cased in `runner.py`.

**Known fallback triggers:**
- `REGEXP_REPLACE` → explicitly detected in `src/fallback.cpp`
- VARCHAR-heavy GROUP BY on strings (GPU groupby on strings sometimes fails)

---

## 7. Configuration Options

```sql
-- Memory
SET use_pin_memory = true;                  -- pinned CPU↔GPU transfers (default: true)

-- Execution
SET use_cudf_expr = true;                   -- libcudf expression compilation
SET use_custom_top_n = true;                -- custom Top-N CUDA kernel
SET use_opt_table_scan = true;              -- multi-stream table scan
SET opt_table_scan_num_streams = 4;         -- concurrent CUDA streams for scan
SET opt_table_scan_memcpy_size = 67108864;  -- per-stream memcpy chunk size

-- Debug
SET print_gpu_table_max_rows = 1000;
SET enable_fallback_check = true;           -- check for unsupported ops before exec
SET enable_regex_jit_impl = true;           -- JIT compile regex patterns
```

---

## 8. Build Notes (x86_64 machine, Quadro RTX 6000)

The `cudf-25.12` branch was originally written for the **GH200** (ARM64, Hopper sm_90a). The following changes were made to run on this x86_64 + RTX 6000 (Turing sm_75) machine:

| Change | File | Before | After |
|--------|------|--------|-------|
| Platform flag | `build.sh` | `linux_arm64` | `linux_amd64` |
| CUDA arch | `build.sh` | hardcoded `90a-real` | detected via `nvidia-smi` → `75-real` |
| CUDA arch env | `pixi.toml` | `CUDAARCHS = "90a-real"` | removed (handled in build.sh) |
| LIBCUDF_ENV_PREFIX | `build.sh` | not set | set to `$PIXI_ENV` |

**Substrait** is NOT a git submodule — it must be cloned manually:
```bash
cd duckdb && mkdir -p extension_external && cd extension_external
git clone https://github.com/duckdb/substrait.git
cd substrait && git reset --hard ec9f8725df7aa22bae7217ece2f221ac37563da4
```

**Rebuild from scratch:**
```bash
cd ~/sirius
~/.pixi/bin/pixi install   # only needed once
bash build.sh clean        # clean + rebuild
bash build.sh              # incremental rebuild
```

**Versions:**
- DuckDB: v1.2.1 (submodule @ `8e52ec4`)
- libcudf: 25.12.0 (via pixi)
- librmm: 25.12.0 (via pixi)
- CUDA: 12.6 (system) / 12.9 (pixi nvcc)
- GPU: Quadro RTX 6000, sm_75, 24GB VRAM

---

## 9. Running Benchmarks

**TPC-H (SF1):**
```bash
# Generate data (one-time)
cd test_datasets/tpch-dbgen && chmod +x dbgen && ./dbgen -s 1 -f && mkdir -p s1 && mv *.tbl s1/

# Load
export LD_LIBRARY_PATH=".pixi/envs/default/lib:$LD_LIBRARY_PATH"
./build/release/duckdb tpch.duckdb < tpch_load.sql

# Run (wraps each query in gpu_processing())
bash run-tpch.sh
```

**ClickBench (1 parquet shard ~1M rows for dev, full 100M for benchmarking):**
```bash
# Download one shard for dev/testing
wget -O test_datasets/hits_0.parquet \
  "https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet"

# Load
./build/release/duckdb clickbench.duckdb \
  -c "CREATE TABLE hits AS SELECT * FROM read_parquet('test_datasets/hits_0.parquet');"

# Run via runner
cd scripts/clickbench_runner
python3 runner.py --dataset_path ../../test_datasets/hits_0.parquet \
  --caching_region_size "4 GB" --processing_region_size "4 GB"
```

---

## 10. Key Observations for ClickBench Optimization

1. **VARCHAR GROUP BY falls back to CPU** — queries grouping on `SearchPhrase`, `URL`, `MobilePhoneModel` trigger GPU fallback. libcudf supports string groupby, so this is likely a type-mapping or expression issue worth fixing.

2. **Q30 (89× SUM expressions)** — 89 separate `SUM(ResolutionWidth + N)` expressions. May benefit from fused kernel instead of 89 separate expression evaluations.

3. **Q24 requires 80GB cache** — the `SELECT *` all-columns query needs the entire dataset in GPU memory. On a 24GB GPU this will need to fall back to batched/streamed execution.

4. **`REGEXP_REPLACE` (Q29)** — currently always falls back. Could potentially add libcudf string::replace_re() support.

5. **Table scan is the bottleneck for small queries** (Q1, Q2, Q20 LIKE) — PCIe bandwidth is the limit; pinned memory and multi-stream scan help.

6. **Expression executor path**: most WHERE predicates compile to libcudf column ops (`use_cudf_expr=true`). Complex expressions may fall through to custom CUDA kernels which are less optimized.
