---
name: doris-query-testing
description: Iterate on testing SQL queries against the Sirius-Doris cluster — start/stop the cluster, warm up BEs, run single-BE and multi-BE exchange queries, inspect logs, and diagnose issues.
allowed-tools: Bash, Read, Grep, Glob, Write, Edit
---

# Doris Query Testing

You are testing SQL queries against a Sirius-Doris cluster: a Doris Frontend (FE) coordinating GPU-accelerated Sirius Backend (BE) nodes.

## Quick Reference

| Action | Command |
|--------|---------|
| Start cluster (2 BEs) | `pixi run -e doris-nixl cluster-start` |
| Stop cluster | `pixi run -e doris-nixl cluster-stop` |
| Cluster status | `pixi run -e doris-nixl cluster-status` |
| Run SQL | `pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root < query.sql` |
| BE logs | `/tmp/sirius-cluster/be-{1,2}.log` |
| Build Rust | `pixi run -e doris-nixl doris-build` |
| Build C++ | `pixi run -- ninja` (from repo root) |

## Running Queries

**Always use a SQL file** — inline `-e` doesn't work well with Doris syntax (commas, quotes).

```bash
cat > /tmp/claude-1000/query.sql << 'EOF'
SET query_timeout = 600;
SELECT * FROM local("file_path" = "/data/tpch/sf1/snappy/nation.parquet", "format" = "parquet", "backend_id" = "BACKEND_ID_HERE") LIMIT 10;
EOF
pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root < /tmp/claude-1000/query.sql
```

### Get Backend IDs

```bash
pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root -N -e "SHOW BACKENDS" | awk -F'\t' '{print "id=" $1, "host=" $2, "alive=" $10}'
```

Typical output: BE1 = `1772613976744` (127.0.0.2), BE2 = `1772613976783` (127.0.0.3). IDs change on restart.

### Doris `local()` TVF Syntax

```sql
-- Doris 4.0 syntax (NOT standard SQL function call syntax)
SELECT * FROM local("file_path" = "/path/to/file.parquet", "format" = "parquet", "backend_id" = "ID");
```

- Use `"param" = "value"` (double-quoted, equals sign), NOT `param => 'value'`
- `file_path`: absolute host path (no `file://` prefix)
- `format`: `parquet`, `csv`, `json`
- `backend_id`: target a specific BE (from SHOW BACKENDS)
- FE needs `enable_outfile_to_local=true` and `enable_access_file_without_broker=true` in fe.conf

### Data Locations

TPC-H data is at `/data/tpch/sf{1,10,100,1000}/snappy/*.parquet`:
- `nation.parquet` (25 rows) — fast, good for smoke tests
- `region.parquet` (5 rows) — tiny
- `lineitem.parquet` (6M rows at SF1) — stress test
- `customer.parquet`, `orders.parquet`, `supplier.parquet`, `part.parquet`, `partsupp.parquet`

## Warm-Up Procedure

GPU cold start on a fresh DuckDB connection compiles the Sirius GPU pipeline, which takes ~30s. The FE's per-fragment RPC timeout is 30s, so the first query often times out. **Always warm up before testing.**

```bash
# Warm up each BE individually with a simple SELECT *
# (count(*) won't work — DuckDB optimizes it to metadata-only, no GPU scan)
cat > /tmp/claude-1000/warmup.sql << 'EOF'
SET query_timeout = 600;
SELECT * FROM local("file_path" = "/data/tpch/sf1/snappy/nation.parquet", "format" = "parquet", "backend_id" = "BE1_ID") LIMIT 5;
EOF
pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root < /tmp/claude-1000/warmup.sql
# Repeat with BE2_ID
```

**Important**: Use `SELECT *` (not `count(*)`) for warmup. `count(*)` gets optimized to `DUMMY_SCAN` → `EXPRESSION_GET` which bypasses the GPU scan pipeline entirely and hangs the GPU executor (no scan operators → no tasks scheduled → `future.get()` blocks forever).

## Testing Patterns

### Single-BE Query (no exchange)

```sql
SET query_timeout = 600;
SELECT * FROM local("file_path" = "/data/tpch/sf1/snappy/nation.parquet", "format" = "parquet", "backend_id" = "BE1_ID");
```

Executes entirely on one BE. Tests: GPU scan, plan translation, result delivery.

### Multi-BE Exchange (UNION ALL)

```sql
SET query_timeout = 600;
SELECT * FROM local("file_path" = "/data/tpch/sf1/snappy/nation.parquet", "format" = "parquet", "backend_id" = "BE1_ID")
UNION ALL
SELECT * FROM local("file_path" = "/data/tpch/sf1/snappy/nation.parquet", "format" = "parquet", "backend_id" = "BE2_ID");
```

FE sends fragments to both BEs. Each BE scans locally, then sends results via exchange (nixl GPU-direct or bRPC) to the result-collecting BE. Tests: exchange path, PBlock encoding, nixl transfer.

Expected: 50 rows (25 × 2).

### Aggregate Over Exchange

```sql
SET query_timeout = 600;
SELECT n_regionkey, count(*) FROM local("file_path" = "/data/tpch/sf1/snappy/nation.parquet", "format" = "parquet", "backend_id" = "BE1_ID")
GROUP BY n_regionkey
UNION ALL
SELECT n_regionkey, count(*) FROM local("file_path" = "/data/tpch/sf1/snappy/nation.parquet", "format" = "parquet", "backend_id" = "BE2_ID")
GROUP BY n_regionkey;
```

**Warning**: Hash-partitioned exchange (GROUP BY across BEs without UNION ALL) is NOT yet implemented — each AGG_FINAL receives ALL data, causing doubled values. Use UNION ALL for now.

## Inspecting Logs

### Filtering BE Logs

```bash
# Remove heartbeat noise
grep -v heartbeat /tmp/sirius-cluster/be-1.log | grep -v publish_topic | tail -50

# Key patterns to grep for:
grep -E "gpu_execution|execute_plan|retain|detect_execution|nixl|exchange" /tmp/sirius-cluster/be-1.log

# Exchange flow
grep -E "merge_fragment|remote exchange|exchange data|exchange PBlock|exchange fragment" /tmp/sirius-cluster/be-1.log

# GPU pipeline diagnostics
grep -E "logical_tree|scan_operators|schedule_next|prepare_for_query" /tmp/sirius-cluster/be-1.log

# Errors and warnings only
grep -E "WARN|ERROR|error|fail|panic" /tmp/sirius-cluster/be-1.log
```

### Key Log Messages and What They Mean

| Log Pattern | Meaning |
|-------------|---------|
| `merged fragment plans for single-BE execution merged_fragments=1` | Fragments merged — single-BE path |
| `fragment has remote exchanges, skipping merge` | Multi-BE exchange path triggered |
| `retain_gpu_buffers set before GPU execution` | GPU buffer retention enabled for nixl |
| `gpu_execution_substrait starting substrait_bytes=N` | GPU execution starting |
| `executed via gpu_execution_substrait elapsed_ms=N` | GPU execution completed |
| `detect_execution_location: GPU buffers found num_buffers=N` | GPU buffers detected for nixl |
| `detect_execution_location done detect_ms=N gpu=true` | Will use nixl GPU-direct path |
| `detect_execution_location done detect_ms=N gpu=false` | Will use bRPC CPU path |
| `nixl GPU-direct transfer complete` | nixl transfer succeeded |
| `nixl transfer failed, falling back to bRPC` | nixl failed, using bRPC (common for self-transfer) |
| `exchange fragment execution complete` | Exchange fragment done |
| `NO scan operators in queue — pipeline will hang!` | GPU plan has no scan ops (count(*) optimization, or plan error) |
| `[prepare_for_query] found N scan operators` | GPU pipeline has N scan operators ready |
| `logical_tree: type=GET children=0` | Parquet scan in logical plan (good) |
| `logical_tree: type=DUMMY_SCAN children=0` | No real scan — count(*) optimization (bad for GPU) |

### Nixl Transfer Flow (successful)

```
BE1 (sender):
  retain_gpu_buffers set → gpu_execution → GPU buffers found →
  registered GPU buffers with nixl agent → loaded remote nixl metadata →
  nixl transfer completed → transfer_complete acknowledged → exchange send complete

BE2 (receiver):
  transfer_complete: nixl transfer done → building PBlock from IPC →
  released pending GPU buffers → fed PBlock into ExchangeBuffer →
  all exchange data received → decoded exchange PBlocks → execute exchange SQL
```

## Diagnosing Hangs

### GPU Execution Hangs (`gpu_execution_substrait starting` but no completion)

1. **Check for scan operators**: Look for `[prepare_for_query] found 0 scan operators` and `NO scan operators in queue`. This means the Substrait plan produced a trivial logical plan (DUMMY_SCAN, EXPRESSION_GET) with no real data scan.
   - **Cause**: DuckDB optimizer pushes count(*) to metadata, or the Substrait plan doesn't include a ReadRel
   - **Fix**: Use `SELECT *` instead of `count(*)`, or ensure the plan has actual scan ops

2. **Connection busy**: Check if `retain_gpu_buffers` was called before the hang. If using `execute_batch()` with SELECT table functions, the connection gets stuck.
   - **Fix**: Use `prepare() + query_arrow().collect()` to properly consume results

3. **Engine mutex contention**: If exchange fragment async tasks hold `engine.lock()` before the leaf can acquire it.
   - **Diagnostic**: Check if multiple fragments try to lock simultaneously

### Exchange Query Hangs

1. **FE timeout loop**: FE sends fragment, times out after 30s, cancels, retries — but the GPU execution from the first attempt still holds the engine mutex.
   - **Fix**: Warm up BEs first; increase `query_timeout`

2. **Exchange data never arrives**: Check `all exchange data received` on the receiving BE. If missing, the sender failed or bRPC didn't connect.
   - **Check**: Sender logs for `exchange send complete`; receiver logs for `registering exchange for fragment`

### Query Returns Wrong Results

1. **Hash-partitioned exchange**: GROUP BY across BEs doubles values (not yet implemented). Use UNION ALL.
2. **Self-transfer nixl failure**: Expected — nixl rejects same-agent metadata. Falls back to bRPC automatically.
3. **Stale GPU buffers**: Exchange fragments (SqlCpuOnly) skip GPU buffer detection to avoid detecting stale buffers from a previous leaf execution.

## Build-Test Cycle

```bash
# 1. Edit Rust code (doris/crates/*)
# 2. Build
pixi run -e doris-nixl doris-build

# 3. Restart cluster (picks up new binary)
pixi run -e doris-nixl cluster-stop && pixi run -e doris-nixl cluster-start

# 4. Warm up both BEs (see above)

# 5. Run test query
pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root < /tmp/claude-1000/query.sql

# 6. Inspect logs
grep -v heartbeat /tmp/sirius-cluster/be-1.log | grep -v publish_topic | tail -50
```

For C++ changes (Sirius extension):
```bash
# 1. Edit C++ code (src/*)
# 2. Build
pixi run -- ninja

# 3. Same restart + warmup + test cycle
```

## Cluster Architecture

```
┌──────────┐     gRPC (exec_plan_fragment)     ┌─────────────┐
│ Doris FE │────────────────────────────────────│  Sirius BE1 │
│          │     Thrift (heartbeat, backend)    │  127.0.0.2  │
│ :9030    │────────────────────────────────────│  :19050/     │
│ :8030    │                                    │   18060      │
└──────────┘                                    └──────┬──────┘
      │                                                │ nixl (GPU→GPU)
      │           gRPC (exec_plan_fragment)             │ or bRPC
      └────────────────────────────────────────┌───────┴──────┐
                                               │  Sirius BE2  │
                                               │  127.0.0.3   │
                                               │  :29050/     │
                                               │   28060      │
                                               └──────────────┘
```

- FE: query parsing, planning, fragment distribution
- BE: GPU execution (Sirius/DuckDB), exchange (nixl/bRPC), result delivery (Arrow Flight)
- Exchange: leaf fragments scan data → send via exchange → collector fragment assembles results
