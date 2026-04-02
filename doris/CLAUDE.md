# Doris BE (Rust) — Sirius GPU Backend

GPU-accelerated Apache Doris Backend in Rust. Receives query fragments from the
Doris FE, translates them to Substrait/SQL, executes via DuckDB (GPU or CPU),
returns results via Arrow Flight.

## Build Commands

```bash
# C++ GPU engine (from repo root)
pixi run -- ninja                        # incremental (uses build/release/)
pixi run make release                    # full rebuild

# Rust BE (includes nixl C++ build)
pixi run -e doris-nixl doris-build       # with nixl (GPU-direct exchange)
pixi run -e doris doris-build            # without nixl (CPU/bRPC only)

# Doris FE (Java)
pixi run -e doris-fe doris-fe-build

# Substrait DuckDB extension
pixi run -e doris substrait-build
```

## Running

### Cluster startup (preferred — handles FE, BE registration, GPU memory, lock files)

```bash
# Start FE + 2 Sirius GPU BEs (handles cleanup, registration, health checks)
pixi run -e doris -- bash doris/scripts/start-cluster.sh 2

# Check status
pixi run -e doris -- bash doris/scripts/start-cluster.sh --status

# Stop everything
pixi run -e doris -- bash doris/scripts/start-cluster.sh --stop
```

The script:
- Kills any previous FE/BEs, cleans lock files
- Starts FE (via pixi doris-fe env for JAVA_HOME), waits for health
- Drops stale backends from FE
- Starts N BEs with separate HOME dirs (avoids sirius.lock conflicts)
- Waits for all BEs to register and become alive
- Config: `~/.sirius/sirius.cfg` is copied to each BE's HOME. For 2 BEs,
  set `usage_limit_fraction ≤ 0.3` to avoid GPU OOM.

### Manual startup (individual terminals)

```bash
pixi run -e doris-fe doris-fe-start      # FE (terminal 1)
pixi run -e doris sirius-be              # BE 1 (terminal 2)
pixi run -e doris sirius-be-2            # BE 2 (terminal 3, separate ports/home)
pixi run -e doris-fe doris-fe-add-sirius # manual FE registration if needed
```

Connect: `pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root`

### GPU warmup (required before queries)

First query triggers GPU pipeline compilation (~30s). FE times out at 30s.
Always warm up each BE individually before running real queries:

```bash
# Get backend IDs
pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root -N -e "SHOW BACKENDS" | awk -F'\t' '{print "id="$1, "host="$2, "alive="$10}'

# Warm up each BE (use SELECT *, NOT count(*) — count gets optimized away)
pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root -e "SET query_timeout=600; SELECT * FROM local(\"file_path\"=\"/data/tpch/sf1/snappy/nation.parquet\", \"format\"=\"parquet\", \"backend_id\"=\"BE_ID\") LIMIT 5"
```

## Testing

```bash
# Plan translator unit/integration tests (109 tests)
pixi run -e doris -- cargo test -p plan-translator

# All workspace tests
pixi run -e doris -- cargo test --workspace

# End-to-end TPC-H (requires running FE + BE)
pixi run -e doris -- bash doris/scripts/run-tpch.sh --skip-build --queries 1,6,14
```

TPC-H data: `/data/tpch/sf{1,10,100,1000}/snappy/*.parquet` (single files),
`/data/tpch/sf1/p16/snappy/{table}/` (16 partitions per table).

### Distributed query testing

Use `shared_storage=true` with partitioned data for multi-BE queries. The FE
splits individual files across BEs (directory paths are expanded by the BE's
glob RPC). Always use SQL files for queries (inline `-e` breaks on commas):

```bash
cat > /tmp/test.sql << 'EOSQL'
SET query_timeout = 600;
SELECT l_returnflag, COUNT(*) FROM local(
  "file_path"="/data/tpch/sf1/p16/snappy/lineitem/",
  "format"="parquet", "shared_storage"="true"
) GROUP BY l_returnflag ORDER BY l_returnflag;
EOSQL
pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root < /tmp/test.sql
```

## Workspace Crates

| Crate | Purpose |
|-------|---------|
| `sirius-doris-be` | Main binary — server startup, FE registration, CLI args |
| `doris-rpc` | All RPC handlers (~7k LOC): gRPC, Thrift, bRPC, Arrow Flight, exchange, nixl |
| `plan-translator` | Doris TPlan → Substrait plans or SQL (109 regression tests) |
| `result-formatter` | ResultStore + Arrow Flight server |
| `sirius-ffi` | DuckDB FFI — GPU/CPU/SQL execution, file registration |
| `doris-thrift` | Thrift codegen from Doris IDL (build.rs) |
| `doris-proto` | Protobuf codegen: gRPC + bRPC + nixl protos (tonic-build) |
| `nixl-test` | Standalone nixl exchange tests |

## Code Conventions

- **Tracing**: Use `#[instrument]` spans with field attributes, NOT `info!`/`warn!` events.
  Subscriber uses `FmtSpan::NEW | FmtSpan::CLOSE`.
- **Error handling**: `anyhow` for application errors, `thiserror` for library crate errors.
- **Async**: Thrift servers are blocking (dedicated threads); gRPC/Flight/bRPC are async (tokio).
- **Arrow versions**: Workspace uses arrow 54; sirius-ffi uses arrow 56 (matches DuckDB crate).
  Two tonic versions coexist: 0.12 (arrow-flight) and 0.13 (doris-rpc).

## Key Pitfalls

- **DuckDB connection is !Sync**: Wrapped in `Arc<Mutex<SiriusEngine>>`. Use `prepare() +
  query_arrow().collect()` for table functions, NOT `execute_batch` (leaves connection "busy").
- **Exchange fragments use GPU Substrait**: Packed GPU tables (from nixl) are registered
  via `register_packed_table` and executed on GPU. Only CPU-registered exchange tables
  (from bRPC PBlock decode) must use `SqlCpuOnly` to avoid DuckDB INTERNAL errors.
- **`Notify::notify_one()` not `notify_waiters()`**: The latter doesn't store a permit —
  notifications are lost if no task is waiting. ExchangeBuffer uses `notify_one()`.
- **Thrift deserialization**: FE uses TBinaryProtocol (NOT compact) for fetch_data results
  and TFileScanRange. Check protocol before adding new Thrift deserialization.
- **Result keying**: FE calls fetch_data with `query_id`, not fragment instance IDs.
- **Stale GPU buffers**: After `SqlCpuOnly`, `get_last_gpu_result_buffers()` returns stale
  buffers from a previous GPU execution. Skip `detect_execution_location` → always CPU.
- **sender_id**: Each exchange fragment must use its `local_params[0].sender_id`, not 0.
- **count(*) without GROUP BY**: DuckDB optimizes to DUMMY_SCAN, GPU engine hangs.
  Always use `SELECT *` for GPU warmup, not `count(*)`.
- **DuckDB SetRel(UNION_ALL) broken**: Exchange-collecting fragments use SQL path instead.
- **Hash partitioned exchange**: GPU path uses `cudf::hash_partition` (MURMUR3) + per-partition
  `cudf::chunked_pack` into staging, transferred via nixl. CPU fallback available with
  `--allow-brpc-fallback` (uses CRC32/CRC32C hash in `hash_partitioner.rs`).
  Only SLOT_REF partition expressions supported (not CAST/FUNCTION_CALL).

## Doris Protocol Notes

- FE sends VERSION_3: `TPipelineFragmentParamsList` with shared fields in first fragment.
- `EXCHANGE_NODE(0 children)`: result-collector (single-BE) or exchange receiver (multi-BE).
- `per_exch_num_senders`: determines local vs remote exchange routing.
- PBlock encoding: raw bytes if ≤256B, else StreamVByte (fixed) or LZ4 (string chars).
- bRPC uses baidu_std wire format: `"PRPC"` magic + body_size + meta_size + payload.
- `local()` TVF: use `file_path` (not `filepath`), needs `backend_id` or `shared_storage`.

## Fragment Execution Flow

1. FE sends `exec_plan_fragment` (gRPC, VERSION_3)
2. Deserialize Thrift `TPipelineFragmentParamsList`, merge shared fields
3. Classify: leaf (FILE_SCAN), exchange root, intermediate
4. Merge decision via `per_exch_num_senders` (local → merge, remote → async wait)
5. Register parquet views, translate TPlan → Substrait or SQL
6. Execute: GPU (`gpu_execution_substrait`) → CPU (`from_substrait`) → SQL fallback
7. Route result: exchange sink (bRPC/nixl) or result sink (Arrow Flight)
