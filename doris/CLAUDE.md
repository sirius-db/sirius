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
pixi run substrait-build
```

## Running

```bash
pixi run -e doris-fe doris-fe-start      # FE (terminal 1)
pixi run -e doris sirius-be              # BE 1 (terminal 2)
pixi run -e doris sirius-be-2            # BE 2 (terminal 3, separate ports/home)
pixi run -e doris-fe doris-fe-add-sirius # manual FE registration if needed
```

Connect: `pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root`

## Testing

```bash
# Plan translator unit/integration tests (109 tests)
pixi run -e doris -- cargo test -p plan-translator

# All workspace tests
pixi run -e doris -- cargo test --workspace

# End-to-end TPC-H (requires running FE + BE)
./doris/scripts/run-tpch.sh --skip-build --queries 1,6,14
```

TPC-H data at `/data/tpch/sf{1,10,100,1000}/snappy/*.parquet`.

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
- **Exchange fragments MUST use `SqlCpuOnly`**: Running `gpu_execution()` on a CPU-side
  exchange table causes DuckDB INTERNAL error that invalidates the database.
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
- **Hash partitioned exchange (CPU path)**: `hash_partitioner.rs` routes rows by CRC32/CRC32C
  hash of partition columns. GPU path (Phase 4) not yet implemented — falls back to CPU bRPC.
  Supports `HASH_PARTITIONED` and `BUCKET_SHUFFLE_HASH_PARTITIONED`; only SLOT_REF partition
  expressions (not CAST/FUNCTION_CALL). Legacy DATE (type 9) hashes as string repr.

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
