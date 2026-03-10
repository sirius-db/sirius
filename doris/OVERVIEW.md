# Sirius-Doris: GPU-Accelerated Query Execution for Apache Doris

## Overview

Sirius-Doris is a GPU-accelerated query execution backend for [Apache Doris](https://doris.apache.org/). It replaces the standard Java-based Doris Backend (BE) with a Rust service that wraps the **Sirius C++ GPU engine** — a DuckDB extension providing native GPU execution for analytical queries via CUDA and RAPIDS/cuDF.

The system enables:
- **GPU-accelerated parquet scans** (native CUDA parquet reader)
- **GPU-accelerated analytical operators** (hash join, grouped aggregate, sort, top-N)
- **Multi-BE distributed execution** with inter-BE data exchange
- **GPU-direct memory transfers** between BEs via NVIDIA NIXL (UCX/RDMA)
- Full compatibility with the Doris FE (Frontend) query planner and MySQL wire protocol

```
                           Architecture Overview

  ┌─────────────────────────────────────────────────────────────────────┐
  │                        MySQL Client                                 │
  │                    mysql -h 127.0.0.1 -P 9030                       │
  └────────────────────────────┬────────────────────────────────────────┘
                               │ MySQL Protocol
                               ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      Apache Doris FE (Java)                         │
  │                                                                     │
  │  SQL Parser → Query Planner → Fragment Distribution → Result Merge  │
  │                                                                     │
  │  Ports: 9030 (MySQL), 8030 (HTTP), 9020 (RPC)                      │
  └──────┬──────────────────────────────────┬───────────────────────────┘
         │ gRPC (exec_plan_fragment)         │ Arrow Flight (fetch results)
         │ Thrift (heartbeat)                │
         ▼                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    Sirius GPU Backend (Rust)                         │
  │                                                                     │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
  │  │ Heartbeat    │  │ PBackend     │  │ Arrow Flight │              │
  │  │ Service      │  │ Service      │  │ Service      │              │
  │  │ (Thrift)     │  │ (gRPC)       │  │ (tonic)      │              │
  │  │ port 9050    │  │ port 8060    │  │ port 8071    │              │
  │  └──────┬───────┘  └──────┬───────┘  └──────────────┘              │
  │         │                 │                                         │
  │         │    ┌────────────┴─────────────────┐                       │
  │         │    │     Fragment Execution        │                       │
  │         │    │                               │                       │
  │         │    │  Thrift Deserialize           │                       │
  │         │    │  → Plan Translate (Substrait) │                       │
  │         │    │  → File Registration          │                       │
  │         │    │  → Execute (GPU or CPU)       │                       │
  │         │    │  → Result / Exchange          │                       │
  │         │    └────────────┬─────────────────┘                       │
  │         │                 │                                         │
  │         │    ┌────────────┴─────────────────┐                       │
  │         │    │       Sirius FFI             │                       │
  │         │    │   (DuckDB + Extensions)       │                       │
  │         │    └────────────┬─────────────────┘                       │
  │         │                 │                                         │
  └─────────┼─────────────────┼─────────────────────────────────────────┘
            │                 │
            │    ┌────────────┴─────────────────┐
            │    │   Sirius C++ GPU Engine       │
            │    │   (DuckDB Extension)          │
            │    │                               │
            │    │  GPU Pipeline Executor        │
            │    │  ├── PARQUET_SCAN (native)    │
            │    │  ├── HASH_JOIN               │
            │    │  ├── GROUPED_AGGREGATE       │
            │    │  ├── SORT / TOP_N            │
            │    │  ├── FILTER                  │
            │    │  └── RESULT_COLLECTOR        │
            │    │                               │
            │    │  GPU Buffer Manager (RMM)     │
            │    │  Last GPU Buffers (FFI)       │
            │    └──────────────────────────────┘
            │
            ▼
  ┌─────────────────────┐
  │  Doris FE           │
  │  (heartbeat acks)   │
  └─────────────────────┘
```

---

## Multi-BE Exchange Architecture

When queries span multiple BEs (e.g., `UNION ALL` across partitioned data), the FE distributes fragments to each BE and coordinates data exchange:

```
  ┌────────────┐                                      ┌────────────┐
  │   BE 1     │                                      │   BE 2     │
  │            │                                      │            │
  │ ┌────────┐ │     bRPC (PBlock)                    │ ┌────────┐ │
  │ │  Scan  │ │  ─────────────────────────────────▶  │ │Exchange│ │
  │ │Fragment│ │     or                               │ │Fragment│ │
  │ │(leaf)  │ │     NIXL GPU-direct (RDMA)           │ │(agg)   │ │
  │ └────────┘ │                                      │ └────────┘ │
  │            │                                      │            │
  │  GPU VRAM  │     ExchangeMetadata (gRPC)          │  GPU VRAM  │
  │ ┌────────┐ │  ◀─────────────────────────────────  │ ┌────────┐ │
  │ │src bufs│ │  ─────────────────────────────────▶  │ │dst bufs│ │
  │ └────────┘ │     UCX RDMA Transfer                │ └────────┘ │
  │            │  ─────────────────────────────────▶  │            │
  │            │     TransferComplete (gRPC)          │            │
  │            │  ─────────────────────────────────▶  │            │
  └────────────┘                                      └────────────┘

  Exchange Paths:
  ─────────────────────────────────────────────────────────────────
  CPU path (bRPC):   Arrow IPC → PBlock encode → baidu_std wire
                     → PBlock decode → DuckDB table → execute

  GPU path (NIXL):   GPU buffers → register with nixl agent
                     → exchange metadata (gRPC) → UCX RDMA transfer
                     → transfer complete → GPU→CPU → PBlock → execute
```

---

## Fragment Execution Flow

```
  FE sends exec_plan_fragment (gRPC, VERSION_3)
  │
  ├─ Deserialize Thrift TPipelineFragmentParamsList
  │   └─ Merge shared fields from first fragment (desc_tbl, query_globals, etc.)
  │
  ├─ Classify fragments:
  │   ├─ Leaf:         FILE_SCAN_NODE (no EXCHANGE children)
  │   ├─ Intermediate: Has EXCHANGE_NODE(0 children) as non-root
  │   └─ Exchange root: EXCHANGE_NODE(0 children) at root
  │
  ├─ Merge decision (per_exch_num_senders):
  │   ├─ All local → merge leaf into intermediate, single execution
  │   └─ Remote exchanges → async wait via ExchangeBuffer
  │
  ├─ For each executable fragment:
  │   ├─ Extract FILE_SCAN_NODE → file paths + format
  │   ├─ Register parquet views: CREATE VIEW ... parquet_scan([files])
  │   ├─ Translate TPlan → Substrait (or SQL fallback)
  │   ├─ Execute:
  │   │   ├─ GPU:     gpu_execution_substrait(plan_bytes)
  │   │   ├─ CPU:     from_substrait(plan_bytes)
  │   │   └─ SQL:     execute_sql(sql_string)
  │   │
  │   ├─ Detect execution location (GPU buffers? or CPU IPC?)
  │   │
  │   ├─ If EXCHANGE sink:
  │   │   ├─ GPU → NIXL transfer (or bRPC fallback)
  │   │   └─ CPU → bRPC (PBlock encoded, baidu_std wire)
  │   │
  │   └─ If result sink:
  │       └─ Store in ResultStore → FE fetches via Arrow Flight
  │
  └─ Return PStatus OK
```

---

## Rust Workspace Structure

```
doris/
├── Cargo.toml                       # Workspace: 8 crates
├── crates/
│   ├── sirius-doris-be/             # Main binary
│   │   ├── src/main.rs              #   Server startup, FE registration
│   │   └── src/config.rs            #   CLI args (clap derive)
│   │
│   ├── doris-thrift/                # Thrift codegen from Doris IDL
│   │   └── build.rs                 #   thrift --gen rs + post-processing
│   │
│   ├── doris-proto/                 # Protobuf codegen (gRPC + bRPC + nixl)
│   │   ├── build.rs                 #   tonic_build + prost_build
│   │   └── proto/                   #   brpc_meta, nixl_exchange, nixl_service
│   │
│   ├── doris-rpc/                   # All RPC protocol handlers (~7000 LOC)
│   │   ├── grpc_service.rs          #   PBackendService: exec_plan_fragment
│   │   ├── heartbeat_service.rs     #   FE heartbeat (Thrift)
│   │   ├── backend_service.rs       #   BackendService (Thrift, mostly stubbed)
│   │   ├── exchange_buffer.rs       #   Concurrent buffer for incoming exchange data
│   │   ├── exchange_sender.rs       #   bRPC transmit_block sender
│   │   ├── brpc_server.rs           #   bRPC frame parser + dispatcher
│   │   ├── pblock_decoder.rs        #   PBlock → Arrow (StreamVByte, LZ4)
│   │   ├── arrow_to_pblock.rs       #   Arrow → PBlock encoder
│   │   ├── nixl_exchange.rs         #   NIXL agent + UCX GPU transfer
│   │   ├── nixl_integration.rs      #   ExecutionLocation detection + routing
│   │   ├── nixl_service.rs          #   NixlMetadataService gRPC
│   │   ├── cuda_driver.rs           #   CUDA driver API via dlopen
│   │   └── fragment_manager.rs      #   Fragment lifecycle tracking
│   │
│   ├── plan-translator/             # Doris TPlan → Substrait (~109 tests)
│   │   ├── lib.rs                   #   translate_fragment()
│   │   ├── node_translator.rs       #   TPlanNode → Substrait Rel
│   │   ├── scan_translator.rs       #   FILE_SCAN_NODE → ReadRel(LocalFiles)
│   │   ├── expr_translator.rs       #   TExpr → Substrait expression
│   │   ├── descriptor_table.rs      #   TDescriptorTable parser
│   │   └── type_mapper.rs           #   Doris ↔ Substrait ↔ Arrow types
│   │
│   ├── result-formatter/            # Result storage + delivery
│   │   ├── result_store.rs          #   DashMap<FinstId, ResultEntry> + Notify
│   │   └── arrow_flight.rs          #   Arrow Flight server (GetSchema, DoGet)
│   │
│   ├── sirius-ffi/                  # DuckDB FFI + extension loading
│   │   └── src/lib.rs               #   SiriusEngine: GPU/CPU exec, file reg
│   │
│   └── nixl-test/                   # Standalone NIXL exchange tests
│
├── thirdparty/
│   ├── apache-doris/                # FE source + Thrift/Proto IDL (submodule)
│   ├── duckdb-substrait-extension/  # Substrait extension for DuckDB
│   ├── nixl-install/                # Pre-built NIXL binaries
│   └── thrift-0.16.0/              # Thrift compiler source
│
├── docker/
│   ├── docker-compose.yml           # FE + 2x GPU BE cluster
│   ├── fe-host.conf                 # FE config for local (non-Docker) runs
│   ├── sirius-host.cfg              # GPU memory config (libconfig)
│   └── sirius-doris-be/             # Docker BE binary
│
└── scripts/
    └── run-host-test.sh             # End-to-end test script
```

### C++ Sirius Engine (in `src/`)
```
src/
├── sirius_engine.cpp/.hpp           # Pipeline execution engine
├── gpu_buffer_manager.cpp/.hpp      # GPU memory management (RMM pools)
├── last_gpu_buffers.hpp             # GPU result buffer tracking (FFI bridge)
├── host_parquet_representation_converters.cpp  # H2D parquet copy
├── op/                              # Physical GPU operators
│   ├── sirius_physical_parquet_scan.*    # Native GPU parquet reader
│   ├── sirius_physical_hash_join.*      # Hash join
│   ├── sirius_physical_grouped_aggregate.*
│   ├── sirius_physical_sort*.hpp        # Sort / Top-N
│   └── ...                              # Filter, concat, table scan, etc.
├── pipeline/
│   └── sirius_pipeline.cpp/.hpp     # Pipeline scheduling
└── include/                         # Headers
```

---

## Key Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `tokio` | 1 | Async runtime for all RPC |
| `tonic` | 0.13 | gRPC (PBackendService, NIXL) |
| `thrift` | 0.17 | Thrift runtime (heartbeat, backend) |
| `arrow` | 54 | Arrow IPC for results |
| `arrow-flight` | 54 | Result delivery (uses tonic 0.12) |
| `substrait` | 0.52 | Substrait plan protobuf types |
| `duckdb` | 1.4.4 | CPU-only DuckDB (bundled) |
| `cudarc` | 0.19 | CUDA driver API (dlopen) |
| `mysql_async` | 0.34 | FE self-registration |
| `dashmap` | 6 | Concurrent maps (ExchangeBuffer, ResultStore) |
| `prost` | 0.13 | Protobuf serialization |
| `clap` | 4 | CLI arg parsing |
| `lz4_flex` | 0.11 | PBlock string compression |

---

## Protocol Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Wire Protocols                               │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│  Thrift      │  gRPC/Proto  │  bRPC        │  Arrow Flight          │
│  (blocking)  │  (async)     │  (baidu_std) │  (streaming)           │
├──────────────┼──────────────┼──────────────┼────────────────────────┤
│ Heartbeat    │ exec_plan_   │ transmit_    │ GetSchema              │
│ Service      │ fragment     │ block        │ DoGet                  │
│ (port 9050)  │ (port 8060)  │ (port 8060)  │ (port 8071)           │
│              │              │              │                        │
│ Backend      │ fetch_table_ │              │                        │
│ Service      │ schema       │              │                        │
│ (port 9060)  │ glob         │              │                        │
│              │              │              │                        │
│              │ NIXL:        │              │                        │
│              │ Exchange     │              │                        │
│              │ Metadata     │              │                        │
│              │ Transfer     │              │                        │
│              │ Complete     │              │                        │
└──────────────┴──────────────┴──────────────┴────────────────────────┘

bRPC Wire Format (baidu_std):
  [0:4]  "PRPC" magic
  [4:8]  body_size (u32 BE)
  [8:12] meta_size (u32 BE)
  [body] = [RpcMeta (protobuf)] + [Payload (PTransmitDataParams)]

PBlock Column Encoding:
  If size ≤ 256 bytes: raw bytes
  If size > 256 bytes:
    Fixed-width: StreamVByte (4 values/control byte, LE truncated)
    Strings:     offsets + LZ4-compressed chars
  Nullable:      null_map (bit vector) + inner column data
```

---

## Plan Translation Pipeline

```
  Doris FE TPlan (Thrift)
  │
  ├─ TPlanNode tree:
  │   FILE_SCAN_NODE, AGG_NODE, SORT_NODE, JOIN_NODE,
  │   EXCHANGE_NODE, UNION_NODE, EMPTY_SET_NODE, etc.
  │
  ├─ TExpr tree (per-node conjuncts, output expressions):
  │   SLOT_REF, INT_LITERAL, STRING_LITERAL, BINARY_PRED,
  │   FUNCTION_CALL, CAST_EXPR, CASE_EXPR, IN_PRED, etc.
  │
  ▼
  ┌────────────────────────────────────┐
  │       plan-translator crate        │
  │                                    │
  │  translate_fragment()              │
  │  ├─ node_translator: TPlanNode    │
  │  │   → ReadRel, FilterRel,        │
  │  │     AggregateRel, SortRel,     │
  │  │     JoinRel, CrossRel,         │
  │  │     FetchRel, SetRel           │
  │  │                                │
  │  ├─ scan_translator:              │
  │  │   FILE_SCAN_NODE               │
  │  │   → ReadRel::LocalFiles        │
  │  │     (parquet_scan([files]))     │
  │  │                                │
  │  ├─ expr_translator: TExpr        │
  │  │   → Substrait Expression       │
  │  │                                │
  │  └─ type_mapper: Doris ↔ Substrait│
  │                                    │
  └────────────────┬───────────────────┘
                   │
                   ▼
  Substrait Plan (protobuf bytes)
  │
  ▼
  ┌──────────────────────────────────────┐
  │         sirius-ffi (DuckDB)          │
  │                                      │
  │  GPU path:                           │
  │    gpu_execution_substrait(plan)     │
  │    → Sirius C++ pipeline executor    │
  │    → GPU VRAM results                │
  │                                      │
  │  CPU path:                           │
  │    from_substrait(plan)              │
  │    → DuckDB native execution         │
  │    → Arrow IPC results               │
  │                                      │
  │  SQL path:                           │
  │    execute_sql(sql)                  │
  │    → DuckDB SQL engine               │
  │    → Arrow IPC results               │
  └──────────────────────────────────────┘
```

---

## What's Working

### Single-BE Query Execution
- **Local file scans**: `SELECT * FROM local("file_path"="...", "format"="parquet", "backend_id"="<id>")`
- **Parquet/CSV/JSON**: File format detection, schema extraction, data scanning
- **GPU execution**: Full GPU pipeline for parquet scans with analytical operators
- **CPU fallback**: Automatic fallback via DuckDB `from_substrait()` when GPU can't handle plan
- **Plan translation**: 109 regression tests covering all supported node/expression types
- **Result delivery**: Arrow Flight streaming to FE, MySQL text protocol formatting

### GPU Pipeline (Sirius C++ Engine)
- **Native GPU parquet reader**: `sirius_physical_parquet_scan` — reads parquet directly to GPU VRAM
- **Analytical operators**: HASH_JOIN, GROUPED_AGGREGATE, UNGROUPED_AGGREGATE, SORT, TOP_N, FILTER
- **TPC-H compatibility**: Q1, Q2, Q4, Q5, Q6, Q12, Q14 verified at SF1, SF10, SF100, SF1000
- **GPU buffer management**: RMM memory pools, configurable cache/processing sizes
- **Error recovery**: GPU exceptions caught as recoverable `runtime_error`, not FATAL `InternalException`
- **H2D copy**: Fixed buffer overflow in `host_parquet_representation_converters.cpp`

### Multi-BE Distributed Execution
- **Fragment distribution**: FE distributes fragments across multiple BEs correctly
- **bRPC exchange**: Inter-BE data exchange via baidu_std protocol + PBlock encoding
- **Exchange buffer**: Concurrent `DashMap` with `Notify` for async data arrival
- **Multi-fragment merging**: Leaf + intermediate fragments merged for single-BE optimization
- **Remote exchange detection**: `per_exch_num_senders` count determines local vs. remote
- **UNION ALL**: Verified correct results (e.g., 8 rows from 2 BEs, 4+4)
- **AGG merge over exchange**: `generate_exchange_agg_merge_sql()` for count→SUM, sum→SUM, etc.

### NIXL GPU-Direct Exchange
- **GPU buffer detection**: `get_last_gpu_result_buffers()` via LastGPUBuffers singleton
- **nixl agent management**: Agent creation, UCX backend, metadata exchange
- **CUDA driver API**: `cuMemAlloc/Free/cpy` via dlopen (no compile-time CUDA dependency)
- **End-to-end transfer**: BE1 GPU → UCX RDMA → BE2 GPU (verified in Docker, 2 BEs)
- **bRPC fallback**: Automatic when nixl unavailable or GPU buffers missing
- **Self-transfer skip**: Same-BE exchange detected, falls back to bRPC

### Local Development Setup
- **Pixi environments**: `default` (C++), `doris` (Rust), `doris-fe` (Java), `doris-nixl`
- **Local FE**: `pixi run -e doris-fe doris-fe-build && doris-fe-start`
- **Local BEs**: `pixi run -e doris sirius-be` / `sirius-be-2` (separate ports)
- **Docker deployment**: `docker-compose.yml` with CDI GPU passthrough
- **Self-registration**: BE auto-registers with FE via HTTP API on startup

---

## What's Not Working / Known Issues

### GPU Engine Limitations
- **DUMMY_SCAN hang**: `count(*) FROM parquet_scan(...)` without GROUP BY produces a degenerate `DUMMY_SCAN → RESULT_COLLECTOR` plan. The GPU engine hangs on `future.get()` because there's no scan operator to produce data. Falls back to CPU.
- **Column ordering in joins**: TPC-H Q3, Q10 — DuckDB optimizer reorders join columns but Substrait `Root.names` are applied positionally, causing column mismatch.
- **ORDER BY not preserved**: TPC-H Q13 — DuckDB's `from_substrait` doesn't always preserve `SortRel/FetchRel` sort order.
- **Multi-input SetRel**: 8-way UNION ALL fails (DuckDB limitation: max 2 inputs per SetRel).
- **DuckDB SetRel(UNION_ALL) broken**: Neither GPU engine nor DuckDB substrait extension handles SetRel correctly. Exchange-collecting fragments use SQL path instead.

### Exchange Limitations
- **Hash partitioned exchange NOT IMPLEMENTED**: Exchange sender broadcasts to ALL destinations. For `GROUP BY` with multi-BE, this causes doubled values (each `AGG_FINAL` receives all data instead of its hash partition). Only UNION ALL and single-destination exchanges work correctly.
- **FE distributes local() scans across all BEs**: With shared storage, this doubles data (each BE scans the same files). Need to handle `backend_id` filtering or partition-aware scanning.

### NIXL / GPU-Direct Issues
- **RMM sub-allocation detection**: UCX may not recognize RMM pool sub-allocations as GPU memory (warns "memory is detected as host"). Dynamic detection via stderr capture falls back to bRPC.
- **`cudaMemcpyBatchAsync` disabled**: CUDA 13.0 batch API fails in Docker CDI. Individual `cudaMemcpyAsync` used instead.
- **--nixl-only testing incomplete**: Testing blocked by RESULT_COLLECTOR port crash (now fixed) and substrait extension version mismatch.

### Build / Version Issues
- **Extension version pinning**: Sirius `.duckdb_extension` must be built for exact same DuckDB version as the Rust `duckdb` crate (currently v1.4.4).
- **Substrait extension**: Must also match the DuckDB version — separate build from `duckdb-substrait-extension/`.
- **Two tonic versions**: `arrow-flight = "54"` pulls in `tonic = "0.12"`, while `doris-rpc` uses `tonic = "0.13"`. Both coexist but add compile time.

---

## Next Steps

### Near-Term
1. **Substrait extension update**: Rebuild `substrait.duckdb_extension` for DuckDB v1.4.4 (currently still v1.4.3).
2. **Test nixl GPU-direct end-to-end locally**: With RESULT_COLLECTOR fix merged, re-test `--nixl-only` with 2 local BEs.
3. **DUMMY_SCAN workaround**: Detect degenerate plans (no scan operator) and route to CPU immediately instead of attempting GPU execution.
4. **TPC-H Q3/Q10 column ordering**: Fix Substrait Root.names to match DuckDB's reordered output columns.

### Medium-Term
5. **Hash partitioned exchange**: Implement hash-based routing in exchange sender so each destination only receives its partition. Required for correct multi-BE `GROUP BY` results.
6. **Partition-aware scanning**: Handle `backend_id` in `local()` TVF so each BE only scans its assigned files, preventing data duplication.
7. **GPU exchange table registration**: Instead of GPU→CPU→table→CPU execution for exchange receivers, register GPU buffers directly as DuckDB tables (skip CPU copy for GPU-to-GPU path).

### Long-Term
8. **Multi-GPU support**: Distribute pipeline stages across multiple GPUs on the same node.
9. **Persistent table support**: Beyond `local()` TVF — support Doris managed tables with GPU-accelerated storage.
10. **Query result caching**: Cache GPU-resident intermediate results for repeated sub-queries.
11. **Streaming execution**: Large result sets that exceed GPU memory — spill to host or stream.

---

## Running Locally

### Prerequisites
- [Pixi](https://pixi.sh/) package manager
- NVIDIA GPU with CUDA support
- TPC-H data in parquet format (e.g., `/data/tpch/sf1/snappy/`)

### Build & Start

```bash
# Build Doris FE (first time only)
pixi run -e doris-fe doris-fe-build

# Build Rust workspace
pixi run -e doris doris-build

# Build C++ GPU engine (release)
pixi run make release

# Terminal 1: Start FE
pixi run -e doris-fe doris-fe-start

# Terminal 2: Start GPU BE 1
pixi run -e doris sirius-be

# Terminal 3: (optional) Start GPU BE 2
pixi run -e doris sirius-be-2

# Register BE with FE (if self-registration fails)
pixi run -e doris-fe doris-fe-add-sirius
```

### Test Queries

```sql
-- Connect
mysql -h 127.0.0.1 -P 9030 -u root

-- Simple test
SELECT 1;

-- File scan (replace <backend_id> with actual ID from SHOW BACKENDS)
SELECT * FROM local(
  "file_path"="/data/tpch/sf1/snappy/lineitem.parquet",
  "format"="parquet",
  "backend_id"="<id>"
) LIMIT 10;

-- TPC-H Q1 (GPU-accelerated)
SELECT l_returnflag, l_linestatus,
       SUM(l_quantity), SUM(l_extendedprice),
       SUM(l_extendedprice * (1 - l_discount)),
       COUNT(*)
FROM local("file_path"="/data/tpch/sf1/snappy/lineitem/*.parquet",
           "format"="parquet", "backend_id"="<id>")
WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;

-- Multi-BE UNION ALL (requires 2 BEs)
SELECT * FROM local("file_path"="...", "format"="parquet", "backend_id"="<be1>")
UNION ALL
SELECT * FROM local("file_path"="...", "format"="parquet", "backend_id"="<be2>");
```

### Configuration

**GPU Memory** (`~/.sirius/sirius.cfg`):
```
gpu_memory : {
    usage_limit_fraction = 0.2;
    reservation_limit_fraction = 0.2;
    downgrade_trigger_fraction = 0.15;
    downgrade_stop_fraction = 0.1;
};
```

**Doris FE** (`fe.conf`):
```
enable_outfile_to_local = true
enable_access_file_without_broker = true
priority_networks = 127.0.0.0/8
```

**BE CLI flags**:
```
--force-cpu          # Skip GPU, use DuckDB CPU
--no-cpu-fallback    # Error instead of CPU fallback
--nixl-only          # Require GPU-direct exchange (error if bRPC needed)
--gpu-cache-size     # GPU cache buffer size (default: 2GB)
--gpu-processing-size # GPU processing buffer size (default: 2GB)
```
