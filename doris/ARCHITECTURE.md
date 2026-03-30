# Sirius-Doris Architecture Diagrams

Contributor-oriented diagrams for the Sirius GPU Backend for Apache Doris.
See also [OVERVIEW.md](OVERVIEW.md) for status, build instructions, and test queries.

---

## 1. System Overview

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                          MySQL Client                                    │
 │                    mysql -h 127.0.0.1 -P 9030 -u root                   │
 └───────────────────────────────┬──────────────────────────────────────────┘
                                 │ MySQL wire protocol
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                       Apache Doris FE (Java)                             │
 │                                                                          │
 │   SQL ──► Parser ──► Analyzer ──► Planner ──► Fragment Distributor       │
 │                                                                          │
 │   The FE produces a distributed physical plan as "fragments".            │
 │   Each fragment is a tree of TPlanNodes (Thrift) that the FE             │
 │   assigns to one or more BEs based on data locality.                     │
 │                                                                          │
 │   Ports: 9030 (MySQL), 8030 (HTTP), 9020 (Thrift RPC)                   │
 └────────┬────────────────────────────────────────┬────────────────────────┘
          │                                        │
          │  gRPC: exec_plan_fragment              │  Arrow Flight: fetch results
          │  Thrift: heartbeat, backend_service    │
          │                                        │
          ▼                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                    Sirius GPU BE (Rust binary)                           │
 │                                                                          │
 │   This is our code. It replaces the standard Java/C++ Doris BE           │
 │   with a Rust service that wraps the Sirius C++ GPU engine.              │
 │                                                                          │
 │   ┌─────────────────────────────────────────────────────────────────┐    │
 │   │                      Service Layer                              │    │
 │   │                                                                 │    │
 │   │  HeartbeatService     PBackendService      Arrow Flight         │    │
 │   │  (Thrift, blocking)   (gRPC, async)        (tonic, async)       │    │
 │   │  port 9050            port 8060            port 8071            │    │
 │   │  ─────────────────    ──────────────────   ──────────────────   │    │
 │   │  FE health checks     Fragment execution   Result streaming     │    │
 │   │  BE registration       File schema/glob     to FE (Arrow IPC)   │    │
 │   │  Status reporting      Plan execution                           │    │
 │   └─────────────────────────────────────────────────────────────────┘    │
 │                                                                          │
 │   ┌─────────────────────────────────────────────────────────────────┐    │
 │   │                    Inter-BE Communication                       │    │
 │   │                                                                 │    │
 │   │  bRPC Server           bRPC Client          NixlMetadataService │    │
 │   │  (TCP, async)          (TCP, async)         (gRPC, async)       │    │
 │   │  port = brpc_port      connects to          port = grpc_port    │    │
 │   │  ─────────────────     other BEs            ──────────────────  │    │
 │   │  Receives PBlocks      Sends PBlocks        GPU-direct metadata │    │
 │   │  from other BEs        to other BEs         exchange (nixl)     │    │
 │   └─────────────────────────────────────────────────────────────────┘    │
 │                                                                          │
 │   ┌─────────────────────────────────────────────────────────────────┐    │
 │   │                    Execution Engine                              │    │
 │   │                                                                 │    │
 │   │  SiriusEngine (Arc<Mutex<>>)                                    │    │
 │   │  ├── DuckDB connection + loaded extensions                      │    │
 │   │  ├── GPU path: gpu_execution_substrait(plan_bytes)              │    │
 │   │  ├── CPU path: from_substrait(plan_bytes)                       │    │
 │   │  └── SQL path: execute_sql(sql_string)                          │    │
 │   └─────────────────────────────────────────────────────────────────┘    │
 │                                                                          │
 │   ┌─────────────────────────────────────────────────────────────────┐    │
 │   │                    Shared State (concurrent)                     │    │
 │   │                                                                 │    │
 │   │  ResultStore         ExchangeBuffer        NixlExchange         │    │
 │   │  DashMap<FinstId,    DashMap<ExchKey,       nixl::Agent          │    │
 │   │    ResultEntry>        ExchEntry>           + staging bufs       │    │
 │   │  ──────────────────  ─────────────────     ──────────────────   │    │
 │   │  Stores Arrow IPC    Buffers incoming      GPU-direct transfer   │    │
 │   │  results until FE    PBlocks from other    via UCX/RDMA          │    │
 │   │  calls fetch_data    BEs; notifies when    (optional feature)    │    │
 │   │                      all senders done                            │    │
 │   └─────────────────────────────────────────────────────────────────┘    │
 └──────────────────────────────────────────────────────────────────────────┘
          │
          │ FFI (C linkage)
          ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                    Sirius C++ GPU Engine                                  │
 │                    (loaded as DuckDB extension)                           │
 │                                                                          │
 │   CALL gpu_execution('SELECT ...')   or   gpu_execution_substrait(plan)  │
 │                                                                          │
 │   Pipeline Executor ──► GPU Operators (cuDF/CUDA):                       │
 │     PARQUET_SCAN, FILTER, HASH_JOIN, GROUPED_AGGREGATE,                  │
 │     SORT, TOP_N, RESULT_COLLECTOR                                        │
 │                                                                          │
 │   GPU Buffer Manager (RMM memory pools)                                  │
 │   cuCascade (tiered memory: GPU → Host → Disk)                           │
 └──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Crate Dependency Graph

```
                        sirius-doris-be
                        (main binary)
                       /       |       \
                      /        |        \
                     ▼         ▼         ▼
               doris-rpc   sirius-ffi   result-formatter
               (~7k LOC)   (DuckDB FFI) (ResultStore +
              /    |    \               Arrow Flight)
             /     |     \
            ▼      ▼      ▼
  plan-translator  doris-thrift  doris-proto
  (TPlan→Substrait) (Thrift IDL)  (Proto IDL)
  109 regression     build.rs:     build.rs:
  tests              thrift --gen   tonic-build
                     rs             prost-build


  Key external dependencies per crate:

  sirius-doris-be:  clap (CLI), mysql_async (FE registration), tokio
  doris-rpc:        tonic 0.13 (gRPC), dashmap, lz4_flex, cudarc
  sirius-ffi:       duckdb 1.4 (bundled), arrow 56
  result-formatter: arrow-flight 54 (tonic 0.12), dashmap
  plan-translator:  substrait 0.52
  doris-thrift:     thrift 0.17
  doris-proto:      prost 0.13, tonic 0.13
```

---

## 3. Query Execution: Single-BE Flow

Shows the happy path for a single-BE query like:
`SELECT * FROM local("file_path"="...", "format"="parquet", "backend_id"="10000")`

```
  FE                              GPU BE                          C++ Engine
  │                                │                                │
  │ ─── exec_plan_fragment ──────► │                                │
  │     (gRPC, VERSION_3)         │                                │
  │     Contains:                  │                                │
  │     - TPipelineFragmentParams  │                                │
  │     - desc_tbl (schema)        │                                │
  │     - query_globals            │                                │
  │     - fragment (TPlan tree)    │                                │
  │                                │                                │
  │ ◄── PStatus { OK } ────────── │  (returns immediately)         │
  │                                │                                │
  │                                ├─ Deserialize Thrift            │
  │                                │  (TBinaryProtocol)             │
  │                                │                                │
  │                                ├─ Classify fragment:            │
  │                                │  "leaf" (has FILE_SCAN_NODE)   │
  │                                │                                │
  │                                ├─ Extract file info:            │
  │                                │  path, format from scan ranges │
  │                                │                                │
  │                                ├─ Register parquet view:        │
  │                                │  "CREATE VIEW t AS             │
  │                                │   SELECT * FROM                │
  │                                │   parquet_scan([files])"       │
  │                                │                                │
  │                                ├─ Translate TPlan → Substrait   │
  │                                │  (plan-translator crate)       │
  │                                │                                │
  │                                ├─ Execute on GPU ──────────────►│
  │                                │  gpu_execution_substrait()     │
  │                                │                          ┌─────┤
  │                                │                          │ GPU │
  │                                │                          │ ops │
  │                                │                          └─────┤
  │                                │◄── Arrow IPC bytes ───────────│
  │                                │    (or GPU buffer pointers)    │
  │                                │                                │
  │                                ├─ Store in ResultStore          │
  │                                │  key = query_id (not frag id!) │
  │                                │                                │
  │ ─── GetFlightInfo ───────────► │                                │
  │     (Arrow Flight, port 8071)  │                                │
  │ ◄── schema + ticket ────────── │                                │
  │                                │                                │
  │ ─── DoGet(ticket) ───────────► │                                │
  │ ◄── Arrow IPC stream ───────── │  (streams record batches)      │
  │                                │                                │
  │  FE converts to MySQL rows     │                                │
  │  and sends to client           │                                │
```

---

## 4. Query Execution: Multi-BE Exchange Flow

Shows a distributed query (e.g., `GROUP BY` across 2 BEs with partitioned data).
The FE creates 3 fragments: 2 leaf scans + 1 aggregation receiver.

```
            FE distributes fragments to 2 BEs:
            ┌───────────────────────────────────────────────────────────┐
            │  Fragment 0 (leaf):   FILE_SCAN → AGG_LOCAL → EXCHANGE   │──► BE 1
            │  Fragment 1 (leaf):   FILE_SCAN → AGG_LOCAL → EXCHANGE   │──► BE 2
            │  Fragment 2 (root):   EXCHANGE → AGG_FINAL → SORT        │──► BE 1
            └───────────────────────────────────────────────────────────┘


  BE 1 (leaf + receiver)                              BE 2 (leaf only)
  ═══════════════════════                             ═══════════════════
  │                                                   │
  │ Receives frag 0 + frag 2                          │ Receives frag 1
  │                                                   │
  │ ┌─ Fragment 0 (leaf) ──────┐                      │ ┌─ Fragment 1 (leaf) ──────┐
  │ │ Scan local parquet files │                      │ │ Scan local parquet files │
  │ │ GPU: scan → agg_local    │                      │ │ GPU: scan → agg_local    │
  │ │ Result: partial agg data │                      │ │ Result: partial agg data │
  │ └──────────┬───────────────┘                      │ └──────────┬───────────────┘
  │            │                                      │            │
  │            │ EXCHANGE sink                        │            │ EXCHANGE sink
  │            │                                      │            │
  │            ▼                                      │            ▼
  │  ┌─────────────────────┐                          │  ┌─────────────────────┐
  │  │ Route exchange data │                          │  │ Route exchange data │
  │  │                     │                          │  │                     │
  │  │ Hash-partition rows │                          │  │ Hash-partition rows │
  │  │ by GROUP BY columns │                          │  │ by GROUP BY columns │
  │  │ (CRC32 hash)        │                          │  │ (CRC32 hash)        │
  │  │                     │                          │  │                     │
  │  │ Dest 0 → local      │                          │  │ Dest 0 → to BE 1   │
  │  │ Dest 1 → to BE 2    │                          │  │ Dest 1 → local      │
  │  └────┬────────┬───────┘                          │  └────┬────────┬───────┘
  │       │        │                                  │       │        │
  │       │        │ bRPC (PBlock)                    │       │        │
  │       │        └──────────────────────────────────│───────│────────┘
  │       │                    bRPC (PBlock)          │       │
  │       │  ◄────────────────────────────────────────│───────┘
  │       ▼                                           │
  │  ┌─────────────────────────┐                      │
  │  │ Fragment 2 (receiver)   │                      │
  │  │                         │                      │
  │  │ ExchangeBuffer collects │                      │
  │  │ PBlocks from both BEs   │                      │
  │  │                         │                      │
  │  │ When all senders EOS:   │                      │
  │  │ ├─ Decode PBlocks       │                      │
  │  │ ├─ Register as table    │                      │
  │  │ ├─ AGG_FINAL merge:     │                      │
  │  │ │  count→SUM, sum→SUM   │                      │
  │  │ │  min→MIN, max→MAX     │                      │
  │  │ ├─ Execute merge SQL    │                      │
  │  │ └─ Store in ResultStore │                      │
  │  └─────────────────────────┘                      │
  │            │                                      │
  │            ▼                                      │
  │  Arrow Flight → FE → Client                      │
```

---

## 5. bRPC Wire Protocol (Inter-BE)

Used for CPU-path exchange data between BEs. This is the Baidu-standard
RPC format that Doris uses internally (NOT standard gRPC/protobuf framing).

```
  Sender BE                                             Receiver BE
  ──────────                                            ───────────
  │                                                     │
  │  Arrow record batch                                 │
  │  ├─ Encode to PBlock (protobuf):                    │
  │  │  ├─ num_rows: u32                                │
  │  │  ├─ per column:                                  │
  │  │  │  ├─ null_map (if nullable)                    │
  │  │  │  ├─ fixed data (int/float/date)               │
  │  │  │  └─ string data (offsets + chars)             │
  │  │  │                                               │
  │  │  ├─ Compression (if col > 256 bytes):            │
  │  │  │  ├─ Fixed cols:  StreamVByte encoding          │
  │  │  │  ├─ Null maps:   StreamVByte encoding          │
  │  │  │  └─ String chars: LZ4 block compression       │
  │  │  │                                               │
  │  │  └─ Wrap in PTransmitDataParams proto            │
  │  │     ├─ finst_id (sender fragment instance)       │
  │  │     ├─ node_id (dest EXCHANGE_NODE id)           │
  │  │     ├─ sender_id (unique per sender)             │
  │  │     ├─ eos (true on last block)                  │
  │  │     └─ block (the PBlock bytes)                  │
  │  │                                                  │
  │  └─ Frame as baidu_std:                             │
  │                                                     │
  │     ┌────────┬───────────┬───────────┬──────────┐   │
  │     │ "PRPC" │ body_size │ meta_size │  body    │   │
  │     │ 4 byte │  u32 BE   │  u32 BE   │          │   │
  │     │ magic  │           │           │          │   │
  │     └────────┴───────────┴───────────┼──────────┤   │
  │                                      │ RpcMeta  │   │
  │                                      │ (proto)  │   │
  │                                      ├──────────┤   │
  │                                      │ Payload  │   │
  │                                      │ (proto)  │   │
  │                                      └──────────┘   │
  │                                                     │
  │ ──── TCP connect to receiver:brpc_port ──────────►  │
  │ ──── send frame ─────────────────────────────────►  │
  │                                                     │
  │                                      ┌──────────────┤
  │                                      │ Parse frame  │
  │                                      │ Decode meta  │
  │                                      │ Decode params│
  │                                      │ Extract block│
  │                                      │              │
  │                                      │ Store in     │
  │                                      │ ExchangeBuffer
  │                                      │ key=(query_id,│
  │                                      │     node_id) │
  │                                      │              │
  │                                      │ If eos:      │
  │                                      │ track sender │
  │                                      │ When all EOS:│
  │                                      │ notify_one() │
  │                                      └──────────────┤
  │                                                     │
  │ ──── send EOS frame (eos=true) ──────────────────►  │
  │                                                     │
```

---

## 6. NIXL GPU-Direct Exchange

When GPU results are available and nixl is enabled, data transfers happen
directly between GPU memory on different BEs via UCX/RDMA, bypassing CPU
serialization entirely.

```
  Sender BE (GPU result in VRAM)                Receiver BE
  ══════════════════════════════                ════════════
  │                                             │
  │ detect_execution_location()                 │
  │ → ExecutionLocation::Gpu {                  │
  │     buffers, column_info,                   │
  │     packed_metadata                         │
  │   }                                         │
  │                                             │
  │                                             │
  │  Step 1: Exchange metadata                  │
  │ ──────────────────────────                  │
  │                                             │
  │   ExchangeMetadata gRPC ──────────────────► │
  │   {                                         │ Allocate dst GPU buffers:
  │     query_id,                               │ ├─ Try pre-registered
  │     sender_id,                              │ │  staging buffer first
  │     columns: [{                             │ ├─ Fallback: cuMemAlloc
  │       gpu_ptr, size,                        │ │  + nixl register
  │       data_type                             │ │
  │     }]                                      │ Store pending state
  │   }                                         │
  │                                             │
  │   ◄──────────── ExchangeMetadataResponse ── │
  │   {                                         │
  │     dst_columns: [{                         │
  │       gpu_ptr, size                         │
  │     }]                                      │
  │   }                                         │
  │                                             │
  │                                             │
  │  Step 2: GPU-direct transfer                │
  │ ────────────────────────────                │
  │                                             │
  │   Create nixl transfer request:             │
  │   src_descs = sender GPU addrs              │
  │   dst_descs = receiver GPU addrs            │
  │                                             │
  │   nixl_agent.post_xfer_req()                │
  │   │                                         │
  │   │    ╔═══════════════════════════╗        │
  │   │    ║  UCX / RDMA transport     ║        │
  │   │    ║                           ║        │
  │   │    ║  GPU VRAM ─────► GPU VRAM ║        │
  │   │    ║  (zero CPU copy)          ║        │
  │   │    ╚═══════════════════════════╝        │
  │   │                                         │
  │   └── poll xfer_status until DONE           │
  │                                             │
  │                                             │
  │  Step 3: Signal completion                  │
  │ ──────────────────────────                  │
  │                                             │
  │   TransferComplete gRPC ──────────────────► │
  │   {                                         │ Register GPU table in DuckDB:
  │     query_id,                               │ ├─ gpu_register_table(
  │     packed_metadata (if any)                │ │    addrs, metadata)
  │   }                                         │ ├─ Deregister from nixl
  │                                             │ └─ Notify exchange task
  │                                             │
  │                                             │
  │                                             │ Exchange task resumes:
  │                                             │ ├─ GPU table available
  │                                             │ ├─ Execute remaining
  │                                             │ │  pipeline on GPU
  │                                             │ └─ Store final result
  │                                             │
  │                                             │
  │  Fallback triggers:                         │
  │  ────────────────────                       │
  │  • nixl agent unavailable → bRPC            │
  │  • UCX doesn't detect GPU mem → bRPC        │
  │  • same-BE transfer → bRPC (nixl rejects)   │
  │  • exchange_metadata errors → bRPC          │
```

---

## 7. Plan Translation Pipeline

Shows how Doris physical plan nodes map to Substrait relations,
and the SQL fallback for unsupported patterns.

```
  Doris TPlan (Thrift)              plan-translator              Substrait / SQL
  ════════════════════              ═══════════════              ═══════════════

  TPlanNode tree                    translate_fragment()
  │                                 │
  ├─ FILE_SCAN_NODE ───────────────►├─ ReadRel(LocalFiles)
  │  conjuncts: [TExpr]            │  (parquet_scan with filter pushdown)
  │  scan_ranges: [files]          │
  │                                │
  ├─ HASH_JOIN_NODE ───────────────►├─ JoinRel
  │  join_type, eq_conjuncts       │  (INNER, LEFT, RIGHT, OUTER, CROSS)
  │  other_conjuncts               │
  │                                │
  ├─ AGGREGATION_NODE ─────────────►├─ AggregateRel
  │  agg_functions, group_by       │  (groupings + measures)
  │                                │
  ├─ SORT_NODE ────────────────────►├─ SortRel or FetchRel
  │  sort_info, offset, limit      │  (ORDER BY + LIMIT/OFFSET)
  │                                │
  ├─ EXCHANGE_NODE ────────────────►├─ (handled in fragment routing,
  │  (0 children = receiver)       │   not translated to Substrait)
  │                                │
  ├─ UNION_NODE ───────────────────►├─ SetRel(UNION_ALL) ← broken in DuckDB
  │                                │   Falls back to SQL generator
  │                                │
  └─ EMPTY_SET_NODE ───────────────►└─ VirtualTable (empty)


  TExpr tree                        expr_translator              Substrait Expr
  ──────────                        ───────────────              ──────────────

  ├─ SLOT_REF ─────────────────────►  FieldReference(index)
  ├─ INT_LITERAL ──────────────────►  Literal(i32/i64)
  ├─ STRING_LITERAL ───────────────►  Literal(string)
  ├─ FLOAT_LITERAL ────────────────►  Literal(fp32/fp64)
  ├─ DECIMAL_LITERAL ──────────────►  Literal(decimal)
  ├─ DATE_LITERAL ─────────────────►  Literal(date)
  ├─ BOOL_LITERAL ─────────────────►  Literal(bool)
  ├─ NULL_LITERAL ─────────────────►  Literal(null, typed)
  ├─ BINARY_PRED ──────────────────►  ScalarFunction(eq/lt/gt/...)
  ├─ COMPOUND_PRED ────────────────►  ScalarFunction(and/or/not)
  ├─ FUNCTION_CALL ────────────────►  ScalarFunction(name, args)
  ├─ CAST_EXPR ────────────────────►  Cast(input, target_type)
  ├─ CASE_EXPR ────────────────────►  IfThen(ifs, else)
  ├─ IN_PRED ──────────────────────►  SingularOrList
  ├─ IS_NULL_PRED ─────────────────►  ScalarFunction(is_null/is_not_null)
  └─ LIKE_PRED ────────────────────►  ScalarFunction(like)


  Three execution paths through sirius-ffi:

  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │   ① GPU (preferred):  gpu_execution_substrait(plan_bytes)           │
  │      → Sirius C++ pipeline executor → GPU VRAM results              │
  │      Used for: leaf fragments with supported operators              │
  │                                                                     │
  │   ② CPU Substrait:   from_substrait(plan_bytes)                     │
  │      → DuckDB native execution → Arrow IPC results                  │
  │      Used for: fallback when GPU rejects plan                       │
  │                                                                     │
  │   ③ SQL string:      execute_sql(sql_string)                        │
  │      → DuckDB SQL engine → Arrow IPC results                        │
  │      Used for: UNION_NODE, AGG merge, exchange collection           │
  │      (SQL generator in plan-translator handles these cases)         │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Thread Model

```
  Process startup (main.rs)
  │
  ├─ Main thread
  │  ├─ Parse CLI args (clap)
  │  ├─ Init SiriusEngine (DuckDB + GPU extensions)
  │  ├─ Init GPU buffers: gpu_buffer_init(cache_size, processing_size)
  │  ├─ Self-register with FE (MySQL INSERT into system table)
  │  │
  │  ├─ Spawn: Heartbeat thread (blocking, dedicated)
  │  │         └─ thrift::TServer on port 9050
  │  │            Runs in a loop, one connection at a time
  │  │            FE sends heartbeat every ~5s
  │  │
  │  ├─ Spawn: Backend service thread (blocking, dedicated)
  │  │         └─ thrift::TServer on port 9060
  │  │            Mostly stubbed for GPU BE
  │  │
  │  └─ Enter: Tokio async runtime (multi-threaded)
  │     │
  │     ├─ gRPC server (tonic) on port 8060
  │     │  └─ PBackendService handlers:
  │     │     ├─ exec_plan_fragment → spawns async task per fragment
  │     │     ├─ fetch_table_schema → DuckDB schema query
  │     │     └─ glob → local file listing
  │     │
  │     ├─ Arrow Flight server (tonic) on port 8071
  │     │  └─ GetSchema / DoGet handlers
  │     │     Read from ResultStore, stream Arrow IPC
  │     │
  │     ├─ bRPC listener (raw TCP) on brpc_port
  │     │  └─ Per-connection handler (async):
  │     │     Parse baidu_std frames → store PBlocks in ExchangeBuffer
  │     │
  │     ├─ NixlMetadataService (tonic, optional) on grpc_port
  │     │  └─ exchange_metadata / transfer_complete handlers
  │     │
  │     └─ Fragment execution tasks (spawned per fragment)
  │        └─ Each task:
  │           ├─ Deserialize Thrift (sync, in spawn_blocking)
  │           ├─ Translate plan (sync)
  │           ├─ Execute engine (sync, in spawn_blocking)
  │           │  └─ Holds Mutex<SiriusEngine> lock during execution
  │           ├─ Route result (async):
  │           │  ├─ Exchange: bRPC send or nixl transfer
  │           │  └─ Result: store in ResultStore
  │           └─ Await exchange data if needed (async)
  │              └─ ExchangeBuffer::wait_for() with Notify


  Important concurrency notes:
  ─────────────────────────────
  • SiriusEngine is Arc<Mutex<>> because DuckDB Connection is !Sync
  • Only ONE fragment executes on the engine at a time (mutex)
  • Fragment execution uses spawn_blocking to avoid starving tokio
  • ExchangeBuffer and ResultStore use DashMap (lock-free concurrent)
  • Notify::notify_one() stores a permit (safe for late waiters)
```

---

## 9. Result Delivery Path

```
  Fragment finishes execution
  │
  ├─ Get result:
  │  ├─ GPU path: Arrow IPC bytes from gpu_execution_substrait()
  │  │            (or GPU buffer pointers for nixl exchange)
  │  └─ CPU path: Arrow IPC bytes from from_substrait() / execute_sql()
  │
  ├─ Is this an EXCHANGE sink fragment?
  │  │
  │  ├─ YES → route to exchange (see diagrams 4-6)
  │  │
  │  └─ NO → this is the result-producing fragment
  │     │
  │     ├─ Parse Arrow IPC bytes into Schema + RecordBatches
  │     │
  │     ├─ Store in ResultStore:
  │     │  key:   query_id (hi: i64, lo: i64)   ◄── NOT fragment instance id!
  │     │  value: ResultEntry {
  │     │           schema: Schema,
  │     │           batches: Vec<RecordBatch>,
  │     │           notify: Notify              ◄── wakes up waiting FE fetch
  │     │         }
  │     │
  │     └─ notify.notify_one()                   ◄── FE may already be waiting
  │
  │
  FE fetches results (may arrive before or after execution completes):
  │
  ├─ GetFlightInfo (Arrow Flight, port 8071)
  │  ├─ ticket = FinstId = 16 bytes (hi LE 8 + lo LE 8)
  │  └─ Returns: FlightInfo with schema
  │
  └─ DoGet(ticket) (Arrow Flight, port 8071)
     ├─ Look up ResultStore by FinstId
     ├─ If not ready: await notify (ResultStore.wait_for())
     ├─ Stream schema as first message (IPC schema bytes)
     └─ Stream each RecordBatch as Arrow IPC
        │
        FE receives and converts to MySQL text protocol rows:
        └─ Each row = concatenated length-encoded column strings
           Wrapped in TResultBatch, serialized with TBinaryProtocol
```

---

## 10. Key File Map

Quick reference for finding code by concern:

```
  "Where do I find...?"

  Server startup & config     doris/crates/sirius-doris-be/src/main.rs
                              doris/crates/sirius-doris-be/src/config.rs

  Fragment execution          doris/crates/doris-rpc/src/grpc_service.rs
    (exec_plan_fragment)        → exec_plan_fragment()
                                → execute_fragment_async()

  Plan translation            doris/crates/plan-translator/src/lib.rs
    Doris TPlan→Substrait       → translate_fragment()
    Node mapping                → node_translator.rs
    Expression mapping          → expr_translator.rs
    SQL generation              → sql_generator.rs
    Type mapping                → type_mapper.rs

  DuckDB engine wrapper       doris/crates/sirius-ffi/src/lib.rs
    GPU/CPU/SQL execution       → SiriusEngine

  Exchange buffer              doris/crates/doris-rpc/src/exchange_buffer.rs
    Concurrent PBlock store     → ExchangeBuffer

  bRPC server (receiver)      doris/crates/doris-rpc/src/brpc_server.rs
    baidu_std frame parser      → handle_brpc_connection()

  bRPC client (sender)        doris/crates/doris-rpc/src/exchange_sender.rs
    PBlock transmission         → send_transmit_block()

  PBlock encode/decode        doris/crates/doris-rpc/src/arrow_to_pblock.rs
                              doris/crates/doris-rpc/src/pblock_decoder.rs

  Hash partitioning           doris/crates/doris-rpc/src/hash_partitioner.rs
    Row routing for GROUP BY    → HashPartitioner

  NIXL GPU-direct             doris/crates/doris-rpc/src/nixl_exchange.rs
    Agent + staging buffers     → NixlExchange
  NIXL integration            doris/crates/doris-rpc/src/nixl_integration.rs
    ExecutionLocation detect    → detect_execution_location()
  NIXL metadata service       doris/crates/doris-rpc/src/nixl_service.rs
    gRPC handlers               → NixlMetadataService

  Result storage              doris/crates/result-formatter/src/result_store.rs
                              → ResultStore, FinstId

  Arrow Flight server         doris/crates/result-formatter/src/arrow_flight.rs
    Result streaming to FE      → SiriusFlightService

  Heartbeat service           doris/crates/doris-rpc/src/heartbeat_service.rs
  Backend service (stubs)     doris/crates/doris-rpc/src/backend_service.rs

  Thrift codegen              doris/crates/doris-thrift/build.rs
  Proto codegen               doris/crates/doris-proto/build.rs

  Docker compose              doris/docker/docker-compose.yml
  Cluster start script        doris/scripts/start-cluster.sh
  TPC-H test script           doris/scripts/run-tpch.sh
```
