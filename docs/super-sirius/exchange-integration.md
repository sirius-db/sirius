# Exchange Integration

This document proposes moving the distributed exchange path from the Rust backend into Sirius Core as a first-class executor, enabling overlap between GPU compute and network communication.

> **Status:** Design proposal — not yet implemented.

## Motivation

The current exchange architecture has three fundamental problems:

### 1. No Overlap Between Compute and Communication

After each fragment's GPU execution completes, there is a **blocking boundary** before exchange begins. The C++ engine finishes, hands packed data to Rust via C API (`sirius_exchange_c_api.cpp`), and Rust handles the entire NIXL transfer lifecycle. The GPU sits idle during transfers, and transfers wait for compute to finish.

```
TIME ═══> [  GPU compute  ][ BLOCKING ][ NIXL transfer ][ BLOCKING ][ GPU compute ]
```

### 2. Partitioning Happens Outside the Execution Pipeline

Hash partitioning (`cudf::hash_partition`) and packing (`cudf::pack`) currently happen inside `sirius_physical_result_collector.cpp` (lines 282-490) as a post-processing step during result collection. This is not a streaming pipeline operator — it runs after the GPU pipeline completes and cannot overlap with upstream computation. Partitioning is tightly coupled to the result collector rather than being a composable operator in the pipeline graph.

### 3. Boundary Crossing and Duplicated Systems

The data path crosses four boundaries per transfer:

```
C++ engine → C API → Rust staging (bump allocator) → NIXL → Rust gRPC → remote Rust → C API → C++ engine
```

The Rust bump allocator (`gpu_staging_buffer.rs`) manages staging memory independently of cuCascade's reservation system. This means:
- Staging memory is invisible to the GPU memory pressure manager
- The downgrade executor cannot reclaim staging buffers under memory pressure
- The task creator cannot schedule downstream work until Rust completes the transfer

### Goal

Make partitioning and communication first-class pipeline operators that overlap with GPU compute, using cuCascade-managed memory throughout.

```
TIME ═══> [ GPU compute+partition batch 1 ][ GPU compute+partition batch 2 ][ ... ]
                  [ staging + RDMA batch 1 ][ staging + RDMA batch 2 ]
                          ^^^ OVERLAPPED: compute, partition, transfer ^^^
```

## Architecture Overview

### Current Architecture (Blocking)

```
┌─────────────────────────────────────────────────────────────┐
│                     Sirius Core (C++)                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ GPU Pipeline  │───>│   Result     │───>│  C API       │  │
│  │ Executor      │    │  Collector   │    │  (blocking)  │──┼──┐
│  │               │    │ (partitions  │    │              │  │  │
│  │               │    │  + packs)    │    │              │  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │  │
└─────────────────────────────────────────────────────────────┘  │
                                                                 │
┌─────────────────────────────────────────────────────────────┐  │
│                  Sirius Backend (Rust)                       │  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │  │
│  │ Staging Bump  │<───│  NIXL        │<───│  gRPC        │<─┼──┘
│  │ Allocator     │───>│  Exchange    │───>│  Service     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Proposed Architecture (Overlapped)

```
┌───────────────────────────────────────────────────────────────────────┐
│                          Sirius Core (C++)                            │
│                                                                       │
│  GPU Pipeline Executor                                                │
│  ┌────────────────────────────────────────────┐                       │
│  │ operators... → exchange_partition (sink)    │                       │
│  │   (hash_partition + pack per partition)     │                       │
│  └─────────────────────┬──────────────────────┘                       │
│                        │ publish to                                    │
│                        v                                              │
│              ┌────────────────────┐                                    │
│              │  shared_data_repo   │  (per-partition exchange repos)   │
│              └────────┬───────────┘                                    │
│                       │ task_creator watches repos,                    │
│                       │ creates communication_task per partition       │
│                       v                                               │
│  Communication Executor                                               │
│  ┌────────────────────────────────────────────┐                       │
│  │ Manager Loop         Worker Pool            │                       │
│  │ ┌─────────────┐     ┌────────────────────┐ │                       │
│  │ │ reserve slot │────>│ 1. pack to staging │ │                       │
│  │ │ make reserv. │     │ 2. register (NIXL) │ │  C++ direct           │
│  │ │ dispatch     │     │ 3. send metadata ──┼─┼──── FFI ──> Rust     │
│  │ └─────────────┘     │ 4. RDMA transfer   │ │  C++ direct (NIXL)   │
│  │                      │ 5. send complete ──┼─┼──── FFI ──> Rust     │
│  │                      └────────────────────┘ │                       │
│  └────────────────────────────────────────────┘                       │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────────────┐
│                      Sirius Backend (Rust)                            │
│  gRPC Client (tonic)           gRPC Server (tonic)                    │
│  ┌──────────────────┐         ┌──────────────────────────────┐        │
│  │ ExchangeMetadata  │         │ ExchangeMetadata handler     │        │
│  │ TransferComplete  │         │   → FFI into C++:            │        │
│  │ (called by C++    │         │     reserve_recv_buffer()    │        │
│  │  worker threads)  │         │ TransferComplete handler     │        │
│  └──────────────────┘         │   → FFI into C++:            │        │
│                                │     ingest_transfer()        │        │
│                                └──────────────────────────────┘        │
└───────────────────────────────────────────────────────────────────────┘
```

### Component Ownership

| Component | Location | Notes |
|-----------|----------|-------|
| `sirius_physical_exchange_partition` | Sirius Core (C++) | New pipeline operator for hash partitioning + pack |
| `communication_executor` | Sirius Core (C++) | New sender-side executor (manager loop + worker pool) |
| `communication_task` | Sirius Core (C++) | New per-partition transfer task |
| NIXL agent (direct C++ calls) | Sirius Core (C++) | New C++ wrapper — NIXL is a native C++ library |
| Staging buffer management | Sirius Core (C++) | cuCascade reservations replace Rust bump allocator |
| Receiver C API | Sirius Core (C++) | New `extern "C"` functions for reserve + ingest |
| Hash partitioning | Sirius Core (C++) | Moved from result_collector post-processing to pipeline operator |
| gRPC client (sender) | Sirius Backend (Rust) | Stays — tonic client, called from C++ via FFI |
| gRPC server (receiver) | Sirius Backend (Rust) | Stays — tonic server, handlers FFI into C++ |
| NIXL Rust bindings (`nixl_sys`) | Sirius Backend (Rust) | **Removed** — C++ calls NIXL directly |
| `nixl_exchange.rs` | Sirius Backend (Rust) | **Removed** — NIXL calls move to C++ |
| `nixl_integration.rs` | Sirius Backend (Rust) | **Refactored** — orchestration moves to C++; gRPC wrappers remain |
| `nixl_service.rs` | Sirius Backend (Rust) | **Simplified** — handlers become thin FFI shims into C++ |
| `gpu_staging_buffer.rs` | Sirius Backend (Rust) | **Removed** — cuCascade replaces Rust bump allocator |
| `sirius_exchange_c_api.cpp` | Sirius Core (C++) | **Replaced** — old capture API replaced by new receiver C API |
| Exchange code in `result_collector` | Sirius Core (C++) | **Removed** — partitioning moves to exchange_partition operator |
| Fragment execution | Sirius Backend (Rust) | Unchanged — still handles fragment dispatch, plan translation |
| bRPC exchange | Sirius Backend (Rust) | Unchanged — remains as CPU fallback path |

## Sender Side: `communication_executor`

### Class Design

**Files:** `src/include/pipeline/communication_executor.hpp`, `src/pipeline/communication_executor.cpp`

Extends `itask_executor` following the same pattern as `gpu_pipeline_executor`:

```cpp
class communication_executor : public sirius::parallel::itask_executor {
 public:
  explicit communication_executor(
    exec::thread_pool_config config,
    cucascade::memory::memory_space* staging_mem_space,
    sirius::parallel::downgrade_executor* downgrade_executor = nullptr);

  void set_task_creator(sirius::creator::task_creator* task_creator);
  void set_completion_handler(completion_handler* handler) noexcept;

 protected:
  void manager_loop() override;

 private:
  cucascade::memory::memory_space* _staging_mem_space;
  sirius::parallel::downgrade_executor* _downgrade_executor{nullptr};
  sirius::creator::task_creator* _task_creator{nullptr};
  completion_handler* _completion_handler{nullptr};
};
```

Key differences from `gpu_pipeline_executor`:
- No CUDA stream pool — worker threads do not run GPU kernels (packing uses the default stream or a dedicated pack stream)
- No `task_request_publisher` — communication tasks do not compete for GPU compute slots
- Uses a staging-dedicated `memory_space` (or a carved-out region of the GPU memory space)

### Manager Loop

The manager loop stays lightweight — it only reserves memory and dispatches work. This follows the same pattern as `gpu_pipeline_executor::manager_loop()` (see [Pipeline Execution](pipeline-execution.md)):

```
while (_running):
  1. slot = _bounded_pool->reserve()          // Block until worker slot available
  2. task = _task_queue.pop()                  // Block until communication task queued
  3. estimate staging_bytes from task input    // Size of cudf::chunked_pack output
  4. reservation = _staging_mem_space->make_reservation(staging_bytes)
  5. if partial and _downgrade_executor:       // Same pattern as GPU executor
       trigger downgrade, retry reservation
  6. attach reservation to task local state
  7. _bounded_pool->dispatch(slot, worker_fn)  // Hand off to worker thread
```

The manager does **not** pack data — packing happens in the worker thread so the manager stays responsive for the next task.

### Worker Thread Protocol

Each worker thread executes a 4-phase transfer lifecycle:

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1 — Pack (C++ direct)                             │
│   cudf::chunked_pack into staging buffer                │
│   Register staging buffer with NIXL agent               │
├─────────────────────────────────────────────────────────┤
│ Phase 2 — Metadata Exchange (FFI → Rust tonic)          │
│   Send ExchangeMetadata gRPC to receiver                │
│   Wait for response: dst buffer addresses + NIXL meta   │
├─────────────────────────────────────────────────────────┤
│ Phase 3 — RDMA Transfer (C++ direct — hot path)         │
│   Load receiver NIXL metadata (cached per peer)         │
│   create_xfer_req() + post_xfer_req()                   │
│   Poll get_xfer_status() until complete                 │
├─────────────────────────────────────────────────────────┤
│ Phase 4 — Completion (FFI → Rust tonic)                 │
│   Send TransferComplete gRPC to receiver                │
│   Release staging reservation + deregister NIXL buffer  │
└─────────────────────────────────────────────────────────┘
```

**Boundary crossing summary:** NIXL operations (phases 1, 3) are called directly from C++ — NIXL is a native C++ library (`doris/thirdparty/nixl/src/api/cpp/`). gRPC operations (phases 2, 4) cross into Rust via `extern "C"` FFI. The RDMA hot path stays in C++; only the metadata RPCs (small messages, two round-trips per transfer) cross the boundary.

### FFI Surface (C++ → Rust, sender side)

New `extern "C"` functions exported by the Rust `doris-rpc` crate for the sender:

```c
// Send ExchangeMetadata gRPC and block until response.
// Called by C++ worker thread in Phase 2.
int sirius_exchange_send_metadata(
    const char* dest_addr,              // Receiver gRPC endpoint
    const sirius_exchange_metadata* req, // Buffer descriptors + NIXL metadata
    sirius_exchange_metadata_response* resp); // Out: dst addresses + receiver NIXL meta

// Send TransferComplete gRPC notification.
// Called by C++ worker thread in Phase 4.
int sirius_exchange_send_complete(
    const char* dest_addr,
    const sirius_transfer_complete* req);
```

These functions block the calling C++ thread. Since each worker thread handles one transfer at a time and the worker pool is bounded, this is acceptable — the tonic runtime handles the async gRPC underneath.

### Retry Mechanism

Three retry triggers, following the OOM reschedule pattern from `gpu_pipeline_task.cpp` (lines 225-312):

| Trigger | Where | Behavior |
|---------|-------|----------|
| **Reservation failure** | Manager loop | Back off 5ms, trigger downgrade, retry. Max 10 retries per task. |
| **Receiver NACK** | Phase 2 (metadata) | Receiver returns `status_code != 0` (insufficient memory). Re-enqueue task with exponential backoff. |
| **RDMA failure** | Phase 3 (transfer) | NIXL transfer timeout or error. Retry with fresh metadata exchange (addresses may have changed). After N failures, fall back to bRPC CPU transfer. |

Retry tracking uses the same `retry_count` + `original_task_id` pattern as `gpu_pipeline_task_local_state`.

## Receiver Side

### gRPC Service (Rust)

The gRPC server stays in Rust as `NixlMetadataService` in `nixl_service.rs`. It continues to use tonic and the existing proto interface (`nixl_service.proto`). The handlers become thin FFI shims that delegate memory management and data ingestion to C++.

### ExchangeMetadata Handler

```
1. Rust receives ExchangeMetadata gRPC request
2. Parse buffer descriptors and sender NIXL metadata
3. FFI into C++: sirius_exchange_reserve_recv_buffer()
   ├── memory_space->make_reservation_or_null(total_recv_size)
   ├── If fail: return error code → Rust sends NACK response
   ├── Allocate receive buffer from reservation
   ├── Register receive buffer with NIXL agent (C++ API)
   └── Return dst buffer addresses through FFI
4. Rust loads sender NIXL metadata (cached per peer)
5. Return gRPC response with dst addresses + local NIXL metadata
```

### TransferComplete Handler

```
1. Rust receives TransferComplete gRPC notification
2. FFI into C++: sirius_exchange_ingest_transfer()
   ├── cudf::unpack() with packed metadata → cuDF table
   ├── Wrap as cucascade::data_batch
   ├── Push into shared_data_repository
   └── task_creator->schedule(output_consumer)
3. Return gRPC success response
```

### FFI Surface (Rust → C++, receiver side)

New `extern "C"` functions exposed by Sirius Core:

```c
// Called by Rust on ExchangeMetadata RPC.
// Reserves GPU memory and allocates a receive buffer.
int sirius_exchange_reserve_recv_buffer(
    uint64_t query_id_hi,
    uint64_t query_id_lo,
    int32_t node_id,
    size_t total_size,
    const uint8_t* packed_cudf_metadata,   // For cudf::unpack schema
    size_t packed_cudf_metadata_size,
    sirius_recv_buffer_info* out);         // Out: GPU addr, NIXL agent metadata

// Called by Rust on TransferComplete RPC.
// Unpacks transferred data and pushes to the data repository.
int sirius_exchange_ingest_transfer(
    uint64_t query_id_hi,
    uint64_t query_id_lo,
    int32_t node_id,
    int32_t sender_id,
    uint32_t num_rows,
    const uint8_t* packed_cudf_metadata,
    size_t packed_cudf_metadata_size);
```

## Exchange as Pipeline Operator

### `sirius_physical_exchange_partition`

**Files:** `src/include/op/sirius_physical_exchange_partition.hpp`, `src/op/sirius_physical_exchange_partition.cpp`

Currently, hash partitioning lives in `sirius_physical_result_collector.cpp` as post-processing (lines 282-490). In the new design, this becomes a dedicated pipeline operator:

- Extends `sirius_physical_operator`
- Registered in `sirius_physical_plan_generator` for exchange sink plans
- Acts as a pipeline sink (barrier) — similar to the existing `sirius_physical_partition` operator (`src/op/sirius_physical_partition.cpp`)
- During `execute()`: runs `cudf::hash_partition` on the input data
- During `sink()`: packs each partition via `cudf::pack` and publishes per-partition packed data to exchange data repositories
- Each partition's data repository feeds into a `communication_task`

This design means partitioning overlaps with upstream GPU compute — as batch N is being partitioned, batch N+1 can already be computing in the GPU pipeline executor.

### Communication Task

**Files:** `src/include/pipeline/communication_task.hpp`, `src/pipeline/communication_task.cpp`

```cpp
class communication_task : public sirius::parallel::sirius_pipeline_itask {
 public:
  void execute(rmm::cuda_stream_view stream) override;

 private:
  // Local state
  std::unique_ptr<cucascade::data_batch> _packed_data;  // From exchange_partition
  std::string _dest_addr;                                // Receiver gRPC endpoint
  int _partition_id;
  int _retry_count{0};
  uint64_t _original_task_id;
  std::unique_ptr<cucascade::memory::reservation> _staging_reservation;

  // Global state (shared across tasks for same query)
  nixl_agent* _nixl_agent;    // NIXL C++ agent handle
  // gRPC goes through Rust FFI, no C++ channel needed
};
```

### Task Creation

In `task_creator` (`src/creator/task_creator.cpp`):

- `task_creator` watches exchange data repositories for ready partitions
- When a partition's data is available, creates a `communication_task`
- One task per partition per batch (or per destination node)
- Routes the task to `communication_executor` instead of `gpu_pipeline_executor`

Data flow:
```
GPU pipeline task
  → operators... → exchange_partition (sink)
    → cudf::hash_partition + cudf::pack
    → publish per-partition data to exchange data repos
      → task_creator detects ready partition
        → creates communication_task
          → communication_executor handles transfer
```

## Integration Points

| Component | File | Change |
|-----------|------|--------|
| `pipeline_executor` | `src/pipeline/pipeline_executor.hpp` | Add `communication_executor` alongside `_gpu_executors` |
| `task_creator` | `src/creator/task_creator.cpp` | New branch: exchange data repo ready → create `communication_task` |
| `completion_handler` | `src/pipeline/completion_handler.hpp` | Track per-partition per-destination completion |
| `SiriusContext` | `src/sirius_context.cpp` | Initialize NIXL agent, register receiver FFI functions |
| `sirius_config` | `src/config.cpp` / `src/include/config.hpp` | New config: comm executor threads, staging limits, retry policy |
| `sirius_physical_plan_generator` | `src/planner/sirius_physical_plan_generator.cpp` | Register `exchange_partition` for exchange sink plans |

## Memory Architecture

### Staging via cuCascade

The Rust bump allocator (`gpu_staging_buffer.rs`) is replaced by per-transfer cuCascade reservations:

| Aspect | Current (Rust) | Proposed (C++) |
|--------|----------------|----------------|
| Allocation | Pre-allocated bump pool (1GB default) | On-demand reservation per transfer |
| Memory pressure | Invisible to cuCascade | Participates in downgrade/upgrade |
| Lifetime | Epoch-based reset when all leases drop | RAII reservation, released after transfer |
| Overflow | Fallback to individual `cuMemAlloc` | Downgrade executor frees GPU memory |

### Sender Reservation Flow

```
task_creator creates communication_task with packed partition data
  → communication_executor manager_loop
    → _staging_mem_space->make_reservation(staging_bytes)
    → if partial: trigger downgrade, retry
    → attach reservation to task local state
    → dispatch to worker thread
      → worker: allocate staging buffer from reservation
      → worker: cudf::chunked_pack into staging
      → worker: register with NIXL, do RDMA transfer
      → worker: release reservation after transfer complete
```

### Receiver Reservation Flow

```
ExchangeMetadata RPC arrives (Rust → C++ FFI)
  → memory_space->make_reservation_or_null(recv_size)
  → if fail: return error → Rust sends NACK → sender retries
  → allocate receive buffer from reservation
  → register with NIXL, return addresses
  → ... RDMA transfer happens (sender-initiated) ...
TransferComplete RPC arrives (Rust → C++ FFI)
  → cudf::unpack() → cuDF table → data_batch
  → push to shared_data_repository (reservation ownership transfers to batch)
  → task_creator->schedule(downstream operator)
```

## Configuration

New entries in `sirius.yaml`:

```yaml
communication:
  # Worker thread count for the communication executor
  thread_count: 4
  # Maximum staging reservation per transfer (bytes)
  max_staging_reservation: 1073741824  # 1GB
  # Retry limits
  max_reservation_retries: 10
  max_receiver_nack_retries: 5
  max_rdma_retries: 3
  # Backoff
  reservation_retry_backoff_ms: 5
  nack_retry_initial_backoff_ms: 10
  nack_retry_max_backoff_ms: 1000

nixl:
  # NIXL transport backend
  transport: ucx
  # gRPC port for the receiver metadata service
  receiver_grpc_port: 9099
```

## Migration Path

1. **Phase 1**: Implement `communication_executor`, `communication_task`, and `sirius_physical_exchange_partition` in C++. Implement NIXL C++ agent wrapper. Implement sender-side FFI functions (`sirius_exchange_send_metadata`, `sirius_exchange_send_complete`).

2. **Phase 2**: Implement receiver-side FFI functions (`sirius_exchange_reserve_recv_buffer`, `sirius_exchange_ingest_transfer`). Simplify `nixl_service.rs` handlers to thin FFI shims.

3. **Phase 3**: Wire into `pipeline_executor` and `task_creator`. Remove exchange code from `result_collector`. Register `exchange_partition` in the plan generator.

4. **Phase 4**: Integration testing with existing NIXL proto format (reuse `nixl_service.proto`, `nixl_exchange.proto`).

5. **Phase 5**: Remove deprecated Rust code (`nixl_exchange.rs`, `gpu_staging_buffer.rs`, NIXL Rust bindings). Replace `sirius_exchange_c_api.cpp` with new receiver C API.

## Open Questions

1. **Staging memory space**: Should staging use a dedicated cuCascade memory space (separate capacity/thresholds) or share the GPU compute space? A dedicated space prevents exchange from starving compute, but reduces total available GPU memory.

2. **NIXL agent lifecycle**: One per `SiriusContext` (shared across queries) or one per query? Per-context is simpler but requires thread-safe access. Per-query provides isolation but increases NIXL initialization overhead.

3. **STRING column offset corruption**: The known NIXL RDMA bug that corrupts STRING column offsets (see `.planning/codebase/CONCERNS.md`) needs to be addressed regardless of C++ vs Rust. The current workaround uses `cudf::pack` instead of `cudf::chunked_pack` for STRING columns — this workaround carries over to the C++ implementation.

4. **bRPC fallback**: Should the bRPC CPU fallback path also move to C++ eventually, or remain in Rust as a separate code path?

5. **Host-pinned receive buffers**: Should the receiver support allocating host-pinned memory (via cuCascade HOST tier) as a fallback when GPU memory is exhausted, instead of NACK-ing the sender?
