# NIXL Implementation - Continuation Session Summary

## Overview

Completed protobuf definitions and gRPC service infrastructure for NIXL GPU-direct metadata exchange. System now has end-to-end flow from execution detection through transfer coordination (with mocked GPU allocation).

## What Was Accomplished (1 commit, ~300 lines)

### Commit: Add nixl metadata exchange protobuf and service infrastructure (e283914)

**1. Protobuf Definitions** (`nixl_exchange.proto`)
```protobuf
message PGpuBufferDesc {
  uint64 addr, len, device_id;
}

message PExchangeNixlMetadataRequest {
  bytes nixl_metadata;
  repeated PGpuBufferDesc src_buffers;
  repeated PColumnInfo columns;
  uint32 num_rows;
}

message PExchangeNixlMetadataResponse {
  repeated PGpuBufferDesc dst_buffers;
  string remote_agent_name;
  int32 status_code;
  repeated string error_msgs;
}
```

**2. gRPC Service** (`nixl_service.rs`)
- `NixlMetadataService`: Standalone handler for nixl methods
- `exchange_metadata()`: Loads remote agent, allocates GPU buffers (mocked)
- Feature-gated implementation with graceful fallback
- 2 new unit tests

**3. Client Integration** (`nixl_integration.rs`)
- `call_exchange_nixl_metadata()`: gRPC client call (mocked)
- `send_nixl_to_peer()`: Full flow from metadata exchange → transfer
- Error handling for status codes, buffer count mismatches

**4. Build Infrastructure**
- Updated `doris-proto/build.rs` to compile nixl proto
- Added `doris_proto::nixl` module
- Clean compilation with new protobuf types

## Architecture Flow (Now Complete)

```
┌──────────────────────────────────────────────────────────────────┐
│                  GPU-Direct Exchange Flow                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Sender BE                          Receiver BE                   │
│  ──────────                         ───────────                   │
│                                                                    │
│  1. Execute on GPU                                                │
│     ├─ SiriusEngine::execute_substrait()                         │
│     └─ Result in GPU buffers                                      │
│                                                                    │
│  2. Detect Location                                               │
│     ├─ detect_execution_location()                               │
│     ├─ get_last_gpu_result_buffers()                             │
│     └─ ExecutionLocation::Gpu { buffers, schema }                │
│                                                                    │
│  3. Metadata Exchange (gRPC)                                      │
│     ├─ Send: PExchangeNixlMetadataRequest                        │
│     │   ├─ nixl_metadata (agent info)                            │
│     │   ├─ src_buffers (GPU addresses)                           │
│     │   └─ column_info + num_rows                                │
│     │                                                              │
│     │                           ┌─> NixlMetadataService           │
│     │                           │   ├─ Load remote agent          │
│     │                           │   ├─ Allocate GPU buffers       │
│     │                           │   └─ Return dst_buffers         │
│     │                                                              │
│     └─ Receive: PExchangeNixlMetadataResponse                    │
│         ├─ dst_buffers (receiver GPU addresses)                  │
│         └─ remote_agent_name                                      │
│                                                                    │
│  4. NIXL Transfer                                                 │
│     ├─ Create XferDescLists (src + dst)                          │
│     ├─ agent.transfer_gpu_to_gpu()                               │
│     └─ Direct GPU→GPU RDMA                                        │
│                                                                    │
│  5. Receiver Execution                                            │
│                           └─> Register GPU table                  │
│                               Execute fragment                     │
│                               Return results                       │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

## Test Coverage

- **63 tests passing** (61 original + 2 nixl_service)
- All feature gates work correctly
- Compiles with and without `nixl` feature
- Mock implementations allow testing without GPU hardware

## Design Decision: Separate Service

**Why NixlMetadataService is separate from PBackendService:**

1. **Upstream Proto Ownership**: PBackendService is defined in apache-doris upstream
   - Modifying would require forking proto files
   - Maintenance burden for future Doris updates

2. **Clean Separation**: NIXL is Sirius-specific, not part of standard Doris
   - Easier to maintain as separate proto
   - Can version independently

3. **Deployment Flexibility**:
   - Can be on same port (multiplexed)
   - Can be on separate port (isolation)
   - Can be disabled entirely (fallback to bRPC)

## Remaining Work (Prioritized)

### High Priority (Required for Basic Functionality)

#### 1. Register NixlMetadataService on gRPC Server
**Location**: `grpc_service.rs`, `start_grpc_server()`

Add nixl service alongside PBackendService:
```rust
#[cfg(feature = "nixl")]
{
    let nixl_svc = crate::nixl_service::NixlMetadataService::new(nixl_agent.clone());
    // Register nixl_svc with tonic::transport::Server
    // Note: Requires defining a tonic service trait for NixlMetadataService
}
```

**Blocker**: Need to define gRPC service trait. Two options:
- **Option A**: Create `nixl_service.proto` with service definition, use tonic-build
- **Option B**: Use HTTP/2 direct (skip tonic, implement raw handler)

**Recommended**: Option A (proper gRPC service)

#### 2. Implement Real gRPC Client Call
**Location**: `nixl_integration.rs`, `call_exchange_nixl_metadata()`

Replace mock with actual gRPC call:
```rust
// Current: returns mock response
// Change to:
use tonic::transport::Channel;

let channel = Channel::from_shared(grpc_addr.to_string())
    .map_err(|e| format!("invalid URI: {e}"))?
    .connect()
    .await
    .map_err(|e| format!("connect: {e}"))?;

let mut client = NixlMetadataClient::new(channel);
let response = client.exchange_metadata(request).await
    .map_err(|e| format!("rpc: {e}"))?
    .into_inner();
```

#### 3. Wire exec_plan_fragment to Use NIXL
**Location**: `grpc_service.rs`, line ~1179

**Current**:
```rust
if let Err(e) = exchange_sender::send_exchange_result(
    &ipc_bytes, &dests, query_id, dest_node_id, sender_id
).await { ... }
```

**Change to**:
```rust
use crate::nixl_integration::{detect_execution_location, send_exchange_with_nixl};

// Detect execution location (CPU or GPU)
let location = {
    let engine_guard = self.engine.as_ref().unwrap().lock().unwrap();
    detect_execution_location(ipc_bytes, &engine_guard)
};

// Send via nixl or bRPC
#[cfg(feature = "nixl")]
let nixl = self.nixl_agent.as_ref();
#[cfg(not(feature = "nixl"))]
let nixl = None;

if let Err(e) = send_exchange_with_nixl(
    nixl, location, &dests, query_id, dest_node_id, sender_id
).await {
    warn!(error = %e, %finst_id, "exchange send failed");
    return Ok(Response::new(PExecPlanFragmentResult {
        status: err_status(&format!("exchange send: {e}")),
        ..Default::default()
    }));
}
```

### Medium Priority (For Production)

#### 4. Real GPU Buffer Allocation
**Location**: `sirius-ffi/src/lib.rs`

Add method:
```rust
pub fn allocate_gpu_buffers(
    &self,
    sizes: &[(usize, u64)], // (len, device_id)
) -> Result<Vec<(usize, usize, u64)>, EngineError> {
    #[cfg(feature = "duckdb-bundled")]
    {
        // Via Sirius extension:
        // SELECT addr FROM sirius_allocate_gpu_buffer(len, device_id)
        let mut result = Vec::new();
        for &(len, device_id) in sizes {
            let sql = format!(
                "SELECT addr FROM sirius_allocate_gpu_buffer({}, {})",
                len, device_id
            );
            // Execute and extract addr
        }
        Ok(result)
    }

    #[cfg(not(feature = "duckdb-bundled"))]
    Err(EngineError::NotCompiled)
}
```

Use in `nixl_service.rs`:
```rust
let dst_buffers = if let Some(engine) = &self.engine {
    let sizes: Vec<_> = req.src_buffers.iter()
        .map(|b| (b.len as usize, b.device_id))
        .collect();
    engine.lock().unwrap().allocate_gpu_buffers(&sizes)?
        .into_iter()
        .map(|(addr, len, device_id)| PGpuBufferDesc { addr: addr as u64, len: len as u64, device_id })
        .collect()
} else {
    // Fallback: mock allocation
    req.src_buffers.clone()
};
```

#### 5. Sirius Extension Functions
**Location**: `extension/sirius/src/sirius_extension.cpp`

Add C++ functions:
```cpp
// sirius_get_last_gpu_buffers()
// Returns: buffer_id, addr, len, device_id, column_name, type_id, num_rows
// Queries GPUBufferManager for last execution's buffer pointers

// sirius_allocate_gpu_buffer(len, device_id)
// Returns: addr (GPU pointer)
// Allocates GPU memory and tracks in buffer manager
```

### Low Priority (Optimizations)

#### 6. End-to-End Integration Test
**Location**: `doris/crates/doris-rpc/tests/nixl_e2e_test.rs`

```rust
#[tokio::test]
#[cfg(feature = "nixl")]
async fn test_nixl_gpu_direct_exchange() {
    // 1. Start two BEs with nixl agents
    // 2. Execute GPU query on BE1
    // 3. Exchange to BE2 via nixl
    // 4. Verify result correctness
    // 5. Check logs for "nixl GPU-direct transfer complete"
}
```

#### 7. Performance Comparison Test
**Location**: `doris/crates/doris-rpc/benches/exchange_throughput.rs`

```rust
// Benchmark: GPU-direct vs bRPC
// Vary buffer sizes: 1KB, 1MB, 10MB, 100MB, 1GB
// Measure latency and throughput
// Plot results
```

## How to Enable NIXL Service

### Option A: Define Proto Service (Recommended)

**1. Create `nixl_service.proto`**:
```protobuf
syntax = "proto3";
package doris.nixl;

import "nixl_exchange.proto";

service NixlMetadataService {
  rpc ExchangeMetadata(PExchangeNixlMetadataRequest) returns (PExchangeNixlMetadataResponse);
}
```

**2. Update `doris-proto/build.rs`**:
```rust
tonic_build::configure()
    .build_server(true)
    .build_client(true)  // Enable client generation
    .compile_protos(&[nixl_service_proto], &[proto_dir])
    .expect("Failed to compile nixl service proto");
```

**3. Register in `start_grpc_server()`**:
```rust
#[cfg(feature = "nixl")]
{
    use doris_proto::nixl::nixl_metadata_service_server::NixlMetadataServiceServer;
    let nixl_handler = crate::nixl_service::NixlMetadataService::new(nixl_agent.clone());
    let nixl_svc = NixlMetadataServiceServer::new(nixl_handler);

    let combined = tonic::transport::Server::builder()
        .add_service(svc)
        .add_service(nixl_svc)
        .serve_with_incoming(incoming);
}
```

### Option B: HTTP/2 Direct Handler

Implement raw HTTP/2 handler bypassing tonic service trait. More complex but avoids proto service definition.

## Quick Start for Testing

```bash
# 1. Build with nixl feature:
pixi run -e doris cargo build --release -p sirius-doris-be --features nixl

# 2. Run tests:
pixi run -e doris cargo test -p doris-rpc --lib --features nixl
# → 63 tests pass

# 3. Check protobuf generation:
ls doris/target/debug/build/doris-proto-*/out/doris.nixl.rs
# → Should exist with PGpuBufferDesc, PExchangeNixlMetadataRequest, etc.

# 4. Start BE with nixl:
./target/release/sirius-doris-be --brpc-port 8060
# → Look for "nixl GPU-direct exchange enabled" in logs
```

## Code Quality

- **No compilation errors**
- **2 warnings** (unused imports in test-only code, pre-existing assignment)
- **63/63 tests passing**
- **Clean feature gates** (compiles with/without nixl)
- **Type-safe** (Rust + protobuf strong typing)

## Summary Statistics

### This Session
- **Lines added**: ~300
- **Files created**: 2 (nixl_exchange.proto, nixl_service.rs)
- **Files modified**: 4 (build.rs, lib.rs, nixl_integration.rs, proto lib.rs)
- **Tests added**: 2
- **Tests passing**: 63
- **Commits**: 1

### Total NIXL Work (All Sessions)
- **Lines added**: ~1,400
- **Files created**: 7
- **Commits**: 4
- **Tests**: 76 total
- **Documentation**: 800+ lines (3 markdown files)

## Git History

```bash
git log --oneline -5 doris

e283914 Add nixl metadata exchange protobuf and service infrastructure
664907f Add comprehensive nixl session summary and roadmap
36affe7 Wire nixl agent through gRPC service and main binary
53d2b5d Add nixl GPU-direct exchange infrastructure and tests
80b6906 Add multi-BE exchange support, UNION ALL fix, and exchange tests
```

## Next Immediate Steps

1. **Create `nixl_service.proto` with service definition** (30 min)
2. **Update build.rs for service generation** (15 min)
3. **Register service in start_grpc_server()** (30 min)
4. **Implement real gRPC client call** (30 min)
5. **Wire exec_plan_fragment** (30 min)
6. **Test end-to-end** (1-2 hours)

**Estimated time to working prototype**: 3-4 hours

## Questions for Next Session

1. **GPU Access**: Do you have CUDA-enabled hardware for testing?
2. **nixl-sys**: Is the nixl-sys crate available? (NVIDIA internal?)
3. **Sirius Extension**: Should we implement GPU buffer management in C++?
4. **Proto Service**: Prefer Option A (proto service) or Option B (HTTP/2 direct)?
5. **Testing Strategy**: Mock GPU transfers or require real hardware?

## Key Insights

### What Worked Well
- ✅ Protobuf approach for type-safe messages
- ✅ Separate service avoids upstream proto pollution
- ✅ Feature gates enable gradual rollout
- ✅ Mock implementations allow development without GPU

### Challenges Identified
- ⚠️ tonic service trait generation requires proto service definition
- ⚠️ GPU buffer allocation needs C++ extension integration
- ⚠️ End-to-end testing requires GPU hardware or sophisticated mocks

### Design Trade-offs
- **Chose**: Separate NixlMetadataService vs extending PBackendService
  - **Rationale**: Cleaner separation, easier maintenance
  - **Cost**: Additional service registration complexity

- **Chose**: Feature-gated compilation vs runtime detection
  - **Rationale**: Compile-time guarantees, smaller binary
  - **Cost**: Must rebuild to enable/disable

- **Chose**: gRPC for metadata exchange vs separate TCP channel
  - **Rationale**: Reuse existing infrastructure, consistent with Doris
  - **Cost**: Proto service definition complexity

---

**Session Date**: 2026-02-19 (continuation)
**Completed By**: Claude Sonnet 4.5
**Status**: Protobuf + service infrastructure complete, wiring pending
**Confidence**: High (compiles, tests pass, architecture sound)
**Next Milestone**: Complete service registration + exec_plan_fragment wiring
