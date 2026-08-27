# Plan: multi-fragment execution across multiple compute nodes

Two Sirius compute-node processes on one host, splitting the GPU, executing the fragments of a
single query across both — with the split configurable through `EngineConfig`, and **cross-node
transport with nixl as the final goal** (M6.x), reached through a brpc/Arrow correctness baseline
first. Extends the single-node native-batch exchange documented in
[new-exchange-design.md](new-exchange-design.md); the transport tiers and memory constraints come
from [old-exchange-design.md](old-exchange-design.md) §5–§6, whose upstream source
(mbrobbel/sirius `starrocks-exchange-design`) is byte-identical to the local copy, and whose doris
prior art is reachable in this repo's git objects at `origin/doris` (head `9d3d7fe9` —
`doris/crates/doris-rpc/src/{nixl_exchange,nixl_service,gpu_staging_buffer}.rs`, line citations
verified).

**nixl feasibility verdict (deep-research, 2026-08-05): POSSIBLE — and the design is forced, not
chosen.** Batches live in `cudaMallocAsync`-pool memory (`rmm::cuda_async_memory_resource`,
constructed non-exportable at `cucascade/src/memory/memory_space.cpp:120`). That memory is
*registrable* with nixl but cannot ride its GPU-to-GPU fast path on this hardware: UCX `cuda_ipc`
exports mempool memory only via fabric handles (IMEX/NVLink-domain hardware, absent on a PCIe
L4), and legacy `cudaIpcGetMemHandle` "cannot be used on any memory allocated through
cudaMallocAsync" (NVIDIA stream-ordered-allocator docs). Pool memory over nixl would *appear* to
work while silently taking staged host copies — this subsystem's signature failure shape. The
design therefore commits to old-exchange-design §6's staging arena: a **`cudaMalloc` exchange
arena registered with nixl once at startup, bump-leased per transfer**, with
`cudf::chunked_pack` *into* the send lease as the wire format (the pack gather must write
somewhere — the lease is that somewhere, so staging costs **zero extra copies**) and zero-copy
`cudf::unpack` + `make_data_batch_from_view` on receive. nixl has official Rust bindings
(`nixl-sys = "1.3"` on crates.io, safe API over the full agent surface).

Every file:line below was verified against this tree (branch `demo-streaming-integration`) by a
four-domain research pass; claims marked **GENERAL** are StarRocks knowledge not re-verified here.

---



## 1. Goal and finish line

**Goal.** One query, two CN processes: the FE places the scan fragment's instances and the
aggregate fragment on different nodes; intermediate data crosses the node boundary over an
explicit transport; the answer is correct; and each node runs inside its configured slice of the
GPU.

**The finish line is falsifiable, in the tradition of the single-node milestone:**

```sql
SET new_planner_agg_stage = 1;
-- TPC-H Q6 over FILES() with TWO parquet files  ->  61567694.95019999
```

with, in the logs:

- CN-A: `transmitted batches to remote exchange stream_id=... dest=127.0.0.1:8061 batches=N (N>0)`
- CN-B: `received remote batches stream_id=... sender_id=... batches=N` and
`relayed native batches across a fragment boundary ...` for its own local sender
- `nvidia-smi`: two CN processes, each within its configured pool budget

Two files in the scan makes the cross-node placement **deterministic**, not a coin flip (§4.7).
The negative control: the same query on the single-CN cluster returns the same answer, and a
sender whose destination is remote **fails loudly** if the transport is disabled — never a silent
hang (§7, trap 1).

## 2. What exists (verified)

- **The FE is ready for two CNs out of the box.** CNs self-register
(`ALTER SYSTEM ADD COMPUTE NODE "host:heartbeat_port"`, `src/lib.rs:191-193, 1189-1243`);
identity is `(host, heartbeat_port)` (`SystemInfoService.java:261-268`); the FE learns each
CN's brpc port from the heartbeat reply (`src/lib.rs:435-460`) and hands every sender the
receiver's real address: `dest.setBrpc_server(worker.getBrpcIpAddress())`
(`ExecutionDAG.java:560`). Placement round-robins across CNs — shuffled per-scan-range for the
scan fragment (`FileScanNode.java:552-558, 457-460`), a process-wide round-robin for the gather
fragment (`RemoteFragmentAssignmentStrategy.java:89-114`, `DefaultWorkerProvider.java:168-183`).
- **The protocol surface is already compiled into the CN.** `PTransmitChunkParams`
(`internal_service.proto:73-98`) and the `transmit_chunk` rpc (`:807`) are generated into the
binary (`build.rs:13-24`); unimplemented methods return a clean brpc `ENOMETHOD`
(`build.rs:118-128`, `prpc.rs:133-137`). The PRPC layer carries opaque attachments both ways
(`prpc.rs:36, :88`) — stock StarRocks moves chunk bytes in the attachment too
(`exchange_sink_operator.cpp:794-813`).
- **The engine's cross-boundary primitives are done and proven** (single node): sender-set EOS,
`push`/`close_input`, parked fragments, `relay_from`, TPC-H Q6 over native batches with the
temp-parquet path deleted. Full suite 2173 cases, 0 failures.
- **But the FFI has no wire exits or entrances.** `result_to_arrow` throws on any fragment that
declared an output stream (`sirius_ffi.cpp:541-546`) — a parked sender's output cannot be
exported; and no Arrow *import* exists anywhere in `sirius_ffi.cpp`. Both directions of the
process boundary are new surface (D4, D5).
- **FE dispatch is receiver-first with a synchronous per-stage ACK**
(`AllAtOnceExecutionSchedule.java:130-136`, `Deployer.java:208-213`) — the receiver is
registered before any sender runs, under the default scheduler.
- **GPU carve-outs are a config-file feature that already exists.** The Sirius YAML supports
`memory.gpu.usage_limit_bytes` (absolute) or `usage_limit_fraction` — **of TOTAL device
memory, not free** (`reservation_manager_configurator.cpp:294-295`) — and the pool is primed
eagerly and retained forever (rmm `cuda_async_memory_resource`, release threshold unset ⇒
uint64-max; `cucascade/src/memory/common.cpp:251-256`). Device selection:
`topology.gpu_ids` in YAML or `CUDA_VISIBLE_DEVICES` per process — the documented canonical
way (`docs/super-sirius/multi-gpu-architecture.md:240`).



## 3. What breaks with two CNs (the gap, verified)


| #   | Gap                                                                                                                                                            | Where                                                                         |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| G1  | The sender **never reads** `destination.brpc_server`; every destination is assumed to be this process                                                          | `compute_node_service.rs:387-392`; zero references to `brpc_server` in `src/` |
| G2  | The receiver is executed **inline in the last sender's RPC thread** — impossible when the last sender is on another node                                       | `compute_node_service.rs:413-415`, `local_exchange.rs:102-115`                |
| G3  | **No brpc client exists** — the CN can serve but never call a peer; `Frame::for_request` is `#[cfg(test)]`                                                     | `prpc.rs:227-259`                                                             |
| G4  | **No transmit endpoint** — `transmit_chunk`/`transmit_data` fall through to `method_not_implemented`                                                           | `compute_node_service.rs:89-205`, `build.rs:125`                              |
| G5  | The FFI cannot ingest external data: the only boundary crossing is `relay_from` over a **local** parked fragment; no `push`/`close_input` is exposed to Rust   | `sirius_ffi.cpp:495-514`, `rust/crates/sirius-sys/src/lib.rs:33-123`          |
| G6  | Parked output is engine-thread-local (`HashMap<SenderSlot, Fragment<'_>>`) — nothing can put it on a wire                                                      | `engine.rs:151`                                                               |
| G7  | Two default-configured engines **cannot coexist**: each primes 0.95 × total device memory at bring-up; the second OOMs                                         | `sirius_config.cpp:384-398`, §2 above                                         |
| G8  | Port defaults collide: heartbeat 9050, thrift 9060, brpc 8060 all bound                                                                                        | `lib.rs:148-169, 886-945`, `brpc.rs:51-59`                                    |
| G9  | Multi-file `FILES()` schema inference is rejected — blocking the deterministic cross-node test shape                                                           | `compute_node_service.rs:633-639`                                             |
| G10 | A cross-node placement today is a **silent hang**: CN2's receiver waits for a `push_sender` that lands on CN1; the FE polls `fetch_data` until `query_timeout` | `local_exchange.rs:121-122`, `result_store.rs:84-111`                         |


The failure mode of G10 is this subsystem's signature: success-shaped nothing. The plan's first
code change (M1) converts it into a loud error before any transport exists.

## 4. Design decisions



### D1 — Two processes, one per engine. Never two engines in one process

`SiriusContext` is `!Send`/`!Sync` and explicitly one-per-process
(`rust/crates/sirius/src/lib.rs:30-31`). The GPU is split by **configured pool carve-outs**, not
by process isolation tricks: each CN primes its own `cudaMallocAsync` pool at bring-up and the
in-process cap is the `reservation_aware_resource_adaptor` over it
(`memory_space.cpp:104-136`). MPS is optional (throughput, not correctness) and MIG is
unsupported on the L4 — neither is needed for the test topology. **On a 2-GPU host the same**
`EngineConfig` **splits by device instead** (`CUDA_VISIBLE_DEVICES` / `topology.gpu_ids`), so
"share one GPU" and "one GPU each" are the same mechanism with different values.

### D2 — `EngineConfig` grows real GPU knobs; C++ stays untouched

Today `EngineConfig` is one flag: `--sirius-config <yaml>` (`main.rs:55-61`). Extend it:

```rust
struct EngineConfig {
    /// Explicit YAML wins over everything below.
    #[arg(long)] sirius_config: Option<PathBuf>,
    /// Absolute GPU pool budget, e.g. "8GiB". Mutually exclusive with fraction.
    #[arg(long)] gpu_memory_limit: Option<String>,
    /// Fraction of TOTAL device memory (documented: total, not free).
    #[arg(long)] gpu_memory_fraction: Option<f64>,
    /// Device selection; sets CUDA_VISIBLE_DEVICES before engine bring-up.
    #[arg(long)] gpu_device: Option<u32>,
    /// Pinned-host budget, e.g. "16GiB" (default 8GiB/NUMA doubles with 2 CNs).
    #[arg(long)] host_memory_limit: Option<String>,
    /// Per-instance log/telemetry root (SIRIUS_LOG_DIR + telemetry output_directory).
    #[arg(long)] engine_dir: Option<PathBuf>,
}
```

When the override flags are used without `--sirius-config`, the CN **derives a YAML** in its
`engine_dir` and passes it to `SiriusContext::from_config_file` — the low-level
`sirius.space.gpu` schema already accepts `device_id` and absolute `memory_capacity`
(`sirius_config.cpp:35-48`), so no FFI or C++ change is required. A programmatic
`make_context_from_options(...)` FFI is a recorded follow-up, not on this path (it would need a
new public setter on `sirius_config`, whose only populators today are `load_from_file` /
`apply_defaults`, `sirius_config.hpp:141-198`).

Guardrail: if neither a YAML nor a limit is given **and** the CN detects another Sirius process
on the device (via `nvidia-smi --query-compute-apps` at startup), refuse to start with a message
naming G7 — the alternative is an OOM abort mid-priming.

Test values for this host (1× L4, 23 GiB): `--gpu-memory-limit 8GiB --host-memory-limit 16GiB`
per CN, leaving headroom for two CUDA contexts and the untracked default-pool allocations
(`common.cpp:237-248`).

### D3 — Transport v1: `transmit_chunk` over brpc, Arrow IPC in the attachment

Both endpoints are our binary, so the wire format is ours. Ride the **already-declared**
`transmit_chunk` rpc: protobuf fields carry addressing (`finst_id` = receiver instance,
`node_id` = exchange node id = engine stream id, `sender_id`, `eos`, `sequence`); the batch
payload is **Arrow IPC stream bytes in the brpc attachment** (add `arrow-ipc = "59"` to match the
pinned arrow crates). We deliberately do not mimic `ChunkPB`'s column layout — no Rust
implementation of StarRocks' serde exists, and no stock BE is a peer in this topology
(`REFUTED as practical` in the protocol research). Sequence numbers detect loss; frames stay
under the decoder's 256 MiB cap (`prpc.rs:16`) by splitting large batches across calls.

This is, explicitly, the **bRPC/CPU fallback tier** from old-exchange-design §5, built first as
the correctness baseline: GPU → host → wire → host → GPU. The invariant becomes: **native
batches within a process, Arrow IPC only at process boundaries and at the MySQL result edge** —
until M6.4, when the nixl tier keeps intermediate data device-resident across the process
boundary too. The brpc tier is not throwaway: the design's own retry policy (retry once over
nixl → fall back to brpc → only then fail the query, old-exchange-design:684-687) **requires**
it to exist permanently. Tier selection lands in M6.4; hand-rolled fd-mempool IPC
(`cudaMemPoolExportToShareableHandle` over a unix socket) is explicitly rejected — staging-free
at the CUDA level but it bypasses nixl and does not generalize cross-node, forking the
transport.

The client side is small: PRPC framing is symmetric, `Frame::encode()` is not test-gated
(`prpc.rs:323`), so a client is connect → write frame → match `correlation_id`, with one
connection per peer kept in a map (there is exactly one peer in this topology).

### D4 — Sender-side routing: "is this destination me?" — and remote senders run as *result* fragments

The CN learns its own identity at startup (`advertise_host`, `brpc_port`) and compares each
`destination.brpc_server` against it — **hostname AND port equality**, the same rule the stock
BE uses (`exchange_sink_operator.cpp:140-143`), which is what makes two CNs on one host
correctly see each other as remote. The decision happens **before** `executor.run`
(`compute_node_service.rs:396-400`), and the service must be handed the CN's identity — today
it lives only in `ComputeNodeConfig` and is not plumbed into `SiriusComputeNodeService`.

Local destination → today's path (park + in-process rendezvous), byte-for-byte unchanged.

Remote destination → **run the fragment as a result fragment: skip** `declare_output`**.** This is
forced by the FFI, not a preference: `Fragment::result_to_arrow` throws on any fragment that
declared an output stream (`sirius_ffi.cpp:541-546`), and the surface exposes no `pull` and no
per-batch export — parked output has **no wire exit** (adversarial-critique finding A1).
`declare_output` is the whole path selector (new-exchange-design §6), so a remote sender simply
selects the other path: its subtree runs rooted in the result collector, `into_arrow()` yields
owned `RecordBatch`es on the CN thread, and the transmit loop sends one `transmit_chunk` per
batch with a final `eos=true` frame (which may carry the last batch, as stock does,
`data_stream_mgr.cpp:166-176`). Parking is meaningless for a remote consumer — the bytes must
leave the process regardless — so nothing of value is lost, and sender memory is bounded by the
Arrow copy of one fragment result.

**Spike required at M2 start:** confirm a DATA_STREAM_SINK fragment translated and run without
`declare_output` produces identical rows through the result-collector path for the demo plan
shapes (the translator exercises this path only for RESULT_SINK fragments today). Fallback if
the spike fails: a new `Fragment::output_to_arrow(stream_id, ...)` FFI that drains the streaming
sink's repository through the existing host-conversion chain — more C++, same wire contract.

### D5 — Receiver-side ingest: buffer at the rendezvous, then park-then-relay

The receiver CN's `transmit_chunk` handler appends `(sender_id, bytes)` to the exchange
rendezvous and records `close` on `eos` — it does **not** touch the engine. When the sender
*set* completes (counting local parked senders and remote closed senders alike), the ready
fragment is executed exactly as today, with one addition on the engine thread: remote inputs are
fed through a new FFI pair before `run()`:

```cpp
// sirius_ffi.hpp additions
void Fragment::push_arrow(uint64_t stream_id, /* Arrow IPC bytes */ rust::Slice<const uint8_t>);
void Fragment::close_input(uint64_t stream_id, uint32_t sender_id);
```

`push_arrow` deserializes the IPC stream, converts via Arrow interop to a device table, wraps it
as a `data_batch`, and calls `session().push(stream_id, batch)` — the mirror of the result-edge
`into_arrow()`. There is **no Arrow import anywhere in** `sirius_ffi.cpp` **today** (only export via
`ResultArrowArrayStreamWrapper`) — this is genuinely new engine-facing surface, legal between
`build()` and `run()` exactly where `relay_from` sits in the ordered lifecycle.

This is **park-then-relay**, and it is forced, not chosen: the context is thread-confined
(`RefCell<UniquePtr<Context>>`, `!Send`/`!Sync`) and `Fragment::run()` blocks that thread until
the pipelines finish — while a receiver runs, no FFI call can inject data. Stated plainly: **the
streaming primitives' receive/compute overlap is deliberately unused at the process boundary**
in v1; all remote bytes are on the receiver CN (host RAM) before its fragment runs. This also
matches new-exchange-design §9's warning that the live producer has never fired under a running
engine — the network will not be its first caller.

The engine-thread request shape must change too: `ExecuteRequest.inputs` today carries parked
`SenderSlot`s, and the engine **errors** on a slot with no locally parked fragment
(`engine.rs:200-204, 237-240`) — remote inputs cannot masquerade as slots. `ExecuteRequest`
gains a `remote_inputs: Vec<(stream_id, sender_id, Vec<ArrowIpcPayload>)>` field, fed through
`push_arrow` + `close_input` before `run()`.

Sender output **names** ride in the Arrow IPC schema header, replacing the local-path
`SenderOutput.names` the receiver's deferred translation needs (`compute_node_service.rs:421-457`).
The mismatch rule mirrors today's local cross-check: all senders' schemas must agree, first
registered wins as the reference, disagreement is a loud error.

### D6 — The rendezvous generalizes; receiver execution leaves the RPC thread

`LocalExchange` keeps its shape (key: `ExchangeKey{receiver_instance, node_id}`; readiness =
per-exchange sender count met) and its `SenderOutput` becomes a two-variant source:

```rust
enum SenderSource {
    LocalParked { names: Vec<String>, slot: SenderSlot },          // today's path
    Remote { schema: SchemaRef, batches: Vec<Bytes>, closed: bool } // fed by transmit_chunk
}
```

Ready fragments are handed to a dedicated dispatch task (mpsc) instead of being executed inline
in whichever RPC thread completed the set (G2). This is required for cross-node and healthy for
single-node: the sender's `exec_plan_fragment` RPC stops blocking on the receiver's entire
execution.

EOS mapping is 1:1 with the engine primitive: `eos=true` → `close_input(node_id, sender_id)`
(`stream_session.hpp:95-99`); `per_exch_num_senders` (the **cluster-wide** sender count,
`ExecutionDAG.java:502-509`) is already plumbed into `declare_input_sender`
(`compute_node_service.rs:481-488`, `engine.rs:206-212`). StarRocks' `(sender_id, be_number)`
disambiguator collapses to `sender_id` — unique per sender instance within a query. `be_number`
is ignored, recorded as such.

**Duplicate delivery is handled from day one**, because brpc retry can replay a frame: chunk
`sequence` numbers are tracked per `(exchange, sender)` — a duplicate is dropped idempotently, a
gap is an error; duplicate `eos` from the same sender is idempotent (the engine's `_closed` is an
identity set — a repeated close from sender 0 can never stand in for sender 1, the exact property
the sender-*set* design exists for).

**Dispatch-ordering contract.** The default FE scheduler deploys fragments in topological order
with a **synchronous per-stage ACK** (`AllAtOnceExecutionSchedule.java:130-136`,
`Deployer.java:208-213` `waitForDeploymentCompletion`) — the receiver CN's `register_receiver`
has completed before any sender is dispatched, so v1 does not need stock-BE-style
early-chunk-before-registration buffering. The `transmit_chunk` handler still treats an unknown
exchange as a **retryable error** rather than trusting this: the guarantee is scheduler-specific
(it does not hold under `PhasedExecutionSchedule`) and does not survive FE RPC retry.

### D7 — Determinism: two files in, two scan instances out

With ≥2 scan ranges, the FE's min-load selector spreads the scan fragment across both CNs
(`NormalBackendSelector.java:65-137`) while the gather receiver lives on exactly one — **at
least one sender is remote, every run**. This requires fixing G9: multi-file `FILES()` schema
inference reads the first file's footer and validates arity across the rest. Single-file runs
stay ~50/50 cross-node (two independent round-robins) — fine for soak, wrong for CI. As a debug
aid, the vendored FE supports `EXPLAIN SCHEDULER <query>` to observe placement without
executing (with the caveat that it advances the round-robin counter).

Bonus: this shape is a genuine **cross-node fan-in** (2 senders, distinct `sender_id`s, one
remote), which closes the "fan-in never tested at fragment level" gap from
new-exchange-design §9 in the same stroke.

## 5. Milestones

Each lands independently, tests green, and is bisectable. C++FFI additions gate on the C++
suite; CN changes gate on `cargo test -p sirius-starrocks-cn` (pure-Rust CI path) plus the
GPU-gated engine tests locally.

### M0 — Two engines on one GPU (config only)

`EngineConfig` per D2 (derived-YAML path, `gpu_device` → `CUDA_VISIBLE_DEVICES`, distinct
`SIRIUS_LOG_DIR`/telemetry dirs); `pixi run cluster2` task: FE + CN1 (defaults) + CN2
(`--heartbeat-port 9051 --thrift-port 9061 --brpc-port 8061 --http-port 8041 --starlet-port 9071`,
own budget). No FE conf change is needed (nothing in `fe.conf` lists nodes).
**Verify:** `SHOW COMPUTE NODES` lists both; both pools visible in `nvidia-smi` within budget;
single-fragment queries succeed on whichever CN they land.
**Known limitation, stated in the task description:** exchange queries may hang at this
milestone when placed cross-node (G10) — which is why M1 follows immediately.
*Files:* `main.rs`, `engine.rs`, `pixi.toml`, `lib.rs` (port docs). *Tests:* YAML-derivation unit
tests; a `--gpu-memory-limit` parse/suffix test; cluster2 smoke by hand.

### M1 — Identity, routing decision, and the death of the silent hang

CN self-identity; read `destination.brpc_server` (G1); local → unchanged, remote → **explicit
error** (`"remote exchange destination 127.0.0.1:8061 — cross-node transport not yet built"`),
propagated through the fragment failure path so the FE errors instead of timing out. Rendezvous
refactor per D6 (SenderSource enum, dispatch task) with only the Local variant constructible.
**Verify:** destination-is-me decision matrix (same host/port, same host/different port,
different host); receiver-no-longer-inline test; all existing exchange tests byte-identical.
*Files:* `compute_node_service.rs`, `local_exchange.rs`, `fragment_executor.rs`.

### M2 — The wire: FFI ingest + transmit_chunk + PRPC client

Opens with the **D4 spike** (sender-as-result-fragment row equivalence); its outcome selects the
sender path before any transport code is written. Then three parallel-reviewable pieces:

1. **FFI (C++):** `Fragment::push_arrow` + `Fragment::close_input` per D5, with C++ tests
  proving Arrow IPC → device batch → `session().push` round-trips values (not counts — the
   subsystem's history demands value assertions), and that `push_arrow` after EOS is a loud
   error.
2. **Server:** implement `transmit_chunk` on `SiriusComputeNodeService` (the trait method is the
  entire delta — routing/decoding/attachment plumbing are generated); buffer into the
   rendezvous Remote variant; `eos` completes the sender; duplicate `sequence` is idempotent,
   gaps are an error.
3. **Client:** minimal PRPC client (symmetric framing); sender remote path per D4
  (drain → Arrow IPC → per-batch transmit → final eos), with the >256 MiB split.

**Verify:** loop-back integration test — one process, the real server dispatch fed by the real
client encoder over a local socket, a stub executor recording pushes; asserts batch bytes, order,
eos mapping, sequence-gap error. Engine-gated test: a receiver fragment fed via `push_arrow`
produces the same values as the `relay_from` path for identical input.

### M3 — Deterministic cross-node e2e

Fix multi-file `FILES()` schema inference (G9). Add the acceptance query (two-file Q6) to
`cluster2`; wire the three log lines from §1.
**Verify (the finish line):** correct revenue; CN-A logs remote transmit with `batches>0`; CN-B
logs remote receive AND its own local relay (fan-in of local + remote senders); single-CN cluster
gives the same answer; `nvidia-smi` shows both processes within budget throughout.

### M4 — Hardening: failure, cancellation, and state GC

- Transmit failure → fragment error → FE; the sender frees any transient state on failure.
- Receiver-side await timeout (senders that never close); host-RAM cap on buffered remote bytes
with a loud error, and the accounting note tied to new-exchange-design §8 (remote-buffered
bytes are host-side and invisible to the downgrade executor — same conflict, new tier; and the
PRPC layer has **no flow control**, so the cap is the only guard).
- **Per-query GC and a cancel surface.** Today parked fragments are cleared only when a *later*
query fails (`engine.rs:156-162`), `cancel_plan_fragment` returns `method_not_implemented`,
and `SenderSlot` carries no `query_id` (`fragment_executor.rs:41-49`) — so a wedged cross-node
query leaks parked GPU output on one CN, buffered host chunks and a `PendingReceiver` +
reserved `ResultStore` entry on the other, indefinitely. With two 8 GiB carve-outs that is a
large slice of a node pinned by one failure. Deliverables: widen the exchange/parked keys with
`query_id` (or a side map), GC on new-query arrival, and a `cancel_plan_fragment` handler that
clears all three stores.
- Kill-one-CN mid-query test: the other CN errors, the FE reports, nothing hangs.



### M5 — PR packaging

The stack mirrors the single-node discipline: one reviewable concern per PR —
(1) EngineConfig + cluster2, (2) routing + rendezvous refactor, (3) FFI ingest pair,
(4) transport, (5) multi-file FILES() + e2e, each with its tests. Docs: extend
new-exchange-design §7 with the two-CN sequence diagram and correct §0's "no cross-CN anything"
line when it stops being true.

### M6.x — The nixl tier (the final goal)

Grounded by the feasibility verdict (header). Each sub-milestone lands independently.

**M6.0 — Hardware/stack ruling (probes, no engine changes).** 30-line binaries settling every
remaining UNVERIFIED before design code: `cudaDevAttrMemoryPoolSupportedHandleTypes` +
`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED` on the L4; a nixl v1.3.2 + UCX two-process
`cudaMalloc`→`cudaMalloc` transfer with `UCX_TLS=cuda_copy,cuda_ipc,sm,tcp,self` (nixl issue
#1628's same-host recipe); the NO_IPC pool-memory bandwidth baseline vs the arena path — the
number that justifies M6.1-M6.4 effort. *Files:* `tools/transport_probe/`, a pixi task.
*Reviewable because:* purely additive; converts research claims into committed fact.

**M6.1 — Exchange staging arena (C++, nixl-free).** `cudaMalloc` arena: reserved **once**
against the cuCascade budget at startup (old-exchange-design:563-569), bump allocator + RAII
leases, send/receive partitions, `EngineConfig` knob (`--gpu-staging-bytes`). v1 lease policy is
**copy-out-on-arrival** (one D2D into an ordinary pool batch), which bounds lease lifetimes and
makes received batches fully accounted + spillable immediately — deferring §6 constraints 1-3
(lease-aware spill, receive floor, copy-out credit) to M6.5; constraint 4's **send-copy credit**
(one max batch) ships here, because a spilled batch must `clone_to<gpu>` before packing and that
allocation can deadlock without it. *Tests:* lease exhaustion / accounting / RAII return in
`sirius_unittest`. *Reviewable because:* a self-contained allocator with GPU-free invariants.

**M6.2 — Packed FFI pair (device-resident export/import).**
`Fragment::export_packed(stream_id) → {metadata, device_ptr, len, ready_event}`: pull →
`read_only_data_batch` lock held for the whole transfer (residency + immutability +
spill-exclusion are **mutex-enforced**, `convertible_data_batch.hpp:87-93,312-333`) → wait the
batch's writer event (STREAM-LINEAGE, `data_batch.hpp:303-321`) → `chunked_pack` into a send
lease with an explicit `temp_mr`. `Fragment::push_packed(stream_id, metadata, device_ptr)`:
zero-copy `unpack` (`contiguous_split.hpp:314` — "No new device memory is allocated") →
copy-out-on-arrival → `session().push`. Siblings of M2's `push_arrow`; both funnel to the same
engine entry. `cudf::pack` also erases the buffer-walk constraints (sliced tables unsupported,
device-read STRING sizing) that a per-buffer descriptor design would inherit. *Tests:* value
round-trip `export_packed`→`push_packed` ≡ `relay_from`; sliced-table case; push-after-EOS loud
error; re-run the D4 spike with `export_packed` as the drain target. *Reviewable because:* pure
FFI + cudf, pinned to the proven in-process path by an equivalence test.

**M6.3 — nixl agent tier (Rust, `nixl-sys = "1.3"`).** One agent per CN named
`{host}:{brpc_port}` (the doris `sirius-be-{host}` naming collides for two CNs on one host);
arena registered at startup; `getLocalMD` blobs exchanged over the **existing brpc control
plane** (no etcd, no p2p socket); `createXferReq(WRITE)` lease→lease; completion notification
blob `(query_id, stream_id, sender_id, seq)` via `getNotifs`; **EOS stays on brpc →
`close_input`** — one source of truth for sender-set completion; a single dedicated transport
task owns the agent (the Rust binding documents a multithreading deadlock caveat). *Tests:*
two-process same-host loopback over raw arenas; MD-cache invalidation on peer restart.
*Reviewable because:* transport in isolation, no engine coupling.

**M6.4 — Tier integration + fallback.** Sender selects nixl-gpu when peer MD is cached and a
lease is acquired, else the D3 brpc tier; retry-once-then-fallback per
old-exchange-design:684-687; arena-exhaustion→brpc test. **The nixl finish line:** the same
two-file Q6 from §1, now with `transmitted batches via nixl stream_id=... bytes=N` on CN-A and
zero host serialization on the hop — same falsifiable shape, one tier deeper.

**M6.5 — Recorded, not gating.** Lease-aware spill + receive-staging floor + copy-out credit
(§6 constraints 1-3) for staging-backed residency; the host-staging deployment tier (a spilled
batch could send from a host lease without re-upgrading); the direct pool-registration
experiment (fabric hardware or UCX fd-mempool support, tracking openucx/ucx#7110); and the §8
spillability decision — still the precondition for making this tier default under memory
pressure. `data_batch::release_or_copy_table` (cuCascade PR #148) is **absent in-tree and on
upstream main** — the zero-copy sink steal is not available; the send path is built on the
read-only lock + deep-copy-under-credit shape until upstream rules on #148.

## 6. Test plan summary


| Level                      | What                                                                                                                                | Gate                     |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| Unit (Rust, no GPU)        | destination-is-me matrix; SenderSource rendezvous (local, remote, mixed, duplicate eos, sequence gap); YAML derivation; frame split | CI (`cn-test-no-engine`) |
| Unit (C++)                 | `push_arrow` value round-trip; post-EOS push is loud; `close_input` idempotence via FFI                                             | `sirius_unittest`        |
| Integration (Rust, no GPU) | client→server loop-back over a socket with stub executor                                                                            | CI                       |
| Engine-gated (GPU)         | `push_arrow`-fed receiver ≡ `relay_from`-fed receiver, values compared                                                              | local                    |
| Cluster                    | `cluster2` two-file Q6, the §1 finish line + probes; kill-one-CN test                                                               | manual/demo              |




## 7. Risks and traps (inherited and new)

1. **Silent-empty is this subsystem's signature failure.** Three single-node defects presented
  as success-with-no-rows. Rule: every unroutable/unsupported case errors loudly (M1 does this
   before the transport exists); every e2e assertion compares **values**, and the transmit log
   line carries a batch count precisely so `batches=0` is visible.
2. **G7 pool priming.** The second default-configured CN dies by OOM mid-bring-up. `EngineConfig`
  guardrail + fail-fast at `SiriusEngine::start` (already fail-fast, `engine.rs:80-87`).
3. **Fraction is of TOTAL, not free** — two `usage_limit_fraction: 0.5` processes do not fit
  (contexts + untracked default-pool allocations). Prefer absolute `--gpu-memory-limit`; state
   it in the flag's help text.
4. **The FE's round-robin is process-global state** — an `EXPLAIN SCHEDULER` shifts the next
  query's placement. Never use it inside the deterministic test; two files make placement
   deterministic without it.
5. **The lifecycle mutex** (`QueryBeginStandalone` non-recursive; a failing path that skips
  `QueryEnd` deadlocks the next statement silently). The transmit/dispatch task must never
   touch the engine outside the `ExecuteRequest` channel — the engine thread remains the only
   caller, preserving the existing discipline.
6. **A hung test holds the GPU** and `pgrep -f "sirius_unittest$"` misses filtered runs; with two
  CNs the confusion doubles. Bound every run (`timeout --signal=KILL`), check
   `nvidia-smi --query-compute-apps` before believing anything finished.
7. **256 MiB frame cap** (`prpc.rs:16`) — large single batches must split; test with an
  artificially low cap.
8. **Result routing:** the FE fetches only from the root fragment's instance-0 worker
  (`DefaultCoordinator.java:646-662`) — the receiving CN must be the one buffering encoded rows,
   which D5/D6 preserve. No change needed, but any future refactor that moves result encoding
   off the receiver CN breaks fetch.



## 8. Out of scope

- **Shuffle / partitioned output** — still blocked on two-phase aggregation in the translator
and `output_partition` handling; unchanged from the single-node backlog (issues G/H).
- **Partition-hash parity** with StarRocks (only observable with mixed-vendor peers or
bucket-shuffle).
- **Concurrent queries per CN** — the engine remains one-query-at-a-time; two CNs give
cluster-level parallelism without touching per-process lifecycle isolation (Stage B).
- **Live producer / push-into-running-fragment** — park-then-relay everywhere (D5). The waker
path under a running engine remains the highest-value engine test to add, but this plan does
not depend on it.
- **Direct nixl registration of pool batch memory** — demoted to an M6.5 experiment: UCX's
  cuda_ipc mempool export is fabric-only (absent on the L4) and its posix-fd path is
  VMM-oriented; the staging arena is the committed design. Hand-rolled fd-mempool IPC is
  rejected outright (forks the transport, doesn't generalize cross-node).
- **Receive/compute overlap across nixl** — transfers land in the receive lease before the
  receiver fragment runs (park-then-relay preserved); overlap needs the live-producer path,
  which needs Stage B.
- **Stock-BE interop** — both exchange endpoints are our CN; `ChunkPB` layout compatibility is
explicitly not attempted.



## 9. Open questions

1. **The D4 spike** — does a DATA_STREAM_SINK fragment run *without* `declare_output` produce
  identical rows through the result-collector path for every demo plan shape? Gate on this at
   M2 start; the `output_to_arrow` FFI is the fallback.
2. **Arrow →** `data_batch` **ingestion mechanism** for `push_arrow` — cudf's Arrow interop vs
  DuckDB-side ingestion. No import path exists today; choose at M2 with a value-round-trip test
   either way.
3. **FE behavior when** `cancel_plan_fragment` **returns** `method_not_implemented` at
  `query_timeout` — retry, blacklist, or leak? Unread FE code; M4's cancel handler makes it
   moot, but verify before shipping M4.
4. `enable_gather_fragment_locality_optimization` **default** (claimed `false`,
  `SessionVariable.java:1262-1263`, not re-verified) — matters only for the nondeterministic
   fallback harness, not the two-file deterministic path.
5. **Multi-file** `FILES()` **blocker** (`compute_node_service.rs:633-639`) — cited but not re-read
  by the critique pass; re-verify the exact rejection before M3 commits to the two-file trick.
6. **Where does the remote-ingest buffer meet the parked-fragment model?** D5 buffers bytes in
  the rendezvous and pushes at build time; an alternative is a receiver-side "ghost fragment"
   parked per remote sender. The buffer approach wins on simplicity but holds bytes in host RAM
   twice (rendezvous + IPC decode) — measure at M2, revisit if the copy shows up.
7. `be_number` — dropped from the EOS key (D6). Correct for one query per exchange with
  FE-assigned unique `sender_id`s; re-verify if instance-level parallelism (`dest_dop`,
   `pipeline_driver_sequence`) is ever enabled.
8. **Descriptor-table cache across two CNs** — the FE sends the full table at least once per
  worker (`Deployer.java:145-146, 280-285`), matching the CN cache. Verified in code, not yet
   observed live with two CNs (M0 verifies).
9. `enable_exchange_pass_through` — the FE may set it for co-located instances; the CN
  ignores it today. Confirm at M3 that it is absent or harmless for same-host different-port.
10. **Host pinned budget** — whether `fixed_size_host_memory_resource` pins eagerly at init
  (doubling matters for two CNs) — and the empirical validation that 8 GiB per CN plus two
    CUDA contexts plus untracked default-pool allocations actually fits the 23 GiB L4. Both at M0.
11. **Does the FE probe the advertised-but-unbound http/starlet ports** when a second CN joins?
  Single-CN tolerates it; M0 smoke will show whether the second registration trips anything.



## 9b. Open questions from the nixl verdict (M6.0 settles these empirically)

- Does the L4/driver report `posix_file_descriptor` in `cudaDevAttrMemoryPoolSupportedHandleTypes`,
  and does UCX's per-allocation posix-fd path (`cuMemRetainAllocationHandle`) ever succeed on
  cudaMallocAsync pool suballocations, or strictly VMM memory? Decides whether the M6.5
  direct-registration experiment is even testable on this box.
- Does UCX cuda_ipc lane selection require the sender's *local* memh to be IPC-capable for a
  put, or only the remote rkey? Pack-into-send-lease makes it moot; bounds any future
  send-from-pool optimization.
- Which UCX version the nixl v1.3.2 install bundles (README says "tested with UCX 1.22.x"); the
  mempool dispatch analysis was against UCX master's `cuda_ipc_md.c`.
- NO_IPC staged-copy bandwidth on the L4 vs cuda_ipc arena D2D — the number that sizes the arena
  and justifies the tier.
- `cudf::pack` gather cost on demo-sized batches vs the per-column converter walk — measure
  before M6.2 freezes the wire (the registration argument already forces staging regardless).
- cuCascade PR #148's close reason (layering rejection vs rework) — determines whether a
  zero-copy sink steal can ever replace deep-copy-under-credit.
- `mr_factory_fn` injection leaves the reservation adaptor's `pool_handle` nullptr
  (`memory_space.cpp:114-135`) — check what depends on it before the M6.5 experiment injects an
  exportable pool.

## 10. Provenance

Two deep-research passes. First (~636k tokens): four parallel reports (CN architecture, protocol
surface, GPU memory config, FE scheduling) plus an adversarial critique that re-verified the
load-bearing claims. Second (~496k tokens, for the nixl goal): four reports (upstream
exchange-design diff + doris prior art; nixl v1.3.2 API/backends/bindings; cuCascade data model
and lock semantics; cudf/rmm wire mechanics at the pinned 26.06 versions) plus an adversarial
verdict that self-verified the three most load-bearing claims (the non-exportable pool
construction, UCX's fabric-only mempool export, and `unpack`'s zero-allocation contract). The critique materially changed the plan twice: D4 (remote senders as result
fragments — the original drain-parked design was impossible against the FFI guard at
`sirius_ffi.cpp:541-546`) and M4 (per-query GC + cancel, from the leak analysis). Contract notes
inherited from new-exchange-design §10 and stated here as commitments: the stream-id routing
table is `finst_id + node_id + sender_id` on the wire with `node_id == engine stream_id`; the
demo is fenced to `new_planner_agg_stage = 1`; the bind catalog stays one-fragment-at-a-time per
connection (the single engine channel preserves this).