# Review guide — the `demo-multi-cn` stack

Eight commits (`d6cce3ae..3473a686` on base `55fd14bd`) plus one submodule commit
(`04cd3136` in `experimental/starrocks/starrocks`, branch `demo-multi-cn-proto`).
Together they take the single-process StarRocks-FE + Sirius-CN demo to a working
two-CN cluster on one GPU where a query's exchange hop crosses processes as GPU
VRAM over nixl/UCX `cuda_ipc` — no Arrow, no host serialization of data.

```bash
git log --oneline 55fd14bd..demo-multi-cn        # the stack
cd experimental/starrocks/starrocks && git show 04cd3136   # the proto commit
```

Live proof: `pixi run cluster2`, TPC-H Q6 over two byte-identical lineitem files →
`61567694.95020001` (single-node reference `61567694.95019999`; ulp difference is
summation order), with log lines `nixl bandwidth canary … gbps="67.3"` and
`transmitted batches via nixl … bytes=457856`. Reproduction steps: `TWO-CN-NIXL-DEMO.md`
and `experimental/starrocks/DEMO.md`.

---

## Suggested review order (differs from commit order)

1. **3f7a9756** — the C++ staging arena AND the C++ packed FFI. Everything sits on
   its two contracts (cudaMalloc-by-contract, lease/release semantics). Read the
   arena header comment first.
2. **83d68072** — the Rust bridge and `packed_hop_matches_relay_hop`, the
   value-equality proof for #1's export/push pair.
3. **ca774e84** — routing + dispatch worker + `Failed` result state. This is where
   the concurrency model changes; hold the "who executes what, on which thread"
   question the whole way through.
4. **Submodule 04cd3136** — 65 proto lines; read before dead333b so the wire shape
   is in your head.
5. **dead333b** — the big one. Review `send_fragment` and `push_remote_frame` side
   by side against the proto comments; then `run_fragment_inner`'s lease
   discipline; then the canary.
6. **afb8fbb3** — small, but only makes sense after you've seen ca774e84 move the
   receiver off the RPC path.
7. **d6cce3ae**, **34931de5** — independent enablers, reviewable in isolation any time.
8. **3473a686** — docs last, as a cross-check that the claims match the code you
   just read.

## Cross-cutting invariants to hold in mind

- **Loud-failure rule.** No path may degrade or drop silently: unrouteable
  destination, missing `brpc_server`, seq gap, frame-after-EOS, changed column
  names, push-after-EOS, unconfigured arena, exhausted arena, sub-floor bandwidth,
  fetch timeout — all named, specific errors. The two failure modes that *can't*
  error (pool-memory cuda_ipc degradation; the nixl-sys dlopen stub) each get a
  dedicated tripwire: the mandatory bandwidth canary, and
  `NIXL_NO_STUBS_FALLBACK=1` + the arena test's `cudaPointerGetAttributes` check.
- **Engine-thread confinement.** `SiriusContext` is `!Send`; every context touch —
  Run, StagingInfo/Lease/Release, ExportNext, DropParked — is a message to the one
  engine thread. The transport thread and dispatch worker mirror the same actor
  shape (nixl's Rust binding documents a multithreading deadlock caveat). Any
  review suggestion that "just calls the context directly" is wrong by construction.
- **Copy-out-on-arrival.** A received packed batch is deep-copied into ordinary
  pool memory before `push_packed` returns. That single property is what makes
  (a) immediate lease release safe, (b) the received batch spillable/accounted
  like any other batch, and (c) reset-at-zero arena reclamation viable. The
  `packed_hop` test's release-before-receiver-runs probe is the proof.
- **cudaMalloc-by-contract.** The arena is plain `cudaMalloc`, never pool /
  stream-ordered memory, and sits OUTSIDE both the cuCascade budget and
  `--gpu-memory-limit` — operators size `carve-out + arena + context` against the
  device (see the cluster2 comment in `experimental/starrocks/pixi.toml`). Anyone
  "optimizing" the arena onto the rmm pool reintroduces a ~220× silent bandwidth
  cliff (0.38 vs 85–90 GB/s, correct bytes, no error).
- **Lease released exactly once, even on failure** — enforced in three separate
  places: `export_packed`'s catch block (C++), `run_fragment`'s released-set sweep
  (engine thread), and `bandwidth_canary`'s unconditional local release. The one
  direction NOT covered is the peer-side lease on a failed send (open item, see
  "riskiest lines" #1).

---

## 1. `d6cce3ae` — feat(starrocks): per-instance GPU carve-outs via EngineConfig

**Purpose.** Without it, a second CN on the same GPU aborts at bring-up: the
engine's default config primes ~0.95× of TOTAL device memory. Adds CLI carve-out
flags that derive a minimal Sirius YAML (reservation == usage limit) and the
`cluster2` pixi task — the physical precondition for every later commit.

**Hunks that matter.**
- `experimental/starrocks/src/engine_settings.rs:29-71` (`derive_sirius_config_yaml`)
  — the invariant: `reservation_limit_fraction: 1.0` is emitted with *every* GPU
  carve-out (lines 52, 56), so the limit is the whole budget, not 100% of the
  device. The `gpu:` mapping is emitted only when a GPU limit was asked for — a
  host-only carve-out must never implicitly reserve the device. Byte strings pass
  through verbatim; the C++ `parse_bytes` is the single authoritative parser.
- `experimental/starrocks/src/main.rs:59-64` (clap `conflicts_with_all`) +
  `main.rs:215-247` (`EngineConfig::resolve`) — `--sirius-config` hard-conflicts
  with the memory flags rather than silently winning; the derived YAML is written
  to `<engine_dir>/derived-sirius-config.yaml` and fed through the *existing*
  `from_config_file` path (zero C++ changes).
- `experimental/starrocks/src/main.rs:256-310` (`ensure_gpu_unclaimed`) — fail-fast
  guardrail: with no config and no carve-out, `nvidia-smi --query-compute-apps` is
  consulted BEFORE priming; if another compute process holds the device, refuse
  with a message quoting the exact rmm abort. Missing/failing `nvidia-smi` only
  warns (rmm remains the backstop) — warn-and-proceed, never silently skip.
- `experimental/starrocks/src/engine.rs:132-160` (`configure_engine_environment`)
  — `SIRIUS_LOG_DIR` and `CUDA_VISIBLE_DEVICES` set in the pre-engine-thread
  window; an operator-exported `CUDA_VISIBLE_DEVICES` wins and an ignored
  `--gpu-device` is *warned about*, not dropped.
- `experimental/starrocks/pixi.toml` `cluster2` task — CN2 shifts all five
  advertised ports (heartbeat 9052 / thrift 9062 / brpc 8062 / http 8042 /
  starlet 9072); both CNs at 8 GiB GPU / 12 GiB host with per-instance
  `.cn1`/`.cn2` dirs.

**Questions to ask.**
1. *Is `unsafe { std::env::set_var }` actually safe here?* `main` is
   `#[tokio::main]`, so runtime worker threads already exist when
   `SiriusEngine::start` runs. Safety rests on the convention (inherited from the
   pre-existing `configure_duckdb_extensions` precedent) that nothing reads env
   concurrently at that point — by discipline, not enforcement. Open note.
2. *Can the Rust shape-validator accept a string the C++ `parse_bytes` rejects?*
   `parse_byte_size` (`main.rs:94`) accepts e.g. `"8 GiB"` (internal whitespace).
   A mismatch fails only at engine bring-up — still loud, but later than the CLI.
   The `full_document_snapshot` test pins the YAML *keys*, not the value grammar.
3. *TOCTOU on the preflight?* Two default-config CNs starting simultaneously both
   pass `ensure_gpu_unclaimed` and one still dies in rmm. Answered by the code's
   own contract: the preflight is an ergonomics layer; the rmm abort remains the
   backstop (`main.rs:256` doc comment).

**Verify.** `cd experimental/starrocks && pixi run cn-test-no-engine`.
Load-bearing tests: `full_document_snapshot`, `host_only_omits_gpu_mapping`,
`reservation_limit_fraction_is_always_one`,
`sirius_config_conflicts_with_each_memory_flag`, `byte_size_validator_*`.
Live: `pixi run cluster2`, confirm ~8388 MiB per CN in `nvidia-smi`.

**Coupling.** Depends on nothing in the stack. Depended on by ca774e84/dead333b
(two live CNs to route between), dead333b (extends `cluster2` env), 3473a686.

---

## 2. `34931de5` — feat(starrocks): multi-file FILES() schema inference

**Purpose.** The FE resolves a FILES() table with ONE `get_file_schema` rpc
carrying every file; the CN rejected `ranges.len() > 1`, so any multi-file
FILES() failed at table-resolution time. Multi-file is what makes a two-CN scan
real — the FE round-robins whole files across nodes — so without this commit no
query can ever produce a cross-node exchange.

**Hunks that matter.**
- `experimental/starrocks/src/file_schema.rs:62-102` (`parquet_files_schema`) —
  fail-closed contract: schema inferred from the FIRST file, every other file must
  agree on column *name* (ASCII case-insensitive, matching StarRocks resolution;
  first file wins the spelling) and *type*; positional fields deliberately not
  compared. Disagreement is a loud error naming both files and the column. This
  intentionally diverges from the native scanner's sample-and-promote because this
  scan reads EVERY file with the inferred schema.
- `experimental/starrocks/src/compute_node_service.rs:624-651`
  (`file_schema_from_attachment`) — every range's format is now validated
  (previously only `ranges[0]`); the format error names the offending path.
- Test `get_file_schema_attachment_infers_across_multiple_ranges` — drives a real
  two-range binary-thrift `TGetFileSchemaRequest` through the attachment handler,
  pinning the FE's actual wire shape.

**Questions to ask.**
1. *Type equality too strict?* Whole-proto `PartialEq` — logically-equal but
   differently-encoded types from different parquet writers would fail. Exactly
   right for the demo's byte-identical files, and failing closed is the stated
   contract; real heterogeneous datasets will hit it. By design; note it.
2. *N sequential file opens on the RPC path?* Serial, one `parquet_file_schema`
   await per file. Deliberate simplicity; fine at demo scale.
3. *Does case-insensitive acceptance lose information downstream?* No — binding is
   positional after resolution and the first file's spelling is what the FE sees
   (test `multi_file_accepts_case_differing_column_names`).

**Verify.** `pixi run cn-test-no-engine`. Load-bearing:
`multi_file_rejects_column_count_mismatch` / `..name_mismatch` /
`..type_mismatch`, `multi_file_missing_file_error_names_its_path`,
`get_file_schema_attachment_rejects_non_parquet_range_by_path`.

**Coupling.** Independent of the routing/transport chain. Required by the
end-to-end demo (3473a686's two-file Q6); no code depends on it.

---

## 3. `ca774e84` — feat(starrocks): route exchange senders by destination address

**Purpose.** The CN never looked at where a destination LIVES — it always
rendezvoused in-process, which with two CNs is a *silent hang* (receiver on CN-B
waits for a sender that landed on CN-A until FE query_timeout). This commit
classifies destinations against the CN's own exchange identity, makes remote a
loud error (the seam dead333b fills), and moves receiver execution off the RPC
thread onto a dispatch worker — mandatory once the last sender can live on
another node.

**Hunks that matter.**
- `experimental/starrocks/src/compute_node_service.rs:70-72`
  (`ExchangeIdentity::matches`) — hostname AND port equality, the stock BE's
  locality rule; the port comparison is exactly what makes two CNs on one host see
  each other as remote. Identity is built from `advertise_host` + `brpc_port`.
- `compute_node_service.rs:581-599` (`route_destination`) + the `DestinationRoute`
  match — two invariants: (a) a destination without `brpc_server` is a malformed
  dispatch, never an implicit "local" (the FE always sets it —
  ExecutionDAG.java:560); (b) routing is decided BEFORE the sender runs, so a
  remote placement fails with zero GPU work (a test asserts `executor.calls == 0`).
- `compute_node_service.rs:127-171` (ServiceCore + `dispatch`) and `:175-186`
  (`dispatch_worker`) — receiver execution leaves the RPC thread: ready fragments
  flow through an mpsc to one dedicated std::thread. The worker holds
  `Arc<ServiceCore>` but no `ready_fragments` sender, so the channel closes and
  the worker exits when the last service clone drops — engine teardown stays
  ordered behind the servers. The worker chases receiver→sender→next-receiver
  chains inline.
- `compute_node_service.rs:334-375` (`run_ready_fragment`) +
  `result_store.rs:61,104` (`FragmentState::Failed`, `ResultStore::fail`) — the
  failure route for errors that happen after every RPC already returned OK: a
  dispatched result fragment's failure lands in its reserved result entry and
  `fetch_data` surfaces it as INTERNAL_ERROR naming the cause; the failure
  *sticks* across polls. An intermediate receiver's failure is parked under its
  own id and logged loudly — full cancellation propagation is deliberately out of
  scope, so the downstream result fragment still waits out the FE timeout.
- `experimental/starrocks/src/local_exchange.rs:27` — `SenderOutput` becomes the
  single-variant enum `SenderSource::LocalParked`, so dead333b's `Remote` variant
  forces every destructuring site to handle it at compile time.

**Questions to ask.**
1. *Race between a receiver registering and the last sender pushing on two RPC
   threads?* Both paths funnel through `LocalExchange`'s one mutex and
   `take_ready`, which hands out the `ReadyFragment` exactly once; duplicate
   senders are a loud error (`push_sender`, `local_exchange.rs:108-121`).
2. *One dispatch thread serializes every query's receivers — can it deadlock?*
   Not on itself (chains run inline; the worker never blocks on another dispatched
   fragment), but one slow receiver head-of-line-blocks all queries. Acceptable
   single-stream demo scope; note for multi-query work.
3. *Known regression planted here:* moving the receiver off the RPC path makes the
   FE's first `fetch_data` poll race receiver execution. This commit's
   `ResultStore::Waiting` reply (not-ready, `eos=false`) consumes an FE
   packet-sequence slot — the live bug afb8fbb3 fixes. Reading this commit alone,
   ask "what does the FE do with the not-ready reply?"; the answer (desync,
   THRIFT_RPC_ERROR) arrives two commits later.

**Verify.** `pixi run cn-test-no-engine`. Load-bearing:
`exchange_identity_requires_host_and_port_equality`,
`data_stream_sink_to_remote_destination_is_a_loud_error` (proves routing precedes
GPU work), `data_stream_sink_destination_without_brpc_server_is_a_loud_error`,
`sender_rpc_returns_before_the_dispatched_receiver_executes`,
`dispatched_receiver_failure_surfaces_through_fetch_data`,
`failed_fragment_reports_its_cause_on_every_poll`.

**Coupling.** Depends on d6cce3ae only operationally. Depended on by dead333b
(fills the `Remote` arm, extends `SenderSource`, reuses the dispatch worker and
`ResultStore::fail`) and afb8fbb3 (fixes the polling regression this introduces).

---

## 4. `3f7a9756` — feat(exec): cudaMalloc exchange staging arena (+ the C++ half of the packed FFI)

**Purpose.** Cross-process GPU transfer needs IPC-capable memory, and the engine's
rmm `cudaMallocAsync` pool memory over UCX `cuda_ipc` does not fail — it silently
degrades ~220× (0.38 vs 85–90 GB/s, correct bytes, no error). This commit adds the
one plain-`cudaMalloc` staging region both transfer directions register, plus —
note, in THIS commit, not 83d68072 — the C++ `Fragment::export_packed` /
`push_packed` / `close_input` and `Context::staging_*` FFI methods.

**Hunks that matter.**
- `src/include/exec/exchange_staging_arena.hpp:24-49` (class comment +
  `kAlignment`/`kCapacityEnvVar`) — the two contracts: plain `cudaMalloc`, never
  pool memory (the entire reason the class exists), and *unset
  `SIRIUS_EXCHANGE_STAGING_BYTES` = no arena = every staging call errors loudly*
  instead of a silent slow path. The arena can never move (copy/move deleted)
  because the base pointer is handed to transport registration.
- `src/exec/exchange_staging_arena.cpp:56-100` (`lease`/`release`) — bump
  allocation under a mutex; explicit lease/release because leases cross the FFI
  (no RAII); zero-length leases rejected (they would alias the next offset and
  break release-by-offset); release of a non-outstanding offset is a loud "double
  release?" error; the bump head resets ONLY when outstanding hits zero — the
  reclamation model *relies on leases being short-lived* (copy-out-on-arrival).
- `src/sirius_ffi.cpp:555-627` (`Fragment::export_packed`) — the send path's
  ordering discipline: pull → `to_read_only()` shared lock
  (residency + immutability + spill-exclusion held for the whole pack) →
  `cudaStreamWaitEvent` on the batch's writer event (STREAM-LINEAGE) →
  `cudf::chunked_pack` gathers directly into the lease (the staging copy IS the
  pack's gather; no extra copy) → `build_metadata` → `stream.synchronize()` before
  returning so the caller may transmit immediately. Lease sized
  `total + kPackChunkBytes` because every `next()` span must be a full chunk long.
  The `catch (...) { arena.release(lease_offset); throw; }` keeps "lease released
  exactly once even on failure". Non-GPU-resident (spilled) batches are rejected
  loudly, not exported wrong.
- `src/sirius_ffi.cpp:629-675` (`Fragment::push_packed`) — the receive mirror:
  bounds-check the lease range against capacity, `cudf::unpack` (a
  zero-allocation view over the lease), then a *deep copy into ordinary pool
  memory* + `stream.synchronize()` — copy-out-on-arrival, making the received
  batch spillable/accounted and the lease immediately releasable. A push after
  end-of-stream throws.
- `src/sirius_ffi.cpp:169-171` (bring-up ordering in `context_state::bring_up`) —
  the arena's `cudaMalloc` happens AFTER engine bring-up so it comes out of the
  headroom the operator left beside the pool budget; the arena sits OUTSIDE the
  cuCascade budget by explicit contract.

**Questions to ask.**
1. *Stale comment at `sirius_ffi.cpp:553-555`:* "1 MiB is cudf's minimum" while
   `kPackChunkBytes = 8u << 20` (8 MiB). Not a bug (8 MiB > minimum) but the
   comment misstates the constant. Confirmed nit — fix the wording.
2. *A long-lived lease pins the whole arena.* Reset-at-zero reclamation means one
   stuck lease makes the bump head grow monotonically until exhaustion; the
   exhaustion error is loud and names requested/free/capacity/outstanding
   (`exchange_staging_arena.cpp:66-75`), but recovery is process restart.
   Deliberate design for short-lived leases; `outstanding()`/`high_water()` exist
   for exactly this. Residual risk noted.
3. *If `stream.synchronize()` or `build_metadata` throws after the batch was
   pulled, the batch is gone.* The lease is released (catch block) but the pulled
   batch was consumed destructively. Any such error propagates and fails the query
   loudly, so no wrong answer is possible; data loss inside an already-failed
   query is acceptable. Also `device_id=0` is hardcoded in `push_packed`'s
   memory-space lookup — correct because `CUDA_VISIBLE_DEVICES` pins one device
   per process (d6cce3ae), but it couples the two commits' assumptions.

**Verify.** `pixi run make test`, or directly
`pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[staging_arena]"`.
Load-bearing: ARENA-1 (reset-at-zero + no-free-list: releasing SOME leases must
not move the head), ARENA-2 (`cudaPointerGetAttributes` proves
`cudaMemoryTypeDevice` — the anti-pool tripwire), ARENA-3 (exhaustion text),
ARENA-4 (double release). The FFI pair's behavioral test lives in 83d68072 (GPU).
Commit message reports the full C++ suite at 2179 cases / 0 failures.

**Coupling.** Depends on nothing in the stack (pure engine-side). Depended on by
83d68072 (binds these exact C++ symbols) and dead333b (registers the arena with
nixl; every WRITE lands in it). Commit-boundary note: 83d68072 is titled "the
packed FFI pair" but the C++ implementation of the pair is HERE; 83d68072 is the
Rust bridge + proof.

---

## 5. `83d68072` — feat(ffi): export_packed / push_packed — a device-resident fragment boundary (Rust side)

**Purpose.** The FFI could previously move a fragment's output only in-process
(`relay_from`) or as host Arrow (`result_to_arrow`); neither can feed a
cross-process GPU transport. This commit exposes the C++ packed pair through cxx
to safe Rust and proves, on the GPU, that the packed hop is value-equal to the
proven relay hop.

**Hunks that matter.**
- `rust/crates/sirius-sys/src/lib.rs:85-114` (bridge decls) — `push_packed` is the
  one `unsafe fn` in the bridge, with the safety contract spelled out (metadata
  pointer/len must outlive the call); `export_packed` returns
  `UniquePtr<CxxVector<u8>>` where null = drained, mapped to `Ok(None)` in the
  safe layer — drained is a state, not an error.
- `rust/crates/sirius/src/lib.rs:269-315` (safe `export_packed`/`push_packed`) —
  the safe wrapper discharges the unsafe contract by borrowing `batch.metadata`
  for the call (SAFETY comment at `lib.rs:295-306`); metadata is copied out of the
  C++ vector into an owned `Vec<u8>` so `PackedBatch` is a plain, wire-shippable
  value type (`lib.rs:323-331`).
- `rust/crates/sirius/src/lib.rs:586-680` (`packed_hop_matches_relay_hop`) — the
  decisive test: identical two-fragment plan run twice, relay hop vs packed hop,
  `assert_eq!(rows(relay), rows(packed))`; plus, in the same test: drained export
  is `None`, push-after-`close_input` errors with "already ended", and *every
  lease is released before the receiver runs* with a probe lease asserting the
  bump head reset to 0 — which can only pass if `push_packed` really copied the
  data out of the lease (the copy-out-on-arrival proof).

**Questions to ask.**
1. *Is `staging_lease(&self)` on an `&self` handle safe against engine
   thread-confinement?* It goes through `self.inner.borrow_mut()` (a RefCell) —
   single-threaded by construction in this crate; the multi-threaded discipline is
   enforced one layer up (dead333b's engine-thread actor). Answered.
2. *Env-var choreography in the GPU test* (`set_var`/`remove_var` for
   `SIRIUS_EXCHANGE_STAGING_BYTES` under `GPU_CONTEXT_LOCK`) — a panic mid-test
   leaks the env var into later tests in the same process. Low risk; acceptable.
3. *`PackedBatch.offset/len` are trusted, unvalidated inputs to `push_packed`.*
   The C++ side bounds-checks against arena capacity but cannot verify the offset
   is a live lease or that the metadata matches the payload — garbage metadata
   over a valid range is undefined at the `cudf::unpack` level. Within this stack
   the only producers are `export_packed` and a nixl WRITE into a granted lease;
   the trust boundary is the CN's own brpc port. Open hardening note (a
   malicious/buggy peer can crash the receiver).

**Verify.** GPU required:
`cargo test -p sirius packed_hop_matches_relay_hop` from `rust/` with
`LD_LIBRARY_PATH=build/release/extension/sirius`. This test is the only
value-equality proof of the packed path in the whole stack below the live Q6.

**Coupling.** Hard dependency on 3f7a9756 (will not link without its C++
symbols). Depended on by dead333b (`engine.rs` calls
`export_packed`/`push_packed`/`close_input` via engine-thread requests).

---

## 6. `dead333b` — feat(starrocks): carry the exchange hop over nixl (+ submodule `04cd3136`)

**Purpose.** Fills ca774e84's `Remote` arm: a sender whose destination is another
CN drains its parked output GPU-to-GPU through nixl/UCX `cuda_ipc` via the staging
arenas — no Arrow, no host data serialization; only control frames on brpc.
Without it a two-CN placement is a (loud) query error and the demo does not exist.

**Proto (submodule commit `04cd3136`, branch `demo-multi-cn-proto`; gitlink
updated here).** `gensrc/proto/internal_service.proto:652-711 + 866-869`: three
CN-only rpcs on PInternalService —
- `exchange_nixl_md` — agent metadata handshake, idempotent;
- `request_staging_lease` — receiver-granted lease → `remote_addr` = arena base +
  offset, the WRITE target;
- `transmit_packed` — per-batch signaling: finst_id/node_id/sender_id, per-sender
  `seq`, `eos`, lease offset/length, `column_names` repeated on every frame
  (cudf pack metadata carries no names), `canary` flag.

Stock FE/BE never call these.

**Hunks that matter.**
- `experimental/starrocks/src/nixl_transport.rs:480-553`
  (`TransportState::send_fragment`) — the sender flow and its lease discipline:
  `export_packed_next` (local lease) → `rpc_request_lease` (peer lease) →
  `write_and_wait` (WRITE local→remote, polled to DONE with a 30 s bound) →
  `rpc_transmit` (metadata in the brpc attachment) → release the LOCAL lease →
  seq++; final frame is `eos` with the next seq; then `drop_parked(slot)`.
  WRITE-based with receiver-granted leases so *every lease's lifetime stays
  process-local* — no cross-node ack protocol. Zero-payload batches take no lease
  and no WRITE.
- `experimental/starrocks/src/nixl_transport.rs:398-478` (`ensure_session` +
  `bandwidth_canary`) — the mandatory first-contact canary: warmed 16 MiB WRITE,
  timed; below `CANARY_FLOOR_GBPS = 2.0` (`nixl_transport.rs:182`) the tier
  REFUSES loudly. This guards the one failure nothing else can catch: pool memory
  over cuda_ipc transfers correct bytes ~220× slower with no error. The local
  canary lease is released on both success and failure. Agent named
  `{advertise_host}:{brpc_port}` — per-host naming would collide for two CNs on
  one host. Names are cross-checked fail-closed on both sides.
- `experimental/starrocks/src/local_exchange.rs:159-241` (`push_remote_frame`) +
  `:59-64` (`is_complete`) + `:94` (`remote_seq`) — the receiver-side rendezvous
  protocol: duplicate seq (below expected) drops idempotently (brpc
  reconnect-retry can replay a frame); a gap is "a frame was lost", a loud error —
  silently dropping rows is the subsystem's cardinal sin; frame-after-eos,
  missing/changed names, empty metadata, and remote-colliding-with-local-sender
  are all loud errors. A remote sender counts toward readiness only once
  `closed`; EOS and sender-set completion stay on the existing rendezvous — one
  source of truth.
- `experimental/starrocks/src/engine.rs:224-317` (`EngineRequest` variants + the
  loop) and `:299-360 + 421-460` (`run_fragment` / `run_fragment_inner`
  remote-input push) — engine-thread confinement: the context is `!Send`, so
  StagingInfo/Lease/Release/ExportNext/DropParked all funnel through the request
  channel; the transport thread never touches the context. The remote-input push
  loop enforces "lease released exactly once even on failure": each lease released
  immediately after its successful `push_packed`, released offsets recorded in a
  set, and an error sweep releases only the not-yet-released, non-sentinel
  (len>0) leases. A failed transmit best-effort-drops the parked sender so a dead
  query does not pin GPU output until restart.
- `experimental/starrocks/src/compute_node_service.rs:448-509 + 760-790 + 842-846`
  — the service seam: `handle_transmit_packed` validates every required field
  loudly and routes canary frames straight to `staging_release` without touching
  the rendezvous; the Remote route arm parks exactly like the local path then
  blocks the (already-blocking) RPC thread on `transport.send_fragment`, so any
  failure fails the sender's dispatch and the FE sees it.
- `experimental/starrocks/src/prpc_client.rs:63-118 + 133-176`
  (`PrpcClient::call` / `try_call`) — the CN could serve but never call; this is
  the minimal blocking client: one request in flight, correlation-id checked
  (mismatch = framing lost sync = untrusted connection), 60 s timeouts, transport
  failures on a *cached* connection retried once over a fresh one (safe because
  the receiver's seq protocol makes duplicate delivery idempotent), brpc-level
  rejections never retried.
- Build hygiene (`Cargo.toml`, `pixi.toml`): `nixl-transport` is a default
  feature implying `sirius-engine`; `NIXL_NO_STUBS_FALLBACK=1` is mandatory
  (nixl-sys otherwise silently builds a dlopen stub — the same silent-degradation
  class the canary exists for); cargo's linker pinned to `/usr/bin/gcc` (conda
  cross-gcc can't resolve libnixl's glibc-2.38 symbols); the CI path
  `--no-default-features` stays nixl-free.

**Questions to ask.**
1. *Receiver-side lease leak on a failed send.* If `write_and_wait` or a later
   step fails AFTER `rpc_request_lease` succeeded, the PEER's lease is never
   released — there is no remote-release rpc except delivering the frame (or
   canary). Because the arena reclaims only at zero-outstanding, one such leak
   permanently pins the peer's bump head until process restart; subsequent
   exhaustion errors are loud but the root cause is a hop away. The commit
   message's "no cross-node ack protocol to get wrong" trades this off
   consciously. OPEN — real, bounded by demo scope, surfaced by
   `outstanding()`/exhaustion text.
2. *Late replayed frame after the receiver already left the rendezvous.*
   `take_ready` removes both the sources entry and the `remote_seq` entries
   (`local_exchange.rs:300`). A transport-level retry arriving after that
   re-enters with `expected_seq=0`, matches `seq==0`, and creates a phantom
   `Remote` source under a consumed exchange key — orphaned bookkeeping
   referencing an already-released lease offset. It can never execute, so it is a
   memory-bookkeeping wart, not a correctness bug — but the duplicate-drop
   comment's "leaks nothing" claim doesn't hold in that window. OPEN (narrow:
   requires a reply-lost retry racing receiver completion).
3. *Cross-CN blocking chain.* `request_staging_lease` on the peer queues behind
   the peer's engine thread, and `send_fragment` occupies the sender's single
   transport thread while blocking on the peer. Two CNs simultaneously sending to
   each other: A's transport waits on B's engine; B's transport waits on A's
   engine; the engine threads never block on the network, so this is latency
   coupling, not deadlock — bounded by the 60 s `REPLY_TIMEOUT` / 30 s
   `XFER_TIMEOUT`. Caveat: a >60 s engine-thread fragment on the peer fails the
   sender's query spuriously. (At N>2 the *transport-to-transport* first-contact
   wait DOES become a real deadlock — see ROADMAP-8CN-TPCH.md §DGX.)

**Verify.**
- CI seam (no GPU/libnixl): `pixi run cn-test-no-engine` — 20 new tests.
  Load-bearing: `transmit_packed_frames_feed_a_dispatched_receiver` (full receiver
  half minus the device WRITE), `transmit_packed_sequence_gap_is_an_internal_error`,
  `remote_frame_after_eos_is_a_loud_error`, `remote_names_must_match_across_frames`,
  `mixed_local_and_remote_sender_set`,
  `remote_transmit_failure_fails_the_sender_dispatch`,
  `data_stream_sink_to_remote_destination_hands_the_parked_output_to_the_transport`,
  `transmit_packed_canary_releases_the_lease_without_touching_the_rendezvous`,
  prpc_client `client_round_trips_a_method_call_against_the_real_dispatch` /
  `client_reconnects_after_the_peer_drops_the_connection`.
- GPU: `pixi run cn-test` adds `engine_pushes_staged_remote_batches` (the
  engine-actor mirror of the packed-hop proof, incl. exactly-once lease release
  and double-`drop_parked` erroring).
- GPU + libnixl smoke: source `tools/nvda_nixl/ENV.sh`, then
  `cargo test -p sirius-starrocks-cn nixl_cross_agent_write_between_arena_leases -- --ignored`
  (agent bring-up, VRAM registration, md handshake, cross-agent WRITE at
  ~84 GB/s; does NOT verify bytes — value verification is the live Q6).
- Proto: `cd experimental/starrocks/starrocks && git show 04cd3136`.

**Coupling.** Depends on ca774e84 (Remote seam, `SenderSource`, dispatch worker),
3f7a9756 + 83d68072 (arena + packed FFI), d6cce3ae (cluster2, carve-outs), and
the submodule gitlink 04cd3136. Depended on by afb8fbb3 (the first real two-CN
run this enabled exposed the fetch_data desync) and 3473a686.

---

## 7. `afb8fbb3` — fix(starrocks): fetch_data long-polls — a not-ready reply desyncs the FE

**Purpose.** Found live on the first two-CN run: the hop crossed correctly, yet
the FE cancelled with THRIFT_RPC_ERROR / "receive packet failed, expect=1,
receive=0". The FE's ResultReceiver counts EVERY fetch_data reply against a
packet sequence; ca774e84's dispatch thread made the previously-unreachable
"not-ready empty reply" the common first poll, consuming sequence 0 so the real
rows arrived stale. Without this fix every dispatched-receiver query fails at the
last step.

**Hunks that matter.**
- `experimental/starrocks/src/result_store.rs:127-158` (`wait_ready`) + `:95-99`
  (the `ready` condvar) — the invariant, stated in the field comment: an empty
  reply is NOT harmless; block on the condvar until the fragment leaves `Waiting`
  (rows buffered or failure recorded — both `insert` and `fail` `notify_all`).
  A timeout is a *loud `Failed`* naming the fragment ("its exchange senders may
  have stalled"), never a sequence-consuming empty.
- `experimental/starrocks/src/compute_node_service.rs:283` (handler) — the wait
  runs in `spawn_blocking` with a 600 s bound, off the current-thread brpc
  runtime, so a parked poll never starves other RPCs.
- Test change: the obsolete intermediate not-ready probe removed from the
  self-exchange test — an assertion that the old reply shape is gone from the
  protocol.

**Questions to ask.**
1. *Can the FE poll before the entry exists?* No — `results.reserve(id)` happens
   on the exec_plan_fragment path before that RPC returns OK, and the FE only
   polls after dispatch succeeded; an id with no entry still returns the immediate
   "no buffered result" error (unknown ≠ wait).
2. *600 s server hold vs FE-side timeouts.* If the FE's brpc-channel timeout for
   fetch_data is shorter, a stalled sender produces an FE-side rpc timeout before
   the CN's loud message. Still fail-loud on both sides; which error the operator
   sees is timeout-ordering dependent. Open (FE constant unverified; cosmetic).
3. *Window between condvar wake and `take_next`.* A second concurrent poller for
   the same id could consume the rows first. The FE has exactly one
   ResultReceiver per fragment, so single-poller in practice; note if fetch ever
   becomes concurrent.

**Verify.** `pixi run cn-test-no-engine`; load-bearing:
`wait_ready_blocks_until_rows_arrive` (producer on another thread — the
dispatch-worker shape) and `wait_ready_times_out_loudly_instead_of_replying_not_ready`.
Live gate: the cluster2 Q6 in DEMO.md returning `61567694.95020001`.

**Coupling.** Fixes the regression planted by ca774e84 and exposed by dead333b's
first live run. Depended on by 3473a686 (the documented Q6 only works with it).

---

## 8. `3473a686` — docs(starrocks): the two-CN demo crosses the exchange hop over nixl

**Purpose.** Adds the "Two compute nodes, one GPU" section to
`experimental/starrocks/DEMO.md` (+49 lines): cluster2, the byte-identical
two-file layout (so the FE's min-load selector places one whole file per CN
deterministically), the finish-line Q6 with log evidence, the ulp difference and
why, and the honest boundary.

**Accuracy checks (docs, not code).**
- The four log lines match the actual `info!` call sites: `nixl bandwidth canary`
  (`nixl_transport.rs:466`), `transmitted batches via nixl`
  (`nixl_transport.rs:546`), `relayed native batches across a fragment boundary`
  (pre-existing engine path), `received remote batches` (`engine.rs:452`). The
  cross-node fan-in claim (sender 0 in-process, sender 1 over nixl) matches the
  `mixed_local_and_remote_sender_set` semantics.
- The stated limitation is real and loud: multi-destination senders (GROUP BY on
  two CNs = hash shuffle) fail with "a data stream sink with 2 destinations needs
  partitioned streaming" — the two-CN surface today is Q6-class scalar aggregation.
- The memory arithmetic ("~8.9 GiB each — pool + arena + context") is consistent
  with d6cce3ae's 8 GiB carve-out + dead333b's 512 MiB arena outside
  `--gpu-memory-limit`.

**Known doc limitations.** Hardcodes the machine-specific
`file:///home/ubuntu/git/sirius/scratch/tpch_sf1/lineitem_multi/*.parquet` path;
the reference value and canary figure are point-in-time measurements no test pins.
Acceptable for a demo doc.

**Verify.** Reproduce per DEMO.md: `pixi run cluster2`, run the quoted Q6, check
the four log lines and the sum. No automated gate.

---

## The 5 riskiest lines in the stack, ranked

1. **`experimental/starrocks/src/nixl_transport.rs:485-492` (dead333b) —
   `rpc_request_lease` → `write_and_wait` with no remote-release on failure.**
   A failed WRITE after a granted peer lease permanently pins the peer's arena
   (reset-at-zero never triggers again) until restart. The only
   unbounded-blast-radius failure path in the stack.
2. **`src/sirius_ffi.cpp:602-612` (3f7a9756) — the chunked_pack loop:
   `arena.lease(total + kPackChunkBytes)` and
   `packer->next(span(lease + written, kPackChunkBytes))`.** The packed wire
   format's integrity rests on "every next() span must be exactly one chunk long,
   final span may overrun the payload by < 1 chunk into the slack". A cudf
   behavior change here corrupts payloads with no local error; the
   `written != total` check and the relay-equality test are the only nets. (Also
   carries the stale "1 MiB is cudf's minimum" comment.)
3. **`experimental/starrocks/src/local_exchange.rs:182-193` (dead333b) — the seq
   accept/drop/gap decision.** `seq < expected` drop is what makes the prpc
   reconnect-retry safe; `seq > expected` error is what makes lost frames
   impossible to silently absorb. Any weakening either double-ingests a batch or
   drops rows — and the post-`take_ready` replay window already lives at this
   boundary.
4. **`experimental/starrocks/src/result_store.rs:127-151` (afb8fbb3) —
   `wait_ready`'s Waiting-loop and loud timeout.** One empty reply from this
   function desyncs the FE's packet counter and kills the query with an error
   that points nowhere near the cause. The condvar wiring (`insert`/`fail` both
   notify) is the correctness of the entire FE-facing result path.
5. **`experimental/starrocks/src/engine_settings.rs:52,56` (d6cce3ae) —
   `reservation_limit_fraction: 1.0`.** Delete or mistune this one line and the
   carve-out silently reverts to reserving against the whole device — two CNs
   then OOM each other at bring-up, i.e. the exact failure the commit exists to
   prevent, reintroduced through config derivation.

## Open questions / accepted gaps

- **dead333b:** peer-side lease leak on failed send (riskiest line #1) — accepted
  demo scope, or does it need a release-lease rpc / lease TTL before merge?
- **dead333b:** a `transmit_packed` retry arriving after `take_ready` cleared
  `remote_seq` re-registers as seq 0 and creates an orphaned phantom source —
  bookkeeping leak only. Worth keying `remote_seq` retention to query lifetime?
- **afb8fbb3:** which side times out first, the CN's 600 s `wait_ready` or the
  FE's fetch_data brpc timeout — unverified FE constant; affects only which loud
  error the operator sees.
- **d6cce3ae:** `unsafe std::env::set_var` after `#[tokio::main]` spawned runtime
  threads — sound by convention, not enforcement.
- **d6cce3ae:** Rust `parse_byte_size` accepts shapes (`"8 GiB"`) whose acceptance
  by the authoritative C++ `parse_bytes` is unverified — a mismatch surfaces only
  at engine bring-up.
- **3f7a9756:** stale "1 MiB is cudf's minimum" comment at
  `src/sirius_ffi.cpp:553-555` (constant is 8 MiB) — wording fix.
- **83d68072:** `push_packed` trusts caller-supplied offset/len/metadata beyond a
  capacity bounds-check — a buggy/malicious peer on the brpc port can feed
  garbage metadata into `cudf::unpack`. Hardening note for anything past the demo.
  *Partially narrowed by `64977ebb` (below): the unpacked table's column count and
  cudf types are now validated against the declared stream schema before the deep
  copy — garbage metadata still reaches `cudf::unpack` itself, but a schema lie no
  longer reaches the engine.*

---

# Part 2 — the two-phase aggregation stack (`64977ebb..11625add`, 2026-08-05)

Retires the `new_planner_agg_stage = 1` workaround for scalar aggregation: the
FE's **default** two-phase plan (partial agg per scan fragment → gather →
merge) now translates and runs. Plan/decision record: TWO-PHASE-AGG-PLAN.md.
Review order = commit order; the stack was built engine-truth-first.

## Suggested lens

Two decisions carry the whole stack; read them before any diff:

1. **No FFI phase marker.** The engine's Substrait consumer ignores
   `AggregateFunction.phase` and `Measure.output_type`
   (`substrait/src/from_substrait.cpp:707-783`), so the phase is resolved
   entirely in the translator: the merge node is a *plain aggregate with
   substituted functions* (sum→sum, count→**sum**, min→min, max→max — the
   engine's own merge table, `gpu_merge_impl.cpp`), and the engine's
   auto-inserted MERGE_AGGREGATE wrap does the cross-CN reduce. "Substitute
   first, label second": the emitted plan must be correct for a phase-ignoring
   consumer.
2. **The FE's intermediate slot type is wrong, not just opaque** (DECIMAL128
   declared where the wire column is FP64; VARBINARY for avg). One pure
   function (`partial_state::wire_type`) models the engine's real binding from
   FE-identical thrift inputs; both fragments derive their side of the exchange
   from it, and the engine *validates* the agreement at every hop.

## 9. `64977ebb` — feat(ffi): reject a hop whose batch schema disagrees with the declared stream

**Purpose.** The safety net everything later leans on: nothing compared a
batch's actual cudf types against the receiver's declared stream schema, so a
wrong declaration meant reinterpreted bits. Now `relay_from` checks the source
sink's logical types (metadata only, before any batch moves;
`streaming_fragment::sink_types()` is new) and `push_packed` checks the
unpacked `table_view` (before the deep copy).
**Skeptical questions.** Can a legitimate hop have unequal-but-compatible
types? (No — the declared schema is what the plan binds against; any mismatch
is a bug by definition.) Does the guard cost GPU work? (No — metadata on the
relay leg; the unpacked view is already in hand on the packed leg.)
**Verify.** `cargo test -p sirius --lib` — the two `*_rejects_a_mismatched_schema`
negatives; full `pixi run make test` (2181 passed / 1 skipped).

## 10. `c6c98bf8` — test(exec): partial aggregates merge to the one-shot answer

**Purpose.** Pins on GPU, before any translator work, the two behaviours the
design rests on: an ungrouped aggregate under a STREAMING_SINK emits its
single-row partial state, and the substituted merge reproduces the one-shot
answer. FRAG-6 (scalar, incl. an **empty-input sender**), FRAG-7 (grouped —
pinned now so unblocking grouped later is not an engine question), FRAG-8
(decimal min/max identity round-trip). These tests answered the plan's open
questions Q1-Q3 empirically (sum(int)/count are BIGINT on the wire; empty
senders don't hang; decimal min/max keeps its type).
**Verify.** `sirius_unittest "[streaming_fragment]"` — 8 cases, 140 assertions.

## 11. `fee8c5b7` — feat(starrocks): classify aggregation phases

**Purpose.** Replaces BOTH legacy guards atomically with
`agg_phase::classify(need_finalize × is_merge_agg)`. The atomicity is
load-bearing: the node-level tuple-id check is dead on new-optimizer plans, so
a merge node passed the node guard and only the expression-level rejection
prevented a silent double-aggregation. OneShot unchanged; Partial/Merge still
rejected here (precise messages); merge-serialize (3/4-phase DISTINCT) and
mixed-measure nodes permanently rejected.
**Riskiest review point.** Convince yourself no path reaches the measures loop
with an unclassified merge measure.

## 12. `42e1b23e` — feat(starrocks): the partial-state wire-type model

**Purpose.** `partial_state::wire_type` — the riskiest ~60 lines of the stack,
reviewed alone. sum(decimal)→FP64, sum(int)→I64, count→I64, min/max identity,
avg→loud error (cardinality change: one FE slot, two Sirius columns). Each row
cites the engine behaviour it mirrors and is pinned by commit 10's GPU tests.
**Skeptical question.** What happens if the engine's binding changes? (Commit
9's hop guards turn the drift into a loud error naming the column.)

## 13. `47cc2245` — feat(starrocks): translate the partial phase

**Purpose.** The Partial arm: plain measures over raw rows,
`phase=InitialToIntermediate` (advisory), measure output types from the model.
Grouped two-phase gets an explicit translator error ("needs partitioned
streaming output (#838)") because on a single CN the merge fragment has one
destination and the service-level guard would never fire — reachable-but-
untested is what the loud-failure rule forbids. DISTINCT and avg rejected.

## 14. `41af1387` — feat(starrocks): translate the merge phase + exchange override

**Purpose.** The stack's crux commit. A preorder pre-pass
(`merge_exchange_overrides`) finds every Merge-classified aggregation, requires
its direct child to be an EXCHANGE_NODE, and computes positional wire-type
overrides; `translate_exchange` rewrites the exchange `NamedStruct` before
anything derives from it (one `Type`, two consumers: ReadRel base_schema and
the engine stream declaration). The Merge arm substitutes functions (the single
most important line: `"count" => "sum"`), re-registers a merged count under the
arithmetic URN, and skips the decimal→FP64 argument cast (the state column is
already FP64).
**Skeptical questions.** Is "next preorder node" really the direct child?
(num_children==1 for aggregations; preorder puts the only child next.) Can an
override land on the wrong column? (Positions are `keys + measure_index` over
the same materialized-slot layout both the tuple and the sender's output use;
an out-of-range position errors.) What if a merge measure's argument is not a
bare SLOT_REF? (It resolves through the exchange row like any expression; the
type override is positional, not expression-dependent.)
**Verify.** `cargo test -p starrocks-plan-translator` — the substitution/override
positives, `merge_over_a_scan_is_rejected`, and
`two_phase_wire_types_agree_end_to_end`.

## 15. `11625add` — feat(starrocks): run the FE's default plan end to end

**Purpose.** Sharpened destination-guard text (behavior unchanged); DEMO.md
recipes drop the SET for scalar queries (GROUP BY keeps it, with the reason).
The live evidence: two-CN Q6 **without** the session variable →
`61567694.95020001`, EXPLAIN shows `update serialize → EXCHANGE → merge
finalize`, and the nixl hop carries **bytes=64** (one partial row) where the
one-phase plan shipped 457 KB. Single-CN: default plan still two-fragment,
grouped guard fires in the translator, `agg_stage=1` regression unchanged.
**Deviation from the plan doc:** no new service-level unit test for the guard
text — the guard's behavior is pre-existing and both refusal texts were
exercised live; noted here so the reviewer can ask for one if wanted.

## Riskiest lines added by this stack

1. **`partial_state.rs::wire_type` (42e1b23e)** — a hand-model of the engine's
   binding. Drift = loud hop error (thanks to 64977ebb), but the *model being
   wrong on day one* for an untested type would refuse valid queries. The
   allowlist is intentionally tiny.
2. **`node_translator.rs` merge substitution `"count" => "sum"` (41af1387)** —
   reverting it to `count` double-counts silently on every two-phase count.
   Pinned by FRAG-6/7 and the translator substitution test.
3. **The guard pairing (fee8c5b7)** — any future edit that relaxes the
   node-level classifier without keeping the measure flags in view reopens the
   silent double-aggregation hole the old code narrowly avoided.

## Open questions added

- **avg follow-up:** sum+count expansion needs a synthetic-slot mechanism
  (descriptor width checks currently enforce one column per slot).
- **Grouped two-phase:** delete the translator guard + wire the partitioned
  sink when #838 lands; FRAG-7 already pins the merge semantics.
- **fmt drift:** `cargo fmt --check` fails on pre-existing CN files untouched
  by this stack (nixl_transport.rs, prpc_client.rs, …) — a one-off
  `cargo fmt` commit would be pure churn removal, kept out of this stack.

---

# Part 3 — the byte-range split stack (`a5c25f76..8c2ebea5`, 2026-08-06)

Clears the survey's dominant blocker (18/22 queries: "byte-range splits do not cover the whole
parquet file"). Plan/decision record: BYTE-RANGE-SPLITS-PLAN.md; survey: TPCH-SURVEY.md.
Review order = commit order (rule → engine plumbing → plan carrier → CN emission → hardening →
fix + docs). The single correctness property everything serves: **N splits of one file read
every row exactly once**, deterministic under any placement.

## The three decisions to hold while reading

1. **Ownership rule** = StarRocks reader convention: a row group's start offset is the min of
   its first column's data/index/dictionary page offsets (+ `RowGroup.file_offset`), each
   counting only when present (cudf has no `__isset`: 0 = absent); a range owns a row group
   iff it contains its start. Straddle → owned by the range holding the start; range inside
   one row group → owns nothing (valid EMPTY split, never a whole-file fallback).
2. **Ranges ride the plan** (`FileOrFiles.start/length`) but DuckDB's consumer (a submodule)
   and `parquet_scan` can't carry them, so `lower_substrait` extracts them into a per-plan
   `ClientContextState` that `build_parquet_table_info` claims. The discipline (single-shot
   claims, `assert_all_consumed`, throw-on-unknown-rel, always-replace-the-state) all guards
   ONE failure mode: an emitted-but-unapplied range silently degrades to a whole-file read
   that duplicates rows N times.
3. **The CN refuses what it cannot honor exactly**: overlap (would double-read under start
   ownership), past-EOF, negative, zero-owned-bytes instances, `has_more` incremental
   delivery, compressed containers. An exact whole-file tiling still collapses to a
   whole-file read, keeping every pre-split plan byte-identical.

## 16. `a5c25f76` — the rule, wired to nothing
`parquet_byte_range.{hpp,cpp}` + exact-tiling sweep (k=1..16), edge cases, real-footer check.
Also answered open question Q1: cudf's `filter_row_groups_with_byte_range` agrees with the
StarRocks rule on the real footer (kept as an informational cross-check).

## 17. `ad2400b7` — the ingestible honors per-file ranges
`resolved_file_ranges` parallel to the paths; applied between `all_row_groups` and stats
pruning; empty selection rides the existing all-pruned fallback (`set_num_rows(0)`). The
pinned cache refuses ranged identities in both directions (a whole-file pin serving a ranged
scan returns extra rows). Skeptical question: can `set_row_groups` see an empty list? (No —
the all-pruned fallback catches it first; empty-list semantics flipped between cudf versions.)

## 18. `bf36716e` — ranges ride the plan
The registry + extraction walk + claims + `assert_all_consumed` + S3 refusal. The Rust FFI
test executes splits on the GPU: two half-file ranges partition 30k rows exactly; **both
splits in one plan (duplicate path, two LocalFiles items) equal the whole file** — answering
open question Q3 (DuckDB's MultiFileList handles the duplicate-path case; `attach_byte_ranges`
expands per-file ranges regardless of dedup behavior since disjoint ranges make only the
per-file union matter).

## 19. `d1e5d969` — the CN emits splits
`resolve_ranges` replaces `validate_complete_files`; translation tests use the live FE capture
shapes (162140518-byte lineitem split at 81070259). The `has_more` refusal matters more than
it looks: the old coverage check was the only thing that made partial delivery loud, and it is
gone.

## 20. `02e90ef6` — compressed-container refusal
Small hardening. Deviation from the plan: no `num_of_columns_from_file` check (only diverges
when path-derived columns exist — already refused).

## 21. `e0180970` — the canonicalization fix (read this one's story)
First live run tripped the unclaimed-range guard: the FE spells `file:/…`, DuckDB's bind
reports the plain path, the claim never matched. **The guard converted what would have been
silent N×-read duplication into a plan-time error** — the exact failure mode decision 2's
discipline exists for, caught on its first real opportunity. Fix: one canonicalizer at insert
and lookup. Live gate after the fix: `count(*) = 6001215` over the single split 155 MB
lineitem (exactly-once), Q6 = `61567694.9502`, `agg_stage=1` regression identical, GROUP
BY/avg guards unchanged.

## 22. `8c2ebea5` — DEMO.md
The byte-identical-file recipe is retired; the demo now scans one large file split by the FE.

## Riskiest lines added by this stack

1. **`parquet_byte_range.cpp::row_group_start_offset`** — the ownership rule. A divergence
   from what the FE assumes shifts row groups between splits (dup or loss). Pinned by the
   tiling sweep, the real-footer test, and the live count(*) gate.
2. **`substrait_scan_ranges.cpp::canonical_scan_path`** — a path spelling that misses both
   insert and lookup would resurrect the unclaimed-range failure; today that is loud
   (`assert_all_consumed`), so the risk is refused queries, not wrong rows.
3. **`scan_paths.rs::resolve_ranges` overlap refusal** — deleting it re-admits double-reads
   the engine cannot detect (each range is individually valid).

## Open questions / accepted gaps added

- Cross-CN tiling cannot be verified by any single CN — the FE is the documented trust
  boundary (per-CN refusals of overlap/past-EOF/has_more are the net).
- S3 byte ranges refused (`sirius_read_parquet` path untested against ranges).
- The full engine suite ran green over the complete stack at `8c2ebea5`
  (2187 passed / 1 skipped) in addition to the per-commit targeted suites and the live e2e.
