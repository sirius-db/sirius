# PLAN-02 — Query-scoped park ownership + a real `cancel_plan_fragment`

**Status:** plan only. Nothing in this document has been implemented.
**Scope:** `experimental/starrocks/` (the Rust StarRocks compute node). No C++ engine change is
required or proposed.
**Repo:** `/home/ubuntu/sirius`, branch `demo-multi-cn`. Default/PR branch is `dev`.
**Item:** #1 of the five ranked in `bench/rtxpro6000-2gpu/SF500-CONFIG-AND-ARCHITECTURE.md:122-144`.
**Estimated size:** days. ~7 files touched, one new trait method, one new engine request variant,
two RAII guards, one new tunable.

Every file:line in this document was read and verified against the working tree at the time of
writing unless explicitly marked **UNVERIFIED**. Discrepancies found against the brief that
commissioned this plan are recorded in [§11](#11-evidence-i-could-not-re-verify).

---

## 0. Orientation for a fresh session

You are working on **Sirius as a StarRocks compute node**. A StarRocks frontend (FE) plans a
query, splits it into *fragments*, and dispatches each fragment instance over BRPC/PRPC to a
compute node (CN). The CN is a Rust binary (`experimental/starrocks/`) that links the Sirius C++
GPU engine through a cxx FFI (`rust/crates/sirius/`, `src/sirius_ffi.cpp`).

Three facts you need before reading anything else:

1. **A sender fragment's output stays on the GPU.** When fragment A feeds fragment B, A runs,
   *parks* its output as native cudf batches inside a live `sirius::Fragment`, and returns. B
   runs later and *relays* those batches in. Nothing is serialised, copied to host, or written to
   disk in between. `experimental/starrocks/src/fragment_executor.rs:90-99`.
2. **That park deliberately outlives its own query window on the C++ side.** The exchange
   repositories are created outside `data_repository_manager_` precisely so `QueryEnd()`'s
   `clear_all_repositories()` cannot destroy them —
   `src/exec/streaming_fragment.cpp:64-66` and `src/include/exec/streaming_fragment.hpp:67-72`.
3. **Nothing on the CN owns that park at query granularity.** There is no per-query teardown. That
   is the bug this plan fixes. The trait's own doc comment already names the gap —
   `experimental/starrocks/src/fragment_executor.rs:97-99`: *"Concurrency needs per-query lifecycle
   isolation in the engine."*

### Build and test

```bash
# C++ engine (needed only by the engine-linked CN tests)
pixi run make

# CN, pure Rust — no engine, no GPU. This is what CI runs.
pixi run --manifest-path experimental/starrocks/pixi.toml cn-test-no-engine

# CN, engine-linked — needs the build tree and a GPU.
pixi run --manifest-path experimental/starrocks/pixi.toml cn-test
pixi run --manifest-path experimental/starrocks/pixi.toml cn-build
```
Task definitions: `experimental/starrocks/pixi.toml:131-148`.

### Getting engine logs (read this before any measurement)

The CN reads `SIRIUS_LOG_BACKEND`, `SIRIUS_LOG_DIR`, `SIRIUS_LOG_LEVEL` in
`src/sirius_ffi.cpp:170-178`, which calls `duckdb::install_configured_log_sink(nullptr)` at
`src/sirius_ffi.cpp:177`.

**Only `duckdb`, `spdlog`, `noop` are accepted.** An unknown value is **silently ignored on the CN
path**: the `throw InvalidInputException` for an unknown backend at
`src/sirius_context.cpp:1573-1578` is guarded by `else if (db)`, and the CN passes `nullptr`. You
will get no logs and no diagnostic. Set `SIRIUS_LOG_BACKEND=spdlog` exactly.

`/opt/dlami/nvme/sirius-build/up-sf500-x.sh:31-33` already exports all three; if you launch the CN
another way you must set them yourself.

### Reading order for the code

| File | Why |
|---|---|
| `experimental/starrocks/src/engine.rs:1-22` | Module doc: the single engine thread, and why staging calls bypass its request channel. |
| `experimental/starrocks/src/engine.rs:245-320` | The engine thread's request loop and the park maps. **The core of this plan.** |
| `experimental/starrocks/src/engine.rs:424-620` | `run_fragment_inner`: relay in, run, park out. |
| `experimental/starrocks/src/fragment_executor.rs:37-48, 68-145` | `SenderSlot`, `FragmentRun`, the `FragmentExecutor` trait. |
| `experimental/starrocks/src/compute_node_service.rs:832-1051` | `execute_fragment_with_inputs`: routing, running, park, rendezvous, remote drains. |
| `experimental/starrocks/src/local_exchange.rs:87-320` | The rendezvous. No eviction anywhere. |
| `experimental/starrocks/src/nixl_transport.rs:395-419, 700-790` | The two consumer-driven `drop_parked` calls. |
| `docs/super-sirius/streaming-sessions.md` | The C++ side of the streaming boundary. |

`src/legacy/` is dead code — do not modify it.

---

## 1. Problem statement

### 1.1 The leak

**A parked sender output whose destination is LOCAL has no teardown release path.**

All three release paths require a *consumer to actually run*:

| Path | Site | Requires |
|---|---|---|
| `release_slot` after a successful `relay_from` | `experimental/starrocks/src/engine.rs:533` | The receiver fragment to be dispatched and to run. |
| `drop_parked` after a successful remote drain | `experimental/starrocks/src/nixl_transport.rs:777` | The transport thread to finish transmitting. |
| `drop_parked` after a *failed* remote transmit | `experimental/starrocks/src/nixl_transport.rs:410` | A remote destination to have been attempted at all. |

The **only** consumer-free release is the process-wide blanket wipe on fragment failure —
`experimental/starrocks/src/engine.rs:274-298`:

```rust
if let Err(err) = &result {
    ...
    poisoned.clear();
    for slot in parked_slots.keys() { poisoned.insert(slot.clone(), err.clone()); }
    parked.clear();          // engine.rs:297
    parked_slots.clear();    // engine.rs:298
}
```

So on a **successful** query whose receiver never runs (or runs but leaves a sibling destination
undrained), the park is immortal. It is released only when some *later, unrelated* query fails on
that CN.

### 1.2 The one per-query end-of-life signal the FE sends is a stub

`experimental/starrocks/src/compute_node_service.rs:374-379`, verbatim:

> Real teardown (aborting the engine run, freeing GPU buffers, dropping parked exchange state) is
> a separate work item.

The handler (`:381-402`) logs, calls `self.core.results.cancel(id, reason)` (`:398`), and returns
OK. It never touches `self.core.exchanges` or `self.core.executor`.

The FE sends it for **every fragment instance of every query**. The proto is
`experimental/starrocks/starrocks/gensrc/proto/internal_service.proto:473-482`; the reason enum is
`:464-471`, where `QUERY_FINISHED = 5` (`:470`) — i.e. the dominant cancel is not an abort at all,
it is the normal end-of-query signal. `query_id` is field 11 and is `optional` (`:477`), so the CN
must not require it.

### 1.3 It is a hard ledger debit, not fragmentation or accounting drift

`cucascade::memory::reservation_aware_resource_adaptor::get_total_allocated_bytes()` is a pure
software counter — `cucascade/src/memory/reservation_aware_resource_adaptor.cpp:316`:

```cpp
std::size_t impl_type::get_total_allocated_bytes() const { return _total_allocated_bytes.load(); }
```

It is incremented at `:464` (`_total_allocated_bytes.try_add(tracking_bytes, _capacity)`) **before**
`_upstream.allocate(...)` at `:468`, and decremented at `:513`. Critically, `try_add` is also the
**admission gate**: when it fails, the allocation is refused at `:478-482` with
`"not enough capacity to allocate memory"` / `MemoryError::LIMIT_EXCEEDED` — *before the CUDA
driver is ever consulted*.

Consequence: **bytes retained by a park are subtracted directly from the next query's budget.**
A leak here is indistinguishable, from the allocator's point of view, from the pool being smaller.

The pool figures are logged at `src/sirius_context.cpp:254`:
```
[gpu_pool] GPU:{} {} allocated={} bytes peak={} bytes reserved={} bytes
```
and the per-fragment window events at `src/sirius_context.cpp:503` (`[window] ... outcome={}`),
with `outcome=unwind` emitted from the `StandaloneQueryScope` destructor at `:612`.

### 1.4 The retained memory is unspillable

The downgrade sweep enumerates only the per-query registry —
`src/downgrade/downgrade_executor.cpp:223`:

```cpp
auto const managers = _data_repo_registry.get_all();
```

and exchange repositories are **by construction** outside it (`src/exec/streaming_fragment.cpp:64-66`).
That is not an oversight; it is the mechanism that makes streaming work
(`src/include/exec/streaming_fragment.hpp:67-72`). So the downgrade executor cannot see, let alone
spill, a parked output. Do **not** "fix" this by registering exchange repos with the manager —
`QueryEnd()`'s `clear_all_repositories()` would then destroy a sender's output before its receiver
runs.

### 1.5 Measured cost

Recorded in `bench/rtxpro6000-2gpu/SF500-CONFIG-AND-ARCHITECTURE.md:126-136`. Reproduced here with
the arithmetic re-checked (see [§11](#11-evidence-i-could-not-re-verify) for the two figures that
did not reconcile):

- **≈11.31 GiB retained per q07 run per CN**, held by 4 parked fragments. The retained halves sum
  to `7,284,904,448 + 4,543,639,808 + 306,675,200 + 7,466,496 = 12,142,685,952 B = 11.3088 GiB`.
- It survives ~207 s of idle byte-for-byte and is paid again by the next run
  (22.732 GiB ≈ **2.010×** the floor — checked: `22.732 / 11.309 = 2.0101`).
- In one log: **386** `not enough capacity to allocate memory` (LIMIT_EXCEEDED, i.e. thrown at
  `reservation_aware_resource_adaptor.cpp:478` before the driver) against **0** driver refusals
  (`cudaErrorMemoryAllocation`), and `reserved=0` in all 1,273 `[gpu_pool]` records.
- **356 downgrade requests freed 0 bytes.**
- Release machinery itself works: in one trace, window 12 released 36,834,794,752 B — exactly what
  windows 5-11 parked — 2.7 s later, leaving the ~11.31 GiB floor untouched.
- On all 18 `outcome=unwind` windows the next `QueryBegin` on that instance reads `allocated=0`
  within 6-160 ms. The only code running in that gap is `parked.clear(); parked_slots.clear();`
  (`engine.rs:297-298`). **A failure path releases 11-56 GiB that no success path releases.**

### 1.6 Cost visible in the committed benchmark CSVs (verified in-tree)

`bench/rtxpro6000-2gpu/results/sf500xcold.csv`:
```
q07,0,cold,pass,83302,4
q07,1,warm,pass,289294,4      →  warm/cold = 3.4728×
```

`bench/rtxpro6000-2gpu/results/sf500x.csv` (no per-query restart):
```
q05,0,cold,pass,22482,5
q05,1,warm,refused,300104,0
q07,0,cold,pass,83251,4
q07,1,warm,refused,319030,0
```
Both queries pass cold and **fail on a second run of the same cluster**. The sweep that produced
the full 22 has to restart the cluster before every query to work at all
(`/opt/dlami/nvme/sirius-build/sweep-sf500x-cold.sh`, whose header says so explicitly), and the
benchmark harness's own documentation records the cause —
`experimental/starrocks/benchmarks/tpch/bench.sh:44-49`:

> `RESTART_CMD` command that fully restarts the cluster. The CN has no `cancel_plan_fragment` yet,
> so a hung or failed query strands its fragments and eventually starves the CNs ("No available
> backends") — without a restart every later measurement is invalid.

---

## 2. Full lifecycle trace of a parked output

### 2.1 Creation

1. FE dispatches a `DATA_STREAM_SINK` fragment → `exec_plan_fragment`
   (`compute_node_service.rs:321-343`) → `spawn_blocking` (`:331`) →
   `exec_single_attachment` → `process_fragment` (`:731-764`) →
   `execute_fragment_with_inputs` (`:832-1051`).
2. Every destination is routed **before** any GPU work (`:951-981`), producing a `SenderSlot`
   per destination (`:957-961`) and a `DestinationRoute::Local | Remote` (`:968`).
   `SenderSlot` is keyed by the **receiver** — `fragment_executor.rs:37-48`:
   ```rust
   pub struct SenderSlot { fragment_instance_id, node_id, sender_id }
   ```
   It carries **no query id**. Nothing could age it out even if something wanted to.
3. `self.executor.run(FragmentRun { outputs: slots.clone(), .. })` — `:985-992`.
4. `SiriusEngine::run` (`engine.rs:667-680`) posts `EngineRequest::Run` on the single FIFO channel
   and **blocks** on the respond channel (`engine_call`, `engine.rs:640-655`).
5. The engine thread runs `run_fragment` → `run_fragment_inner` (`engine.rs:427-618`). After
   `fragment.run()` (`engine.rs:583-585`), with `outputs` non-empty:
   ```rust
   let park_id = *next_park_id; *next_park_id += 1;             // engine.rs:590-591
   for (stream, slot) in request.outputs.iter().enumerate() {
       if parked_slots.contains_key(slot) { return Err(...) }   // engine.rs:593-597
       parked_slots.insert(slot.clone(), (park_id, stream as u64));  // engine.rs:598
   }
   parked.insert(park_id, ParkedOutput { fragment, outstanding: request.outputs.len() });
                                                                 // engine.rs:600-607
   return Ok(None);                                              // engine.rs:608
   ```
   `ParkedOutput` is `engine.rs:43-46`: `{ fragment: sirius::Fragment<'ctx>, outstanding: usize }`.
   **The GPU memory is held by that `sirius::Fragment`.** Dropping it drops the C++
   `streaming_fragment`, its `_output_repos`, and the batches.

### 2.2 Release path A — local receiver runs

`push_sender` (`compute_node_service.rs:998-1008`) registers `SenderSource::LocalParked { names, slot }`
in the `LocalExchange`. When the receiver's sender set is complete, `take_ready`
(`local_exchange.rs:248-313`) hands back a `ReadyFragment` carrying that slot. The dispatch worker
(`compute_node_service.rs:304-314`) runs `execute_ready_fragment` (`:1129-1195`), which turns the
`LocalParked` sources into `inputs` (`:1170`) and passes them to `execute_fragment_with_inputs`
(`:1194`). On the engine thread, `run_fragment_inner` relays and releases —
`engine.rs:519-541`:
```rust
let moved = fragment.relay_from(&mut sender.fragment, sender_stream, stream_id, sender_id)?;
release_slot(parked, parked_slots, poisoned, slot)?;   // engine.rs:533
```
`release_slot` (`engine.rs:363-381`) decrements `outstanding` and removes the `ParkedOutput` at
zero.

### 2.3 Release path B — remote destination drains

`transport.start_fragment(spec)` (`compute_node_service.rs:1034`) posts a drain to the transport
thread. On success the transport calls `self.executor.drop_parked(spec.slot)` after the eos frame
(`nixl_transport.rs:777`); on failure the transport-thread loop calls it too
(`nixl_transport.rs:405-418`, the `drop_parked` at `:410`).

### 2.4 Release path C — blanket wipe on any fragment failure

`engine.rs:274-298`, described in §1.1. Process-wide, not query-scoped.

### 2.5 Leak paths

Six, all verified.

| # | Site | Why it leaks |
|---|---|---|
| **L1** | **No teardown at all** for a local park whose receiver never runs | `LocalExchange` entries leave `receivers`/`sources`/`remote_seq` **only** inside `take_ready` (`local_exchange.rs:282-308`), downstream of `if complete != expected { return Ok(None); }` (`:277-279`). No TTL, no cap, no GC, no cancel hook. `cancel_plan_fragment` (`compute_node_service.rs:381-402`) does not touch it. **This is the 11.31 GiB.** |
| **L2** | `compute_node_service.rs:1008` | The `?` on `push_sender`. A duplicate sender (`local_exchange.rs:150-152`) or any rendezvous error unwinds `execute_fragment_with_inputs` **after** `executor.run` already parked at `:985`. `executor.run` returned `Ok`, so the blanket wipe at `engine.rs:274` never fires. |
| **L3** | `compute_node_service.rs:1039-1042` | `transport.start_fragment` failed to post. `drains.join()` then `return Err(err)` — every slot not yet posted (and every local slot already pushed) is abandoned. |
| **L4** | `compute_node_service.rs:1161`, `:1163`, `:1179-1185` | `execute_ready_fragment` drops the `ReadyFragment` on error. Its `SenderSource::LocalParked` slots were **already removed** from `LocalExchange` by `take_ready` (`local_exchange.rs:295`), so nothing can ever find them again. `:1161` = schema/collect error, `:1163` = translation failure, `:1179` = an open remote source. The same drop also strands every `SenderSource::Remote { batches }` **staging-arena lease** (`local_exchange.rs:44`) — a second, independent leak, in the arena rather than the pool. |
| **L5** | `compute_node_service.rs:151-154` | `FragmentOutcome::join_into_ready` does `self.drains.join()?; Ok(self.ready)` — the `?` discards `self.ready`, a `Vec<ReadyFragment>`. Same class as L4. |
| **L6** | `compute_node_service.rs:257-261`, `:277-281` | `dispatch` returns `Err` when the worker has exited; `dispatch_then_join` records the error and drops the `ReadyFragment`. Same class as L4. |

There are no `Drop` impls anywhere in the crate that would catch these —
`impl Drop` exists only at `prpc_client.rs:244`, `lib.rs:862`, `lib.rs:960`,
`nixl_transport.rs:151`, `nixl_transport.rs:254`, `engine.rs:711`, and none of them touch parks.

---

## 3. Design

Five pieces. They compose; none is useful alone.

### 3.1 `query_id` on the park

**`FragmentRun` gains a required field** — `experimental/starrocks/src/fragment_executor.rs:68-88`:

```rust
pub struct FragmentRun<'a> {
    pub plan: &'a TranslatedPlan,
    /// The query every fragment instance of this plan belongs to. Required: an
    /// unattributable park is exactly the thing that leaks, so a fragment with no query
    /// id is refused rather than parked.
    pub query_id: FragmentInstanceId,
    pub inputs: Vec<(i32, Vec<SenderSlot>)>,
    pub remote_inputs: Vec<(i32, i32, Vec<StagedBatch>)>,
    pub outputs: Vec<SenderSlot>,
    pub broadcast: bool,
    pub hash_keys: Vec<usize>,
}
```

`FragmentInstanceId` is `Copy` (`result_store.rs:22-23`), so this costs nothing to thread.

**Every construction site** (exhaustive — `grep -n 'FragmentRun {'`):

| Site | Kind | Value to pass |
|---|---|---|
| `compute_node_service.rs:845` | production, RESULT_SINK branch | `query_id` hoisted at the top of `execute_fragment_with_inputs` |
| `compute_node_service.rs:985` | production, sender branch | same hoisted `query_id` |
| `engine.rs:867` | `#[cfg(test)]` (`mod tests` starts at `engine.rs:731-732`) | `FragmentInstanceId::from_halves(1, 0)` |
| `engine.rs:1058` | `#[cfg(test)]` | `FragmentInstanceId::from_halves(3, 0)` |
| `engine.rs:1083` | `#[cfg(test)]` | same as `:1058` (same query) |
| `engine.rs:1203` | `#[cfg(test)]` | `FragmentInstanceId::from_halves(1, 0)` |
| `engine.rs:1218` | `#[cfg(test)]` | same as `:1203` (same query) |

Hoist once, as the first statement of `execute_fragment_with_inputs`
(`compute_node_service.rs:832-840`):
```rust
let query_id = Self::query_id(params)
    .ok_or_else(|| format!("{} carries no query id; a park it produced could never be \
                            attributed to a query and would leak", Self::fragment_context(params)))?;
```

**This refuses nothing that runs today.** `Self::query_id` (`compute_node_service.rs:1355-1360`)
returns `Some` whenever `params.params` is present, and both branches of
`execute_fragment_with_inputs` already require `params.params`: the RESULT_SINK branch via
`fragment_instance_id` (`:841-843`, which reads `params.params`, `:1346-1351`) and the sender
branch explicitly at `:904-906`. The exchange-receiver path already hard-requires it at `:750-751`.

**`StubExecutor`** (`fragment_executor.rs:150-160`) ignores the field, as it ignores the rest.

### 3.2 The park index and `drop_query` on the engine thread

`ParkedOutput` (`engine.rs:43-46`) gains the id:
```rust
struct ParkedOutput<'ctx> {
    fragment: sirius::Fragment<'ctx>,
    outstanding: usize,
    query_id: FragmentInstanceId,
}
```

The engine thread's state (`engine.rs:250-257`) gains one index and one set:
```rust
let mut parked:          HashMap<u64, ParkedOutput<'_>>          = HashMap::new();   // unchanged
let mut parked_slots:    HashMap<SenderSlot, (u64, u64)>         = HashMap::new();   // unchanged
let mut parked_by_query: HashMap<FragmentInstanceId, HashSet<u64>> = HashMap::new(); // NEW
let mut torn_down:       BoundedIdSet                            = BoundedIdSet::new(cap); // NEW, §5
```

Maintenance rules — get these wrong and the index becomes the new leak:

- **park** (`engine.rs:600-607`): `parked_by_query.entry(query_id).or_default().insert(park_id)`.
- **release_slot** (`engine.rs:363-381`): when `entry.outstanding` reaches 0 and the `ParkedOutput`
  is removed, also remove `park_id` from `parked_by_query[query_id]`, and remove the query's entry
  when its set is empty.
- **drop_query**: `parked_by_query.remove(&q)` → for each id, `parked.remove(&id)` → then
  `parked_slots.retain(|_, (id, _)| !ids.contains(id))`, recording each removed slot in `poisoned`
  with a "query {q} was torn down" cause so a late export says *why* (the `missing_slot` mechanism
  at `engine.rs:321-333` exists for exactly this and its comment explains why it matters).
  Finally `torn_down.insert(q)`.

New engine request variant (`engine.rs:79-97`):
```rust
/// Drop every parked output belonging to `query_id` and remember the query as torn down, so a
/// sender that parks afterwards refuses instead of leaking. Fire-and-forget: the caller must
/// NOT wait (see §4).
DropQuery { query_id: FragmentInstanceId },
```
handled in the loop at `engine.rs:262-311` alongside `ExportNext` and `DropParked`. Log
`query_id`, `parks_dropped`, `slots_dropped`, and the resulting `parked.len()`.

**Refuse a park for a torn-down query.** In `run_fragment_inner`, immediately before
`parked.insert` (`engine.rs:600`):
```rust
if torn_down.contains(&request.query_id) {
    return Err(format!("query {} was torn down before this sender parked; its output is \
                        discarded", request.query_id));
}
```
Returning `Err` here drops `fragment` on the spot (it is a local by value at that point) and frees
the GPU memory. It also trips the existing blanket wipe (`engine.rs:274-298`), which is harmless
and, during the staged rollout of §9.4, is the belt to this braces.

### 3.3 `drop_query` on the `FragmentExecutor` trait

`experimental/starrocks/src/fragment_executor.rs:100-145`:

```rust
/// Drops every parked output belonging to `query_id`, releasing the GPU memory its batches
/// hold, and remembers the query as torn down so a sender that parks afterwards refuses.
///
/// POSTED, NOT AWAITED. The engine serialises everything on one thread; waiting here would
/// park a BRPC blocking-pool thread behind whatever fragment that thread is running (§4).
/// FIFO ordering on the request channel is what makes "posted" sufficient: a `run` posted
/// after this is guaranteed to see the teardown.
///
/// Idempotent: the FE sends a cancel per fragment instance, so this is called many times per
/// query and all but one call have nothing to do.
fn drop_query(&self, query_id: FragmentInstanceId) -> Result<(), String> {
    let _ = query_id;
    Ok(())          // an executor that parks nothing has nothing to drop
}
```

`StubExecutor` parks nothing (`fragment_executor.rs:153-158`: `if !run.outputs.is_empty() { return Ok(None); }`),
so the default `Ok(())` is correct for it and for every test double that does not override `run`.

`SiriusEngine::drop_query` (`engine.rs:666-709`) sends `EngineRequest::DropQuery` **without**
waiting. Add a test-only blocking variant so the engine-linked test can assert deterministically:
```rust
#[cfg(test)]
pub(crate) fn drop_query_blocking(&self, q: FragmentInstanceId) -> Result<usize, String>
```
(a `DropQuery` carrying an optional respond channel; the loop already ignores respond-send
failures — `engine.rs:260-262`).

### 3.4 A real `cancel_plan_fragment`

Replace `compute_node_service.rs:381-402`. The RPC must still **always return OK** — the existing
rationale at `:374-377` is sound (an unrouted/errored reply is a PRPC-level error frame that the FE
mis-attributes on its shared jprotobuf channel) and this change must not weaken it.

Order matters. Do it in exactly this sequence:

```
1. resolve query_id   := request.query_id, else core.query_of_instance[finst_id], else None
2. if None            → today's behaviour (log + results.cancel) + WARN, return OK
3. if !tunables.park_teardown → today's behaviour + INFO naming the knob, return OK
4. core.exchanges.drop_query(query_id)
      -> under the LocalExchange mutex: mark cancelled, remove every `receivers` entry whose
         instance belongs to Q, every `sources` entry keyed by such an instance, every
         `remote_seq` entry likewise; COLLECT the evicted StagedBatches and RETURN them.
      -> AFTER the guard is dropped: executor.staging_release(offset) for each len > 0.
         (Never hold the exchange mutex across an arena call — §4.4.)
5. core.executor.drop_query(query_id)          // posts; does not block
6. core.descriptor_tables.lock().remove(&query_id)      // compute_node_service.rs:188, :798
7. core.results.cancel(id, reason)                       // unchanged, compute_node_service.rs:398
8. core.query_of_instance evicts every instance of Q
9. log query_id, cancel_reason, exchange_entries_evicted, leases_released, and return OK
```

Step 4 before step 5 is deliberate: evicting the rendezvous first means no concurrent
`push_sender` can complete a sender set and dispatch a receiver for a park that step 5 is about to
destroy.

**`LocalExchange::drop_query`** is new (`local_exchange.rs`, next to `take_ready`):
```rust
/// Evicts every rendezvous entry belonging to `query_id` and returns the staging leases the
/// evicted remote sources were holding, for the caller to release outside this lock.
///
/// Also records the query as cancelled, so a `push_sender` that arrives afterwards refuses
/// instead of readying a receiver whose senders have already been torn down.
pub(crate) fn drop_query(&self, query_id: FragmentInstanceId)
    -> (usize /*entries*/, Vec<StagedBatch> /*leases to release*/)
```

Wiring the instance→query relation: `ExchangeKey` (`local_exchange.rs:19-22`) holds the
*receiver's* `fragment_instance_id`, not a query id. Two options; take the first:

- **(a) Store the query id.** `register_receiver` (`local_exchange.rs:105-139`) already receives
  the receiver's `params`; take `query_id` as an explicit argument and keep a
  `HashMap<FragmentInstanceId /*query*/, HashSet<FragmentInstanceId> /*instances*/>` inside
  `ExchangeState` (`local_exchange.rs:87-95`). Explicit, O(1), and the call site
  (`compute_node_service.rs:754-759`) already has the query id two lines up at `:750`.
  `push_sender`/`push_remote_frame` do not need the query id: they key off an instance that
  `register_receiver` has already mapped, and an instance with no mapping cannot belong to a
  cancelled query the CN has seen.
- (b) Scan every key on cancel. O(entries) per cancel × 629 cancels per sweep. Rejected.

### 3.5 `ParkGuard` and `ReadyGuard` — RAII over the post-park window

Both leak classes are "an owner of GPU/arena resources was dropped on an error path". Fix them with
ownership, not with more error handling.

#### `ParkGuard` — covers L2 and L3

Lives in `compute_node_service.rs`. Constructed the instant `executor.run(...)` returns `Ok` at
`:985-992`; owns every slot in `slots` until each is handed to a real consumer.

```rust
/// The slots a sender just parked, owned until each is handed to the consumer that will
/// release it. Dropping the guard with slots still owned releases them.
///
/// This is the only thing standing between an error on the post-park path and a permanent
/// GPU leak: `executor.run` already returned Ok, so the engine's blanket wipe
/// (engine.rs:274-298) will never fire for these.
struct ParkGuard<'a> {
    executor: &'a dyn FragmentExecutor,
    owned: Vec<SenderSlot>,
    armed: bool,             // false under the kill-switch: log, do not release
}

impl ParkGuard<'_> {
    /// This slot now belongs to `LocalExchange` (push_sender returned Ok) or to the transport
    /// thread (start_fragment returned a ticket).
    fn hand_off(&mut self, slot: &SenderSlot);
    /// Everything handed off.
    fn disarm(self);
}

impl Drop for ParkGuard<'_> {
    fn drop(&mut self) {
        for slot in std::mem::take(&mut self.owned) {
            if !self.armed { warn!(slot = ?slot, "park teardown disabled: leaking a slot"); continue; }
            if let Err(err) = self.executor.drop_parked(slot) {
                warn!(slot = ?slot, error = %err, "failed to release a park abandoned on an error path");
            }
        }
    }
}
```

Hand-off points in `execute_fragment_with_inputs`:
- **local**: after `push_sender` returns `Ok` (`:998-1008`) — ownership passes to `LocalExchange`
  (and, if it returned `Some(ready)`, onward to the `ReadyGuard` below).
- **remote**: on `Ok(ticket)` only (`:1035`) — the transport thread owns the exactly-once
  `drop_parked` from then on (`nixl_transport.rs:410`, `:777`). On `Err` at `:1039-1042` the guard
  releases this slot *and every slot not yet posted*. That is L3, closed.
- `disarm()` immediately before `Ok(FragmentOutcome { .. })` at `:1047-1050`.

#### `ReadyGuard` — covers L4, L5, L6

A `ReadyFragment` (`local_exchange.rs:74-79`) that never reaches `run_fragment_inner` strands both
its `LocalParked` slots and its `Remote` staging leases. Give it an owner from the moment it leaves
`take_ready`.

```rust
/// A ready receiver plus ownership of everything its sources hold: the parked outputs of its
/// local senders and the staging leases of its remote ones. `take_ready` has already removed
/// these from the rendezvous, so if this is dropped nothing else can ever find them.
struct ReadyGuard {
    executor: Arc<dyn FragmentExecutor>,
    ready: Option<ReadyFragment>,
    armed: bool,
}
impl ReadyGuard {
    fn new(executor: Arc<dyn FragmentExecutor>, ready: ReadyFragment) -> Self;
    /// Ownership passes to the FragmentRun that is about to consume them.
    fn into_ready(mut self) -> ReadyFragment;
}
impl Drop for ReadyGuard { /* drop_parked each LocalParked slot; staging_release each len>0 batch */ }
```

`Arc<dyn FragmentExecutor>` (not a borrow) because the guard crosses the dispatch channel.
`FragmentExecutor: Send + Sync` (`fragment_executor.rs:100`) and `ServiceCore.executor` is already
`Arc<dyn FragmentExecutor>` (`compute_node_service.rs:182`), so this is free.

Type changes that follow:
- `FragmentOutcome.ready: Vec<ReadyGuard>` (`compute_node_service.rs:127-129`).
- `SiriusComputeNodeService.ready_fragments: mpsc::Sender<ReadyGuard>` (`:86-88`), the channel at
  `:243`, `dispatch` at `:257-261`, `dispatch_worker` at `:304-314`.
  **Bonus property to call out in review:** `mpsc::SendError<ReadyGuard>` *carries the guard back*,
  so the existing failure at `:258-260` releases by construction — L6 closes itself.
- `execute_ready_fragment` (`:1129-1195`) takes `ReadyGuard`, calls `into_ready()` only at `:1166`
  where the sources are moved into `inputs`/`remote_inputs`. Everything above that — `:1161`,
  `:1163` — is now covered. `:1179-1185` sits inside the `for input in ready.inputs` loop; restructure
  so the `closed == false` check happens *before* `into_ready` (validate the guard's contents, then
  consume), or hold a second small guard over the partially-drained loop. The former is simpler.
- `join_into_ready` (`:151-154`) and `dispatch_then_join` (`:275-297`) need no change beyond the
  type: dropping a `Vec<ReadyGuard>` now releases. L5 closes itself.

### 3.6 Query-scoped teardown on the failure path

`run_ready_fragment`'s error arm (`compute_node_service.rs:688-722`) already knows `query_id`
(`:679`) and already calls `self.results.fail_query(query_id, id, error)` (`:709`). Add there:
```rust
self.exchanges.drop_query(query_id);   // + release the returned leases
self.executor.drop_query(query_id);
```
This is the same teardown as cancel, on the path that currently relies on the blanket wipe.

**Do not delete the blanket wipe in the same change.** See §9.4 for the staged rollout.

---

## 4. Concurrency and ordering

### 4.1 The threads

| Thread | What it is | Sends to the engine? |
|---|---|---|
| BRPC current-thread runtime | `main.rs:577-586`, one `spawn_blocking`-hosted runtime owning the listener | no — every handler offloads |
| BRPC **blocking pool** | `tokio::task::spawn_blocking` from the handlers | **yes, and it blocks** |
| `fragment-dispatch` | dedicated `std::thread`, `compute_node_service.rs:245-248`, loop at `:304-314` | yes, blocks |
| `nixl-transport` | dedicated `std::thread`, `nixl_transport.rs:353`, loop at `:395-431` | yes, blocks (`export_packed_next`, `drop_parked`) |
| `sirius-engine` | dedicated `std::thread`, loop at `engine.rs:262-311` | is the engine |

Handlers that use the blocking pool: `exec_plan_fragment` `:331`, `exec_batch_plan_fragments`
`:357`, `fetch_data` `:419`, `exchange_nixl_md` `:486`, `request_staging_lease` `:519`,
`transmit_packed` `:554`. The design rule is stated at `engine.rs:16-22`:

> Staging-arena calls deliberately BYPASS the request channel … Funneling leases through the engine
> thread turns any engine stall into a peer's exchange stall — a fragment wedged inside `run()`
> starved the peer CN's `request_staging_lease` for the PRPC timeout and failed the whole query …

### 4.2 Why `cancel_plan_fragment` must not block

`SiriusEngine::engine_call` (`engine.rs:640-655`) posts on one FIFO channel and blocks on
`respond_rx.recv()`. A blocking `drop_query` therefore waits out whatever fragment the engine
thread is running — measured up to **289,294 ms** for q07 SF500
(`bench/rtxpro6000-2gpu/results/sf500xcold.csv`). With the FE sending a cancel per fragment
instance, a wedged query produces a burst of blocked pool threads sitting in front of
`transmit_packed` and `fetch_data` — the exact starvation `engine.rs:16-22` documents.

**Therefore: `drop_query` posts and returns.** `cancel_plan_fragment` does only mutex-bounded work
on its own thread (exchange eviction, arena releases, result-store cancel) and returns OK
immediately.

This is safe because the request channel is **FIFO into a single consumer**. Any `Run` posted after
a `DropQuery` is dequeued after it. There is no "the drop hasn't happened yet" window for anything
the CN itself subsequently asks the engine to do.

### 4.3 The race: a sender parks AFTER its query's teardown arrived

Two sub-cases:

- **Sender's `Run` was already queued (or in flight) when the cancel arrived.** FIFO: the `Run`
  completes and parks, *then* `DropQuery` is dequeued and sees the park. Closed by ordering alone.
- **Sender's `Run` is posted after `DropQuery`.** Reachable: a receiver of Q was handed to the
  dispatch worker before the cancel, and the worker posts its `Run` later. Closed by the engine
  thread's `torn_down` set (§3.2): the park-time check at `engine.rs:600` refuses, returning a
  named error and dropping the fragment immediately.

Symmetrically on the CN side, `LocalExchange::drop_query` marks Q cancelled under its own mutex, so
a `push_sender` that arrives afterwards (`local_exchange.rs:142-155`) refuses instead of readying a
receiver whose senders are gone. That error propagates through `compute_node_service.rs:1008` —
which, with `ParkGuard` in place, now *releases* instead of leaking.

Both sides of the race need their own marker because the two data structures are guarded by
different things (one mutex, one thread). Do not try to share one.

### 4.4 Lock ordering

Two hard rules, both new invariants worth a comment in the code:

1. **Never call an engine request (`drop_query`, `drop_parked`, `export_packed_next`) while holding
   the `LocalExchange` mutex.** `LocalExchange::drop_query` therefore *returns* the leases it
   evicted rather than releasing them itself.
2. **Never call `staging_release` while holding the `LocalExchange` mutex.** `staging_release` takes
   the arena's internal mutex (`engine.rs:107-120`, `engine.rs:696-700`); acquiring the two in one
   order here and the opposite order anywhere else is a deadlock.

### 4.5 Interaction with an in-flight nixl drain

`release_slot` is exactly-once and loud about a double release (`engine.rs:360-362`:
"a second release of the same slot is a loud error, never a silent double-drop"). If `drop_query`
destroys a slot the transport thread is mid-drain on, the transport's next
`export_packed_next`/`drop_parked` returns `Err` with the `missing_slot` message
(`engine.rs:321-333`) — and because both go through the same FIFO channel and the transport thread
blocks on its own respond, this is a clean error, never a use-after-free.

Make the `poisoned` cause read `"query {q} was torn down"` so that error explains itself.

**A `QUERY_FINISHED` cancel cannot race a live drain.** The sender's RPC does not return until every
drain it posted has been joined — `compute_node_service.rs:263-269` ("The RPC still does not return
until every drain has been joined — the FE may only be told the sender succeeded once each
destination's copy is actually across the wire") and the implementation at `:275-297`,
`:126-137`. So by the time the FE can know the query finished, no drain of it is outstanding.
A `USER_CANCEL`/`TIMEOUT`/`INTERNAL_ERROR` cancel *can* race one, and there a failed drain is the
correct outcome.

(`SenderDrains::join` is `compute_node_service.rs:102-119`; it never short-circuits, so a cancel
that turns one drain into a failure still lets every sibling run to completion and still performs
each sibling's exactly-once `drop_parked` on the transport thread.)

Watch item: `"failed to drop the parked output of a failed remote transmit"`
(`nixl_transport.rs:411-415`) appearing in a **passing** sweep would mean this reasoning is wrong.
Treat it as a bug, not noise.

---

## 5. Bounded state

Nothing in this change may add an unbounded map. Two are added; both get a cap.

### 5.1 New state

| Structure | Owner | Bound |
|---|---|---|
| `parked_by_query: HashMap<query, HashSet<park_id>>` | engine thread | Bounded by `parked.len()` **provided** `release_slot` maintains it (§3.2). Add a debug assertion that the total of all sets equals `parked.len()`. |
| `torn_down: BoundedIdSet` | engine thread | `SIRIUS_CN_TORNDOWN_QUERY_CAP`, default **4096**. |
| `query_of_instance: Mutex<BoundedIdMap<instance, query>>` | `ServiceCore` | `SIRIUS_CN_QUERY_INDEX_CAP`, default **65536** instances; eagerly evicted by `drop_query`. |
| `LocalExchange` instances-per-query index (§3.4a) | `ExchangeState` | Bounded by `receivers.len()`; removed with the receiver in `take_ready` and in `drop_query`. |
| `cancelled` set in `ExchangeState` | `ExchangeState` | Shares `SIRIUS_CN_TORNDOWN_QUERY_CAP`. |

### 5.2 `BoundedIdSet` / `BoundedIdMap`

One small type, insertion-ordered ring, FIFO eviction at the cap:
```rust
struct BoundedIdSet { cap: usize, order: VecDeque<FragmentInstanceId>, live: HashSet<FragmentInstanceId> }
```
FIFO is the right policy because the set's *only* job is to catch a park that arrives shortly after
its teardown. An entry older than `cap` queries cannot do that. Log at `debug` when an eviction
happens.

**Cap rationale.** One measured sweep produced 629 cancels across 22 queries — ~29 fragment
instances per query. 4096 queries of scrollback is ~5 minutes of the busiest observed traffic and
costs `4096 × 16 B ≈ 64 KiB`. 65536 instances ≈ 1 MiB.

### 5.3 Existing unbounded state this change improves

Not the goal, but worth recording — `drop_query` gives all three their first eviction path:

- `ServiceCore.descriptor_tables` (`compute_node_service.rs:188`, inserted at `:798`) — never
  evicted today.
- `ResultStore.query_results` / `query_failures` / `fragments`
  (`result_store.rs:121-197`) — never evicted today.
- `LocalExchange.receivers` / `sources` / `remote_seq` (`local_exchange.rs:88-95`) — evicted only
  by `take_ready`.

If you want to keep this change small, evicting `descriptor_tables` and the `LocalExchange` maps is
in scope (they hold the leaked resources); the `ResultStore` maps hold only ids and messages and can
be left for a follow-up. Say which you chose in the PR description.

### 5.4 The existing `poisoned` map

`engine.rs:257` documents its bound: *"Bounded: an entry is dropped as soon as the slot is parked
again"*, and the blanket wipe enforces it with `poisoned.clear()` before re-inserting
(`engine.rs:293-296`). A **scoped** drop must not clear the whole map, so that bound is lost.
Convert `poisoned` to the same bounded ring, cap `SIRIUS_CN_POISONED_SLOT_CAP`, default **8192**.

---

## 6. Kill switch

Use the repo's env-knob validation seam — `experimental/starrocks/src/tunable.rs`, added by commit
`a27615d5` ("feat(cn): validate transport tunables at bring-up"). **Not raw `std::env::var`.**

The module's three rules (`tunable.rs:7-17`): out-of-range and unparsable values are **rejected,
never clamped and never ignored**; the resolved set is **logged at bring-up**; unset means the
documented default.

### 6.1 The knob

```rust
/// Master switch for query-scoped park teardown (PLAN-02).
///
/// `false` restores the pre-PLAN-02 behaviour exactly: `cancel_plan_fragment` logs and cancels
/// the result entry only, and the RAII guards log what they WOULD have released instead of
/// releasing it. That makes an A/B of the whole change possible on ONE binary, which is how the
/// 11.31 GiB/run figure was measured in the first place.
const PARK_TEARDOWN: Knob<bool> = Knob {
    name: "SIRIUS_CN_PARK_TEARDOWN",
    default: true,
    min: false, max: true,     // or a bool-specific Knob shape without a range
};
```

`tunable.rs` today has only `Knob<u64>` (`:116-149`) and `Knob<f64>` (`:151-183`). Add a
`Knob<bool>` following the same shape: accept `true/false/1/0/on/off/yes/no` case-insensitively,
**reject** anything else with the same `rejected()` message form (`:143-148`). A range check is
meaningless for a bool, so give `Knob<bool>` its own struct without `min`/`max` rather than
faking them.

### 6.2 Wiring

| Change | Site |
|---|---|
| Add the field to `Tunables` | `tunable.rs:201-215` |
| Add it to `DEFAULTS` | `tunable.rs:219-226` |
| Read it in `from_env` | `tunable.rs:234-248` |
| Add it to the resolve log line | `tunable.rs:274-282` |
| Document it | `experimental/starrocks/docs/TUNABLES.md`, "Transport (validated registry)" table |

`Tunables::resolve()` is called at `main.rs:162` — **before** any port is bound, any GPU pool is
reserved, or any RPC is served. `Tunables::get()` returns `DEFAULTS` before that
(`tunable.rs:286-292`), which is correct for unit tests (they get teardown ON) and for builds
without the transport feature.

The three consumers read `Tunables::get().park_teardown`:
`cancel_plan_fragment` (step 3 of §3.4), `ParkGuard::new` (`armed`), `ReadyGuard::new` (`armed`).
The engine-side `torn_down` check is naturally inert when nothing ever posts `DropQuery`.

### 6.3 A second knob if the drain race bites

If §4.5's watch item fires, add `SIRIUS_CN_PARK_TEARDOWN_ON_QUERY_FINISHED` (default `true`) so
teardown can be limited to the abort reasons while `QUERY_FINISHED` reverts to the stub. Do not add
this preemptively.

---

## 7. Tests

### 7.1 The query-scoped leak assertion

**Do not** assert `QueryEnd allocated == QueryBegin allocated`. `src/sirius_context.cpp:226-228`
names that as "the leak signature", and for a monolithic query it is — but here a `[window]`
(`src/sirius_context.cpp:503`) is **one fragment**, not one query, and a sender's parked output
outliving its own window is *required* by streaming
(`src/include/exec/streaming_fragment.hpp:67-72`, `src/exec/streaming_fragment.cpp:64-66`). Such an
assertion would fire on every sender fragment and be switched off within a day.

The correct invariant is **per query, per CN**, asserted on CN state in Rust:

```
after cancel_plan_fragment(Q) has been handled on a CN:
    executor.live_parks_for(Q)     == 0
    exchanges.entries_for_query(Q) == 0
    executor.live_parks_for(R)     unchanged   for every other query R
```

The third clause is the point: this must be *scoped*, not another blanket wipe. Expose
`live_parks_for` on the test double, and (behind `#[cfg(test)]`) on `SiriusEngine` via a
`DropQuery`-style query request.

The pool-bytes version of this is a **system-level** check comparing the first `[gpu_pool]
allocated=` of run N+1 against the post-run floor of run N *for the same query on the same cluster*
— a query-to-query comparison. It belongs in §8, not in a unit test.

### 7.2 Unit tests — one per leak path

Run with `pixi run --manifest-path experimental/starrocks/pixi.toml cn-test-no-engine`
(`pixi.toml:147-148`; this is what CI runs — no GPU, no build tree).

Existing fixtures to build on, all in `compute_node_service.rs`'s `mod tests`:
`SiriusComputeNodeService::new()` (`:203-209`) and `with_executor` (`:213-219`);
`CountingExecutor` (`:1545-1555`), `is_receiver_run` (`:1559-1561`), `GatedExecutor` (`:1565-1584`),
`FailingReceiverExecutor` (`:1587-1597`), `FailingIntermediateExecutor` (`:1601-1611`),
`ReceiverSignalExecutor` (`:2197-2209`), `drain_specs` (`:2215-…`), `RecordingExecutor`
(`:2514-2534`), `transmit_params` (`:2538-…`), `route` / `assert_exec_ok` / `exec_params` /
`fragment_params` / `local_destination` (`:1620-…`), and the two existing cancel tests at
`:2971-3018` and `:3020-3053` (which must keep passing: OK status, unblocked poll, nothing
fabricated for an unknown instance).

Add a `ParkTrackingExecutor` that records `run`'s `outputs` and `query_id`, and every
`drop_parked` / `drop_query` / `staging_release`, exposing `live_parks_for(query)`.

| # | Test | Covers |
|---|---|---|
| 1 | `a_duplicate_push_sender_releases_the_slots_it_just_parked` — two senders with the same `(receiver, node, sender_id)` so `push_sender` errors (`local_exchange.rs:150-152`); assert every parked slot released. | **L2** (`:1008`) |
| 2 | `a_failed_transport_post_releases_the_slots_it_did_not_hand_off` — a fake transport whose request channel is closed so `start_fragment` errors (`nixl_transport.rs:225`); assert not-yet-posted slots dropped, posted ones NOT (the transport owns those). | **L3** (`:1039-1042`) |
| 3 | `a_ready_fragment_dropped_before_it_runs_releases_its_parks_and_leases` — three variants forcing `:1161` (two sources with different output names), `:1163` (untranslatable receiver params), `:1179` (a `Remote` source with `closed == false`); assert every `LocalParked` slot dropped and every `len > 0` `Remote` lease released. | **L4** |
| 4 | `a_drain_failure_does_not_strand_the_ready_receivers` | **L5** (`:151-154`) |
| 5 | `a_dispatch_send_failure_returns_the_guard_and_releases_it` — drop the worker, assert `SendError<ReadyGuard>` releases on drop. | **L6** (`:257-261`) |
| 6 | `cancel_plan_fragment_drops_every_park_of_its_query` — park two senders for Q whose receivers never arrive **and one for an unrelated query R**; cancel Q; assert `live_parks_for(Q) == 0` **and `live_parks_for(R)` unchanged**. | **L1**, and scoping |
| 7 | `cancel_plan_fragment_without_a_query_id_resolves_it_from_the_instance_index` — `query_id: None` is legal (`internal_service.proto:475-477`). | robustness |
| 8 | `cancel_plan_fragment_is_idempotent_across_every_instance_of_a_query` — send 8 cancels, assert one effective drop, 7 no-ops, no error, OK every time. | the 629-cancels-per-sweep reality |
| 9 | `a_sender_that_parks_after_its_query_was_cancelled_refuses_instead_of_leaking` | §4.3 |
| 10 | `cancelling_a_query_releases_the_staging_leases_of_its_undelivered_remote_frames` — `transmit_packed` frames for a receiver that never readies, then cancel; assert `staging_release` per `len > 0` batch. | the arena half of L1 |
| 11 | `the_torn_down_query_set_is_bounded` — insert `cap + N`, assert `len <= cap` and the newest survive. | §5 |
| 12 | `park_teardown_can_be_switched_off` — with `SIRIUS_CN_PARK_TEARDOWN=false`, assert nothing is released and the log names the knob. | §6 |
| 13 | `a_bool_knob_takes_true_false_and_rejects_anything_else` in `tunable.rs`'s `mod tests`. | §6.1 |

Knob tests must go through `Tunables::from_env`, **not** `resolve()` — `RESOLVED` is a `OnceLock`
and the first caller wins (`tunable.rs:259-265`). Follow the existing pattern:
`with_env` (`tunable.rs:321-338`) + `Tunables::from_env` (`tunable.rs:341-361`). `with_env` takes
the whole variable set at once because `ENV_LOCK` is non-reentrant (`tunable.rs:316-321`).

### 7.3 Engine-linked test (needs a GPU)

`pixi run --manifest-path experimental/starrocks/pixi.toml cn-test` (`pixi.toml:136-140`).

14. `drop_query_releases_the_gpu_memory_a_parked_sender_holds`, modelled on the existing park/export
    test at `engine.rs:1023-1100`: park a sender under query Q; `drop_query_blocking(Q)` returns 1;
    a subsequent `export_packed_next(slot)` errors with a message naming the teardown; a subsequent
    `drop_parked(slot)` also errors (exactly-once, `engine.rs:360-363`, as the existing test already
    asserts for the double-drop case at `engine.rs:1076-1079`).

### 7.4 SF500 measurement — the proof

`bench.sh` has **no correctness gate**. Verbatim, `experimental/starrocks/benchmarks/tpch/bench.sh:54-56`:

> NOTE: this script times and counts rows only -- it does not check answers.

So every measurement below is followed by `compare.py`. It reads `<q>.r<N>.out` from the directory
holding the timings CSV (`bench.sh:86` `OUT=$(dirname "$OUT_CSV")`, `:169` `f=$OUT/$q.r$r.out`) and
diffs cell-by-cell against `<q>.tsv`, numeric cells at a 1e-6 relative tolerance
(`/opt/dlami/nvme/sirius-build/compare.py`).

**A/B on one binary.**

```bash
source /opt/dlami/nvme/sirius-build/env.sh

# --- A: teardown OFF (today's behaviour) --------------------------------------
GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB \
  SIRIUS_CN_PARK_TEARDOWN=false \
  /opt/dlami/nvme/sirius-build/up-sf500-x.sh
# two warm runs of q07 on the SAME cluster (r0 cold, r1/r2 warm) — sweep-sf500x.sh does NOT restart
GPU_MEM=60GiB STAGING=32GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB \
  OUT=/opt/dlami/nvme/sirius-build/bench/SF500-PARK-OFF/timings.csv RUNS=2 \
  /opt/dlami/nvme/sirius-build/sweep-sf500x.sh q07
python3 /opt/dlami/nvme/sirius-build/compare.py \
  /opt/dlami/nvme/sirius-build/bench/SF500-PARK-OFF \
  /opt/dlami/nvme/sirius-build/oracle-sf500f64

# --- B: teardown ON -----------------------------------------------------------
GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB \
  SIRIUS_CN_PARK_TEARDOWN=true \
  /opt/dlami/nvme/sirius-build/up-sf500-x.sh
GPU_MEM=60GiB STAGING=32GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB \
  OUT=/opt/dlami/nvme/sirius-build/bench/SF500-PARK-ON/timings.csv RUNS=2 \
  /opt/dlami/nvme/sirius-build/sweep-sf500x.sh q07
python3 /opt/dlami/nvme/sirius-build/compare.py \
  /opt/dlami/nvme/sirius-build/bench/SF500-PARK-ON \
  /opt/dlami/nvme/sirius-build/oracle-sf500f64
```

Note `sweep-sf500x.sh` uses `bench.sh --cold` (records run 0, no per-query restart) while
`sweep-sf500x-cold.sh` uses `--cold-restart` (fresh cluster per query). **A/B must use
`sweep-sf500x.sh`** — a restart before every query is precisely what hides this bug.

**Full 22, for the regression gate:**
```bash
GPU_MEM=60GiB STAGING=32GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB \
  OUT=/opt/dlami/nvme/sirius-build/bench/SF500PARK/timings.csv \
  /opt/dlami/nvme/sirius-build/sweep-sf500x-cold.sh
python3 /opt/dlami/nvme/sirius-build/compare.py \
  /opt/dlami/nvme/sirius-build/bench/SF500PARK \
  /opt/dlami/nvme/sirius-build/oracle-sf500f64
```
Baseline to beat: `bench/rtxpro6000-2gpu/results/sf500xcold.csv` (21/22, q08+q09 refused; q08 has
since been fixed per `results/sf500e5.csv`).

**Reading the pool floor.** Engine logs land in `$SIRIUS_LOG_DIR` (`up-sf500-x.sh:32`, default
`/opt/dlami/nvme/sirius-build/siriuslog`). Grep `[gpu_pool]` (`src/sirius_context.cpp:254`) and
`[window]` (`:503`) and correlate on the window id. `reserved=` should stay 0; `allocated=` at the
first `QueryBegin` of run N+1 is the number this plan moves.

---

## 8. Success criteria

Numeric, in priority order.

| # | Criterion | Today |
|---|---|---|
| 1 | **The floor returns.** For q07 SF500 run twice on ONE cluster, `[gpu_pool] allocated=` at the second run's first `QueryBegin` equals the first run's post-run floor within **±64 MiB** on both CNs. | 22.732 GiB vs 11.309 GiB = **2.010×** |
| 2 | **No standing parks.** Within 1 s of the last cancel of a query on an otherwise idle cluster, the CN logs `live_parks=0` for that query, and the process-wide live-park count returns to 0. | unbounded growth |
| 3 | **Warm/cold ratio.** q07 SF500 `warm_ms / cold_ms` **≤ 1.2×**; **< 2.0× is a pass for this work item** (the remainder is the scheduler stall, `SF500-CONFIG-AND-ARCHITECTURE.md:164-169`, out of scope here). | **3.4728×** (289,294 / 83,302, `results/sf500xcold.csv`) |
| 4 | **q05 and q07 pass a warm second run without a restart.** | both `refused` on run 1 (`results/sf500x.csv`: `q05,1,warm,refused,300104,0`, `q07,1,warm,refused,319030,0`) |
| 5 | **Correctness does not regress.** `compare.py` vs `oracle-sf500f64`: no query that MATCHed with teardown OFF fails with it ON; total ≥ 21/22. | 21/22 |
| 6 | **No new exchange errors.** Zero occurrences of `no parked sender output` / `sender output for {slot:?} was discarded` (`engine.rs:321-333`) and zero `failed to drop the parked output of a failed remote transmit` (`nixl_transport.rs:411-415`) in a passing sweep. | — |
| 7 | **`RESTART_CMD` becomes optional.** A `--cold` sweep (no per-query restart) completes the full 22 with the same MATCH count as the `--cold-restart` sweep. This is the operational statement of the fix, and it retires the caveat at `bench.sh:44-49`. | restart mandatory |
| 8 | **Tests green.** `cn-test-no-engine` and `cn-test`. | — |

---

## 9. Risks

### 9.1 Premature release → wrong answers or use-after-free
**Highest-severity risk.** Mitigations, in order of strength:
1. **Every park mutation happens on the single engine thread** (`engine.rs:262-311`). A drop can
   never interleave with a `relay_from`. The worst outcome is a *loud* `Err`, not corruption.
2. `release_slot` is exactly-once by construction and says so (`engine.rs:360-363`).
3. `drop_query` is keyed by the **sender fragment's** query id. A receiver of that query is by
   definition also dead.
4. No park is shared between queries: `SenderSlot` is `(receiver instance, node, sender)`
   (`fragment_executor.rs:37-48`) and instance ids are per query.
5. The kill-switch (§6) reverts the whole thing without a rebuild.

### 9.2 Cancel racing a live nixl drain
Argued closed for `QUERY_FINISHED` in §4.5 (the sender's RPC joins every drain before returning —
`compute_node_service.rs:263-269`, `:275-297`). Open by design for abort reasons, where a failed
drain is correct. Watch item: §8 criterion 6.

### 9.3 Deadlock
Two new invariants (§4.4): never call an engine request under the `LocalExchange` mutex; never call
`staging_release` under it. `cancel_plan_fragment` posts rather than blocks, so a BRPC blocking-pool
thread is never parked behind a running fragment (§4.2).

### 9.4 Removing the blanket wipe too early
`engine.rs:274-298` is *measured* to be the only thing releasing 11-56 GiB across 18 `outcome=unwind`
windows. Staged rollout:
1. **Ship the scoped teardown with the blanket wipe untouched.** Add a log line on the wipe path
   reporting how many slots it still found. If the scoped path works, that number is 0.
2. Only once it is observed to be 0 across a full sweep, narrow the wipe to the failing query
   (`run_fragment` now knows `request.query_id`), keeping a `warn!` when it finds parks belonging to
   *other* queries.
3. Never remove it entirely without a replacement: a fragment failure that bypasses `drop_query`
   would silently reintroduce the leak.

### 9.5 Blast radius of the `ReadyGuard` type change
`mpsc::Sender<ReadyFragment>` → `Sender<ReadyGuard>` touches `compute_node_service.rs:86-88`,
`:243`, `:257-261`, `:304-314`, `:127-129`, `:151-154`, `:275-297`, `:1129-1195`, plus every test
that constructs the service. Mechanical but wide. Land it as its own commit so a bisect can separate
it from the behavioural change.

### 9.6 Making `FragmentRun.query_id` required
Audited in §3.1: it refuses nothing that runs today. If a fixture trips, fix the fixture — an
unattributable park is the leak, and accepting one would defeat the plan.

### 9.7 Bounded sets evicting too aggressively
Would reopen §4.3's race on a very long sweep. Cap is 4096 queries; log at `debug` on eviction, and
at `warn` if an evicted id still had live parks.

### 9.8 Interaction with the downgrade executor
None expected: exchange repositories are by construction outside `_data_repo_registry`
(`src/exec/streaming_fragment.cpp:64-66`, `src/downgrade/downgrade_executor.cpp:223`), which is
*why* the retained bytes were unspillable (356 requests, 0 bytes freed). This change does not make
them spillable — it makes them short-lived. **Do not** "fix" it by registering exchange repos with
the manager: `QueryEnd()`'s `clear_all_repositories()` would then destroy a sender's output before
its receiver runs (`src/include/exec/streaming_fragment.hpp:67-72`).

### 9.9 Stale citations in the code (fix while you are here)
`compute_node_service.rs:107-108` cites *"nixl_transport.rs:669 on success, :306-317 on failure"* for
the exactly-once `drop_parked`. The actual sites are **`nixl_transport.rs:777`** (success) and
**`:410`** (failure). Correct them in this change — a plan that hinges on those two lines should not
leave a doc comment pointing somewhere else.

---

## 10. Execution checklist

Land as four commits so each is separately revertible and bisectable.

**Commit 1 — plumbing (no behaviour change).**
- [ ] `fragment_executor.rs`: add `FragmentRun.query_id`; add the `drop_query` trait method with the
      `Ok(())` default and the "posted, not awaited" doc.
- [ ] Update all 7 construction sites (§3.1 table).
- [ ] `engine.rs`: `ParkedOutput.query_id`; `parked_by_query`; maintain it in the park path
      (`:600-607`) and in `release_slot` (`:363-381`).
- [ ] Fix the stale citations at `compute_node_service.rs:107-108` (§9.9).
- [ ] `cn-test-no-engine` + `cn-test` green.

**Commit 2 — the RAII guards (closes L2-L6).**
- [ ] `ParkGuard` + hand-off points in `execute_fragment_with_inputs`.
- [ ] `ReadyGuard` + the `mpsc` type change.
- [ ] Restructure `execute_ready_fragment` so `:1179`'s validation precedes `into_ready()`.
- [ ] Tests 1-5.

**Commit 3 — teardown (closes L1).**
- [ ] `tunable.rs`: `Knob<bool>` + `SIRIUS_CN_PARK_TEARDOWN`; `docs/TUNABLES.md`.
- [ ] `BoundedIdSet`/`BoundedIdMap`; convert `poisoned`.
- [ ] `EngineRequest::DropQuery` + the engine-thread handler + the `torn_down` park-time check.
- [ ] `SiriusEngine::drop_query` (post-only) and `drop_query_blocking` (`#[cfg(test)]`).
- [ ] `LocalExchange::drop_query` + the instances-per-query index + the cancelled marker.
- [ ] `ServiceCore.query_of_instance`.
- [ ] Rewrite `cancel_plan_fragment` per §3.4; add `drop_query` to `run_ready_fragment`'s error arm.
- [ ] Tests 6-14; the two existing cancel tests (`:2971`, `:3020`) still pass.

**Commit 4 — measurement.**
- [ ] Run §7.4 A/B and the full 22; `compare.py` both.
- [ ] Record the CSVs in `bench/rtxpro6000-2gpu/results/`.
- [ ] Update `bench/rtxpro6000-2gpu/SF500-CONFIG-AND-ARCHITECTURE.md:122-144` with the measured
      result, and `bench.sh:44-49` if criterion 7 holds.

Before opening the PR: `pixi run pre-commit run -a`. Branch and PR against **`dev`**.

---

## 11. Evidence I could not re-verify

Marked so a reader does not mistake inherited numbers for freshly checked ones.

| Claim | Status |
|---|---|
| Sum of the four retained halves | **Corrected.** `7,284,904,448 + 4,543,639,808 + 306,675,200 + 7,466,496 = 12,142,685,952 B`, not `12,142,686,208` (off by 256 B). |
| "11,309 MiB (11.309 GiB)" | **Inconsistent in the source brief.** `12,142,685,952 B = 11.3088 GiB = 11,580.2 MiB`. `11,309 MiB` would be `11,858,345,984 B`. This document uses **≈11.31 GiB** throughout. |
| `22.732 GiB = 2.010×` the floor | **Arithmetic checks out**: `22.732 / 11.309 = 2.0101`. |
| q07 warm/cold `289,294 / 83,302 = 3.4728×` | **Verified in-tree**: `bench/rtxpro6000-2gpu/results/sf500xcold.csv`. |
| q05/q07 refused on warm run 1 | **Verified in-tree**: `bench/rtxpro6000-2gpu/results/sf500x.csv`. |
| 11.309 GiB survives 206.74 s of idle; 4 parked fragments; 386 LIMIT_EXCEEDED vs 0 driver refusals; `reserved=0` in all 1,273 `[gpu_pool]` records; 629 cancels with `query_id=Some(..)` in 629/629; 356 downgrade requests / 0 bytes freed; window 12 released 36,834,794,752 B; 18 `outcome=unwind` windows with the next `QueryBegin` at `allocated=0` in 6-160 ms | **UNVERIFIED by re-read.** The only engine log present in this checkout is `/opt/dlami/nvme/sirius-build/siriuslog/sirius_2026-08-19.log` (50,845 lines, 20:17-21:57), which contains **no** `[gpu_pool]` records; the analysed log is not in the tree. All of these are consistent with the code paths cited in §1.3-§1.5, which **were** verified. Re-derive them from a fresh run before quoting them in a PR. |
| Every file:line in §1-§7 other than the above | **Verified** against the working tree on branch `demo-multi-cn`. |
