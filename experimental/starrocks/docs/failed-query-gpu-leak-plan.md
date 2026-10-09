# Plan: free a failed query's GPU memory in the StarRocks CN

## Status (2026-10-06): implemented

All seven work-plan steps are in the CN. Results on 4× GB200:

| run | before | after |
|---|---|---|
| SF3000, q05 then q07 (experiment A) | q07 FAIL | q07 **PASS**; the q05 purge freed 91 received batches and 7 parked fragments across the 4 CNs |
| SF3000, all 7 queries | 2/7 | **4/7** (q07 and q12 now pass); every leak check clean |
| SF1000, all 7 queries | 7/7 | 7/7; every leak check clean |
| SF1, 2 CNs, injected failure in q05 | — | fails as injected, nothing held afterwards, then 7/7 |
| CN unit tests | 86 | 92, including 6 new ones (purge, refusals, bound, cancel, failed fragment, input guard) |

Where the code departs from the design below:

- **Query identity** uses only the FE's `hi` invariant (`FragmentInstanceId::query_hi`). It
  covers both registered and orphan sources, so the explicit instance-to-query map would add
  nothing.
- **Counters** are a `leak counters` log line, not a `/debug/leaks` endpoint, because the CN
  serves no HTTP. `tests/cn_leak_check.sh` reads them after every query in both e2e scripts.
- **Cancel** is wired on brpc only. The FE cancels over brpc, with the query id and a dummy
  instance id (`ExecutionDAG.java:637`). The thrift handler still returns not-implemented.
- **No `fetch_data` timeout trigger.** The FE cancels a query whose fetch fails, and the cancel
  path purges it.
- **Tombstones keep the first error**, so a refused late frame repeats the cause rather than a
  bare "already failed". Without this, the full SF3000 run reported q05 as a `transmit_chunk`
  refusal instead of its OOM. This change is unit-tested only; its e2e rerun was blocked by
  another cluster on the GPUs.
- **NIXL quarantine** reclaims a held WRITE only once it reports `Success`. A WRITE that never
  succeeds stays held as before, but it is now logged. It isn't exercised end to end: no WRITE
  timed out in any run.
- **Failure injection** is `SIRIUS_CN_FAIL_ONCE_FILE` with `INJECT_FAILURE_QUERY` in
  `4cn_tpch_joins_sf1000.sh`, which also runs 2-CN SF1 (`GPUS="0 1"`).
  `2cn_tpch_joins.sh` is tied to another machine's env file, so it got the leak check only.
  The injected failure fires on the first fragment that runs, before any data moves. It covers
  the cancel and receiver purge paths; the buffer-heavy path is covered by the SF3000 runs.

## Problem

When a query fails partway, the Sirius CN keeps part of that query's GPU memory for the life of
the process. Later queries start with less memory and fail with out-of-memory errors they would
not hit on a fresh CN.

### Evidence (SF3000, 4× GB200, 2026-10-06)

Two runs, each on freshly started CNs, with `GPU_FRACTION=0.85` (a 156.4 GiB pool per GPU):

| run | first query | then q07 | q07 hash-join OOMs (all GPUs) |
|---|---|---|---|
| A | q05 fails (OOM) | **FAIL** | 3 GPUs; 80–130 GB already in use when the join asked for 1.9 GB |
| B | q14 passes | **PASS**, 27 s, matches DuckDB | 0 |

q07 and q12 both pass on a fresh cluster and both fail when they run after a failed query. q05,
q08 and q09 fail even on a fresh cluster. Those are real per-query limits and are out of scope
here.

## Root cause

The leak is in the CN's exchange layer (Rust). The C++ engine is not the cause: on error it
drains its task queues and clears the query's repositories (`sirius_engine::execute` →
`drain_after_error`, then `run_mandatory_cleanup`).

1. **Receivers waiting on senders that never finish.**
   - `LocalExchange` (`local_exchange.rs:96`) holds three maps, all
     keyed by fragment instance:
     - `receivers`: receivers waiting for their senders;
     - `sources`: the data senders have delivered;
     - `remote_seq`: frame sequence counters.
   - Entries leave these maps only in `take_ready` (`:242`), and only once *every* expected
     sender has finished.
   - When a query fails, some senders never send end-of-stream, so the receiver's entries stay
     forever, along with what they hold:
     - `SenderSource::LocalParked { slot }` pins a parked engine fragment in `ParkedRegistry`,
       which holds its GPU output repositories.
     - `SenderSource::Remote { batches }` holds direct-exchange receive buffers (`token`s) in
       `DirectExchange::_entries`. These are freed only by `take`, `release`, or process exit.
2. **No cancellation path.**
   - `ResultStore::fail_query` ([result_store.rs:118](../src/result_store.rs)) only marks
     result slots as failed.
   - The thrift `cancelPlanFragment` returns not-implemented ([lib.rs:551](../src/lib.rs)).
   - The brpc `PInternalService.cancel_plan_fragment` falls through to the generated default,
     `method_not_implemented`.
   - So when the FE cancels a failed query on every CN, nothing happens.
3. **Inputs leaked by errors before the engine runs.**
   - In `execute_ready_fragment` ([compute_node_service.rs:637](../src/compute_node_service.rs)),
     `TokenGuard` (`:951`) releases remote tokens only.
   - If `exchange_inputs`, `translate_fragment_logged`, or the pre-run checks in
     `execute_fragment` fail, the `LocalParked` slots are never released.
   - The release-on-error loop in [engine.rs:161](../src/engine.rs) runs only when `run_fragment`
     itself fails.
4. **NIXL WRITE timeout or failure.**
   - `nixl_transport.rs:366` and `:422` `mem::forget` the transfer
     request, and both batches stay held. This is deliberate, because the NIC may still be
     writing.
   - The cleanup loop (`:292`) sends `Release` to the peer only when `complete()` succeeds, so
     the peer's receive buffers leak too.

## Goal and non-goals

**Goal:** after a query fails or is cancelled, the CN releases everything it holds for that
query: waiting receivers, parked sender outputs, received NIXL buffers, and sequence state. Once
the CN is idle again, its leftover counters are back to zero.

**Non-goals:**
- making q05, q08 or q09 fit at SF3000 (that needs spilling, rebalancing, or more GPUs);
- interrupting a fragment the engine is already running. A running fragment finishes or fails
  on its own, and its guards free its inputs.

## Design

### Query identity

Record each fragment instance's query when it is first seen:

- **Receivers** carry `exec.query_id` at `register_receiver`.
- **Sources** can arrive before their receiver registers, keyed only by `finst_id`. Use the FE's
  invariant here: a fragment instance ID has the same `hi` as its query ID, and
  `lo = query.lo + n` (`ExecutionDAG.setInstanceId`, `ExecutionDAG.java:610`).
- Store an explicit `instance → query` map wherever the query is known. Match on `hi` only for
  orphan sources that have no receiver yet, and document that dependency next to the code.

A sturdier alternative is to add `query_id` to `NixlEnvelope::Packed`, since we own both ends
of that format. Prefer this if the `hi` invariant looks fragile; it changes the wire format.

### `LocalExchange::purge_query(query) -> Purged`

A new method that, under the existing lock:

1. removes the query's `receivers` entries;
2. removes its `sources`, collecting `LocalParked` slots and `Remote` batch tokens;
3. removes its `remote_seq` entries;
4. adds the query to a bounded **tombstone** set (for example the last 1024 queries, or a
   10-minute TTL).

It returns `Purged { slots, tokens }` and frees nothing itself. The caller then:
- releases tokens with `nixl.release(token)`;
- releases slots with `executor.drop_parked(slot)`.

All GPU frees therefore happen outside the exchange lock and on the threads that own those
resources: the engine thread for parked fragments, `DirectExchange` for tokens.

### Tombstones stop late frames

After a purge, a late `push_remote_frame` or `push_sender` for that query would recreate
entries through `entry().or_default()`. With the tombstone set:

- `push_remote_frame` rejects the frame. `handle_nixl_chunk` already releases the token of a
  refused Packed frame (`:317`).
- `push_sender` returns the slot so the caller can drop it.
- `register_receiver` for a tombstoned query fails the query immediately.

### Triggers

1. **Local failure:** at the existing `fail_query` call sites:
   - `process_fragment` register error, `:290`;
   - `drain_ready` receiver error, `:614`.
   
   Also call `purge_query` when a leaf fragment fails in `process_fragment`/`execute_fragment`.
   Wrap both calls in one `fail_and_purge(query, err)` helper so they can't drift apart.
2. **FE cancel:** implement the brpc `cancel_plan_fragment` in `SiriusComputeNodeService`.
   - Read `query_id` from `PCancelPlanFragmentRequest`; it's field 11. If it's missing, derive
     the query from `finst_id.hi`.
   - Call `fail_and_purge`.
   - Answer OK. It must be idempotent, because the FE sends one cancel per instance.
   - Make the thrift `cancelPlanFragment` call the same path.
3. **Fetch-data timeout:** if `fetch_data` gives up on a waiting slot, purge that query too.

### Guard for inputs before the engine runs

Replace `TokenGuard` with a `ReadyInputsGuard` that owns both the remote tokens and the
`LocalParked` slots of a `ReadyFragment`.
- On drop it releases both kinds.
- Once the inputs are handed to `executor.run`, it is disarmed (`std::mem::take` the slots).
  From then on the engine owns them, and the release loop in `engine.rs:161` covers engine-side
  failures.

This closes root cause 3.

### NIXL timeout reclaim (lower priority)

Don't `forget` timed-out requests outright. Put them in a `quarantine` list together with their
local and remote tokens. A reaper (the transport thread, between sends) polls
`get_xfer_status`:
- when the status is terminal, it releases the request and the local batch, and sends `Release`
  to the peer;
- after a hard limit (for example 10× the timeout), it logs and gives up as today.

This closes root cause 4. It doesn't affect the SF3000 symptom, where no timeouts were seen, so
it goes last.

## Observability (do first)

Add a single `leak_counters()` snapshot:

| counter | source |
|---|---|
| `exchange.receivers`, `exchange.sources`, `exchange.remote_batches` | `LocalExchange` |
| `parked.fragments`, `parked.slots` | `ParkedRegistry` (engine thread, via a new `EngineRequest::Stats`) |
| `direct.outstanding` | `DirectExchange::outstanding()` (already exported, `sirius_ffi.cpp:356`; implemented in `src/exec/exchange_direct.cpp:387`) |
| `nixl.quarantined` | transport (once the reclaim work lands) |

- Log it at `info` after every `fail_and_purge` and every completed result fetch.
- Serve it on the CN HTTP port (`/debug/leaks`) so scripts can assert on it.
- Run once **before** the fix to confirm the counters are non-zero after run A's q05.

## Work plan

Each step ends in a state that can be verified on its own:

| # | change | verify |
|---|---|---|
| 1 | `leak_counters()` and logging; turn on `SIRIUS_LOG_BACKEND` in the run scripts behind an option | rerun experiment A: counters stay non-zero after q05 fails and the CN is idle |
| 2 | `LocalExchange::purge_query`, tombstones, `Purged` | unit tests in `local_exchange.rs` (below) |
| 3 | `fail_and_purge` at the local failure sites | experiment A: q07 passes; counters back to 0 after q05 |
| 4 | brpc and thrift `cancel_plan_fragment` | 2-CN test: fail a fragment on CN0; CN1's counters go to 0 without a local error |
| 5 | `ReadyInputsGuard` | unit test: translation error on a ready receiver releases its parked slots |
| 6 | NIXL quarantine and reaper | inject a WRITE timeout (`SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS=0`); the quarantine drains |
| 7 | e2e regression in the SF1000/SF3000 scripts: assert `/debug/leaks` is zero after every query | full SF3000 run: q07 and q12 pass in sequence; SF1000: 7/7 still pass |

Steps 1–3 fix the observed symptom. Steps 4–5 cover the other ways a query can fail. Step 6
hardens the transport.

## Tests

**Unit (`local_exchange.rs`):**
- `purge_query` returns every slot and token of the query and none of another query's;
- purging an unknown query is a no-op;
- after a purge, a late remote frame is refused, a late local sender gets its slot back, and a
  late receiver registration fails;
- the tombstone set stays bounded.

**Service (`compute_node_service.rs` tests, fake executor and fake NIXL):**
- a failed receiver releases its own inputs *and* every other pending entry of its query;
- `cancel_plan_fragment` is idempotent and releases tokens through the fake NIXL;
- a translation error releases parked slots.

**Failure injection for CI (SF1, 2 CNs):**
- add `SIRIUS_CN_FAIL_FRAGMENT=<node_id>` (test-only) so a chosen fragment returns an error;
- extend `2cn_tpch_joins.sh` to run a query with injection, then the same query without it, and
  check that the second passes and the counters read zero.

**Scale validation:**
- SF3000 full run: q07 and q12 move from FAIL to PASS (4/7 expected);
- experiment A passes;
- SF1000 stays 7/7.

## Risks and open questions

- **Allocator accounting.** Direct-exchange receive buffers are allocated under a temporary
  reservation that is then detached (`src/exec/exchange_direct.cpp:309-329`). Confirm in cucascade that
  freeing a detached allocation returns the bytes to the tracker the next query reserves from.
  Otherwise memory is freed in RMM but still counted as used.
- **Fragmentation.** Even with everything freed, a 156 GiB pool can end up fragmented after a
  failure. Step 3's verification will show whether q07 passes after a purge. If it doesn't
  while the counters are zero, look at pool fragmentation next.
- **Running fragments.** A fragment already on the engine thread can't be stopped by a purge.
  Its own guards release its inputs when it ends; its outputs may park after the purge, and the
  tombstone check in `push_sender` must drop them.
- **Two FE invariants:** the `hi` match between instance and query IDs, and the FE always
  sending cancels on failure. If either is in doubt, carry `query_id` in the Packed envelope.

## Out of scope / follow-ups

- **q05, q08, q09 at SF3000.** Each fails on backend 10001 (GPU 1). Next step: measure the
  per-CN bytes received, to check for skew in the shuffle. Options include operator spill to
  host memory, better partition balance, or more GPUs.
- **Engine logging is off by default in the CN** (no-op sink). Consider turning on `spdlog`
  with a per-CN `SIRIUS_LOG_DIR` in the run scripts, since `sirius.log` has a fixed name.
- **Retry loop.** The 100-retry OOM loop in `gpu_pipeline_executor.cpp` hides the real failure
  for about 5 s per task. Separately, consider failing fast when the pool is full and nothing
  else is running.
