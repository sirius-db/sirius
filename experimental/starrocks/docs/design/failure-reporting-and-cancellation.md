# Plan: failure reporting and cancellation

## Why

How the StarRocks FE (4.1.3) learns that a fragment failed. A failure reaches the coordinator in one of three ways:
- the `exec_plan_fragment` status, if the fragment fails during dispatch;
- `FrontendService.reportExecStatus`, if it fails while running;
- the `fetch_data` status, if the result-sink instance fails.

The first non-OK status fails the query, and the FE then cancels every other instance.

If a fragment fails but its node still answers heartbeats, and nothing reports the failure, the FE waits until the statement timeout. To the user it looks like a hang. The same happens when two nodes can't reach each other but both still answer the FE.

Today the CN uses only the first and third channels, and has no failure signal between CNs.

## Today

Line numbers refer to `f4780e74`. `CNS` is `experimental/starrocks/src/compute_node_service.rs`.

- **Dispatch errors** are returned in the reply (CNS:147). That works for a leaf fragment, which runs before the CN replies.
- **A receiver replies OK before it runs** (CNS:445-448). It runs later in one of two places:
  - inside a local sender's dispatch, where its error lands in that sender's reply;
  - on an `exchange-receiver` thread, where its error is only logged (CNS:721).
- **A scan deferred for a runtime filter** (#2062) only purges locally on error (CNS:538-541).
- **No failure frame between CNs.** The SRNX envelope has `Md`, `Alloc`, `Packed` and `Release` (`nixl_chunk.rs`). A failing sender stops without sending end-of-stream (`nixl_transport.rs:701-727`), and the remote receiver waits with no deadline. Peers learn of the failure only if they try to send more frames to a CN that already purged the query.
- **`fetch_data` waits up to 600 s** for a result (`RESULT_WAIT`, CNS:54).
- **`reportExecStatus` is never called,** and `TExecPlanFragmentParams.backend_num` and `TQueryOptions.query_timeout` are never read.
- **`cancel_plan_fragment`** (CNS:278-299) purges exchange state in the background. It never reads `cancel_reason`, so QUERY_FINISHED / LIMIT_REACH cancels, which the FE sends after many successful queries, are logged as "purged a failed query".
- **A cancel that overtakes dispatch:**
  - late receivers and frames are refused (`local_exchange.rs:175,200,232`);
  - a late leaf fragment still runs fully on the GPU;
  - the purged-query list is a FIFO of 1024 entries.
- **No interruption.** One engine thread runs one request at a time (`engine.rs:192`), and `fragment.run()` blocks to completion (TODO at `fragment_executor.rs:204-208`). Freeing a purged query's parked output also goes through that thread, so it waits for whatever fragment is running. At SF3000 this held a failed q09's memory for 55 seconds after its cancel (Step 4).
- **Reusable piece:** `report_to_frontend_once` (`lib.rs:1068-1150`) already opens a `FrontendServiceSyncClient` to the FE address learned from the heartbeat.

## Steps

### Step 1: a failure frame between CNs

- Add `Failed { error }` to `NixlEnvelope` (`nixl_chunk.rs`), sent in place of end-of-stream.
- On a sender failure, `announce_all` (`nixl_transport.rs`) sends it to every hop that hasn't sent end-of-stream yet.
- On receipt, the receiving CN fails the pending exchange with that error and calls `fail_and_purge`. Its result slot, if any, fails at once.
- **Tests:**
  - a CN unit test: a receiver fed by a sender that fails remotely fails at once, and the leak counters return to zero;
  - in the harness: `INJECT_FAILURE_QUERY` on a non-result fragment fails the query in seconds, not at 600 s.

### Step 2: `reportExecStatus` for failures

- Extract the client setup from `report_to_frontend_once` into a helper.
- Add `report_exec_status(coord, query_id, backend_num, fragment_instance_id, status, done)` in a new `fe_report.rs`.
- Record `backend_num`, `coord` and the instance id per instance at dispatch.
- Send **one** final report with a non-OK status and `done = true` from every place that today only logs or purges:
  - the `exchange-receiver` thread (CNS:721);
  - deferred scans (CNS:538);
  - the Step 1 failure path.
- The FE drops anything after `done = true`, so send exactly one final report per instance.
- Treat a NOT_FOUND reply as success: the FE has already finished the query.
- **Tests:**
  - a fake `FrontendService` in a CN unit test receives exactly one report with the right `backend_num`;
  - end to end, a failure injected into a receiver run on another CN fails the query fast with that CN's error.

### Step 3: cancel the way the FE uses it

- Read `cancel_reason`:
  - QUERY_FINISHED and LIMIT_REACH purge quietly, with no failure log and no report;
  - INTERNAL_ERROR, a timeout or a user kill purge as failures.
- At the top of `process_fragment`, check the purged-query list, so a late fragment of any kind is refused before it runs.
- Size the purged list by time (the query timeout) rather than a fixed 1024 entries.
- Keep cancel idempotent: it must be safe before dispatch, during the run, and after the instance has finished.
- **Tests:** CN unit tests for cancel-before-dispatch (no GPU work), cancel-after-finish (no error), and a QUERY_FINISHED reason (not logged as a failure).

### Step 4: release a cancelled query's GPU memory promptly

**Evidence (SF3000, all 7 join queries in sequence, Oct 8).** A failed q09 left its GPU memory on CN 3 for 55 seconds after the cancel, long enough to fail q12. The memory was released in the end, so #2041's purge is complete but not prompt. CN 3's log:

| time | CN 3 |
|---|---|
| 01:24:34 | q09's filtered `lineitem` scan starts on the engine thread |
| 01:25:24 | q09 fails on another CN; the FE's cancel arrives. The purge removes q09's exchange state and releases its 75 NIXL buffers at once, but **4 parked fragments stay** (`parked_fragments=4`) |
| 01:25:24–01:26:19 | The scan keeps running, retrying an out-of-memory error up to 100 times (`gpu_pipeline_executor.cpp:400`) |
| 01:25:46 | q12 starts; its exchange buffers on CN 3 fail with "0 available" |
| 01:26:19 | The scan gives up. The purge, blocked until now, drops 3 parked fragments; the scan's own fragment frees the 4th |
| 01:26:23 | Every counter is zero; q19 then passes |

**Three causes:**
- **A running fragment can't be stopped.** Nothing tells Sirius the query was cancelled, so the scan retries for 55 seconds. Part of the memory it waits for is held by its own query's parked fragments.
- **Freeing parked output needs the busy engine thread.** `fail_and_purge` frees each parked fragment with `executor.drop_parked(slot)`, a synchronous request to the CN's single engine thread (`engine.rs:233`). That request waits behind the running fragment, and the purge thread waits with it.
- **The next query fails instead of waiting.** A receive allocation that finds the pool full fails at once, even while memory of a purged query is about to be freed.

**Fixes, in order:**
1. **Interrupt the running fragment (engine and FFI).**
   - The FFI already holds a `duckdb::Connection` (`sirius_ffi.cpp:206`). Add `Context::interrupt()`, which calls `Connection::Interrupt()`. DuckDB designed that call to be made from another thread.
   - Today Sirius checks `IsInterrupted()` only when a query enters its lifecycle window (`sirius_context.cpp:1855,1868`). Add the check to the pipeline executor, both before scheduling a task and before rescheduling an out-of-memory retry, so a cancelled fragment stops within one task.
   - Clear the flag when the next fragment starts.
   - The CN calls `interrupt()` from `fail_and_purge`, only when the fragment running on the engine thread belongs to the purged query.
2. **Don't block the purge on the engine thread (CN).** Send `DropParked` without waiting for the reply, so the purge returns at once and the drops run as soon as the engine thread is free. With fix 1, that's within one task. Better still, register a local fragment's parked output as direct-exchange tokens, the way remote outputs already are; then the purge frees it from any thread, as it already does for the 75 NIXL buffers.
3. **Let an allocation wait for a pending purge (CN).** Count purges whose drops haven't run yet. A receive allocation that finds the pool full while that count is non-zero waits for it to reach zero (bounded by the query timeout), then retries. A late release then delays the next query instead of failing it.
4. **Harness:** report how long each CN took to return to zero, not just whether it did so within 20 s. That separates a late release from a real leak.

- **Tests:**
  - an FFI test: `interrupt()` stops a fragment that is retrying out-of-memory, within one task;
  - a CN unit test: a purge with parked fragments returns while the engine thread is busy;
  - a CN unit test: a receive allocation waits for a pending purge, then succeeds;
  - at SF3000, all 7 join queries in sequence: q12 passes after a failed q09, and the leak check reaches zero within a few seconds.

### Step 5: deadlines from the query timeout

- Read `TQueryOptions.query_timeout`, and use it to bound receiver waits and `fetch_data` instead of the fixed 600 s.
- **Test:** a receiver whose sender never arrives fails at the query timeout.

## Order

Do 1 → 2 → 3 → 4 → 5.
- Steps 1–3 are CN-only.
- Step 4 touches the engine and the FFI.
- Steps 1 and 2 are prerequisites for streaming receivers. Once a receiver replies to dispatch before it has data, `reportExecStatus` is the only way its failure reaches the FE.

## Out of scope

- **Success reports and profiles.** SELECT returns on `fetch_data` end-of-stream. Final reports only finish the profile, and INSERT/loads wait for them, but we don't support INSERT.
- **`reportFragmentFinish`** (`enable_phased_scheduler`, off by default).
- **Streaming receivers** (Oct 2 item 8). That's its own plan, after this one.
