# TPC-H SF1 two-CN sweep: hangs, loud refusals, and the cluster cascade

Branch `demo-multi-cn` @ `830380f4`. Evidence: `/tmp/sirius-tpch-bench/bench/A/` (sweep of
2026-08-07 00:35–00:57) plus four fresh single-query reproductions on `pixi run cluster2`
(2026-08-07 01:17–01:35).

## 1. Executive summary

Four queries (q02, q03, q07, q08) returned nothing for 300 s in the sweep. Only **two of them**
are actually broken; q03 and q08's sweep results were collateral damage from an earlier query
that had already wedged a compute node.

Run one at a time on a clean cluster:

| query | solo behaviour | class |
|---|---|---|
| q02 | hangs (120 s, no error) | cross-CN staging-lease deadlock |
| q03 | **fails loudly in 2 s** | sort-tuple column-order defect (same as q05) |
| q07 | hangs (sweep evidence) | `year()` wire-type mismatch |
| q08 | hangs (120 s, no error) | `year()` wire-type mismatch |

Every hang has the same shape: a fragment fails, the CN logs the failure at ERROR, and the FE is
never told. Whether a failure is loud or silent is decided by **which fragment failed** — a
RESULT_SINK fragment has an FE-visible channel (`fetch_data`), an intermediate receiver fragment
has none. It is not decided by transport leg: the same guard fires on both the relay leg and the
packed leg, and both are silent when they land on an intermediate fragment.

A silent failure then wedges the whole compute node for 600 s, because the FE's `fetch_data`
long-poll blocks the CN's single per-connection frame loop. Every later query — including ones
that only need the CN for *planning* — queues behind it. That is what turned two real defects
into a 20-query wipe-out.

## 2. Failure classes observed

### Class A — silent hang (q02, q07, q08)

A non-root fragment fails; the error is logged and parked, never reported.

Sweep, q02 (`019fd9a5-66d7-…`):
```
00:37:27.156 ERROR dispatched intermediate receiver fragment failed; its downstream result
  fragment will wait until the FE times out
  error=request_staging_lease to 127.0.0.1:8060 (after reconnect):
        failed to read reply frame: failed to read PRPC header
```
Sweep, q07 (`019fd9ae-9dee-…`):
```
00:45:28.032 ERROR dispatched intermediate receiver fragment failed; …
  error=failed to relay sender 0: Fragment: relay into stream 24 column 2
        is declared SMALLINT but the source sink produces BIGINT
```
Fresh solo repro, q08 (rc=124 after 120 s):
```
01:21:10.470 ERROR dispatched intermediate receiver fragment failed; …
  error=failed to relay sender 0: Fragment: relay into stream 34 column 0
        is declared SMALLINT but the source sink produces BIGINT
```
Fresh solo repro, q02 (rc=124 after 120 s) reproduced the sweep byte-for-byte — same stream ids
(7/22/26/29/31), same 10,256,640-byte transfer, same `request_staging_lease` failure after two
60 s attempts. Deterministic.

The client sees nothing at all: `q02.r0.out`, `q07.r0.out`, `q08.r0.out` are 0 bytes.

### Class B — loud refusal (q05, and q03 when run alone)

Same guard, but the failing fragment is the RESULT_SINK fragment, whose reserved `fetch_data`
entry carries the error to the FE.

Sweep, q05, 772 ms:
```
ERROR 1064: fragment instance 019fd9ae-97bc-… failed: failed to relay sender 0:
  Fragment: relay into stream 30 column 0 is declared VARCHAR but the source sink produces DOUBLE
```
Fresh solo repro, q03, 2 s:
```
ERROR 1064: fragment instance 019fd9ce-cd8e-… failed: failed to relay sender 0:
  Fragment: relay into stream 14 column 0 is declared DATE but the source sink produces DOUBLE
```
q03's sweep entry (`wedge`, 300 005 ms) is therefore a **misclassification**: q03 is a Class B
query that never got to run.

### Class C — cluster cascade (q03, q08 in the sweep; q09; q10–q22)

After a Class A hang the CN stops answering anything for 600 s. Sweep trace, CN1's FE-facing
connection unblocking at 00:45:24 — exactly 600 s after q02 stalled at 00:35:24:
```
handle_connection{peer=127.0.0.1:43040}:fetch_data:        close time.busy=16.0µs time.idle=600s
handle_connection{peer=127.0.0.1:43040}:get_file_schema:   close time.busy=104µs  time.idle=177µs
handle_connection{peer=127.0.0.1:43040}:exec_plan_fragment:close time.busy=8.30µs time.idle=230µs
handle_connection{peer=127.0.0.1:43040}:                   close time.busy=7.65ms time.idle=612s
```
`get_file_schema` and `exec_plan_fragment` were sent by the FE minutes earlier and sat unread
behind `fetch_data`; they were served in under a millisecond once it returned. q03 spent its
entire 300 s inside that window, blocked in **FE planning**, not execution — its coordinator was
never created (no `DefaultCoordinator.cancel … because client closed` on client close, unlike
q02). q08 sat inside the identical 600 s window opened by q07 on CN2 (`peer=45046`,
`fetch_data … time.idle=600s`).

The same head-of-line block hits the CN↔CN leg: `handle_connection{peer=127.0.0.1:35908}:
request_staging_lease: … time.idle=1341s`.

Live confirmation: on a clean cluster q06 runs in ~190 ms; run immediately after q08 wedged the
cluster it timed out at 90 s.

q09's error is a *misattributed* reply from the poisoned channel:
```
ERROR 1064: Access storage error … failed to get file schema:
  errorCode=2001 errorMessage:method 'cancel_plan_fragment' is not implemented by the Rust CN
```
The FE stack trace (`FragmentInstanceExecState.waitForDeploymentCompletion:291` →
`ProtobufRpcProxy$2.get`) shows the error arriving on a *deployment* call, and the netty handler
raising `NullPointerException` in `RpcClientServiceHandler.channelRead0` — the shared jprotobuf
channel is left inconsistent after the timed-out futures are reaped. q10–q22 then fail at
planning with `No alive backend or compute node in warehouse default_warehouse`.

### Residual state

The CN left stranded by the sweep was still alive 40 minutes later holding **9.4 GiB of GPU
memory**, and ignored `SIGTERM` — it required `SIGKILL`. Same on both teardowns during this
investigation. Shutdown joins the engine thread, which is inside the stalled fragment.

## 3. Root cause per class

### CONFIRMED — Class A propagation gap: `compute_node_service.rs:542-575`

`ServiceCore::run_ready_fragment` branches on fragment role:

- `:549-554` result fragment → `self.results.fail(id, error)` on the id the FE polls → loud.
- `:555-567` intermediate receiver → `self.results.fail(id, error)` on an id **nobody polls**,
  plus an ERROR log that says so verbatim: *"its downstream result fragment will wait until the
  FE times out"*.

**The exact gap**: the failure is never forwarded from the intermediate fragment instance to the
downstream result fragment instance, and the CN has no FE-facing status channel at all — it
implements 7 PRPC methods (`compute_node_service.rs:215,242,271,340,374,408,432`) and
`report_exec_status` is not one of them. Nothing exists to carry a fragment failure to the
coordinator. This is the pre-existing gap `TWO-PHASE-AGG-PLAN.md` §5 non-goal 6 lists as "the
failed-sender cancellation gap".

Note the CN *does* already hold an FE thrift client and the FE address
(`src/lib.rs:22,242,1132`, used for `FrontendService.report` after heartbeat), so the plumbing
for a real status report exists.

### CONFIRMED — Class A trigger #1: `year()` return-type mismatch (q07, q08)

Minimal reproduction, built and run during this investigation:

```sql
WITH orders AS (SELECT * FROM FILES("path"="…/tpch_sf1/orders/*.parquet","format"="parquet"))
SELECT EXTRACT(year FROM o_orderdate) AS y, count(*) AS c FROM orders GROUP BY y ORDER BY y;
```
→ silent hang (rc=124 at 90 s). CN log, **both legs**:
```
relay into stream 3 column 0 is declared SMALLINT but the source sink produces BIGINT
packed batch for stream 3 column 0 (col_10) is declared SMALLINT (int16_t) but carries int64_t
```
Control with a non-`year` grouping key (`GROUP BY o_orderstatus ORDER BY o_orderstatus`) returns
correct counts in 2 s.

StarRocks' `year()` returns SMALLINT; DuckDB/Sirius returns BIGINT. The translator maps
`year|month|day` straight through with no return-type cast — `expr_translator.rs:602`:
```rust
"year" | "month" | "day" => (URN_DATETIME, name),
```
The FE-declared slot stays SMALLINT, the engine produces BIGINT, and the C1 guards
(`src/sirius_ffi.cpp:616` relay, `:759` packed) refuse the hop. This is the *same class* as the
HUGEINT residual closed by `bb066e90`, which fixed it for merge measures by wrapping them in
throwing casts to the FE-declared output-slot types — scalar functions in an ordinary projection
have no such cast.

### CONFIRMED — Class B trigger: sort-tuple column order (q03, q05)

Already documented in `TWO-PHASE-AGG-PLAN.md` §8 (2026-08-07 entry). The sender projects
`sort_tuple_slot_exprs` in FE list order (order-by keys first) —
`node_translator.rs:1219-1235`, `project_rel(child, expressions, vec![sort_tuple])` — while the
receiver derives the stream schema from the sort tuple's materialized slot order. q05
(`ORDER BY revenue DESC`) and q03 (`ORDER BY revenue DESC, o_orderdate`) both put a non-leading
slot first. Loud, not wrong results.

### CONFIRMED (mechanism) / HYPOTHESIZED (precise interleaving) — Class A trigger #2: staging-lease deadlock (q02)

Every engine interaction, including staging-arena leases, is funnelled through one thread:
`engine.rs:67` — *"the only caller of `SiriusContext`, which is `!Send`"*; the thread is spawned
at `engine.rs:126`; `staging_lease` / `staging_release` are ordinary engine requests
(`engine.rs:270,276`) sent over the same channel and blocked on (`engine.rs:587-602`).

`compute_node_service.rs:371-372` states the consequence in its own doc comment:
> *"The lease request queues behind whatever the engine thread is running; the peer's client
> timeout bounds the wait."*

So when CN1's engine thread is inside a fragment that is waiting for input from CN2, and CN2's
sender needs a staging lease from CN1 to deliver exactly that input, neither can proceed. The
60 s `REPLY_TIMEOUT` (`prpc_client.rs:22-25`) breaks it twice (once on the cached connection,
once after reconnect), then the sender fails — on an intermediate fragment, so silently.

The two-minute silence between q02's last engine event and the error (identical in the sweep and
the fresh repro) matches 60 s + 60 s exactly. CONFIRMED as a mechanism and reproducible;
HYPOTHESIZED is only *which* fragment pair closes the cycle — confirm with
`SIRIUS_CN_DUMP_FRAGMENTS` plus a per-fragment `run()` enter/exit trace.

### CONFIRMED — Class C: head-of-line blocking, `brpc.rs:150-193` + `compute_node_service.rs:271-297`

`handle_connection` is a strictly serial loop per connection: read frame → `await
call_service` → write response → read next frame. The FE multiplexes **all** its RPCs to a CN
over one shared channel (`ProtobufRpcProxy.proxy():261 Use global share channel pool`), so one
slow handler blocks everything else on that CN, including `get_file_schema` used during FE
planning of *unrelated* queries.

`fetch_data` is exactly such a handler: `compute_node_service.rs:284` long-polls
`wait_ready(id, Duration::from_secs(600))`. 600 s is hardcoded and unrelated to the query's
timeout.

## 4. The cascade mechanism

1. An intermediate fragment fails; the error is parked under an id nobody polls
   (`compute_node_service.rs:555-567`).
2. The FE's `fetch_data` on the result fragment blocks for the full 600 s
   (`compute_node_service.rs:284`).
3. The CN's per-connection frame loop (`brpc.rs:156-192`) stops serving that connection —
   FE→CN and CN→CN alike (`request_staging_lease … time.idle=1341s`).
4. The FE's jprotobuf futures time out (`RpcTimerTask … timeout with bound channel`), the CN
   answers minutes later, and replies land against the wrong pending calls
   (`RpcClientServiceHandler.channelRead0` NPE). Queries fail with errors that belong to other
   RPCs — that is why q03 and q09 report *"failed to get file schema: … method
   'cancel_plan_fragment' is not implemented"*.
5. The FE gives up on the backends: `DefaultCoordinator.cancel … because No alive backend or
   compute node in warehouse default_warehouse` → q10–q22 fail at planning with
   *"No available backends"*.

**Where `cancel_plan_fragment` is missing.** Two surfaces, both unimplemented:

- **brpc / PInternalService** — not among the router's implemented methods
  (`src/compute_node_service.rs:215-432`), so the generated router returns a PRPC-level error
  frame, `build.rs:125` → `prpc::Error::method_not_implemented` → text at `src/prpc.rs:130-134`.
  This is a *transport* error, not a StatusPB failure, which is why it poisons the channel.
- **thrift / BackendService** — `src/lib.rs:554` `handle_cancel_plan_fragment` returns
  `not_implemented_status("BackendService.cancelPlanFragment")`.

Because the FE cannot cancel, a stranded fragment holds its GPU memory and its engine thread
until the process is `SIGKILL`ed.

## 5. Proposed solutions, ranked

### (a) Propagate every fragment failure to the FE — highest value

**a1 — intra-CN propagation (do first).** When `run_ready_fragment` fails an intermediate
receiver, fail every result-fragment instance registered for the same query on this CN, not just
the failing id. The exchange registry already holds the receiver set; fragment instance ids share
the query-id prefix. Also wake `wait_ready` immediately.
Files: `src/compute_node_service.rs:542-575`, `src/result_store.rs` (query-scoped `fail`).
Effort: **S** (half a day). Risk: **low** — strictly widens an existing error path.
Effect: q02, q07, q08 fail loudly in ~1 s instead of hanging 300 s; no wedge, so no cascade.
Limitation: does not cover a failure on CN B whose result fragment lives on CN A.

**a2 — FE status report (do next).** Implement `FrontendService.reportExecStatus` with
`done=true` and the error status for the failed instance. The CN already has the FE thrift
client and address (`src/lib.rs:22,242,1132`).
Files: new FE-report call site in `src/compute_node_service.rs`, client in `src/lib.rs`.
Effort: **M** (1–2 days). Risk: **medium** (FE contract). Effect: any fragment failure on any CN
reaches the coordinator immediately; makes a1's limitation moot and removes reliance on
`fetch_data` as the only error channel.

### (b) Implement `cancel_plan_fragment`

**b1 — stub that succeeds (do immediately, S, ~1 h).** Register the PRPC method and return OK.
Files: `src/compute_node_service.rs`. Risk: **low**. Effect: stops the channel poisoning and the
misattributed errors on q03/q09; removes the "not implemented" cascade even before real
cancellation exists.

**b2 — best-effort terminate + free (M).** On cancel: mark the query's instances failed, wake
`wait_ready`, drop registered receivers and parked sender batches, release staging leases.
Files: `src/compute_node_service.rs`, `src/result_store.rs`, `src/lib.rs:554`.
Risk: **medium** — must not free buffers the engine thread is still using, so this is
bookkeeping-only; aborting a fragment mid-`run()` needs a cancellation token in the engine and is
a separate item.

### (c) The wire-type defects (make the queries actually pass)

**c1 — sort-tuple column order** (task #22, mechanism in `TWO-PHASE-AGG-PLAN.md` §8). Reorder
the sort projection to materialized-slot order, or override the receiver schema positionally like
the merge pre-pass. File: `node_translator.rs:1219-1235`. Effort: **S**. Risk: **low-medium**
(sort semantics). Effect: q03, q05 → pass.

**c2 — scalar-function return types.** `year|month|day` (`expr_translator.rs:602`) — and any
other FE-narrowed scalar — must cross the boundary as the FE-declared slot type. Reuse the
`merge_projection` pattern from `bb066e90`: wrap projection outputs in throwing casts to the
FE-declared slot types. Effort: **S-M**. Risk: **low** (the cast is checked; a wrong prediction
still fails loudly). Effect: q07, q08 → pass, or reveal the next blocker.

**c3 — extend the parity gate.** `experimental/starrocks/src/wire_type_parity.rs` covers
aggregate wire types only. Extend it to scalar functions crossing a fragment boundary; c2 would
have been a CI failure rather than a 300 s hang. Effort: **S**. Risk: **none**.

### (d) Defense in depth

**d1 — concurrent per-connection RPC handling.** `brpc.rs:150-193` must spawn each request and
write replies by correlation id instead of awaiting inline (a write mutex serialises the
socket). Effort: **S-M**. Risk: **medium**. Effect: one slow query can no longer block another
query's planning or dispatch on the same CN — this is what turned two defects into 20 failures.

**d2 — bound `fetch_data` by the query timeout.** Replace the hardcoded 600 s at
`compute_node_service.rs:284` with `query_options.query_timeout`. Effort: **S**. Risk: **low**.
Effect: a wedge clears in seconds instead of ten minutes.

**d3 — per-query watchdog.** Fail any fragment instance with no progress within N seconds, with
the last observed stream/sender in the message. Effort: **M**. Risk: **low**. Catches hang modes
nobody has enumerated yet.

**d4 — take the staging arena off the engine thread** (q02's root cause). Serve
`request_staging_lease` / `staging_release` from a separate arena mutex so a peer can always
obtain a lease while the engine thread is inside a fragment. Files: `src/engine.rs` (arena
ownership — currently inside the `!Send` `SiriusContext`), `src/compute_node_service.rs:371-403`.
Effort: **M**. Risk: **medium-high**. Cheaper interim: have the receiver pre-lease at fragment
build so no lease is ever requested while its engine thread is busy.

### (e) Harness

`bench.sh` must restart the cluster after any non-pass result. One wedge invalidated 20 of 22
queries in this sweep, and the recorded errors for q03/q09 were not even their own.

## 6. What each fix does per query

| fix | q02 | q03 | q05 | q07 | q08 | q09–q22 |
|---|---|---|---|---|---|---|
| (a1) intra-CN propagation | hang → loud, ~1 s | already loud solo | — | hang → loud | hang → loud | no longer collateral; sweep completes |
| (a2) FE status report | reinforces a1 | — | — | reinforces a1 | reinforces a1 | — |
| (b1) cancel stub | — | removes misattributed error | — | — | — | removes "not implemented" cascade |
| (b2) best-effort cancel | frees GPU/CN | — | — | frees GPU/CN | frees GPU/CN | no "No available backends" |
| (c1) sort-tuple order | — | **pass** | **pass** | — | — | — |
| (c2) scalar return types | — | — | — | **pass** | **pass** | may unblock others using `year()` |
| (d1) concurrent RPC | — | never blocked in planning | — | — | never blocked in planning | wedges stay query-local |
| (d2) timeout-bounded poll | 600 s → seconds | — | — | 600 s → seconds | 600 s → seconds | — |
| (d4) arena off engine thread | **pass** or next blocker | — | — | — | — | — |

Suggested order: **b1 → a1 → c2 → c1 → d2 → d1 → a2 → b2 → d4**. The first two are under a day
together and convert every observed silent hang into a loud, attributable error.

## 7. Reproductions used

All on `pixi run --manifest-path experimental/starrocks/pixi.toml cluster2`, one query per
cluster lifetime, cluster fully restarted between runs.

| # | query | result |
|---|---|---|
| 1 | q03 solo | `ERROR 1064 … stream 14 column 0 declared DATE, source sink produces DOUBLE`, 2 s |
| 2 | q08 solo | rc=124 at 120 s; CN: `stream 34 column 0 declared SMALLINT … produces BIGINT` on an intermediate fragment |
| 3 | q06 after (2) | rc=124 at 90 s (passes in ~190 ms on a clean cluster) — cascade confirmed |
| 4 | q02 solo | rc=124 at 120 s; CN: `request_staging_lease to …:8060 (after reconnect)` — byte-identical to the sweep |
| 5 | `GROUP BY EXTRACT(year …)` minimal | rc=124 at 90 s; relay **and** packed guards both fire |
| 6 | `GROUP BY o_orderstatus` control | correct counts, 2 s |

## Post-fix status (2026-08-07, commits c858e79a / 4beca977 / 4323197d)

Sweep after the b1+a1+c2+c1 fixes (3 warm runs, 30 s ceiling, restart-on-failure):
**15 pass / 6 loud refusals / 1 wedge — zero cascade** (every query after a failure passed).

Pass (warm median ms): q01 338, q04 400, q05 1175, q06 327, q07 974, q08 1236, q11 851,
q12 408, q13 499, q15 741, q17 1033, q19 831, q20 892, q21 1507, q22 645.

Remaining failures:
- q02 wedge: the staging-lease deadlock (d4) stands; the propagation fix does deliver its
  error to the FE, but only after ~124 s (2× PRPC reply timeout) — invisible at a 30 s
  client gate — and the CN afterwards holds 9.4 GiB GPU and ignores SIGTERM (needs b2,
  engine-side abort).
- q03 refused 1.6 s: the sort fix cleared the q05 shape; q03's two-key sort still
  misdeclares the hop (now DATE vs BIGINT, was DATE vs DOUBLE).
- q09 refused: exchange staging arena exhausted (648 MB requested vs 512 MB capacity).
- q10 refused: hash partition key DECIMAL(15,2) unsupported.
- q14 refused: unsupported TExprNodeType(29).
- q16 refused: descriptor slot 2 (tuple 8) not in row_tuples [9].
- q18 refused: stream declared INTEGER, source produces VARCHAR.

New correctness finding (pre-existing, highest-priority follow-up): every
sum(l_extendedprice*(1-l_discount)) lands ~0.1 % low while counts/base sums/avgs on the
same rows are exact; a DuckDB simulation of "rows with l_discount=0.10 compute (1-d) 0.01
low" reproduces q01 A|F to 3.6 ppm. Decimal literal/scale suspect in the multi-fragment
expression path. (Tracked as task #24.)

## Final status (2026-08-07)

Second fix wave, one commit per remaining refusal class from the table above:

- `4323197d` fixed sort tuples; `7bdcd312` applies the same materialized-slot-order rule to
  aggregation grouping keys — clears the q03 refusal (DATE vs BIGINT hop).
- `a94e8660` serves staging leases off the engine thread (fix d4) and adds SIGTERM→SIGKILL
  escalation — a wedged CN no longer survives teardown holding 9.4 GiB of GPU memory.
- `8c23e7e7` hashes DECIMAL partition keys through a FLOAT64 cast — clears the q10 refusal.
- `1d4428da` grows the exchange staging arena to 1280 MiB per CN — clears the q09 exhaustion
  (648 MB requested vs 512 MB capacity).
- `90750142` — harness waits for both backends and kills the right CN binary (no more phantom
  wedges from a half-started cluster).
- **Pending commit** (working tree: `descriptor_table.rs`, `expr_translator.rs`,
  `tests/translate.rs`): CLONE_EXPR (TExprNodeType 29) translation plus a unique-slot-id
  fallback for slot refs whose tuple is not in `row_tuples` — clears q16 (the earlier
  "descriptor slot 2 (tuple 8)" refusal); q14 hits a new blocker behind it (below).

Full sweep (22 queries × 3 timed runs, harness `90750142`, 30 s ceiling; CSV
`/tmp/sirius-tpch-bench/bench/A4/timings.csv`):

| Q | result | ms (3 runs) | rows | notes |
|---|---|---|---|---|
| q01 | pass | 347/337/368 | 4 | max value drift 0.096 % (#24, in band) |
| q02 | **wedge** | 30004 (run0) | 0 | #26 engine-thread stall (below) |
| q03 | pass* | 596/569/498 | 10 | values out of #24 band (below) |
| q04 | pass | 357/580/519 | 5 | exact |
| q05 | pass | 1145/843/884 | 5 | max drift 0.111 % (#24, in band) |
| q06 | pass | 257/288/296 | 1 | exact |
| q07 | pass | 913/923/913 | 4 | |
| q08 | pass | 1095/1056/1056 | 2 | |
| q09 | pass | 1043/954/1012 | 175 | arena fix 1d4428da |
| q10 | pass* | 581/580/580 | 20 | values out of #24 band (below) |
| q11 | pass | 965/959/879 | 1048 | |
| q12 | pass | 499/499/500 | 2 | |
| q13 | pass | 478/459/459 | 42 | |
| q14 | **refused** | 250 (run0) | — | NEW descriptor blocker (below) |
| q15 | **wedge** | 1712 (run0) | 0 | #29 empty-result flake; bench.sh records empty as wedge |
| q16 | pass | 518/488/489 | 18314 | byte-identical to DuckDB; CLONE_EXPR patch |
| q17 | pass | 953/953/1064 | 1 | |
| q18 | pass | 752/701/682 | 57 | exact |
| q19 | pass | 408/409/397 | 1 | |
| q20 | pass | 742/812/832 | 186 | |
| q21 | pass | 944/934/912 | 100 | |
| q22 | pass | 523/524/545 | 7 | |

**19/22 run to completion with exact key sets, counts, and self-consistent ordering; 17/22
additionally hold every value inside the 0.25 % tolerance** (q03/q10 are the pass* rows).
Zero cascade: every non-pass row reproduces solo with a characterized cause, and every query
after each restart ran clean. Teardown verified: 0 leftover CN/FE processes, nvidia-smi 0 MiB.

Remaining open, in priority order:

- **#26 — q02 wedge (primary).** Deterministic engine-thread stall; times out at 30 s with
  0 rows. Note the lease decoupling (`a94e8660`) did NOT clear it — d4 landed, so q02's stall
  is deeper than the staging-lease cycle; needs real engine-side abort (b2) plus the
  per-fragment watchdog (d3) to even localize it.
- **q14 — NEW loud refusal**: `descriptor error: slot 35 (tuple 5) is not part of
  row_tuples [5]`, raised translating intermediate fragment F04; deterministic, reproduced on
  two healthy clusters. Per EXPLAIN VERBOSE, slot 35 = `[31:cast]*[34:cast]` (the
  `l_extendedprice*(1-l_discount)` multiply) lives in Project node 6's `common_slot_map`; the
  update-serialize AGGREGATE node 7 references it directly inside
  `if(p_type LIKE 'PROMO%', [35], 0)`, but slot 35 is not among tuple 5's materialized slots
  and common-expr slot registrations are local to the project node's translation — resolution
  finds zero candidates (the unique-slot-id fallback also finds none). Fix direction: make
  `common_slot_map` registrations visible to sibling nodes in the same fragment. The
  CLONE_EXPR translation itself is proven by q16.
- **#29 — q15 flake.** FP64 equality race (`total_revenue = max(total_revenue)` over two
  independently reduced FP64 sums): 3/6 correct on a warm cluster (correct runs return
  supplier 8449, total_revenue drift 0.043 % — in band), 3/6 empty. Worse than the ~1/3
  previously documented.
- **#24 — revenue-sum deficit, now out of band on two queries.** q03 and q10: key sets,
  counts, and Sirius self-ordering all exact, values deterministically LOW (stable across 3
  reruns), 2 rows each beyond 0.25 %: q03 orderkeys 2435712 (−0.300 %) and 2456423 (−0.390 %);
  q10 custkeys 143347 (−0.291 %) and 146149 (−0.397 %). q10 also shows a non-adjacent 3-rank
  rotation (custkey 6226 drops two ranks) — legal under "ordering per Sirius's own values" but
  beyond the documented adjacent-swap behavior. Same low-bias signature as #24, larger than
  the documented 0.1–0.2 %.

## 20/22 (2026-08-07, commit fe236e8b)

The q14 common-slot materialization landed (fe236e8b: a project's `common_slot_map` slots
consumed by ancestor nodes are appended as carried trailing columns, dropped at every
re-materializing node, refused at joins). Final sweep (A5, 3 timed runs, 30 s ceiling,
restart-on-failure): **20 of 22 pass** — every query except:

- q02: the engine-thread wedge (#26) — hard hang, needs engine-side abort/watchdog.
- q15: the empty-result flake (#29) — intermittent 0-row result (~1 in 4), FP64
  exact-equality race; passes with in-band values otherwise.

q14 = 16.381152163162234 vs DuckDB 16.380778626395543 (+0.0023%). q03/q10 pass with exact
key sets and internally-exact ordering; individual revenue values run up to 0.40% low (the
deferred #24 deficit). CSV: /tmp/sirius-tpch-bench/bench/A5/timings.csv.
