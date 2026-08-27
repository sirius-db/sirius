# TPC-H SF100 — the 5 Engine A query failures

**Measured** 2026-08-09 on the 4× GB200 box. Engine A = 4 Sirius CNs (one per GPU), HEAD `05d3c7f4`.
Engine B = stock StarRocks 3.5.20, CPU, 2 BEs — used here as a **control**, not a rival.
Data: `/raid/prestouser/kkristensen/tpch_parquet_sf100` (local NVMe), identical for both engines.
Protocol: 1 cold run + 3 timed warm runs per query, 180 s warm client cut, 300 s cold.

Raw data: `benchmark-results/tpch-sf100-abc/results.csv`, `sweep-engineA.log`, `sweep-engineB.log`.
A deeper root-cause pass is in flight and will land at
`benchmark-results/tpch-sf100-abc/FAILURES.md`.

---

## 1. What actually happened

Engine A completed **18 of 22**; Engine B completed **20 of 22**.

| query | Engine A — cold, w1, w2, w3 | Engine B | class |
|---|---|---|---|
| **q05** | 1.8s · **wedge 180s** | 2.4 2.4 2.3 2.5 | 🔴 A-only, degrades after run 1 |
| **q08** | **refused 51.1s** · **refused 51.1s** | 229.0s · wedge | 🟠 both, but A errors early |
| **q09** | **refused 64.0s** · **refused 130.9s** | refused 300.2s · wedge | 🟠 both, genuinely heavy |
| **q10** | **refused 121.3s** · **wedge 180s** | 2.2 2.3 2.3 2.5 | 🔴 A-only |
| **q18** | 1.1s · 1.0s · **61.5s** · **wedge 180s** | 4.7 3.9 4.2 6.6 | 🔴 A-only, clean degradation curve |

The 17 queries not listed all passed on both engines, and **Engine A was faster than Engine B on
every one of them** — q17 0.9 s vs 3.3 s, q21 1.6 s vs 6.9 s, q22 0.6 s vs 2.3 s.

---

## 2. The dominant pattern: state accumulates across runs

**This is the single most important observation in this document.** Three of the five failures show
the same shape — the query works, then stops working, on the *same cluster* with *no change in
input*:

```
q18:  1.1s  →  1.0s  →  61.5s  →  wedge(180s)      a 60× degradation across three runs
q05:  1.8s  →  wedge(180s)
q10:  refused(121s) → wedge(180s)
```

q18's curve is the clue: 1.0 s → 61.5 s → wedge is monotonic, not bimodal. Something grows with
each execution until the query can no longer complete. That is a **leak or an unreleased resource**,
not a query-shape problem — a query that is too hard fails on run 1, not on run 3.

**Candidates to check, in order of suspicion** (all *inferred*, none yet traced):

1. **Exchange staging arena leases.** 16 GiB of bare `cudaMalloc` outside RMM. A cumulative arena
   leak of exactly this shape was fixed once before in `0092077c` — read that commit first; this may
   be a second leak of the same family, or a regression of it.
2. **Parked sender outputs** (`engine.rs:264-274`). See §3.
3. **Result store / channel registry** entries not reaped after a failed or cancelled query.
4. **nixl transfer descriptors** or registered buffers not released per query.
5. **GPU pool fragmentation** — would show as allocation slowdown rather than a hang, so less likely.

**Decisive experiment** (cheap): run q18 five times on a fresh cluster while sampling the engine's
own `[gpu_pool]` peak, the staging-arena outstanding-lease count, and per-run telemetry entity
counts. Whichever counter rises monotonically alongside latency is the leak. The saved telemetry
under `tpch-sf100-abc/engineA/` may already contain this — compare run 1 against run 3 of q18.

---

## 3. q08 — CONFIRMED: `05d3c7f4` created a head-of-line deadlock. Revert or fix before anything else.

**Verdict: the commit did NOT create the OOM, but it made that OOM unrecoverable, and then hid it.**
Diagnosed from `benchmark-results/tpch-sf100-abc/engineA/cluster.log.gz`; every claim below was
re-verified by hand.

### What actually fails
Three consecutive log lines, q08 cold, within 65 µs of each other:

```
18:43:51.916104 WARN  failed to drop the parked output ... error=no parked sender output to drop
18:43:51.916130 ERROR dispatched intermediate receiver fragment failed ...
                error=GPU pipeline task exceeded maximum retry limit (100) for original task 889:
                      OOM at operator HASH_JOIN (index 0)
18:43:51.916169 ERROR a remote destination's drain failed
                error=no parked sender output to export for SenderSlot{...}
```

The real failure is a **GPU OOM in HASH_JOIN**, retried to exhaustion by
`src/pipeline/gpu_pipeline_executor.cpp:348-364` (MAX_RETRIES = 100) and giving up after ~51 s.
The `no parked sender output` message an operator sees is emitted **39 µs later** and is derived.

### Why the retries can never converge — the new hazard
1. The engine thread is **strictly single-request**: `engine.rs:257`,
   `while let Ok(request) = requests.recv()`.
2. `ExportNext` and `DropParked` are serviced **on that same queue** (`engine.rs:279-286`), reached
   from the transport thread via `engine_call` inside the drain loop (`nixl_transport.rs:688`).
3. `05d3c7f4` made the sender dispatch its **local receiver before joining the remote drains**
   (`compute_node_service.rs:281-286`). So `run_fragment` for that receiver occupies the engine
   thread for 51 s, while the drains for its **3 sibling remote destinations sit queued behind it**.
4. Those drains are the only thing that can release the sender's parked GPU batches. **The receiver
   retries 100 times waiting for memory that only a request queued behind it can free.**

Measured confirmation: `exec_plan_fragment: close time.busy=10.4µs time.idle=50.3s` — the sender's
RPC thread sat 50.3 s inside `SenderDrains::join`.

**Before this commit the shape was impossible**: `send_fragment` blocked, so every drain completed
and released its claim *before* the local receiver was dispatched
(`git show 05d3c7f4^:...compute_node_service.rs`, lines 883-899).

**The codebase already knows this hazard.** `engine.rs:636-637`:
> *"The three staging calls below run on the CALLER's thread, never the engine thread: a peer's
> lease request must succeed even while the engine is deep inside a fragment run."*

`ExportNext`/`DropParked` were never given that treatment.

### Why the FE saw the wrong error
`engine.rs:267-274` does `parked.clear(); parked_slots.clear();` on any `run_fragment` error, wiping
outputs whose drains are mid-flight. The siblings then fail at `engine.rs:304`, and
`dispatch_then_join` returns the *first drain error* in preference to the real one — so the OOM
never reached the FE at all.

### Evidence quality
Exactly **3** `maximum retry limit ... OOM at operator HASH_JOIN` lines in the whole sweep (q08 cold,
q08 warm, q09 cold), each immediately followed by its 3 siblings' `no parked` errors, with **zero
counter-examples**. `git show --stat 05d3c7f4 -- src/` is **empty** — the commit touches no C++
engine file, so it cannot have changed HASH_JOIN's memory behaviour.
*(One discrepancy: the diagnosing agent reported 9 `no parked sender output to export` lines; a
direct `zgrep -c` returns 17. The 3 OOM lines and the 1:1 ordering are unaffected, but the exact
count should be re-derived before quoting it.)*

### Decision
**Revert `nixl_transport.rs` + `compute_node_service.rs` to `c3bfe660`.** The change was A/B'd on
this box and bought **nothing measurable** (q14 958→944 ms, q06 865→892 ms — both inside the noise
band), while introducing a deadlock that costs 2 queries. Keep the rest of `05d3c7f4`, which
contains the `http_port` blacklist fix.

**Proper fix, if the change is wanted later:** move `ExportNext`/`DropParked` off the engine thread,
exactly as the three staging calls already are (`engine.rs:636-637`). Until then the
dispatch-before-join ordering is unsafe by construction.

---

## 3b. Original suspicion (superseded by §3, kept for the reasoning trail)

```
ERROR 1064 (HY000): no parked sender output to export for SenderSlot
                    { fragment_instance_id: FragmentInstanceId(019fe7d5-de8c-...) }
```

That string matches, exactly, a hazard the fragment-dataflow change in HEAD documented and
deliberately did **not** fix:

> *"A same-query receiver dispatched early can now fail while its own siblings' drains are in
> flight, and `engine.rs:264-274` wipes all parked outputs, turning one failure into K 'no parked
> sender output to export' errors."*

**The signature also changed.** Before that commit, q08 was recorded as *"refused at 60758 ms,
≈ the hardcoded 60 s REPLY_TIMEOUT"* — a **timeout**. It now fails at **51 s with a parked-output
error**, i.e. *before* any timeout, through a different path.

**Counter-evidence, stated honestly:** the earlier sweep ran on a cluster that may have had 2 of 4
CNs silently blacklisted, so it is not a clean baseline, and the comparison is weaker than it looks.

**Also note this error is very likely a *symptom*.** `parked.clear()` amplifies one failure into K,
so the real defect may be whatever failed first — the log should be read for a *preceding* error
before anyone "fixes" the parked-output message.

**Verdict needed: CAUSED / UNMASKED / UNRELATED.** The settling experiment is a revert of
`nixl_transport.rs` + `compute_node_service.rs` to `c3bfe660`, rebuild, re-run q08 — which needs
exclusive box time.

**Named fix**, from the change's own author: release only the failed run's own `request.inputs`
instead of wiping all parked outputs. It lives in `engine.rs`, which is behind the `sirius-engine`
feature and therefore **cannot be compiled or tested in the 132-test no-engine profile** — that is
why it was left out, and it remains the gating difficulty.

---

## 4. q08 / q09 — how much is Sirius, and how much is the query?

Engine B is the control, and it is unambiguous:

| | Engine A | Engine B |
|---|---|---|
| q08 | refused at **51 s** with an internal error | **passed in 229 s**, then hit the 180 s warm cut |
| q09 | refused at **64 s** with an internal error | refused at **300 s** (`query_timeout`) |

So **q08 and q09 are genuinely heavy at SF100** — a CPU engine needs 229 s and >300 s. Two separate
things follow:

* **A harness bug, not an engine bug:** Engine B's q08 *succeeded* in 229 s and was then failed by a
  180 s warm cut. The timeout must scale with the query and the engine, or the harness manufactures
  failures. Fix the harness before drawing any conclusion about q08 on Engine B.
* **A real Sirius defect regardless:** Engine A errors out at 51–64 s, far below any timeout, with
  `no parked sender output to export` and `failed to export a packed batch`. Those are internal
  failures, not slowness. Duration does not excuse them.

---

## 5. q10 — one defect or two?

```
cold:  refused 121.3s  —  rpc failed with 127.0.0.1: exec rpc error. backend [id=10002]
w1:    wedge 180s
```

An FE-side message: the FE's exec RPC to CN 10002 failed. Engine B passes q10 in 2.2 s, so this is
Sirius-side.

**The two lines are probably one defect.** A failed exec RPC can blacklist that CN, and the very next
run then executes on a degraded cluster — which is exactly what a 180 s wedge on w1 looks like.
Blacklist entries now evict in ~2.5 s (fixed in `05d3c7f4`), but *during* a sweep the window is still
live. Two consequences:

* The harness must **restart the cluster after any refusal or wedge**, or every subsequent number in
  that sweep is untrustworthy. `bench.sh` has `RESTART_CMD` for exactly this; `run-abc.sh` should
  use it unconditionally after a non-pass.
* Assert **`SHOW COMPUTE NODE BLACKLIST` is empty** after each failure, not just at start-up.

---

## 6. The worst problem is not a failure, it is the silence

q05 and q18 produce **`rc=124` and nothing else** — no error, no log line, no telemetry marker. A
180 s wall-clock cut is all anyone gets.

Making a wedge *loud* is worth more than fixing any single query here, because it makes every future
wedge self-diagnosing. Concretely:

1. **Arm the query watchdog.** `SIRIUS_QUERY_WATCHDOG_SECS` exists and is **unset** in
   `configs/gb200-4gpu/engine-a.env` (`${SIRIUS_QUERY_WATCHDOG_SECS:-0}` — 0 = disabled). Set it
   below the client cut so the engine aborts and *reports* before the client gives up.
2. **Dump on wedge.** On watchdog fire, log which fragment, which operator, how many outstanding
   staging leases, and how many parked outputs.
3. **Flush telemetry on abort.** Telemetry is buffered in memory and written only at engine
   shutdown, so a wedged run that is killed produces *nothing* — the exact case where it is most
   needed.
4. There is a known `forcing process exit: graceful shutdown did not finish — engine thread wedged
   inside a fragment run` path; find whether it fired here and why it did not surface.

---

## 7. Proposed order of work

| # | Action | Why now | Effort |
|---|---|---|---|
| 0 | **Revert `nixl_transport.rs` + `compute_node_service.rs` to `c3bfe660`** (§3) | CONFIRMED head-of-line deadlock in HEAD, costing q08 and q09. The change bought nothing measurable. Keep the rest of `05d3c7f4`. | minutes |
| 1 | ~~Settle the `05d3c7f4` regression verdict~~ — **done, see §3** | — | — |
| 2 | **Find the cumulative leak behind q18's 1.0→61.5→wedge** | One root cause plausibly explains q05, q10 and q18 — 3 of 5 failures. Best ratio in the list. | days |
| 3 | **Make wedges loud** (watchdog + dump + flush-on-abort) | Turns every future silent failure into a diagnosable one. Prerequisite for #2 being efficient. | hours |
| 4 | **Harness: restart after any non-pass; scale timeouts with SF** | Engine B's q08 "failure" was a harness artifact. Without this, results stay contaminated. | hours |
| 5 | **q09 resource analysis** | Estimate peak in-flight exchange bytes vs the 16 GiB arena; it may simply need more. | hours |

**Do not** start with the parked-output message. It is very likely a symptom, and `parked.clear()`
converts one failure into K — the first error in the log is the real target.

---

## 8. What is NOT established

* No root cause is **traced in code** yet; everything in §2 is inferred from timing shape.
* The `05d3c7f4` regression is **suspected, not proven** — the pre-change baseline is itself suspect.
* No live reproduction was run (the box was in use); every claim here comes from saved logs and
  telemetry.
* q08/q09 on Engine B were measured with a 180 s warm cut that is **too short for those queries**, so
  Engine B's "failures" there are partly the harness's doing.
* Engine C is not in this document — its run had not completed.
