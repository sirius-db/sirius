# Status — 2026-08-20

Box: 2× RTX PRO 6000 Blackwell (**94.97 GiB allocatable**/card — 638 MiB is driver-reserved, so the
old 95.59 GiB figure is wrong). 2 CNs, one per GPU. Branch `demo-multi-cn`. Nothing committed.

## Where TPC-H stands

| Scale | Result | Config |
|---|---|---|
| SF100 | 22/22 correct | 60/32 |
| SF300 | 21/22 | 60/32 |
| **SF500** | **21/22 correct** (was 14/22) | 60/32 + 1 GiB operator budgets |

SF500 detail: 19 verified against the DuckDB oracle (max rel dev **3.3e-10**), q11 correct-empty
(matches oracle), q21 passes but intermittently stalls, **q09 is the only real failure**.

Full analysis: [SF500-CONFIG-AND-ARCHITECTURE.md](SF500-CONFIG-AND-ARCHITECTURE.md).
Raw data: `results/sf500xcold.csv` (full 22), `sf500e5.csv` (q08 fixed), `sf500e{3,4a,6,7,8}.csv`.

## The working config

```bash
GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB NUM_CNS=2
hash_partition_bytes=1GiB  max_build_hash_table_bytes=2GiB
scan_task_batch_size=1GiB  concat_batch_bytes=1GiB
SET GLOBAL query_timeout=1800     SIRIUS_LOG_BACKEND=spdlog
```

Operator budgets need `--sirius-config`; the `--gpu-memory-limit` flag path cannot express them.

## What we learned (short form)

1. **Operator budgets were the decisive lever, not the memory split.**
   `derived_default_batch_size()` (`src/sirius_config.cpp:38`) sizes every budget from
   `prop.totalGlobalMem` (measured 94.9706 GiB here — it already excludes the driver reserve), not the
   configured pool. Derived default 2.3743 GiB; 2.4x oversized at a 60 GiB pool, and far worse under
   a carve-out (`--gpu-memory-limit 8GiB` yields budgets at 29.7% of pool). Cutting them to pool-proportional values is what fixed q08.
2. **Arena occupancy is a pressure gauge for the pool, not independent demand.**
   `push_packed` deep-copies arena→pool *before* releasing the lease, so a pool at its ceiling stalls
   the drain and the arena ratchets to capacity. Same q21 measured 47.40 GiB (starved pool) vs
   ≤26.78 GiB (healthy). Retire `STAGING ≈ 96GiB × SF/500 / N`.
3. **Parked sender outputs leak** — 11.3 GiB per q07 run per CN, unspillable (356 downgrade requests
   freed 0 bytes). Local destinations have no teardown release; `cancel_plan_fragment` is a stub.
4. **Two measurement defects** invalidated earlier data: `SIRIUS_LOG_BACKEND` silently drops unknown
   values on the CN path, and `bench.sh` never raised the FE's 300 s `query_timeout`.

## Pending — ranked

Each item has a **self-contained plan** in [`plans/`](plans/). Every plan is written to be executed
in a **fresh session with zero prior context** — open the plan, follow it, no back-reference needed.

### Code (none written; all analysis only)

| # | Plan | Item |
|---|---|---|
| 1 | [PLAN-01](plans/PLAN-01-copy-out-on-arrival.md) | **Copy-out on arrival** — highest value, and **what q09 needs**. Batches accumulate in `SenderSource::Remote{ batches: Vec<StagedBatch> }`; leases release only once the receiver's whole sender set closes, so arena demand = the receiver's entire remote input. `engine.rs:543` / `fragment_executor.rs:75` already *claim* copy-out-on-arrival. Making it true collapses arena demand to ~2.4 GiB and frees ~30 GiB/CN. |
| 2 | [PLAN-02](plans/PLAN-02-park-ownership-teardown.md) | **Query-scoped park ownership + real `cancel_plan_fragment`** — closes the 11.3 GiB leak. Note a `[window]` is one *fragment*, not one query, so "QueryEnd == QueryBegin" is NOT a valid assertion. |
| 3 | [PLAN-03](plans/PLAN-03-batch-size-derivation.md) | **`derived_default_batch_size()` off the configured pool** + bring-up validation rejecting `pool + arena + overhead > allocatable` instead of a later opaque `bad_alloc`. |
| 4 | [PLAN-04](plans/PLAN-04-scheduler-stall.md) | **Scheduler stall (Defect B)** — q21's 600 s hang and 207 s of q07's warm run are one un-cancellable fragment head-of-line-blocking the CN's single engine thread until the watchdog. Investigation-first. |
| 5 | [PLAN-05](plans/PLAN-05-bench-harness.md) | **`bench.sh`** — no correctness gate at all; 0-row answers logged as wedges (q11); FE `query_timeout` never raised from 300 s. |
| 9 | [PLAN-09](plans/PLAN-09-exchange-backpressure.md) | **No backpressure on the exchange lease path.** `push_packed` deep-copies into pool memory (`sirius_ffi.cpp:849`) and only *then* releases the lease (`engine.rs:563`), so arena drain is gated on pool availability — while `lease()` either grants or throws, with no credit, window, or retry. Pool at ceiling ⇒ arena ratchets to capacity and never recovers. Problem writeup, two reproductions, and the backpressure design space. Companion to PLAN-01. |

### Measurement gaps

| # | Plan | Item |
|---|---|---|
| 6 | [PLAN-06](plans/PLAN-06-q21-flake-quantification.md) | **q21 flake rate** — only 3 samples (pass / 600 s hang / pass). Produces the rate + the paired log corpus PLAN-04 consumes. |
| 7 | [PLAN-07](plans/PLAN-07-q15-float-determinism.md) | **q15 intermittent 0 rows** — exact float equality against a GPU aggregate; 13/30 at SF100, and arm C showed two adjacent doubles 1 ULP apart. Weighs do-nothing vs deterministic reduction vs CTE reuse vs query tolerance. |
| 8 | [PLAN-08](plans/PLAN-08-measurement-gaps.md) | **Four gaps**: SF300/SF100 never re-run with the new operator budgets; arena high-water largely unmeasured (teardown prints only on clean shutdown — measured 48 of 96 arena lifetimes, i.e. 50%, lost because restart killed CNs inside the 15 s SHUTDOWN_GRACE); disk spill never confirmed to fire (downgrade success logs at DEBUG, we ran at INFO); q08/q09 still on hand-reordered FROM clauses pending real `FILES()` statistics. |

### Not worth re-testing
- Pool/arena splits for q09: 60/32, 65/27, 68/24, 70/22, 76/16 all fail, alternating pool-OOM and
  arena-exhaustion. The window is empty — the arena floor (~24–27 GiB) and q09's >60 GiB pool need
  exceed the card.
- Smaller batch sizes for q09: identical failure at 2.3743 GiB, 1 GiB, 512 MiB. Not a partitionable
  build side.

## Box state

A cluster is up at **65 GiB/27 GiB** (the last q09 probe), *not* the winning config. To restore:

```bash
GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB \
  /opt/dlami/nvme/sirius-build/restart-sf500x.sh
```

Uncommitted in the tree: arena freelist rework, FFI `outstanding()`, `streaming_fragment`
DOUBLE/FLOAT hash keys, q08/q09 FROM reorders, plus this session's docs and CSVs.
