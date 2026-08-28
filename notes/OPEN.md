# Open issues

Living work queue. Historical write-ups live in the dated folders; this file is the only list
to act on. Last synthesized 2026-08-26 from the 2026-08-20 RTX status and the 2026-08-09 GB200
audit.

**Do not re-open:** FE auto-blacklist (M6, fixed 2026-08-09), GDS on GB200 C2C (M5, wont-fix),
aarch64 UCX/nixl bring-up (M0.1–4), “nixl is the bottleneck” (#34), byte-range splits,
partitioned output, two-phase agg, the SF1 22/22 defect list (q02 wedge, arena leak, `year()`
casts, sort-tuple order), ASSERT as a TPC-H blocker, the Path B L4 demo, or bump-allocator
drift (free-list has landed).

---

## Ranked — RTX PRO 6000 ×2 / SF500

Source: [`../bench/rtxpro6000-2gpu/STATUS.md`](../bench/rtxpro6000-2gpu/STATUS.md). Plans are
written to run in a fresh session.

| # | Item | Plan |
|---|---|---|
| 1 | **Copy-out on arrival** — batches accumulate in `SenderSource::Remote`; leases release only when the whole sender set closes. Arena demand = the receiver’s entire remote input. This is what SF500 q09 needs. | [PLAN-01](2026-08-20-rtx-sf500/PLAN-01-copy-out-on-arrival.md) |
| 2 | **Query-scoped park ownership + real `cancel_plan_fragment`** — 11.3 GiB parked-sender leak per q07 per CN; cancel is a stub. | [PLAN-02](2026-08-20-rtx-sf500/PLAN-02-park-ownership-teardown.md) |
| 3 | **`derived_default_batch_size()` off the configured pool** + bring-up reject of `pool + arena + overhead > allocatable`. YAML path is partial; flag path still uses physical HBM. | [PLAN-03](2026-08-20-rtx-sf500/PLAN-03-batch-size-derivation.md) |
| 4 | **Scheduler stall** — q21 600 s hang and 207 s of q07’s warm run; one un-cancellable fragment HOL-blocks the CN engine thread. | [PLAN-04](2026-08-20-rtx-sf500/PLAN-04-scheduler-stall.md) |
| 5 | **`bench.sh` harness** — no correctness gate; 0-row answers logged as wedges; FE `query_timeout` never raised from 300 s. | [PLAN-05](2026-08-20-rtx-sf500/PLAN-05-bench-harness.md) |
| 6 | **q21 flake rate** — only 3 samples (pass / 600 s hang / pass). Produces the corpus PLAN-04 consumes. | [PLAN-06](2026-08-20-rtx-sf500/PLAN-06-q21-flake-quantification.md) |
| 7 | **q15 intermittent 0 rows** — exact float equality against a GPU aggregate (13/30 at SF100). | [PLAN-07](2026-08-20-rtx-sf500/PLAN-07-q15-float-determinism.md) |
| 8 | **Measurement gaps** — SF300/SF100 never re-run with operator budgets; arena high-water 50% lost on restart; disk spill unconfirmed; q08/q09 still on hand-reordered `FROM`. | [PLAN-08](2026-08-20-rtx-sf500/PLAN-08-measurement-gaps.md) |
| 9 | **No backpressure on the exchange lease path** — `lease()` grants or throws; pool at ceiling ⇒ arena ratchets to capacity. Companion to PLAN-01. | [PLAN-09](2026-08-20-rtx-sf500/PLAN-09-exchange-backpressure.md) |

SF500 is **21/22 correct** (q09 the only real failure). Do not re-tune pool/arena splits for q09 —
the window is empty.

## GB200 / multi-CN (still open after the 2026-08-09 audit)

Write-up: [`2026-08-09-gb200-sf100/OPEN-ISSUES.md`](2026-08-09-gb200-sf100/OPEN-ISSUES.md).
Retractions: [`2026-08-09-gb200-sf100/HANDOFF.md`](2026-08-09-gb200-sf100/HANDOFF.md).

| ID | Item |
|---|---|
| **M1.4** | `SIRIUS_QUERY_WATCHDOG_SECS` unset → blocking `future.get()`. `REPLY_TIMEOUT` hardcoded 60 s (`prpc_client.rs`). |
| **M1.5** | Lazy nixl sessions; metadata on the transport thread. Cold first cross-CN query fails at 60 s; warmup hides it. |
| **M2.1** | Derived arena default, fail-loud at bring-up, per-lease cap, log `high_water`. SF100 still needs hand-passed `STAGING`. The free-list itself has landed — do not re-open bump-allocator drift. |
| **M4** | `use_odirect` is media-blind (catastrophic on NFS). Do **not** flip `use_sirius_datasource`. |
| **#24** | Decimal→FP64 in `expr_translator.rs` `translate_arithmetic`, not SUM/AVG lowering. |
| **#31** | Unhandled sink → `Ok([])`; parked-sender export; no mid-query fragment abort. Overlaps PLAN-02. |
| **#32** | Harness: discarded warmup masks M1.5; no row-count oracle; `grep -c true` over-counts Alive. Overlaps PLAN-05. |
| **M0.5** | Engine B `setup-engine-b.sh` rewrites `mem_limit=16G` every run; Docker path. |
| **Phase 2** | 8 CNs across 2 GB200 hosts. Phase 1 (1 CN/host, fabric arena, 98 GB/s) works — see `experimental/starrocks/benchmarks/2NODE-REPLICATE.md`. |
| **SF10000 8-CN** | **14/22** with times (112/64 dop=9; q07 96/80). Empty: q03/q05/q08/q09/q10/q17/q18/q21. Log [`../bench/gb200-8gpu/sf10000/TUNING-DISCOVERY.md`](../bench/gb200-8gpu/sf10000/TUNING-DISCOVERY.md). Next: more CNs on **the same SF**, not a bigger scale; do not raise STAGING for q03/q09/q21. Handoff [`2026-08-28-8gpu-handoff.md`](2026-08-28-8gpu-handoff.md). |

## Query workarounds still live

q08/q09 `FROM` reorder pending real `FILES()` statistics:
[`experimental/starrocks/benchmarks/tpch/QUERY-DEVIATIONS.md`](../experimental/starrocks/benchmarks/tpch/QUERY-DEVIATIONS.md).
Tracked as PLAN-08’s last bullet.
