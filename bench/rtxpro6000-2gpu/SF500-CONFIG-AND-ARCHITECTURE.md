# SF500 on 2× RTX PRO 6000: what configuration achieves, and what it cannot

**Result: 21/22, up from 14/22, by configuration alone — no code change.**
Only q09 remains, and it is a genuine wall rather than a tuning miss.

Box: 2× RTX PRO 6000 Blackwell. `nvidia-smi` reports 97887 MiB/card, but **638 MiB is driver-reserved
and not allocatable — the real budget is 94.97 GiB, not 95.59.** 48 cores, ~1.1 TiB host RAM.
Dataset `/opt/dlami/nvme/tpch/tpch_parquet_sf500_f64` (DECIMAL cast to DOUBLE). 2 CNs, one per GPU.

## The configuration

```bash
GPU_MEM=60GiB  STAGING=32GiB  HOST_MEM=200GiB   NUM_CNS=2

# operator_params — reachable ONLY via --sirius-config; the --gpu-memory-limit flag path
# cannot express them (src/main.rs makes the two mutually exclusive).
hash_partition_bytes:       1GiB     # default here is 2.3743GiB
max_build_hash_table_bytes: 2GiB     # default here is 4.7486GiB
scan_task_batch_size:       1GiB
concat_batch_bytes:         1GiB

SET GLOBAL query_timeout = 1800      # FE default is 300s
SIRIUS_LOG_BACKEND=spdlog            # any other value is SILENTLY discarded on the CN path
```

Bring-up: `/opt/dlami/nvme/sirius-build/up-sf500-x.sh` (all values are env knobs).

## Per-query status

| Status | Queries | n |
|---|---|---|
| Pass, verified against the DuckDB oracle | q01–q08, q10, q12–q20, q22 | 19 |
| Correct empty — matches the oracle exactly | q11 | 1 |
| Passes, but intermittently stalls | q21 | 1 |
| Fails — genuine per-fragment memory wall | q09 | 1 |

Max relative deviation across all verified queries: **3.3e-10**; most are exactly 0.
q08's answer agrees with DuckDB to 1 ULP (`0.04051322057535745` vs `…744`).

**q11 is not a failure.** `oracle-sf500f64/q11.tsv` is 17 bytes — header only; DuckDB returns 0 rows too,
because `queries/q11.sql` hardcodes the SF1 threshold `0.0001` where the spec scales it `0.0001/SF`.
`bench.sh:175`'s `[ -s "$f" ]` files a correct empty answer as a wedge.

**q21 is flaky, not broken.** 3 samples: 17.7s pass, 600s hang, 20.6s pass. The hang is a scheduler
stall, not memory — the ledger sat at 35 GiB of 60 while a fragment that already had its inputs made
no progress for 280s with 8 and 24 un-consumed batches.

## What actually mattered, in order

1. **The operator budgets, not the memory split.** `derived_default_batch_size()`
   (`src/sirius_config.cpp:38`) computes `clamp(prop.totalGlobalMem/40, 512MiB, 5GiB)` from the
   **card**, not the configured pool. Measured on this box `totalGlobalMem` = 101,973,950,464 B =
   94.9706 GiB (it ALREADY excludes the 638 MiB driver reserve), so the derived default is
   **2,549,348,761 B = 2.3743 GiB**, and it is the
   default for `scan_task_batch_size`, `hash_partition_bytes`, `concat_batch_bytes`,
   `sort_sample_bytes`, with `max_build_hash_table_bytes` at 2x = 4.7486 GiB. With a 60 GiB pool every
   budget is therefore **2.4× oversized**. Correcting the ratio is what let q08 through — not more
   memory, just memory used in smaller pieces. `natural_num_partitions() = ceil(total_bytes /
   hash_partition_bytes)` (`src/include/op/sirius_physical_partition_consumer_operator.hpp:65`), so
   lowering it genuinely raises partition count and lowers per-partition bytes.
2. **The arena split, 45/48 → 60/32.** Fixed q17, q18, q21 outright.
3. **Instrumentation.** Every prior SF500 run was blind (see below).

## Two measurement defects found

* `SIRIUS_LOG_BACKEND` accepts only `duckdb`, `spdlog`, `noop`. On the CN's FFI path an unknown value
  is **silently dropped** — `install_configured_log_sink`'s `throw` is guarded by `if (db)` and the CN
  passes `nullptr` (`src/sirius_context.cpp:1573`). No `[gpu_pool]`, downgrade, or arena telemetry was
  ever captured before this session.
* `bench.sh` sets only a client-side timeout and never raises the FE's server-side `query_timeout`
  (default 300s), so slow-but-healthy queries were recorded as failures.

## The arena is a pressure gauge, not an independent demand

`Fragment::push_packed` deep-copies each staged batch **into pool memory** (`src/sirius_ffi.cpp:849`)
and only then releases the lease (`experimental/starrocks/src/engine.rs:563`). **The arena's drain
rate is gated on pool availability**, and there is no backpressure on the lease path — `lease()`
either grants or throws. So when the pool is at its ceiling, copies stall, leases are never released,
senders keep writing, and arena occupancy ratchets to capacity.

Proof: at 45/48 the four `arena exhausted` events 82 seconds apart report **byte-identical** state —
`642284544 free of 51539607552 capacity, 67 leases outstanding holding 50897323008 bytes`. Once
wedged, not one byte was released.

The same q21, same SF, same CN count measured **47.40 GiB** with a starved pool and **≤26.78 GiB**
with a healthy one. No `(SF, N)` formula can produce a 1.8× spread, so
`STAGING ≈ 96GiB × SF/500 / N` should be retired.

**Diagnostic rule for `arena exhausted`:**

| Reading | Meaning | Action |
|---|---|---|
| arena >90% full **and** pool peak == cap | pathological | **do not raise the arena — fix the pool** |
| arena >90% full, pool under cap | real demand | raise the arena |
| arena <70%, `largest_free < request ≤ total_free` | fragmentation | a bigger arena may not help |

Measured arena high-water: **SF100 full sweep 6.51 GiB**; SF500 healthy max **26.78 GiB** (q18 18.68,
q17 16.06). The old "SF100 needs >16 GiB" datapoint is **dead** — it was q08 under the
`part × supplier` cross-join from missing `FILES()` statistics, since fixed by the FROM reorder.

## Why q09 cannot be configured through

Within a single window q09 grows from 5.85 GiB to **56.53 GiB** and peaks at 64,410,655,403 B =
**99.98% of the 60 GiB cap**. It does not respond to batch size (identical failure at 2.3743 GiB, 1 GiB
and 512 MiB), so it is not a partitionable build side. Every split fails, alternating which resource
breaks:

| Pool / Arena | Budgets | q08 | q09 |
|---|---|---|---|
| 60/32 | 2.3743GiB / 1GiB / 512MiB | pass at <=1GiB | pool OOM (all three) |
| 68/24 | default | **arena** (cold), pool OOM (warm) | pool OOM (both) |
| 76/16 | default | **arena** (both) | pool OOM |
| 70/22 | 512MiB | not run | **arena** (warm); cold died in `export_packed` |
| 65/27 | 512MiB | not run | **arena** (both) |

Re-derived from the surviving sweep logs (`/tmp/sweep-e4a.log`, `-e7`, `-e8`) rather than from memory —
an earlier version of this table collapsed q08 and q09 into one column and mislabelled 68/24 as
pool-only. Note the failure mode FLIPS with the split, which is the whole point: there is no split at
which both fit.

The arena needs ≥24 GiB (q08 breaks at 24; other queries measure 26.78) and ~2 GiB is overhead, so
against 94.97 GiB allocatable the pool cannot exceed ~65 GiB — while q09 needs more than 60. **The
window is empty.** q09 at SF500 exceeds the per-CN device budget of this box.

## Re-architecture required

Ranked by queries unlocked per unit of effort.

### 1. Query-scoped park ownership + a real `cancel_plan_fragment` — days

**The leak.** A parked sender output with a *local* destination has **no teardown release path**. The
three release paths (`engine.rs:533`, `nixl_transport.rs:777`, `:410`) all require a consumer to
actually run; the only consumer-free release is the blanket `parked.clear()` on failure. And the one
per-query end-of-life signal the FE sends is a stub — `compute_node_service.rs:374`, verbatim:
*"Real teardown (aborting the engine run, freeing GPU buffers, dropping parked exchange state) is a
separate work item."* The FE sends it for every fragment instance of every query — **629 times** in
one sweep — and the handler logs, cancels the result, and returns OK.

Measured cost: **11.309 GiB per q07 run per CN**, surviving 206.7s of idle byte-for-byte and paid
again by the next run (2.010× the floor). It is unspillable: the downgrade sweep enumerates only the
per-query registry (`src/downgrade/downgrade_executor.cpp:223`), and exchange repositories are by
construction outside it — **356 downgrade requests freed 0 bytes**.

Fix: give `ParkedOutput` a `query_id`, index parks by query, add `drop_query`, and call it from
`cancel_plan_fragment` and from the fragment-failure path. Add a `ParkGuard` RAII over the post-park
window (four early returns leak today because `run()` already returned `Ok`, so the blanket wipe never
fires).

Caveat: **a `[window]` is one fragment, not one query.** An assertion that `QueryEnd allocated ==
QueryBegin allocated` would fire on every sender fragment — parked output outliving its own window is
required by streaming (`src/include/exec/streaming_fragment.hpp:67`).

### 2. Copy-out on arrival — days

`handle_transmit_packed` only records the batch in the rendezvous; batches accumulate in
`SenderSource::Remote{ batches: Vec<StagedBatch> }` (`local_exchange.rs:240`) and leases are released
only inside `run_fragment_inner`'s push loop, which cannot start until the receiver's entire sender
set has closed. So **peak arena demand is the receiver's whole remote input**, not batch × window.

The code's own comments already claim this is fixed — `engine.rs:543` and `fragment_executor.rs:75`
both say *"copy-out-on-arrival makes that safe"* — but the copy-out happens at **run** time, not
**arrival** time. Making the comment true collapses arena demand to `p_max + 8 MiB ≈ 2.4 GiB`
regardless of SF or CN count, and frees ~30 GiB per CN for the pool. **This is what unlocks q09.**

### 3. Fix `derived_default_batch_size()` to key off the configured pool — hours

One-line class of change; removes the 2.4× oversizing that made q08 fail. Add a bring-up validation
that **refuses** a config whose pool + arena + overhead exceeds the allocatable card, instead of
failing later with `bad_alloc`.

### 4. Diagnose the scheduler stall (Defect B) — unknown

q21's 600s hang, and 207 of q07's 289s warm run, are an un-cancellable fragment head-of-line-blocking
the CN's single engine thread until the 280s watchdog. `task_scheduler.cpp:244` reports the stall but
not its cause. Log the per-pipeline blocked reason — which port is not EOS, whether a reservation is
outstanding — on the watchdog path before `report_error`.

### 5. Harness fixes — hours

`bench.sh:175` must treat rc=0 + empty + no ERROR as a **pass with rows=0** (q11), and should set the
FE's `query_timeout`. The harness has no correctness gate at all — always diff against the oracle.

## Reproduce

```bash
# cluster
GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB \
  /opt/dlami/nvme/sirius-build/up-sf500-x.sh

# full sweep, fresh cluster per query
GPU_MEM=60GiB STAGING=32GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB \
  /opt/dlami/nvme/sirius-build/sweep-sf500x-cold.sh

# correctness (the harness does NOT check this)
python3 /opt/dlami/nvme/sirius-build/compare.py \
  bench/SF500XCOLD /opt/dlami/nvme/sirius-build/oracle-sf500f64
```

Results in `results/sf500xcold.csv` (full 22), `sf500e5.csv` (q08 fixed), `sf500e{3,4a,6,7,8}.csv`
(the q09 ladder).
