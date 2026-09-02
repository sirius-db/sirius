# TPC-H Benchmark Plan — `presto-gb200-gcn-17` (4× GB200)

**Box:** [`HARDWARE.md`](../../bench/gb200-4gpu/HARDWARE.md) · **Query set:** `QUERYSET.md`
**Engines:** A — Sirius · B — StarRocks · C — cudf-polars
**Re-targeting:** [`RETARGETING.md`](../../bench/common/RETARGETING.md) · **Derived from:** [`../../TPCH-BENCHMARK-BRIEF.md`](TPCH-BENCHMARK-BRIEF.md)

This plan contains **no configuration values**. Every engine's settings live in that engine's own
file so the folder can be re-targeted by swapping `HARDWARE.md` and the three engine configs.

---

## Scope

Three studies, two scale factors, one box. **4 GPUs, not 8** — so scale-out is **1 → 2 → 4**.

| # | Study | Question | Engines | Regime |
|---|---|---|---|---|
| **1** | **Scale-Out** | Does Sirius scale 1 → 2 → 4 GPUs? | A | pinned (SF500) / cold (SF1000) |
| **2** | **GPU Shootout** | Sirius vs cudf-polars, same box, no bias | A, C | **cold** |
| **3** | **Cost Efficiency** | $ per run vs a CPU engine | A, B | warm |

### Data — both local, no generation step

| Scale | Path | Size |
|---|---|---|
| SF500 | `/raid/prestouser/aocsa/tpch_parquet_sf500` | **132 GB** |
| SF1000 | `/raid/tpch-sf1000` | **283 GB** |

### Public chart labels

Engine A/B/C are **internal code names only**. Charts say:

| Internal | Chart label | Marker |
|---|---|---|
| A | **Sirius (GPU)** | green square |
| B | **StarRocks (CPU)** | salmon circle |
| C | **cudf-polars (GPU)** | teal circle |

**Filled = hot/pinned (steady state) · open = cold (fresh process, cache dropped).**

---

## Study 1 — Scale-Out

**Chart:** `Sirius GPU Scale-Out · TPC-H SF500` / `· SF1000`
**Subtitle:** `1 → 2 → 4 × GB200 · single node · all pairs NV18`

### Arms

| Arm | CNs | GPUs | CPUs | Notes |
|---|---|---|---|---|
| 1-GPU | 1 | GPU0 | `0-71`, membind 0 | Socket-local baseline |
| 2-GPU | 2 | **GPU0 + GPU2** | `0-71` / `72-143`, membind 0 / 1 | **Both sockets** — see below |
| 4-GPU | 4 | GPU0–3 | `0-35`/`36-71`/`72-107`/`108-143` | Full box |
| *2-GPU NUMA variant* | 2 | GPU0 + GPU1 | `0-35` / `36-71`, both membind 0 | Isolates host-NUMA cost |

> **The 2-GPU choice is the interesting one.** Because **every GPU pair is NV18**, GPU↔GPU
> bandwidth is identical regardless of which two you pick. So GPU0+GPU2 — which gives each CN a
> full socket (72 cores, ~479 GiB) — costs nothing on the fabric while doubling host resources
> versus GPU0+GPU1, which crams both CNs onto socket 0 and leaves socket 1 idle.
>
> Running **both** 2-GPU arms is the cleanest host-NUMA experiment this box supports: the GPU
> fabric is held exactly constant, so **100% of the delta is host-side**. Report it — it is a
> result in its own right, and it directly informs how to place CNs on any 2-socket box.

### Regime

- **SF500 — pinned.** Fits at every topology (132 GB → 132 / 66 / 33 GB per GPU). Pinning is
  legitimate when comparing Sirius to itself and isolates *compute* scaling from *I/O* scaling.
- **SF1000 — cold, all arms.** 283 GB does not fit on one GPU and is tight on two. Running some
  arms pinned and others cold is not a scaling curve. Consistency beats the flattering number.

### Output

X = GPU count (1/2/4, linear) · Y = total time over the 8-query headline set (s, log) · speedup
annotated per point. Second panel: per-query speedup, to separate queries that scale from
communication-bound ones.

**Expected finding to check:** the prior 4-query GB200 sweep showed a **2-CN regression on q06**.
Reproduce or refute it — it is the one anomaly in the existing scaling data.

### Cost angle

On a whole-box price, **scaling efficiency converts directly to cost efficiency**: if 4 GPUs are
3× faster than 1, they are 3× cheaper per run. Worth stating on the chart.

---

## Study 2 — Sirius vs cudf-polars

**Chart:** `Sirius vs cudf-polars · TPC-H SF500`
**Subtitle:** `4 × GB200 · cold — page cache dropped per run · same box, same data`

### Why this is the strongest study available

Both engines run on **the same box, the same GPUs, the same filesystem, the same data**. Hardware
is exactly matched — which is the one thing the SF100 three-way comparison could never claim.

### The five biases from the SF100 audit, and their fixes

| Audit finding | Fix here |
|---|---|
| A numerically wrong on 6 queries | Tier 3 excluded from aggregates; defect magnitude reported separately (`QUERYSET.md`) |
| C ran `--interleave=all` — **11.4% slower on 22/22** | Never `--interleave=all` on this box. See `engine-c-cudf-polars.md` |
| "cold" was first-touch, not cache-dropped | `drop_caches` before each run for A; `--io-mode cold` for C — **symmetric** |
| A used KvikIO, not the fastest path | Sirius uring datasource, tuned. See `engine-a-sirius.md` |
| A capped with no spill; C spills device→host | A's `downgrade_*` path + disk spill root closes this. **Verify it engages under a CN** |
| C quoted from `--iterations 1` | `--iterations 4`: run 0 cold, 1–3 timed |

### Regime conflict — resolved

The SF1000 repro config is a **pinned/hot** config (`SIRIUS_PIN_TIER_*=gpu` for all 8 tables).
Pinning is **unfair here**: cudf-polars has no equivalent — `--io-mode hot` still re-reads parquet
every iteration. A pinned Sirius against a re-reading cudf-polars would be a worse violation than
anything in the SF100 audit.

**→ Study 2 runs cold, `SIRIUS_PIN_TIER_*` unset.** Keep the config's *tuning*; drop its *pinning*.
Pinned numbers belong to Study 1, where Sirius is compared to itself.

### The trap that will silently ruin this study

`ast_jit` compiles kernels with NVRTC on first use — **~19 s cold**; suite first-iteration
28.99 s → 10.50 s warm. The cache lives at `$HOME/.cudf/$VERSION/$ARCH` and survives process restart.

**Pre-warm it, and never drop it as part of "cold."** *Cold* means **page cache dropped**, not
*compiler cache dropped*. Otherwise this measures NVRTC, not Sirius.

### Aggregates

cudf-polars completes queries Engine A cannot, and is correct where Engine A is not. **Any geomean
or "N× faster" must be computed over the 8-query headline set for BOTH engines**, with the
restriction stated on the chart. Otherwise the comparison is not honest.

---

## Study 3 — Cost Efficiency

**Chart:** `TPC-H SF500 — Sirius (GPU) vs StarRocks (CPU) · Cost per Run`

### The problem this box creates

`presto-gb200-gcn-17` is **on-prem**. It has **no $/hr**. A cost-per-run chart needs a price, and
inventing one for the GPU box while using real AWS prices for the CPU box would be the single
easiest way to produce a dishonest chart.

### Design — measure timing here, model cost explicitly

Engine B runs **locally on this box's 144 Grace cores** (2 BEs, one per NUMA node), exactly as in
the SF100 audit. That gives A and B timings on **identical hardware, same data, same filesystem** —
a cleaner timing comparison than any cloud pairing.

Cost is then a **derived, disclosed** quantity:

```
cost_per_run_usd = (wall_seconds / 3600) × declared_$/hr
```

**Report the break-even ratio as the primary result, not absolute dollars.** It is the honest
framing: it survives price changes, needs no on-prem price at all, and is what a reader actually
needs to make a decision.

> Sirius must beat StarRocks by **`price_gpu / price_cpu`×** to break even per run.

With the reference prices on record — 8× A100 $13.25/hr, `m8gd.48xlarge` $8.83/hr,
`m8i.48xlarge` $12.19/hr — the bar is **1.50×** against Graviton4 and **1.09×** against Intel.
At SF100 Engine A beat StarRocks on **all 17** passing queries, several by 3–4× (q17 0.9 s vs
3.3 s, q21 1.6 s vs 6.9 s, q22 0.6 s vs 2.3 s). If that holds it clears both bars comfortably.

### Mandatory disclosure on the chart

- Timings **measured** on `presto-gb200-gcn-17`; dollar figures **modeled** from declared prices
- The GPU price is a **stand-in** — this box is on-prem and not sold by the hour
- Both engines ran on the **same physical box**, so the timing comparison is exact and the cost
  comparison is the only modeled part
- Aggregates over the **8-query** headline set

### Secondary, price-free metrics

Worth reporting alongside, because they need no pricing assumption at all: **GPU-seconds per
query**, **core-seconds per query**, and **energy per run** if `nvidia-smi` power sampling and
node-level power are available.

---

## Sequencing

1. **Study 2 at SF500** — one box, no new infra, lowest risk. Shakes out the config and proves the
   query set survives at scale.
2. **Study 1 at SF500** — same box, add the 1/2/4 topologies plus the 2-GPU NUMA variant. Pinned.
3. **Study 3 at SF500** — Engine B locally; no new hardware needed.
4. **All three at SF1000** — reuse everything; only memory knobs change. Study 1 runs all-cold here.

SF1000 costs roughly 2× the wall clock per sweep. **Do not start SF1000 until the SF500 oracle diff
is green** — an unvalidated sweep at 2× the cost is 2× the waste.

## Blocking prerequisites

- [ ] Stand up the DuckDB oracle at SF500 with a **relative-tolerance (1e-12)** diff wired into the harness
- [ ] Verify the `downgrade_*` spill path actually engages under a CN (force pressure, watch the spill dir)
- [ ] Pre-warm the NVRTC cache and confirm it persists across process restart
- [ ] Arm `SIRIUS_QUERY_WATCHDOG_SECS` (currently 0)
- [ ] Confirm q11's `FRACTION` is `0.0001/SF`
- [ ] Verify box idle and GPUs at the ~28 MiB floor before every engine switch

## What NOT to do

- Never quote cudf-polars from `--iterations 1` — 1.4–4.6× warm-up penalty. **This single mistake
  produced the earlier "Sirius beats cudf-polars" chart.**
- Never compare a **pinned** Sirius against a re-reading cudf-polars
- Never treat `status=pass` as correct — the harness never compares values
- Never use exact equality in the oracle diff — q06 varies ±1 ULP legitimately
- Never drop the NVRTC cache as part of "cold"
- Never publish "all 22 queries complete" for Engine A — it is 17, aggregates over 8
- Never run the Tier 5 failures in the measurement sweep
- Never `numactl --interleave=all` on this box — it lands host pages in GPU HBM
- Never verify NUMA with `Mems_allowed_list` — use `/proc/<pid>/numa_maps`
- Never set a **percentage** memory limit — `/proc/meminfo` counts GPU HBM and `Swap: 0`
- Never run Engine C from a directory containing a `duckdb/` child
- Never present modeled dollars as measured
