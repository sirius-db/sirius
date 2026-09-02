# TPC-H Benchmark Plan — 8× A100-SXM4-80GB

**Box:** [`HARDWARE.md`](../../bench/a100x8/HARDWARE.md) · **Query set:** `../common/QUERYSET.md`
**Engine A config:** [`engine-a-sirius.yaml`](../../bench/a100x8/engine-a-sirius.yaml)
**Re-targeting:** [`../common/RETARGETING.md`](../../bench/common/RETARGETING.md) · **Sibling:** `../gb200-4gpu/`

No configuration values live in this file — they are in the per-engine files.

---

## Scope

Three studies, two scale factors. **8 GPUs** → scale-out is **1 → 2 → 4 → 8**, four points.

| # | Study | Question | Engines | Regime |
|---|---|---|---|---|
| **1** | **Scale-Out** | Does Sirius scale 1 → 2 → 4 → 8 GPUs? | A | **cold** (see below) |
| **2** | **GPU Shootout** | Sirius vs cudf-polars, same box | A, C | **cold** |
| **3** | **Cost Efficiency** | $ per run vs a CPU engine | A, B | warm |

### Data — must be staged

| Scale | Size | Status |
|---|---|---|
| SF500 | 132 GB | 🔍 **stage before booking box time** |
| SF1000 | 283 GB | 🔍 **stage before booking box time** |

415 GB total. On an hourly box, staging time is billable — copy before the measurement window, and
confirm the target is local NVMe (`findmnt -T`), not network storage.

---

## What makes this box different from the GB200

Read [`HARDWARE.md`](../../bench/a100x8/HARDWARE.md) for the full comparison. Three items change the studies:

1. **80 GiB/GPU → a ~68 GiB usable pool** vs the GB200's ~159 GiB. Per-GPU working room is the
   binding constraint, and it drives every regime decision below.
2. **No GPU→socket affinity and no GPU HBM NUMA nodes.** The GB200's `--interleave=all` trap does
   **not** exist here, and its "GPU0+GPU2 vs GPU0+GPU1" NUMA experiment has **no analogue** — all
   GPUs report `0-239` / `NUMA 0-1` / `GPU NUMA ID N/A`. CN placement is a free choice; balance
   CNs across the two nodes deliberately.
3. **It has a real price — $13.25/hr.** Unlike the on-prem GB200, Study 3 here yields a **measured**
   cost-per-run, not a modeled one. That makes this the better box for Study 3.

---

## Study 1 — Scale-Out

**Chart:** `Sirius GPU Scale-Out · TPC-H SF500` / `· SF1000`
**Subtitle:** `1 → 2 → 4 → 8 × A100 80 GB · single node · all pairs NV12`

### Arms

| Arm | CNs | GPUs | CPUs | membind |
|---|---|---|---|---|
| 1-GPU | 1 | 0 | `0-119` | 0 |
| 2-GPU | 2 | 0, 4 | `0-119` / `120-239` | 0 / 1 |
| 4-GPU | 4 | 0,1,4,5 | `0-59`,`60-119` / `120-179`,`180-239` | 0,0 / 1,1 |
| 8-GPU | 8 | 0–7 | 30 cores each | 0×4 / 1×4 |

### Regime — run **cold**, all arms, both scale factors

This is forced, not preferred. Pinned-tier feasibility on 80 GiB cards:

| GPUs | SF500 (132 GB) | SF1000 (283 GB) |
|---|---|---|
| 1 | impossible | impossible |
| 2 | 77% — too tight | impossible |
| 4 | 38% — OK | 82% — too tight |
| 8 | 19% — OK | 41% — OK |

Only the 8-GPU arm pins at both scale factors. A curve whose arms use different regimes is not a
scaling curve, so **run every arm cold** and keep the comparison internally valid.

> The GB200 box pins SF500 at *every* topology, so its Study 1 can run pinned end-to-end. That is a
> genuine methodological advantage of the GB200 box, and worth noting when the two are compared.
> If a pinned number is wanted here, report **8 GPUs only**, labelled as a ceiling.

### Output

X = GPU count (1/2/4/8, linear) · Y = total time over the 8-query headline set (s, log) · speedup
annotated. Second panel: per-query speedup.

**Expect worse scaling than the GB200 box on communication-bound queries** — `NV12` is fewer links
and an older NVLink generation than `NV18`. That is a real finding if it appears; do not tune it away.

### Cost angle

The box is priced whole at **$13.25/hr** regardless of GPUs used, so **scaling efficiency converts
directly to cost efficiency**: 8 GPUs 3× faster than 2 means 3× cheaper per run. State it.

---

## Study 2 — Sirius vs cudf-polars

**Chart:** `Sirius vs cudf-polars · TPC-H SF500`
**Subtitle:** `8 × A100 80 GB · cold — page cache dropped per run · same box, same data`

Same design as the GB200 folder. Both engines on the same box, same GPUs, same filesystem, same
data — hardware exactly matched.

### The five SF100 biases, and their fixes

| Audit finding | Fix |
|---|---|
| A numerically wrong on 6 queries | Tier 3 excluded from aggregates (`../common/QUERYSET.md`) |
| C ran `--interleave=all` — 11.4% slower | Harmless *on this box* (no GPU HBM nodes), but prefer per-worker `bind_to_gpu(hardware_binding)` anyway |
| "cold" was first-touch, not cache-dropped | `drop_caches` for A; `--io-mode cold` for C — **symmetric** |
| A used KvikIO | uring datasource, tuned — [`engine-a-sirius.yaml`](../../bench/a100x8/engine-a-sirius.yaml) |
| A capped, no spill; C spills | A's `downgrade_*` + disk spill root. **Verify it engages under a CN** |
| C from `--iterations 1` | `--iterations 4`: run 0 cold, 1–3 timed |

### Regime — cold, `SIRIUS_PIN_TIER_*` unset

Pinning is unfair against cudf-polars, which has no equivalent (`--io-mode hot` still re-reads
parquet every iteration). Keep the config's *tuning*; drop its *pinning*.

### The trap that will silently ruin this study

`ast_jit` compiles kernels with NVRTC on first use — **~19 s cold**, and **`sm_80` starts empty on
this box**. Pre-warm `$HOME/.cudf/$VERSION/$ARCH` and **never drop it as part of "cold."**
*Cold* = page cache dropped, **not** compiler cache dropped.

### Aggregates

cudf-polars completes queries Engine A cannot. Any geomean or "N× faster" must be over the
**8-query headline set for both engines**, with the restriction on the chart.

---

## Study 3 — Cost Efficiency

**Chart:** `TPC-H SF500 — Sirius (GPU) vs StarRocks (CPU) · Cost per Run`
**Subtitle:** `17 of 22 queries · on-demand $/hr · lower-left is better`

**This box is where Study 3 belongs** — it has a real hourly price, so the cost axis is measured
rather than modeled.

```
cost_per_run_usd = (wall_seconds / 3600) × $/hr
```

| Engine | Instance | $/hr | $/second |
|---|---|---|---|
| Sirius (GPU) | 8× A100 80 GB | **$13.25** | $0.003681 |
| StarRocks (CPU) | `m8gd.48xlarge` — 192 vCPU Graviton4, 768 GB, local NVMe | **$8.83** | $0.002453 |
| StarRocks (CPU) | `m8i.48xlarge` — 192 vCPU Intel, 768 GB | **$12.19** | $0.003386 |

### The break-even, stated up front

Prices are close, so **this study is decided almost entirely by speed**:

| Sirius must beat | by | to break even |
|---|---|---|
| `m8i.48xlarge` | **1.09×** | any GPU win is a cost win |
| `m8gd.48xlarge` | **1.50×** | the real bar |

At SF100 Engine A beat StarRocks on **all 17** passing queries, several by 3–4×. If that holds, it
clears both bars. **Put the break-even ratio on the chart** — it survives price changes, which a
raw cost ratio does not.

### m8gd vs m8i — pick m8gd

They differ in more than architecture: **m8gd has local NVMe, m8i does not** (it needs EBS). Running
both confounds arch-vs-storage. Use **m8gd as primary** — cheaper, local NVMe matching the GPU box,
and Graviton matches the SF100 audit's lineage. Use m8i only as a secondary x86 reference, with the
storage difference named.

---

## Sequencing

Because this box is **billed hourly**, sequence to minimise idle time:

1. **Before booking:** stage both datasets, complete the x86_64 build, dry-run the harness.
2. **Study 2 at SF500** — proves the config and query set.
3. **Study 1 at SF500** — same cluster, add topologies.
4. **Study 3 at SF500** — Engine B runs on AWS independently and in parallel; only the Sirius arm
   needs this box.
5. **SF1000** for Studies 1 and 2 — ~2× the wall clock.

**Do not start SF1000 until the SF500 oracle diff is green.** An unvalidated sweep at 2× the cost is
2× the waste — and here that waste is denominated in dollars.

## Blocking prerequisites

- [ ] **x86_64 rebuild against CUDA 12.8** — the GB200 build is aarch64/13.x. Confirm the pixi env resolves
- [ ] Stage SF500 + SF1000 (415 GB) on local NVMe; `findmnt -T` to confirm
- [ ] Set `SIRIUS_EXCHANGE_STAGING_BYTES=8GiB` — **not** the 16 GiB default (20% of this card)
- [ ] `mkdir` per-CN spill dirs on local NVMe; `df` before committing 4 TB
- [ ] Pre-warm the `sm_80` NVRTC cache
- [ ] DuckDB oracle at SF500 with **relative-tolerance (1e-12)** diff
- [ ] Verify the `downgrade_*` path engages under a CN
- [ ] Arm `SIRIUS_QUERY_WATCHDOG_SECS`
- [ ] Sweep `scan_task_batch_size` ∈ {2,3,4} GB and `uring_n_reactors` ∈ {4,8,32}
- [ ] Confirm q11's `FRACTION` = `0.0001/SF`

## What NOT to do

- Never copy `usage_limit_fraction` from the GB300 (0.95) or GB200 (0.86) config — **0.85 here**
- Never leave `SIRIUS_EXCHANGE_STAGING_BYTES` at 16 GiB — that is 20% of an 80 GiB card
- Never copy `hash_partition_bytes: 32GB` — that is 47% of this box's entire pool
- Never quote cudf-polars from `--iterations 1`
- Never compare a **pinned** Sirius against a re-reading cudf-polars
- Never treat `status=pass` as correct — the harness never compares values
- Never use exact equality in the oracle diff — q06 varies ±1 ULP legitimately
- Never drop the NVRTC cache as part of "cold"
- Never publish "all 22 queries complete" — it is 17, aggregates over 8
- Never run the Tier 5 failures in the measurement sweep
- Never spill to network storage — on PCIe-attached GPUs the downgrade path is already a cliff
- Never set a **percentage** memory limit for Engine B
- Never run Engine C from a directory containing a `duckdb/` child
