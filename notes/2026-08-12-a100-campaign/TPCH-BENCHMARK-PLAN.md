# TPC-H Benchmark Plan — three studies, SF500 and SF1000

**One-page brief:** [`TPCH-BENCHMARK-BRIEF.md`](TPCH-BENCHMARK-BRIEF.md)
**Date authored:** 2026-08-12
**Datasets:** both verified present on `/raid`, no generation needed —
SF500 `/raid/prestouser/aocsa/tpch_parquet_sf500/` (**132 GB**),
SF1000 `/raid/tpch-sf1000/` (**283 GB**, all 8 tables + `metadata.json`)
**Engine A config:** [`bench/sf500-gb200/sirius-sf500.yaml`](../../bench/sf500-gb200/sirius-sf500.yaml) ·
derivation in [`bench/sf500-gb200/README.md`](../../bench/sf500-gb200/README.md)
**Tuning lineage:** [`bench/sf1000-repro/`](../../bench/sf1000-repro/) — PR #1371, merged on `demo-multi-cn`
**Known failures:** [`TPCH-SF100-FAILURES.md`](../2026-08-09-gb200-sf100/TPCH-SF100-FAILURES.md)

Reference artifacts live outside the repo in `../benchmark-results/`:
`ENGINE-CONFIGS-AND-EQUIVALENCE.md` (apples-to-apples audit), `scan-defaults-sweep.md`
(uring vs KvikIO, O_DIRECT), `tpch-sf100-abc/` (the SF100 three-way run + correctness files).

---

## Naming

Engine A/B/C are **internal code names only** — provenance files, invocation scripts, this doc.
Public chart labels are self-explanatory.

| Internal | Public chart label | Marker | Runs on |
|---|---|---|---|
| Engine A | **Sirius (GPU)** | green square | 8× A100 box |
| Engine B | **StarRocks (CPU)** | salmon circle | AWS m8i / m8gd |
| Engine C | **cudf-polars (GPU)** | teal circle | 8× A100 box |

Marker fill follows the existing chart series: **filled = hot/pinned (steady state)**,
**open = cold (fresh process, cache dropped)**.

---

## Hardware

| Role | Instance | Spec | $/hr |
|---|---|---|---|
| **Engines A + C** | `massedcompute_A100_sxm4_80Gx8` | 8× A100 80 GB SXM4 (640 GB VRAM), 1500 GB host | **$13.25** |
| **Engine B** (arm) | `m8gd.48xlarge` | 192 vCPU Graviton4, 768 GB, local NVMe, arm64 | **$8.83** |
| **Engine B** (x86) | `m8i.48xlarge` | 192 vCPU Intel, 768 GB, x86_64 | **$12.19** |

> **Assumption flagged:** the brief lists the A100 box under "Engine A, B" but then lists the AWS
> boxes under "Engine B". Read as **Engines A and C share the A100 box** (both are GPU engines and
> it is the only GPU hardware listed); Engine B runs on AWS. This is forced by elimination — but
> if Engine C was meant to run elsewhere, say so, because Study 2 depends on it.

**This supersedes the earlier brev / `g7e` / `m9g` recommendation** in favour of hardware you
already have access to.

### What the A100 box changes

1. **Study 1 no longer needs two boxes.** 2/4/8 GPUs all fit in one node over NVLink.
   `presto-gb200-gcn-17` is out of scope, and with it the cross-host UCX/TCP fallback, the
   cross-host FE registration, and the interconnect discovery work. **This is a large simplification
   and removes the plan's three biggest infrastructure blockers.**
2. **80 GB HBM per GPU, not 185 GiB.** That is **43%** of a GB200 and **31%** of the GB300 the
   tuning came from. Every memory-sized knob must shrink. See the re-derivation below.
3. **x86_64, not aarch64.** Sirius must be **rebuilt for x86**. The GB200 build does not transfer.
   Same for the cuDF patches if we take them.
4. **A100 is `sm_80`, not Blackwell.** The NVRTC JIT cache is keyed per-arch, so it starts cold.
   `cuco` bucket sizes and the Bloom fast-range win were tuned on GB300 — they should still hold
   (they are algorithmic, not arch-specific) but the *magnitudes* will differ.
5. **No Grace C2C.** Host↔device is PCIe, not a 900 GB/s coherent link. **Device→host spill is
   dramatically more expensive here than on GB200.** The `downgrade_*` path becomes a cliff rather
   than a gentle slope — configure to avoid spilling, not to spill gracefully.
6. **Total VRAM 640 GB.** Pinning feasibility depends on both scale factor and GPU count — see
   the table below. It is the single biggest constraint on which arms can run.

---

## Scale factors

Both datasets are already staged on `/raid` (11 TB free), so there is no generation step.

| | SF500 | SF1000 |
|---|---|---|
| Path | `/raid/prestouser/aocsa/tpch_parquet_sf500/` | `/raid/tpch-sf1000/` |
| Size on disk | **132 GB** | **283 GB** (2.14×) |
| Tables | 8 | 8 + `metadata.json` |

### Pinned-tier feasibility — 80 GB per A100, before working memory

| GPUs | SF500 (132 GB) | SF1000 (283 GB) |
|---|---|---|
| 2 | 66.0 GB/GPU — **marginal**, expect spill | 141.5 GB/GPU — **impossible** |
| 4 | 33.0 GB/GPU — comfortable | 70.8 GB/GPU — **will not fit** with working memory |
| 8 | 16.5 GB/GPU — comfortable | **35.4 GB/GPU — fits** |

> **Consequence for Study 1:** the SF1000 scale-out sweep can only run **pinned at 8 GPUs**. Its
> 2- and 4-GPU arms must run **cold/streaming**, or the study reduces to a single point. Either run
> SF1000 scale-out entirely cold (consistent across arms, directly comparable), or report SF1000
> pinned only at 8 GPUs as a ceiling number. **Recommend all-cold for SF1000 Study 1** — internal
> consistency matters more than the absolute number.
>
> SF500 has no such problem: pin at 4 and 8, note the 2-GPU spill.

### Reference target at SF1000

`bench/sf1000-repro/` records **8.180 s** for the full 22-query suite on **one GB300**, pinned,
best-of-3, 22/22 byte-identical — peaking at 251.7 of 256 GB with under 2 GB of headroom.

Eight A100s have **2.5× the aggregate VRAM** (640 vs 256 GB) but **31% of the per-GPU HBM**
(80 vs 256 GB). That is the interesting tension: this is a test of whether distribution across
many smaller cards beats one large one. The comparison is already apples-to-apples on data and
query set, so **quote it** — but only over the same 17-query set, since the GB300 run included
the five queries Engine A cannot complete.

### Query-set changes at SF1000

The scale-dependent Tier 4 projections get worse, with one non-obvious exception:

- **q13** — the `2³¹` chars-per-string-column cap on `o_comment` is a **per-CN** limit, so
  **scaling out CNs is itself the mitigation**: 8 CNs halve the per-CN share versus 4. At SF1000
  on 8 CNs the per-CN share is comparable to SF500 on 4 CNs. Run it at 8 GPUs before assuming
  it fails.
- **q17, q21** — staged-inbound volume scales linearly; ~2× the SF500 pressure against the same
  staging arena. Expect these to fail at SF1000 before they fail at SF500.
- **Tier 1/2 (the 8 headline queries)** carry no known scale-dependent failure mode. They are the
  set most likely to survive intact at SF1000, which is another reason to draw the headline
  from them.

Do **not** assume the SF500 result set transfers. Re-run the oracle diff at SF1000; the
`(1 - l_discount)` defect magnitude is itself scale-dependent and worth recording at both points.

---

## The query set — this is the scoping decision

**Instruction taken:** run only the queries known good at SF100 for Engine A.

### Ground truth

Engine A completes **17 of 22** at SF100 — and was faster than StarRocks on **every one of them**.

> **Correction:** `ENGINE-CONFIGS-AND-EQUIVALENCE.md:290` says 18/22. It is **wrong** — q18's 4th
> run wedged. Verified by re-reading all 88 Engine A rows of `tpch-sf100-abc/results.csv`. Use 17.

**Five queries fail** — and none is a query-shape problem:

| query | SF100 behaviour | mechanism |
|---|---|---|
| q05 | 1.8 s → **wedge 180 s** | task enters Computing and never leaves (undiagnosed) |
| q08 | **refused 51.1 s** ×2 | HASH_JOIN requests **142.2 GiB from a 140 GiB pool** off a 12.5 MB input — a sizing bug, 101 identical retries |
| q09 | **refused 64 s / 131 s** | **1.13 TiB single request**; cross-join, O(SF²) intermediates; staging-arena leases leaked |
| q10 | **refused 121 s** → wedge | same freeze family as q05 |
| q18 | 1.1 → 1.0 → **61.5 s** → wedge | monotonic degradation = **a leak**, not query difficulty |

> The dominant pattern is **state accumulating across runs on the same cluster**. q18's
> 1.0 s → 61.5 s → wedge curve is monotonic, not bimodal. A query that is too hard fails on run 1,
> not run 3. More GPUs cannot fix any of these, and every mechanism gets **worse at 5× the data**.

### Verified against the SF500 evidence

An exhaustive sweep of every SF500 artifact on the GB200 box found: **no 22-query Engine A sweep
has ever been run at SF500, on any build.** Every SF500 artifact is restricted to the same four
queries — q01, q04, q06, q14 — plus a `select1` liveness probe. And of those four, **q01 and q14
return numerically wrong values at SF500** (q01 low by 9.56e-04 relative, consistently, across all
12 `scale_test` payloads and all 24 `apply-wins` rows).

So the *directly confirmed, value-sound* Engine A SF500 set is **two queries: q04 and q06**. That
is not a benchmark. The 17-query SF100-derived set is the right scope — it just has to be labelled
as *measured at SF500 for the first time*, not as *known good at SF500*.

### The set

**Run these 17** (all 22 minus the five failures), in this order — `run-abc.sh` preserves the order
you pass, so clean queries run before risky ones and a restart cannot contaminate what matters:

```
q01 q02 q03 q04 q06 q07 q11 q12 q14 q15 q16 q19 q20 q22 q13 q17 q21
```

**Tiers, and which tier the headline comes from:**

| Tier | Queries | Status | Use |
|---|---|---|---|
| **1 — anchor** | `q04 q06` | value-sound at SF500, **measured** | Safe for any claim today |
| **2 — expansion** | `q02 q11 q12 q16 q20 q22` | complete at SF100 with values **byte-identical to the DuckDB oracle**; none touches `(1 - l_discount)` | **Headline, once measured at SF500.** The real path from a 2-query to an **8-query** headline |
| **3 — timing-only** | `q01 q03 q14 q15 q19` | complete, **values wrong** (decimal defect) | Timings + defect magnitude only. **Never in a correctness claim or an aggregate** |
| **4 — probe** | `q13 q17 q21 q07` | each has a *named, quantified* SF500 failure projection. **q07 is also numerically wrong** | Run once, report outcome, exclude from aggregates |
| **5 — excluded** | `q05 q08 q09 q10 q18` | measured hard failures at SF100 | Not in the sweep. See below |

> **q13/q17 caveat.** These are value-correct at SF100, so on correctness alone they belong in
> Tier 2. They sit in Tier 4 purely on **SF500 memory risk**. If they survive the first SF500 run,
> promote them and the headline set grows from 8 to 10.

**Why Tier 4 is worth running:** each of the three fails *loudly and cheaply* (allocator refusal /
arena exhaustion in seconds) rather than wedging, and each has a specific prediction to convert
into a measurement:
- **q13** — `o_comment` is at **85% of cuDF's 2³¹ chars-per-string-column cap** at SF100 per CN.
  SF500 projects to ~4.2× **over** the cap; the 2³¹ guards are unlanded.
- **q17** — in the 1.9–6 GB/CN staged-inbound set; 5× puts it at ~9.5–30 GB/CN against the staging
  arena. Engine A's FP64 decimal lowering also forces a full `sorted_order`+gather over lineitem.
- **q21** — same staged-inbound set, plus actual arena-exhaustion history.

**Why Tier 5 must not be in the sweep:** at SF500 the harness derives warm=900 s / cold=3000 s.
Five wedging queries is **~5.5 hours of wall clock** burned re-confirming known failures — and with
`SIRIUS_QUERY_WATCHDOG_SECS=0` a wedged handler blocks the FE↔CN connection and **poisons the
queries after it**. Run them in a separate, isolated pass with a fresh cluster per query.

### The decimal defect — NOT fixed at HEAD, and the root cause is one function

Verified at HEAD `4e6439c8` on three independent lines:

1. **The code still does it.** `experimental/starrocks/crates/starrocks-plan-translator/src/expr_translator.rs:459-481`
   (`translate_arithmetic`) casts **both operands of every decimal `+ - * / %` to FP64 and declares
   FP64 output**. So `l_extendedprice * (1 - l_discount)` is already FP64 *before SUM ever sees it*.
2. **The crate is untouched since the audit.** `git diff --stat 1d2bbae2..HEAD -- experimental/starrocks/crates/`
   is **empty**. The 8 post-audit commits touch config, nixl, the staging arena and FFI — nothing
   in any expression, decimal, cast or aggregation path.
3. **Post-audit reruns reproduce the wrong values bit-identically.** Two runs dated 2026-08-10
   (`cn2-vs-cn4/`, `nfs-a-vs-c/`) both give q14 = `16.640448956692076` against an oracle of
   `16.640357433254103` — two different CN topologies, two different storage backends, same error.

[`OPEN-ISSUES.md`](../2026-08-09-gb200-sf100/OPEN-ISSUES.md) **#24** tracks this as 🔴 OPEN and warns that fixing the SUM/AVG
lowering at `:826-833` — *what the doc tells you to do* — **would change nothing**. Start at
`translate_arithmetic`.

> **This is the highest-leverage fix available.** It is one function in the Rust translator, it is
> localized to the CN path (standalone Sirius computes these exactly on the same files), and it
> **takes the headline set from 8 queries to 14**. Worth doing before the benchmark, not after.

**The blast radius is 7 queries, not 6.** q05 is a seventh victim: its cold run *passed* with 5
correctly-ordered rows, every value low by ~0.096%. The audit missed it because it diffed only the
`r1` files and `q05.r1.out` is 0 bytes from the warm wedge. q05 stays excluded for wedging, but
record the defect set correctly: **q01 q03 q05 q07 q14 q15 q19**.

> ⚠️ **Two traps that look like fixes — do not cite either.**
> `engineA-fixed-q01-q04-q06-q14.png` is dated **2026-08-08, the day *before* the audit**;
> "fixed"/"httpfix" refer to the CN `http_port` advertisement fix (`1d2bbae2`) — an *availability*
> fix, not an arithmetic one. Its subtitle claims *"results bit-identical to the DuckDB oracle"*,
> which is **false** for the q01 and q14 bars it draws.
> And `REVIEW-benchmark-findings.md:187` localizes the defect to the CN wrapper — that is a
> diagnosis, not a repair.

### Mandatory: the harness does not check values

`run-abc.sh` defines `status=pass` as *"exit code 0 and at least one row."* **It never compares
values.** Every result must be diffed against a pure-CPU DuckDB oracle (`SET gpu_execution = false`)
on the same SF500 files, using a **relative tolerance of 1e-12 — not string equality**.

> q06 legitimately returns three adjacent doubles across runs
> (`61662234676.307495` / `.3075` / `.30751`, exactly ±1 ULP). An exact comparator flags it as a
> failure about a third of the time. Without a relative-tolerance oracle diff this sweep produces
> another table of liveness checks and we learn nothing about correctness.

### Chart labelling consequence

The existing chart's subtitle *"all 22 queries complete"* is **false for Engine A and must change**:

> `17 of 22 queries · q05/q08/q09/q10/q18 excluded (Engine A failures) · aggregates over the 8-query Tier 1+2 set`

Engines B and C pass **more** queries than Engine A. Every aggregate must be restricted to the
Engine A set, and that restriction named in the footnote.

---

## Study 1 — GPU Scale-Out

**Internal:** `scale-out` · **Chart:** `Sirius GPU Scale-Out · TPC-H SF500`
**Subtitle:** `2 → 4 → 8 × A100 80 GB SXM4 · single node, NVLink`

### Goal

How does Sirius scale as GPU count doubles, with I/O held constant? Prior GB200 data (1→2→4 CNs,
4 queries) showed a **2-CN regression on q06** — worth understanding, and this sweep should
reproduce or refute it.

| Config | CNs | GPUs | Pinning feasible? |
|---|---|---|---|
| **2× A100** | 2 | 0–1 | **Marginal** — 66 GB/GPU of 80 GB before working memory. Expect spill |
| **4× A100** | 4 | 0–3 | Yes — 33 GB/GPU |
| **8× A100** | 8 | 0–7 | Yes — 16.5 GB/GPU |

**Run this study pinned/hot.** Pinning is legitimate when comparing Sirius to itself, and it
isolates *compute* scaling from *I/O* scaling — which is exactly the question. Report the cold arm
separately; if the 2-GPU pinned arm spills, report it as a finding rather than forcing it.

**Cost note:** the box is priced whole at **$13.25/hr** regardless of how many GPUs are used, so
scaling efficiency converts **directly** into cost efficiency. If 8 GPUs are 3× faster than 2, they
are 3× cheaper per run. Say so on the chart — it is the strongest framing available.

**Chart:** X = GPU count (2/4/8, linear), Y = total time for the Tier 1+2 set (s, log), speedup
annotated per point. Second panel: per-query speedup heatmap, to separate the queries that scale
from the communication-bound ones.

---

## Study 2 — Sirius vs cudf-polars

**Internal:** `gpu-shootout` · **Chart:** `Sirius vs cudf-polars · TPC-H SF500`
**Subtitle:** `8 × A100 80 GB · cold — page cache dropped per run · same box, same data`

### Goal

A defensible GPU-vs-GPU comparison with the five biases from the SF100 audit removed. Both engines
on the **same box**, so hardware is exactly matched for the first time.

### Fixes vs the contaminated SF100 run

| Audit item | Fix |
|---|---|
| A wrong on 6 queries | Tier 3 excluded from aggregates; defect magnitude reported separately |
| C ran `--interleave=all` (−11.4%) | Drop `numactl`; let cudf-polars `bind_to_gpu(hardware_binding)` bind per worker |
| "cold" was first-touch, not cache-dropped | `sync && drop_caches` before each run (A); `--io-mode cold` (C) — **symmetric** |
| A used KvikIO, not the fastest path | `use_sirius_datasource: true` + tuned uring. **Confirmed by the SF1000 config, which also uses uring** |
| Memory policy mismatch (A capped, no spill; C spills) | The SF1000 config's `downgrade_*` fractions + disk spill root **close this gap** — verify they engage under a CN |
| C quoted from `--iterations 1` | `--iterations 4`: run 0 cold, 1–3 warm |

### The regime conflict — resolved

The supplied SF1000 config is a **pinned/hot** config (`SIRIUS_PIN_TIER_*=gpu` for all 8 tables in
`run.sh`); the brief asks for **cold**. These are different measurements, and pinning is
**unfair here**: cudf-polars has no equivalent — `--io-mode hot` still re-reads parquet every
iteration. A pinned Sirius vs a re-reading cudf-polars would be a worse violation than anything in
the SF100 audit.

**→ Study 2 runs cold, with `SIRIUS_PIN_TIER_*` unset.** Keep the config's *tuning*; drop its
*pinning*. Pinned numbers belong to Study 1 (Sirius vs itself), where they are legitimate.

### The cold-mode trap that will silently ruin this study

`ast_jit` compiles kernels with NVRTC on first use — **~19 s on a cold cache**, suite first-iteration
28.99 s → 10.50 s warm. The cache lives at `$HOME/.cudf/$VERSION/$ARCH` and **survives process
restart**, but A100 is a new arch so it starts empty.

**Pre-warm the JIT cache before any timed run, and never drop it as part of "cold".** "Cold" means
*page cache dropped*, not *compiler cache dropped*. Otherwise we measure NVRTC, not Sirius.

### Engine C

```bash
cd <a directory with NO duckdb/ child>   # the Sirius repo's duckdb/ submodule shadows the
                                          # duckdb package as an empty namespace package and
                                          # kills cudf-polars in 2 s. Cost us a full sweep once.
python -m cudf_polars.streaming.benchmarks.pdsh <tier 1+2 query list> \
  --frontend ray --num-gpus 8 \
  --path /raid/.../tpch_parquet_sf500 --suffix '' \
  --io-mode cold --iterations 4
```

---

## Study 3 — Cost Efficiency

**Internal:** `cloud-cost` · **Chart:** `TPC-H SF500 — Sirius (GPU) vs StarRocks (CPU) · Cost per Run`
**Subtitle:** `17 of 22 queries · on-demand $/hr · lower-left is better`

```
cost_per_run_usd = (wall_seconds / 3600) × $/hr
```

| Engine | Instance | $/hr | $/second |
|---|---|---|---|
| Sirius (GPU) | 8× A100 80 GB | $13.25 | $0.003681 |
| StarRocks (CPU) | m8i.48xlarge (Intel) | $12.19 | $0.003386 |
| StarRocks (CPU) | m8gd.48xlarge (Graviton4) | $8.83 | $0.002453 |

### The break-even, stated up front

Because the prices are close, **this study is decided almost entirely by speed**:

| Sirius must beat | by | to break even on cost |
|---|---|---|
| m8i.48xlarge | **1.09×** | nearly free — any GPU win is a cost win |
| m8gd.48xlarge | **1.50×** | the real bar |

At SF100 Engine A beat StarRocks on **every** one of the 17 passing queries, several by 3–4×
(q17 0.9 s vs 3.3 s, q21 1.6 s vs 6.9 s, q22 0.6 s vs 2.3 s). If that holds at SF500, Sirius
clears both bars comfortably. **State the break-even ratio on the chart** — it is more honest and
more useful than a raw cost ratio, because it survives price changes.

### Engine B configuration — carry forward from the SF100 audit

| Setting | Value | Why |
|---|---|---|
| BE count | **1 per NUMA domain** | Check `numactl -H` on each instance; do not assume 2 |
| `num_cores` | vCPU ÷ BE count, **explicitly** | StarRocks' `CpuInfo` never calls `sched_getaffinity`, so a pinned BE otherwise reports all 192 |
| `mem_limit` | **explicit GiB, never `"90%"`** | These are `Swap: 0` boxes; a percentage resolves against the wrong total and guarantees an OOM-kill. 768 GB total → ~240 GB/BE for 2 BEs, leaving ~290 GB page cache = 2.2× the dataset |
| Spill | **OFF** (default) | A loud refusal beats a silently slow pass |
| Storage | local NVMe | m8gd has it; **m8i does not** — it needs EBS, which is a real I/O asymmetry. Disclose it |

> **m8gd vs m8i is not just arm-vs-x86.** m8gd has local NVMe, m8i does not. If both are run, that
> confounds the architecture comparison with a storage comparison. **Recommend m8gd as primary** —
> it is cheaper, has local NVMe matching the GPU box's `/raid`, and Graviton matches the SF100
> audit's lineage. Use m8i only as a secondary x86 reference, with the storage difference named.

---

## Config: re-deriving for A100 80 GB

The committed [`bench/sf500-gb200/sirius-sf500.yaml`](../../bench/sf500-gb200/sirius-sf500.yaml) targets
GB200 (185 GiB). **A100 is 80 GB — 43% of that.** Deltas, on top of the GB200 derivation in
[`bench/sf500-gb200/README.md`](../../bench/sf500-gb200/README.md):

| Knob | GB300 source | GB200 | **A100 80 GB** | Why |
|---|---|---|---|---|
| `usage_limit_fraction` | 0.95 | 0.86 | **~0.80** | The staging arena is out-of-pool. On 80 GB a 16 GiB arena is **20% of the card** — shrink the arena to 4–6 GiB first, then set the fraction |
| `scan_task_batch_size` | 8GB (peaked 253.9/256 GB) | 6GB | **sweep {2, 3, 4} GB** | Naive scale of 8 GB by 80/256 is 2.5 GB. SF500 is half of SF1000, which relieves it. The source calls this *the one knob that matters* — a step, not a gradient |
| `host.capacity_bytes` | 471.2 GB | 160 GiB | **~120 GiB/CN** | 1500 GB ÷ 8 CNs = 187.5 GB; leave real page cache for a 132 GB dataset |
| `downgrade_*` | 0.9 / 0.85 | 0.9 / 0.85 | **keep, but tune to avoid spilling** | No Grace C2C — host spill crosses PCIe. A cliff, not a slope |
| `hash_partition_bytes` | 32GB | 32GB | **8–16GB** | 32 GB of partition state does not fit alongside data on an 80 GB card |
| `max_build_hash_table_bytes` | 32GB | 32GB | **8–16GB** | Same |
| `scan_manager.num_threads` | 18 (72c) | 9 | **host cores ÷ 8 CNs** | Measured inert 18 vs 24 — do not over-think, just don't exceed the CN's core allotment |
| `uring_n_reactors` | 4 | sweep 4/8/32 | **sweep 4/8/32** | Different storage, different optimum |
| `pipeline.num_threads` | 8 | 8 | **8** | Measured inert 6/8/12 |
| `enable_prefetch_cache` | false | false | **false** | Two independent sources: `true` is a **2.1× regression** |
| `expression_evaluator_strategy` | `ast_jit` | `ast_jit` | **`ast_jit`** | **−4.17% suite for zero code.** `src/config.cpp:27` ships the slow `AST_INTERPRET`. Free win — take it everywhere |

### What we inherit from PR #1371 for free

`bench/sf1000-repro/` is **merged on `demo-multi-cn`**, so the six Sirius-side wins are already in
the build: q17 `LOGICAL_DELIM_GET`, `interruptible_mpmc` sentinels, q16 count-distinct radix label
(**q16 −39%**), q19 OR-branch + dictionary pushdown, Bloom fast-range (**q21 −15.2%**), `cuco`
bucket 1→4. Only the **three cuDF patches** are external (`strings::like` → q13 −36.5%;
memcpy threshold → q9 −5.8%; groupby shmem → q1 −5%).

**Decision:** the cuDF patches are *not* required for a valid benchmark, and building them for x86
is real work. Recommend running **without** them, and disclosing that — the numbers are then
reproducible from merged code alone, which is a stronger claim. Revisit only if q13 becomes the
bottleneck.

---

## Prerequisites

**Blocking:**
- [ ] **Rebuild Sirius for x86_64** — the GB200 aarch64 build does not transfer
- [ ] Stage the SF500 dataset on the A100 box; confirm local NVMe, not NFS (`findmnt -T`)
- [ ] Shrink `SIRIUS_EXCHANGE_STAGING_BYTES` from 16 GiB (20% of an 80 GB card) and re-derive
      `usage_limit_fraction`
- [ ] `mkdir` per-CN downgrade spill dirs on local NVMe; check free space × 8
- [ ] Pre-warm the NVRTC cache for `sm_80` and confirm it persists

**Before trusting any number:**
- [ ] Stand up the DuckDB oracle at SF500 and wire a **relative-tolerance (1e-12)** diff into the harness
- [ ] Verify the `downgrade_*` path actually engages under a CN (force pressure; watch the spill dir)
- [ ] Sweep `scan_task_batch_size` ∈ {2,3,4} GB and `uring_n_reactors` ∈ {4,8,32}
- [ ] Arm `SIRIUS_QUERY_WATCHDOG_SECS` (currently 0)
- [ ] Confirm q11's `FRACTION` is `0.0001/500` at SF500 — the literal-vs-spec choice silently
      changes the answer by 500× and historical CSVs used the wrong one
- [ ] Confirm `numactl -H` topology on each AWS instance before choosing BE count

**Sequencing:**

1. **Study 2 at SF500** — single box, no new infra, lowest risk. Establishes the queryset actually
   works at scale and shakes out the config.
2. **Study 1 at SF500** — same box, add the 2/4/8 topologies. Pinned at 4 and 8.
3. **Studies 1 + 2 at SF1000** — reuse everything; only the memory knobs change. Run Study 1
   all-cold here (pinning only fits at 8 GPUs).
4. **Study 3** — needs AWS provisioning, so it runs independently and can start in parallel with
   any of the above. Run it at **both** scale factors; the cost curve's shape across SF is the
   result, not any single point.

SF1000 costs roughly 2× the wall clock of SF500 per sweep. Budget accordingly, and do not start
SF1000 until the SF500 oracle diff is green — an unvalidated sweep at 2× the cost is 2× the waste.

---

## What NOT to do

- Never quote cudf-polars from `--iterations 1` — first-touch, 1.4–4.6× warm-up penalty. **This
  single mistake produced the earlier "Sirius beats cudf-polars" chart.**
- Never compare a **pinned** Sirius against a re-reading cudf-polars
- Never treat `status=pass` as correct — the harness never compares values
- Never use exact equality in the oracle diff — q06 varies by ±1 ULP legitimately
- Never drop the NVRTC cache as part of "cold" — that measures the compiler
- Never publish "all 22 queries complete" for Engine A — it is 17, and aggregates are over 8
- Never run the Tier 5 failures in the measurement sweep — ~5.5 h of wedges that poison what follows
- Never set `SIRIUS_QUERY_WATCHDOG_SECS=0`
- Never set `enable_prefetch_cache: true` — 2.1× regression, confirmed twice
- Never set StarRocks `mem_limit="90%"` on a `Swap: 0` box
- Never verify NUMA with `Mems_allowed_list` — use `/proc/<pid>/numa_maps`
- Never run Engine C from a directory with a `duckdb/` child
- Never use `LD_LIBRARY_PATH` for a patched libcudf — `DT_RPATH` wins; use `LD_PRELOAD`
