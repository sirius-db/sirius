# Adapting the SF1000 repro config to Engine A on GB200

**Source:** [`felipeblazing/sirius@901287a1`](https://github.com/felipeblazing/sirius/blob/901287a1d55d01614966c83b4745a11b67443a52/bench/sf1000-repro/sirius-sf1000.yaml) · `bench/sf1000-repro/`
**Fetched:** 2026-08-12 · **In-repo source:** [`../sf1000-repro/sirius-sf1000.yaml`](../../bench/sf1000-repro/sirius-sf1000.yaml)
— already merged here as PR #1371, so the fetched copy was redundant and was dropped.
**Adapted config:** [`sirius-sf500.yaml`](../../bench/sf500-gb200/sirius-sf500.yaml) ·
**Study plan:** [`../../TPCH-BENCHMARK-PLAN.md`](TPCH-BENCHMARK-PLAN.md)

> **Scope note:** this file derives the config for **GB200 (185 GiB)**. The benchmark has since
> been retargeted to **8× A100 80 GB**, which is 43% of that — see the A100 re-derivation table in
> the study plan. The reasoning below still applies; the numbers shrink again.

---

## Read this first — three things the config file alone does not tell you

I read the config's sibling `README.md` and `run.sh` in the same directory. Three facts
materially change how this config can be used, and none are visible in the YAML:

### 1. It is Engine D, not Engine A

The README states plainly: *"The implementation uses Sirius, a GPU-accelerated query engine
built on DuckDB, **not StarRocks**."* The runner invokes
`python3 test/tpch_performance/performance_test.py --engine gpu`, not a StarRocks FE + CN cluster.

The YAML's `sirius:` tree **does** map onto an Engine A CN — a CN is `num_gpus: 1` and reads the
same config schema — so the memory / executor / operator_params sections transfer. But the
**measured 8.18 s and the 22/22 claim were produced by Engine D**, and do not transfer.

### 2. It requires three out-of-tree libcudf patches

`build-libcudf.sh` builds a patched libcudf from `felipeblazing/cudf`, and `run.sh`
`LD_PRELOAD`s it. The patches and their measured value:

| Patch | Commit | Effect |
|---|---|---|
| `strings::like` backtrack skip | `4a345cc` | q13 **−36.5%** |
| memcpy 2 MiB threshold | `9af88b0` | q9 **−5.8%** |
| groupby shmem replication | `7375a46` | q1 **~−5%** |

*"These patches are not proposed upstream."* The branch itself is **"for reproduction only.
It is not proposed for merge"**, and *"the project's own unit and SQLLogic suites have not been
run against this stack."*

**→ Without the patched libcudf, this config does not reproduce its numbers.** Building it is a
prerequisite task, not a detail.

### 3. It is a PINNED (hot) benchmark — data is resident on the GPU

`run.sh` sets:

```bash
SIRIUS_PIN_TIER_LINEITEM=gpu   SIRIUS_PIN_TIER_ORDERS=gpu    SIRIUS_PIN_TIER_PART=gpu
SIRIUS_PIN_TIER_CUSTOMER=gpu   SIRIUS_PIN_TIER_SUPPLIER=gpu  SIRIUS_PIN_TIER_NATION=gpu
SIRIUS_PIN_TIER_REGION=gpu     SIRIUS_PIN_TIER_PARTSUPP=gpu
SIRIUS_PRE_SQL="pin_table_compression = true;
                pin_table_input_compression_plan_dir = <plans/>;
                expression_evaluator_strategy = 'ast_jit'"
--mode grouped --engine gpu --pin host --iterations 3
```

All eight tables are **pinned into GPU memory before timing**. There is essentially **no parquet
read inside the timed loop.**

This is the same regime as the green "hot — steady state, data resident" squares in the existing
AWS chart, whose footnote confirms it: *"Pinned tier per SF — SF100/300/500 GPU-tier; SF1000
host-tier compressed... pinned steady state is box-invariant within ~3%."*

**→ This directly conflicts with the brief for Study 2**, which specified *cold mode with the
fastest datasource for this box*. If the tables are pinned, the datasource is almost irrelevant —
you are no longer measuring I/O at all. See "The regime conflict" below.

---

## The diagnostic this hands us for free

Engine D on this stack gets **22/22 byte-identical** at SF1000. Engine A at SF100 fails
q05/q08/q09/q10 and returns numerically wrong values on q01/q03/q07/q14/q15/q19.

Corroborating from our own box: the `snmg-sf500` run (**Engine D**, standalone) verified
q01 at SF500 as `RF 18878152156.0028307869` — **matching the pure-CPU DuckDB oracle exactly**.
Engine A gets the same query wrong.

> **The `(1 - l_discount)` decimal defect and the q05/q08/q09/q10 failures live in the
> StarRocks CN integration layer — FE planning, fragment distribution, or the
> StarRocks→Substrait→Sirius type mapping — not in the Sirius GPU engine.**

None of the three libcudf patches touch decimal arithmetic (they are `strings::like`, memcpy
sizing, and groupby shared-memory), so the patched stack is not what makes Engine D correct.
Engine D was already correct.

This is worth an hour of investigation before the benchmark runs: it reframes the defect from
"a GPU kernel bug" to "a plan-translation bug", which is a much cheaper fix and would unlock
6 queries for the headline correctness claim.

---

## Hardware deltas

| | GB300 (source) | GB200 (`presto-gb200-gcn-17/18`) | Ratio |
|---|---|---|---|
| HBM per GPU | 256 GB | **189,471 MiB = 185.0 GiB = 198.7 GB** | **0.776×** |
| SMs | 152 | (GB200, compute cap 10.0) | — |
| Host CPU | 72-core Grace, 1 socket | 144 cores, **2 CPU NUMA nodes** | 2× |
| Host RAM | ~500 GB implied | 956.82 GiB LPDDR (478.41 GiB/node) | — |
| GPUs used | 1 | 4 per box (1 per CN), 8 across two boxes | — |
| Scale factor | SF1000 | **SF500** | 0.5× |

Two effects pull in opposite directions: **HBM is 22% smaller** (pushes batch sizes down) but
**the dataset is half the size** (pushes them up). Neither dominates cleanly, so the memory-sized
knobs must be swept, not guessed.

---

## Per-knob adaptation

| Knob | Source value | Engine A / GB200 / SF500 | Rationale |
|---|---|---|---|
| `topology.num_gpus` | `1` | **`1`** — keep | A CN owns exactly one GPU. Maps directly. |
| `gpu.usage_limit_fraction` | `0.95` | **`0.95`** → 175.8 GiB | Replaces the audit's 140 GiB hard cap. **+26% GPU memory per CN**, a real change. |
| `gpu.downgrade_trigger_fraction` | `0.9` | **`0.9`** | See note below — this is the spill path. |
| `gpu.downgrade_stop_fraction` | `0.85` | **`0.85`** | |
| `host.capacity_bytes` | `471200000000` (471.2 GB) | **~235 GB for 4 CNs**, ~470 GB for 2 CNs | 2 NUMA nodes × 478.41 GiB. With 4 CNs, two share each node → halve. Oversubscribing invites the OOM-kill the audit warns about on this `Swap: 0` box. |
| `host.initial_number_pools` | `392` | **196** (4 CNs) / 392 (2 CNs) | Scale with `capacity_bytes`; 392 × 8 × 64 MiB ≈ 201 GB of pool structure. |
| `host.pool_size` / `block_size` | `8` / `67108864` | **unchanged** | Not memory-total dependent. |
| `disk.downgrade_root_dirs` | `/localhome/local-faramburu/...` | **`/raid/prestouser/aocsa/sirius_disk_memory`** | **MUST change — the source path does not exist here.** Put it on local NVMe (`/dev/md0`), never NFS. Needs a distinct subdir per CN to avoid collisions. |
| `disk.capacity_bytes` | `1000000000000` (1 TB) | **1 TB**, verify free space × N CNs | 4 CNs spilling 1 TB each = 4 TB. Check `df /raid` first. |
| `scan_manager.use_sirius_datasource` | `true` | **`true`** | Confirms the uring path over KvikIO — matches the Study 2 recommendation and `scan-defaults-sweep.md`. |
| `scan_manager.uring_n_reactors` | `4` | **SWEEP 4 / 8 / 32** | Our box's own sweep found r=8 best on NFS and r=32 best on NVMe; the source used 4. Different media, different optimum. |
| `scan_manager.num_threads` | `18` (of 72 cores) | **18** (2-CN, 72 cores each) / **9** (4-CN, 36 cores each) | Source measured 18 vs 24 as inert (−0.33%), so exact value is not critical — but do not exceed the CN's `physcpubind` width. |
| `scan_manager.memory_prefetcher` | `enable: true, num_threads: 3` | **keep** | Not previously in our configs. Distinct from `enable_prefetch_cache`. |
| `scan_manager.enable_prefetch_cache` | `false` | **`false`** | Independently confirmed by our `scan-defaults-sweep.md`: `true` is a 2.1× regression. Two sources agree — do not flip. |
| `pipeline.num_threads` | `8` | **`8`** | Source measured 6/8/12 within 0.55%. Inert. |
| `downgrade.num_threads` | `1` | **`1`**, consider 2–4 | Source barely spilled at 0.95 usage. At SF500 on 22% less HBM we may spill more, making this a live knob. |
| `task_creator.num_threads` | `4` | **`4`** | |
| `scan_task_batch_size` | `8GB` | **SWEEP 5 / 6 / 8 GB** | **The one knob the source says mattered** (5→8 GB = −1.85%, q4 −25%, q12 −18%) and explicitly a *step, not a gradient*. But 8 GB peaked at **253.9 of 256 GB** — 99.2% of HBM. Naively scaled to 198.7 GB that is 6.2 GB. SF500 is half the data, which relieves it. **Must be measured.** |
| `hash_partition_bytes` | `32GB` | **32GB** | Source: 16 vs 32 GB = −0.06%. Inert. |
| `max_build_hash_table_bytes` | `32GB` | **32GB**, watch at SF500 | Join build sides scale with data; a 140→176 GiB cap change interacts here. |
| `concat_batch_bytes` | `5GB` | **5GB** | |
| `max_sort_partition_bytes` | `0` | **`0`** (unlimited) | |
| `telemetry.enable_quent` | `false` | **`false`** for timed runs | Turn on only for the diagnostic pass — it is measurement overhead. |

### The `downgrade_*` fractions are more important than they look

The SF100 audit recorded Engine A as having **"no host spill implemented"** and a hard 140 GiB
cap — listed as divergence #3 against cudf-polars, *"latent at SF100, first-order at SF≥500."*

This config sets `downgrade_trigger_fraction: 0.9` / `downgrade_stop_fraction: 0.85` **plus a
disk spill root**. That is a working device→host→disk downgrade path. If it functions under
Engine A, it **closes divergence #3** — the single biggest structural unfairness against Sirius
in the SF500 comparison, since cudf-polars has had device→host spill all along.

**Verify it actually engages under a CN**, do not assume: force pressure and confirm the
downgrade executor runs and the spill dir fills.

---

## The regime conflict — this must be resolved before Study 2 runs

The brief for Study 2 says *"both in cold mode but with the fastest datasource for this box."*
The supplied config is a **pinned/hot** config. These are mutually exclusive measurements:

| | Cold / streaming | Pinned / hot (this config) |
|---|---|---|
| What is timed | parquet read + decode + compute | compute only |
| Datasource choice | **decisive** | nearly irrelevant — data already resident |
| `SIRIUS_PIN_TIER_*` | unset | `gpu` for all 8 tables |
| cudf-polars counterpart | `--io-mode cold` | no true equivalent — it re-reads parquet every iteration |
| Answers | "how fast end to end from storage" | "how fast is the compute engine" |

There is a further problem with the pinned regime for a **comparison**: cudf-polars has no
equivalent of pinning tables into GPU memory across queries — `--io-mode hot` still re-reads
parquet from page cache every iteration. **A pinned Sirius vs a re-reading cudf-polars is not a
fair fight**, and would be a worse apples-to-apples violation than anything in the SF100 audit.

**Recommendation:** run **both regimes**, and label them as the existing chart already does
(filled = hot/pinned, open = cold):

1. **Cold / streaming** — for the Sirius-vs-cudf-polars comparison (Study 2). Both engines read
   parquet, caches dropped. This is the defensible head-to-head. Use this config's tuning but
   with `SIRIUS_PIN_TIER_*` **unset**.
2. **Pinned / hot** — for Sirius scale-out (Study 1) and as a Sirius-only "engine ceiling" number.
   Pinning is legitimate when comparing Sirius to itself across 2/4/8 GPUs, and it isolates
   compute scaling from I/O scaling, which is exactly what Study 1 wants.

At SF500 the existing chart's footnote says the pin tier is **GPU** (not host-compressed as at
SF1000) — but that was on a 96 GB card. SF500 lineitem alone is large; confirm the GPU tier
actually fits in 185 GiB per CN before relying on it.

---

## Prerequisite checklist before any run

- [ ] Build the patched libcudf (`build-libcudf.sh`, three commits from `felipeblazing/cudf`)
      and confirm `LD_PRELOAD` works under a **CN**, not just the standalone binary
- [ ] Decide whether the CN inherits `SIRIUS_PRE_SQL` — `expression_evaluator_strategy = 'ast_jit'`
      and `pin_table_compression` are session settings; confirm the CN applies them
- [ ] `mkdir -p /raid/prestouser/aocsa/sirius_disk_memory/cn{0..3}` and check `df /raid` headroom
- [ ] Verify the downgrade path engages under Engine A (see above)
- [ ] Sweep `scan_task_batch_size` ∈ {5, 6, 8} GB at SF500 on GB200
- [ ] Sweep `uring_n_reactors` ∈ {4, 8, 32} on `/raid`
- [ ] Confirm whether the compression `plans/` directory is portable or must be regenerated
      for SF500 on this box
- [ ] Resolve: does `usage_limit_fraction: 0.95` (175.8 GiB) leave room for the 16 GiB
      `SIRIUS_EXCHANGE_STAGING_BYTES` arena, which is a bare `cudaMalloc` **outside** the RMM pool?
      175.8 + 16 = 191.8 GiB > 185 GiB. **This looks like an over-commit — reduce the fraction to
      ~0.86 for Engine A, or shrink the staging arena.**

> The last item is the most likely thing to bite immediately. The source config was written for a
> standalone engine with **no exchange staging arena at all**; Engine A adds a 16 GiB out-of-pool
> allocation per CN that `usage_limit_fraction` knows nothing about.
