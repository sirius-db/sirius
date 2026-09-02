# Tuning Sirius + StarRocks on this 4× GB200 box

Self-contained procedure for the StarRocks CN path on **this machine**: 4 CNs, one per GB200
GPU, TPC-H against external parquet. How the knobs work, which values are committed here, and
how to switch scale factor **without rebuilding**.

Companion to:

| Need | Doc |
|---|---|
| Box facts (NUMA, NV18, HBM nodes) | [`HARDWARE.md`](HARDWARE.md) |
| Build + smoke on this fleet | [`BUILD-AND-SMOKE.md`](BUILD-AND-SMOKE.md) |
| Memory arithmetic, every comment | [`../../experimental/starrocks/configs/gb200-4gpu/engine-a.env`](../../experimental/starrocks/configs/gb200-4gpu/engine-a.env) |
| Launcher | [`../../experimental/starrocks/configs/gb200-4gpu/cluster4-numa.sh`](../../experimental/starrocks/configs/gb200-4gpu/cluster4-numa.sh) |
| Transport env vars | [`../../experimental/starrocks/docs/TUNABLES.md`](../../experimental/starrocks/docs/TUNABLES.md) |
| 22-query harness (any box) | [`../rtxpro6000-2gpu/TPCH-SWEEP-RUNBOOK.md`](../rtxpro6000-2gpu/TPCH-SWEEP-RUNBOOK.md) |
| Method this file specialises | [`../rtxpro6000-2gpu/SIRIUS-TUNING-RUNBOOK.md`](../rtxpro6000-2gpu/SIRIUS-TUNING-RUNBOOK.md) |

Do **not** copy timings from `notes/2026-08-09-gb200-sf100/`. Do **not** launch with
`benchmarks/cluster8.sh` here — it has no NUMA/HBM interlock and no port preflight.

---

## 0. The two config paths are mutually exclusive

Same trap as every other StarRocks CN box.

| Path | How | Reaches |
|---|---|---|
| **A — derived** | `cluster4-numa.sh` always passes `--gpu-memory-limit` / `--host-memory-limit`; the CN writes `.cn<i>/derived-sirius-config.yaml` | memory limits, CPU affinity, telemetry dir — **nothing else** |
| **B — full file** | `--sirius-config <path>` (or `$SIRIUS_CONFIG_FILE`) | everything, including `operator_params` |

**`--sirius-config` conflicts with every memory flag.** Path A is what this launcher does.
Operator knobs (`scan_task_batch_size`, `hash_partition_bytes`, …) stay at
`derived_default_batch_size()` until someone drops those two CLI flags and puts
`usage_limit_bytes` in YAML instead.

What path A writes on this box (one CN, `SCALE_FACTOR=1000`):

```yaml
sirius:
  topology: {num_gpus: 1}
  memory:
    gpu:  {usage_limit_bytes: "128GiB", reservation_limit_fraction: 1.0}
    host: {capacity_bytes: "112GiB"}
  executor:
    scan_manager:  {cpu_affinity: [<that GPU's local_cpulist>]}
    task_creator:  {cpu_affinity: [...]}
    downgrade:     {cpu_affinity: [...]}
  telemetry: {output_directory: ".cn0/telemetry"}
```

`cpu_affinity` is discovered from the GPU's NUMA node. With `CPU_SPLIT=disjoint` (the
`engine-a.env` default) that is 36 cores, not 144.

---

## 1. Units

Byte-valued keys use Kubernetes/systemd conventions:

- `K/KB/M/MB/G/GB/T/TB` are powers of **1000**
- `Ki/KiB/Mi/MiB/Gi/GiB/Ti/TiB` are powers of **1024**

`8GB` in YAML is 8e9 bytes, not 8 GiB. Committed configs mix both (`scan_task_batch_size: 8GB`
next to `GPU_MEM=128GiB`). Do not assume.

---

## 2. This card's device budget

```
usable HBM / GPU     188,416 MiB = 184.00 GiB     (nvidia-smi used+free; NOT nameplate 189,471)
CUDA ctx + cuDF      779 MiB     = 0.76 GiB       (measured on this box, two CN generations)
CPU LPDDR            979,783 MiB = 956.82 GiB     (NUMA 0+1 only)
swap                 0
```

`free -g` reports ~1692 GiB. That **counts the four GPU HBM NUMA nodes**. Never budget from it.

```
device occupancy = GPU_MEM  +  STAGING  +  0.76 GiB
                   (RMM pool)  (bare cudaMalloc, OUTSIDE the pool)
```

**Staging is not subtracted from `--gpu-memory-limit`.** Raising `STAGING` without lowering
`GPU_MEM` overruns the card. SF100 keeps `GPU_MEM + STAGING ≤ 159,744 MiB` (156 GiB). SF1000
accepts 160 GiB of that sum and 12.6 % headroom — still inside the 85–97 % band used everywhere
else, just less slack.

### SCALE_FACTOR presets (path A, 4 CNs)

These are what `engine-a.env` exports when you pass `SCALE_FACTOR=<N>` and do not override the
individual knobs. Switching SF is a **relaunch**. The binary does not change.

| SCALE_FACTOR | Dataset on this box | Size | GPU_MEM | STAGING | HOST_MEM | occupancy | watchdog | RPC timeout |
|---|---|---|---|---|---|---|---|---|
| **100** (default) | `/raid/prestouser/aocsa/tpch_parquet_sf100` | 26 G | 140 GiB | 16 GiB | 160 GiB | 156.8 / 184 = 85 % | 0 | 60 s |
| **500** | `/raid/prestouser/aocsa/tpch_parquet_sf500` | 132 G | 132 GiB | 24 GiB | 160 GiB | 156.8 / 184 = 85 % | 180 | 180 s |
| **1000** | `/raid/prestouser/aocsa/tpch_parquet_sf1000_f64` | **380 G** | 128 GiB | 32 GiB | **112 GiB** | 160.8 / 184 = 87 % | 300 | 300 s |
| **3000** | `/raid/prestouser/aocsa/tpch_parquet_sf3000_f64` | 1.2 T | 120 GiB | 36 GiB | 16 GiB | 156.8 / 184 = 85 % | 600 | 900 s |
| **10000** | `/raid/prestouser/aocsa/tpch_parquet_sf10000_f64` | 3.8 T | 112 GiB | 44 GiB | 16 GiB | 156.8 / 184 = 85 % | 1800 | 1800 s |

`HOST_MEM` is a lazily-grown **ceiling**, not a reservation. Measured CN host RSS at rest is
1.0–1.5 GiB. It yields so page cache can hold the parquet:

- SF100 / SF500: 4×160 = 640 GiB committed; leftover cache many times the dataset.
- SF1000: 4×112 = 448 GiB committed; leftover **~393 GiB** against a 380 GiB f64 tree (≈1.03×).
  Use **112**, not 120 — 4×120 leaves 373 GiB, under the dataset.
- SF3000 / SF10000: the dataset is larger than CPU LPDDR. `HOST_MEM=16GiB` is intentional.
  Scans miss cache and hit `/raid` NVMe. That is the only legal split; it is not a claim the
  suite will finish.

**Do not use** `STAGING ≈ 96 GiB × (SF/500) / N`. That formula is retired (A100 campaign floor;
it underestimated even there). It would ask for 144 GiB of arena per CN at SF3000, which does
not fit 184 GiB of HBM. The table above **caps occupancy** and grows staging only until the
pool would drop below ~110 GiB.

### HOST_MEM and NUMA

`--membind` is **0 or 1 only**. Nodes 2 / 10 / 18 / 26 **are** the four GPUs' HBM. `cluster4-numa.sh`
refuses a membind onto a CPU-less node. Never `--interleave=all`.

Under a hard membind, 200 GiB/CN exceeds the per-node ceiling on node 1. That is why SF100 here
is 160 GiB, not the unpinned baseline's 200.

---

## 3. Operator knobs (path B only)

All default to:

```
derived_default_batch_size() = clamp(min_visible_GPU_total_bytes / 40, 512 MiB, 5 GiB)
```

On this card: `clamp(184 GiB / 40) = 4.6 GiB`, under the 5 GiB clamp. The GB300 measured
optimum for `scan_task_batch_size` was `8GB` (a **step**, not a gradient). The auto-derivation
cannot reach that. Committed ratios cluster at **~3 % of HBM** for scan batches → **~6 GB**
here. Hash knobs do **not** interpolate linearly; leave them unless HASH_JOIN OOM persists
after memory is sized.

`cluster4-numa.sh` cannot set these today. Path B means editing the launcher to drop
`--gpu-memory-limit` / `--host-memory-limit` and putting `usage_limit_bytes: 128GiB` (or the
preset's pool) in YAML.

`bench/sf1000-repro/sirius-sf1000.yaml` is a **1-GPU GB300** in-process config (`num_gpus: 1`,
`usage_limit_fraction: 0.95`, no staging arena). Do not copy 0.95 onto this 4-CN path with a
32 GiB out-of-pool arena.

---

## 4. What has been measured — do not re-derive this

Quoted from the RTX / GB300 / A100 campaigns and from this box's own smokes. GPU-busy on GB300
was 91–97 % of wall: thread-pool knobs were inert. Measure GPU-busy first; if it is that high
here, skip `pipeline.num_threads` / `scan_manager.num_threads`.

**MEASURED CATASTROPHIC — leave alone**

- `enable_dynamic_filter_pushdown: false` **livelocks** at high occupancy.
- `enable_prefetch_cache: true` is a **2.1×** regression.
- `local.use_odirect: true` is a **12.5×** regression on NFS. Data on `/raid` (local NVMe) is
  the opposite case — odirect is fine there. `$HOME` is NFS; do not put TPC-H there.

**MEASURED, BUT NOT A YAML KEY**

- `SET expression_evaluator_strategy = 'ast_jit'` — **−4.17 %** suite on GB300. Default is the
  slower interpreter. Warm `$HOME/.cudf/$VERSION/$ARCH` before timing (cold NVRTC ≈ 19 s).

**THIS BOX, already run**

- SF100 4-CN unpinned baseline: q05 / q09 wedge at 180 s; q08 refused at **60758 ms** (the 60 s
  peer RPC bound). That bound is now `SIRIUS_CN_RPC_TIMEOUT_SECS` (range 1–3600), not a
  recompile. The SCALE_FACTOR presets raise it.
- SF1000 4-CN smoke (2026-08-27, `128/32/120`, this host): Q6 and a GROUP BY on full
  `lineitem/*.parquet` **matched DuckDB** (`61635169685.0692`). That is not a 22-query pass.

**HONEST LIMIT** (`engine-a.env`): you cannot buy a full SF1000 suite with memory knobs.
Projected q21 peaks 100–150 GB/CN; arena demand under bump-reset was 105–125 GB/CN. Trailing
reclamation makes 32 GiB comfortable in the common case. It is not a proof q05 / q08 / q09 /
q21 pass.

---

## 5. Staging arena

`SIRIUS_EXCHANGE_STAGING_BYTES` is an **environment variable, not a YAML key**, and it has
**no engine default**. Unset means no arena: the CN boots, registers, answers local queries,
and every remote exchange fails. `cluster4-numa.sh` always sets it from `STAGING`.

Exhaustion is self-naming:

```
exchange staging arena exhausted: requested … bytes, … free of … capacity with N leases outstanding
(raise SIRIUS_EXCHANGE_STAGING_BYTES)
```

Raise `STAGING`, lower `GPU_MEM` by the same amount. `OOM at operator HASH_JOIN` after 100
retries is **not** the same bug — more pool did nothing for RTX q09; the lever is more CNs
(you already have 4) or operator budgets / engine work.

---

## 6. A tuning procedure that will not waste your time

1. **Size memory first (§2).** `SCALE_FACTOR=<N>` then confirm with
   `nvidia-smi --query-compute-apps=used_memory`. Four distinct `gpu_uuid`s.
2. **Read the two self-naming errors.** Arena exhausted → raise `STAGING`. HASH_JOIN OOM after
   100 retries → not automatically "more `GPU_MEM`".
3. **Measure GPU-busy.** If >90 % of wall, skip thread-pool knobs.
4. **Warm NVRTC** and `SET expression_evaluator_strategy = 'ast_jit'` before timing.
5. **Only then** consider `scan_task_batch_size` (~6 GB), and only via path B.
6. **One change per arm. Oracle every quoted query against DuckDB.** `bench.sh` has no
   correctness gate. `run-abc.sh --q11-fraction spec` rewrites q11's `0.0001` to `0.0001/SF`
   (required at SF≠1).

---

## 7. Worked example — this box, SF1000, 4 CNs

Usable 184 GiB. Dataset `/raid/prestouser/aocsa/tpch_parquet_sf1000_f64` (380 G, f64 decimals,
60 `lineitem` parts).

### Launch (own terminal; the EXIT trap tears the cluster down)

```bash
source /raid/prestouser/sirius-build/env.sh
cd /home/prestouser/aocsa/sirius/experimental/starrocks
unset CUDA_VISIBLE_DEVICES

SCALE_FACTOR=1000 ./configs/gb200-4gpu/cluster4-numa.sh 2>&1 | tee /tmp/cluster-sf1000.log
```

Equivalent explicit override (same numbers the preset selects):

```bash
GPU_MEM=128GiB STAGING=32GiB HOST_MEM=112GiB \
SIRIUS_QUERY_WATCHDOG_SECS=300 SIRIUS_CN_RPC_TIMEOUT_SECS=300 \
  ./configs/gb200-4gpu/cluster4-numa.sh
```

Wait until column 9 of `SHOW COMPUTE NODES` is `true` for **exactly 4** rows
(`awk -F'\t' '$9=="true"'` — `grep -c true` overcounts). Confirm four distinct GPU UUIDs.
`UCX_TLS` must stay `cuda_copy,cuda_ipc,tcp,self`. Do not add `ib`/`rc`.

### Sweep

`run-abc.sh --sf` already exports `SCALE_FACTOR`, searches `tpch_parquet_sf<N>_f64`, and
scales client timeouts (`max(90, 1.8×SF)` warm / `max(300, 6×SF)` cold).

```bash
cd /home/prestouser/aocsa/sirius/experimental/starrocks
./benchmarks/tpch/run-abc.sh --sf 1000 --engines A \
  --data /raid/prestouser/aocsa/tpch_parquet_sf1000_f64
```

Or the lower-level harness (`QUERY_TIMEOUT` default is **30 s** — unusable at SF1000):

```bash
SR=/home/prestouser/aocsa/sirius/experimental/starrocks
export PATH=$SR/.pixi/envs/default/bin:$PATH
TPCH_DATA=/raid/prestouser/aocsa/tpch_parquet_sf1000_f64 \
QUERY_TIMEOUT=1800 COLD_TIMEOUT=6000 MIN_BACKENDS=4 \
  $SR/benchmarks/tpch/bench.sh /tmp/bench/A-sf1000/timings.csv 3
```

Oracle must be a **plain** DuckDB (`/raid/prestouser/sirius-build/oracle`).
`$REPO/build/release/duckdb` auto-loads Sirius and fights the CNs for HBM.

Teardown: kill CN/FE **by PID**. `pkill -f` matches the launcher script. Then
`nvidia-smi --query-compute-apps=pid` must be empty (idle `memory.used` is 30–33 MiB, never 0).

---

## 8. Switching SF500 / SF3000 / SF10000 — no rebuild

The stack is already built. Parquet is `FILES()` at query time. Memory, staging, watchdog, and
peer RPC timeout are env vars resolved when the CN process starts.

```bash
# tear the previous cluster down first (ports 9030 / 9100–9134 must be free)
SCALE_FACTOR=500   ./configs/gb200-4gpu/cluster4-numa.sh
SCALE_FACTOR=3000  ./configs/gb200-4gpu/cluster4-numa.sh
SCALE_FACTOR=10000 ./configs/gb200-4gpu/cluster4-numa.sh
```

Or one shot:

```bash
./benchmarks/tpch/run-abc.sh --sf 500  --engines A
./benchmarks/tpch/run-abc.sh --sf 3000 --engines A \
  --data /raid/prestouser/aocsa/tpch_parquet_sf3000_f64
```

`--sf` looks under `$TPCH_DATA_ROOTS` for `tpch_parquet_sf<N>` then `tpch_parquet_sf<N>_f64`.
On this box SF500 matches the unsuffixed tree; SF1000/3000/10000 need the `_f64` name (or
`--data`).

Override any one knob without touching the others:

```bash
SCALE_FACTOR=1000 STAGING=40GiB GPU_MEM=116GiB ./configs/gb200-4gpu/cluster4-numa.sh
```

`bench.sh` / `run-abc.sh` do not compile. A config change that needs a **rebuild** is a code
change (new operator, proto, libsirius). `SIRIUS_CN_RPC_TIMEOUT_SECS` used to be that; it is
now an env var (1–3600 s, fail-closed at bring-up).

---

## 9. Is this project ready for those scale factors?

**Ready to relaunch at another SF without compiling: yes.**

The CN binary, FE, nixl, and libsirius are scale-agnostic. Datasets for 500 / 1000 / 3000 /
10000 already sit on `/raid`. `SCALE_FACTOR` plus `run-abc.sh --sf` is the runtime switch.

**Ready to finish the 22-query suite: no, not at every SF.**

| SF | Launch | Dataset | Full suite |
|---|---|---|---|
| 100 | yes (validated topology) | yes | **no** — q05/q09 wedge, q08 historically 60 s RPC |
| 500 | yes (preset is a starting split) | yes (132 G) | **unproven here**; A100 closed SF500 at 8 and 4 CNs with a 24 GiB arena at 4 CNs |
| 1000 | yes | yes (380 G f64) | **smoke only** (Q6 + GROUP BY). Engine caveat: q21 / arena not bought with 128/32 |
| 3000 | yes (will boot) | yes (1.2 T) | **not expected to close**. Dataset > LPDDR; shuffle working set will not fit 184 GiB HBM if it scales with SF |
| 10000 | yes (will boot) | yes (3.8 T) | **probe queries only**. Same wall, 10× the data of SF1000 |

What is still engine work, not config:

- q05 / q09 wedges (watchdog converts them into a clean error so they stop poisoning the sweep;
  it does not fix them).
- HASH_JOIN OOM on the widest builds after 100 retries — more HBM per CN did not fix this on RTX.
- Arena occupancy under a long-lived low-offset lease (trailing reclaim helps; eager per-lease
  reclaim is still the SF1000+ prerequisite in `engine-a.env`).
- Path B operator YAML is not wired into `cluster4-numa.sh`.
- `bench.sh` still has no correctness gate; `run-abc.sh` compares row counts across engines but
  you still need a DuckDB oracle for values.

What **is** ready: bring-up, NUMA pinning, fail-closed transport knobs, dataset layout, and
changing SF by restarting processes.
