# Tuning Sirius + StarRocks for a given box

Self-contained procedure for choosing the Sirius engine and StarRocks CN configuration on a new
machine — how the knobs work, which ones have measurement behind them, and how to size the rest
from GPU HBM and TPC-H scale factor.

Companion to `TPCH-SWEEP-RUNBOOK.md` (how to run the benchmark) and `BUILD-SIRIUS-STARROCKS.md`
(how to build the stack). Reference config: `bench/sf1000-repro/sirius-sf1000.yaml`.

---

## 0. The two config paths are mutually exclusive

This is the first thing to understand, and it constrains everything else.

| Path | How | Reaches |
|---|---|---|
| **A — derived** | `cluster8.sh` passes `--gpu-memory-limit` / `--host-memory-limit`; the CN *generates* `.cn<i>/derived-sirius-config.yaml` | memory limits, CPU affinity, telemetry dir — **nothing else** |
| **B — full file** | `--sirius-config <path>` (or `$SIRIUS_CONFIG_FILE`, `./sirius.yaml`, `~/.sirius/sirius.yaml`) | everything |

**`--sirius-config` conflicts with every memory flag** — "a full config already decides memory".
Since `cluster8.sh` always passes `--gpu-memory-limit` and `--host-memory-limit`, **you cannot use
a full config with the stock launcher.** Choosing path B means editing the launcher to drop those
two flags and putting the memory settings in the YAML instead.

Here is exactly what path A produces (measured, 2 GPUs, `GPU_MEM=40GiB HOST_MEM=128GiB`):

```yaml
sirius:
  topology: {num_gpus: 1}
  memory:
    gpu:  {usage_limit_bytes: "40GiB", reservation_limit_fraction: 1.0}
    host: {capacity_bytes: "128GiB"}
  executor:
    scan_manager:  {cpu_affinity: [0..47]}
    task_creator:  {cpu_affinity: [0..47]}
    downgrade:     {cpu_affinity: [0..47]}
  telemetry: {output_directory: ".cn0/telemetry"}
```

Every `operator_params` key in `sirius-sf1000.yaml` — `scan_task_batch_size`,
`hash_partition_bytes`, `concat_batch_bytes`, `max_build_hash_table_bytes` — is absent, so it sits
at its built-in default. **Path A cannot tune the operator layer at all.**

`cpu_affinity` is derived per GPU: the CN resolves its device ordinal, finds the PCI BDF, reads
that device's NUMA node and `local_cpulist`, and writes it into the three thread-pool affinities.
On a 1-NUMA box that is all cores, which is why the block above is `0..47`.

---

## 1. Units — read this before writing any number

Byte-valued keys use Kubernetes/systemd conventions:

- `K/KB/M/MB/G/GB/T/TB` are powers of **1000**
- `Ki/KiB/Mi/MiB/Gi/GiB/Ti/TiB` are powers of **1024**

So `8GB` in the YAML is 8,000,000,000 bytes, **not** 8 GiB. Fractional values (`1.5Gi`) are
allowed. The committed configs mix both conventions — `scan_task_batch_size: 8GB` alongside
`GPU_MEM=140GiB` — so do not assume.

---

## 2. The device budget, and the one hard interlock

```
device occupancy = usage_limit_bytes  +  SIRIUS_EXCHANGE_STAGING_BYTES  +  ~2 GiB
                   (the RMM pool)        (bare cudaMalloc, OUTSIDE the pool)   (CUDA ctx, cudf, frag)
```

**The staging arena is outside the RMM pool and `usage_limit_fraction` knows nothing about it.**
Every GiB of arena costs a GiB of pool. This is the interlock that breaks naive configs: raising
`STAGING` without lowering `GPU_MEM` by the same amount overruns the card.

`usage_limit_fraction` multiplies **total** device memory, never free memory.
`usage_limit_bytes` and `usage_limit_fraction` are mutually exclusive, and **bytes wins** if both
are present.

### Sizing recipe

1. `STAGING(N, SF) ≈ 96 GiB × (SF/500) / N`, rounded up to the next 4 GiB — **a floor, not a
   prediction** (§5).
2. `GPU_MEM = CARD − STAGING − 2 GiB`.
3. Check the total lands at **85–97 % of CARD**. Every validated config does:

| Machine | HBM/GPU | N | GPU_MEM | STAGING | HOST_MEM | % HBM |
|---|---|---|---|---|---|---|
| A100x8 8-CN (validated SF500) | 80.0 GiB | 8 | 66 GiB | 12 GiB | 100 GiB | 97.5 % |
| A100x8 4-CN (validated SF500) | 80.0 GiB | 4 | 54 GiB | 24 GiB | 200 GiB | 97.5 % |
| GB200 4-GPU (validated SF100) | 184.0 GiB | 4 | 140 GiB | 16 GiB | 160 GiB | 84.8 % |
| GB200 4-GPU (SF1000) | 184.0 GiB | 4 | 128 GiB | 32 GiB | 112 GiB | 84.9 % |
| GB300 standalone (SF1000) | 256 GB | 1 | `fraction 0.95` | no arena | 471.2 GB | 95 % |

`usage_limit_fraction` is the one knob whose committed value moves monotonically **down** as HBM
shrinks — 0.95 (256 GB) → 0.86 (185 GiB) → 0.85 (80 GiB) — precisely because the fixed
out-of-pool arena is a growing fraction of a smaller card. The a100x8 config says so in capitals:
*"DO NOT COPY 0.95 FROM THE GB300 CONFIG, OR 0.86 FROM THE GB200 ONE."*

### HOST_MEM is per NUMA region

`memory.host.capacity_bytes` is per NUMA region, **not** machine-wide: one host memory space per
distinct NUMA node of the configured GPUs. On a 1-NUMA box, `N` CNs each take their full
`HOST_MEM` from the same RAM. It is routinely over-provisioned — a measured high-water of 631 MiB
against a 200 GiB ceiling — so prefer leaving RAM for page cache over the parquet files.

Host pool granule: `block_size × pool_size × initial_number_pools` is the **initial pinned
footprint**. All three committed configs use `block_size: 67108864` (64 MiB) and `pool_size: 8`
(a 512 MiB granule) and scale only `initial_number_pools`, holding the same ratio: initial
allocation ≈ **44.6–44.7 % of `capacity_bytes`** on all three machines. Code defaults are much
smaller (1 MiB / 128 / 4).

---

## 3. The operator knobs, and what is actually known about them

All default to a single runtime-derived value:

```
derived_default_batch_size() = clamp(min_visible_GPU_total_bytes / 40, 512 MiB, 5 GiB)
```

2.5 % of the **smallest visible GPU's total memory**, memoized once per process, CUDA_VISIBLE_DEVICES-aware,
falling back to 800 MiB when no GPU is visible.

**This ignores the CN's carve-out entirely.** With `N` CNs sharing a card, each derives its batch
size from the *whole* card, not from its `1/N` slice — so a multi-CN-per-GPU layout over-sizes
batches by a factor of `N`. Worth checking whenever CNs share a device.

| Key | Default | Units | Controls |
|---|---|---|---|
| `scan_task_batch_size` | derived (2.5 %) | bytes | target coalesced scan batch; forwarded as `approximate_batch_size` into the parquet/duckdb ingestibles |
| `hash_partition_bytes` | derived | bytes | partition count `ceil(total/this)`, floored to `num_gpus` past the small-table threshold. **Zero is rejected** |
| `concat_batch_bytes` | derived | bytes | CONCAT emits when `!concat_all && total > this` |
| `max_build_hash_table_bytes` | 2 × derived | bytes | BUILD_PROBE gate |
| `sort_sample_bytes` | derived | bytes | input sampled before sort boundaries. Not set in any committed config |
| `max_sort_partition_bytes` | 0 = auto | bytes | when 0, budget = **available** (not total) GPU memory × fraction |
| `max_sort_partition_memory_fraction` | 0.33 | fraction | range-checked [0,1] |
| `max_broadcast_join_size` | 256 MiB | bytes | broadcast eligibility |

Ratios implied by the committed configs, as % of total HBM:

| Knob | GB300 256 GB | GB200 198.7 GB | A100 85.9 GB | Spread |
|---|---|---|---|---|
| `scan_task_batch_size` | 8 GB = 3.12 % | 6 GB = 3.02 % | 3 GB = 3.49 % | **tight, ~3.0–3.5 %** |
| `concat_batch_bytes` | 5 GB = 1.95 % | 5 GB = 2.52 % | 2 GB = 2.33 % | ~2–2.5 % (0.63–0.83× scan) |
| `hash_partition_bytes` = `max_build_hash_table_bytes` | 32 GB = 12.5 % | 32 GB = 16.1 % | 8 GB = 9.3 % | **does not scale linearly** |

`scan_task_batch_size` is the only knob whose committed values cluster tightly enough to
extrapolate from HBM. The hash knobs were cut 4× for a 3× smaller card — do not interpolate them.

### Note the derivation ceiling

`derived_default_batch_size()` **clamps at 5 GiB**. On the GB300 the formula wants 6.4 GiB and
gets 5 GiB; the measured optimum is `8GB`. **The auto-derivation cannot reach the measured
optimum on any card above ~200 GB** — that is exactly why the sf1000 config sets it by hand, and
the single strongest argument for using config path B on large cards.

---

## 4. What has been measured — do not re-derive this

Quoted from the committed configs. Twelve knobs were swept on GB300/SF1000; **only one mattered.**

**MEASURED TO MATTER**

> `scan_task_batch_size` is the ONE knob that mattered: 5GB → 8GB is **−1.85 %** (q4 −25 %,
> q12 −18 %), and it is a **STEP not a gradient** — 10GB adds nothing and costs q9 2.6 %.
> 8GB peaks at 253.9 GB of 256 GB, so do not raise it further.

**MEASURED INERT — do not bother retuning**

| Knob | Result |
|---|---|
| `pipeline.num_threads` 6 / 8 / 12 | all within 0.55 % |
| `hash_partition_bytes` 16GB vs 32GB | −0.06 % |
| `scan_manager.num_threads` 18 vs 24 | −0.33 % |
| `scan_task_batch_size` 2GB | +0.10 % |
| `mark_join_build_switch_ratio` 3.0 / 0 | −0.03 % / −0.28 % — never fires |
| `dynamic_filter_keep_threshold` 0.5 / 0.2 | −0.11 % / +0.69 % — gate never binds |
| `max_broadcast_join_size` 8GB | −0.24 % |
| `enable_dynamic_zone_map_filter` true | +0.06 % — scattered keys prune nothing |

The reason is structural: **GPU-busy was 91–97 % of wall**, so scheduling and parallelism knobs
cannot help. Measure GPU-busy first; if it is that high on your box, skip the whole thread-pool
family.

**MEASURED CATASTROPHIC — leave alone**

- `enable_dynamic_filter_pushdown: false` **livelocks**. At 251 GB of a 256 GB card it is
  load-bearing for *memory feasibility*, not speed: without it more rows survive the scan,
  intermediates grow, the executor cannot get a reservation, and it spins on
  `reschedule (retry 1/100)` forever with the GPU at 0 %.
- `enable_prefetch_cache: true` is a **2.1× regression** (confirmed by two independent sources).
- `local.use_odirect: true` is a **12.5× regression on NFS** (fine on local NVMe). **If the data
  is on network storage, set it false.**

**MEASURED, BUT NOT A YAML KEY**

- `SET expression_evaluator_strategy = 'ast_jit'` — **−4.17 % suite for zero code**; the shipped
  default is the slower `AST_INTERPRET`. The NVRTC cache at `$HOME/.cudf/$VERSION/$ARCH` persists
  across restarts; a cold cache costs ~19 s in run 0 (28.99 s first iteration → 10.50 s warm).
  **Warm the cache before timing anything.**

**KNOWN BUG-SHAPED INTERLOCK**

The runtime scan-task coalescer ignores the configured value and uses the compile-time constant:
`accumulated_bytes >= config::DEFAULT_SCAN_TASK_BATCH_SIZE` (800 MiB) with
`max_batches_per_task = 32`. So raising `scan_task_batch_size` does not move *that* path. Verify
against your build before attributing a null result to the knob.

**Non-obvious defaults worth knowing**

- All three configs raise `downgrade_trigger_fraction` 0.8 → **0.9** and `downgrade_stop_fraction`
  0.6 → **0.85**, labelled a measured delta.
- All three enable `memory_prefetcher` (`enable: true, num_threads: 3`); the code default is
  **false**.
- `scan_manager.num_threads` is validated `> 2` — the minimum accepted is **3**, not 1.
- Disk spilling is off unless `downgrade_root_dirs` is non-empty **and** `capacity_bytes != 0`;
  otherwise the configurator silently registers no disk mount.
- `telemetry.enable_quent` defaults **true** and all three configs set it false — it costs time.

---

## 5. The staging arena — the knob that decides whether queries run at all

> **Canonical env-var overview:** [`experimental/starrocks/docs/TUNABLES.md`](../../experimental/starrocks/docs/TUNABLES.md).
> Transport knobs are validated at startup; a bad value refuses to boot rather than
> surfacing as an unexplained timeout mid-sweep.

`SIRIUS_EXCHANGE_STAGING_BYTES` is an **environment variable, not a YAML key**, and it has **no
engine default**. Unset means *no arena*: the CN boots healthy, registers with the FE, answers
local queries, and every remote exchange destination fails. The "default" you see quoted is a
launcher value and differs per script — 8 GiB in `cluster8.sh`, 16 GiB in `script-box.sh`, 2 GiB
in `nixl-echo-2node.sh`.

It must hold the **sum of concurrent leases**, not the largest one; 9–82 simultaneous leases have
been observed at 97.7–99.4 % occupancy. It never degrades gracefully — exhaustion is a hard,
self-naming failure that tells you the fix:

```
exchange staging arena exhausted: requested 1242515456 bytes (1242515456 aligned),
778297088 free of 17179869184 capacity with 14 leases outstanding
(raise SIRIUS_EXCHANGE_STAGING_BYTES)
```

Scaling: **inversely with CN count, directly with scale factor.**

```
arena(N, SF) ≈ 96 GiB × (SF/500) / N,  rounded up to the next 4 GiB
```

Measured on 8× A100 at SF500: 8 CN needs 12 GiB (fails at 8); 4 CN needs 24 GiB (fails at 16);
2 CN has no working split, because 48 GiB of arena plus a usable pool does not fit 80 GiB.

**The formula is a floor and it has been observed to underestimate** — see §7.

---

## 6. A tuning procedure that will not waste your time

1. **Size memory first (§2).** Nothing else matters if queries OOM. Confirm
   `GPU_MEM + STAGING` lands at 85–97 % of the card, and confirm with
   `nvidia-smi --query-compute-apps=used_memory`.
2. **Run the sweep and read the two self-naming errors.** `exchange staging arena exhausted` →
   raise `STAGING`. `exceeded 100 retries … OOM at operator HASH_JOIN` → raise `GPU_MEM`. Both
   name the fix; do not guess.
3. **Measure GPU-busy.** If it is >90 % of wall, every thread-pool knob is inert (§4) — stop.
4. **Warm the NVRTC cache** and set `expression_evaluator_strategy='ast_jit'` before timing.
5. **Only then** consider `scan_task_batch_size`, and only via config path B (§0) — remembering it
   is a step function, not a gradient, and the auto-derivation clamps at 5 GiB (§3).
6. **Change one thing per arm, and re-verify correctness against the DuckDB oracle each time.**
   The harness has no correctness gate; a config change that makes a query fast and wrong is
   recorded as a win.

Discipline that the committed configs model well: record *inert* results too. Half the value of
`sirius-sf1000.yaml` is its list of twelve knobs that did nothing.

---

## 7. Worked example — 2× RTX PRO 6000 Blackwell (95.6 GiB), SF100, 2 CNs

Measured 2026-08-19. Card reports 97887 MiB = 95.6 GiB.

### Arm A — the inherited config

```
NUM_CNS=2 GPU_MEM=40GiB STAGING=16GiB HOST_MEM=128GiB SIRIUS_QUERY_WATCHDOG_SECS=90
```

Device occupancy 40 + 16 = 56 GiB = **58.6 % of the card** — far below the 85–97 % band every
validated config occupies. This config was frozen from a **4-CN** topology; at 2 CNs each CN
carries twice the fan-out.

Result: **19/22 recorded pass**. Two failures, both naming their own cause:

- **q09** — `OOM at operator HASH_JOIN (index 0)`, 100 retries then terminated.
- **q08** — `arena exhausted: … 14 leases outstanding`.

Applying §2 to this box: `STAGING(2, 100) = 96 × 0.2 / 2 = 9.6 → 12 GiB` by formula, yet 16 GiB
was **measured exhausted**. This is the formula's floor behaviour — trust the measurement.

### Arm B — sized by the rule

```
NUM_CNS=2 GPU_MEM=60GiB STAGING=32GiB HOST_MEM=128GiB SIRIUS_QUERY_WATCHDOG_SECS=90
```

Device occupancy 92 GiB; **measured 94788 MiB of 97887 = 96.8 % of card**, 3 GiB margin — in the
validated band. Predicted overhead ~0.6 GiB, matching the measurement almost exactly, so the
`+2 GiB` term in §2 is conservative on this card.

**Outcome — the arm separated two failures that looked alike:**

| Query | Arm A (40+16) | Arm B (60+32) | Verdict |
|---|---|---|---|
| q08 | `arena exhausted, 14 leases outstanding` | `no parked sender output to export for SenderSlot` | **was sizing** — the bigger arena removed that defect and exposed an exchange head-of-line deadlock beneath it |
| q09 | `100 retries → OOM at HASH_JOIN` | `100 retries → OOM at HASH_JOIN`, both CNs | **not sizing** — a 50 % larger pool changed nothing |
| q15 | cold wedge, 1/3 warm pass | cold pass, 0/3 warm pass | flaky either way, not config-sensitive |
| pass count | 19/22 | 18/22 | the delta is only q15's flake |
| 18 common queries | 24195 ms | **24105 ms (−0.4 %)** | **50 % more pool bought no speed** |

Correctness was **bit-identical across both arms** — the same six queries carry the same
decimal drift to the same digits. That rules out memory pressure and the downgrade path as its
cause, and means a memory-sizing arm can be evaluated on pass/fail alone.

Two lessons worth carrying into any tuning study:

1. **An undersized resource masks other defects.** q08's real bug was invisible until the arena
   was big enough to stop failing first. Size memory correctly *before* triaging anything else —
   which is why §6 puts it at step 1.
2. **`OOM at operator HASH_JOIN` does not imply "raise `GPU_MEM`".** q09 is the widest join in
   TPC-H; at 2 CNs the per-node build side exceeds what 60 GiB holds, and the lever is **more CNs**
   (shrinking each node's build) or engine-side work — not more HBM per CN. Adding memory to a
   partitioning problem does nothing.

At SF100 the timings being flat across a 50 % larger pool also says these queries are not
memory-bound here: the extra HBM buys failure-avoidance on the big shuffles, not throughput.

### What this box implies for the operator knobs

`derived_default_batch_size()` = `clamp(95.6 GiB / 40, 512 MiB, 5 GiB)` = **2.39 GiB**, comfortably
inside the clamp, so unlike the GB300 this card's auto-derivation is *not* truncated. Against the
§3 target band of 3.0–3.5 % of HBM, 2.39 GiB is **2.5 %** — slightly low but within reach of the
committed spread, so `scan_task_batch_size` is a plausible (untested here) first arm at ~3 GB via
config path B.

**Nothing in the operator layer was tuned on this box.** Arms A and B differ only in memory
sizing. Treat the §3 ratios as the starting point for the next study, not as validated values here.
