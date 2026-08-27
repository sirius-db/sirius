# Re-targeting this folder to different hardware

This folder is a **template**. Everything hardware-specific is isolated so that moving to another
box means editing a known, bounded set of values rather than re-deriving the benchmark.

## The rule

| File | Hardware-dependent? | What to do |
|---|---|---|
| [`HARDWARE.md`](HARDWARE.md) | **Entirely** | **Replace.** Re-probe the new box and rewrite |
| [`engine-a-sirius.yaml`](engine-a-sirius.yaml) | **Heavily** | Re-derive — see the table below |
| [`engine-a-sirius.env`](engine-a-sirius.env) | **Heavily** | CN count, GPU/CPU/NUMA mapping, memory |
| [`engine-b-starrocks.conf`](engine-b-starrocks.conf) | **Heavily** | BE count, `num_cores`, `mem_limit`, storage paths |
| [`engine-c-cudf-polars.env`](engine-c-cudf-polars.env) | Moderately | `--num-gpus`, NUMA policy, thread counts |
| [`PLAN.md`](PLAN.md) | Lightly | Campaign plans now live under `notes/` (dated). Keep a short local plan only if this box is still being designed |
| Query set | **No** | All 22 TPC-H queries. Deviations and workarounds: [`../../experimental/starrocks/benchmarks/tpch/QUERY-DEVIATIONS.md`](../../experimental/starrocks/benchmarks/tpch/QUERY-DEVIATIONS.md). Per-box pass/fail: `STATUS.md` / `TPCH-STATUS.md` on that box — never reuse a dropped-query list from another campaign |
| `RETARGETING.md` | No | This file |

---

## Step 1 — probe the new box

```bash
hostname
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m
numactl -H
lscpu | grep -E '^CPU\(s\)|Model name|Architecture|NUMA node[0-9]+ CPU'
df -h /raid                       # or wherever the data lives
findmnt -T <data path>            # CONFIRM local, not NFS — this changes the I/O config
free -g; cat /proc/meminfo | head -3
swapon --show                     # Swap:0 makes percentage memory limits fatal
```

Rewrite `HARDWARE.md` from that output. **Do not skip `nvidia-smi topo -m`** — the GPU↔GPU link
type and the GPU→CPU-affinity mapping determine the entire CN placement.

## Step 2 — recompute the derived quantities

These are the numbers everything else depends on.

| Quantity | Formula | This box |
|---|---|---|
| GPU pool fraction | `(HBM − staging_arena − ~1 GiB) / HBM` | `(185.03 − 16 − 1) / 185.03` → **0.86** |
| Host capacity per CN | `total_RAM / CN_count`, leaving ≥ 1.5× dataset as page cache | 160–240 GiB |
| CPU range per CN | Split each GPU's `CPU Affinity` range evenly among the CNs on that socket | 36 or 72 cores |
| `scan_task_batch_size` | Scale from the reference by HBM ratio, then sweep ±1 step | 6 GB |
| Pinning feasible? | `dataset_bytes / GPU_count < HBM − working_set` | See `HARDWARE.md` |

## Step 3 — the values that change, by knob

### Engine A — Sirius

| Knob | Depends on | How to derive |
|---|---|---|
| `topology.num_gpus` | — | Always `1` per CN |
| `gpu.usage_limit_fraction` | **HBM size**, staging arena | Formula above. The arena is **out-of-pool** — `usage_limit_fraction` does not know about it. On a small card shrink the arena first |
| `SIRIUS_EXCHANGE_STAGING_BYTES` | **HBM size** | Keep ≤ ~10% of HBM. 16 GiB is 8.6% of a GB200 but **20% of an 80 GB card** |
| `host.capacity_bytes` | **RAM**, CN count, dataset size | Leave ≥ 1.5× dataset as page cache |
| `host.initial_number_pools` | `capacity_bytes` | Hold the reference ratio: `pools × pool_size × block_size ≈ 0.45 × capacity` |
| `disk.downgrade_root_dirs` | **Storage layout** | Local NVMe, **never NFS**, one subdir per CN |
| `scan_manager.num_threads` | **Cores per CN** | Do not exceed the CN's `physcpubind` width |
| `uring_n_reactors` | **Storage media** | Sweep {4, 8, 32}. NVMe and NFS have different optima |
| `local.use_odirect` | **Storage media** | `true` is fine on local NVMe. **On NFS it is a 12.5× regression — must be `false`** |
| `scan_task_batch_size` | **HBM size**, scale factor | The one knob that matters. A *step*, not a gradient — sweep, don't interpolate |
| `hash_partition_bytes`, `max_build_hash_table_bytes` | **HBM size** | 32 GB on a 185 GiB card; 8–16 GB on an 80 GB card |
| `pipeline.num_threads` | — | Measured inert (6/8/12 within 0.55%). Leave at 8 |
| `enable_prefetch_cache` | — | **Always `false`.** `true` is a 2.1× regression, confirmed twice |
| `expression_evaluator_strategy` | — | **Always `ast_jit`.** −4.17% for zero code |

### Engine B — StarRocks

| Knob | Depends on | How to derive |
|---|---|---|
| BE count | **CPU NUMA nodes** | One BE per CPU NUMA domain. Check `numactl -H`; do not assume 2 |
| `num_cores` | **Core count**, BE count | `cores / BE_count`, set **explicitly** — `CpuInfo` never calls `sched_getaffinity`, so a pinned BE otherwise reports all cores |
| `mem_limit` | **RAM** | **Absolute bytes, never a percentage.** On a GPU box `/proc/meminfo` counts HBM, so `"90%"` resolves against the wrong total. With `Swap: 0` that is an OOM-kill |
| `storage_root_path` | **Storage** | Local NVMe. Never NFS |
| numactl wrapper | **NUMA layout** | `--membind=<node>` per BE. Never `--interleave=all` on a box with GPU HBM NUMA nodes |

### Engine C — cudf-polars

| Knob | Depends on | How to derive |
|---|---|---|
| `--num-gpus` | **GPU count** | Mutually exclusive with `CUDA_VISIBLE_DEVICES` |
| NUMA policy | **NUMA layout** | Prefer per-worker `bind_to_gpu(hardware_binding)`. Otherwise `--interleave=0,1` — **never `all`** on a box with GPU HBM nodes |
| `KVIKIO_NTHREADS`, `RAPIDSMPF_NUM_STREAMING_THREADS` | **Core count** | Scale with cores per worker |
| `min_device_size`, `target_partition_size` | **HBM size** | Scale with per-GPU memory |
| cwd | — | **Never a directory containing a `duckdb/` child** — it shadows the `duckdb` package as an empty namespace package and kills the run in 2 s |

---

## Step 4 — the checks that catch a bad re-target

Run these before trusting any number on new hardware.

```bash
# 1. Memory arithmetic: pool + staging must fit in HBM
#    (usage_limit_fraction × HBM) + staging_arena + 1 GiB  <  HBM

# 2. CN placement matches GPU affinity — a CN must never own a cross-socket GPU
nvidia-smi topo -m | grep -A1 'CPU Affinity'

# 3. NUMA binding actually took effect (Mems_allowed_list is the WRONG probe)
grep -m1 mempolicy /proc/<cn_pid>/numa_maps          # expect bind:<node>
grep -o 'N[0-9]*=' /proc/<cn_pid>/numa_maps | sort -u # expect a single node

# 4. No host pages landed in GPU HBM nodes
#    (GPU HBM node IDs come from the 'GPU NUMA ID' column of nvidia-smi topo -m)

# 5. Data is where you think, and on the media you think
findmnt -T <data path>

# 6. Box is idle and GPUs are at the clean floor
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

## Step 5 — re-validate correctness, always

**Never assume the query set transfers.** Different hardware means different memory pressure, which
means different failure points. Re-run the DuckDB oracle diff at relative tolerance `1e-12` on the
new box before publishing anything, and re-check which Tier 4 probes survive.

The `(1 − l_discount)` decimal defect is **hardware-independent** — it is a plan-translation bug, so
it will follow you to every box until `translate_arithmetic` is fixed.
