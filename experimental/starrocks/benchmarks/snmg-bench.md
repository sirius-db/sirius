# SNMG Benchmark: TPC-H SF200 on 2× H100 SXM5 (`repro/sf1000-8.5s`)

Single Node Multi-GPU (SNMG) run of all 22 TPC-H queries at SF200 (~200 GB raw)
using the `repro/sf1000-8.5s` branch on a dual-H100 SXM5 node.

> This document adapts [`repro-sf1000.md`](repro-sf1000.md) for a 2-GPU H100 node.
> Read that doc first for background on the branch changes and gotchas.

---

## Target machine

| Resource | This node | Notes |
|---|---|---|
| GPU | 2× H100 SXM5 80 GB | 160 GB total HBM3, sm_90 |
| GPU interconnect | NVLink 4.0 | 900 GB/s bidirectional P2P |
| GPU HBM bandwidth | 3.35 TB/s per GPU | 6.7 TB/s aggregate |
| Host RAM | 450 GB | ~94 GB pinned host pool for SF200 |
| Disk | ≥ 400 GB free | NVMe preferred |
| CUDA | 12.x+ | sm_90 requires CUDA ≥ 12.0 |
| OS | Linux (Ubuntu 22.04 / 24.04) | |

### Why SF200 fits comfortably on 2× 80 GB H100

SF1000 (1 TB raw) peaks at 251.7 GB of compressed GPU-resident data on one GB300.
SF200 is 1/5 the data: **~50 GB total compressed**, split ~25 GB per GPU.
Each H100 uses roughly 31% of its 80 GB HBM for pinned table data, leaving
55 GB of headroom for operator intermediates (hash tables, sort buffers, intermediates).
The downgrade executor will not fire under any of the 22 queries at this scale.

---

## How Sirius distributes work across 2 GPUs

Understanding the SNMG dispatch before you run helps interpret the logs:

| Layer | 2-GPU behavior |
|---|---|
| **Scan dispatch** | Round-robin: each scan batch is stamped with `preferred_device = gpu_id % 2`. Lineitem chunks alternate between GPU 0 and GPU 1. |
| **Partitioned operators** (hash join, grouped aggregate) | Partition count is clamped to `max(N, num_gpus)`. Each partition is pinned to `partition_idx % 2`. The hash table for partition 0 lives on GPU 0 and never moves. |
| **Cross-GPU data movement** | `cudaMemcpyPeerAsync` over NVLink (900 GB/s). If P2P is unavailable on a given pair, Sirius silently falls back to host-staged copy. |
| **Scheduling** | Two independent `gpu_pipeline_executor` instances. Device-pinned tasks go to their GPU; tasks with no preference go to whichever GPU signals `device_ready` first. |
| **No NCCL / no all-reduce** | All coordination is point-to-point cucascade batch conversion. |

At init, Sirius logs P2P status for every GPU pair:

```
[info] cudaDeviceCanAccessPeer(0 → 1): true  → peer access enabled
[info] cudaDeviceCanAccessPeer(1 → 0): true  → peer access enabled
```

If you do **not** see these lines, NVLink P2P is not being used and cross-GPU
transfers go through host memory at ~48 GB/s (PCIe). This would significantly
hurt multi-GPU performance.

---

## Step 1 — Check out the branch and build Sirius

```bash
git checkout repro/sf1000-8.5s
git submodule update --init --recursive

pixi run make          # full release build, all cores (~3–5 min first time)
```

If the build fails after a previous attempt:

```bash
pixi run make clean && pixi run make
```

---

## Step 2 — Build the patched libcudf

Three cuDF patches are required (same as SF1000). On H100, `CMAKE_CUDA_ARCHITECTURES=NATIVE`
produces `sm_90` instead of `sm_103a` — the build script handles this automatically.

```bash
pixi run bash bench/sf1000-repro/build-libcudf.sh
```

Build time: 25–50 min (H100 has more SMs than GB300 — PTX compilation takes longer).
Output: `$HOME/cudf-src/cpp/build/libcudf.so`

| Patch | Effect |
|---|---|
| `strings::like` backtrack skip | q13 −36.5% |
| `cuda_memcpy` 2 MiB threshold | q9 −5.8% |
| groupby shared-memory replication | q1 ~−5% |

---

## Step 3 — Generate the SF200 dataset

```bash
cd test/tpch_performance

pixi run bash generate_tpch_data.sh 200 \
  --output ~/tpch_parquet_sf200 --jobs 16

cd ../..
```

Output: `~/tpch_parquet_sf200/` (~200 GB raw, one subdirectory per table).
Generation time: ~5–10 min with 16 threads.

> **Do not use `CALL dbgen(sf=200)`** — DuckDB's Parquet writer produces 122K-row
> row groups, too small for GPU columnar scans. The `tpchgen-rs` path writes
> 2M–10M-row groups.

---

## Step 4 — Create the SF200 2-GPU config

Save the following as `bench/sf200-2gpu/sirius-sf200-2gpu.yaml`.
Values are scaled from the SF1000 single-GPU baseline; see the annotations.

```yaml
sirius:
  topology:
    num_gpus: 2                   # SNMG: both H100s

  memory:
    gpu:
      # Applied per-GPU. Each H100 has 80 GB; SF200 pins ~25 GB per GPU (~31%).
      usage_limit_fraction:       0.95   # 76 GB usable per GPU
      reservation_limit_fraction: 1.0    # pipelines may use all claimed memory
      downgrade_trigger_fraction: 0.9    # trigger at ~68 GB reserved per GPU
      downgrade_stop_fraction:    0.85   # stop at ~64 GB — won't fire at SF200

    host:
      # SF200 host pool = SF1000 (471 GB) × 0.2 ≈ 94 GB total
      capacity_bytes:       94240000000  # ~94 GB pinned (cudaMallocHost)
      initial_number_pools: 80           # 392 × 0.2, rounded up
      pool_size:            8            # keep: block_size × pool_size = 512 MiB/pool
      block_size:           67108864     # 64 MiB — eliminates per-segment H2D overhead
      # memory_prefetcher MUST be disabled when num_gpus > 1.
      # The engine forces this anyway and logs a warning; set it explicitly.
      memory_prefetcher:
        enable:      false
        num_threads: 3

    disk:
      disk_id: 0
      capacity_bytes: 200000000000      # 200 GB safety cap (spill disabled below)
      downgrade_root_dirs: ""           # leave empty — disk tier disabled

  executor:
    scan_manager:
      num_threads:           18
      # use_sirius_datasource is forced true by the engine on num_gpus > 1.
      # Set it explicitly to avoid the WARN log.
      use_sirius_datasource: true
      uring_n_reactors:      4
      enable_prefetch_cache: false
      memory_prefetcher:
        enable:      false              # same as above — must be false for multi-GPU
        num_threads: 3
    pipeline:
      num_threads: 8
    downgrade:
      num_threads: 1
    task_creator:
      num_threads: 4

  operator_params:
    scan_task_batch_size:        8GB   # ceiling; actual batches are smaller at SF200
    max_sort_partition_bytes:    0
    hash_partition_bytes:        16GB  # halved vs SF1000 — tables are 5× smaller
    concat_batch_bytes:          2GB
    max_build_hash_table_bytes:  16GB  # halved vs SF1000

  telemetry:
    enable_quent:     false
    output_directory: telemetry_data
    engine_name:      siriusDB
```

### Key differences from the SF1000 single-GPU config

| Parameter | SF1000 1-GPU | SF200 2-GPU | Reason |
|---|---|---|---|
| `num_gpus` | 1 | **2** | SNMG |
| `host.capacity_bytes` | 471 GB | **94 GB** | 1/5 data |
| `host.initial_number_pools` | 392 | **80** | 1/5 data |
| `memory_prefetcher.enable` | true | **false** | required for multi-GPU |
| `use_sirius_datasource` | true | **true** (explicit) | forced by engine; avoid WARN |
| `hash_partition_bytes` | 32 GB | **16 GB** | tables 5× smaller |
| `max_build_hash_table_bytes` | 32 GB | **16 GB** | tables 5× smaller |
| `concat_batch_bytes` | 5 GB | **2 GB** | right-sized for SF200 batches |

---

## Step 5 — Create the run script

Save as `bench/sf200-2gpu/run.sh`:

```bash
#!/usr/bin/env bash
# Reproduce TPC-H SF200 SNMG on 2× H100 SXM5
# Run from the repo root:  pixi run bash bench/sf200-2gpu/run.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

DATA="${DATA:-$HOME/tpch_parquet_sf200}"
CUDF_SO="${CUDF_SO:-$HOME/cudf-src/cpp/build/libcudf.so}"
PLANS="${PLANS:-$REPO/bench/sf1000-repro/plans}"   # schema-level; works at any SF
CFG="${CFG:-$HERE/sirius-sf200-2gpu.yaml}"
NAME="${NAME:-sf200_2gpu}"

[ -d "$DATA" ]    || { echo "ERROR: no SF200 parquet at $DATA (set DATA=)"; exit 1; }
[ -f "$CUDF_SO" ] || { echo "ERROR: no patched libcudf at $CUDF_SO -- run build-libcudf.sh first"; exit 1; }

export LD_PRELOAD="$CUDF_SO"

export SIRIUS_PRE_SQL="SET pin_table_compression = true; \
SET pin_table_input_compression_plan_dir = '$PLANS'; \
SET expression_evaluator_strategy = 'ast_jit'"

# Pin all 8 tables to GPU tier (compressed, distributed across 2 GPUs)
for t in LINEITEM ORDERS PART CUSTOMER SUPPLIER NATION REGION PARTSUPP; do
  export "SIRIUS_PIN_TIER_$t=gpu"
done

python3 "$REPO/test/tpch_performance/performance_test.py" \
  --input  "$DATA"  \
  --mode   grouped  \   # per-query pin — avoids union-pin filling GPU memory
  --iterations 3    \   # best-of-3 per query
  --engine gpu      \
  --pin    host     \   # stage disk→host first; SIRIUS_PIN_TIER_* overrides to gpu
  --queries 1-22    \
  --config "$CFG"   \
  --name   "$NAME"
```

Run it:

```bash
mkdir -p bench/sf200-2gpu
# (place sirius-sf200-2gpu.yaml and run.sh here first)
pixi run bash bench/sf200-2gpu/run.sh
```

Override paths without editing the script:

```bash
DATA=~/my_sf200 CUDF_SO=~/alt/libcudf.so \
  pixi run bash bench/sf200-2gpu/run.sh
```

---

## Step 6 — Validate results

Run with `--engine both --validation` before collecting timing data. This byte-compares
GPU output against DuckDB CPU for all 22 queries.

```bash
export LD_PRELOAD="$HOME/cudf-src/cpp/build/libcudf.so"
export SIRIUS_PRE_SQL="SET pin_table_compression = true; \
  SET pin_table_input_compression_plan_dir = 'bench/sf1000-repro/plans'; \
  SET expression_evaluator_strategy = 'ast_jit'"
for t in LINEITEM ORDERS PART CUSTOMER SUPPLIER NATION REGION PARTSUPP; do
  export "SIRIUS_PIN_TIER_$t=gpu"
done

pixi run python3 test/tpch_performance/performance_test.py \
  --input ~/tpch_parquet_sf200 \
  --mode grouped --iterations 1 \
  --engine both --validation \
  --queries 1-22 \
  --config bench/sf200-2gpu/sirius-sf200-2gpu.yaml \
  --name sf200_2gpu_validation
```

All 22 queries should pass. A correctness failure most commonly means:
- Patched libcudf not loaded (`LD_PRELOAD` missing or path wrong)
- `l_shipinstruct` codec changed from `dictionary` (see gotchas)

---

## Step 7 — Collect and summarize timings

Output lands in `test/tpch_performance/output/sf200_2gpu/`:

```
output/sf200_2gpu/
  csv/runtimes.csv     # per-query timings, all 3 iterations
  metadata.json        # config snapshot + git sha
  log_dir/             # combined Sirius spdlog
  gpu/q<N>/result.txt  # GPU query results
```

Summarize:

```bash
awk -F',' 'NR>1 { sum += $3 } END { printf "Total best-of-3: %.3f s\n", sum }' \
  test/tpch_performance/output/sf200_2gpu/csv/runtimes.csv
```

Verify both GPUs received work (look for `GPU 0` and `GPU 1` in the log):

```bash
grep -E "GPU [01]|device=[01]|preferred_device=[01]" \
  test/tpch_performance/output/sf200_2gpu/log_dir/*.log | head -40
```

---

## Pinning strategy: why `--mode grouped` matters even with 2 GPUs

With `--mode sequential` (union-pin), all 22 queries' columns are pinned simultaneously
across both GPUs — ~25 GB per GPU. This is fine at SF200 (31% per GPU), so sequential
mode doesn't cause downgrade issues here the way it does at SF1000.

However, `--mode grouped` is still preferred because:
- Each query's pin/unpin cycle lets Sirius pick the freshest scan batches for dispatch
- Operator memory reservations are tighter (no stale pinned batches holding space)
- It matches the SF1000 baseline methodology, making results comparable

| Mode | SF200 GPU peak per GPU | Downgrade fires? |
|---|---|---|
| `sequential` | ~25 GB (31%) | No — well under trigger at 68 GB |
| `grouped` | ~15 GB (19%) | No |

---

## SNMG-specific gotchas

### 1. `memory_prefetcher` must be disabled

The background host→GPU prefetcher is single-GPU only. Setting `enable: true`
with `num_gpus: 2` logs a warning and disables itself. Set `enable: false` explicitly
in both `scan_manager.memory_prefetcher` and `memory.host.memory_prefetcher` to
suppress the warning.

### 2. Hash-join retry warnings are expected

When `num_gpus=2`, the probe task on GPU 1 can reach
`get_next_task_input_data_for_build_probe` while the build task on GPU 0 is still
scheduled but not yet running. This logs:

```
[warn] invalid hash table build state 2 — retrying (attempt N/100)
```

This is handled: `MAX_RETRIES=100` with 50 ms backoff. Queries complete correctly.
If retries exceed 100 on any query, it indicates a scheduling starvation issue —
check GPU utilization and `pipeline.num_threads`.

### 3. Verify NVLink P2P is active

```bash
# Before running the benchmark:
nvidia-smi nvlink --status -i 0

# Should show active NVLink lanes.
# If NVLink shows "Inactive", cross-GPU transfers use PCIe (~48 GB/s) instead
# of NVLink (~900 GB/s) — expect substantially longer query times for joins.
```

### 4. `use_sirius_datasource` is forced true

The engine automatically sets `use_sirius_datasource = true` when `num_gpus > 1` and
logs a `WARN`. Set it explicitly in the config to suppress the log noise.

### 5. `LD_PRELOAD` over `LD_LIBRARY_PATH` (same as SF1000)

```bash
# Correct:
export LD_PRELOAD="$HOME/cudf-src/cpp/build/libcudf.so"

# Wrong — pixi's unpatched cuDF loads first:
export LD_LIBRARY_PATH="$HOME/cudf-src/cpp/build:$LD_LIBRARY_PATH"
```

### 6. JIT warm-up adds ~19 s to the first query execution

`expression_evaluator_strategy = 'ast_jit'` compiles on first execution regardless
of scale factor. Best-of-3 reporting absorbs this. Do not report iteration 1 timing
from a cold process.

### 7. `l_shipinstruct` must stay `dictionary` (same as SF1000)

In `bench/sf1000-repro/plans/lineitem.txt`, do not change `l_shipinstruct` from
`dictionary` codec. Q19 decode-time predicate pushdown requires it. Changing to
`identity` silently disables the pushdown at any scale factor.

---

## Quick-start checklist

```
[ ] git checkout repro/sf1000-8.5s && git submodule update --init --recursive
[ ] pixi run make
[ ] pixi run bash bench/sf1000-repro/build-libcudf.sh
[ ] nvidia-smi nvlink --status -i 0   # confirm NVLink active
[ ] pixi run bash test/tpch_performance/generate_tpch_data.sh 200 \
      --output ~/tpch_parquet_sf200 --jobs 16
[ ] mkdir -p bench/sf200-2gpu
    # write sirius-sf200-2gpu.yaml and run.sh (see above)
[ ] pixi run bash bench/sf200-2gpu/run.sh   # validation pass first
[ ] pixi run bash bench/sf200-2gpu/run.sh   # timing run (best-of-3)
[ ] awk -F',' 'NR>1{sum+=$3}END{print sum}' \
      test/tpch_performance/output/sf200_2gpu/csv/runtimes.csv
```
