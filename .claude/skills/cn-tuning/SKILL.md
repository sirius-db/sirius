---
name: cn-tuning
description: >
  Size Sirius StarRocks CN memory and operator budgets for a given GPU box and TPC-H scale factor.
  Use when choosing GPU_MEM, STAGING, HOST_MEM, hash_partition_bytes, scan_task_batch_size, or
  when a query OOMs / exhausts the exchange arena / looks like an engine bug that is actually
  mis-sizing. Covers the derived-flag path vs a full sirius.yaml.
---

Read, in this order:

1. [`bench/rtxpro6000-2gpu/SIRIUS-TUNING-RUNBOOK.md`](../../../bench/rtxpro6000-2gpu/SIRIUS-TUNING-RUNBOOK.md) — how the knobs work
2. [`bench/a100x8/TUNING.md`](../../../bench/a100x8/TUNING.md) — measured method on 8× A100
3. [`experimental/starrocks/docs/TUNABLES.md`](../../../experimental/starrocks/docs/TUNABLES.md) — env vars (fail-closed registry)
4. Per-box facts: `bench/<box>/HARDWARE.md`, `CONFIGURATIONS.md`, `STATUS.md`

Open work on derivation: [`notes/OPEN.md`](../../../notes/OPEN.md) PLAN-03.

## Two paths — mutually exclusive

| Path | How | Reaches |
|---|---|---|
| **A — derived** | `cluster8.sh` always passes `--gpu-memory-limit` / `--host-memory-limit` | memory + CPU affinity + telemetry. **No `operator_params`.** |
| **B — full YAML** | `--sirius-config` | everything, including operator budgets |

`--sirius-config` conflicts with the memory flags. Stock `cluster8.sh` cannot do path B; edit the launcher to drop those flags. Operator budgets (`hash_partition_bytes`, `max_build_hash_table_bytes`, `scan_task_batch_size`, `concat_batch_bytes`) are why SF500 q08 moved — they need path B.

## Units

`K/M/G/T` are SI (1000). `Ki/Mi/Gi/Ti` are 1024. `8GB` ≠ `8GiB`. Committed configs mix both.

## What actually failed on RTX

- `derived_default_batch_size()` sizes every operator budget from **`prop.totalGlobalMem` (physical HBM)**, not the configured pool. At `GPU_MEM=60GiB` on a 95 GiB card the default is ~2.4× oversized. Cutting budgets to ~1 GiB fixed q08; it did **not** fix q09.
- Arena high-water is a **pool-pressure gauge**, not independent demand. `push_packed` deep-copies into the pool before releasing the lease. A starved pool ratchets the arena to capacity.
- **Retire** `STAGING ≈ 96 GiB × SF/500 / N`. Do not re-split pool/arena for RTX q09 — 60/32 through 76/16 all fail; the window is empty. q09 needs copy-out-on-arrival (PLAN-01).

## Sizing procedure

1. Probe HBM, NUMA, link type (`nvidia-smi topo -m`). Never membind cpuless GPU-HBM nodes.
2. Pick STAGING from a **validated** row for this card class (A100: `CONFIGURATIONS.md`; RTX: `STATUS.md` 60/32; GB200: `tpch-bench` / `engine-a.env`). Not the retired formula.
3. `GPU_MEM = allocatable_HBM − STAGING − ~2 GiB`. Target 85–97% occupancy. More pool does not buy speed (RTX arm A vs B: −0.4%).
4. `HOST_MEM × N` must fit CPU DRAM with `SwapTotal 0`, leaving page cache for the dataset. `%` mem limits on GPU boxes resolve against HBM-inflated `MemTotal` — use absolute GiB.
5. If shuffle queries OOM at HASH_JOIN after a healthy arena, the lever is **operator budgets** (path B), not another STAGING bump.
6. Confirm the CN log line `exchange staging arena: N bytes` matches what you passed. Teardown `high water N of M` is the measurement — 50% of those lines are lost if you kill CNs inside SHUTDOWN_GRACE.

`use_odirect` is catastrophic on NFS (OPEN M4). Do not flip `use_sirius_datasource` to “fix” that.
