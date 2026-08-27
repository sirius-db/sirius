---
name: tpch-cn-sweep
description: >
  Run the 22-query TPC-H sweep against a Sirius StarRocks CN cluster on any box, scale factor,
  data path, or GPU count. Use when launching bench.sh, sizing GPU_MEM/STAGING, checking results
  against the DuckDB oracle, or triaging pass/wedge/refused rows. Not the in-process Sirius-vs-DuckDB
  benchmark skill; not the GB200-hardcoded tpch-bench ops skill.
---

Read [`bench/rtxpro6000-2gpu/TPCH-SWEEP-RUNBOOK.md`](../../../bench/rtxpro6000-2gpu/TPCH-SWEEP-RUNBOOK.md)
fully before acting. Harness contract: [`experimental/starrocks/benchmarks/tpch/README.md`](../../../experimental/starrocks/benchmarks/tpch/README.md).
Query text deviations: [`QUERY-DEVIATIONS.md`](../../../experimental/starrocks/benchmarks/tpch/QUERY-DEVIATIONS.md).
Living defects: [`notes/OPEN.md`](../../../notes/OPEN.md).

## Inputs

`$DATA` (parquet root `<table>/*.parquet`), `$SF`, `$N` (CNs = GPUs), `$CARD` (HBM the card reports).

## Traps (these silently corrupt results)

- **No correctness gate.** `bench.sh` scores `pass` on exit 0 + non-empty file + no `ERROR` on line 1. Quote nothing without the DuckDB oracle (`tools/oracle.py` + `tools/compare.py` in `bench/rtxpro6000-2gpu/tools/`). Use pip `duckdb`, not `build/release/duckdb` (it loads Sirius and fights the CN for GPU).
- **Correct empty is a `wedge`.** `mysql --batch` prints nothing for 0 rows. q11 is empty at SF≥100 against the oracle — not a bug.
- **Preflight or you corrupt registries.** `pgrep` CN/FE, `ss` on 8030/9010/9020/9030/91xx, `nvidia-smi` compute apps — all must be empty.
- **`grep -c true` over-counts Alive.** Count column 9: `awk -F'\t' '$9=="true"'`.
- **Unset `CUDA_VISIBLE_DEVICES`** before launch; an exported value wins over `--gpu-device`.
- **`QUERY_TIMEOUT` default 30 s** collapses hangs and refusals. Set 180 (SF100) or 1800 (SF500). Raise FE `query_timeout` too (`SET GLOBAL query_timeout=...`) — `bench.sh` does not.
- **`bench.sh q05` without a run count** shifts args wrong (`RUNS=q05`, sweeps all 22).
- Engines A and B share port 9030 — never simultaneous.

## Sizing

`occupancy = GPU_MEM + STAGING + ~2 GiB` must fit `$CARD`. Pick STAGING from a **measured** config for this box/SF (`STATUS.md`, `CONFIGURATIONS.md`, or `cn-tuning`), then `GPU_MEM = CARD − STAGING − 2 GiB`.

**Do not use** `STAGING ≈ 96 GiB × SF/500 / N`. That formula is retired: arena occupancy is a pool-pressure gauge (`push_packed` copies arena→pool before releasing the lease). STATUS.md 2026-08-20.

Working RTX reference: `GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB` plus 1 GiB operator budgets via `--sirius-config` (the flag path cannot express them).

## Sweep skeleton

```bash
# 1. preflight  2. bring cluster up  3. wait Alive-column count == N
TPCH_DATA=$DATA QUERY_TIMEOUT=180 MIN_BACKENDS=$N RESTART_CMD=./restart.sh \
  experimental/starrocks/benchmarks/tpch/bench.sh /tmp/bench/A/timings.csv 3
# 4. oracle + compare before quoting
```

`mysql` exists only in the starrocks pixi env. Copy the CSV off `/tmp`. After any wedge/refusal, restart before the next row (`cancel_plan_fragment` is a stub).
