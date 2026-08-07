# Reproducing the A-vs-B TPC-H comparison by hand

The one-command path is `TPCH_DATA=… ./run-comparison.sh`; this file is the same procedure
unrolled, with the checks and the traps.

## 0. Prerequisites

- One NVIDIA GPU (the reference run used an L4, 23 GiB; two CNs share it — each takes an
  8 GiB carve-out + a 1.25 GiB exchange staging arena, see `pixi.toml`).
- TPC-H parquet at `$TPCH_DATA/<table>/*.parquet` (the reference run used SF1, one file per
  table). Both engines scan these same files via `FILES()` — there is no loading step.
- Docker (only to extract stock StarRocks binaries for engine B).
- JDK 17+ for engine B's FE (`JAVA_HOME`); engine A's toolchain comes entirely from pixi.
- A `mysql` client on the PATH (engine A's pixi env has one:
  `pixi run --manifest-path experimental/starrocks/pixi.toml mysql …`).

Run one engine at a time: they share the FE port (9030), the backend port pairs, and the
host CPUs. Measuring one while the other runs invalidates both.

## 1. Engine A — Sirius GPU compute nodes

```bash
cd experimental/starrocks
pixi run cluster2          # builds the CN + FE on first run; keep it in its own terminal
```

Wait until `mysql -h127.0.0.1 -P9030 -uroot -e "SHOW COMPUTE NODES;"` shows **two** rows
with `Alive=true` — one alive CN means the cluster is still booting, and a sweep started
then records phantom failures.

```bash
TPCH_DATA=/path/to/tpch_sf1 \
RESTART_CMD='pkill -f "[s]irius-starrocks-cn"; pkill -f "[S]tarRocksFE"; sleep 5; \
  (cd <repo>/experimental/starrocks && nohup pixi run cluster2 >/tmp/cluster2.log 2>&1 &); sleep 20' \
  ./benchmarks/tpch/bench.sh /tmp/bench/A/timings.csv 3
```

Protocol per query: 1 discarded warm-up + 3 timed runs, wall clock around `mysql`,
refusals recorded once, hangs cut at 30 s. `RESTART_CMD` restores the cluster after any
failure so later measurements stay valid. Expected on the reference code (branch
`demo-multi-cn`, commit `312e4535` or later): **22/22 pass**, warm medians ~0.3–1.3 s at SF1.

Tear down: `pkill -f '[s]irius-starrocks-cn'; pkill -f '[S]tarRocksFE'` — then verify
`nvidia-smi` shows 0 MiB used. (The CN binary is `sirius-starrocks-cn`; a pattern naming
anything else kills nothing.)

## 2. Engine B — stock StarRocks, same topology

```bash
JAVA_HOME=/usr/lib/jvm/<jdk17+> ./benchmarks/tpch/setup-engine-b.sh   # once; ~/starrocks-bench
~/starrocks-bench/fe/bin/start_fe.sh --daemon
~/starrocks-bench/be1/bin/start_be.sh --daemon
~/starrocks-bench/be2/bin/start_be.sh --daemon
mysql -h127.0.0.1 -P9030 -uroot -e \
  'ALTER SYSTEM ADD BACKEND "127.0.0.1:9050"; ALTER SYSTEM ADD BACKEND "127.0.0.1:9052";'
```

The backends MUST run as BEs, not shared-nothing CNs — CNs are not schedulable for
`FILES()` external scans and every query fails with "No available backends". Wait for
`SHOW BACKENDS;` to show both `Alive=true`, then:

```bash
TPCH_DATA=/path/to/tpch_sf1 ./benchmarks/tpch/bench.sh /tmp/bench/B/timings.csv 3
```

(No `RESTART_CMD` needed — stock StarRocks cleans up after failed queries.)

## 3. Compare

```bash
./benchmarks/tpch/analyze.py /tmp/bench/A/timings.csv /tmp/bench/B/timings.csv \
    results.md tpch_a_vs_b.png
```

Markdown table of per-query medians + geometric-mean speedup, and a log-scale bar plot.
`analyze.py` needs matplotlib for the plot (the table prints regardless).

## Interpreting the results (reference-run caveats)

- **Correctness**: every engine-A result was validated against DuckDB over the same
  parquet. Counts, keys, and orderings are exact. Revenue-shaped sums
  (`x*(1-l_discount)`) run up to ~0.4 % low on a few rows: the FE plans exact DECIMAL,
  the translator lowers to FP64 (documented as the open decimal-aggregation work item).
- **Fairness**: A executes on the GPU — that is the point of the comparison; B's BE is a
  mature vectorized CPU engine. At SF1, fixed per-query overheads (fragment dispatch,
  first-touch allocation) matter as much as scan throughput. Indicative, not TPC-compliant.
- **Known flakes**: back-to-back sweeps occasionally hit a loud "exchange staging arena
  exhausted" refusal (q05/q21 class); the restart protocol recovers and a solo rerun
  passes. Raise `SIRIUS_EXCHANGE_STAGING_BYTES` in `pixi.toml` if it recurs.
- The engine-A journey from 3/22 to 22/22, with per-fix commits and root causes, is in
  `QUERY-TIMEOUT-ANALYSIS.md` and `ROADMAP-8CN-TPCH.md` ("What integrating Sirius as a
  StarRocks CN actually required") at the repo root.
