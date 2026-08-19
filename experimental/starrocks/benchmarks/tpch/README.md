# TPC-H benchmark harness: Sirius GPU CNs (A) vs stock StarRocks (B)

Times TPC-H Q1–Q22 over external parquet against whatever FE answers on port 9030,
so the same harness measures both engines. Both engines scan the **same files** via
`FILES()` CTEs — no data loading step, the FE byte-splits large tables across backends
in both cases.

## Layout

- `queries/qNN.sql` — Q1–Q22 with `FILES()` preludes; `__TPCH_DATA__` is substituted
  with `$TPCH_DATA` at run time (a directory holding `<table>/*.parquet`).
- `bench.sh` — per-query sweep: 1 discarded warm-up + N timed runs, wall-clock ms
  around `mysql -e`, medians taken later. Refusals (ERROR) are recorded once; hangs
  are cut at `$QUERY_TIMEOUT` (default 30 s) and recorded as wedges. After either,
  `$RESTART_CMD` runs — required for engine A (see caveat below).
  **`--cold`** records the warm-up (run 0) instead of discarding it, tagged
  `phase=cold` and cut at `$COLD_TIMEOUT` (default 180 s) — run 0 is the only run
  that exercises first contact (lazy nixl session setup, plan-cache misses).
  `--cold-restart` additionally restarts before each query so every run 0 is a true
  cold cluster.
- `analyze.py` — merges two result CSVs into a markdown table + log-scale bar plot
  (median ms per query, geometric-mean speedup over the comparable set). It compares
  the two engines' **row counts** as well as their times: a query where they
  disagree is reported at the top of the markdown, dropped from the geomean, hatched
  in the plot, and exits the script 1 (`--allow-mismatch` to report only). Cold rows
  are kept out of the warm medians and listed separately.
- `setup-engine-b.sh` — lays out stock StarRocks (1 FE + 2 BEs) from the prebuilt
  artifacts Docker image.

## Engine A (Sirius GPU CNs)

```bash
# from experimental/starrocks/ — builds and starts 1 FE + 2 GPU CNs
pixi run cluster2   # keep it in its own terminal / background task

TPCH_DATA=/path/to/tpch_sf1 \
RESTART_CMD='pkill -f "[s]irius-starrocks-cn"; pkill -f "[S]tarRocksFE"; sleep 5; \
  (cd <repo>/experimental/starrocks && nohup pixi run cluster2 >/tmp/cluster2.log 2>&1 &)' \
  ./bench.sh /tmp/bench/A/timings.csv 3
```

**Caveat (as of 2026-08-07):** the CN does not implement `cancel_plan_fragment`, so a
hung or mid-execution-failed query strands its fragments; the stranded fragments
eventually starve the CNs and the FE reports "No available backends" for everything
after. `RESTART_CMD` is therefore mandatory for A — without it, every measurement
after the first failure is invalid.

## Engine B (stock StarRocks baseline)

```bash
JAVA_HOME=/usr/lib/jvm/<jdk17+> ./setup-engine-b.sh   # once
# start FE + BEs and register them (the script prints the exact commands)
TPCH_DATA=/path/to/tpch_sf1 ./bench.sh /tmp/bench/B/timings.csv 3
```

Run one engine at a time: A and B share the FE port (9030), the backend port pairs,
and the host CPUs. Take A fully down before measuring B and vice versa.

## Compare

```bash
./analyze.py /tmp/bench/A/timings.csv /tmp/bench/B/timings.csv results.md tpch_a_vs_b.png
```

## Multiple GPUs / more CNs

`cluster2` starts two CNs on one GPU (each with its own memory carve-out). To spread
CNs across GPUs, replicate the `cluster2` task pattern in `pixi.toml` with one CN per
GPU and `CUDA_VISIBLE_DEVICES=<i>` per CN process, and register each CN's heartbeat
port on the FE. The harness itself is topology-agnostic — it only talks to the FE;
`wait_alive` accepts any mix of alive compute nodes/backends. Set `MIN_BACKENDS` to
the real node count: it is the expected size, not a floor, and the sweep aborts if
more nodes are alive than you declared (a threshold below the real topology can be
satisfied mid-boot, and the sweep then measures a half-started cluster).

## Before calling a query broken: check whether it was a transport timeout

At SF500 and above, a *refused* query is often a healthy CN that ran out of clock, not a bug.
The transport is serialized (one thread per CN, blocking round trips), so a peer's
`request_staging_lease` queues behind whatever fragment its engine thread is currently running.
The default 60 s bound then fails a query that would have completed — SF100 **q08 refused at
60758 ms** in `OPEN-ISSUES.md` is exactly this.

Tell the two apart in the CN log:

| | Looks like |
|---|---|
| Transport timeout | `request_staging_lease to <host>:<port>: failed to read reply frame: …` — names the peer and method, no peer-side status |
| Real query failure | `… failed with status <code>: <messages>` — a `StatusPB` the peer actually sent |

For the first, raise `SIRIUS_CN_RPC_TIMEOUT_SECS` and re-run before concluding anything. For
the second, raising it will not help — the peer answered and refused.

High-level knob list: [`../../docs/TUNABLES.md`](../../docs/TUNABLES.md).

## Fairness notes

A executes on the GPU (that is the point of the comparison); B's BE is a mature
vectorized CPU engine. At small scale factors, fixed per-query overheads (fragment
dispatch, first-touch allocation) matter as much as scan throughput. Results are
indicative, not a TPC-compliant benchmark.
