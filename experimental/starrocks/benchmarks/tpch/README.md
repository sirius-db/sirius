# TPC-H benchmark harness for Sirius GPU compute nodes

Times TPC-H queries over external parquet against whatever FE answers on port 9030, then
checks the answers against DuckDB. The queries scan the files through `FILES()` CTEs — no
data loading step; the FE byte-splits large tables across the compute nodes.

This tree ships all 22 queries. Every departure from the stock TPC-H text is recorded in
[`QUERY-DEVIATIONS.md`](QUERY-DEVIATIONS.md) (never as `--` comments inside the `.sql`
files, which `mysql -e` would swallow). q11 keeps its `0.0001` fraction spec-correct at any
scale through `__TPCH_SF__`, substituted with `$TPCH_SF` (default 1) by `bench.sh` and
`oracle.py`; at SF1000 without it the query returns zero rows on every engine.

## Layout

- `queries/qNN.sql` — the queries with `FILES()` preludes; `__TPCH_DATA__` is substituted
  with `$TPCH_DATA` at run time (a directory holding `<table>/*.parquet`, the layout
  [`bench/common/gen-tpch.sh`](../../../../bench/common/gen-tpch.sh) writes) and
  `__TPCH_SF__` with `$TPCH_SF`.
- `bench.sh` — per-query sweep: 1 discarded warm-up + N timed runs, wall-clock ms
  around `mysql -e`, medians taken later. Refusals (ERROR) are recorded once; hangs
  are cut at `$QUERY_TIMEOUT` (default 30 s) and recorded as wedges. After either,
  `$RESTART_CMD` runs — required for the Sirius CN (see caveat below).
  **`--cold`** records the warm-up (run 0) instead of discarding it, tagged
  `phase=cold` and cut at `$COLD_TIMEOUT` (default 180 s) — run 0 is the only run
  that exercises first contact (plan-cache misses, first-touch allocation).
  `--cold-restart` additionally restarts before each query so every run 0 is a true
  cold cluster.
- [`../../tools/oracle.py`](../../tools/oracle.py) — runs the same SQL through DuckDB on
  the CPU over the same parquet and writes one `<q>.tsv` per query, formatted like
  `mysql --batch`. Needs the `duckdb` Python package (the repo's pixi env has it).
- [`../../tools/compare.py`](../../tools/compare.py) — diffs EVERY `<q>.rN.out` of the
  sweep against the oracle's TSVs: row count first, then cell by cell (relative
  tolerance for numerics, exact match otherwise). A query's verdict is the worst of its
  runs, so a cold run that returned zero rows next to two matching warm runs is reported
  as the flake it is; it exits non-zero unless every run of every query matches. This is
  the harness's correctness gate; `bench.sh` alone times and counts rows.
- [`../cluster8.sh`](../cluster8.sh) — 1 FE + N Sirius CNs, one CN per GPU, cross-CN
  exchange over nixl.

## Run

```bash
# from experimental/starrocks/ — 1 FE + one CN per GPU (NUM_CNS defaults to 8).
# Keep it in its own terminal or background task.
NUM_CNS=4 GPU_MEM=48GiB STAGING=8GiB ./benchmarks/cluster8.sh

# The sweep. MIN_BACKENDS is the real CN count (see below), not a floor.
TPCH_DATA=/path/to/tpch_sf1 MIN_BACKENDS=4 QUERY_TIMEOUT=120 \
  ./benchmarks/tpch/bench.sh /tmp/bench/A/timings.csv 3 q01 q06

# The answers. oracle.py runs DuckDB over the same files; compare.py diffs the two.
python3 tools/oracle.py benchmarks/tpch/queries /path/to/tpch_sf1 /tmp/bench/oracle q01 q06
python3 tools/compare.py /tmp/bench/A /tmp/bench/oracle
```

At SF1, q01 returns 4 rows and q06 one row (`revenue` 123141078.2283, low digits varying
with double-precision summation order — that is what the relative tolerance is for);
`compare.py` reports `2/2 match`. Setting `ORACLE_DIR=/tmp/bench/oracle` on the `bench.sh`
line runs the compare step at the end of the sweep instead of as a separate command, and
the sweep then exits with compare.py's status; the oracle still has to be produced first.

A two-CN smoke test on one GPU is `pixi run cluster2` with `MIN_BACKENDS=2`;
[`DEMO.md`](../../DEMO.md) walks through a Q6-shaped query on that cluster (different date
and discount bounds, so its `61567694.9502` is not q06's answer).

**Caveat:** the CN's `cancel_plan_fragment` retires a cancelled query's parked output
and rendezvous state and refuses its later fragments, but it cannot abort a fragment
already inside the engine's `run()`; a query wedged there still strands that fragment
until it ends, and the stranded fragment starves the CN (the FE reports "No available
backends" for everything after). `RESTART_CMD` therefore stays mandatory for a Sirius
cluster — without it, every measurement after a wedge is invalid:

```bash
RESTART_CMD='pkill -f "[s]irius-starrocks-cn"; pkill -f "[S]tarRocksFE"; sleep 5; \
  (cd <repo>/experimental/starrocks && NUM_CNS=4 nohup ./benchmarks/cluster8.sh >/tmp/cluster8.log 2>&1 &)'
```

## Multiple GPUs / more CNs

`cluster2` starts two CNs on one GPU (each with its own memory carve-out).
[`cluster8.sh`](../cluster8.sh) puts one CN per GPU: `NUM_CNS` CNs, each with a contiguous
10-port block from `PORT_BASE`, its own `--gpu-device`, `GPU_MEM`/`HOST_MEM` limits and a
`STAGING`-sized exchange arena. It clears an inherited `CUDA_VISIBLE_DEVICES` (which would
otherwise win over `--gpu-device` and land every CN on one GPU) and exports
`SIRIUS_QUERY_WATCHDOG_SECS` so a stalled query fails loudly instead of wedging a CN. Check
that `nvidia-smi` shows exactly one CN process per GPU before measuring anything.

The harness itself is topology-agnostic — it only talks to the FE; `wait_alive` accepts any
mix of alive compute nodes/backends (the `Alive` column, resolved from the header row, not a
`grep` over the whole row). Set `MIN_BACKENDS` to the real node count: it is the expected
size, not a floor, and the sweep aborts if more nodes are alive than you declared (a
threshold below the real topology can be satisfied mid-boot, and the sweep then measures a
half-started cluster).

Whether every CN actually did work is a separate question from whether the answer was
right: [`../../scripts/cn-distribution.py`](../../scripts/cn-distribution.py) reads the
per-CN telemetry under `.cn<N>/` and reports the per-CN split, carrying a CN that produced
nothing as an explicit zero.

## Before calling a query broken: check whether it was a transport timeout

At SF500 and above, a *refused* query is often a healthy CN that ran out of clock, not a bug.
The transport is serialized (one thread per CN, blocking round trips), so a peer's
`request_staging_lease` queues behind whatever fragment its engine thread is currently running.
The default 60 s bound then fails a query that would have completed — an SF100 q08 refused at
60758 ms, just past the bound, was exactly this.

Tell the two apart in the CN log:

| | Looks like |
|---|---|
| Transport timeout | `request_staging_lease to <host>:<port>: failed to read reply frame: …` — names the peer and method, no peer-side status |
| Real query failure | `… failed with status <code>: <messages>` — a `StatusPB` the peer actually sent |

For the first, raise `SIRIUS_CN_RPC_TIMEOUT_SECS` and re-run before concluding anything. For
the second, raising it will not help — the peer answered and refused.

High-level knob list: [`../../docs/TUNABLES.md`](../../docs/TUNABLES.md).

## Notes

The harness only talks to the FE, so the same sweep times a stock StarRocks FE + BE cluster
on port 9030 as a CPU baseline. Run one engine at a time: they share the FE port, the backend
port pairs and the host CPUs. At small scale factors, fixed per-query overheads (fragment
dispatch, first-touch allocation) matter as much as scan throughput. Results are indicative,
not a TPC-compliant benchmark.
