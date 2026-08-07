---
name: tpch-bench
description: Operate the Sirius-as-StarRocks-CN demo and its TPC-H benchmark. Use when bringing the 2-CN GPU cluster up or down, running a query or the 22-query sweep, triaging a failing/hanging/slow query, verifying results against DuckDB, or reproducing the A-vs-B comparison against stock StarRocks.
---

Everything here runs from the repo root unless noted. Deep references: the sweep protocol
lives in `experimental/starrocks/benchmarks/tpch/REPRODUCE.md`, per-defect history in
`QUERY-TIMEOUT-ANALYSIS.md`, reference numbers in
`experimental/starrocks/benchmarks/tpch/results/`. Prefer those over re-deriving.

## Facts the environment does not confess

- The CN binary is `sirius-starrocks-cn` (a pkill pattern naming anything else kills nothing).
  The FE java class is `StarRocksFE`. FE MySQL port: 9030.
- One alive CN means the cluster is STILL BOOTING — act only on **2× Alive=true**.
- Engines A and B share port 9030 and the host CPUs: run one at a time.
- The engine A CN has no real query cancellation: a hung or killed query can strand state;
  after any hang, restart the cluster rather than trusting the next result.
- Queries in `experimental/starrocks/benchmarks/tpch/queries/` carry a `__TPCH_DATA__`
  placeholder; `bench.sh` substitutes `$TPCH_DATA`. For manual runs, sed it yourself.
- Passing `-e "$Q"` through nested shells mangles the `FILES()` quotes — pipe the file
  into mysql instead.

## Cluster (engine A)

```bash
cd experimental/starrocks
pixi run cluster2 &                                   # background/own terminal; first run builds
until [ "$(mysql -h127.0.0.1 -P9030 -uroot -N -e 'SHOW COMPUTE NODES;' 2>/dev/null | grep -c true)" -ge 2 ]; do sleep 5; done

# teardown -- verify both lines
pkill -f '[s]irius-starrocks-cn'; pkill -f '[S]tarRocksFE'
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # must be 0 MiB
```

Inside a Claude session use the pixi env for mysql:
`pixi run --manifest-path experimental/starrocks/pixi.toml bash -c 'mysql ...'`.

## One query, with the DuckDB oracle

```bash
export TPCH_DATA=/path/to/tpch_sf1
sed "s|__TPCH_DATA__|$TPCH_DATA|g" experimental/starrocks/benchmarks/tpch/queries/q05.sql > /tmp/q.sql
timeout 30 mysql -h127.0.0.1 -P9030 -uroot --batch < /tmp/q.sql

# oracle: same SQL, parquet read directly
pixi run python3 - <<'PY'
import duckdb, re
sql = open("/tmp/q.sql").read()
sql = re.sub(r'FILES\("path"="file://([^"]+)","format"="parquet"\)', r"read_parquet('\1')", sql)
print(duckdb.sql(sql).df())
PY
```

Comparison bar: counts, keys, and ordering must be exact; values on `x*(1-l_discount)`
expressions may run up to ~0.4% low (the open decimal-lowering item, task history in
`QUERY-TIMEOUT-ANALYSIS.md`).

## The sweep

```bash
TPCH_DATA=$TPCH_DATA \
RESTART_CMD='pkill -f "[s]irius-starrocks-cn"; pkill -f "[S]tarRocksFE"; sleep 5; \
  (cd '"$PWD"'/experimental/starrocks && nohup pixi run cluster2 >/tmp/c2.log 2>&1 &); sleep 20' \
  ./experimental/starrocks/benchmarks/tpch/bench.sh /tmp/bench/A/timings.csv 3
```

1 discarded warm-up + 3 timed runs per query; refusals recorded once; hangs cut at 30 s.
Expected at the reference code: 22/22 pass, warm medians ~0.3–1.3 s at SF1.
Full A-vs-B (engine B setup, sweep, table + plot):
`TPCH_DATA=$TPCH_DATA ./experimental/starrocks/benchmarks/tpch/run-comparison.sh`.

## Triage a non-pass row

| Symptom | Meaning | Move |
|---|---|---|
| `refused` + "staging lease ... arena exhausted" | cumulative arena leak under back-to-back load, NOT this query's fault | restart cluster, rerun the query solo (warm-up + 3), use the solo median |
| `refused` + "declared X but the source sink produces Y" | a translator/engine schema disagreement on a fragment hop | real defect: capture EXPLAIN + the exact error; check QUERY-TIMEOUT-ANALYSIS.md for the class before diagnosing fresh |
| `refused` + other error text | loud, query-local; the text names the layer | read it literally — the loud-failure net is trustworthy |
| `wedge` (30 s timeout, no error) | the worst class: something silent survived | restart, rerun solo; if it reproduces solo, treat as a new silent-failure bug — capture both CN logs and a stack of the CN (`gdb -p <pid> -batch -ex 'thread apply all bt'`) while wedged |
| pass but wrong values | check the ~0.4% revenue band first | outside the band → correctness bug: bisect with the oracle per sub-expression |

After ANY wedge or refusal, the queries that follow are only trustworthy if the cluster was
restarted — that is what `RESTART_CMD` is for.

## Verifying a fix

Suites, in cost order (all must be green before trusting a live run):

```bash
pixi run --manifest-path experimental/starrocks/pixi.toml bash -c \
  'cd experimental/starrocks/crates/starrocks-plan-translator && cargo test'   # translator
pixi run --manifest-path experimental/starrocks/pixi.toml cn-test-no-engine
pixi run --manifest-path experimental/starrocks/pixi.toml cn-test             # engine-linked, incl. the 76-case wire-type parity gate
pixi run make && pixi run make test                                           # only if src/** C++ changed
pixi run bash -c 'export LD_LIBRARY_PATH=$PWD/build/release/extension/sirius:$LD_LIBRARY_PATH; \
  cargo test --manifest-path rust/Cargo.toml -p sirius --lib -- --test-threads=1'  # GPU harness (watchdogged)
```

Then the live gate: the affected query solo vs the oracle, its regressions, and a sweep.
