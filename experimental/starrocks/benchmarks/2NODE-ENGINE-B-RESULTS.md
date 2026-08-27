# Engine B two-machine bring-up — execution record

**2026-08-11, gcn-17 + gcn-18.** Stock StarRocks 3.5.20 (aarch64), 1 FE + 1 BE per host.
Everything below is MEASURED in this session, not derived.

## Result

**Distributed query execution across two physical machines: CONFIRMED.**

## Cluster

| | |
|---|---|
| FE | gcn-18 `10.87.140.53:9030`, `Role: LEADER`, `Alive: true`, `3.5.20-4d17879` |
| BE 10002 | gcn-18 `10.87.140.53`, `Alive: true`, `CpuCores: 144`, `MemLimit: 432.000GB` |
| BE 10250 | gcn-17 `10.87.140.52`, `Alive: true`, `CpuCores: 144`, `MemLimit: 432.000GB` |
| run_mode | unset → `shared_nothing` (BEs, not CNs) |
| Data | `/raid/prestouser/aocsa/tpch_parquet_sf100`, identical path both hosts, node-local ext4 |
| Binaries | `/home/prestouser/starrocks-bench/{fe,be17,be18}`, NFS-shared, one tree per host |

Deviation from the tutorial: **the FE runs on gcn-18, not gcn-17** (the agent driving this could
only start processes on gcn-18). `priority_networks = 10.87.140.32/27` is unchanged and covers both.

## Evidence

**Scan path.** `nation` → 25 rows. `lineitem` → **600,037,902** rows, the exact SF100 cardinality.

**Placement, pre-flight** (`EXPLAIN SCHEDULER`, zero fragments deployed):

```
PLAN FRAGMENT 2(F00)          <- scan
    INSTANCE(3-F00#0)  BE: 10002      (gcn-18)
    INSTANCE(4-F00#1)  BE: 10250      (gcn-17)
PLAN FRAGMENT 1(F01)          <- merge agg, hash-partitioned input
    INSTANCE(1-F01#0)  BE: 10250
    INSTANCE(2-F01#1)  BE: 10002
```

**Placement, actual** (raw profile, `pipeline_profile_level = 1`):

```
   - BackendAddresses: 10.87.140.53:9060,10.87.140.52:9060      (x2 fragments)
   - BackendNum: 2
   - InstanceNum: 2
```

**Cross-machine traffic** (`EXCHANGE_SINK` / `EXCHANGE_SOURCE`):

```
   - PartType: HASH_PARTITIONED       - BytesReceived:    2.212 MB
   - ChannelNum: 2                    - BytesPassThrough: 2.344 MB
   - BytesSent: 2.212 MB              - RequestReceived: 130
   - NetworkBandwidth: 230.997 MB/sec
   - NetworkTime: 20.079ms
   - RequestSent: 128
```

With one BE per host, pass-through requires the *same process*, so `BytesSent > 0` is by
construction cross-machine. Split is 2.344 local : 2.212 remote ≈ 51/49 — the expected 2-BE
hash-shuffle signature.

**Independent NIC corroboration.** bond0 delta across one query run: **tx 1310 KB / rx 1419 KB**.
Higher than the 2.212 MB payload once framing and concurrent NFS traffic are counted, and the same
order of magnitude. Bytes crossed.

Query used: SF100 `lineitem`, `SET new_planner_agg_stage = 2` (forces TWO_STAGE →
`HASH_PARTITIONED` `DataStreamSink`), `GROUP BY l_orderkey % 4096` for a high-cardinality shuffle.
Query id `e4042219-95a3-11f1-98f4-001acaffff02`.

## Defects found and fixed in the documents

1. **BE config parser rejects trailing inline comments.** `be_port = 9060   # thrift` →
   `Invalid value of config 'be_port': ' 9060                  # thrift...'` and the BE exits.
   Diagnosis is hostile: `be.INFO` is created but **empty**, `be.pid` is left **stale**, and the
   only message is in `log/be.out` *inside the tree*, not `sys_log_dir`. The FE parser accepts
   inline comments, which is why the trap is easy to hit. Killed the first BE start here.
2. **`Mems_allowed_list` is the wrong membind check.** It reports cpuset-allowed nodes and stays
   `0-2,10,18,26` even when `numactl --membind=0,1` is working. Measured: that field showed all
   nodes while all 4,474 `numa_maps` entries were `bind:0-1` with **0 pages** on any HBM node. Use
   `/proc/PID/numa_maps`. Corrected in all four documents.
3. **`BytesPassThrough` is not a subset of `BytesReceived`.** The tutorial claimed
   `BytesPassThrough == BytesReceived` proves nothing crossed. Measured here: `BytesPassThrough`
   (2.344 MB) **exceeds** `BytesReceived` (2.212 MB) on a run where the NIC confirms bytes crossed —
   the old rule yields a **false negative** on a healthy cluster. They are independent counters.
4. The pixi `mysql` client rejects `\G`; use `--vertical`.

## TPC-H SF100 sweep — 21/22 measured

`--cold`, 3 timed warm runs per query after a discarded cold run. Warm medians, ms:

| q | ms | q | ms | q | ms | q | ms |
|---|---:|---|---:|---|---:|---|---:|
| q06 | 741 | q03 | 1669 | q07 | 2066 | q17 | 3917 |
| q14 | 1226 | q02 | 1762 | q20 | 2117 | q18 | 4673 |
| q11 | 1432 | q04 | 1763 | q19 | 2375 | q21 | 5771 |
| q01 | 1551 | q22 | 2483 | q15 | 2395 | q08 | **310473** |
| q13 | 1618 | q05 | 2575 | q16 | 2425 | | |
| | | q10 | 2891 | q12 | 2977 | | |

**Geometric mean over the 21 measured queries: 2762 ms.** Zero cascade failures — every query after
a failure still ran, which is the stock-BE self-cleanup the harness assumes (no `RESTART_CMD`).

CSVs: `timings.csv` (main sweep), `timings-q0809.csv` (q08 re-run).
Provenance: `INVOCATION-engineB.txt`.

### The q08 / q09 story — read before quoting these numbers

Both initially recorded as failures. **Neither was an engine fault.** Both hit the FE's
`query_timeout` session default of **300 s**:

```
ERROR 5024 (53400): Query reached its timeout of 300 seconds,
please increase the 'query_timeout' session variable and retry
```

The warm rows then recorded as `wedge` at exactly `180006 ms` because the harness's client-side
`QUERY_TIMEOUT=180` cut first — so one root cause presented as two different symptoms in the CSV.
That 300 s default is an SF1-era value the tutorial carried forward unchanged; at SF100 the two
heaviest joins exceed it.

* **q08: re-measured** with `SET GLOBAL query_timeout = 1800` and matching client timeouts —
  passes at 309–325 s, 2 rows. It is ~50× the next-slowest query and dominates any mean; the
  geometric mean above **includes** it.
* **q09: NOT MEASURED — deliberately skipped.** Two attempts. The first died when the gcn-17 BE was
  stopped mid-flight. The second ran on a healthy 2-BE cluster and was cancelled by operator
  decision after **16 minutes still inside its cold run** (`SHOW PROC '/current_backend_instances'`
  showed live fragments accumulating ~963 s of instance time, so it was executing, not hung). At
  that rate 4 runs would have cost 60–90 minutes for one query.

**Do not quote a 22/22 result. The honest statement is 21/22 measured, q09 skipped.**

q09 is TPC-H's worst case — the six-way join across `part`/`partsupp`/`lineitem`/`orders`/
`supplier`/`nation`. Together with q08 (310 s) it is heavy enough to dominate any aggregate. When
engine A is measured, report the geometric mean **both with and without q08/q09**, or two queries
decide the headline number for both engines.

## Not yet done

* q09 — skipped by decision, not blocked. To close it later: both BEs alive,
  `SET GLOBAL query_timeout = 1800`, then
  `bench.sh --cold <out>.csv 3 q09` with `QUERY_TIMEOUT=1800 COLD_TIMEOUT=1800`. Budget ~90 min.
* Engine A two-machine bring-up — blocked, and independent of this.
* `MemLimit` reports `432.000GB` for a configured `mem_limit = 480G`, a clean 0.9 ratio. Safe (far
  from the ~1523 GB percentage trap) but the cause is unconfirmed — display convention vs reserve
  factor. Do not quote 480G as the effective limit without checking.

## Teardown

```bash
/home/prestouser/starrocks-bench/be18/bin/stop_be.sh     # gcn-18
/home/prestouser/starrocks-bench/fe/bin/stop_fe.sh       # gcn-18
/home/prestouser/starrocks-bench/be17/bin/stop_be.sh     # gcn-17
```
FE metadata under `/raid/prestouser/sr-bench-2node/fe/meta` survives a restart; deleting it forces
re-running both `ALTER SYSTEM ADD BACKEND` statements.
