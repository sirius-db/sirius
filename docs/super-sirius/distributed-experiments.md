# Distributed experiments on NVL4 and NVL72 nodes

How to reproduce a TPC-H benchmark of distributed Sirius and get numbers that can be trusted.
Start here, then follow the document for your hardware:

| Read | For |
|---|---|
| [Distributed Quick Start](distributed-quickstart.md), first | What the gateway, node and client commands do, the `sirius.distributed` launch settings, and the launch errors. This document does not repeat them |
| This document | GB200 NVLink nodes that Slurm schedules (NVL4, NVL72) |
| [Distributed experiments without Slurm](distributed-experiments-no-slurm.md) | A machine Slurm does not schedule, such as a standalone NVL4 node: the same config, `run.sql` and checks, with one script that starts and stops the gateway and the ranks |
| [Distributed experiments on AWS g7e instances](distributed-experiments-aws.md) | An AWS instance (no Slurm, no NVLink, data on local NVMe): launch, storage, build, dataset generation, and a config for 96 GB GPUs |

| System | What it is | Ranks |
|---|---|---|
| NVL4 | One node: 4 GPUs, 2 Grace sockets | 4, one per GPU |
| NVL72 | One rack: 18 such nodes in one NVLink domain | 4 per node |

Every run goes through the gateway, on one node as on several, so the numbers compare: a
gateway, one rank per GPU, and `sirius-client` for the SQL. Ranks on different nodes exchange
over NVLink (MNNVL). Tested with 4, 8 and 16 ranks.
Last followed from an empty account on 2026-11-10: clone, build, and SF10000 on one node.

## Rules

1. **On a node Slurm schedules, use Slurm for everything.** Start it with `sbatch` or `srun`,
   from a host that is not in the job. Never `ssh` to such a node or leave a process on one:
   where the Slurm prolog checks for stray user processes or a full `/`, the next job fails and
   the node is drained. Run without Slurm only on machines Slurm does not schedule.
2. **One job per node at a time.** After a job whose ranks were killed, wait until the node is
   `idle`, then 60 s more.
3. **Ask before holding 4 or more nodes.**
4. **Start narrow:** the few queries the question is about, one cold and one warm run. Run all
   22 only for a number that will be quoted.
5. **Keep the default scan backend.** `scan_manager.backend: kvikio` on GPFS stalled every rank
   and left the node's file system client unusable.

## Setup

Needs Git, [Pixi](https://pixi.sh/latest/installation/) 0.78 or newer and the Slurm client
commands on `PATH` (export `SLURM_CONF` if `sinfo` finds no configuration), and the quick
start's [prerequisites](distributed-quickstart.md#prerequisites). What the machine needs is in
[docs/README.md](../README.md#requirements).

The repository holds no cluster names or paths. Export these first:

| Variable | What |
|---|---|
| `PARTITION`, `NODES` | The GPU partition, and idle nodes, comma-separated (`sinfo -N -o '%N %T %E'`) |
| `CHECKOUT` | Where the Sirius checkout goes, on a file system every node mounts |
| `DATA` | The Parquet dataset, one folder per table, same path on every node |
| `OUT` | A new run folder, for `config.yaml`, `run.sql` and the output |

`$HOME` needs free space: the build puts about 8 GB there (Pixi's package cache;
`PIXI_CACHE_DIR` moves it). With `$HOME` at its quota, jobs fail within seconds.

```bash
git clone --branch experimental/distributed-sirius https://github.com/sirius-db/sirius.git "$CHECKOUT"
cd "$CHECKOUT" && git submodule update --init --depth=1 --jobs 5 \
  duckdb substrait cucascade sirius-duckdb/duckdb sirius-duckdb/extension-ci-tools
pixi run build-distributed            # once: Sirius and the gateway
mkdir -p "$OUT"
```

The build takes 35 to 40 minutes on 144 cores and the checkout grows to 14 GB. `pixi run` may
warn that its cache is on a network file system: ignore it.

Before the first job on a set of nodes:

```bash
for n in ${NODES//,/ }; do              # every node sees the build and the data
  srun -p "$PARTITION" -w "$n" -N1 --time=2 bash -c \
    "hostname; ls -d $CHECKOUT/build/release/libsirius.so $DATA/lineitem"
done
srun -p "$PARTITION" -w "${NODES%%,*}" -N1 --gpus=4 --time=2 \
  bash -c 'nvidia-smi -q | grep -A4 "^ *Fabric$"'
```

An `idle` node can still have a stale mount, so do run the `ls`. The fabric check must print
`State: Completed` and the same `CliqueId` and `ClusterUUID` on every node; otherwise the
exchange goes over the network. On a new cluster, also measure the exchange alone:
`scripts/distributed_test_nixl_pairs.sh <node1>,<node2> "$OUT/pairs"` writes `summary.md`. One
GPU pair should move about 700 GB/s; under 150 GB/s, UCX is not using NVLink.

## Config

`$OUT/config.yaml` holds the engine settings of every rank ([Configuration](configuration.md)
explains the keys). The launch settings, the quick start's
[`sirius.distributed`](distributed-quickstart.md#configure) section, stay at their defaults: the
Slurm script names the gateway and counts the ranks itself. The SF10000 config, tuned on 1 node
and used unchanged on 2, and for a first check at SF100:

```yaml
sirius:
  memory:
    gpu:
      usage_limit_fraction: 0.8841
      downgrade_trigger_fraction: 0.9089
      downgrade_stop_fraction: 0.6786
    host: { capacity_bytes: 200Gi }
    disk: { disk_id: 0, capacity_bytes: 1000000000000, downgrade_root_dirs: "<spill folder>" }
  executor:
    pipeline: { num_threads: 7 }
    task_creator: { num_threads: 1 }
    downgrade: { num_threads: 2 }
    scan_manager: { num_threads: 9, uring_n_reactors: 1, cache: { mode: none } }
  operator_params:
    scan_task_batch_size: 901Mi
    hash_partition_bytes: 1067Mi
    concat_batch_bytes: 2552Mi
    sort_sample_bytes: 201Mi
    max_build_hash_table_bytes: 6358Mi
    max_build_probe_resident_bytes: 16Gi
    max_sort_partition_memory_fraction: 0.4330
    mark_join_build_switch_ratio: 10.1054
    dynamic_filter_domain_coverage_threshold: 0.6257
    dynamic_filter_keep_threshold: 0.8144
    max_dynamic_filter_bloom_bytes_per_gpu: 2577Mi
```

For SF30000 and SF100000 (2 and 4 nodes), only these were set:

```yaml
sirius:
  memory:
    host: { capacity_bytes: 180Gi }
    disk: { disk_id: 0, capacity_bytes: 1000000000000, downgrade_root_dirs: "<spill folder>" }
  executor:
    scan_manager: { num_threads: 32, uring_n_reactors: 4, cache: { mode: none } }
  operator_params:
    max_dynamic_filter_bloom_bytes_per_gpu: 8Gi
    max_build_probe_resident_bytes: 64Gi
    max_build_hash_table_bytes: 64Gi
```

- **Host tier** is per rank. A node has about 478 GiB per socket for 2 ranks: 200Gi works,
  250Gi and more got ranks killed by the kernel.
- **Exchange and UCX:** keep the defaults. Never list `bond0` with the rail NICs in
  `UCX_NET_DEVICES`.
- The Bloom cap is set in the file only; a `SET` fails.
- **Spill folder:** create it first, on a disk with free space.

## Run

One arm is one Slurm job with fresh ranks: start the gateway, send the SQL, cancel the job.
What `scripts/sirius_slurm.sbatch` starts is in the quick start's
[Slurm](distributed-quickstart.md#slurm) section. As a script:

```bash
export SIRIUS_CONFIG_FILE=$OUT/config.yaml
export SIRIUS_RANK_LOG_ROOT=$OUT/rank-logs           # one log folder per rank
export SIRIUS_PRIORITY_DOWNSTREAM_FIRST=1            # the reference numbers below use it
export SIRIUS_GATEWAY_STATEMENT_TIMEOUT=300          # seconds before a hung statement is cancelled
cd "$CHECKOUT" && git log -1 --oneline > "$OUT/env.txt"
job=$(sbatch --parsable -N"$(tr ',' '\n' <<< "$NODES" | wc -l)" -p "$PARTITION" -w "$NODES" \
  --time=40 -o "$OUT/job.out" scripts/sirius_slurm.sbatch)
until [ -s "$HOME/.sirius/gateways/$job/gateway.conf" ]; do
  squeue -j "$job" -h -o %T | grep -q . || { echo "job $job ended: see $OUT/job.out"; exit 1; }
  sleep 5
done
SIRIUS_CLUSTER=$job pixi run sirius-client -bail < "$OUT/run.sql" > "$OUT/client.txt" 2>&1
sleep 5                                              # the ranks flush their logs every 3 s
scancel "$job"
```

`-bail` stops the client at the first failed statement: without it, every statement of a run
whose ranks never start waits its turn to fail. If `rank exited` keeps repeating in `job.out`,
the ranks die at start (usually the config): cancel the job and read the rank's log.

`run.sql`, here for five queries:

```bash
q=$CHECKOUT/test/tpch_performance/tpch_queries/orig
{
  printf '.mode csv\n.timer on\n'
  for t in customer lineitem nation orders part partsupp region supplier; do
    echo "CREATE OR REPLACE VIEW $t AS FROM read_parquet('$DATA/$t/*.parquet');"
  done
  for n in 1 3 6 11 12; do                              # $(seq 1 22) for all
    for run in 1 2; do
      echo "SELECT 'Q$n run $run' AS marker;"
      sed 's/\* 0\.00001$/* 0.00000001/' "$q/q$n.sql"   # Q11's fraction, 0.0001 / SF: here SF10000
    done
  done
  echo "FROM sirius_gateway_status();"                  # every rank up, no last_error
} > "$OUT/run.sql"
```

- Each query runs twice in a row: "cold" is its first run in the job, "warm" the second. The
  first query of a job is slower still (Q6 at SF10000: 10.8 s as the first query, 6.4 s as the
  third), so keep the query order when comparing runs.
- At SF100000, footer statistics are per query: Q3 needs
  `SET parquet_footer_statistics = false`, Q9 needs the default (`true`).

## Check the run

**Answers.** Compare every answer with an independent engine on the same data and query texts
(DuckDB on CPU handles SF10000 in about 15 minutes with 1 TB of memory). The build's shell is a
plain DuckDB when started as `pixi run sirius-duckdb/build/release/duckdb < ref.sql`, with
`ref.sql` the same views and queries. Compare rows in any order, numbers within `1e-6` relative
or `0.01` absolute. Q3 and Q18 end in `ORDER BY ... LIMIT` and may return different rows that
tie at the cut: that is a tie, not an error, once the differing rows are shown to have the same
order-column values. `scripts/compare_answers.py` applies these rules:

```bash
python3 scripts/compare_answers.py "$OUT/client.txt" --ref-client "$OUT/ref.txt" \
  --ties 3:revenue,o_orderdate --ties 18:o_totalprice,o_orderdate
```

**Logs.** In `client.txt` each run is its marker, the answer, then the query's `Run Time` (the
second after the marker). Each rank's log in `rank-logs/<host>-<port>/` should show every peer
as:

```text
exchange: rank 0 (GPU ...) -> rank 5 (GPU ..., NVLink domain ...): NIXL over the NVLink fabric
exchange: query 24 on rank 0: sent 1.655 GB in 322 batches, received 1.602 GB in 301 batches
```

Anything other than `NIXL over the NVLink fabric` means that pair uses the network. The `sent`
lines, summed over ranks, give the bytes a query moved between ranks. Query numbers count every
statement: with the `run.sql` above, the first run is query 10 and each run adds 2. A
`Transparent execution fallback ... quack_nop_authorization(...)` line precedes every statement
and is not the query: a query on the GPU logs `physical plan replaced with GPU operator`. Rank
logs contain the gateway's token: remove it before sharing them.

**Dataset layout.** Row-group size changes the result by 2× and more, so report it with every
number. With small row groups a scan is bound by read requests, not bytes.

```sql
SELECT count(DISTINCT file_name) AS files, count(*) AS row_groups, median(row_group_num_rows) AS rows
FROM (SELECT DISTINCT file_name, row_group_id, row_group_num_rows
      FROM parquet_metadata('<DATA>/lineitem/*.parquet'));
```

## Reference numbers

GB200, 4 ranks per node, the sum of the 22 TPC-H query times in seconds.

| Scale | Scans | 1 node | 2 nodes | 4 nodes |
|---|---|---|---|---|
| SF10000 | GDS | 94.7 | 54.4 | 36.3 |
| SF30000 | GDS | 321.7 | 184.8 | 109.2 |
| SF100000 | GDS | 1,641.1 | 747.7 | 336.8 |
| SF10000 | io_uring | 237.1 | 128.0 | 72.2 |
| SF30000 | io_uring | 720.4 | 384.7 | 208.7 |
| SF100000 | io_uring | 3,263.0 | 1,508.9 | 725.8 |

- **GDS** is `scan_manager.backend: gds`, set up as in the [GDS runbook](gds-runbook.md). Its
  integration is not finished. It is faster than io_uring at the three scales by 2.50x, 2.24x
  and 1.99x on one node, 2.35x, 2.08x and 2.02x on two, and 1.99x, 1.91x and 2.15x on four.
- **io_uring** is the default scan backend: the config has no `scan_manager.backend` line.
- **Totals:** SF10000 and SF30000 run each query twice and take the faster run. SF100000 runs
  each query once.
- **SF100000, io_uring, four nodes:** in the single pass one node's storage reads stopped
  for about two minutes during Q5 and the pass took 849.7 s. Q5 run again alone on the same nodes
  took 48.2 s, which gives the 725.8 s of the table.
- **SF100000 on one node:** Q9 and Q13 do not fit in memory and spill to disk (about 2.5 TB and
  0.6 TB), so that run has spill compression on.
- The build of these runs carries the planner rewrites for Q2, Q11, Q13, Q17 and Q18.

These runs are on [accel-etl](https://accel-etl.nvidia.com/runs). A run there holds every pass,
so its totals are per pass and differ from the faster-run totals above. The SF10000 GDS run on
one node is a separate session with the standard query parameters. In the SF30000 io_uring runs
on two and four nodes each pass has one statement that paused for more than 90 s; the faster-run
totals leave those out.

| Scale | Scans | Nodes | Run | First pass (s) |
|---|---|---|---|---|
| SF10000 | GDS | 1 | [11807](https://accel-etl.nvidia.com/runs/11807/) | 99.4 |
| SF10000 | GDS | 2 | [11899](https://accel-etl.nvidia.com/runs/11899/) | 58.3 |
| SF10000 | GDS | 4 | [11900](https://accel-etl.nvidia.com/runs/11900/) | 39.5 |
| SF10000 | io_uring | 1 | [11808](https://accel-etl.nvidia.com/runs/11808/) | 245.3 |
| SF10000 | io_uring | 2 | [11901](https://accel-etl.nvidia.com/runs/11901/) | 136.0 |
| SF10000 | io_uring | 4 | [11902](https://accel-etl.nvidia.com/runs/11902/) | 78.1 |
| SF30000 | GDS | 1 | [11805](https://accel-etl.nvidia.com/runs/11805/) | 325.4 |
| SF30000 | GDS | 2 | [11903](https://accel-etl.nvidia.com/runs/11903/) | 198.1 |
| SF30000 | GDS | 4 | [11904](https://accel-etl.nvidia.com/runs/11904/) | 112.0 |
| SF30000 | io_uring | 1 | [11806](https://accel-etl.nvidia.com/runs/11806/) | 738.8 |
| SF30000 | io_uring | 2 | [11905](https://accel-etl.nvidia.com/runs/11905/) | 484.5 |
| SF30000 | io_uring | 4 | [11906](https://accel-etl.nvidia.com/runs/11906/) | 322.2 |
| SF100000 | GDS | 1 | [11810](https://accel-etl.nvidia.com/runs/11810/) | 1,641.1 |
| SF100000 | GDS | 2 | [11907](https://accel-etl.nvidia.com/runs/11907/) | 747.7 |
| SF100000 | GDS | 4 | [11908](https://accel-etl.nvidia.com/runs/11908/) | 336.8 |
| SF100000 | io_uring | 1 | [11809](https://accel-etl.nvidia.com/runs/11809/) | 3,263.0 |
| SF100000 | io_uring | 2 | [11909](https://accel-etl.nvidia.com/runs/11909/) | 1,508.9 |
| SF100000 | io_uring | 4 | [11910](https://accel-etl.nvidia.com/runs/11910/) | 725.8 |

### One node since the join-side commits

The numbers above are from a build before commit `35f5573d` (2026-11-18). Since that commit a
hash join holds its filtered side when the unfiltered scan on its other side does not fit in one
GPU, a small lookup table is joined after the aggregation, and a `+`, `-`, `*` expression over
DOUBLE columns is computed below the join (as it already was for DECIMAL). None of this has an
option to set. In TPC-H it changes Q9.

One node, GDS, the datasets, config and method of the table above, in seconds:

| Scale | 22 queries before | 22 queries after | Q9 before | Q9 after |
|---|---|---|---|---|
| SF30000 | 321.7 | 319.7 | 28.5 | 27.1 |
| SF100000 | 1,641.1 | 1,494.6 | 328.8 | 179.3 |

- **SF100000 Q9** has two modes: 179.3 s in the 22-query pass and 122.0 s alone an hour earlier on
  the same node, against 338.8 s for the earlier build run next to it. It now writes 44 to 436 GB
  to disk where it wrote about 2.5 TB. Q13 is unchanged and still writes about 0.6 TB.
- **The other 21 queries at SF100000** sum to 1,315.3 s against 1,312.4 s.
- **Answers** match the expected results for all 22 queries at both scales.
- **Not repeated:** SF10000, the io_uring runs, and every run on two and four nodes. Those cells
  of the table above are still the earlier build's.
- The runs used the same source as `35f5573d` apart from log levels, formatting and two unit tests.

## Past 16 ranks

Not run yet. Grow in steps (4, 8, then 18 nodes) with Q1 and Q9 before any full run, and check
the fabric lines in every rank's log at each step. Each rank's exchange rings take
`16 MiB × (32 + 16 × peers)` of GPU memory: 4.25 GiB with 16 ranks, 18.3 GiB with 72. On a full
rack, `exchange.recv_frames_per_peer: 8` halves that for about 10% less throughput per pair.

## Known failures

A `FAIL` line or a `rank N ...` message from the launch is in the quick start's
[Troubleshooting](distributed-quickstart.md#troubleshooting).

| Symptom | Cause and fix |
|---|---|
| Job fails in seconds, no output | `$HOME` at quota, or a stale mount on the node: run the `ls` check |
| Job fails its prolog, node drained | A user process outside Slurm, or `/` nearly full. An administrator resumes the node |
| Ranks killed mid-query | Host tier too large: use 200Gi |
| A peer is not on `NIXL over the NVLink fabric` | No IMEX channel, different NVLink domains, or `UCX_TLS` without `cuda_ipc`: keep the default `ucx` settings |
| Q3 or Q18 differ from the reference | LIMIT tie: check the order columns |
| Q11 returns no or too few rows | Fraction must be `0.0001 / SF` |
| SF100000 Q3 runs out of GPU memory | `SET parquet_footer_statistics = false` for it |
| SF100000 Q9 spills for minutes | Keep `parquet_footer_statistics` on for it |
| SF100000, 16 ranks: `exchange all-gather ... 15 of 16 ranks` | One rank never joined; seen once, not diagnosed. Rerun; if it repeats, take a stack of that rank with `srun --jobid <job> --overlap` |
| A statement hangs with idle ranks | Seen rarely (Q12, Q15 at SF10000). The statement timeout cancels it and the next statement restarts the ranks |

## Report

The commit, config and nodes; the dataset path and its lineitem layout; cold and warm seconds
per query; how each answer was checked; what failed or was not run.
