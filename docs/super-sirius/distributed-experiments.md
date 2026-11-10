# Distributed experiments on NVL4 and NVL72 nodes

How to run a distributed Sirius benchmark on GB200 NVLink nodes under Slurm and get numbers
that can be trusted. Without Slurm, follow
[Distributed experiments without Slurm](distributed-experiments-no-slurm.md) instead. Read
[Distributed Quick Start](distributed-quickstart.md) first for the commands and the
`sirius.distributed` settings.

| System | What it is | Ranks |
|---|---|---|
| NVL4 | One node: 4 GPUs, 2 Grace sockets | 4, one per GPU |
| NVL72 | One rack: 18 such nodes in one NVLink domain | 4 per node |

Every run goes through the gateway, on one node as on several, so the numbers compare: a
gateway, one rank per GPU, and `sirius-client` for the SQL. `scripts/sirius_slurm.sbatch` starts
the gateway and the ranks ([Run](#run)). Ranks on different nodes exchange over NVLink (MNNVL).
CPU fallback is off: a query Sirius cannot run on the GPU fails. Tested with 4, 8 and 16 ranks.
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

Needs Git and [Pixi](https://pixi.sh/latest/installation/) 0.78 or newer on `PATH`, and under
Slurm its client commands (export `SLURM_CONF` if `sinfo` finds no configuration). What the
machine needs is in [docs/README.md](../README.md#requirements).

The repository holds no cluster names or paths. Export these first:

| Variable | What |
|---|---|
| `PARTITION`, `NODES` | The GPU partition, and idle nodes, comma-separated (`sinfo -N -o '%N %T %E'`) |
| `CHECKOUT` | Where the Sirius checkout goes, on a file system every node mounts |
| `DATA` | The Parquet dataset, one folder per table, same path on every node |
| `OUT` | A new run folder, for `config.yaml`, `run.sql` and the output |

`$HOME` must be shared by the nodes and have free space: the build puts about 8 GB there
(Pixi's package cache; `PIXI_CACHE_DIR` moves it) and the gateway writes
`~/.sirius/gateways/<job id>/`. With `$HOME` at its quota, jobs fail within seconds.

```bash
git clone --branch experimental/distributed-sirius https://github.com/sirius-db/sirius.git "$CHECKOUT"
cd "$CHECKOUT" && git submodule update --init --depth=1 --jobs 5 \
  duckdb substrait cucascade sirius-duckdb/duckdb sirius-duckdb/extension-ci-tools
pixi run build-distributed            # once: NIXL, Sirius with NIXL, the gateway
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

`$OUT/config.yaml` configures every rank. The SF10000 config, tuned on 1 node and used
unchanged on 2, and for a first check at SF100:

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

One arm is one Slurm job with fresh ranks: start the gateway, send the SQL, cancel the job. As a
script:

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
SIRIUS_CLUSTER=$job pixi run sirius-client < "$OUT/run.sql" > "$OUT/client.txt" 2>&1
sleep 5                                              # the ranks flush their logs every 3 s
scancel "$job"
```

If `rank exited` keeps repeating in `job.out`, the ranks die at start (usually the config):
cancel the job and read the rank's log.

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

## Run without Slurm

On a machine Slurm does not schedule, such as a standalone NVL4 node, follow
[Distributed experiments without Slurm](distributed-experiments-no-slurm.md): the same config,
`run.sql` and checks, from an empty account, with one script that starts the gateway and the
ranks, and stops them.

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

GB200, 4 ranks per node, all 22 TPC-H queries, seconds cold / warm, every answer checked, one
run each.

| Scale | Lineitem row groups | 1 node | 2 nodes | 4 nodes |
|---|---|---|---|---|
| SF10000 | 1M rows | 202.4 / 197.2 | 111.1 / 107.3 | |
| SF10000 | 115k rows | 452.2 / 413.8 | | |
| SF30000 | | | 406.4 / 389.3 | 220.7 / 203.5 |

The SF30000 runs predate downstream-first priority. SF100000 on 4 nodes has no full result:
best per query, cold, Q18 28.8, Q3 27.3, Q9 59.8, Q21 86.3, Q1 35.4 s.

## Past 16 ranks

Not run yet. Grow in steps (4, 8, then 18 nodes) with Q1 and Q9 before any full run, and check
the fabric lines in every rank's log at each step. Each rank's exchange rings take
`16 MiB × (32 + 16 × peers)` of GPU memory: 4.25 GiB with 16 ranks, 18.3 GiB with 72. On a full
rack, `exchange.recv_frames_per_peer: 8` halves that for about 10% less throughput per pair.

## Known failures

| Symptom | Cause and fix |
|---|---|
| Job fails in seconds, no output | `$HOME` at quota, or a stale mount on the node: run the `ls` check |
| `gateway <cluster> already runs (lock ...)` | Another gateway has that name: set another `SIRIUS_CLUSTER` |
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
