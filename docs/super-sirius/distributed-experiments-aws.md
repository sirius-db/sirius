# Distributed experiments on AWS g7e instances

How to take one AWS `g7e` instance from launch to a checked TPC-H benchmark through the gateway.
An instance is a machine without Slurm and without NVLink. This document covers what AWS adds
(launch, storage, build, dataset, config) and what the numbers mean there; the launch script and
the checks are those of
[Distributed experiments without Slurm](distributed-experiments-no-slurm.md). Read
[Distributed Quick Start](distributed-quickstart.md) first for the commands and the
`sirius.distributed` settings.

Nothing below hard-codes a GPU count, core count or memory size. One rank runs per GPU, on one
GPU as on eight, so the numbers compare.

**Tested:** `g7e.24xlarge` only (4 x RTX PRO 6000 Blackwell 96 GB, 96 vCPUs, 999 GiB RAM,
2 x 3.5 TB instance-store NVMe, x86_64, driver 595.91.07), TPC-H SF3000 with 1, 2 and 4 ranks,
at commit `e11487d2`. That commit started the gateway and the ranks by hand. The launch script
and `scripts/compare_answers.py` are newer: steps 5 to 7 were run as written here on a GB200
machine at SF100, not on an instance. Other `g7e` sizes were not run: check step 1 on yours.

## Rules

1. **The data is on ephemeral storage.** Instance-store NVMe is erased when the instance is
   stopped or terminated; a reboot keeps it. Never stop the instance while you need the data,
   and copy results off before terminating.
2. **Know when the instance ends.** If it was launched with a self-shutdown timer,
   `cat /run/systemd/shutdown/scheduled` shows it. Cancel it with `sudo shutdown -c`, or move it
   with `sudo shutdown -c && sudo shutdown -h +<minutes>`.
3. **One heavy job at a time.** Dataset generation, a CPU reference run and a benchmark compete
   for the same disks and cores. A benchmark timed next to either is not a result.
4. **Start narrow:** Q1, Q6 and Q12, cold and warm, checked. Run all 22 only after those match.
5. **Keep the default scan backend.** Do not set `scan_manager.backend: kvikio`.
6. **Leave no rank behind.** After each run
   `nvidia-smi --query-compute-apps=pid --format=csv,noheader` must print nothing. If it does
   not, [stop what is left](distributed-experiments-no-slurm.md#stop-what-a-killed-run-left).

## Launch

Any way of launching works. What the instance needs:

| Setting | Value |
|---|---|
| AMI | A Deep Learning Base AMI with the NVIDIA driver, for the region (tested: "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260818"). Sirius's CUDA 13 environment needs driver 580 or newer |
| Root volume | 200 GB. The build and caches go on the instance store, but leave room for the OS |
| Purchase | On-demand. A spot reclaim mid-run wastes the build and the data |
| Network | SSH (22) from your address only |

- `g7e` capacity is zone-limited, and `InsufficientInstanceCapacity` in every zone of a region is
  common. Retry another zone or region; the AMI, key pair and security group are per region.
- The "Running On-Demand G and VT instances" quota counts vCPUs: the instance's vCPU count must
  fit (`g7e.24xlarge` needs 96).

## 1. Look at the machine

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m                      # which GPUs share a PCIe bridge
nvidia-smi topo -p2p r                  # which GPU pairs can copy directly
lscpu | grep -E 'Architecture|^CPU\(s\)|Model name|Socket|NUMA node\(s\)'
free -g
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL
df -h
```

Write down the GPU count, vCPUs, RAM and the instance-store size. Stop if there is no NVIDIA
driver, or no instance-store NVMe ("Amazon EC2 NVMe Instance Storage" in `lsblk`).

On `g7e.24xlarge` the GPUs are two PCIe pairs (0-1 and 2-3). Peer-to-peer copies work inside a
pair only; data between pairs is staged through host memory. No `g7e` size has NVLink, so skip
the fabric check and the NIXL pairs test of the other documents.

## 2. Storage

The Deep Learning AMI mounts all instance-store disks as one striped LVM volume at
`/opt/dlami/nvme`. If your AMI did not, show `lsblk` to whoever owns the machine before
formatting anything, and only format unmounted instance-store devices that hold no file system.
Never touch the root volume or an attached EBS data volume.

```bash
export WORK=/opt/dlami/nvme/work
sudo mkdir -p "$WORK" && sudo chown "$USER" "$WORK"
mkdir -p "$WORK"/{cache/pixi,cache/rattler,logs,runs,spill,duckdb_tmp,ref}
```

Everything large goes under `$WORK`. Space needed:

| What | SF3000 | SF10000 |
|---|---|---|
| Checkout, build and caches | about 120 GB | same |
| Dataset | 802 GB | 2.7 TB |
| Spill | up to 1 TB (none was used at SF3000) | not run |

Check `df -h "$WORK"` against this before choosing a scale factor: smaller `g7e` sizes have less
instance storage.

## 3. Build

```bash
curl -fsSL https://pixi.sh/install.sh | bash
cat > "$WORK/env.sh" << EOF
export WORK=$WORK PIXI_CACHE_DIR=$WORK/cache/pixi RATTLER_CACHE_DIR=$WORK/cache/rattler
export CARGO_HOME=$WORK/cargo RUSTUP_HOME=$WORK/rustup
export PATH=\$HOME/.pixi/bin:\$CARGO_HOME/bin:\$PATH
EOF
. "$WORK/env.sh"
curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path      # the dataset generator needs cargo

cd "$WORK" && git clone --branch experimental/distributed-sirius https://github.com/sirius-db/sirius.git
cd sirius && git submodule update --init --depth=1 --jobs 5 \
  duckdb substrait cucascade sirius-duckdb/duckdb sirius-duckdb/extension-ci-tools
tmux new-session -d -s build "pixi run build-distributed > $WORK/logs/build.log 2>&1; echo \$? > $WORK/logs/build.rc"
```

- A non-interactive `ssh host '<command>'` does not read `~/.bashrc`: start every remote command
  with `. /opt/dlami/nvme/work/env.sh`.
- Run long jobs in a detached `tmux` session with a log and an exit-code file, as above. Do not
  hold an SSH connection open for them.
- The build took **905 s** on 96 vCPUs (NIXL, Sirius, gateway), with no change needed on x86_64.
  That run cloned every submodule (`--recursive`), which the 120 GB above includes; the five
  named here are the ones the build uses.
- The rank and the client use `gateway/build/release/duckdb`. The plain DuckDB without Sirius,
  for reference answers, is `sirius-duckdb/build/release/duckdb` (`build/release/duckdb` is a
  directory).

## 4. Dataset

Use an existing dataset if there is one (one folder per table). Otherwise:

```bash
cd "$WORK/sirius"
tmux new-session -d -s datagen "pixi run bash test/tpch_performance/generate_tpch_data.sh 3000 \
  --format parquet --output $WORK/tpch_sf3000 > $WORK/logs/datagen.log 2>&1; echo \$? > $WORK/logs/datagen.rc"
```

SF3000 took **464 s** (802 GB) on 96 vCPUs, including building the generator. Row-group size
changes scan results by 2x and more, so keep row groups near 1 million rows and report the
layout with every number.

- The script's defaults give about 1.06 million rows per row group and scale the file count with
  the scale factor (lineitem: 180 files at SF3000).
- For a fixed layout, call the generator (`test_datasets/tpchgen-rs/target/release/tpchgen-cli`,
  built by the script) directly. It takes a byte target per row group, not a row count, and
  writes `floor(rows_per_file x avg_row_bytes / target) + 1` equal row groups per file. SF10000
  with lineitem 600 files x 96 row groups, orders 174 x 83, partsupp 109 x 70, part 16 x 120,
  customer 32 x 45 and supplier 2 x 48 took **1,026 s** (2.7 TB), with `--parts` as the file
  count and `--parquet-row-group-bytes` of 67015706, 78369905, 148900336, 73221757, 176966292
  and 172631578 for those tables, and the other options as the generator's
  `scripts/generate_tpch.py` passes them.

Check the data before benchmarking, from the Parquet metadata (seconds, no full scan), in
`pixi run sirius-duckdb/build/release/duckdb -no-agent`:

```sql
-- rows and files per table: lineitem about 6 million x SF, orders 1.5 million x SF
SELECT sum(num_rows), count(*) FROM parquet_file_metadata('<DATA>/lineitem/*.parquet');
-- layout
SELECT count(DISTINCT file_name) AS files, count(*) AS row_groups, median(row_group_num_rows) AS rows
FROM (SELECT DISTINCT file_name, row_group_id, row_group_num_rows
      FROM parquet_metadata('<DATA>/lineitem/*.parquet'));
-- per column: no nulls, and min differs from max (o_shippriority is always 0)
SELECT path_in_schema, sum(stats_null_count), min(stats_min_value) = max(stats_max_value)
FROM parquet_metadata('<DATA>/lineitem/*.parquet') GROUP BY ALL;
```

## 5. Config and queries

`$OUT/config.yaml` configures every rank. Use a new `$OUT` for every run.

```bash
. /opt/dlami/nvme/work/env.sh
export CHECKOUT=$WORK/sirius DATA=$WORK/tpch_sf3000 OUT=$WORK/runs/<name>
ranks=$(nvidia-smi -L | wc -l)                                        # one per GPU
host=$(($(free -g | awk '/^Mem/ { print $2 }') * 6 / 10 / ranks))     # GiB of host memory per rank
mkdir -p "$OUT" && cat > "$OUT/config.yaml" << EOF
sirius:
  memory:
    host: { capacity_bytes: ${host}Gi }
    disk: { disk_id: 0, capacity_bytes: 1000000000000, downgrade_root_dirs: "$WORK/spill" }
  executor:
    pipeline: { num_threads: 4 }
    scan_manager: { num_threads: 32, uring_n_reactors: 4, cache: { mode: none } }
  operator_params:
    max_dynamic_filter_bloom_bytes_per_gpu: 2Gi
    max_build_probe_resident_bytes: 32Gi
    max_build_hash_table_bytes: 32Gi
EOF
```

| Key | How to choose | Tested |
|---|---|---|
| `host.capacity_bytes` | Per rank: `0.6 x RAM / ranks`, which leaves room for the OS and the page cache. Too large gets ranks killed by the kernel | 150Gi with 4 ranks on 999 GiB |
| `pipeline.num_threads` | A rank runs this many scan tasks at once. With 1 or 2 ranks, 4 threads per rank cannot keep the disks busy: use 16. With 4 ranks, 4 already saturate the disks | 4, 8, 16 |
| `scan_manager.num_threads` | 32 per rank, as on NVL nodes. That is 128 threads for 4 ranks on 96 vCPUs; it worked, and fewer was not tested | 32 |
| `scan_manager.cache.mode` | `none` reads with `O_DIRECT` and is bound by the disks. `os` uses the page cache: much faster once the columns are cached, and it scales with ranks. See [What we learned](#what-we-learned) | `none`, `os` |
| Bloom, build/probe, hash table | 2Gi and 32Gi: half the values used on NVL nodes, for 96 GB GPUs | all 22 queries at SF3000 |

Change the config only if a run fails, one change at a time, and record each change and why. The
Bloom cap is set in the file only; a `SET` fails. Keep the `exchange` and `ucx` settings at
their defaults.

Write `$OUT/run.sql` with the snippet of
[step 4](distributed-experiments-no-slurm.md#4-config-and-queries) of the other document. Set
its `frac` (Q11's fraction, `0.0001 / SF`) to `0.0000000333333` at SF3000 and to `0.00000001` at
SF10000, and its query list to `1 6 12` for the first run.

## 6. Run

Save the script of [step 5](distributed-experiments-no-slurm.md#5-run) of the other document as
`$OUT/run.sh`. One run of it is one arm, with a fresh gateway and fresh ranks. Then, in the
shell of step 5 here:

```bash
sed -i "s/MIN_RANKS=4 /MIN_RANKS=$ranks /" "$OUT/run.sh"
tmux new-session -d -s run ". $WORK/env.sh; export CHECKOUT=$CHECKOUT DATA=$DATA OUT=$OUT
  cd $CHECKOUT && pixi run bash $OUT/run.sh > $OUT/run.out 2>&1; echo \$? > $OUT/run.rc"
```

The run is over when `$OUT/run.rc` exists; it holds 0 unless the script stopped early, and
`run.out` says why. What the script writes to `$OUT` is listed in the other document.

- **Fewer GPUs**, to compare GPU counts on one machine: see
  [Variations](distributed-experiments-no-slurm.md#variations), and set `ranks` to match before
  step 5. For 2 GPUs use two that share a PCIe bridge (`0,1` on `g7e.24xlarge`).
- A one-GPU instance runs the same way with one rank.
- Telemetry (`telemetry_data/` in the checkout, one folder per rank) is on by default and holds
  per-task timings. Sessions of 26 to 75 s that were killed left their telemetry files empty;
  sessions of 150 s and more had them.

## 7. Check the run

A time without a checked answer is not a result. Run the checks of
[step 6](distributed-experiments-no-slurm.md#6-check-the-result) of the other document, with two
differences.

**The exchange path.** There is no NVLink to count. This prints how many rank pairs use each
path:

```bash
grep -h "exchange: rank .* -> rank" "$OUT"/rank-logs/*/sirius_*.log | sed 's/.*): //' | sort | uniq -c
```

Each rank's log has a line per peer, and a warning that is expected without NVLink:

```text
exchange: rank 0 has no fabric memory (cuMemCreate: CUDA_ERROR_NOT_PERMITTED (is an IMEX channel open?)); ...
exchange: rank 0 (GPU 0000:52:00.0) -> rank 1 (GPU 0000:53:00.0, NVLink domain none): NIXL
exchange: rank 0 (GPU 0000:52:00.0) -> rank 2 (GPU 0000:63:00.0, NVLink domain none): host-staged through pinned shared memory
```

`NIXL` is a peer-to-peer pair (UCX `cuda_ipc`); `host-staged through pinned shared memory` is a
pair that cannot copy directly. UCX's defaults here are `UCX_TLS=cuda_copy,cuda_ipc,tcp,self` on
the instance's network interface.

**Reference answers.** They come from the plain DuckDB on the CPU, same data and query texts,
while no rank is running. Compute them once per dataset, one process per query, so that one
failure does not lose the rest:

```bash
frac=0.0000000333333                                    # as in run.sql
q=$CHECKOUT/test/tpch_performance/tpch_queries/orig
cd "$CHECKOUT" && for n in $(seq 1 22); do
  {
    echo ".mode csv"
    echo "SET memory_limit='800GB'; SET temp_directory='$WORK/duckdb_tmp';"
    for t in customer lineitem nation orders part partsupp region supplier; do
      echo "CREATE VIEW $t AS FROM read_parquet('$DATA/$t/*.parquet');"
    done
    echo ".output $WORK/ref/q$n.csv"
    sed "s/\* 0\.00001$/* $frac/" "$q/q$n.sql"
  } | pixi run sirius-duckdb/build/release/duckdb -no-agent
done
python3 scripts/compare_answers.py "$OUT/client.txt" --ref-dir "$WORK/ref" --ref-pattern "q{n}.csv" \
  --ties 3:revenue,o_orderdate --ties 18:o_totalprice,o_orderdate
```

Size `memory_limit` to about 80% of RAM. At SF3000 on 96 vCPUs all 22 took 531 s; Q21 alone
took 216 s.

**LIMIT ties in Q3 and Q18 are expected, not rare.** TPC-H's generator repeats its random
streams after 2^31 - 2 orders, so above about SF1430 whole orders repeat with identical line
items (order keys about 2^33 apart). Rows that tie at the cut can differ between Sirius and the
reference, and between a cold and a warm run. `--ties` accepts a difference only when the
differing rows have the order-column values of the last row.

**Exchange bandwidth** on one machine (the two-node pairs test does not apply):

```bash
GPUS=0,1 TEST='[nixl_bandwidth]' pixi run bash scripts/distributed_test_nixl.sh 2    # one pair
TEST='[nixl_bandwidth]' pixi run bash scripts/distributed_test_nixl.sh 4             # all to all
grep BANDWIDTH /tmp/distributed_nixl_rank_*.log
```

Expect about 40 to 55 GB/s per pair on PCIe Gen5, against about 700 GB/s for an NVLink pair.
Each reading moves under 4 GB, so repeat it; single readings as low as 12 GB/s were seen.

## Reference numbers

`g7e.24xlarge`, TPC-H SF3000, lineitem in 180 files with row groups of about 1.06 million rows,
the config above (`pipeline.num_threads: 4`, `cache.mode: none`), one run each, every answer
checked against DuckDB.

| Ranks | Queries | Cold s | Warm s |
|---|---|---|---|
| 4 | all 22 | 270.9 | 269.7 |
| 4 | Q1 - Q7 | 72.0 | 71.8 |
| 2 | Q1 - Q7 | 74.9 | 74.5 |

Slowest queries on 4 ranks (warm): Q21 28.9, Q9 22.4, Q8 20.1, Q17 19.8 s. Nothing spilled to
disk.

Raw GPU copy speeds on this machine: pinned host to GPU 55.6 GB/s, GPU to host 57.2, GPU to GPU
inside a pair 53 (peer access on), across pairs 43.

## What we learned

**At SF3000 on this machine the benchmark measures the disks, not the GPUs.** Going from 2 to 4
ranks made Q1 - Q7 4% faster. The cause, with the evidence:

1. **The queries are scan-bound.** In the task telemetry, scan pipelines hold all but 0.5 s of
   each query's compute time, and each query ends within 0.1 s of the slowest rank's scan.
2. **The disks have a fixed ceiling of about 13.5 GB/s** (two instance-store NVMe, striped;
   `dd` with `O_DIRECT`, the same with 16 and with 64 parallel readers).
3. **`cache.mode: none` reads with `O_DIRECT`, so every scan hits that ceiling.** Disk read rate
   during Q6 and Q1, sampled each second from `/proc/diskstats`:

   | Setup | Disk read rate (median) | Q6 warm s | Q1 warm s |
   |---|---|---|---|
   | 1 rank, 4 pipeline threads | 10.3 GB/s | 10.25 | 14.88 |
   | 1 rank, 16 pipeline threads | 14.0 GB/s | 9.26 | 11.01 |
   | 2 ranks | 13.1 GB/s | 9.80 | 12.19 |
   | 4 ranks | 13.0 GB/s | 9.52 | 11.44 |

   One rank with enough threads reads as fast as four. Q6 needs 105 GB of column data: at
   13.5 GB/s that is 7.8 s before any GPU work.
4. **With the disks out of the way, ranks scale.** The same queries with `cache.mode: os`, after
   the columns are in the page cache (the disks read nothing):

   | Ranks | Q6 warm s | Q1 warm s |
   |---|---|---|
   | 1 | 16.89 | 19.68 |
   | 2 | 8.22 | 10.04 |
   | 4 | 4.69 | 5.66 |

5. **One rank is starved.** With several ranks reading, one of them gets about a quarter of the
   others' share of the disk (about 1 GB/s against 4 GB/s each on 4 ranks) until they finish,
   and the query waits for it. Not diagnosed.
6. **A second ceiling sits behind the first.** With the page cache and 16 pipeline threads, one
   rank ran Q6 in 4.66 s and four ranks in 4.49 s. Serving lineitem from a `tmpfs` copy gave
   uneven results (Q6: 5.2 s on 1 rank, 9.5 s on 2, 5.6 s on 4). Not diagnosed; CPU use stayed
   under 20% of the cores, and GPU utilisation under 50%.

What to do about it:

- **Report the storage with every number**: the read ceiling, the cache mode, and whether the
  data was cached. A scan-heavy result on this hardware says little about GPUs.
- **Measure the ceiling first**: the bytes this reads, divided by its seconds. A query's floor
  is its column bytes divided by the ceiling.

  ```bash
  time (ls $DATA/lineitem/*.parquet | head -16 |
    xargs -P 16 -I{} dd if={} of=/dev/null bs=16M iflag=direct count=160 status=none)   # 16 x 2.5 GiB
  ```

- **To measure GPU scaling, take the disks out**: `cache.mode: os` with a warm-up run, on a
  dataset whose scanned columns fit in RAM beside the ranks' host memory. Otherwise use a scale
  factor or storage where the query's column bytes divided by the read ceiling is small.
- **With 1 or 2 ranks, raise `pipeline.num_threads` to 16**, or the ranks cannot reach the disk
  ceiling (Q1 on 1 rank: 14.9 s with 4 threads, 11.0 s with 16).
- **DuckDB on the CPU is not a like-for-like baseline here**: it reads through the page cache,
  which is why it ran Q6 in 4.3 s against 9.4 s for Sirius with `O_DIRECT`.

## Known failures

Failures of the launch are in
[When it goes wrong](distributed-experiments-no-slurm.md#when-it-goes-wrong) of the other
document. On an instance, also:

| Symptom | Cause and fix |
|---|---|
| `InsufficientInstanceCapacity` at launch | No free `g7e` in that zone: try another zone or region |
| `pixi: command not found` over `ssh host '...'` | Non-interactive shells skip `~/.bashrc`: source `env.sh` first |
| `build/release/duckdb: Is a directory` | The plain DuckDB binary is `sirius-duckdb/build/release/duckdb` |
| Adding ranks does not make scans faster | The disks are the limit: see [What we learned](#what-we-learned) |
| Q3 or Q18 differ from the reference | LIMIT tie: pass `--ties` |
| Q11 returns no or too few rows | Fraction must be `0.0001 / SF` |
| Ranks killed mid-query | Host tier too large for the RAM: lower `host.capacity_bytes` |
| Telemetry folders are empty | Seen for sessions under about 75 s that were killed; the cause was not confirmed |

## Report

The instance type and what step 1 found; the commit and build time; the config and every change
to it; the dataset path, whether it was found or generated, and its lineitem layout; the storage
read ceiling and cache mode; cold and warm seconds per query and in total; how each answer was
checked; the exchange paths the rank logs report; what failed or was not run.
