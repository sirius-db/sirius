# GDS runbook: distributed Sirius reading through GPUDirect Storage

How to run the ranks of [distributed Sirius](distributed-sirius.md) with `scan_manager.backend: gds`,
how to tell that the reads really went through GDS, and what to do when a storage NIC stops
delivering data. How the backend works is in [Scan](scan.md#gpudirect-storage-backend); every key
is in [Configuration](configuration.md#scan_managergds--gpudirect-storage-backend-srciogds).

The numbers here were measured in November 2026 on GB200 nodes (4 GPUs, four 400 Gb storage NICs
per node) reading Parquet from a GPFS file system over RDMA, with nvidia-fs 2.28 and cuFile 1.19.

## Contents

- [Before the first run on a node](#before-the-first-run-on-a-node)
- [Config](#config)
- [Environment](#environment)
- [Run](#run)
- [Check that the reads went through GDS](#check-that-the-reads-went-through-gds)
- [When a storage NIC stops](#when-a-storage-nic-stops)
- [What to expect](#what-to-expect)

## Before the first run on a node

- **Build.** `pixi run build-distributed`, as for any distributed run. The GDS backend needs no
  build option.
- **cuFile configuration.** The node's `/etc/cufile.json` must list the storage NICs' addresses in
  `rdma_dev_addr_list`. The launcher copies that file for each rank and turns `allow_compat_mode`
  off in the copy, so nothing has to be edited by hand.
- **Probe.** `scripts/gds/gds_probe.sbatch` issues the same cuFile calls outside the engine, from
  one 16 MiB read up to 60 s on every GPU at once. Run it on one exclusive node before the first
  query, and tell the storage admins first: a GDS run can leave a node's storage path degraded
  for everyone. The probe is a target of its own, so build it once, then submit from a folder
  that is not on the file system under test:

```bash
pixi run cmake --build --preset release --target gds_probe
FILE=/data/lineitem/part.0.parquet CHECKOUT=/path/to/sirius sbatch -w <node> /path/to/sirius/scripts/gds/gds_probe.sbatch
```

## Config

The `sirius.yaml` of the one-node TPC-H SF30000 run (4 ranks, one per GPU):

```yaml
sirius:
  memory:
    host: { capacity_bytes: 180Gi }
    disk: { disk_id: 0, capacity_bytes: 1000000000000, downgrade_root_dirs: "/path/to/spill" }
  executor:
    scan_manager:
      backend: gds
      gds: { num_threads: 8, read_timeout_s: 30, slow_read_ms_per_mib: 0 }
      num_threads: 32
      uring_n_reactors: 4
      cache: { mode: none }
  operator_params:
    max_dynamic_filter_bloom_bytes_per_gpu: 8Gi
    max_build_probe_resident_bytes: 64Gi
    max_build_hash_table_bytes: 64Gi
    enable_complete_group_having: true
```

Only two lines are about GDS: `backend: gds` and the `gds:` line.

| Key | Value used | Default | Why |
|---|---|---|---|
| `gds.num_threads` | 8 | 32 | Fewer reads in flight per rank. On this storage it makes the stall below rarer, at a cost: TPC-H Q9 at SF10000 on 4 ranks took 11.3 s with 8 threads, 9.0 s with 16 and 8.2 s with 32 |
| `gds.read_timeout_s` | 30 | 10 | During the SF30000 stall, held reads returned after up to 29.5 s. With 10 the rank exits sooner |
| `gds.slow_read_ms_per_mib` | 0 (off) | 5 | Turned off to keep a degraded session running long enough to record it. Leave the default unless you are diagnosing |

## Environment

Set these for the gateway and for every node before starting them:

```bash
export SIRIUS_CONFIG_FILE=/path/to/sirius.yaml
export SIRIUS_GDS=1
export SIRIUS_GATEWAY_MIN_RANKS=4
export SIRIUS_RANK_LOG_ROOT=/path/off/the/data/filesystem/rank-logs
export UCX_TLS=cuda_copy,cuda_ipc,tcp,self UCX_NET_DEVICES=bond0 UCX_MODULES=^ib,rdmacm
```

| Variable | What it does |
|---|---|
| `SIRIUS_GDS=1` | `scripts/sirius_rank_node.sh` gives each rank a cuFile configuration that cannot fall back to POSIX reads, and binds the rank to the CPUs of its GPU's socket |
| `SIRIUS_GATEWAY_MIN_RANKS` | The first statement waits for this many ranks (4 on one node, 8 on two). The default is 1, and a rank that registers after the first statement restarts every rank |
| `SIRIUS_RANK_LOG_ROOT` | One log folder per rank. Put it on a file system other than the one the scans read, so that the logs survive when that one stops answering |
| `UCX_TLS`, `UCX_NET_DEVICES`, `UCX_MODULES` | Keep UCX off the storage NICs. By default `scripts/nixl_env.sh` gives UCX the host's fastest RDMA ports that are in no bond, and on these nodes those are the NICs GPFS and cuFile read through. `UCX_NET_DEVICES` names the interface the ranks reach each other on (`bond0` here), and `UCX_MODULES=^ib,rdmacm` tells UCX to leave out its `ib` and `rdmacm` modules |

- With that `UCX_TLS`, `scripts/nixl_env.sh` sets `UCX_RCACHE_ENABLE=n` by itself; see
  [Settings and build](distributed-sirius.md#settings-and-build) for why.
- The UCX settings are a precaution. Every GDS run since the evening of 2026-11-10 used them (the
  first stall, earlier that day, was without them), and the stall below happens with them too.
- On one node the exchange between ranks does not use a NIC: the rank logs report `NIXL over the
  NVLink fabric`.
- Optional: `SIRIUS_GATEWAY_STATEMENT_TIMEOUT=<seconds>` (600 in the SF30000 run) and
  `SIRIUS_PRIORITY_DOWNSTREAM_FIRST=1`, which that run also set.

## Run

The three commands of the [distributed quick start](distributed-quickstart.md), each in a shell
with the environment above:

```bash
pixi run sirius-gateway --serve                      # wait for its "gateway <name>: quack:..." line
pixi run sirius-node --gpus 0,1,2,3                  # one rank per GPU
pixi run sirius-client < run.sql > client.txt 2>&1   # needs the gateway up: it reads its gateway.conf
```

`run.sql` creates one view per table with `read_parquet('/data/<table>/*.parquet')`, then runs
the queries. Run on a node of your own (one exclusive Slurm job), and never build or run GPU work
on a login node.

## Check that the reads went through GDS

Both lines are in each rank's log.

- **At start-up:** `[gds] cuFile ready: nvidia-fs 2.28, posix_fallback_allowed=false,
  cufile_counters=true, num_threads=8, ...`. It shows the settings in effect.
- **Per query:** `[gds] query=N device_bytes=... cufile_reads=... posix_reads=... GB/s=... ...
  held_reads=0 verdict=gds`. `verdict=gds` with `held_reads=0` is a clean query. A few POSIX reads
  are normal: cuFile serves reads of a few KiB that way. The line is written when the query
  finishes on the rank, and the log is flushed every few seconds, so a rank stopped right after
  the last statement can lose that statement's line.

## When a storage NIC stops

**The fault.** Under scan load, the data arriving on one storage NIC drops from tens of GB/s to
almost nothing within a second. The other NICs keep working, but the read threads of the ranks on
that NIC's socket end up waiting on reads bound to it, and the query hangs. It was seen at
SF10000 and at SF30000, with 8, 16 and 32 read threads, on several nodes. The cause is below Sirius,
between one client NIC and the storage, and was not found.

**How it shows.** The rank log gets one line a second while any read has been out for more than a
second:

```
[gds] held: 4 reads have not returned, the oldest a cuFile read for 1.6 s (331782 bytes at 457055839 of /data/lineitem/part.0.parquet); GB/s received per RDMA port: mlx5_0=22.7 mlx5_1=0.0 mlx5_2=0.0 mlx5_3=0.0 mlx5_4=22.8 mlx5_5=18.2 mlx5_6=0.0 mlx5_7=0.0
```

The line lists every RDMA port of the host. Among the storage NICs (`mlx5_0`, `mlx5_1`, `mlx5_4`
and `mlx5_5` on these nodes), the one near zero while the others carry data is the one that
stopped; here it is `mlx5_1`. While no other rank is reading, every port shows 0.0 and the line
cannot tell. A read that comes back
logs `[gds] held: a cuFile read returned after ... s`. After `gds.read_timeout_s` without a return
the rank exits with status 70 and the launcher does not start it again.

**What to do.**

1. Stop the session at the first `[gds] held:` line. Waiting does not help: the port stayed near
   zero for as long as the readers kept running.
2. Check the node before running anything else on it: a few small direct reads first
   (`dd iflag=direct`), then `gds_probe` with cuFile's `rdma_dev_addr_list` cut to one NIC at a
   time. A bad NIC stalls within the first few hundred MB, at the same or nearly the same byte
   count every round; a good one reads it all.
3. If a NIC is bad, leave the node and report it with the time of the first held line and the
   NIC's address. One node recovered within three minutes of stopping the session; another was
   still bad 90 minutes later.
4. Do not route around it. Leaving the bad NIC out of `rdma_dev_addr_list` kept GDS reads working
   for a few minutes, footer reads still hung, and then other NICs of the node stopped.

**Read threads.** With 8 threads per rank about 164 TB were read on two nodes with one stall. With
32 threads the first stall came after 36 TB on one node and 3.8 TB into a session on another; with
16 threads, 20 TB into a session. Fewer threads make it rarer, not impossible: the stall at 8
threads came at under 30 GB/s per NIC.

## What to expect

TPC-H SF30000 on one node (4 GB200, 4 ranks, the config above, 8.81 TB of Parquet on disk): the
22 queries took 463.5 s for their first runs and 403.1 s for their second, reading 30.3 TB per
pass, with engine read rates of 15 to 25 GB/s per rank for most queries. One stall
interrupted it; the session was stopped, the node recovered, and a second session ran the
remaining queries. With 32 read threads the same run on another node lost two ranks in its sixth
statement.
