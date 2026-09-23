# Packed NIXL hop throughput

Measure-then-tune pass over the two-CN packed NIXL hop, following cascade-tpc-shuttle's
Milestone 9 (`/home/ubuntu/cascade-tpc-shuttle/docs/milestones/M9.md`): same benchmark shapes
(distributed-join's all-to-all, 1M-key shuffle, 1M x 1M join), a per-hop timeline first, then
only the changes that timeline justified.

## Setup

- One host, two NVIDIA RTX PRO 4500 Blackwell (topology `NODE`), one CN per GPU, UCX
  `cuda_copy,cuda_ipc,tcp,self`, 1 GiB staging arena per CN. The shuttle's numbers are two GB10
  hosts over 200 Gb/s RoCE (wire 13.3 GB/s); absolute numbers are not comparable, the shape is.
- Wire ceiling here: a single one-way WRITE (the first-contact canary) reaches 32.5 GB/s; with
  both CNs writing to each other at once, each direction gets 23-25 GB/s (`wire_gbs`).
- The NIXL changes below ship in #1794 (`feat/group-by-nixl-shuffle`); the SQL benchmarks and this
  doc sit on top of #1920 (`feat/tpch-join-nixl`). Numbers were measured on the pre-rebase commits
  (`c7f3c733` as the "before" head).

## Running

```bash
L=/home/ubuntu/sirius-wt/all22/gpu-lock.sh
$L experimental/starrocks/tests/2cn_bench_a2a.sh      # all-to-all 64 MB, 1 GB, 2 GB (no FE)
$L experimental/starrocks/tests/2cn_bench.sh          # SQL join 1M x 1M and shuffle 1M, DuckDB-checked
```

`SIRIUS_CN_TIMING=1` (set by both scripts) adds a `packed hop timing` line per hop (md, pack,
lease, write, announce, `wire_busy_pct`, `wire_gbs`) and a `fragment timing` line per fragment
(build, inputs, run). `SIRIUS_CN_NIXL_WINDOW` sets the writes in flight per hop (default 4).

| Benchmark | Shape (after distributed-join / the shuttle's `bench.cpp`) | Timed |
| --- | --- | --- |
| all-to-all | `bytes/2` per peer as 15 MiB frames over the production hop (Md, Lease, WRITE, Packed, EOS); receiver releases each lease on arrival | 4 rounds after 1 warm-up, GB/s per GPU = per-peer bytes x rounds / time |
| SQL join 1M x 1M | 1M build + 1M probe rows per CN (one equal-size parquet file each), INT64 key + INT64 payload, unique shuffled build keys, probe selectivity 0.3 (~600k matches); `SELECT SUM(b_pay), SUM(p_pay) FROM p JOIN [SHUFFLE] b ON p_key = b_key` | FE wall per query, and CN round = first leaf fragment start to last fragment end across both CNs; cold = first run, warm = mean of 3 more |
| SQL shuffle 1M | 1M INT32 keys per CN, `rand % 10M`, one-stage `GROUP BY k` (`new_planner_agg_stage = 1`) so raw keys hash-exchange | same |

The parquet files are padded to identical sizes: StarRocks' `FileScanNode` byte-splits a file
larger than `total / instances`, and this CN only scans whole files.

## Results

Worker 0 (CN0), all-to-all medians of three runs, SQL means of three warm rounds.

| Case | before (`c7f3c733`, #1920 before the fixes) | Md fix | + window 4, async announce | shuttle nixl direct after M9 | distributed-join |
| --- | --- | --- | --- | --- | --- |
| a2a 2 GB (GB/s per GPU) | deadlock (see below) | 23.2 | **22.7** | 13.1 | 13.3 |
| a2a 1 GB | deadlock | 22.4 | 22.5 | 13.0 | 13.3 |
| a2a 64 MB | deadlock | 19.4 | **21.0** | 9.9 | 13.1 |
| wire busy, 2 GB | | 86% | **99%** | 99% | |
| wire GB/s while busy, 2 GB | | 26.6 | 23.3 | 13.4 | |
| SQL join 1M x 1M, warm, CN round (ms) | 39.0 | 40.8 | 40.1 | 7.0 (join only, not SQL) | |
| SQL join 1M x 1M, warm, FE wall (ms) | 90.6 | 90.9 | 93.5 | | |
| SQL join 1M x 1M, cold, FE wall (ms) | 627 | 638 | 629 | | (cold single run 50) |
| SQL shuffle 1M, warm, CN round (ms) | 17.3 | 16.9 | 17.2 | 1.05 (shuffle only, not SQL) | 0.7 |
| TPC-H joins matching DuckDB (Q3 Q5 Q7-Q10 Q12 Q14 Q19) | Q14, Q3, Q5; Q7 hangs, rest never reached | | 9 of 9 | | |

The Md-fix all-to-all column is a single run. Window 1 (stop-and-wait order, async announce
only) as a reference: 18.9 / 22.7 / 22.6 GB/s at 64 MB / 1 GB / 2 GB, wire busy 80% / 92%.

## What the timeline showed

- **All-to-all is at the wire.** Stop-and-wait already kept the wire 86% busy; the idle 14% was
  the serialized Lease RPC (2.5 ms per 1 GB hop) and Packed announce (2.9 ms). A window of 4
  with announces on their own thread takes busy time to 99%, and every hop then runs at the
  23 GB/s the two directions share. That helps the small round (64 MB: 19.4 -> 21.0 GB/s) and
  leaves bulk where the link caps it.
- **The SQL join is not transport-bound.** One warm 40 ms round (CN0, baseline commit):

  | Phase | ms |
  | --- | --- |
  | build-side scan + hash partition fragment | 12.4 |
  | build hop (Md 0.1, pack 0.4, lease 0.05, WRITE 0.25, announce 0.05) | 1.0 |
  | probe-side scan + hash partition fragment | 12.4 |
  | probe hop | 1.0 |
  | join + partial SUM fragment (build 1.6, push_packed 0.2, run 2.5) | 4.3 |
  | final SUM fragment | 2.8 |

  The two NIXL WRITEs are 0.5 ms of the round. The rest is engine work, and the two leaf
  fragments run one after the other on each CN because the engine serializes fragments.
  Before the first fragment starts, the FE spends about 33 ms planning and dispatching.
  The shuttle's 4.5-7 ms join times only the two shuffles plus `cudf::inner_join` on
  resident tables; a SQL round also scans parquet and plans four fragments.

## Changes

1. `febb938f` Timeline (`SIRIUS_CN_TIMING`) and `--bench-a2a-bytes`. No behavior change.
2. `3dade890` **Md deadlock.** When both CNs opened a hop to each other at once, each sent an
   Md RPC whose handler loaded the caller's metadata on the local transport thread. That
   thread was already blocked in its own Md call, so both waited out the 60 s PRPC timeout
   (`PRPC read: failed to read PRPC header`). This was the TPC-H Q7 hang; the all-to-all bench
   hit it on round 1. The handler now replies from the cached blob (only the WRITE initiator
   needs the target's metadata), and a sender loads each peer's metadata once.
3. `a20d3d65` **Window and async announce** (their A2/A3 and B2). The transport packs and
   leases the next batches while up to `SIRIUS_CN_NIXL_WINDOW` WRITEs are in flight. It
   releases a local lease when its WRITE lands, and announces frames in seq order on a scoped
   thread. A failed hop still waits for every posted WRITE before it releases that WRITE's
   local lease.

Not done:

- **B4, adopt the receiver lease instead of copying.** `push_packed` (unpack, deep copy,
  sync) measures 0.2 ms for 16 MB per join round, about 0.5%. Adopting the lease would need a
  cuCascade representation or RMM resource over arena slices, and would change spill and
  accounting for batches living in registered memory.
- **B1, sender partition.** The 12.4 ms leaf fragments are the lever for the SQL join. Running
  the build and probe leaves concurrently, or fusing scan, partition and pack, is engine work
  outside this transport pass.
- **Their Track C** (control frames as NIXL notifications). With the window, the wire is 99%
  busy, so there is nothing left for it to recover.
