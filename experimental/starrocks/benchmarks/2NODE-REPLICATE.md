# Replicating the two-machine Sirius CN experiment

Step-by-step, from a fresh shell, to reproduce the 2026-08-11/12 result: **StarRocks scheduling
plan fragments across two physical machines with NIXL moving GPU-resident data between them over
cross-host NVLink at 98 GB/s.**

Everything here was executed and measured on `presto-gb200-gcn-17` + `presto-gb200-gcn-18`.
Commands are marked **[17]**, **[18]** or **[both]**. Nothing needs root.

> **You need a shell on each host.** The agent session that produced this could only reach gcn-18,
> so every gcn-17 step was hand-run. Two terminals is the simplest setup.

---

## 0. What this reproduces, and the one thing that makes it work

| | |
|---|---|
| Shape | 1 FE (gcn-18) + 1 Sirius CN per host, 1 GPU each |
| Transport | nixl/UCX `cuda_ipc` over the **MNNVL fabric** |
| Cross-host canary | **98.0 GB/s** (0.41 GB/s before the fix) |

The load-bearing change is `SIRIUS_EXCHANGE_STAGING_ARENA=fabric`. `cudaMalloc`'s IPC handle is
node-local **by construction**, so a peer on another host can never map the staging arena; UCX
silently falls back to a host bounce at ~0.4 GB/s, which is below the transport's hard 2.0 GB/s
admission floor, so the peer is refused and **no distributed query can run at all**.

GPUDirect RDMA is *not* the answer on these hosts: `nvidia_peermem` is not loaded, and even with
dma-buf forced the mlx5 memory domain never advertises the `cuda` memory type. Do not put
`rc_mlx5` in `UCX_TLS` — the CN will fail to start.

---

## 1. Prerequisites (verify, don't assume)

**[both]**

```bash
uname -m                                   # aarch64
nvidia-smi -L                              # 4x NVIDIA GB200
ip -br addr show bond0                     # .52 on gcn-17, .53 on gcn-18, /27
nvidia-smi -q | grep -A3 Fabric            # State: Completed / Status: Success
```

The **ClusterUUID must match on both hosts** — that is what makes MNNVL a single fabric domain.
On this pair it was `3482beb4-a3cd-48a4-9b6c-a6ba43bc59a4`.

```bash
ls -d /raid/prestouser/aocsa/tpch_parquet_sf100    # 26 GB, node-local ext4, SAME PATH BOTH HOSTS
```

`/home` is NFS (`master:/home`) and therefore **shared**: the repo, the built engine and the CN
binary are the same files on both hosts. `/raid` is node-local and is **not** — data and logs live
there per host.

---

## 2. Build

Four commits are required; all are on `demo-multi-cn`:

| Commit | Why |
|---|---|
| `d271522a` | CN links against CUDA driver libs + system `ld` — without it the CN will not link |
| `f8360593` | plan substrait inside a transaction — without it **every** `FILES()` query fails |
| `3b19962f` | fabric-handle staging arena — without it cross-host exchange is refused |
| `adacda38` | repairs `test_streaming_fragment.cpp` so `pixi run make` completes |

**[either host — /home is shared, build once]**

```bash
cd /home/prestouser/aocsa/sirius
pixi run make                                     # full build
# If the unit-test target still breaks the build, build just the extension:
#   pixi run ninja -C build/release sirius_loadable_extension
```

Then the CN. **Do not use `pixi run cn-build`** — it `depends-on` the full `make`, so any unrelated
test failure blocks it:

```bash
cd /home/prestouser/aocsa/sirius/experimental/starrocks
NIXL_NO_STUBS_FALLBACK=1 TOOLS_DIR=/home/prestouser/aocsa/tools \
  pixi run bash -lc 'source scripts/cn-env.sh
                     cargo build --release -p sirius-starrocks-cn'
```

`bash -lc` (login shell) matters — `cn-env.sh` prepends `/usr/bin` so the **system** `ld` is used;
conda's `ld` links the conda sysroot's `libpthread` against system libc and dies on 39
`GLIBC_PRIVATE` undefined references.

**Verify the binaries are newer than any running process.** ninja writes a *new inode*, so a CN
started before a rebuild keeps the old `.so` mapped forever while `ls` shows the new one — two
hosts then silently run different engines:

```bash
stat -c '%i %n' build/release/extension/sirius/sirius.duckdb_extension
grep -m1 sirius.duckdb_extension /proc/$(pgrep -f '[s]irius-starrocks-cn')/maps   # inodes must match
```

---

## 3. Start the cluster

### 3a. Reset FE metadata **[18]**

```bash
cd /home/prestouser/aocsa/sirius/experimental/starrocks
rm -rf  /raid/prestouser/sr-eng-a-2node/fe/meta
mkdir -p /raid/prestouser/sr-eng-a-2node/fe/meta
```

Required. `conf/fe.conf` sets `priority_networks = 10.87.140.32/27`, but a metadata dir
bootstrapped under a different address — or left half-written by an interrupted start — makes the
FE exit with `current node is not added to the cluster, will exit`. Engine A creates no persistent
tables (everything is `FILES()` over parquet), so nothing is lost.

### 3b. FE + CN **[18]**

```bash
cd /home/prestouser/aocsa/sirius/experimental/starrocks
SIRIUS_EXCHANGE_STAGING_ARENA=fabric \
UCX_TLS=cuda_copy,cuda_ipc,tcp,self \
NUM_CNS_PER_HOST=1 CN_NODE="0" CN_CPUS="0-71" \
  ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.53
```

### 3c. CN only **[17]**

Wait until gcn-18's CN is alive first (§4), then:

```bash
cd /home/prestouser/aocsa/sirius/experimental/starrocks
SIRIUS_EXCHANGE_STAGING_ARENA=fabric \
UCX_TLS=cuda_copy,cuda_ipc,tcp,self \
NUM_CNS_PER_HOST=1 CN_NODE="0" CN_CPUS="0-71" \
  ./benchmarks/cn-2host.sh 10.87.140.52 10.87.140.53 --no-fe
```

**Both env vars must be set on BOTH hosts.** A fabric arena on one side and `cudaMalloc` on the
other cannot map each other — the export has to be mutual.

**Leave both in the foreground.** The launcher `wait`s on its child and its cleanup trap tears the
cluster down when the script exits; Ctrl-C kills the FE too.

`CN_CPUS="0-71"` gives the single CN GPU 0's whole socket. The launcher's default is the 4-CN
disjoint split.

---

## 4. Verify

**[either]** — `mysql` lives only in the pixi env:

```bash
export PATH=/home/prestouser/aocsa/sirius/experimental/starrocks/.pixi/envs/default/bin:$PATH
mysql -h 10.87.140.53 -P 9030 -uroot --vertical -e "SHOW COMPUTE NODES"
```

Both rows must read `Alive: true` with **recent** `LastHeartbeat`. Two *rows* is not two *nodes* —
the FE persists a registration after the process dies, showing `Alive: false`,
`StatusCode: DISCONNECTED`, `ErrMsg: java.net.ConnectException: Connection refused`.

`LastStartTime` renders in a different timezone than the FE's own `now()` (8 h off here) — do not
correlate it against file mtimes.

Use `--vertical`; this `mysql` rejects `\G` with `-e`.

### The canary — the gate on everything

```bash
grep -a 'nixl bandwidth canary' /tmp/cn-53-0.log | tail -2
```

```
nixl bandwidth canary peer=10.87.140.52:9102 gbps="98.0" bytes=16777216
```

**Anything near 0.4 GB/s means the fabric arena did not take**, and you will see
`below the 2 GB/s floor — Refusing the transport tier`. Check that
`SIRIUS_EXCHANGE_STAGING_ARENA=fabric` reached *both* CNs.

---

## 5. Prove distribution

```bash
export PATH=/home/prestouser/aocsa/sirius/experimental/starrocks/.pixi/envs/default/bin:$PATH
cat > /tmp/q.sql <<'EOF'
SET new_planner_agg_stage = 2;
WITH lineitem AS (SELECT * FROM FILES('path'='file:///raid/prestouser/aocsa/tpch_parquet_sf100/lineitem/*.parquet','format'='parquet'))
SELECT l_suppkey, count(*) AS n FROM lineitem GROUP BY 1 ORDER BY 1 LIMIT 5;
EOF
mysql -h 10.87.140.53 -P 9030 -uroot < /tmp/q.sql
```

Expect ~600 rows per supplier (600M lineitems / 1M suppliers) in well under a second.

> Use a **plain column** as the shuffle key. `l_orderkey % 4096` fails with
> `Unsupported expression in projection (falling back to CPU): mod(...)`.

**FE placement** — zero fragments deployed:

```bash
{ echo "SET new_planner_agg_stage = 2;"; echo "EXPLAIN SCHEDULER"; tail -n +2 /tmp/q.sql; } \
  | mysql -h 10.87.140.53 -P 9030 -uroot | grep -E 'PLAN FRAGMENT|INSTANCE\(|BE: '
```

The scan fragment must list **two distinct** `BE:` ids — map them via `ComputeNodeId`.

**NIXL transfer proof:**

```bash
grep -aE 'transmitted batches via nixl|received remote batches' /tmp/cn-53-0.log | tail -4
```

A `transmitted ... dest=10.87.140.52:9102` line matched by a `received remote batches` line is, by
construction, work crossing the machine boundary. `relayed native batches across a fragment
boundary` is the same-process short circuit — if that is all you see, nothing crossed.

---

## 6. TPC-H sweep

```bash
export PATH=/home/prestouser/aocsa/sirius/experimental/starrocks/.pixi/envs/default/bin:$PATH
mysql -h 10.87.140.53 -P 9030 -uroot -e \
  "SET GLOBAL enable_pipeline_engine=true; SET GLOBAL pipeline_dop=36; SET GLOBAL query_timeout=1800;"

cd /home/prestouser/aocsa/sirius/experimental/starrocks
sed -i 's/0\.0001000000/0.000001000000/' benchmarks/tpch/queries/q11.sql   # SF100 FRACTION; revert after

TPCH_DATA=/raid/prestouser/aocsa/tpch_parquet_sf100 \
FE_PORT=9030 QUERY_TIMEOUT=180 COLD_TIMEOUT=600 MIN_BACKENDS=2 \
  ./benchmarks/tpch/bench.sh --cold /raid/prestouser/bench-2node-A/timings.csv 3

git checkout HEAD -- benchmarks/tpch/queries/q11.sql
```

`pipeline_dop=36` because `--physcpubind=0-71` makes each CN report 72 cores. Pin it, or an A/B has
a free variable. `runs` is positional and mandatory before any query subset —
`bench.sh out.csv q05` sets `RUNS=q05`.

---

## 7. Results obtained, and what they are worth

Warm medians, SF100, 2 CNs across 2 hosts (2026-08-12):

| q01 | q02 | q03 | q04 | q05 | q06 | q07 | q11 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1687 | 505 | 576 | 472 | 1430 | 473 | 662 | 630 |

**8 measured passes, 1 real failure, 13 unmeasured — not "8/22".** Read §8 before quoting anything.

Engine B (stock StarRocks 3.5.20, 1 BE per host, same files) for comparison: 21/22, geomean
2762 ms. See `2NODE-ENGINE-B-RESULTS.md`.

---

## 8. Known failures — read before interpreting a sweep

**The harness has no correctness gate.** `bench.sh` scores `pass` on exit code + non-empty output +
no `ERROR` on line 1. Row counts are recorded and **never compared**. A query returning 1 row
instead of 100,000 registers as a fast win. Nothing above is oracle-validated.

**The staging-lease leak cascades — this is the big one.** A failed query strands its staging
leases (there is no `cancel_plan_fragment`), and the arena never recovers:

```
q08:  OOM at operator HASH_JOIN (index 0)                    <- real, 165.8 s, 100 retries
q09:  exchange staging arena exhausted: 2033670144 free of 17179869184
q22:  exchange staging arena exhausted: 2659840 free ... with 35 leases outstanding
```

16 GiB → 2.6 MB, monotonically, across 14 queries. **Every query after the first failure is
collateral, not a verdict** — their sub-second "refused" times are the tell. Restart both CNs after
any failure before trusting another row.

`RESTART_CMD` normally automates this, but it cannot: restarting requires a shell on gcn-17.

**q08 is a genuine OOM** at this shape — with 2 CNs each holds ~half the data, so the build side per
GPU is larger than in the recorded 4-CN single-host baseline (where q08 failed differently, on a
60 s RPC timeout).

**Do not compare against the 4-CN baseline** in `.claude/skills/tpch-bench/SKILL.md`. It read from
`/home` (**NFS**); this reads node-local `/raid`. That difference alone can move scan-bound queries
by an order of magnitude, independent of anything two-machine.

---

## 9. Teardown

**[both]**

```bash
pkill -f '[s]irius-starrocks-cn'
pkill -f 'com.starrocks.[S]tarRocksFE'
sleep 5
pgrep -af '[s]irius-starrocks-cn|com.starrocks.[S]tarRocksFE' || echo "ALL STOPPED"
nvidia-smi --query-compute-apps=pid --format=csv,noheader          # must be empty
```

**Use the bracket form** `'[s]irius-...'`. A bare pattern matches `pkill`'s own command line and
kills the shell — and if the same command *also* mentions the binary path elsewhere, even the
bracket form self-matches. Keep the `pkill` in its own command.

---

## 10. Open work

* **Staging-lease leak** — one failed query permanently degrades the cluster. Blocks any unattended
  sweep. Not root-caused.
* **q08 `HASH_JOIN` OOM** at SF100 with 2 CNs. Does the downgrade/spill path apply here?
* **The 100-retry loop** burns 165 s before failing an OOM that retrying cannot fix.
* **No two-host teardown/restart script** — every gcn-17 step is hand-run.
* **Correctness** — no SF100 result has ever been diffed against the DuckDB oracle.
