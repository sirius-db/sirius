# Engine B — stock StarRocks 3.5.20 baseline on the 4× GB200 Grace box

This is the **CPU baseline** half of the A/B comparison. Engine A is Sirius-as-StarRocks-CN
(4 GPU compute nodes); engine B is unmodified StarRocks 3.5.20 from the arm64 release
tarball. Both read the same TPC-H parquet through `FILES()` — there is no loading step.

---

## ⚠️ ENGINE A AND ENGINE B CAN NEVER RUN AT THE SAME TIME

Two independent reasons, either one fatal:

1. **Port 9030.** Both FEs bind `9030` (plus `8030`, `9020`, `9010`). Engine A's FE lives in
   a *different tree* — `experimental/starrocks/starrocks/output/fe` (see
   `benchmarks/cluster8.sh:34`) — while engine B's FE is `$HOME/starrocks-bench/fe`. So the
   **files** never collide, but the **ports** always do. The second FE to start simply fails
   to bind, or worse, a `mysql -P9030` session silently lands on the wrong engine and the
   benchmark measures the wrong thing.
2. **The 144 host CPUs and the ~980 GB of LPDDR.** Engine B's two BEs are sized to consume
   480 GiB and all 144 cores. Engine A's 4 CNs are sized to consume their own large share.
   Running both means both are memory- and CPU-starved, and on a **`Swap: 0`** box an
   over-commit is an **OOM-kill, not a slowdown**.

Before launching engine B:

```bash
pgrep -af sirius-starrocks-cn     # must print nothing
pgrep -af StarRocksFE             # must print nothing
```

`setup-engine-b-gb200.sh` prints a loud warning if engine A is live, and hard-refuses to
rewrite conf files if engine **B** is live. It never starts or kills anything.

---

## ⚠️ The committed `setup-engine-b.sh` must NOT be reused here

`benchmarks/tpch/setup-engine-b.sh` is retained for its original target and is **wrong for
this box** in four ways:

| Problem | Detail |
|---|---|
| **`mem_limit = 16G`** (lines 44, 55) | An **L4-era number** — that script was sized for a tiny single-GPU host (its own header says so). 16 GiB per BE at SF100 spills continuously and would make engine B look absurdly slow. **Never reuse this value.** It has since been hand-edited on disk (400G → 64G at various points), which is exactly the drift this config set exists to end. |
| **Requires Docker** | It extracts the artifacts from `starrocks/artifacts-ubuntu:3.5.20` via `docker cp`. **There is no Docker on this box.** The trees are already extracted at `$HOME/starrocks-bench/{fe,be}`. |
| **`cat >` heredocs overwrite the shipped confs wholesale** | It silently drops the stock `JAVA_OPTS` from both `fe.conf` (→ FE falls through to the `-Xmx8192m` fallback at `start_fe.sh:124` and loses `-XX:ErrorFile`) and `be.conf` (→ loses the JDK17 `--add-opens` flags). |
| **Everything lands on NFS** | `storage_root_path`, `spill`, `meta_dir` and all logs default under `${STARROCKS_HOME}` = `$HOME/starrocks-bench`, which is `nfs4` on `master:/home`. |

Use `./setup-engine-b-gb200.sh` in this directory instead.

---

## BE topology (recommended: **N = 2**)

| BE | CPU bind | Mem bind | LPDDR on node | `num_cores` | `mem_limit` (SF100 / SF1000) | be_port | heartbeat | brpc | http | starlet | `port` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **be1** | node 0 → CPUs 0–71 | node 0 | 489,960 MiB = 478.48 GiB | 72 | **240G** / 224G | 9060 | **9050** | 8060 | 8040 | 9070 | 20001 |
| **be2** | node 1 → CPUs 72–143 | node 1 | 489,823 MiB = 478.34 GiB | 72 | **240G** / 224G | 9062 | **9052** | 8062 | 8042 | 9072 | 20002 |

FE: `http 8030`, `rpc 9020`, **`query 9030`**, `edit_log 9010`, `-Xmx16g`, `numactl --membind=0,1`.

Ports are **unchanged** from the staged `$HOME/starrocks-bench/be1`/`be2` layout, so those
trees stay compatible. Registration is by **heartbeat** port:
`ALTER SYSTEM ADD BACKEND "127.0.0.1:9050"`, `"...:9052"`.

### 🚨 `--numa` may only ever be `0` or `1`

`start_be.sh --numa N` → `start_backend.sh:134-136` → `numactl --cpubind N --membind N`.
`RUN_NUMA` is substituted into **both** flags. On this box `numactl` reports **34 nodes**, and
nodes **2, 10, 18, 26 are GPU HBM** (188,416 MiB each, **zero CPUs**). `--numa 2` would
membind a BE's heap *into GPU0's HBM* on a node with no CPUs — instant failure, or corruption
of memory a Sirius CN is using. Only nodes 0 and 1 have CPUs and LPDDR.

Verify after launch — this must print `0` or `1`, never `0-2,10,18,26`:

```bash
grep Mems_allowed_list /proc/$(cat $HOME/starrocks-bench/be1/bin/be.pid)/status
```

---

## Memory arithmetic against the **real** ~980 GB

### The trap: `/proc/meminfo` counts GPU HBM

```
/proc/meminfo MemTotal = 1,775,050,816 kB = 1692.6 GiB      <-- what free -g shows
                                          - 736.0 GiB       <-- 4 x 188,416 MiB GPU HBM
                                                              (NUMA nodes 2, 10, 18, 26)
                                          = 956.82 GiB      <-- REAL CPU-addressable LPDDR
                                                              (979,783 MiB, the "~980 GB")
```

`MemInfo::init()` (`be/src/util/mem_info.cpp:66-91`) reads that `MemTotal` line verbatim.
`set_memlimit_if_container()` (`:113-117`) returns early without `/.dockerenv`, and there is
no cgroup limit, so nothing corrects it. Then `exec_env.cpp:184-186`:

```cpp
bytes_limit = ParseUtil::parse_mem_spec(config::mem_limit, MemInfo::physical_mem());
bytes_limit = bytes_limit * 0.9;   // soft limit
```

**The stock `mem_limit = "90%"` therefore resolves to `0.90 × 1692.6 = 1523 GiB` per BE —
1.6× the real LPDDR, on a box with `Swap: 0`.** Two BEs would be 3046 GiB. The
`bytes_limit > physical_mem()` guard at `exec_env.cpp:191` cannot catch this because
`physical_mem()` is itself the wrong number.

> **Rule for this box: `mem_limit` is always an absolute byte value. Never a percentage.**

Note also that `mem_limit` is the **hard RSS ceiling**; the `×0.9` produces the *soft* tracker
limit. Budget against `mem_limit`, not against `0.9 × mem_limit`.

### Fixed tenants (measured, not guessed)

| Tenant | Size | Source |
|---|---|---|
| `mmfsd` (GPFS daemon) | 50.68 GiB permanent RSS | `/proc/774274/status` |
| Engine B FE JVM | ~20 GiB (`-Xmx16g` + metaspace + threads + direct) | `fe.conf` |
| Kernel, slab, daemons | ~10 GiB | — |
| **Total budgeted** | **84 GiB** | |

### SF100 (dataset 26 GiB) — `mem_limit = 240G`

```
  CPU-addressable LPDDR                            956.82 GiB
- fixed tenants                                     84.00
- 2 BEs x 240 GiB                                  480.00
  ----------------------------------------------------------
= kernel page cache                                392.82 GiB   = 402.2 "GB"   >= 400 ✓
```

Per-node check against the **hard** membind ceiling:

| node | capacity | BE | FE | mmfsd share | kernel | slack |
|---|---|---|---|---|---|---|
| 0 | 478.48 GiB | 240 | ~20 | ~25 | ~5 | **~188 GiB** |
| 1 | 478.34 GiB | 240 | — | ~25 | ~5 | **~208 GiB** |

A `--membind` is a **hard cap**: a BE exceeding its node's LPDDR is OOM-killed, it does *not*
fall back to the other socket. ~190 GiB of per-node slack comfortably absorbs jemalloc
fragmentation and the connector-scan thread stacks.

240 GiB is **3–5× any plausible per-BE TPC-H SF100 peak** (the q18/q21 hash builds), so
engine B never spills and is **not handicapped** by this number.

### SF1000 (dataset 265 GB = 246.8 GiB) — `mem_limit = 224G`

```
  956.82 - 84 (fixed) - 2 x 224 = 424.82 GiB page cache   = 435.0 "GB"   >= 400 ✓
                                                          > 246.8 GiB dataset ✓ (fully warm)
```

Change **both** BEs to `224G` (and `datacache_disk_size` to `200G`) before an SF1000 run.

### Resolving the OPEN-ISSUES M1 item 3 conflict, explicitly

M1 item 3 asks for **"≥400 GB of LPDDR left for page cache."** Two things had to be pinned
down before the number could be chosen:

**1. Which "GB"?** The box brief states total LPDDR as "~980 GB", which is `979,783 MiB / 1000`
— i.e. it uses MiB/1000, not GiB and not decimal GB. Read consistently, the ≥400 GB bar is
**390.6 GiB**. That is the *strictest* of the three plausible readings (decimal 400 GB =
372.5 GiB is looser), so **every number here is validated against 390.6 GiB.** Both SF100
(392.82 GiB) and SF1000 (424.82 GiB) clear it.

**2. Is the conflict simultaneous?** For engine A, 4 CNs × 200 GiB = 800 GiB leaves only
~157 GiB and genuinely does conflict. **For engine B the conflict dissolves, because A and B
never run concurrently** (they share port 9030 — see the top of this file). The ≥400 GB
constraint only has to hold **per engine, one at a time**, and engine B owns the whole box
when it runs. That is what makes `240G × 2` affordable.

The remaining honest caveat: **at SF100 the ≥400 GB ask is essentially vacuous** — the
dataset is 26 GiB, so you cannot fill 390 GiB of page cache no matter how you size the BEs.
Choosing 240G rather than a larger 320G is therefore *free* at SF100 (both fully cache the
dataset; 392.82 GiB is 15× it), and it has the real benefit of using **one coherent budget
that satisfies the constraint at both scales**. At SF1000 the ask stops being vacuous and
genuinely binds — the dataset is 246.8 GiB and page cache is the difference between a cold
and a warm scan — which is precisely why the limit yields there rather than the cache.

---

## Why **N = 2** BEs

**There are exactly two CPU+memory NUMA domains on this box.** Two BEs gives a perfect 1:1
process ↔ NUMA-domain mapping.

1. **It matches the hardware.** Node 0 = CPUs 0–71 + 478.48 GiB; node 1 = CPUs 72–143 +
   478.34 GiB. Nodes 2/10/18/26 are GPU HBM with zero CPUs. One BE spanning both sockets
   would pay cross-socket traffic on every hash-table probe, and the StarRocks BE is
   memory-bandwidth-bound. A single BE's pipeline engine scales to 72 cores in-process
   without difficulty.
2. **Sharding is far more expensive for B than for A.** Engine A's shuffle rides NVLink
   (NV18 all-to-all, measured 322–399 GB/s per peer). Engine B's shuffle rides **loopback
   TCP through brpc**. Going to 4 BEs would roughly double the bytes that must cross that
   TCP path *for zero additional hardware parallelism* — it splits each socket into two
   processes that then exchange over TCP what they could have exchanged in-process.
3. **Forcing B to 4 shards "for symmetry with 4 CNs" would be the wrong kind of fairness.**
   The honest comparison is best-A vs best-B on the same 144 cores and the same LPDDR:
   A gets 4 procs × 36 cores, B gets 2 procs × 72 cores — both 144. A additionally gets
   4 GPUs. *That* is the experiment.
4. **It keeps the staged trees compatible.** `$HOME/starrocks-bench/be1` and `be2` already
   exist with these exact ports.

A **4-BE variant is provided in `sensitivity-4be/`** as a sensitivity check, **not the
headline**. It halves `num_cores` to 36 and `mem_limit` to 120G so the per-socket totals are
identical, and it continues the +2 port ladder (be3: 9064/9054/8064/8044/9074, be4:
9066/9056/8066/8046/9076). It requires an **explicit `numactl --physcpubind` wrapper**,
because `--numa N` always cpubinds a *whole* node and cannot express a half-socket.

---

## What changed from the 3.5.20 defaults, and why

Every value is annotated inline in the conf files. Summary of the BE changes:

| key | default (3.5.20) | here | why |
|---|---|---|---|
| `mem_limit` | `"90%"` (`config.h:84`) | **`240G`** | % resolves against HBM-inflated `MemTotal` → 1523 GiB → guaranteed OOM-kill. |
| `num_cores` | `0` (`config.h:574`) | **`72`** | **Mandatory with `--numa`.** `CpuInfo` counts `/proc/cpuinfo` lines and **never calls `sched_getaffinity`** (`cpu_info.cpp:150-152,165-166`), so `--cpubind` does *not* shrink it. `cpu_info.cpp:173` is the only override. Without it each BE builds 144-wide pools for 72 cores and the FE sees 288 cores. |
| `enable_resource_group_bind_cpus` | `true` (`config.h:1514`) | **`false`** | `get_core_ids()` enumerates all 144 cores regardless of the cpuset, so the BE fights its own numactl mask. |
| `pipeline_connector_scan_thread_num_per_cpu` | `8` (`config.h:885`) | **`4`** | This is the pool `FILES()` actually uses: `ceil(v × num_cores)`. Default = 576/BE (1152 total) on 144 cores. |
| `connector_io_tasks_per_scan_operator` | `16` (`config.h:1035`) | **`32`** | GPFS rewards deep IO queues. Adaptive *ceiling*, not a floor. **Primary sweep knob.** |
| `disable_storage_page_cache` | `false` (`config.h:306`) | **`true`** | Caches **native OLAP** pages. There are no native tables — everything is `FILES()`. Otherwise reserves 20% of the tracker for a cache that never fills. |
| `datacache_mem_size` | `"0"` (`config.h:1297`) | **`32G`** | This *is* the cache external parquet uses. `datacache_enable` is `true` by default but **sized zero**, and auto-adjust only grows the disk quota after **7200 s of disk idle** (`config.h:1353`) — so stock 3.5.20 caches nothing during a bench run. Comes out of `mem_limit`; not double-counted. |
| `datacache_disk_size` | `"0"` | **`0`** (explicit) | Pins the behavior instead of leaving it to auto-adjust. Set `200G` for SF1000. |
| `storage_root_path` | `${STARROCKS_HOME}/storage` | `/raid/.../storage` | **Off NFS.** |
| `spill_local_storage_dir` | `${STARROCKS_HOME}/spill` (`config.h:1242`) | `/raid/.../spill` | **Off NFS.** |
| `sys_log_dir` | `${STARROCKS_HOME}/log` | `/raid/.../log` | **Off NFS.** |
| `port` | `20001` (`config.h:243`) | `20001`/`20002` | Every BE otherwise inherits the same value; cheap collision insurance. |
| `JAVA_OPTS` | JDK17 `--add-opens` | *restored* | The committed script's `cat >` heredoc drops it. |

**Deliberately NOT changed** (the common misdirections):
`scanner_thread_pool_thread_num` (48 — sizes the *non-pipeline* `table_scan_io` pool, inert
under the default pipeline engine); `num_threads_per_core` (3 — **dead code**, grepping
3.5.20's `be/src` finds only the definition, zero consumers); `vector_chunk_size` (4096 —
well tuned; raising it inflates per-operator memory by DOP × pipeline count);
`query_cache_capacity`/`enable_query_cache` (off by default, native-table per-tablet only);
`load_process_max_memory_limit_percent` (no load path); `query_max_memory_limit_percent`
(90% of a now-correct tracker); `pipeline_exec_thread_pool_thread_num` and friends (`0` →
`num_cores`, which is now correctly 72).

### Storage: everything moves off NFS

```
/home     master:/home   nfs4   <- $HOME/starrocks-bench lives here. CODE ONLY.
/raid     /dev/md0       ext4   14T (13T free)  <- LOCAL. all data/spill/logs/meta.
/         nvme1n1p3      ext4   1.8T (1.2T free) <- LOCAL
/scratch  gpfs           4.7P   <- source parquet, read-only
```

With stock `${STARROCKS_HOME}/...` paths, tablet storage, the datacache disk tier, spill
scratch, every log file **and the FE's BDB JE metadata journal** all go over NFS — which also
runs ~0.2 s ahead of the local clock. This is the single largest avoidable handicap on
engine B, and it would make the A/B result measure the filesystem rather than the engine.

### FE: coordinator, not a worker

`-Xmx16g`, up from the `8192m` fallback. The FE parses SQL, runs the CBO, enumerates `FILES()`
splits and ships fragments — it **never holds query data**. 8 GiB is thin for SF1000 split
enumeration; 16 GiB is 2× that headroom and deliberately **not** 32 GiB, because every GiB
here comes out of the same LPDDR pool as the BEs and the page cache. `-XX:ErrorFile` is
restored (the `start_fe.sh:124` fallback drops it). `FE_ENABLE_AUTO_JVM_XMX_DETECT` cannot
help — it is a no-op off-container (`common.sh:118-120` needs `/.dockerenv`) and would size
the heap from the HBM-inflated `MemTotal` anyway.

**`run_mode` is deliberately left unset** (defaults to `shared_nothing`). Engine A's FE conf
sets `run_mode = shared_data` because its storage-less Rust CNs are only schedulable that
way. **Do not copy that setting here** — engine B runs real BEs.

**`meta_dir` moves to `/raid`, which bootstraps a brand-new empty FE cluster.** The BEs must
be re-registered with `ALTER SYSTEM ADD BACKEND` after the first start. One-time cost;
`setup-engine-b-gb200.sh` prints the exact statements.

---

## Files

| file | purpose |
|---|---|
| `fe.conf` | → `$HOME/starrocks-bench/fe/conf/fe.conf` |
| `be1.conf` | → `$HOME/starrocks-bench/be1/conf/be.conf` (node 0) |
| `be2.conf` | → `$HOME/starrocks-bench/be2/conf/be.conf` (node 1) |
| `sensitivity-4be/be{1..4}.conf` | 4-BE sensitivity variant. **Not the headline.** |
| `setup-engine-b-gb200.sh` | Idempotent, non-Docker layout. **Starts nothing.** |

## Running it

```bash
# 1. Confirm engine A is fully stopped.
pgrep -af sirius-starrocks-cn        # must print nothing

# 2. Lay out the trees and confs (starts nothing; prints the launch commands).
export JAVA_HOME=/path/to/jdk17
cd /home/prestouser/aocsa/sirius/experimental/starrocks/configs/gb200-4gpu/engine-b
DRY_RUN=1 ./setup-engine-b-gb200.sh   # preview
./setup-engine-b-gb200.sh             # apply

# 3. Follow the printed launch commands. Then verify:
#    - SHOW BACKENDS: Alive=true, CpuCores=72 per BE (NOT 144)
#    - grep Mems_allowed_list /proc/<be-pid>/status  ->  0 or 1 (NOT 0-2,10,18,26)
```

Expected FE-side DOP after this: `min(max_pipeline_dop=64, 72/2)` = **36** per fragment
instance × 2 BEs = 72 concurrent drivers on 144 physical cores. Before the `num_cores` fix
the BEs reported 144 → 72 → silently clamped to 64 by `max_pipeline_dop`.
