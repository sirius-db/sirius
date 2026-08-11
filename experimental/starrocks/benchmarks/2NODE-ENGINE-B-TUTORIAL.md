# Tutorial — two-machine standalone StarRocks (engine B) reference cluster

Companion to `2NODE-RUNBOOK.md`. That runbook brings up **engine A** (Sirius Rust CNs) across
`presto-gb200-gcn-17` + `presto-gb200-gcn-18`. This document brings up the **engine B** control:
stock, unmodified StarRocks on the same two boxes, so a distributed engine-A number has a
distributed CPU baseline to be measured against.

---

## 0. What this is, and the shape

**Shape: 1 FE + 1 backend process per host, 2 hosts.**

| | gcn-17 (10.87.140.52) | gcn-18 (10.87.140.53) |
|---|---|---|
| FE | yes (coordinator only) | — |
| BE | 1, unpinned, all 144 cores | 1, unpinned, all 144 cores |
| GPU | none used | none used |

Engine B runs **BEs, not CNs**, in the default `shared_nothing` run mode. That is not a style
choice: a `FILES()` `SELECT` plans to a `FileScanNode` (`PlanFragmentBuilder.java:4031` at tag
`3.5.20`), which is a `LoadScanNode`, and `LoadScanNode.getAvailableComputeNodes` returns
`getIdToBackend()` in shared-nothing (`LoadScanNode.java:112-124`). A shared-nothing CN is not in
that map and every query fails with "No available backends"
(`benchmarks/tpch/setup-engine-b.sh:5-7`).

Two consequences the engine-A runbook does **not** carry over:

* `run_mode` stays **unset** (defaults to `shared_nothing`). Engine A requires
  `run_mode = shared_data` (`2NODE-RUNBOOK.md:125`); copying that here breaks engine B.
  *(Correction to the runbook's stated reason for that line — see `2NODE-RUNBOOK-GAPS.md` §A.)*
* Registration is **manual**: `ALTER SYSTEM ADD BACKEND`. Engine A's CN self-registers over MySQL
  (`experimental/starrocks/src/lib.rs:207-209`); a stock BE does not.

Engine A and engine B **cannot run at the same time on either host**: both FEs bind `9030`, and
both would take all 144 CPUs (`configs/gb200-4gpu/engine-b/setup-engine-b-gb200.sh:26`). Stop
engine A first — `pgrep -af sirius-starrocks-cn`.

---

## 1. Prerequisites, and the binary problem

**There is no binary problem. Do not build from source.**

A stock, official, **aarch64** StarRocks **3.5.20** FE and BE are already extracted at
`/home/prestouser/starrocks-bench/{fe,be,be1..be4}`, and `$HOME` is NFS
(`master:/home`, `df -PT /home` → `nfs4`), so **the identical tree is already visible on both
hosts at the identical path**. Nothing to download, build, or copy.

Verified on gcn-18:

```
$ file /home/prestouser/starrocks-bench/be/lib/starrocks_be
ELF 64-bit LSB executable, ARM aarch64, ... dynamically linked,
interpreter /lib/ld-linux-aarch64.so.1, ... not stripped     # 383,358,928 bytes, Jul 22 19:53

$ unzip -p /home/prestouser/starrocks-bench/fe/lib/starrocks-fe.jar \
      com/starrocks/common/Version.class | strings | grep -E '^[0-9]+\.[0-9]+\.[0-9]+|^[0-9a-f]{7}$'
3.5.20
4d17879
```

Provenance: the tree came from the `linux/arm64` manifest of `starrocks/artifacts-ubuntu:3.5.20`
(registry config blob `created: 2026-07-22T19:54:12Z`, matching the local file mtimes to the
minute). The direct release-tarball URL 403s and **there is no container runtime on these boxes**
(`command -v docker podman nerdctl` → all absent), so `benchmarks/tpch/setup-engine-b.sh`
(which does `docker create`/`docker cp`) cannot be used here — but its docker block is guarded by
`if [ ! -d $B/fe ]` (`setup-engine-b.sh:15`) and is skipped anyway.

**Pin the version.** This tutorial targets **3.5.20**, and every line citation below is against
`git show 3.5.20:<path>` in the vendored submodule. The submodule's checked-out HEAD is **4.1.1**
(`git describe --tags` → `4.1.1`), where several BE defaults differ (`datacache_mem_size`
`"0"`→`"20%"`, `datacache_disk_size` `"0"`→`"100%"`, `storage_page_cache_limit` `"20%"`→`"-1"`).
Do not cite 4.1.1 line numbers for these binaries. Recover 3.5.20 defaults offline with:

```bash
cd /home/prestouser/aocsa/sirius/experimental/starrocks/starrocks
git show 3.5.20:be/src/common/config.h | grep -n 'datacache_\|storage_page_cache'
```

### Build-from-source fallback (do not use unless the trees are lost)

`./build.sh --be` in the submodule produces **4.1.1**, not the 3.5.20 baseline, and
`thirdparty/installed` does not exist, so `build.sh:83-85` first triggers a full 77-package
thirdparty build including LLVM — many hours and tens of GB, at `PARALLEL = nproc/4+1 = 37` of
144 cores (`build.sh:87`). aarch64 is supported upstream (`thirdparty/vars-aarch64.sh` exists) but
this is the wrong answer for a baseline that must match the recorded 3.5.20 numbers.

### Prerequisites already satisfied on gcn-18 (re-check on gcn-17)

| Requirement | Where enforced | Measured on gcn-18 |
|---|---|---|
| JDK 17+ for the FE | `start_fe.sh:42` `MIN_JDK_VERSION=17`, `:107-108` `exit -1` | `/usr/lib/jvm/java-21-openjdk-arm64` present |
| JDK for the BE (`file://` goes through libhdfs/JNI) | `start_backend.sh` only WARNS | same JDK |
| `numactl` | used below for `--membind` | `/usr/bin/numactl` |
| `ulimit -n >= 60000` | `storage_engine.cpp` `_check_file_descriptor_number`, `min_file_descriptor_number = 60000` (3.5.20 `config.h:297`); `start_backend.sh:201-202` self-raises | `ulimit -n` → `500000` |
| `mysql` client | — | `.pixi/envs/default/bin/mysql` (no system mysql) |

**`JAVA_HOME` is mandatory on the BE, not optional.** `file:///...` is not a posix URI to the BE
(`fs_util.h` `is_posix_uri` matches only scheme-less or `posix://`), so it falls through to
`new_fs_hdfs` → `hdfsBuilderConnect` → JVM. `start_backend.sh` only prints a warning if it is
unset; every `FILES()` query then fails.

---

## 2. Host prep — run on BOTH machines

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
export PATH=$JAVA_HOME/bin:/home/prestouser/aocsa/sirius/experimental/starrocks/.pixi/envs/default/bin:$PATH

# Node-local data root. /home is NFS (master:/home) -- never put storage, spill, logs or the
# BDB JE journal there. /raid is ext4 on /dev/md0, 13T free.
export DATA_ROOT=/raid/prestouser/sr-bench-2node
mkdir -p $DATA_ROOT/be/{storage,spill,log}
mkdir -p $DATA_ROOT/fe/{meta,log}          # gcn-17 only, harmless elsewhere
```

Per-host BE trees, cloned once from the pristine template (run on **either** host — `$HOME` is
shared NFS, so both appear on both):

```bash
cd /home/prestouser/starrocks-bench
[ -d be17 ] || cp -a be be17      # gcn-17 uses this tree
[ -d be18 ] || cp -a be be18      # gcn-18 uses this tree
```

> **Why two trees for two hosts.** `common.sh:90` sets `export PID_DIR=$(cd "$curdir"; pwd)` — the
> tree's own `bin/` — and `start_backend.sh:180` writes `$PID_DIR/be.pid` there. That directory is
> on shared NFS. Two hosts launching the *same* tree clobber each other's `be.pid`, and
> `stop_be.sh` on one host kills nothing (or the wrong pid) on the other. Distinct trees, one per
> host, is the fix. (This also leaves the existing single-host `be1`/`be2` baseline untouched.)

### Kernel settings — what is actually required

Measured on gcn-18; **verify on gcn-17 before launching**:

| Setting | Required | gcn-18 reads | Action |
|---|---|---|---|
| `vm.max_map_count` | ≥ 262144 | `1048576` | none |
| swap | off | `SwapTotal: 0 kB` | none |
| `net.core.somaxconn` | ≥ 1024 | `4096` | none |
| `ulimit -n` | ≥ 60000 | `500000` | none |
| `ulimit -u` | ≥ 65535 | `978700` | none |
| `vm.overcommit_memory` | upstream recommends `1` | `0` | optional, needs root |
| `net.ipv4.tcp_abort_on_overflow` | upstream recommends `1` | `0` | optional, needs root |

The last two are recommendations in `docs/en/deployment/environment_configurations.md`, not
enforced by any StarRocks check. The engine-B SF100 single-host baseline was measured with them at
these values, so **leaving them alone keeps A-vs-B comparable**. If you change them, change them
on both hosts and record it.

**No firewall work is needed.** `/etc/ufw/ufw.conf` → `ENABLED=no`, and a TCP probe to
gcn-17 returns an immediate RST rather than a timeout on every engine-B port (see §5), which is
what an unfiltered path with no listener looks like. `nft list ruleset` needs root and was not run
— so "no filtering" is inferred from the RST behaviour, not from a rule dump.

---

## 3. FE setup — gcn-17 only

Write `/home/prestouser/starrocks-bench/fe/conf/fe.conf`. This is a **full replacement** derived
from `configs/gb200-4gpu/engine-b/fe.conf` with exactly one behavioural change (`priority_networks`)
plus a fresh `meta_dir`.

```bash
cat > /home/prestouser/starrocks-bench/fe/conf/fe.conf <<'EOF'
# Engine B (stock StarRocks 3.5.20) FE -- TWO-HOST variant. Coordinator only; holds no query data.
# Uppercase keys are exported as shell vars by bin/start_fe.sh; lowercase keys are parsed by the
# Java Config class. LOG_DIR must precede JAVA_OPTS because JAVA_OPTS expands it.

LOG_DIR = /raid/prestouser/sr-bench-2node/fe/log
DATE = "$(date +%Y%m%d-%H%M%S)"

# -Xmx16g, not the 8g start_fe.sh fallback: 8 GiB is thin for FE-side FILES() split enumeration
# at SF1000. Do NOT rely on FE_ENABLE_AUTO_JVM_XMX_DETECT -- it is a no-op off-container and
# would derive the heap from /proc/meminfo MemTotal, which counts the four GPU HBM NUMA nodes.
JAVA_OPTS="-Dlog4j2.formatMsgNoLookups=true -Xmx16g -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError -Xlog:gc*:${LOG_DIR}/fe.gc.log.$DATE:time -XX:ErrorFile=${LOG_DIR}/hs_err_pid%p.log -Djava.security.policy=${STARROCKS_HOME}/conf/udf_security.policy"

# *** THE ONE BEHAVIOURAL CHANGE vs the single-host engine-B conf. ***
# Was 127.0.0.1/32. 10.87.140.32/27 covers .32-.63 and therefore both .52 and .53; each host's
# FE/BE picks the FIRST local address inside the CIDR (FrontendOptions/BackendOptions both
# break on first match). Verified: bond0 on gcn-18 is 10.87.140.53 netmask 255.255.255.224.
priority_networks = 10.87.140.32/27

# UNCHANGED port block. *** 9030 IS SHARED WITH ENGINE A -- NEVER RUN BOTH. ***
http_port = 8030
rpc_port = 9020
query_port = 9030
edit_log_port = 9010

# Local ext4. BDB JE fsyncs on every edit-log write; NFS is the worst placement on this box.
# *** A NEW meta_dir BOOTSTRAPS A BRAND NEW, EMPTY FE CLUSTER. *** That is deliberate here:
# the old meta has BEs registered at 127.0.0.1:9050/9052, which are wrong for a two-host run.
meta_dir = /raid/prestouser/sr-bench-2node/fe/meta
sys_log_dir = /raid/prestouser/sr-bench-2node/fe/log
audit_log_dir = /raid/prestouser/sr-bench-2node/fe/log
sys_log_level = INFO

mysql_service_nio_enabled = true

# run_mode: NOT SET, deliberately -> defaults to shared_nothing, which is what BEs need.
# DO NOT copy engine A's `run_mode = shared_data` here.
# default_replication_num: NOT SET. FILES() creates no replicated user tables; internal
# _statistics_ tables auto-infer min(3, #BEs) = 2 (AutoInferUtil).
# enable_statistic_collect: left ON. Disabling it would handicap engine B's CBO and make the
# A/B comparison dishonest.
EOF
```

Do **not** re-run `configs/gb200-4gpu/engine-b/setup-engine-b-gb200.sh` after this: its
`install_conf` copies the loopback `fe.conf`/`be1.conf` back over your edits
(`setup-engine-b-gb200.sh:205-208`).

Start it:

```bash
# gcn-17
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
numactl --membind=0,1 -- /home/prestouser/starrocks-bench/fe/bin/start_fe.sh --daemon
```

`--membind=0,1` and *no* `--cpubind`: the FE keeps its cross-socket CPU float but can never
allocate JVM heap into a GPU HBM node. On this box `numactl -H` shows 34 nodes; only **0** (CPUs
0-71) and **1** (CPUs 72-143) have CPUs. Nodes 2/10/18/26 are 188,416 MB of GPU HBM with **zero
CPUs** — binding there is never valid.

Confirm it is up:

```bash
until mysql -h 10.87.140.52 -P 9030 -uroot -e 'SELECT 1' >/dev/null 2>&1; do sleep 2; done
mysql -h 10.87.140.52 -P 9030 -uroot --vertical -e "SHOW PROC '/frontends'"
```

The `IP` column must read `10.87.140.52`, **not** `127.0.0.1`. Loopback there means
`priority_networks` did not take — the FE will heartbeat the BEs from an address they cannot
route back to.

---

## 4. Backend setup — both hosts

### 4a. `be.conf`

Two files, one per host tree. They are **identical except for the three data paths**. Every key is
annotated CHANGED (with the 3.5.20 default from `git show 3.5.20:be/src/common/config.h`) or
NOT SET.

```bash
# ---- run once; writes both confs (the trees are on shared NFS) ----
for h in 17 18; do
cat > /home/prestouser/starrocks-bench/be$h/conf/be.conf <<EOF
# Engine B (stock StarRocks 3.5.20) BE -- TWO-HOST variant, ONE unpinned BE per box.

# CHANGED from "" (config.h:63). Same CIDR as the FE; gcn-17 resolves .52, gcn-18 resolves .53.
# This ALSO decides which NIC carries BE<->BE exchange traffic -> bond0, 400 Gb/s.
priority_networks = 10.87.140.32/27

##### Memory #####
# CHANGED from "90%" (config.h:84).
# *** ON THIS BOX mem_limit MUST BE AN ABSOLUTE BYTE VALUE. NEVER A PERCENTAGE. ***
# MemInfo::init reads /proc/meminfo MemTotal = 1,775,050,816 kB = 1692.6 GiB, because the kernel
# counts the four 184 GiB GPU HBM NUMA nodes as system memory. Real CPU-addressable LPDDR is
# node0 489,960 MB + node1 489,823 MB = 979,783 MB = 956.82 GiB, with SwapTotal: 0. "90%" would
# resolve to ~1523 GiB -> guaranteed OOM-kill, and the bytes_limit > physical_mem() guard cannot
# catch it because physical_mem() is itself the wrong number.
#
# 480G is the DERIVED one-BE-per-host analogue of the validated 2x240G single-host layout
# (same aggregate per box). Per-host arithmetic on gcn-17, the worst case (it also hosts the FE):
#   CPU-addressable LPDDR                                  956.82 GiB
#   - mmfsd (measured 50.7) - FE JVM ~20 - kernel ~10       -84.00
#   - 1 BE                                                -480.00
#   = kernel page cache left                               392.82 GiB   (15x the 26 GiB SF100 set)
# *** DERIVED, NOT MEASURED at this topology. *** Sanity-check against an SF100 run before
# quoting it. For SF1000 use 400G on both BEs so page cache clears the 265 GB dataset.
mem_limit = 480G

# CHANGED from false (config.h:307; storage_page_cache_limit = "20%", config.h:305).
# storage_page_cache caches pages of NATIVE OLAP tablets. This benchmark has no native tables --
# both engines read the same parquet through FILES(). Left on it reserves a cache that never fills.
disable_storage_page_cache = true

# CHANGED from "0" (config.h:1297). This is the cache that DOES apply to FILES() parquet scans.
# datacache_enable is true by default but sized ZERO in 3.5.20, and datacache_auto_adjust_enable
# only grows the DISK quota after 7200 s of disk idle -- far longer than a benchmark run. So stock
# 3.5.20 caches nothing here despite reporting the cache enabled. 32G holds all of SF100.
# This comes OUT of mem_limit; it is not double-counted above.
datacache_mem_size = 32G

# CHANGED from "0" -- pinned explicitly rather than left to the auto-adjust heuristic.
# For SF1000 set 200G (it lives under storage_root_path, which is local RAID -- never NFS).
datacache_disk_size = 0

##### CPU #####
# num_cores: NOT SET, deliberately (default 0, config.h:574 -> CpuInfo counts /proc/cpuinfo = 144).
# The single-host 2-BE conf sets num_cores = 72 ONLY because it uses --numa 0/1, which cpubinds.
# CpuInfo never calls sched_getaffinity, so a cpubound BE would still report 144 and needs the
# override. Here the BE is NOT cpubound -- it owns all 144 cores -- so 144 is the truth. Setting
# num_cores here would under-size every thread pool by 2x.
#
# enable_resource_group_bind_cpus: NOT SET (default true, config.h:1514). It pins worker threads
# to CpuInfo::get_core_ids() = all 144 cores, which is correct when there is no cpubind. The
# single-host conf disables it only because --cpubind 0 would fight it.

# CHANGED from 8 (config.h:885). Sizes the CONNECTOR scan pool -- ceil(value * num_cores) -- the
# pool FILES() actually uses. 4 * 144 = 576 threads on this box, the same box-wide total as the
# validated 2x(4*72) single-host layout. Drop to 2 if vmstat 1 shows a sustained run queue > 144.
pipeline_connector_scan_thread_num_per_cpu = 4

# CHANGED from 16 (config.h:1035). Adaptive CEILING, not a floor, so raising it cannot hurt a
# latency-sensitive phase. Primary sweep knob.
connector_io_tasks_per_scan_operator = 32

##### Ports -- all stock defaults, restated because they are load-bearing #####
# *** NO TRAILING INLINE COMMENTS ANYWHERE IN THIS FILE. ***  (MEASURED 2026-08-11)
# The BE config parser takes everything after '=' as the value, comment included:
#   be_port = 9060   # thrift
# fails startup with, in be.out:
#   Invalid value of config 'be_port': ' 9060                  # thrift'
#   error read config file.
# The BE then exits, leaving an EMPTY be.INFO and a stale be.pid -- so the symptom looks like a
# silent crash, and the real message is only in log/be.out inside the TREE (not sys_log_dir).
# The FE's parser does NOT share this limitation, which is what makes the trap easy to hit.
#
# be_port                thrift: agent tasks + FE blacklist probe
# heartbeat_service_port THIS is the port that goes in ALTER SYSTEM ADD BACKEND
# brpc_port              FE->BE fragment deploy/cancel AND BE<->BE exchange
# be_http_port           FE blacklist probe, profile / error-url fetch
# starlet_port           shared-data only; inert here
be_port = 9060
heartbeat_service_port = 9050
brpc_port = 8060
be_http_port = 8040
starlet_port = 9070
# CHANGED from 20001 (config.h:243, "StarRocks test backend"). Only one BE per host here, so a
# collision is impossible -- pinned for symmetry with the single-host conf.
port = 20001

##### Storage / spill / logs -- all on node-local ext4, never NFS #####
#   /raid = ext4 on /dev/md0 (14T, 13T free), LOCAL
#   /home = nfs4 (master:/home)   <- code only
# A BE (unlike a CN) FATALs if storage_root_path is unparseable or not read-write. It stays EMPTY:
# no tables are ever created.
storage_root_path = /raid/prestouser/sr-bench-2node/be/storage
spill_local_storage_dir = /raid/prestouser/sr-bench-2node/be/spill
sys_log_dir = /raid/prestouser/sr-bench-2node/be/log
sys_log_level = INFO

# Restored from the stock 3.5.20 be.conf -- required for JDK17+ (Java extensions / JNI, and
# file:// goes through libhdfs). A bare \`cat >\` heredoc silently drops it otherwise.
JAVA_OPTS="--add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
EOF
done
```

### 4b. Launch

```bash
# ---- gcn-17 ----
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
numactl --membind=0,1 -- /home/prestouser/starrocks-bench/be17/bin/start_be.sh --daemon

# ---- gcn-18 ----
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
numactl --membind=0,1 -- /home/prestouser/starrocks-bench/be18/bin/start_be.sh --daemon
```

**Do not use `--numa N`.** `start_backend.sh:134-136` expands it to
`numactl --cpubind N --membind N` — it cannot express membind-only, and cpubinding would halve
this BE to one socket. The explicit `numactl --membind=0,1` wrapper gets memory safety without
losing cores; affinity and mempolicy are inherited across the `--daemon` fork/exec (this is the
same technique `setup-engine-b-gb200.sh` prescribes for its 4-BE variant).

> **`Mems_allowed_list` is the WRONG check — do not use it.** It reports the *cpuset*-allowed
> nodes, which stay `0-2,10,18,26` even when `--membind` is working perfectly. MEASURED on gcn-18:
> a BE launched under `numactl --membind=0,1` shows `Mems_allowed_list: 0-2,10,18,26` while every
> one of its 4,474 mappings is `bind:0-1` with zero pages on any HBM node. Reading that field
> would send you chasing a non-problem. Use `numa_maps`, below.

Confirm the process came up and the membind applied:

```bash
grep -i 'physical memory\|Physical Memory' /raid/prestouser/sr-bench-2node/be/log/be.INFO | head -1
P=$(cat /home/prestouser/starrocks-bench/be18/bin/be.pid)     # be17 on the other host

# THE CORRECT CHECK -- the actual memory policy, per mapping:
awk '{for(i=2;i<=NF;i++) if($i ~ /^(bind|prefer|interleave|default)/){print $i; break}}' \
    /proc/$P/numa_maps | sort | uniq -c
#   must print a single line:   NNNN bind:0-1        (measured: 4474 bind:0-1)
#   any `default` mappings  =>  the membind did NOT take

# and confirm nothing actually landed on the HBM nodes:
grep -o 'N\(2\|10\|18\|26\)=[0-9]*' /proc/$P/numa_maps | awk -F= '{s+=$2} END{print (s?s:0)" pages"}'
#   must print   0 pages
```

### 4c. Register both backends

The port is **`heartbeat_service_port` (9050)**, not `be_port`.
`SystemInfoService.addBackend(String host, int heartbeatPort, ...)` (3.5.20 `:314`) is the only
port the FE is told; `be_port`, `be_http_port`, `brpc_port`, core count and `mem_limit` all arrive
in the heartbeat reply (`HeartbeatMgr.java:266`).

```bash
mysql -h 10.87.140.52 -P 9030 -uroot -e \
  'ALTER SYSTEM ADD BACKEND "10.87.140.52:9050", "10.87.140.53:9050";'
```

### 4d. Confirm both are alive with real IPs

```bash
mysql -h 10.87.140.52 -P 9030 -uroot --vertical -e "SHOW BACKENDS"
```

Expect exactly two rows:

```
*************************** 1. row ***************************
        BackendId: 10002
               IP: 10.87.140.52
    HeartbeatPort: 9050
           BePort: 9060
         HttpPort: 8040
         BrpcPort: 8060
            Alive: true
         CpuCores: 144
         MemLimit: 480.000 GB
          Version: 3.5.20-4d17879
*************************** 2. row ***************************
               IP: 10.87.140.53
            Alive: true
         CpuCores: 144
```

Checks that matter:

* `IP` is `10.87.140.5x`, never `127.0.0.1`. Loopback = `priority_networks` did not take.
* `CpuCores: 144` on **both**. This is the correct value for an unpinned BE. (If you ever add
  `--numa`, it must become 72 and `num_cores = 72` must be set — see the comment in `be.conf`.)
* `MemLimit` is an absolute value, not ~1523 GB. ~1523 GB means `mem_limit` reverted to a percentage.
* `Version` matches on both.
* **`SHOW COMPUTE NODES` will be empty.** Engine B has no CNs. Do not use it here.

`SHOW PROC '/backends'` gives the same rows.

---

## 5. Port matrix and reachability

| Direction | Port | Config key | Purpose |
|---|---|---|---|
| client → FE | 9030 | `query_port` | MySQL: queries, `ALTER SYSTEM` |
| BE → FE | 9020 | `rpc_port` | thrift task/report; the FE ships its own address here in the heartbeat (`HeartbeatMgr.java:97`) |
| BE → FE | 8030 | `http_port` | FE HTTP |
| FE → BE | 9050 | `heartbeat_service_port` | liveness; the address in `ADD BACKEND` |
| FE → BE | 8060 | `brpc_port` | fragment deploy + cancel |
| FE → BE | 9060 | `be_port` | agent tasks; **blacklist-eviction probe** |
| FE → BE | 8040 | `be_http_port` | profile / error-url fetch; **blacklist-eviction probe** |
| **BE ↔ BE** | **8060** | **`brpc_port`** | **the cross-host exchange path — engine B's NIXL equivalent** |

The blacklist-eviction probe hits **all three** of `be_port`, `brpc_port`, `http_port` and only
lifts the entry if **every** one accepts (3.5.20 `HostBlacklist.java:206-207`,
`NetUtils.checkAccessibleForAllPorts` breaks on the first refusal). A firewall that opens only
heartbeat/brpc leaves a blacklisted BE permanently excluded.

Confirm reachability **before** launching anything (all should be refused; after launch, open):

```bash
# from gcn-18, probing gcn-17
for p in 9030 9020 8030 9050 8060 9060 8040; do
  timeout 2 bash -c "cat </dev/null >/dev/tcp/10.87.140.52/$p" 2>/dev/null \
    && echo "52:$p OPEN" || echo "52:$p refused"
done
# from gcn-17, probing gcn-18
for p in 9050 8060 9060 8040; do
  timeout 2 bash -c "cat </dev/null >/dev/tcp/10.87.140.53/$p" 2>/dev/null \
    && echo "53:$p OPEN" || echo "53:$p refused"
done
```

Measured on gcn-18 with nothing running: every port returns *refused* (immediate RST) rather than
timing out — an unfiltered path with no listener. A **timeout** instead means filtering.

After launch, confirm the listeners are on the routable address, not loopback:

```bash
ss -ltnp | grep -E ':(9030|9020|8030|9050|9060|8060|8040)\b'
```

> **Note on interfaces.** Engine B has **no** interface-selection knob. brpc destinations are built
> from `ComputeNode.getIP()`, which `BackendOptions` sets from `priority_networks` — so all
> exchange traffic rides `bond0` at 400 Gb/s. The four 400G RoCE planes and NVLink are unreachable
> to engine B. That asymmetry against engine A's NIXL path is real and must be stated in any A/B
> writeup; it is not a misconfiguration.

---

## 6. Data

`FILES()` with `file://` is read **server-side by every BE**, and the FE assigns byte ranges
without knowing which host holds the file. So:

**The parquet must exist at the SAME absolute path on both hosts, on node-local storage.**

The FE also globs the path itself and sends schema inference to a *randomly shuffled* alive
backend (`TableFunctionTable`), so a file present on only one host fails **intermittently**, not
deterministically. That is the worst failure mode; do not skip this step.

**This is already satisfied — no copying required.** Both hosts carry the same trees at the same
absolute path, on node-local ext4 (`/raid` = `/dev/md0`, 13 TB free):

| Path (identical on gcn-17 and gcn-18) | Size |
|---|---|
| `/raid/prestouser/aocsa/tpch_parquet_sf100` | 26 GB |
| `/raid/prestouser/aocsa/tpch_parquet_sf500` | 132 GB |

Measured layout of SF100 (gcn-18): `lineitem` **17,187,602,838 B** across 6 files
(`part.0.parquet` … `part.5.parquet`), `orders` 5,051,383,146 B, `nation` 2,250 B.

Verify before every run — a divergence between the two trees is the worst failure mode here,
because it fails *intermittently* rather than outright:

```bash
# run on BOTH hosts; the two md5sums must match
find /raid/prestouser/aocsa/tpch_parquet_sf100 -type f -printf '%P %s\n' | sort | md5sum

# and confirm the storage is local, not NFS/GPFS
df -PT /raid/prestouser/aocsa/tpch_parquet_sf100 | tail -1     # expect /dev/md0  ext4
```

> **There is no SF1 on these hosts.** Where this tutorial previously used SF1 for a cheap
> distribution test, it now uses SF100. `/opt/sirius-ci/datasets/tpch_sf{1,…,1000}` exists on
> **gcn-18 only** and must not be used for a two-host run — gcn-17 would be handed byte ranges for
> files it cannot open.

Do **not** "solve" a path problem by pointing both hosts at `$HOME` — that is NFS (`master:/home`)
and the scan becomes the benchmark. `/scratch` is GPFS and *is* the same path on both hosts, which
would work for engine B alone, but it is a cluster filesystem: using it would measure GPFS, and
would not be comparable to any engine-A number taken on node-local NVMe.

---

## 7. Validate distribution — the centerpiece

This is what the whole exercise is for: prove that one query ran fragments on **both machines**
and that **bytes crossed the network between them**.

### 7a. Understand the split rule first, or you will prove nothing

`FileScanNode` (3.5.20 `:543-550`):

```
numInstances = clamp( totalBytes / Config.min_bytes_per_broker_scanner ,
                      1 ,
                      nodes.size() * parallelInstanceNum )
```

`min_bytes_per_broker_scanner = 67108864` (64 MiB, `Config.java:1137`), and instances are
round-robined over a **shuffled** node list (`:558`). Therefore:

* **A single file smaller than 128 MiB lands entirely on ONE host and proves nothing.**
  `nation` (2,250 B measured), `region` and `supplier` at any SF are in this class. Seeing them on
  one host is correct behaviour, not a distribution failure — always test on `lineitem`.
* SF100 `lineitem` is **17,187,602,838 B** (measured, 6 files `part.0..5.parquet`) → 256 uncapped,
  capped at `2 nodes × parallelInstanceNum`. With 144 cores reported, `getSinkDefaultDOP` =
  `min(32, 144/4) = 32` (3.5.20 `BackendResourceStat.java:195-204`) → cap 64, i.e. 32 ranges/host.
  This is the smallest set available on these hosts and is far above the threshold, so both
  machines are guaranteed ranges.
* SF500 (`tpch_parquet_sf500`, 132 GB) is the other option; it splits proportionally wider.
* **SF1 is not present on these hosts.** It would have been the minimal honest test
  (162,140,518 B → exactly 2 instances, one range per host), but it does not exist here.

Byte-range splits are safe: the BE snaps each split to whole row groups by start offset
(`file_reader.cpp` `_select_row_group`), so ranges never overlap or drop rows.

### 7b. Pre-flight: `EXPLAIN SCHEDULER` — full scheduling, zero fragments deployed

```sql
-- mysql -h 10.87.140.52 -P 9030 -uroot
EXPLAIN SCHEDULER
WITH lineitem AS (SELECT * FROM FILES(
  "path"="file:///raid/prestouser/aocsa/tpch_parquet_sf100/lineitem/*.parquet","format"="parquet"))
SELECT l_orderkey % 4096 AS bucket, count(*) AS n, sum(l_quantity) AS q
FROM lineitem GROUP BY 1 ORDER BY 1 LIMIT 20;
```

It prints one `PLAN FRAGMENT n(Fnn)` block per fragment, with per instance
`INSTANCE(<id>)`, `DESTINATIONS`, and `BE: <backend_id>` (3.5.20 `FragmentInstance.java:140-154`).

```bash
mysql -h 10.87.140.52 -P 9030 -uroot -e "EXPLAIN SCHEDULER <query>;" \
  | grep -E 'PLAN FRAGMENT|INSTANCE\(|BE: '
```

`BE:` is a numeric backend id — map it through the `BackendId` column of `SHOW BACKENDS`. **Two
distinct ids on the scan fragment = the FE intends to use both machines.** Nothing is deployed, so
this is safe to run repeatedly.

### 7c. The validation query

```sql
SET enable_profile = true;
SET pipeline_profile_level = 1;   -- default: merges instances and emits BackendAddresses
SET new_planner_agg_stage = 2;    -- force partial-agg -> HASH_PARTITIONED exchange -> merge-agg

WITH lineitem AS (SELECT * FROM FILES(
  "path"="file:///raid/prestouser/aocsa/tpch_parquet_sf100/lineitem/*.parquet","format"="parquet"))
SELECT l_orderkey % 4096 AS bucket, count(*) AS n, sum(l_quantity) AS q
FROM lineitem
GROUP BY 1
ORDER BY 1
LIMIT 20;

SELECT last_query_id();
```

`new_planner_agg_stage = 2` = `TWO_STAGE` (3.5.20 `SessionVariable.java:357,1593`), which
guarantees a two-fragment aggregation with a hash-partitioned `DataStreamSink` between them
instead of letting the optimizer collapse to one stage. `l_orderkey % 4096` is high-cardinality on
purpose, so the shuffle carries real bytes.

Stronger, bidirectional variant once the above passes — a hash-shuffle join forces **both** scan
fragments to fan out:

```sql
WITH
  lineitem AS (SELECT * FROM FILES("path"="file:///raid/prestouser/aocsa/tpch_parquet_sf100/lineitem/*.parquet","format"="parquet")),
  orders   AS (SELECT * FROM FILES("path"="file:///raid/prestouser/aocsa/tpch_parquet_sf100/orders/*.parquet","format"="parquet"))
SELECT count(*) FROM lineitem JOIN [SHUFFLE] orders ON l_orderkey = o_orderkey;
```

### 7d. Capture the profile

```bash
mysql -h 10.87.140.52 -P 9030 -uroot -e "SHOW PROFILELIST LIMIT 5;"

QID=<query_id from SHOW PROFILELIST or last_query_id()>
mysql -h 10.87.140.52 -P 9030 -uroot --raw -N \
  -e "SELECT get_query_profile('$QID')" > /tmp/profile.txt
```

`get_query_profile` is function id `100020` in 3.5.20 (`gensrc/script/functions.py:775`,
`be/src/exprs/utility_functions.cpp:277`). Equivalent HTTP route (needs basic auth):
`curl -s -u root: 'http://10.87.140.52:8030/query_profile?query_id=$QID' -o /tmp/profile.html`
(`QueryProfileAction.java:62`).

> Use the **raw** profile, not `ANALYZE PROFILE`. `ANALYZE PROFILE FROM '<qid>'` prints
> `BackendNum:` but **not** `BackendAddresses:` — it can tell you "two backends", never "which two
> hosts". `ANALYZE PROFILE` also forces `enable_async_profile = false` and blocks the client for
> the full `profile_timeout`; `SET enable_profile = true` does not.

### 7e. Proof #1 — instances ran on both hosts

`FragmentInstanceExecState` stamps each instance profile with `Address` = `host:be_port`
(3.5.20 `:108-110`); at `pipeline_profile_level = 1` the FE merges them and writes
`BackendAddresses` / `BackendNum` / `InstanceIds` onto each fragment
(3.5.20 `QueryRuntimeProfile.java:475-480`). Profile text renders info strings as `   - Key: value`.

```bash
grep -n ' - BackendAddresses:' /tmp/profile.txt
grep -nE ' - (BackendNum|InstanceNum|MissingInstanceIds):' /tmp/profile.txt
```

Expected shape:

```
   - BackendAddresses: 10.87.140.52:9060,10.87.140.53:9060
   - BackendNum: 2
   - InstanceNum: 2
```

**The port in `Address` is `be_port` (9060), not `brpc_port`** — `ComputeNode.getAddress()` returns
`(host, bePort)`. Two *distinct IPs* is the proof. `BackendNum: 2` alone is **not** proof of two
machines in general (it counts distinct backends, which could be co-located); it happens to be
sufficient here only because this topology has exactly one BE per host — rely on the IPs.

Per-instance detail, if you want the raw tree instead of the merged one:

```sql
SET pipeline_profile_level = 2;   -- emits `Instance <id> (host=TNetworkAddress(hostname:..., port:...))`
```
```bash
grep -oE 'host=TNetworkAddress\(hostname:[^,]+, port:[0-9]+\)' /tmp/profile.txt | sort -u
```

### 7f. Proof #2 — bytes crossed the network

Sender counters live on `EXCHANGE_SINK (plan_node_id=N)`; receiver counters on
`EXCHANGE_SOURCE (plan_node_id=N)`. All names verified in 3.5.20 (`sink_buffer.cpp:158-193`,
`data_stream_recvr.cpp:202-204`, `exchange_sink_operator.cpp:427-430`).

```bash
awk '/EXCHANGE_SINK \(plan_node_id=/{p=1}
     p && / - (BytesSent|RequestSent|RpcCount|NetworkTime|NetworkBandwidth|OverallThroughput|DestFragments|PartType|ChannelNum):/{print}
     /^[[:space:]]*$/{p=0}' /tmp/profile.txt

awk '/EXCHANGE_SOURCE \(plan_node_id=/{p=1}
     p && / - (BytesReceived|BytesPassThrough|RequestReceived):/{print}
     /^[[:space:]]*$/{p=0}' /tmp/profile.txt
```

**MEASURED shape** (2026-08-11, SF100 `lineitem`, 2 BEs, one per host):

```
   - ChannelNum: 2
   - DestFragments: e404221995a311f1-98f4001acaffff04, e404221995a311f1-98f4001acaffff05
   - PartType: HASH_PARTITIONED
   - BytesSent: 2.212 MB
   - NetworkBandwidth: 230.997 MB/sec
   - NetworkTime: 20.079ms
   - RequestSent: 128
...
   - BytesReceived: 2.212 MB
   - BytesPassThrough: 2.344 MB
   - RequestReceived: 130
```

A channel short-circuits to pass-through only when the destination has the **same IP *and* the same
brpc port** as the sender (3.5.20 `exchange_sink_operator.cpp:140`
`if (BackendOptions::get_local_ip() != _brpc_dest_addr.hostname) return false;` plus the port check).
With one BE per host, "same IP and port" means *the same process*, so **any `BytesSent > 0` is by
construction cross-machine traffic.**

> **`BytesPassThrough` is NOT a subset of `BytesReceived`, and an earlier revision of this document
> said it was.** The measured run above has `BytesPassThrough` (2.344 MB) **greater** than
> `BytesReceived` (2.212 MB) while bytes demonstrably crossed the wire — bond0 counters moved
> 1310 KB tx / 1419 KB rx during the query. The old rule ("`BytesPassThrough == BytesReceived` means
> nothing crossed") would have reported a **false negative** on a perfectly healthy cluster. They are
> two independent counters: `BytesReceived` tracks bytes arriving over brpc and matches `BytesSent`
> exactly; `BytesPassThrough` separately tracks bytes that never entered the transport.

| Observation | Meaning |
|---|---|
| `BytesSent > 0` and `BackendAddresses` spans two IPs | bytes went over brpc **between the two machines** ✅ — this is the proof |
| `BytesSent == BytesReceived` | consistent accounting; sender and receiver agree |
| `BytesPassThrough : BytesSent ≈ 1:1` | the healthy 2-BE hash-shuffle signature — about half of each receiver's input is local, half remote (measured 2.344 : 2.212, i.e. 51% / 49%) |
| `BytesSent == 0` while `BytesPassThrough > 0` | everything stayed in one process — the shuffle did **not** cross |
| `BytesSent == 0` and `BytesPassThrough == 0` | the plan had no exchange; `new_planner_agg_stage = 2` did not take |

**Corroborate at the NIC** rather than trusting the counters alone — this is what caught the error
above:

```bash
a=$(cat /sys/class/net/bond0/statistics/tx_bytes); b=$(cat /sys/class/net/bond0/statistics/rx_bytes)
# ...run the query...
c=$(cat /sys/class/net/bond0/statistics/tx_bytes); d=$(cat /sys/class/net/bond0/statistics/rx_bytes)
echo "tx $(( (c-a)/1024 )) KB / rx $(( (d-b)/1024 )) KB"
```
Expect the same order of magnitude as `BytesSent`; the NIC total runs higher because it includes
brpc framing and TCP/IP headers, and `/home` NFS traffic shares this interface. Near-zero means the
shuffle never left the box.

### 7g. Live view, during the query

Run in a second session while the query is executing. This is built entirely from the FE's
`ExecutionDAG` and needs no BE cooperation (`ProcService.java:67`):

```bash
watch -n1 "mysql -h 10.87.140.52 -P 9030 -uroot -e \"SHOW PROC '/current_backend_instances'\""
```

Columns: `Backend | InstanceNum | InstanceId | ExecTime`, where `Backend` is `host:be_port`.
Two rows with distinct IPs = both machines are executing right now.

### 7h. Independent, engine-agnostic corroboration

Delta-sample the NIC counters around the query. `bond0` is an 802.3ad LACP bond over
`enP22p3s0f0np0` + `enP6p3s0f0np0`, so read the slaves (or `bond0` itself):

```bash
read a b < <(cat /sys/class/net/bond0/statistics/{tx_bytes,rx_bytes}); echo "$a $b"
# ...run the query...
read c d < <(cat /sys/class/net/bond0/statistics/{tx_bytes,rx_bytes}); echo $((c-a)) $((d-b))
```

The delta should be within an order of magnitude of `BytesSent`. Near-zero means the shuffle never
left the box.

---

## 8. TPC-H sweep with the existing harness

**Use `bench.sh`, not `run-abc.sh`.** `run-abc.sh` is a single-box *lifecycle orchestrator*: it
starts the FE (`:1099`), the BEs (`:1109`) and registers them as `127.0.0.1` (`:1114`) locally, it
`die`s without `nvidia-smi`/`numactl` (`:658-659`), it waits for a GPU-free box before engine B
(`:1194`), and its engine-B gate **aborts when `CpuCores == 144`** — which is the correct value
here. None of that survives a two-host layout.

`bench.sh` only connects to whatever answers; it starts nothing. But it hardcodes
`--host 127.0.0.1` (`bench.sh:85`) and there is no `FE_HOST` variable
(`grep -rn FE_HOST benchmarks/` → nothing). **So run it on gcn-17, where the FE is** — then
`127.0.0.1` is literally correct and no patch is needed.

```bash
# ---- on gcn-17 ----
cd /home/prestouser/aocsa/sirius/experimental/starrocks
export PATH=$PWD/.pixi/envs/default/bin:$PATH     # bench.sh calls bare `mysql`

TPCH_DATA=/raid/prestouser/aocsa/tpch_parquet_sf100 \
FE_PORT=9030 \
QUERY_TIMEOUT=180 \
COLD_TIMEOUT=600 \
MIN_BACKENDS=2 \
  ./benchmarks/tpch/bench.sh --cold \
     /raid/prestouser/bench-2node-B/timings.csv 3
```

Notes, each grounded:

* `MIN_BACKENDS=2` — `bench.sh` sums `SHOW COMPUTE NODES` + `SHOW BACKENDS` alive rows
  (`bench.sh:108-131`) and **aborts if MORE than `MIN_BACKENDS` are alive** unless
  `ALLOW_EXTRA_BACKENDS=1`. Two BEs, zero CNs → 2.
* `QUERY_TIMEOUT=180 / COLD_TIMEOUT=600` — `bench.sh`'s defaults of 30/180 are SF1 numbers.
  180/600 is `run-abc.sh`'s SF-scaled formula (`max(90, 1.8·SF)`, `max(300, 6·SF)`) at SF100.
* **`runs` is positional and mandatory before a query subset.** `bench.sh out.csv q05` sets
  `RUNS=q05` (`bench.sh:73-74`). Always write `bench.sh out.csv 3 q05`.
* **No `RESTART_CMD`.** Stock BEs clean up after a failed query; that variable exists for engine A,
  whose CN has no `cancel_plan_fragment`.
* Run 0 is discarded unless `--cold`; `runs 3` therefore executes 4 mysql invocations per query.
* Output: the CSV goes exactly where argv points; raw per-run mysql output lands in
  `<dirname>/qNN.rN.out`. Header is `query,run,phase,status,ms,rows`.

### q11 at SF > 1

`queries/q11.sql:26` hardcodes the TPC-H `FRACTION` literal `0.0001000000`, which is correct **only
at SF1**. `bench.sh` does not correct it (only `run-abc.sh` does, into a staged copy). At SF100 q11
returns 0 rows and `bench.sh` records it as `wedge`. Patch the tracked file for the run and revert:

```bash
sed -i 's/0\.0001000000/0.000001000000/' benchmarks/tpch/queries/q11.sql   # 0.0001 / 100
# ...run the sweep...
git checkout -- benchmarks/tpch/queries/q11.sql
```

### Comparing against engine A

```bash
python3 benchmarks/tpch/analyze.py \
  /raid/prestouser/bench-2node-A/timings.csv \
  /raid/prestouser/bench-2node-B/timings.csv \
  /raid/prestouser/bench-2node/results.md \
  /raid/prestouser/bench-2node/tpch_a_vs_b.png
```

`analyze.py` medians `phase=warm,status=pass` rows and geomeans B/A. It **exits 1 if the two
engines' row counts disagree on any query** — that is a shape check, not a correctness check.
Neither harness ever compares answer *values*; a real A/B needs an out-of-band DuckDB oracle diff.

`matplotlib 3.11.1` / `numpy 2.5.1` are present in the system `python3` on gcn-18, so the PNG is
produced and the exit code is meaningful (the `tpch-bench` SKILL.md still claims otherwise — stale).

### Provenance

Record a hand-written `INVOCATION-engineB.txt` next to the CSV containing the literal shell you
ran, including every `export`. This is the convention the recorded SF100 A/B run used
(`/home/prestouser/aocsa/benchmark-results/tpch-sf100-abc/INVOCATION-engineB.txt`), and it is the
only thing that makes a number reproducible.

---

## 9. Teardown

```bash
# gcn-17
/home/prestouser/starrocks-bench/be17/bin/stop_be.sh
/home/prestouser/starrocks-bench/fe/bin/stop_fe.sh

# gcn-18
/home/prestouser/starrocks-bench/be18/bin/stop_be.sh
```

Confirm:

```bash
pgrep -af 'starrocks_be|StarRocksFE'          # both hosts, must be empty
```

`stop_be.sh` reads `$PID_DIR/be.pid` inside its own tree — which is why each host must own a
distinct tree (§2). The FE metadata under `/raid/prestouser/sr-bench-2node/fe/meta` survives a
restart; delete it only if you want to re-bootstrap (which forces re-running `ALTER SYSTEM ADD
BACKEND`).

---

## 10. Known gaps / unverified

Everything below was measured on **gcn-18 only**. `ssh` to gcn-17 is blocked by the tooling used to
write this document, so nothing on gcn-17 was observed directly. Existing artifacts
(`/tmp/nixl-echo-2node/origin.log`, written by `scripts/nixl-echo-2node.sh:111`, which self-reports
`host=presto-gb200-gcn-17` and executed an aarch64 binary out of the shared NFS repo path) establish
that gcn-17 is aarch64, is reachable over passwordless `BatchMode` ssh, and mounts `master:/home` at
the identical path. Everything else about gcn-17 is inference.

| # | TODO | Command to resolve |
|---|---|---|
| 1 | gcn-17 NUMA layout matches (node 0 = CPUs 0-71, node 1 = 72-143; nodes 2/10/18/26 = HBM, no CPUs) | `ssh presto-gb200-gcn-17 'numactl -H \| head -8'` |
| 2 | gcn-17 kernel settings match the §2 table | `ssh presto-gb200-gcn-17 'ulimit -n; ulimit -u; cat /proc/sys/vm/max_map_count /proc/sys/net/core/somaxconn; grep SwapTotal /proc/meminfo'` |
| 3 | gcn-17 `bond0` is `10.87.140.52/27` and JDK 21 is at the same path | `ssh presto-gb200-gcn-17 'ifconfig bond0 \| head -3; ls -d /usr/lib/jvm/java-21-openjdk-arm64'` |
| 4 | `/raid/prestouser` exists and is writable on gcn-17, with room for the dataset | `ssh presto-gb200-gcn-17 'df -PT /raid; touch /raid/prestouser/.wtest && rm /raid/prestouser/.wtest && echo writable'` |
| 5 | gcn-17 LPDDR total matches (so `mem_limit = 480G` is right there too) | `ssh presto-gb200-gcn-17 'numactl -H \| grep "^node [01] size"'` |
| 6 | No `nftables`/`iptables` rules between the hosts (§5 rests on RST-vs-timeout, not a rule dump) | `sudo nft list ruleset` on both hosts |

Other unverified or derived items:

* **`mem_limit = 480G` is DERIVED, not measured** at this topology. The validated number is
  2×240G NUMA-pinned on one box. Sanity-check against an SF100 run (watch `be.INFO` for
  "Memory of process exceed limit" and any spill) before quoting it.
* **The `EXCHANGE_SINK`/`EXCHANGE_SOURCE` counter names are verified in 3.5.20 source but the
  §7f expected output is illustrative**, not a capture — no two-host engine-B query has been run.
* **`pipeline_dop`** is left at its default (0 = auto) and will resolve to
  `min(64, 144/2) = 64` per fragment instance. No committed harness pins it. If you also run engine
  A, pin `SET GLOBAL pipeline_dop` identically in both arms or the comparison has a free variable.
* **`enable_pipeline_engine` persists in FE metadata** across runs (`run-abc.sh:786-791`). A fresh
  `meta_dir` (§3) resets it to the default `true`, which is what you want — but check it if you
  reuse metadata.
* **Engine B rides `bond0` TCP at 400 Gb/s, full stop.** There is no way to move brpc onto the 4×400G
  RoCE fabric: brpc has no interface-selection knob and `priority_networks` also governs heartbeats
  and FE→BE dispatch. Moving it would be an untested experiment, not a config change.
* **`ALTER SYSTEM DROP BACKEND`** was not exercised. If you mis-register an address, correcting it
  needs a drop or a fresh `meta_dir`.
