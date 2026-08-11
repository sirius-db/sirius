# 2NODE bring-up — execution record

Authority: [2NODE-BRINGUP-PLAN.md](2NODE-BRINGUP-PLAN.md). This file records what was actually
executed, on **gcn-18 only** (`10.87.140.53`), date 2026-08-11.

**Session constraints.** `ssh` is a user DENY rule, so gcn-17 (`10.87.140.52`) was never contacted;
every gcn-17 fact in the plan remains inference. No cluster, FE, CN, BE, benchmark sweep or GPU
workload was started. Nothing was pushed. Verified at the end of the session:

```
$ ps -eo pid,args --no-headers | grep -E 'sirius-starrocks-cn|StarRocksFE|starrocks_be' | grep -v grep
(no output)
```

---

## Status table

| Plan task | Status | Artifact | Evidence |
|---|---|---|---|
| 1.1 steps 1-3 — FE `priority_networks` → control LAN | **DONE (uncommitted)** | `conf/fe.conf:85`, propagated to `starrocks/output/fe/conf/fe.conf` | §A |
| 1.1 step 3b — FE identity migration | **ADDED TO PLAN, not executed** | plan Task 1.1 step 3b | §D defect 1 |
| 1.1 step 4 — start the FE, confirm advertised IP | **BLOCKED — requires gcn-17** | — | §C |
| 1.2 — cross-host env overlay | **DONE, committed `ec10f8f4`** | `configs/gb200-4gpu/engine-a-2host.env` | §B |
| 1.3 / 1.4 — launch CN0 on each host | **BLOCKED — requires gcn-17 + cluster launch** | plan text corrected (`--engine-dir`) | §C, §D defect 2 |
| 1.5 / 1.6 / 1.7 — registration, distribution proof, SF100 sweep | **BLOCKED** | — | §C |
| 2.1 — two-host CN launcher | **DONE, committed `8eced0f1`** | `benchmarks/cn-2host.sh` | §B |
| 2.2 – 2.5 — 8-CN bring-up, transport, distribution, sweep | **BLOCKED** | — | §C |
| 0.1 – 0.3 — gcn-17 facts, dataset replication, standalone data plane | **BLOCKED** | — | §C |
| 3.x — engine B reference | **BLOCKED** | — | §C |

Local commits on `demo-multi-cn`, **not pushed** (`## demo-multi-cn...origin/demo-multi-cn [ahead 2]`):

```
8eced0f1 feat(bench): two-host CN launcher with HBM-membind interlock
ec10f8f4 feat(bench): cross-host env overlay for two-machine engine A
```

`conf/fe.conf` is deliberately left **modified but uncommitted** in the working tree.

---

## §A — Task 1.1 steps 1-3, literal output

Step 1 (before the change) printed `priority_networks = 127.0.0.1/32` at line 76 and
`run_mode = shared_data`. After step 2's `sed` plus the comment correction (§D defect 4), re-run:

```
$ grep -n 'priority_networks\|run_mode' conf/fe.conf
53:# priority_networks = 10.10.10.0/24;192.168.0.0/16
85:priority_networks = 10.87.140.32/27
89:run_mode = shared_data

$ cp conf/fe.conf starrocks/output/fe/conf/fe.conf
$ grep -n priority_networks starrocks/output/fe/conf/fe.conf
53:# priority_networks = 10.10.10.0/24;192.168.0.0/16
85:priority_networks = 10.87.140.32/27

$ md5sum conf/fe.conf starrocks/output/fe/conf/fe.conf
74ee1a241f099be4010aa1488b895532  conf/fe.conf
74ee1a241f099be4010aa1488b895532  starrocks/output/fe/conf/fe.conf
$ diff conf/fe.conf starrocks/output/fe/conf/fe.conf ; echo "diff exit=$?"
diff exit=0
```

The commented `10.10.10.0/24` example at line 53 is untouched (the `^priority_networks` anchor
skipped it) and `run_mode = shared_data` survived the propagation. The output copy is gitignored
(`.gitignore:7:output` inside the `starrocks` submodule) and must be re-propagated by hand after any
future edit — that is why plan step 3 exists.

**No pre-existing drift.** Before any edit, both files were byte-identical (3094 bytes, md5
`39c1782aa50b08931518b59a8e95eba3`), so the tracked config was an accurate picture of what the FE
had been reading.

## §B — Tasks 1.2 and 2.1, literal output

### Task 1.2 (`configs/gb200-4gpu/engine-a-2host.env`)

```
=== Task 1.2 Step 2 ===
peers=1 tls=cuda_copy,cuda_ipc,rc_mlx5,tcp,self
=== Task 1.2 Step 3 ===
configs/gb200-4gpu/engine-a-2host.env: line 7: NUM_CNS: set NUM_CNS (total CNs across BOTH hosts) before sourcing
exit=1
```

Both match the plan's expected values exactly. The file is byte-identical to the plan's Task 1.2
step 1 heredoc (md5 `da427574f6abd1da34459b4bb51721bc` on both sides).

Hardware cross-checks that back the overlay's values, measured on gcn-18: `mlx5_0/1/4/5` map to
`enp3s0np0`/`enP2p3s0np0`/`enP16p3s0np0`/`enP18p3s0np0`, all `state 4: ACTIVE`, `400 Gb/sec (4X NDR)`;
the excluded `mlx5_2/3/6/7` are the 200 Gb/sec DPU NICs on `100.127.x`. `ucx_info -v` = 1.21.0 and
`ucx_info -f` shows `UCX_MAX_RNDV_RAILS=2`, so the overlay's `=4` is meaningful. `UCX_NET_DEVICES`
does not disable NVLink: `cuda_ipc`/`cuda_copy` are governed by `UCX_SHM_DEVICES`/`UCX_ACC_DEVICES`.

### Task 2.1 (`benchmarks/cn-2host.sh`), after the fixes in §D

```
$ bash -n benchmarks/cn-2host.sh && echo "SYNTAX OK"
SYNTAX OK

=== Step 2: refuses an HBM membind ===
$ CN_NODE="0 0 2 1" ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.52 --no-fe; echo "exit=$?"
CN2: --membind 2 is not a CPU-bearing node (HBM interlock)
exit=1

=== Step 3: refuses a short CN_CPUS ===
$ CN_CPUS="0-35 36-71" ./benchmarks/cn-2host.sh 10.87.140.53 10.87.140.52 --no-fe; echo "exit=$?"
CN_CPUS needs 4 entries
exit=1

=== Step 4: derived warmup peer count ===
$ ( NUM_CNS_PER_HOST=4 NUM_CNS=8 . configs/gb200-4gpu/engine-a-2host.env \
    && echo "peers=$SIRIUS_CN_NIXL_WARMUP_EXPECT_PEERS" ); echo "exit=$?"
peers=7
exit=0
```

Both negative tests exit at the validation guards, **before** the env sourcing, the preflight, the FE
launch and the CN loop. Confirmed after each run:

```
$ ps -eo pid,args --no-headers | grep -E 'sirius-starrocks-cn|StarRocksFE|starrocks_be' | grep -v grep
(no output)
$ ls /tmp/fe.log /tmp/cn-53-*.log
ls: cannot access '/tmp/fe.log': No such file or directory
ls: cannot access '/tmp/cn-53-*.log': No such file or directory
```

(The per-CN log files are created by shell redirection *before* `exec`, so their absence is
independent proof that the launch loop was never entered.)

### Task 2.1 step 4b — the new preflight guards (added, see §D defects 2/3)

Run against a scratch copy whose `target/release/sirius-starrocks-cn` is a symlink to `/bin/true`,
so a broken guard could only fork `/bin/true` — zero launch risk:

```
=== A: no CN binary present -> must refuse ===
cn-2host: no CN binary at .../scratchpad/fakesr/target/release/sirius-starrocks-cn
exit=1

=== C: START_FE with no packaged FE present -> must refuse ===
cn-2host: no packaged FE at .../scratchpad/fakesr/starrocks/output/fe/bin/start_fe.sh
exit=1

=== D: port 9100 occupied -> must refuse before any launch ===
listener bound on 127.0.0.1:9100
cn-2host: required ports already bound: 9100
  A cluster is very likely already running (G1). Shut it down first -- do NOT launch a
  second one on top of it.
exit=1

=== E: FE membind node list derived from hardware ===
FE_NODES=0,1
cpuless(HBM) node excluded: 2 .. 33      (32 HBM nodes, all excluded)

=== F: nothing was launched ===
no CN/FE/BE processes
```

The GPU-claim guard could not be made to fire — all four GPUs are genuinely unclaimed right now
(`nvidia-smi --query-compute-apps=pid ... -i 0` returns empty), so it correctly stays silent. Its
*firing* path is therefore **untested**; it is a verbatim lift of `cluster4-numa.sh:252-267`.

**Plan/file sync proof** — the plan's Task 2.1 step 1 heredoc body and the committed file are
byte-identical, so they cannot drift:

```
plan heredoc body lines 706..869
heredoc md5 = ebeef36c6ea60ea1eb7b61bc87e1046b 7637 bytes
file    md5 = ebeef36c6ea60ea1eb7b61bc87e1046b 7637 bytes
IDENTICAL
```

and the committed blob matches the working tree (`git show HEAD:...cn-2host.sh | md5sum` =
`ebeef36c6ea60ea1eb7b61bc87e1046b`).

---

## §C — BLOCKED — requires gcn-17

`ssh` is a user DENY rule in this session, and hard rule 2 forbids starting any cluster or
long-lived process. Everything below is blocked for one or both of those reasons and must be run by
an operator with gcn-17 access.

| Plan task | Why blocked |
|---|---|
| **0.1** steps 1-5 — collect gcn-17 arch/NUMA/bond0/JDK/`/raid`/HCA/GPU facts; RoCE link state on both hosts; IMEX fabric + ClusterUUID match | every step is a literal `ssh presto-gb200-gcn-17 ...`. **All gcn-17 facts in the plan remain inference.** |
| **0.2** steps 1-4 — stage SF1/SF100 under `/raid`, `rsync` to gcn-17, compare inventories, record `lineitem` size | step 2 is `rsync ... presto-gb200-gcn-17:`; step 3 is `ssh`. Steps 1 and 4 are host-local but pointless alone (G3 requires the same absolute path on *both* hosts). |
| **0.3** steps 1-4 — `nixl-echo-2node.sh` on default config, on multi-rail GPUDirect RDMA, and on MNNVL | the harness is a two-node GPU workload. Barred by hard rule 2 even leaving ssh aside. Consequence: the overlay's `UCX_TLS` (with `cuda_ipc` retained) is **still unmeasured cross-host** — open risk 2. |
| **1.1 step 3b** — migrate the FE's persisted identity | the metadata (`starrocks/output/fe/meta`) is read by the FE **on gcn-17**. Route (b) is a `mv` of a directory that may be open by a process on the unreachable host, so it was not executed here. Procedure is now written into the plan. |
| **1.1 step 4** — start the FE, `SHOW PROC '/frontends'` | starts the FE (hard rule 2) and runs on gcn-17. |
| **1.3** — launch CN0 on gcn-17 | gcn-17, and launches a CN. |
| **1.4** — launch CN0 on gcn-18 | host-local, but launches a CN (hard rule 2) and is useless without gcn-17's FE. |
| **1.5** steps 1-3 — `SHOW COMPUTE NODES`, bandwidth canary, warmup peers | needs a live FE + two live CNs. |
| **1.6** steps 1-7 — `pipeline_dop` pin, `EXPLAIN SCHEDULER`, validation query, the three distribution proofs, acceptance record | needs a live cluster; proofs 1/2 read a running query's profile and the CN logs. |
| **1.7** steps 1-3 — SF100 TPC-H sweep | a benchmark sweep; also must run on gcn-17 (`bench.sh` hardcodes `--host 127.0.0.1`). |
| **2.2** steps 1-5 — clear-box check, launch 8 CNs, verify registration + NUMA pins | step 1 is `ssh` to both hosts; steps 2-3 launch the cluster. |
| **2.3** steps 1-3 — canary per peer pair, `cuda_ipc` vs RDMA split, 7-peer warmup | reads logs from a live 8-CN cluster. This is the check Phase 1 structurally cannot perform. |
| **2.4** steps 1-5 — re-pin `pipeline_dop`, SF100 validation query, 8-host profile, cross-host `dest=` volume | needs a live 8-CN cluster. |
| **2.5** steps 1-3 — Phase 2 sweep, provenance file, Phase 1 vs Phase 2 comparison | benchmark sweeps on gcn-17. |
| **3** steps 1-4 — stop engine A on both hosts, engine B tutorial §2-§6, engine-B distribution proof, three-arm comparison | step 1 is `ssh` to both hosts; the rest brings up a second cluster. |

**Net:** only the three host-local, filesystem-only tasks (1.1 steps 1-3, 1.2, 2.1) were executable
from this session. Nothing in Phase 0 has been resolved, so the plan's open risk 1 ("every gcn-17
fact is inference") is fully intact.

---

## §D — Defects found and fixed

Four defects graded blocker/major by adversarial review. Minor findings (the `NUM_CNS` guard
accepting non-numeric values, `-ge` arity guards accepting over-long lists, the hardcoded `0|1`
membind list not being hardware-derived, `PER_HOST` not validated as a positive integer) were
**deliberately not fixed** and remain open.

### Defect 1 — BLOCKER: the FE cannot start with the new `priority_networks` (Task 1.1)

Changing `priority_networks` does not rewrite the identity persisted in the FE metadata.

**Before** — verified on this tree:

```
$ cat starrocks/output/fe/meta/image/ROLE
#Mon Aug 10 00:53:15 GMT 2026
role=FOLLOWER
hostType=IP
name=127.0.0.1_9010_1786215315408
```

`conf/fe.conf:38` (`meta_dir`) is commented, so `meta_dir = ${STARROCKS_HOME}/meta` — that
directory, which also holds a real 2.2 MB `bdb/00000000.jdb` journal. On the next start
`isFirstTimeStartUp = false`, `selfNode` becomes `10.87.140.52`, and
`NodeMgr.checkCurrentNodeExist()` (`NodeMgr.java:675-683`) → `unprotectCheckFeExist`
(`NodeMgr.java:977-981`) matches with `NetUtils.isSameIP` — pure string equality, no loopback
aliasing — against the persisted host `127.0.0.1`. It returns `null`, the FE logs
`current node is not added to the cluster, will exit` and calls `System.exit(-1)`. Confirmed in
source; `Config.bdbje_reset_election_group` exists at `Config.java:773` and short-circuits that
check at `NodeMgr.java:676`.

Compounding it, plan step 4's `until mysql ... ; do sleep 2; done` had no timeout and no liveness
check, so a dead FE was an infinite hang rather than an error.

**After** — plan Task 1.1 gains **step 3b** (mandatory identity migration, two documented routes:
`bdbje_reset_election_group = true` for one start, or `mv .../meta .../meta.bak-127001` for a clean
bootstrap; `ALTER SYSTEM MODIFY FRONTEND HOST` is ruled out because `NodeMgr.java:827-829` throws
`can not modify current master node`), and step 4's wait loop is now bounded at 90 iterations with a
`kill -0 "$FE_PID"` liveness check that tails `/tmp/fe.log` on exit. The `ROLE` fact is also recorded
in the plan's "Verified environment facts" table.

**Not executed here** — the metadata lives on NFS and is read by the FE on the unreachable gcn-17.
See §C.

### Defect 2 — MAJOR: the FE was launched with no NUMA membind (`cn-2host.sh`)

**Before** (`cn-2host.sh:49`):

```bash
starrocks/output/fe/bin/start_fe.sh --logconsole > /tmp/fe.log 2>&1 &
```

A bare FE keeps `Mems_allowed_list = 0-2,10,18,26`, i.e. a ~10-20 GiB JVM able to allocate into
GPU0's HBM — the exact exposure the CN membind exists to close, and the thing the script's own
header declares as its reason for existing. `configs/gb200-4gpu/cluster4-numa.sh:314-325` already
implements the fix.

**After**: the node list is derived from the hardware (never hardcoded) and the FE is membound to
every CPU-bearing node, with **no** cpubind so its deliberate cross-socket float survives:

```bash
FE_NODES=$(numactl --hardware |
    awk '/^node [0-9]+ cpus:/ && NF > 3 { n = (n == "" ? $2 : n "," $2) } END { print n }')
[ -n "$FE_NODES" ] ||
    { echo "cn-2host: no NUMA node reports CPUs -- refusing to membind the FE" >&2; exit 1; }
numactl --membind="$FE_NODES" -- "$FE_BIN" --logconsole > /tmp/fe.log 2>&1 &
```

Verified on gcn-18: `FE_NODES=0,1`, with all 32 cpuless HBM nodes (2-33) excluded (§B step 4b, E).

### Defect 3 — MAJOR: no preflight before the launch loop (`cn-2host.sh`)

**Before**: no binary check, no `numactl` check, no port scan, no GPU-claim check. The script uses
the same `9100`/stride-10 block as `cluster8.sh` and `cluster4-numa.sh`, so launching it over a
running cluster is an **identity** collision (FE keys a node by `(advertise_host, heartbeat_port)`;
the nixl agent is `{advertise_host}:{brpc_port}`), not a clean bind failure. And the CN's own
`ensure_gpu_unclaimed` preflight is skipped whenever `--gpu-memory-limit` is set — which this script
always does — so nothing stopped two CNs landing on one GPU with a fully pre-reserved RMM pool.

**After**: a preflight block before the launch loop that checks `-x` on the CN binary, `-x` on the FE
binary when `START_FE=1`, `command -v numactl`, a `/proc/net/tcp{,6}` `st == 0A` scan of every
required port (plus `6090 8030 9010 9020 9030` when starting the FE), and
`nvidia-smi --query-compute-apps` per GPU with an `ALLOW_SHARED_GPUS=1` override. Three of the four
guards were made to fire (§B step 4b); the GPU-claim path is untested because no GPU is claimed.

Also fixed in the same block (graded minor, but it is the same failure mode): `trap cleanup EXIT INT
TERM` was installed *after* the launch loop had already forked the FE and every CN, so a Ctrl-C
during the launch window orphaned them. The trap is now armed before the first fork and the handler
gained the `trap - EXIT INT TERM` disarm and `exit "$status"` propagation from
`cluster4-numa.sh:284-294`.

### Defect 4 — MAJOR: `--engine-dir` collides across hosts on the shared NFS checkout

**Before**: plan Tasks 1.3 and 1.4 both passed `--engine-dir .cn0`, and `cn-2host.sh` passed
`--engine-dir ".cn$i"` on both hosts.

`/proc/mounts` confirms `/home` is NFS:

```
master:/home /home nfs4 rw,...,addr=10.87.140.8 0 0
/dev/md0 /raid ext4 rw,noatime,nodiratime,stripe=64 0 0
```

`--engine-dir` is relative and resolved against this checkout (`src/main.rs:252-288`), and holds
`derived-sirius-config.yaml`, `log/` and `telemetry/` — so both hosts' CN0 would race on the same
files.

**After**: `cn-2host.sh` uses `--engine-dir ".cn$i-${ADVERTISE##*.}"` (`.cn0-52` / `.cn0-53`), plan
Task 1.3 uses `.cn0-52` and Task 1.4 uses `.cn0-53`, both with the reason stated inline. Confirmed
safe: nothing in `benchmarks/tpch/bench.sh` or `run-abc.sh` refers to `.cnN`. The NFS fact and its
consequence ("Tasks 1.1/1.2/2.1 are done once, not per host") are now in the plan's facts table.

### Defect 5 — the `fe.conf` comment stated the opposite of the config (graded minor, fixed anyway)

**Before** (`conf/fe.conf:75`, directly above the changed line, and inside the diff hunk):

```
# Keep FE on loopback so it advertises 127.0.0.1 to the CN (matches its --fe-host default).
priority_networks = 10.87.140.32/27
```

**After**: the comment now describes the control-LAN CIDR, states that the CN's `--fe-host` default
is still `127.0.0.1` so every CN must pass `--fe-host 10.87.140.52` explicitly, and points at Task
1.1 step 3b for the metadata migration.

### Also corrected in the plan (both reviewers flagged; not a code defect)

Tasks 1.2 step 4 and 2.1 step 5 wrote a bare `git commit -m "..."`, which on this branch would sweep
a dozen unrelated staged paths into the commit. Both are now pathspec-scoped
(`-- configs/gb200-4gpu/engine-a-2host.env`, `-- benchmarks/cn-2host.sh`), which is what was actually
executed for `ec10f8f4` and `8eced0f1`.

---

## §E — Environment note for whoever runs this next

A `PreToolUse` agent hook implementing the `pre-commit-cleanup` gate fires on **every** `Bash` call,
not just `git commit`, and its verdict is nondeterministic — the identical command was denied and
allowed across attempts with mutually contradictory rationales, including one denial of a read-only
command whose stated reason was that the hook's own `git status`/`git diff` were unavailable. Every
complaint it raised targeted this plan's own staged documents, not the artifacts under change. It
cost a large fraction of the tool budget across all three tasks. Two fixes the hook itself suggested:
scope the matcher to `git commit` commands, and add read-only `Bash(git status *)` / `Bash(git diff *)`
to the project `.claude/settings.json` so the gate can actually read a diff.
