# `2NODE-RUNBOOK.md` — corrections and additions

Patch list against `benchmarks/2NODE-RUNBOOK.md` (untracked, `git status` → `A`). Part A is
everything the audit found **wrong or stale**; points the audit found accurate are not listed.
Part B is drop-in markdown for the gaps between the runbook and the stated goal (one backend per
box, one GPU each, fragments provably on both hosts, best available transport).

Line citations for StarRocks FE/BE are against the vendored submodule at its checked-out **4.1.1**
HEAD unless a tag is named. Sirius citations are against the working tree.

---

## A. Corrections

| § / line | Current text | Corrected text | Evidence |
|---|---|---|---|
| §1 line 34, §3 line 118-123 | "`priority_networks` in `conf/fe.conf`" and a diff shown against that file | The FE reads **`starrocks/output/fe/conf/fe.conf`**, not `conf/fe.conf`. Keep editing the tracked `conf/fe.conf` (it *is* the source of truth) but then either run `pixi run fe-run`, or copy it explicitly: `cp experimental/starrocks/conf/fe.conf experimental/starrocks/starrocks/output/fe/conf/fe.conf`. The runbook's own start command (`starrocks/output/fe/bin/start_fe.sh`, §3 line 135) bypasses the copy. Note a bare `./build.sh --fe` inside the submodule (outside pixi) restores the stock conf. | `starrocks/output/fe/bin/start_fe.sh:77` `export STARROCKS_HOME=$(cd "$curdir/.."; pwd)`, `:90` `export_env_from_conf $STARROCKS_HOME/conf/fe.conf`; propagation is `pixi.toml:216` `cp ../conf/fe.conf output/fe/conf/fe.conf` (last step of `fe-build`, with `conf/fe.conf` declared an input at `pixi.toml:219`); `starrocks/build.sh:602` copies the *submodule's* conf, which lacks every Sirius override; `starrocks/.gitignore:7` ignores `output` |
| §3 lines 125-126 | "`run_mode = shared_data` is required — shared-nothing schedules only on BEs, and a storage-less CN would never receive a fragment." | The directive is right, the reason is wrong. The fragment **scheduler** is not BE-only: `DefaultWorkerProvider` (the SHARED_NOTHING provider) sets `usedComputeNode = true` whenever no Backends are available. The real reason is DDL: with zero BEs, shared-nothing cannot create tables — `LocalMetastore.java:1989` counts only alive Backends and `:2001-2006` throws `DdlException("no alive backends")`; replica placement via `NodeSelector.seqChooseBackendIds` (`NodeSelector.java:110`) also draws only from `getAvailableBackends()`. Keep `run_mode = shared_data`; fix the sentence. | `DefaultWorkerProvider.java:49`, `:139-144`; `SessionVariable.java:1240-1241` (`preferComputeNode = false` default); `LocalMetastore.java:1989`, `:2001-2006` |
| §2 line 98 **and** §4 line 168 | `export UCX_TLS=cuda_copy,rc_mlx5,tcp,self` | `export UCX_TLS=cuda_copy,cuda_ipc,rc_mlx5,tcp,self` — the prescribed line **drops `cuda_ipc`**, contradicting §2's own table, whose 48.7 / 97 GB/s rows are labelled "`+ rc_mlx5`" i.e. *added to* the `cuda_copy,cuda_ipc,tcp,self` baseline. With 4 CNs per host, dropping `cuda_ipc` costs every same-host peer pair the 85-90 GB/s NVLink path, and the transport's own refusal text names "UCX_TLS missing cuda_ipc" as a trap. **Untested config change** — re-measure the 97 GB/s row with `cuda_ipc` retained. | Table at `2NODE-RUNBOOK.md:57-59`; harness default `scripts/nixl-echo-2node.sh:36` `UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}`; refusal text `src/nixl_transport.rs:664-668` |
| §4 line 171 | `export SIRIUS_EXCHANGE_STAGING_BYTES=1280MiB` alongside `--gpu-memory-limit 140GiB` | `export SIRIUS_EXCHANGE_STAGING_BYTES=16GiB`. `1280MiB` is the pixi `cluster2` value, where two CNs share **one 23 GiB L4** at 8 GiB each. The committed GB200 value is 16 GiB, and the arena fails **hard** on exhaustion (it never degrades). | `pixi.toml:88` (`cluster2` env); `configs/gb200-4gpu/engine-a.env:71-72` `GPU_MEM=140GiB` / `STAGING=16GiB`, `:113`; `src/exec/exchange_staging_arena.cpp:85-91` |
| §4 line 224 | `export SIRIUS_CN_NIXL_WARMUP_EXPECT_PEERS=7        # NUM_CNS - 1` | The comment is right; the **value is topology-specific**. `7` is correct only for 8 CNs. For the one-CN-per-host shape it is **`1`**. The loop breaks on `established.len() >= expect`. | `src/nixl_transport/warmup.rs:51-52`, `:109-111`, `:291-292` |
| §4 lines 164-190 (the launch block) | Bare `target/release/sirius-starrocks-cn ...` in a `for` loop | Two omissions vs the committed launcher, both hazards: (a) **no `numactl`** — an unpinned CN runs with `Mems_allowed_list: 0-2,10,18,26`, i.e. GPU HBM is in its allowed set; (b) **no `unset CUDA_VISIBLE_DEVICES`** — an inherited value *overrides* `--gpu-device`, which is merely warned about, silently collapsing every CN onto one GPU. See §B.1 for the corrected block. | `configs/gb200-4gpu/cluster4-numa.sh:46-54` (the unset), `:301-303` (`numactl --physcpubind ... --membind ...`); `src/engine.rs:206-218` (CUDA_VISIBLE_DEVICES precedence) |
| §4 lines 142-144 | "there is no manual `ALTER SYSTEM` step and **no ordering requirement** between FE and CNs" | Bounded, not unbounded. The CN binds all listeners and marks itself ready **first**, then retries registration for ~57 min (`--registration-max-attempts`, default **120**, backon exponential 1 s → 30 s); on exhaustion `run()` returns `Err` and the CN process **exits**. Raise `--registration-max-attempts` if the FE will start much later. | `src/main.rs:24`, `:26`, `:58-59`, `:397-398`, `:413`; readiness before registration at `src/main.rs:204-214` |
| §4 port table, line 207 | `FE → CN \| http \| FE's blacklist-eviction TCP probe` | The probe hits **`be_port` (thrift) + `brpc_port` + `http_port`** and lifts the entry only if **every** one accepts (it breaks on the first refusal). The table is also **missing a row for the CN thrift port** (`--thrift-port`, advertised as `be_port`), which the CN binds and the FE must reach. A firewall opening only heartbeat/http/brpc leaves a blacklisted CN permanently excluded. | `HostBlacklist.java:210-212`, `NetUtils.java:159-165`; CN binds thrift at `src/lib.rs:1000-1001`. Same in 3.5.20: `HostBlacklist.java:206-207` |
| §0 line 20 | "the committed launchers (`cluster8.sh`, `configs/gb200-4gpu/cluster4-numa.sh`)" | Path is **`benchmarks/cluster8.sh`**, and a third single-host launcher **`benchmarks/cluster8-numa.sh`** exists and is unmentioned. The substantive claim (none pass `--advertise-host`/`--fe-host`) holds for all three. | `ls benchmarks/*.sh` → `cluster8-numa.sh cluster8.sh`; `grep -n 'advertise-host\|fe-host' benchmarks/cluster8*.sh configs/gb200-4gpu/cluster4-numa.sh` → comments only |
| §7 bullet 3 | "`priority_networks` is committed as loopback, so a two-host run means editing tracked config." | True but understated: editing **only** the tracked file changes nothing about a running FE — see the §3 correction above. The override mechanism this bullet asks for should be a `cp` step or a `pixi run fe-run` in §3, not a new feature. | as above |
| §0 lines 16-20 (Provenance) | "The transport numbers in §2 are measured on this pair by `scripts/nixl-echo-2node.sh`" | Add: **the script and `src/nixl_transport/nixl_echo.rs` / `two_node_harness.rs` are uncommitted working-tree additions** (`git status` → `A`), so §2 is not reproducible from a clean `dev` checkout. Only the **765 GB/s** row has a surviving raw artifact (`/tmp/nixl-echo-2node/origin.log`, 2026-08-11 02:44, `leg1 763.07 GB/s / leg2 767.80 GB/s / verified true`, `host=presto-gb200-gcn-17` ↔ `host=presto-gb200-gcn-18`). The 0.37 / 48.7 / 97 GB/s rows have no on-disk artifact. | `git status --porcelain`; `/tmp/nixl-echo-2node/origin.log`, `/tmp/nixl-echo-2node/echo.log`; `scripts/nixl-echo-2node.sh:111` |
| §6, after line 273 | (nothing) | Add a caution: **engine A produces no StarRocks query profile.** The CN never calls `FrontendService.reportExecStatus` (0 hits across `src/` + `crates/`; the generated binding has 16) and hardcodes `PFetchDataResult.query_statistics = None`, so `BytesSent`/`BytesReceived` do not exist and `fe.audit.log` `ScanBytes` is empty. `EXPLAIN ANALYZE` additionally blocks for the full `profile_timeout` (10 s) waiting for reports that never arrive. Use FE-side placement + CN tracing instead (§B.2). | `src/compute_node_service.rs:1487` `query_statistics: None`; `benchmarks/tpch/plans/q01.analyze.txt:5,7,27` (`TotalTime: 10s605ms`, `CollectProfileTime: 10s2ms`, `MissingInstanceIds:`); `SessionVariable.java:1619-1620` |

Not corrections, but worth folding into §2 as scope notes:

* The §2 table's row 4 (765 GB/s, fabric arena) was measured with the **default** `UCX_TLS`
  (`cuda_copy,cuda_ipc,tcp,self`) and `UCX_NET_DEVICES=enp3s0np0` only — per the header line in
  `/tmp/nixl-echo-2node/origin.log`. `rc_mlx5` is a requirement of the **`cudaMalloc` arena**, not of
  cross-host operation in general. The table already says this; the prose in §2 "What to do today"
  does not.
* `UCX_MAX_RNDV_RAILS=4` (§2 line 100, §4 line 170) appears in no source file or committed script in
  this repo. Confirm it against the installed UCX build before publishing:
  `/home/prestouser/aocsa/tools/ucx-install/bin/ucx_info -c | grep RNDV_RAILS`. **TODO — unverified.**
* `gdr_copy` is **not built** into the installed UCX 1.21.0 (`ucx_info -b` → `uct_cuda_MODULES ""`,
  zero `gdr_copy` devices in `ucx_info -d`). Do not add it to `UCX_TLS`; it would be a silent no-op.

---

## B. Additions

### B.1 — Drop-in replacement for §4: "one CN per host, one GPU"

> ## 4b. Variant — one CN per host, one GPU each
>
> The minimal two-machine topology: exactly two CNs, one per box, each owning GPU 0. This is the
> shape to bring up first — it has one peer pair, so every cross-host path is exercised and no
> same-host `cuda_ipc` pair can mask a broken fabric.
>
> NUMA affinity on this box (MEASURED, `nvidia-smi topo -m` + `numactl -H`): **GPU0/GPU1 → NUMA
> node 0, CPUs 0-71; GPU2/GPU3 → NUMA node 1, CPUs 72-143.** So GPU 0 pairs with
> `--physcpubind=0-71 --membind=0`.
>
> `--membind` may only ever be `0` or `1`. `numactl -H` reports 34 nodes; nodes 2/10/18/26 are
> 188,416 MB of GPU HBM with **zero CPUs** — binding host pages there eats the HBM of a GPU a CN is
> computing on (`configs/gb200-4gpu/cluster4-numa.sh:135-141` enforces this mechanically).
>
> ```bash
> # ================= gcn-17 (10.87.140.52) =================
> cd ~/aocsa/sirius/experimental/starrocks
>
> # TOOLS_DIR and UCX_TLS FIRST: cn-env.sh only fills them in when unset (cn-env.sh:35), so
> # sourcing first would silently discard the cross-host TLS choice.
> export TOOLS_DIR=/home/prestouser/aocsa/tools
> export UCX_TLS=cuda_copy,cuda_ipc,rc_mlx5,tcp,self
> source scripts/cn-env.sh
>
> # An inherited CUDA_VISIBLE_DEVICES OVERRIDES --gpu-device and is only warned about
> # (src/engine.rs:206-218). Clear it so --gpu-device is authoritative.
> unset CUDA_VISIBLE_DEVICES
>
> export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,enp3s0np0
> export UCX_MAX_RNDV_RAILS=4                      # TODO: confirm with `ucx_info -c | grep RNDV_RAILS`
> export SIRIUS_EXCHANGE_STAGING_BYTES=16GiB       # GATE, not just a size: unset => no arena,
>                                                  # nixl tier disabled, every remote dest fails
> export SIRIUS_CN_NIXL_WARMUP_EXPECT_PEERS=1      # NUM_CNS - 1, and NUM_CNS is 2 here
> export SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS=300    # default 180 assumes a single-box launch
> export SIRIUS_CN_USE_SIRIUS_DATASOURCE=false     # PIN IT. Set by no committed config; it is
>                                                  # worth ~20x on scan-bound queries and the
>                                                  # SF100 anchor ran with =false (kvikio).
> export RUST_LOG=sirius_starrocks_cn=info,info
>
> numactl --physcpubind=0-71 --membind=0 -- target/release/sirius-starrocks-cn \
>     --fe-host           10.87.140.52 \
>     --advertise-host    10.87.140.52 \
>     --bind-host         0.0.0.0 \
>     --gpu-device        0 \
>     --heartbeat-port    9100 \
>     --thrift-port       9101 \
>     --brpc-port         9102 \
>     --http-port         9103 \
>     --starlet-port      9104 \
>     --gpu-memory-limit  140GiB \
>     --host-memory-limit 160GiB \
>     --engine-dir        .cn0 \
>     > /tmp/cn-gcn17.log 2>&1 &
> ```
>
> ```bash
> # ================= gcn-18 (10.87.140.53) =================
> # IDENTICAL except --advertise-host. Ports may stay the same across hosts: FE node identity is
> # (advertise_host, heartbeat_port), so the differing IP already separates the two nodes
> # (src/main.rs:521-523 also makes the nixl agent name {advertise_host}:{brpc_port}).
> cd ~/aocsa/sirius/experimental/starrocks
> export TOOLS_DIR=/home/prestouser/aocsa/tools
> export UCX_TLS=cuda_copy,cuda_ipc,rc_mlx5,tcp,self
> source scripts/cn-env.sh
> unset CUDA_VISIBLE_DEVICES
> export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,enp3s0np0
> export UCX_MAX_RNDV_RAILS=4
> export SIRIUS_EXCHANGE_STAGING_BYTES=16GiB
> export SIRIUS_CN_NIXL_WARMUP_EXPECT_PEERS=1
> export SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS=300
> export SIRIUS_CN_USE_SIRIUS_DATASOURCE=false
> export RUST_LOG=sirius_starrocks_cn=info,info
>
> numactl --physcpubind=0-71 --membind=0 -- target/release/sirius-starrocks-cn \
>     --fe-host           10.87.140.52 \
>     --advertise-host    10.87.140.53 \
>     --bind-host         0.0.0.0 \
>     --gpu-device        0 \
>     --heartbeat-port    9100 --thrift-port 9101 --brpc-port 9102 \
>     --http-port         9103 --starlet-port 9104 \
>     --gpu-memory-limit  140GiB \
>     --host-memory-limit 160GiB \
>     --engine-dir        .cn0 \
>     > /tmp/cn-gcn18.log 2>&1 &
> ```
>
> **Memory, per GPU.** `--gpu-memory-limit` is per-GPU and does **not** scale with CN count — 140 GiB
> stays 140 GiB. Usable HBM is 188,417 MiB (~184.0 GiB), not the 189,471 MiB nameplate. The staging
> arena is a bare `cudaMalloc` **outside** the RMM limit, so the real footprint is
> `GPU_MEM + STAGING + ~779 MiB` of CUDA context. The documented ceiling is
> `GPU_MEM + STAGING ≤ 159,744 MiB`; `140GiB + 16GiB` hits it exactly, leaving 27,893 MiB (14.8%)
> for out-of-RMM consumers. If the arena starves at 2 CNs, the footprint-neutral rebalance is
> `128GiB + 28GiB` — **derived, never measured at 2 CNs.**
> (`configs/gb200-4gpu/engine-a.env:57-96`; `SNMG-PLAN.md:53`.)
>
> **`--host-memory-limit 160GiB`** is safe with margin: the per-node ceiling for a CN membound to
> node 0 is ~386.66 GiB. 320 GiB is also inside the ceiling if you want the single-CN-per-socket
> analogue (`engine-a.env:170-196`; `SNMG-PLAN.md:92`) — **untested at this shape.**
>
> **Pin `pipeline_dop` explicitly.** It is derived from the CPU count the CN reports
> (`src/lib.rs:407` `available_parallelism()` → `hardware_cores` → FE), so `--physcpubind=0-71` makes
> each CN report 72 and `pipeline_dop` resolve to `min(64, 72/2) = 36`; unpinned it would be 64.
> Set it identically in both arms of any A/B, and record it:
> ```bash
> mysql -h 10.87.140.52 -P 9030 -uroot -e \
>   "SET GLOBAL enable_pipeline_engine = true; SET GLOBAL pipeline_dop = 36;
>    SHOW GLOBAL VARIABLES LIKE 'pipeline_dop';"
> ```
> (`enable_pipeline_engine` **persists in FE metadata** across runs — `benchmarks/tpch/run-abc.sh:786-791`.)
>
> **Verify the pin took** (must not read `0-2,10,18,26`):
> ```bash
> P=$(pgrep -f 'sirius-starrocks-cn')
> awk '{for(i=2;i<=NF;i++) if($i ~ /^(bind|default)/){print $i; break}}' /proc/$P/numa_maps | sort | uniq -c
> # expect a single `bind:0` line. Do NOT use Mems_allowed_list: it reports cpuset-allowed nodes
> # and stays 0-2,10,18,26 even when --membind works (MEASURED gcn-18 2026-08-11).
> ```
>
> **Scan fan-out caveat.** A scan fragment gets one instance per worker that actually received
> ranges. With 2 CNs a table only spreads if it has enough bytes/files: SF100 `lineitem` is 6 parquet
> files, `orders` 2, but `nation` is a single file and will legitimately land on one CN.

### B.2 — New subsection for §6: "Prove the fragments landed on both hosts"

> ### Prove the fragments landed on both hosts
>
> `SHOW COMPUTE NODES` proves the CNs *registered*. It does not prove a query *ran* on both. Two
> independent proofs are needed: the FE's placement, and the CN's own transport log.
>
> #### The minimal shuffle query
>
> A `FILES()` scan only splits across hosts if the data is big enough:
> `numInstances = clamp(totalBytes / min_bytes_per_broker_scanner, 1, nodes × parallelInstanceNum)`
> with `min_bytes_per_broker_scanner = 67108864` (64 MiB). **A single file under 128 MiB lands
> entirely on one host and proves nothing** — `nation` is 2,250 B and will always land on one CN,
> which is correct, not a failure. Always test on `lineitem`: SF100 is **17,187,602,838 B** over 6
> files (measured), i.e. 256 instances uncapped, so every CN is guaranteed ranges. There is no SF1
> on these hosts. (`FileScanNode.java:543-558`, `Config.java:1213`.)
>
> ```sql
> SET enable_profile = true;         -- NOT `EXPLAIN ANALYZE`: that forces enable_async_profile=false
>                                    -- and blocks for the full 10 s profile_timeout on engine A
> SET new_planner_agg_stage = 2;     -- TWO_STAGE: forces partial-agg -> HASH_PARTITIONED exchange
>                                    -- -> merge-agg instead of collapsing to one stage
> WITH lineitem AS (SELECT * FROM FILES(
>   "path"="file:///raid/prestouser/aocsa/tpch_parquet_sf100/lineitem/*.parquet","format"="parquet"))
> SELECT l_orderkey % 4096 AS bucket, count(*) AS n, sum(l_quantity) AS q
> FROM lineitem GROUP BY 1 ORDER BY 1 LIMIT 20;
> SELECT last_query_id();
> ```
>
> #### Proof 1 — StarRocks side (FE placement). Works on engine A.
>
> Everything here is FE-side, so it survives engine A's missing BE profile.
>
> ```sql
> -- Pre-flight: full scheduling, ZERO fragments deployed (StmtExecutor -> execWithoutDeploy()).
> EXPLAIN SCHEDULER <the query above>;
> ```
> ```bash
> mysql -h 10.87.140.52 -P 9030 -uroot -e "EXPLAIN SCHEDULER <query>;" \
>   | grep -E 'PLAN FRAGMENT|INSTANCE\(|BE: '
> ```
> `BE:` is a numeric compute-node id (`FragmentInstance.java:105-120`); map it through the
> `ComputeNodeId` column of `SHOW COMPUTE NODES`. Two distinct ids on the scan fragment = the FE
> intends to use both machines.
>
> Live, while the query runs (built entirely from the FE's `ExecutionDAG`, `ProcService.java:67`):
> ```bash
> watch -n1 "mysql -h 10.87.140.52 -P 9030 -uroot -e \"SHOW PROC '/current_backend_instances'\""
> ```
> Columns `Backend | InstanceNum | InstanceId | ExecTime`, where `Backend` is `host:be_port`.
>
> After the fact, from the raw profile:
> ```bash
> QID=<last_query_id()>
> mysql -h 10.87.140.52 -P 9030 -uroot --raw -N \
>   -e "SELECT get_query_profile('$QID')" > /tmp/profile.txt
> grep -n ' - BackendAddresses:' /tmp/profile.txt
> grep -nE ' - (BackendNum|InstanceNum|MissingInstanceIds):' /tmp/profile.txt
> ```
> Expected:
> ```
>    - BackendAddresses: 10.87.140.52:9101,10.87.140.53:9101
>    - BackendNum: 2
>    - MissingInstanceIds: 019fe1c9-...-c971,019fe1c9-...-c972      <- expected on engine A
> ```
> Notes: the port in `Address` is the CN's **thrift** port (`--thrift-port`, `ComputeNode.getAddress()`
> returns `(host, bePort)`), **not** brpc — so with the §B.1 port block it reads `:9101`, while the
> nixl log lines below name peers by `:9102`. `ANALYZE PROFILE` prints `BackendNum` but **never**
> `BackendAddresses`, so it cannot answer "which two hosts". And `BackendNum: 2` alone is not proof
> of two machines in a 4-CN-per-host layout — only the IPs are.
>
> On engine A every operator will read `TotalTime: 0ns (NaN%)` / `OutputRows: ?` and every instance
> will be listed under `MissingInstanceIds`. That is expected and does **not** invalidate
> `BackendAddresses`, which the FE writes from its own assignment.
>
> #### Proof 2 — nixl side (which transport, and did bytes cross). Engine A only.
>
> The CN logs to **stdout only**, so §4's redirects are mandatory or there is nothing to read.
>
> ```bash
> # transport came up and registered the arena
> grep 'nixl transport ready; staging arena registered' /tmp/cn-gcn1*.log
>
> # per-peer measured link speed -- THE gate on trusting any timing
> grep 'nixl bandwidth canary' /tmp/cn-gcn1*.log
> grep 'below the 2 GB/s floor'  /tmp/cn-gcn1*.log      # must be empty
>
> # warmup found its one peer
> grep -E 'pre-established a nixl peer session|nixl session warmup complete|peers left cold' \
>      /tmp/cn-gcn1*.log
>
> # THE per-transfer proof pair -- run these on the two hosts and match them up
> grep 'transmitted batches via nixl'                   /tmp/cn-gcn17.log   # sender
> grep 'received remote batches'                        /tmp/cn-gcn18.log   # receiver
> grep 'relayed native batches across a fragment boundary' /tmp/cn-gcn1*.log # same-process, NOT a hop
> ```
>
> `transmitted batches via nixl` carries `dest=<peer_host>:<peer_brpc_port>`, `batches`, `bytes`
> (`src/nixl_transport.rs:764-770`) and is emitted **only** under `DestinationRoute::Remote`
> (`src/compute_node_service.rs:1028`, routing decision at `:1055-1077`). So a line on gcn-17 with
> `dest=10.87.140.53:9102`, matched against a `received remote batches` line on gcn-18
> (`src/engine.rs:530-535`), is by construction proof that work crossed the machine boundary.
> `relayed native batches across a fragment boundary` (`src/engine.rs:489-493`) is the same-process
> short circuit — if that is all you see, nothing crossed.
>
> There is **no receiver-side byte counter** in CN tracing; read received bytes off the peer's sender
> line. Finer per-frame receive logging exists at DEBUG only
> (`received remote exchange frame`, `src/compute_node_service.rs:652-658`) — needs
> `RUST_LOG=sirius_starrocks_cn=debug,info`.
>
> #### Proof 3 — engine-agnostic corroboration
>
> Delta-sample OS counters around the query. These register on the **GPUDirect RDMA** path only;
> MNNVL bytes ride NVLink and will not appear here.
> ```bash
> for d in mlx5_0 mlx5_1 mlx5_4 mlx5_5; do
>   echo -n "$d "; cat /sys/class/infiniband/$d/ports/1/counters/{port_xmit_data,port_rcv_data} | tr '\n' ' '; echo
> done
> nvidia-smi nvlink -gt d > /tmp/nvlink.before    # ...run query... then repeat and diff
> ```

### B.3 — New subsection for §2: cross-host transport decision table

> ### Which cross-host path will you get, and how to pick it
>
> | Path | Requires | How to select | How to confirm you got it | GB/s |
> |---|---|---|---|---|
> | **MNNVL / cross-host NVLink** | Staging arena allocated via `cuMemCreate` + `CU_MEM_HANDLE_TYPE_FABRIC`; IMEX domain up on both hosts | **Not selectable in the CN today.** `exchange_staging_arena.cpp:42-48` is an unconditional `cudaMalloc`. Only the standalone harness can do it: `NIXL_ECHO_ARENA=fabric ./scripts/nixl-echo-2node.sh` | harness prints `standalone fabric (VMM) arena`; `nvidia-smi nvlink -gt d` deltas are non-zero | **765** (measured: `leg1 763.07 / leg2 767.80`, `/tmp/nixl-echo-2node/origin.log`, 2026-08-11) |
> | **Multi-rail GPUDirect RDMA** | 4× 400G RoCE ACTIVE; `rc_mlx5` in the installed UCX | `UCX_TLS=cuda_copy,cuda_ipc,rc_mlx5,tcp,self` + `UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,enp3s0np0` + `UCX_MAX_RNDV_RAILS=4` | `grep 'nixl bandwidth canary'` reads tens of GB/s; `/sys/class/infiniband/mlx5_*/ports/1/counters/port_xmit_data` climbs | **97** (runbook §2; no surviving raw artifact) |
> | Single-rail GPUDirect RDMA | as above, one NIC | same, `UCX_NET_DEVICES=mlx5_0:1,enp3s0np0` | canary tens of GB/s, one NIC's counters climb | 48.7 (same caveat) |
> | Host-staged TCP (**failure mode**) | nothing | what you get by **default** — `cn-env.sh:35` sets `UCX_TLS=cuda_copy,cuda_ipc,tcp,self` and no `UCX_NET_DEVICES` | CN log: `nixl link to ... below the 2 GB/s floor ... Refusing the transport tier`; `UCX_PROTO_INFO=used` shows a `ucp_put` sourced from *"host memory"* with no device id | **0.37 — peer refused, query fails** |
>
> **Best available today = multi-rail GPUDirect RDMA.** It needs no code change and works with the
> `cudaMalloc` arena. MNNVL needs a Sirius-side allocator change; `two_node_harness::cuda_vmm`
> (`alloc_fabric`, `CU_MEM_HANDLE_TYPE_FABRIC = 8`) is a working reference.
>
> There is **no fallback tier**: any `DestinationRoute::Remote` with no transport errors out with
> "cross-node exchange to {host}:{brpc_port} needs the nixl transport tier"
> (`src/compute_node_service.rs:970-977`). With one CN per host, *every* exchange is remote, so a
> refused peer means the cluster cannot run a single distributed query. The 2.0 GB/s floor is a
> compile-time `const` (`src/nixl_transport.rs:318`) with **no env override** — do not look for one.
>
> Preconditions verified on gcn-18 (re-check on gcn-17):
> ```bash
> ibv_devinfo -l                      # 8 HCAs: mlx5_0..mlx5_7
> ls -d /sys/class/infiniband/*/device/net/*   # mlx5_0->enp3s0np0, mlx5_1->enP2p3s0np0,
>                                              # mlx5_4->enP16p3s0np0, mlx5_5->enP18p3s0np0
> grep -H . /sys/class/infiniband/mlx5_{0,1,4,5}/ports/1/{state,rate,link_layer}
>                                     # 4: ACTIVE / 400 Gb/sec (4X NDR) / Ethernet
> nvidia-imex-ctl -N                  # gcn-17 and gcn-18 both READY, matrix all "C"
> nvidia-smi -q | grep -A3 Fabric     # State: Completed / Status: Success, same ClusterUUID
> $TOOLS_DIR/ucx-install/bin/ucx_info -d | grep -E 'Transport|Device:'   # rc_mlx5 on all four
> ```
> Note `/etc/nvidia-imex/nodes_config.cfg.pending` (Aug 5) is **newer** than the live
> `nodes_config.cfg` (Jul 9). A node-map mismatch disables IMEX communication
> (`nvidia-imex-ctl` legend: `!M! - Node map mismatch, communication disabled`), which would kill the
> MNNVL path mid-run if the daemon reloads. **TODO:** `diff /etc/nvidia-imex/nodes_config.cfg{,.pending}`.
>
> Ordering matters: export `UCX_TLS`/`TOOLS_DIR` **before** `source scripts/cn-env.sh`, which fills
> them in only when unset (`cn-env.sh:35`).

### B.4 — Replacement for §5's dataset paragraph

**This supersedes `2NODE-RUNBOOK.md:248-251`, which is now wrong.** That text says the datasets live
only on gcn-18 and must be replicated. They are already on both hosts.

> ### Dataset paths (already replicated)
>
> The FE assigns byte ranges without host awareness, and *also* sends `FILES()` schema inference to a
> **randomly shuffled** alive backend — so a dataset present on only one host fails
> **intermittently**, not deterministically. That makes the identical-path requirement absolute.
>
> **It is already met.** Both hosts carry the same trees at the same absolute path, on node-local
> ext4 (`/raid` = `/dev/md0`, 13 TB free):
>
> | Path (identical on gcn-17 and gcn-18) | Size |
> |---|---|
> | `/raid/prestouser/aocsa/tpch_parquet_sf100` | 26 GB |
> | `/raid/prestouser/aocsa/tpch_parquet_sf500` | 132 GB |
>
> Measured SF100 layout (gcn-18): `lineitem` **17,187,602,838 B** over 6 files
> (`part.0.parquet` … `part.5.parquet`), `orders` 5,051,383,146 B, `nation` 2,250 B.
>
> Verify rather than copy — run on **both** hosts and compare:
>
> ```bash
> find /raid/prestouser/aocsa/tpch_parquet_sf100 -type f -printf '%P %s\n' | sort | md5sum
> df -PT /raid/prestouser/aocsa/tpch_parquet_sf100 | tail -1     # expect /dev/md0  ext4
> ```
>
> **Do not use `/opt/sirius-ci/datasets/`** (`tpch_sf1 … tpch_sf1000`) or
> `/raid/prestouser/kkristensen/`. Those exist on **gcn-18 only**; a two-host run pointed at them
> hands gcn-17 byte ranges for files it cannot open. **There is no SF1 on either host** — the
> smallest two-host set is SF100.
>
> Do not point both hosts at `$HOME` (NFS, `master:/home`). `/scratch` is GPFS and *is* the same path
> on both hosts, but it is a cluster filesystem — using it measures GPFS, and would not be comparable
> to any number taken on node-local NVMe.

---

## C. Outstanding TODOs left in these documents

### Resolved since this document was written (2026-08-11)

| # | Was | Now |
|---|---|---|
| 1 | `UCX_MAX_RNDV_RAILS` unverified in the installed build | **RESOLVED.** `ucx_info -c` on UCX 1.21.0 → `UCX_MAX_RNDV_RAILS=2` is the default, so setting `4` genuinely engages the extra rails. |
| 2 | IMEX pending node-map may disable MNNVL on reload | **RESOLVED.** `nodes_config.cfg` and `.pending` are byte-identical on gcn-18; no reload hazard. Fabric reports `Completed`/`Success`, ClusterUUID `3482beb4-a3cd-48a4-9b6c-a6ba43bc59a4`. gcn-17's UUID must match — still to confirm. |
| — | Datasets exist on gcn-18 only and must be replicated | **RESOLVED — the premise was wrong.** SF100 and SF500 are already on both hosts at `/raid/prestouser/aocsa/tpch_parquet_sf{100,500}`. See B.4. No SF1 anywhere. |
| 7 | No two-host launcher exists | **RESOLVED.** `benchmarks/cn-2host.sh` (commit `8eced0f1`), with an HBM-membind interlock and preflight; negative tests verified. |

### Still open

| # | TODO | Resolve with |
|---|---|---|
| 3 | gcn-17 was never observed directly (`ssh` blocked for the agent) — NUMA layout, HCA↔netdev mapping, bond0 address, JDK path, LPDDR total, IMEX ClusterUUID | run `benchmarks/collect-host-facts.sh` on gcn-17; it writes `benchmarks/host-facts-presto-gb200-gcn-17.txt` into the NFS-shared repo. gcn-18's baseline is already captured. |
| 4 | The corrected `UCX_TLS` (with `cuda_ipc` retained) has never been measured cross-host | re-run the §2 table's 97 GB/s row with `UCX_TLS=cuda_copy,cuda_ipc,rc_mlx5,tcp,self` |
| 5 | `SIRIUS_EXCHANGE_STAGING_BYTES=16GiB` is validated at 4 CNs, not 2; fewer CNs means each carries more fan-out | run q05 and q09 at SF100 with `140GiB+16GiB` and the footprint-neutral `128GiB+28GiB` |
| 6 | `pipeline_dop = 36` (pinned) vs 64 (unpinned) is unresolved for 2 CNs | run both arms; `SNMG-PLAN.md:113-119` argues higher DOP cuts *against* distributed Sirius |
| 8 | FE metadata identity is `127.0.0.1_9010_…`; changing `priority_networks` makes the FE `exit(-1)` | plan Task 1.1 Step 3b (identity migration) — not yet executed |
