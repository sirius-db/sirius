# Open issues — the work queue for the multi-GPU box

State of the branch this doc ships with: **22/22 TPC-H pass at SF1**, unbounded back-to-back
execution (the arena leak is fixed), every result DuckDB-validated. The journey and evidence
live in `QUERY-TIMEOUT-ANALYSIS.md`; the integration learnings in `ROADMAP-8CN-TPCH.md`; the
benchmark protocol in `experimental/starrocks/benchmarks/tpch/REPRODUCE.md`; day-to-day
operations in the `tpch-bench` skill (`.claude/skills/tpch-bench/SKILL.md`). Reference
numbers: `experimental/starrocks/benchmarks/tpch/results/sf1-2026-08-07.md`.

Recommended order on the GB200 box: **M0 → M1 → M2 → #24 → #31**, with M3 as soon as the
SF100/SF1000 datasets land. M0–M2 unblock everything else on the new hardware. The scale
math behind M1–M3 is `TPCH-PLAN-ANALYSIS.md` §8.

---

> ## ⚠️ AUDIT 2026-08-08 — status markers below are verified against code
>
> Audited on the live 4× GB200 box (aarch64 Grace, NV18 full mesh, 185 GB HBM/GPU, ~980 GB
> LPDDR) with a 4-CN cluster up and a TPC-H **SF100** sweep running. Every `✅/🟡/🔴` marker
> below was checked against the working tree at `03ad7316`, not against this document.
>
> **This doc was committed 2–24 minutes stale.** `0bee7b01` (this file) landed at
> 19:53:04; `ae73d503` (which added `scripts/cn-env.sh` and obsoleted M0 step 3) landed at
> 19:50:49, and `0092077c` (which replaced the arena's bump-at-quiescence reset) at 19:29:11.
> M0 step 3, the M1 premise, and the M2 Phase-2 premise were **already false when written**.
>
> **Revised order given the audit: M1.4 → M1.5 → M2(a,c) → #31.3 → #24.** M0 is done except
> engine B; M1's launcher already existed. See "What to focus on now" at the end of this file.
>
> **Live failure profile at SF100 (4 CNs):** q01/q02/q03/q04/q06/q07/q12/q13/q14 pass;
> **q08 refused at 60758 ms** (≈ the hardcoded 60 s REPLY_TIMEOUT); q05/q09 wedge at the
> 180 s client timeout with no error; q11/q15 return empty. Note **q02 now passes** (2.4 s)
> despite being listed as a hard hang in "Known limitations".

---

> ## ⚠️ AUDIT 2026-08-09 — a measurement-integrity defect invalidated earlier numbers
>
> **Every Engine A number taken before 04:20 UTC on 2026-08-09 may have been measured on a
> cluster silently running at half capacity.** See **M6** below. Two of four CNs were
> auto-blacklisted ~2 s after *every* cluster start, before any query ran, and never came back.
> Re-take any Engine A result you intend to quote.
>
> **Six claims in this document's own history are RETRACTED** (details in M6 and in
> `/home/prestouser/aocsa/benchmark-results/cn-distribution-and-numa.md`):
>
> | Retracted claim | Reality |
> |---|---|
> | Fragments are distributed *unevenly* (~13× imbalance) | Distribution is **even — 1.02–1.06× spread**. It was *exclusion*, not imbalance. |
> | `FILES()` → `FileTableScanNode`, one scan range per file | Wrong node. `FILES()` SELECT builds a **`FileScanNode`** (`PlanFragmentBuilder.java:4254`) which emits real sub-file **byte ranges**. |
> | `parallelInstanceNum` is hardcoded to 1 | Overridden at `PlanFragmentBuilder.java:4270` with `pipeline_dop`; real default ≈ 9–18 per node. |
> | `prefer_compute_node=false` blocks fan-out | FE runs `shared_data`, so `isPreferComputeNode()` returns a literal **`true`**. The LOCALITY branch is unreachable here. |
> | NUMA host-memory tier is dormant | **ACTIVE.** `use_host_per_numa()` runs unconditionally; `numa_id` comes from NVML. |
> | The `host space count (1) != NUMA node count (34)` warning proves misconfiguration | **False alarm** — it counts raw `/sys` node dirs. 1 is correct for a 1-GPU CN. |
>
> **Also note:** `Mems_allowed_list` **cannot** see `numactl --membind` (it reflects the cpuset
> controller; `--membind` sets an `MPOL_BIND` policy visible only in `/proc/<pid>/numa_maps`).
> `cluster4-numa.sh:354-356` tells operators to verify exactly this wrong way — fix it.

---



## M6. CNs silently excluded by a permanent FE blacklist ✅ **FIXED 2026-08-09 — but read this**

**This is the most consequential defect found so far, because it did not fail loudly — it made a
4-GPU cluster quietly behave like a 2-GPU one while reporting `Alive = true` on all four.**

### Symptom
`SHOW COMPUTE NODE BLACKLIST` on a *freshly started* FE, before any query:

```
| ComputeNodeId | AddBlackListType | LostConnectionTime  |
| 10001         | AUTO             | 2026-08-09 04:06:59 |
| 10003         | AUTO             | 2026-08-09 04:06:59 |
```

Still blacklisted 1 m 42 s later against a nominal 500 ms penalty. Reproduced on consecutive
launches, deterministically the same two node IDs. q14 then ran **48.9 / 0 / 0 / 51.1 %** across the
four CNs — the two survivors split the work almost perfectly; the other two did *nothing*
(zero operators, tasks, plans, ports).

### Root cause — two independent defects, both required
1. **Trigger — a start-up ordering race.** The CN built the GPU engine *before* binding any socket.
   Engine start reserves the whole RMM pool: **~6.9 s on a GB200** (measured 04:20:18.70 → 25.61)
   during which every port refused. The FE is up in ~4 s and immediately heartbeats every compute
   node it remembers from **persisted metadata** (`ComputeNodeId` 10001–10004 were byte-identical
   across restarts), so it always probed into that window.
2. **Why it was permanent.** `HostBlacklist.remove()` (`HostBlacklist.java:208-214`) only evicts
   once `NetUtils.checkAccessibleForAllPorts(host, [bePort, brpcPort, httpPort])` connects to **all
   three**. The CN **advertised `http_port` (`src/lib.rs:462`) and never bound it** — no HTTP server
   exists in `Cargo.toml`. The probe always refused, so eviction never ran. `black_host_penalty_min_ms`
   and friends are irrelevant; the port gate sits upstream of all of them.

### Fix (both landed, both verified live)
* **`start_http_server`** (`src/lib.rs`) — bind the advertised port. Minimal listener, fixed 200,
  read/write timeouts. Verified: the FE now auto-evicts in ~1 s
  (`remove nodeID 10003 from blacklist` at 04:20:27.882).
* **Listeners before engine init + `EngineReadiness` gate** (`src/lib.rs`, `src/main.rs`) — bind
  heartbeat/backend/http *first*, build the engine second, open the gate only once BRPC is up.
  While warming, the heartbeat is **answered** (so the node is reachable, not blacklistable) with a
  **non-OK status** (so the FE holds it not-alive and never schedules onto a cold engine). A failed
  *heartbeat* marks a node dead; only a failed *fragment RPC* blacklists it — that asymmetry is what
  makes the gate safe.
  Verified 05:10: ports bound at t+0, engine ready at t+6.9 s, **zero `HostBlacklist` lines**.
  The race no longer produces an entry at all rather than producing one that heals.

### Consequences you must act on
* **Re-take any Engine A measurement from before 2026-08-09 04:20 UTC**, including the CN-scaling
  CSVs. They may be half-capacity runs.
* **The harness must assert an empty blacklist** as a pre- and post-condition of every measured run.
  Being `Alive` is not sufficient and never was.

### Still open here
The blacklist adds were emitted by thread `starrocks-mysql-nio-pool-0` (a **client** thread) 1.3 s
*after* the ports bound — not by `heartbeat-mgr-pool`. The exact call site is **not identified**, and
the determinism (same two IDs every time) suggests something structural rather than a timing
coin-flip. The fix removes the symptom; the mechanism deserves one more look.

---

## M4. `use_odirect` is media-blind — O_DIRECT on NFS costs 20× 🔴 **NEW**

> ### ⚠️ CORRECTED 2026-08-09 — the original claim in this section was WRONG
> This section first read *"the scan datasource default is a 5–21× regression; switch to the cudf
> datasource."* **That finding was an artifact and the recommendation was wrong.** A controlled
> sweep (`benchmark-results/scan-defaults-sweep.md`) established:
>
> - The SF100 dataset lives on **NFS, `rsize=32768`**. `io::uring::config::use_odirect` defaults to
>   **`true`** (`src/include/io/uring/config.hpp:27`), so the uring path bypasses page cache and
>   pays the NFS wire cost on *every* read. The cudf/kvikio path does **buffered** reads and is
>   served from a 448 GB page cache after first touch. **That is the entire 5–21×.**
> - Measured ceilings: **NFS+O_DIRECT is flat at ~0.78 GB/s** at any concurrency (1 thr 0.64 →
>   64 thr 0.78). Sirius's uring path reached 0.69–0.72 GB/s — **~90% of the achievable ceiling.
>   The reader was never the bottleneck.**
> - **Both on local NVMe, stock defaults: cudf is only 1.08× faster — no real gap.**
> - **Tuned uring beats cudf: 1.50× on NFS; with O_DIRECT off, 21.7 vs 14.5 GB/s.**
>
> **Do NOT flip `use_sirius_datasource`.** Flipping it also forfeits multi-GPU, which
> force-enables it anyway (`sirius_config.cpp:507-513`). Fix `use_odirect` instead.
>
> ⚠️ Every benchmark number collected on 2026-08-08 used `use_sirius_datasource=false` on
> NFS-resident SF100. The timings are real; the *attribution* to "cudf's reader is better" is not.

**What**: `io::uring::config::use_odirect{true}` is applied regardless of the underlying media.
O_DIRECT is correct on local NVMe and catastrophic on a network mount. Related undersized
defaults in `src/include/scan_manager/config.hpp`: `uring_n_reactors{1}` (:47),
`thread_pool.num_threads{8}` on a 144-core box (:43), `enable_prefetch_cache{false}` (:55).

**Do**:
1. **Make `use_odirect` media-aware** — probe the filesystem (`statfs` / `findmnt`) and disable
   O_DIRECT on network mounts. This is the whole finding.
2. **Derive `uring_n_reactors` from core count.** On NVMe+O_DIRECT throughput scales
   **2.34 → 25.29 GB/s from 1 to 64 threads**, so `1` is a genuine ~10× bottleneck *on local
   disk* — invisible on NFS only because the mount capped everything. This matters at SF1000,
   where cold scans read ~95 GB/node. Suggested:
   `uring_n_reactors{std::clamp(hardware_concurrency()/16, 4u, 16u)}`.
3. Derive `num_threads` from core count (small measured effect). Keep `enable_prefetch_cache`
   false. `max_n_chunks{1}` showed no measured effect.
4. ⚠️ **Reactor threads are not NUMA-pinned.** `uring_reactor.cpp:31` includes `<numa.h>` but
   makes **no `numa_*` calls**; `cpu_affinity_list` pins only the scan_manager pool, not reactors.
   On this two-domain box CN0/CN1's reactors belong on node 0 and CN2/CN3's on node 1. See M1.2.

**Superseded original text follows.** `scan_manager.use_sirius_datasource` defaults to **`true`**
(`src/include/scan_manager/config.hpp:44`), selecting Sirius's own io_uring reader. Setting it
**`false`** selects the kvikio/cudf datasource and is **5–21× faster** on every query measured.

| Q (SF100, 4 CN) | uring (default) | cudf datasource | speedup |
|---|---|---|---|
| q01 | 6093 ms | **903 ms** | 6.7× |
| q04 | 3700 ms | **651 ms** | 5.7× |
| q06 | 5285 ms | **499 ms** | 10.6× |
| q14 | 7158 ms | **492 ms** | 14.5× |

Standalone single-GPU (DuckDB+Sirius, no FE) is even starker: q06 4841 → **241 ms (20×)**,
q04 3149 → 222 ms, q14 7144 → 271 ms. All results **bit-identical to the DuckDB oracle**.
With the fix Sirius goes from *slowest* of three engines to *fastest* on all four queries
(vs stock StarRocks CPU 4-BE and cudf-polars 4-GPU on the same box and parquet).

**Why it is probably not "cudf's reader is better"**: the uring path runs with
`uring_n_reactors{1}` — **one** I/O thread — and `thread_pool.num_threads{8}` on a **144-core**
box, with `enable_prefetch_cache{false}`. All three defaults are untested on this hardware.
⚠️ **Measure before concluding**: raising `uring_n_reactors` may close most of the gap, in which
case this is an undersized default, not a superior competitor.

**Do**:
1. Sweep `uring_n_reactors` (1 → 8/16/32) and `num_threads` (8 → 72/144) on q06 standalone.
   Cheapest decisive experiment in this document.
2. If uring stays behind, flip the default to `false` for 1-GPU-per-CN deployments.
3. Either way, derive `num_threads` from core count rather than hardcoding 8.

**Plumbing (already landed, uncommitted)**: `experimental/starrocks/src/engine_settings.rs`
now emits `sirius: executor: scan_manager: use_sirius_datasource` into each CN's derived YAML,
gated on env `SIRIUS_CN_USE_SIRIUS_DATASOURCE`. The multi-GPU guard
(`sirius_scan_manager.cpp:244-251`) does **not** trip, because each CN pins one device and its
derived YAML declares `num_gpus: 1`. Verified live: `/proc/<pid>/maps` goes 0 → 8 cufile maps.

**Note it is NOT GDS.** cuFile initialises then falls back to its POSIX pool on *both* NFS and
local NVMe — see M5.

## M5. GPUDirect Storage is unreachable on this box 🔴 **NEW — closes the "enable GDS" idea**

`nvidia_fs` is loaded, GDS 1.17, `/raid` is local ext4 on NVMe — every prerequisite present, yet
`/proc/driver/nvidia-fs/stats` shows `Ops: Read=0 Write=0` and `Registered_MiB=0` on all 4 GPUs
after a full scan. `cufile.log`:

```
cufio-cuda:650 Config doesn't support PCIP2PDMA
cufio-drv:138  pci_p2pdma not supported errornum: 801 CUDA_ERROR_NOT_SUPPORTED
cufio-drv:146  Is PCIP2PDMA Supported: 0  kernel Support: 1  pci_p2pdma_supported_cuda: 0
cufio-drv:510  NVMe : nvfs, compat
```

`kernel Support: 1` but `pci_p2pdma_supported_cuda: 0` — **CUDA** reports P2P DMA unsupported on
GB200, where the GPU attaches over NVLink-C2C rather than the PCIe path cuFile looks for. On NFS
the failure differs (`NFS NVFS symbols not found`) but the outcome is the same: POSIX pool.
**There is no GDS upside to unlock here.** Do not spend effort on it; revisit only on a platform
that reports `pci_p2pdma_supported_cuda: 1`.

## M0. aarch64 port ✅ **DONE (except step 5)** — completed 2026-08-08

**What**: ~~the target node's Grace CPUs are aarch64; the demo's transport stack ships
x86-64 prebuilts.~~ **No longer true.** `tools/nvda_nixl/lib/` now contains *only*
`aarch64-linux-gnu/` (+ `python3`); the x86_64 dir is gone. `tools/ucx-install/lib/libucp.so`
is aarch64 ELF. The CN builds and runs.

**Do**:

1. ✅ **DONE** — UCX 1.21.0 rebuilt from source with `--with-cuda=$CUDA_HOME --enable-mt`.
  📝 *Doc nit*: the source dir is `tools/ucx-1.21.0`, **not** `tools/ucx-src` (never existed).
2. ✅ **DONE** — nixl meson-rebuilt against that UCX; install dir is
  `tools/nvda_nixl/lib/aarch64-linux-gnu/`, `plugins/libplugin_UCX.so` present.
3. ✅ **DONE — AND THIS STEP WAS ALREADY OBSOLETE WHEN WRITTEN. DELETE IT.**
  `pixi.toml` contains **none** of the five strings this step names (no
   `x86_64-linux-gnu`, no `/home/ubuntu`). `scripts/cn-env.sh` (added by `ae73d503`,
   **2 min 15 s before this doc was committed**) derives every path: it globs
   `lib/*-linux-gnu` (`cn-env.sh:25-29`) so the arch dir name is auto-detected, and already
   exports `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER` (`cn-env.sh:48-49`).
   ⚠️ Note `cn-env.sh` reads `TOOLS_DIR` (with the S); the runbook's `TOOL_DIR` is inert.
4. ✅ **DONE** — `pixi install` + full engine and CN rebuild; both aarch64 ELF.
5. 🔴 **OPEN — the assertion "nothing needed for engine B" is FALSE on this box.**
  `setup-engine-b.sh:15-23` extracts artifacts **via** `docker create`**/**`docker cp`, and
   **docker is not installed here**. Engine B was staged instead from the arm64 release
   tarball `StarRocks-3.5.20-ubuntu-arm64.tar.gz`, which downloads fine — contradicting the
   script's own header comment (`:2-3`, "the release tarball URLs 403"). Also
   `setup-engine-b.sh:44,55` **rewrites** `mem_limit = 16G` **on every re-run**, silently
   reverting any re-baseline. Fix: add a non-docker extraction path + stop hardcoding 16G.

**Verify**: ✅ CN starts, loaded .so's are aarch64, **q06 passes (5.3 s at SF100)**.

## M1. Multi-GPU cluster bring-up 🟡 **PARTIAL — items 4 & 5 are the live blockers**

**What**: ~~the demo's cluster task is hardcoded to 2 CNs sharing one GPU. The GB200 box
wants a~~ `cluster4` — ⚠️ **PREMISE STALE/WRONG.** Two parameterized N-CN launchers already
existed when this doc was written: `benchmarks/cluster8.sh:24,65-79` (`NUM_CNS`, default 8)
and `benchmarks/nixl-nvlink/script-box.sh:28,106-120`. **No** `cluster4` **needs creating** —
`NUM_CNS=4 ./benchmarks/cluster8.sh` brought up 4 alive CNs on the live box.

**Where**: `experimental/starrocks/pixi.toml` — the `cluster2` task (~line 60-81) launches
the FE and two `sirius-starrocks-cn` processes with `--gpu-memory-limit 8GiB --host-memory-limit 12GiB --engine-dir .cnN` and per-CN port offsets (+2 per CN:
heartbeat 9050/9052/…, thrift 9060/9062/…, brpc 8060/8062/…, http 8040/8042/…,
starlet 9070/9072/…).

**Do**:

1. ✅ **ADDRESSED** — `cluster8.sh:67-76` already does one CN per GPU via `--gpu-device <i>`
  with distinct `--engine-dir` and a contiguous 10-port block ladder.
   🟡 `MIN_BACKENDS=4` is **not** the default: `benchmarks/tpch/bench.sh:39` is still
   `MIN_BACKENDS=${MIN_BACKENDS:-2}` — env-overridable only, so a 4-CN sweep started with
   the default can begin against a half-booted cluster.
2. 🔴 **OPEN — zero implementation.** Repo-wide grep for `numactl|physcpubind|membind` finds
  **no hit** in `cluster8.sh`, `script-box.sh`, or `cn-env.sh`; the only executable hit is
   stock StarRocks' own `starrocks/bin/start_backend.sh:121`. The CN has no `--numa` flag.
   ✅ Confirmed on the box: nodes **2/10/18/26 are the four 184 GiB HBM regions** (no CPUs) —
   membinding them would be a silent catastrophe. CPU memory is only nodes 0 and 1.
3. 🟡 **PARTIAL** — parameterized but every default is far too small: `cluster8.sh:29`
  defaults `GPU_MEM=64GiB` (A100 sizing), `script-box.sh:42` uses `40GiB`. The live run
   passed `140GiB` by hand.
   ⚠️ **Correction**: this box has ~**980 GB** LPDDR (2 × 490 GB), not 960 GB — and `free -g`
   misleadingly reports ~1692 GB because it counts the 4 × 184 GiB HBM NUMA nodes. At 4 CNs,
   "`--host-memory-limit` 160–200 GiB/CN" (=640–800 GiB) is **incompatible** with "leave
   ≥400 GB for page cache". Pick one; at SF100 (26 GB dataset) page cache barely matters, at
   SF1000 (~355–390 GB) it dominates.
4. 🔴 **OPEN — and WORSE than stated. ← START HERE**
  - **Watchdog is not merely low, it is DISABLED** under the launcher actually in use:
   neither `cluster8.sh` nor `script-box.sh` sets `SIRIUS_QUERY_WATCHDOG_SECS` at all
   (only the `cluster2` pixi task did, at 60). With it unset, `src/sirius_engine.cpp:110-113`
   takes a **blocking** `future.get()` with no timeout → the 180 s silent wedges.
  - **REPLY_TIMEOUT is still a hardcoded compile-time constant**:
  `experimental/starrocks/src/prpc_client.rs:25` → `Duration::from_secs(60)`.
  🎯 **This matches the live q08 failure at 60758 ms almost exactly.**
5. 🔥 **OPEN — AND IT FIRED. NO LONGER THEORETICAL.** Sessions are strictly lazy
  (`nixl_transport.rs:402,497`); `start` takes no peer list (`:191-194`); MD is still
   answered **on the transport thread** (`compute_node_service.rs:949`). Both of the
   ROADMAP's alternative fixes (`ROADMAP-8CN-TPCH.md:489-492`) are also unimplemented.

   **Reproduced 2026-08-08 on 4 CNs.** A fresh cluster, then TPC-H q14 run **once**:
   ```
   run 1 (COLD, first cross-CN query):  FAILED after 121 s
     FE: exec rpc error, backend [id=10002], THRIFT_RPC_ERROR, fragmentId=F05
     cause: errorCode=62 "method request time out ... 60000 (MILLISECONDS)"
            bound channel => R:/127.0.0.1:9102   (9102 = cn0 brpc)
     then: "acknowledging cancel_plan_fragment (best-effort: no engine-side abort yet)"
   run 2 (WARM, same cluster, same SQL):  751 ms, correct result
   ```
   One cold failure, one warm success — the first-contact signature exactly as predicted.
   ⚠️ **This is why it was never seen before:** `bench.sh` runs "1 discarded warm-up + 3 timed",
   so the warm-up absorbed this failure on every sweep — including after every `RESTART_CMD`
   restart. **Every benchmark number in this repo was collected on a cluster whose first
   cross-CN query had already been sacrificed.** See #32.
6. ✅ **ADDRESSED** — CNs self-register via heartbeat (`src/main.rs:21-27`); 4 alive on the
  live box with no FE changes.

**Verify**: ✅ `SHOW COMPUTE NODES` = 4 alive. 🔴 full sweep **not** 22/22 — at SF100:
9 pass, q08 refused @60 s, q05/q09 wedge @180 s, q11/q15 empty. Endurance not yet attempted.

**Watch out**: `UCX_TLS=cuda_copy,cuda_ipc,tcp,self` suffices — cuda_ipc rides NVLink P2P
automatically on the all-to-all NV18 box; keep the 2 GB/s bandwidth canary to catch silent
fallback. The nixl plugin dir env is absolute in pixi.toml — M0 fixes the paths. Multi-NODE
(not just multi-GPU) additionally needs the nixl RDMA tier, which this demo has never
exercised — the 8 idle mlx5 NICs stay a separate project, not a config change.

## M2. Staging arena 🟡 **Phase 1 fully OPEN; Phase 2 partly done**

**What**: `SIRIUS_EXCHANGE_STAGING_BYTES` is a hand-tuned 1280 MiB, ~~bump-reset only at full
quiescence~~ — ⚠️ **premise WRONG since** `0092077c` **(2026-08-07 19:29:11, 24 min before this
doc was committed)**: the old `if (leases_.empty()) { head_ = 0; }` was replaced by trailing
reclamation (`exchange_staging_arena.cpp:112-124`). The doc's *conclusion* still holds for
the receiver side, but for a different reason — see Phase 2 below. Demand scales with fan-in (a receiver holds staging from up to N−1 senders) and
with data volume (TPC-H q09 needs a single ~648 MB lease at SF1 with 2 CNs). At SF100 on
4 CNs, staged inbound per CN is 1.9–6 GB (q03/q05/q17/q18/q21) and single leases exceed the
whole slab (q16 broadcast payload 1.28 GB, q04 3.0 GB) — ≥7 queries fail on day one. At
SF1000 the bump-reset semantics themselves are the limit: demand is cumulative per query
epoch, ~105–125 GB/CN (q05/q18/q21), and no fixed slab coexists with 80–150 GB operator
peaks in 185 GiB HBM (`TPCH-PLAN-ANALYSIS.md` §8.3).

**Where**: the env var is read at CN bring-up (`experimental/starrocks/src/engine_settings.rs`
derives the engine config; the arena itself is `src/exec/exchange_staging_arena.{hpp,cpp}`).

**Do**, in two phases:

1. **Sizing + chunked leases (gates SF100)** — 🔴 **ALL FOUR ASKS OPEN, not one line written.**
  - (a) derived default when env var unset → 🔴 OPEN (`exchange_staging_arena.cpp:55-56`;
   no `cudaMemGetInfo` anywhere outside `src/legacy/`).
  - (b) fail loudly at bring-up → 🔴 OPEN; only a *reactive* `cudaMalloc` guard exists
  (`:39-48`), and the carve-out is accounting-only (`sirius_config.cpp:260-280`).
  - (c) ~256 MB per-lease cap + chunked multi-lease export → 🔴 OPEN (`:75-99` has no cap;
  `sirius_ffi.cpp:743` takes **one lease for a whole table**). 💡 The machinery is already
  there — `chunked_pack` is per-8 MiB; what's missing is *leasing per chunk*.
  - (d) log chosen size + high_water → 🔴 OPEN (`high_water()` is called only by a unit test).
   ℹ️ The live SF100 run sidestepped (a)+(c) entirely by passing `STAGING=16GiB` by hand —
   which is why **q04 passed** despite its ~3.01 GB single lease that the doc predicted
   would fail against the 1280 MiB slab.
2. **Eager per-lease reclamation (gates SF1000)** — 🟡 **PARTIALLY ADDRESSED.**
  - ✅ Trailing reclamation landed in `0092077c` (`:112-124`).
  - ✅ Sender-side eager release is done: `nixl_transport.rs:505-552` releases per batch
  post-WRITE on both success and error paths — exactly the "post-WRITE" release asked for.
  - 🔴 **No free list / ring** — gaps below the head stay unreachable, asserted deliberately
  in `ARENA-1` (`test_exchange_staging_arena.cpp:51-57`).
  - 🔴 **Receiver side is still cumulative per epoch** — `local_exchange.rs:44` buffers
  `batches: Vec<StagedBatch>` per remote sender, holding leases from arrival to drain.
  **This is the one M2 item that actually gates SF1000.**
  - 🔴 Broadcast lease-reuse loop OPEN (`engine.rs:440-449` declares one stream per
  destination; `nixl_transport.rs:496-552` leases per destination).

**Verify**: bring-up on 1-GPU and N-GPU boxes without setting the var; q16 at SF100 (the
single-lease unit test) and q09 pass; q05/q18/q21 at SF1000 with high_water ≪ cumulative
epoch bytes; the oversized-request error still names request/free/capacity; endurance
sweeps.

## M3. A fair benchmark at scale (SF100 / SF1000)

**What**: at SF1 (388 MB), fixed per-query overheads made stock StarRocks ~2× faster
overall (geo-mean 0.48x; Sirius won only q01/q09/q19). The GPU's economics need data. This
is the demo's headline experiment on the GB200 box. Targets: **SF100** (600M lineitem rows;
measured parquet 35.7 GB total, lineitem 22.8 GB, snappy) and **SF1000** (6.0B lineitem
rows; ~355–390 GB total, lineitem ~228 GB). Per-CN share at 4 CNs: 150M / 1.5B lineitem
rows — 1.5B is 70% of cuDF's 2^31 row cap, so 4 CNs is the floor topology for SF1000.

**Do**: generate with the `dataset-manager` skill (or DuckDB
`CALL dbgen(sf=...); EXPORT DATABASE '<dir>' (FORMAT PARQUET);`) as **multiple files per
table, a multiple of 4**: SF100 8–16 files/table; SF1000 lineitem 32–64 files (~2.5–5 GB
each), row groups 128–256 MB — whole-file-contiguous ranges balance the FE's byte-range
splits. Then rerun `benchmarks/tpch/run-comparison.sh` with `TPCH_DATA` pointed at it.

**Storage**: local NVMe, not network — cold SF1000 reads ~90–97 GB/CN. One warm pass
page-caches the entire dataset in the ~960 GB of LPDDR (verify at bring-up, M1), after
which scans come from RAM over C2C — publish cold and warm medians separately.

**Watch out**: arena sizing (M2 phase 1) gates SF100 day one; q08/q09 stay excluded until
the cross-join re-association lands (their intermediates are O(SF²): 2.1 TB / 13 TB at
SF100 — `TPCH-PLAN-ANALYSIS.md` §8.2); q04/q10/q13 need the 2^31 guards at SF1000. Engine B
(stock StarRocks BEs on the 144 Grace cores) needs a re-baselined `mem_limit` in
`setup-engine-b.sh` — don't reuse the L4-host numbers.

---



## #24. Decimal-native aggregation 🔴 **OPEN — but the doc mislocates the real gap**

> **AUDIT 2026-08-08.** Decimal→FP64 is still real, but **only in the Rust translator**; the
> **C++ engine is already substantially decimal-native**, so this item is smaller than it looks
> — and the doc's "Where" list points at the wrong line for the dominant error.
>
> 🎯 **THE UNDOCUMENTED ROOT CAUSE:** `expr_translator.rs:459-481` (`translate_arithmetic`)
> casts **both operands of every decimal** `+ - * / %` to FP64 and declares FP64 output. So
> `l_extendedprice * (1 - l_discount)` is **already FP64 before SUM ever sees it**. Fixing only
> the SUM/AVG lowering at `:826-833` — which is what the doc tells you to do — would change
> **nothing** for q01/q03/q05/q07/q09/q10/q14. Start at `translate_arithmetic`, not at the
> aggregate.
>
> ✅ **Already done, verify only** (the doc calls these gaps; they are not):
>
> - Decimal SUM widening is implemented in **3 of 4 aggregate paths**; decimal AVG already
> divides in fixed point (`sirius_physical_grouped_aggregate_merge.cpp:268-272`).
> - Dedicated DECIMAL64 test coverage exists (`test/cpp/operator/aggregate/aggregate_test_utils.hpp:80-120,224-256`).
> - `cudf_utils.hpp:158-213` DECIMAL(p,s) round-trip; `throw_if_int64_sum_could_overflow`.
> - **The q15 acceptance criterion needs ZERO new code**: `is_order_sensitive_sum`
> (`aggregate_op_util.cpp:225-229`) is FLOAT32/64-**only**, so the canonical-sort path
> self-disables the moment the input becomes decimal.
>
> 📝 **Path correction**: the expression code is `src/expression_evaluator/`, not
> `src/expression/` (which is only AST node definitions).
> ⚠️ Doc's own line numbers for `:826-833`, `:999-1003`, and the 76-case parity gate are **exact**.

**What**: the FE plans money arithmetic as exact DECIMAL64/128; the translator lowers
decimal SUM/AVG to FP64. Consequences, all measured:

- Every `sum(x*(1-l_discount))` lands 0.1–0.4 % LOW vs DuckDB (q01 sum_disc_price/charge,
q03/q05/q07/q09/q10 revenues, q08 mkt_share, q14 promo_revenue) — deterministic,
low-biased, occasionally reorders near-tie ORDER BY rows (q10 rank rotation).
- FP64 hash-groupby accumulates via cudf atomicAdd (bit-nondeterministic), which forced the
q15 workaround: canonical-order + `sorted::YES` (atomics-free path) on every float-SUM
groupby — correct, but pays a sort per aggregation.

**Where** (all mapped during the q15 investigation — wf journals + QUERY-TIMEOUT-ANALYSIS.md):

- Translator: `expr_translator.rs:826-833` (decimal SUM/AVG → `cast_to_fp64`, :999-1003),
`type_mapper.rs` (DECIMAL(p>18) → Fp64 lowering), the partial-state wire-type model
(`partial_state.rs` — decimal sums modeled as FP64 on the wire).
- Engine: cuDF supports `fixed_point` (DECIMAL32/64/128) natively; fixed-point SUM uses
exact integer atomics (`device_aggregators.cuh:126`) — deterministic and exact. The scan
already reads decimal parquet columns; the gaps are the expression path
(`src/expression/`, arithmetic over fixed_point incl. the (1 − DECIMAL) literal shape),
aggregate/merge output types (`gpu_aggregate_impl.cpp`, `gpu_merge_impl.cpp`,
`aggregate_op_util.cpp`), and the exchange schema spelling (`type_mapper.rs` ↔
`get_cudf_type`, `cudf_utils.hpp:158-210` — DECIMAL(p,s) strings already round-trip).
- Mind the 76-case `wire_type_parity` gate (`experimental/starrocks/src/wire_type_parity.rs`)
— it will enforce whatever the model says; update model + engine together.

**Approach hint**: start with SUM(DECIMAL64) end-to-end (scan → partial sum → wire → merge
→ finalize), gate q06/q01 on bit-exactness vs DuckDB, then widen (AVG expansion states,
DECIMAL128, the (1−d) literal). The avg `__count` expansion machinery is orthogonal and
stays.

**Acceptance**: q01/q05/q06/q14 values bit-exact vs DuckDB (no tolerance band); q15 stays
8/8 WITHOUT the canonical-sort path on decimal inputs (keep it for genuine float columns);
the overflow guard (`throw_if_int64_sum_could_overflow`) still covers 64-bit integer sums;
full suite + sweep green.

## #31. The engine-abort surface 🔴 **ALL THREE OPEN — doc is accurate, line numbers exact**

> **AUDIT 2026-08-08.** Verified at `03ad7316`: nothing here has been addressed, and the doc
> is not wrong anywhere. One **scoping correction**: item 1 is a *latent landmine* that
> produces the q05/q09 wedge fingerprint but is **not their cause today** — no TPC-H plan
> emits MULTI_CAST_DATA_STREAM_SINK. **Item 3 is the one that explains the live failures.**
> Confirmed stubs: `compute_node_service.rs:270-299`, `lib.rs:556-566`,
> `src/pipeline/task_scheduler.cpp:238-256`. Item 2's error string is still at `engine.rs:304`.

Three catalogued gaps, one root: the engine cannot abort a running fragment.

1. **Unhandled sink types are silently swallowed**:
  `experimental/starrocks/src/compute_node_service.rs` ~:752-760 — a fragment whose output
   sink is not DATA_STREAM_SINK (e.g. MULTI_CAST_DATA_STREAM_SINK, emitted under
   `cbo_cte_force_reuse_node_count=1`) falls into a let-else returning `Ok(Vec::new())`:
   output discarded, consumers hang, the FE's serial channel wedges cluster-wide.
   Fix: `Err` on any unhandled sink (do this first — 30 minutes); optionally implement
   MULTI_CAST (run the fragment once, park/transmit per nested sink destination) — that also
   enables single-evaluation CTE reuse, structurally removing q15's double-evaluation shape.
2. **Parked-sender bookkeeping defect**: `"no parked sender output to export for
  SenderSlot { node_id: 3, sender_id: 1 }"`, reproducible by adding` CAST(sum AS VARCHAR)`
   to a grouped-CTE probe. Uninvestigated; likely the park/claim accounting for a plan shape
   where an expected sender output was never parked.
3. **Mid-query export failure wedges the CN**: when a sender-side export fails mid-run, the
  client sees a silent timeout (not the loud error) and the wedged statement head-of-line
   blocks the next until restart. The failure propagation (c858e79a) covers pre-run
   failures; mid-run needs the engine to actually abort the fragment. Scaffolding exists:
   the watchdog + `terminate_query` + SIGTERM escalation (a94e8660) and `cancel_plan_fragment`
   stubs (PRPC + thrift) — the missing piece is engine-side fragment abort (interrupt the
   stream waits / task loop; respect the 19d7cca2 manager-thread constraints).

**Acceptance**: no plan shape can hang silently — every unsupported/failed path yields a
loud FE-visible error within seconds AND the next statement runs without restarting; a
real cancel frees GPU memory mid-query.

---



## #32. The benchmark harness hides the defects it should expose 🔴 **NEW — fix before quoting any number**

Three harness properties, each of which silently corrupted a conclusion this session:

1. **The discarded warm-up masks cold-start failures.** `bench.sh` runs 1 warm-up + 3 timed and
   throws the warm-up away. That is precisely the run that exercises first contact (M1.5), so a
   100%-reproducible cold-start hang was invisible across every sweep ever run here.
   **Fix**: a `--cold` mode that *records* run 0; run it at least once per cluster restart.
2. **No correctness gate anywhere.** `bench.sh:73-75` scores `pass` on exit-code + non-empty
   output + no `ERROR` on line 1. `analyze.py` reads **only** `status` and `ms` — it never
   compares row counts between engines. A query returning 1 row instead of 100,000 is recorded
   as a fast **win**. **Fix**: collect `rows`, mark cross-engine mismatches, exclude them from
   the geomean.
3. **`wait_alive` over-counts.** `bench.sh:46-48` does `grep -c true` on the whole
   `SHOW COMPUTE NODES` row, so `HasStoragePath=true` also matches; `MIN_BACKENDS` can be
   satisfied by a half-booted cluster. **Fix**: `awk -F'\t' '$9=="true"'`. (`MIN_BACKENDS`
   also still defaults to 2 — see M1.1.)

## #33. Scale-out economics, measured 🟡 **NEW — informational, sets the roadmap**

Full 1/2/4-CN curves at two scale factors, cudf datasource, 90/90 runs passed, zero wedges:

| Q | SF100 1→4 CN | SF500 1→4 CN | limiter (measured shuffle/run) |
|---|---|---|---|
| q01 | **2.69×** | **3.40×** (85% eff.) | none — 960 **bytes** shuffled |
| q04 | 0.88× (worse) | 1.27× | shuffle-bound — 14.35 GiB @2CN |
| q06 | 0.88× (worse) | 1.11× | **87% serial** — 0 bytes shuffled |
| q14 | 0.95× (worse) | 1.37× | shuffle **grows** with N: 1.69→2.53 GiB |

- **Scale-out is net-negative at SF100 and net-positive at SF500.** Fitted break-even:
  q14 ≈ SF130, q04 ≈ SF179, q06 ≈ SF237; q01 pays at every scale.
- **Overhead is a step, not a slope.** Every regression is at the 1→2 boundary; **2→4 never
  regresses**. The cost is *entering* distributed mode, and it does not compound with N.
- **But 4 GB200s under StarRocks still lose to 1 GB200 standalone at both scale factors**
  (SF100 sweep: 2827 ms vs **1288 ms**). The tax at 1 CN is +212 ms (q06) to +2001 ms (q01) —
  a fixed ~200–400 ms per query, plus a work-proportional inefficiency visible only on q01.
- 🔎 **q14 shuffle growing with N** and **q06's serialized tail** (the GATHER host runs its own
  scan *alone* after the other CNs finish, then tears down and re-opens a DuckDB session) are
  the two concrete leads.

## #34. nixl/UCX is NOT the bottleneck ✅ **CLOSED — measured, stop looking here**

Benchmarked with the CN's own Rust primitives (`nixl_transport.rs` `write_and_wait` over the
engine's staging arena), GPU0→GPU1, 10 timed + 3 warm-up per size:

| payload | tcp-naive | nixl | vs TCP |
|---|---|---|---|
| 1 GiB | 7.93 GB/s | **628.21 GB/s** | **79.2×** |
| 16 MiB | 7.53 | 214.17 (post→DONE **399.91**) | 28.5× |

- NVLink counters account for **99.99–100.00%** of bytes on every nixl phase, **0 B** on every
  TCP phase. A GPU0→GPU0 negative control gave 2525 GB/s with 0 B on NVLink — the counter
  discriminates.
- **Silent-degradation cliff reproduced**: removing `cuda_ipc` → **0.47 GB/s (1349×)**, bytes
  still correct, no error raised. That is ~4× *below* `CANARY_FLOOR_GBPS = 2.0`, so **the 2.0
  floor is correct on Grace** — previously an open question.
- 16 MiB post→DONE **399.91 GB/s** matches the live cluster canary's 322–399 GB/s per peer:
  independent confirmation the harness measures what production measures.
- 📌 **628 GB/s is a software ceiling, not the link's**: a plain `cudaMemcpyPeer` probe reaches
  **773.8 GB/s** on the same box (~16% headroom above nixl at ≥256 MiB). NV18 line rate is
  956 GB/s, so the reference report's "exceeds the unidirectional spec" caveat does not apply.

## 🎯 What to focus on now (audit 2026-08-09 — supersedes the 2026-08-08 list below)

> **The distribution question is CLOSED.** Fragments are distributed evenly (1.02–1.06× spread) once
> the CNs are actually schedulable; see **M6**. NUMA layers 1 and 2 are ACTIVE. What remains is a
> *performance* question, and it is now sharply posed:
>
> ### The central open puzzle
> **At SF100, four GB200s lose to one GB200 on every query — with distribution provably even.**
>
> | | Q01 | Q04 | Q06 | Q14 |
> |---|---|---|---|---|
> | Engine A, 4 CN (all 4 verified) | 913 | 617 | 376 | 569 |
> | Engine D, 1 GPU, no FE | **554** | **222** | **241** | **271** |
>
> FE+client overhead is ~107 ms and does not close the gap. At **SF500** scale-out finally works,
> but unevenly: Q01 reaches **3.5×** on 4 CNs while Q04/Q06/Q14 manage only **1.4–1.6×**, and Q06 is
> *slower* on 2 CNs (1538 ms) than on 1 (1227 ms). Read together: a large **per-query fixed cost**
> that is independent of data size, plus a shuffle that eats most of the parallel gain.
> Attributing that fixed cost is the highest-value work available.
>
> | # | Item | Why | Effort |
> |---|---|---|---|
> | **0** | **Attribute the per-query fixed cost** | Q06 costs 376 ms distributed vs 241 ms on one GPU. Account for that 135 ms — planning, fragment delivery, exchange setup, result assembly. Everything else is guesswork until this is known. | days |
> | **1** | **#32 — harness must assert an empty blacklist** | M6 proved `Alive=true` is not enough. Without this gate the same class of defect silently returns. | hours |
> | **2** | **The exchange fixed cost** | Q01 (no shuffle) 3.5×; shuffling queries 1.4–1.6×. That delta is the shuffle, and at these payload sizes it is setup, not bytes — the transport does 628 GB/s. | days |
> | **3** | **Re-take pre-2026-08-09 Engine A numbers** | They may be half-capacity runs (M6). | hours |
> | **4** | **A/B the `cpu_affinity` change** | It landed in the same binary as the M6 fix, so **no latency delta measured on 2026-08-09 can be attributed to either change alone.** Unpick it. | hours |
> | **5** | **M4 — sweep `uring_n_reactors` / `num_threads`** | Still unswept; `uring_n_reactors{1}` on a box reading ~33 GB/CN at SF500. | hours |
>
> **Measured and settled — do not re-investigate:** nixl/UCX healthy at 628 GB/s (#34) · GDS
> unreachable (M5) · fragment distribution even (M6) · row-group stats pruning useless on this data
> (564/564 lineitem row groups survive q14's predicate) · NUMA L1/L2 active.

---

## 🎯 What to focus on now (audit 2026-08-08, live 4× GB200 + SF100) — SUPERSEDED

> **REVISED after the 2026-08-08 live session.** The list below was written before M4 (the scan
> datasource, 5–21×), M1.5 firing for real, and #32 (the harness masking it). New order:
>
> | # | Item | Why | Effort |
> |---|---|---|---|
> | **0** | **M4 — sweep `uring_n_reactors` / `num_threads`** | Decides whether the 5–21× datasource win is a real fix or three undersized defaults. Cheapest decisive experiment here. | hours |
> | **1** | **M1.5 — pre-establish nixl sessions** | **No longer theoretical — it fired.** A cold cluster's first cross-CN query fails at the FE's 60 s timeout. Users hit this on query #1. | A: hours · B: days |
> | **2** | **#32 — harness gates** | Until `--cold` and a row-count gate exist, every future number can hide the same class of bug this one did. | hours |
> | **3** | **M1.4 — the two timeouts** | q08's 60758 ms ≈ `prpc_client.rs:25`; watchdog unset ⇒ blocking `future.get()`. | hours |
> | **4** | **#31.1 — Err on unhandled sink** | 30 minutes, removes a whole silent-failure class. | 30 min |
> | **5** | **M2.1(a)+(c) — arena default + per-lease cap** | SF100 only works because of a hand-passed `STAGING=16GiB`. | days |
>
> **Deprioritised**: **M5 (GDS)** — impossible on this hardware, do not attempt.
> **#34 (nixl/UCX)** — measured healthy at 628 GB/s, stop investigating the transport.

**Superseded — original guidance follows.** Focus on M1 item 4 — the two timeout knobs. It is
the only item where a live failure maps to a specific constant, and it is a hours-not-days change.


| #     | Item                                            | Why now                                                                                                                                                                                                                 | First change                                                                                                                      |
| ----- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **M1.4 — timeouts**                             | **q08 refused at 60758 ms** ≈ the hardcoded 60 s. And the watchdog is *unset* by `cluster8.sh`, so `sirius_engine.cpp:110-113` blocks on `future.get()` forever → the 180 s silent wedges. Two symptoms, two constants. | Make `prpc_client.rs:25` (`Duration::from_secs(60)`) env-configurable; set `SIRIUS_QUERY_WATCHDOG_SECS` in `cluster8.sh`.         |
| **2** | **M1.5 — pre-establish nixl sessions**          | The first-contact MD deadlock the ROADMAP says is "near-certain under 4-way bidirectional shuffle". **This box is the only place it reproduces** — it never fired at 2 CNs. Sessions are lazy-only.                     | Add a peer list to `nixl_transport.rs` `start` (`:191-194`); or move MD off the transport thread (`compute_node_service.rs:949`). |
| **3** | **M2.1(a)+(c) — arena default + per-lease cap** | Everything at SF100 depends on a hand-passed `STAGING=16GiB`. Unset it and the CN refuses exchange. `chunked_pack` is already per-8 MiB — only the *leasing* is monolithic (`sirius_ffi.cpp:743`).                      | Derive default from `cudaMemGetInfo`; lease per chunk instead of once per table.                                                  |


**Do NOT start with #24 on this box.** It is the highest-value item long-term, but it is a
multi-week change, needs no multi-GPU hardware, and the doc points at the wrong line (see
above) — a naive start would produce zero measurable improvement.

**Prerequisite before any A-vs-B number is quotable:** `bench.sh:73-75` scores a run `pass`
on exit-code + non-empty output only, and `analyze.py` reads **only** `status` and `ms` —
it never compares A's row count against B's. A query returning 1 row instead of 100,000 is
recorded as a fast **win**. Add a row-count mismatch gate before publishing anything.

---



## Verification protocol (any of the above)

The suite ladder and live-gate protocol are in the `tpch-bench` skill; in short:
translator → cn-test-no-engine → cn-test (incl. the wire-type parity gate) → C++ suite if
`src/**` changed → GPU harness → live: the affected queries solo vs the DuckDB oracle →
a full sweep, and for anything touching exchange/arena, an endurance shape (2-3 sweeps,
zero restarts). Every fix lands with the regression test that would have caught it.