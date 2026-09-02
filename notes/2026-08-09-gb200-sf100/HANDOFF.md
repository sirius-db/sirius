# Handoff — CN fragment distribution + NUMA awareness

> # ✅ RESOLVED 2026-08-09 05:15 UTC
>
> **Everything in the "next steps" section below is DONE.** Read
> `benchmark-results/cn-distribution-and-numa.md`
> and `OPEN-ISSUES.md` **M6** instead. This document is kept for the retraction table and the
> traps, which are still accurate.
>
> **Root cause.** Not imbalance — **exclusion**. Two of four CNs were auto-blacklisted ~2 s after
> every cluster start, before any query ran, and never came back, while all four reported
> `Alive = true`. q14 ran 48.9 / 0 / 0 / 51.1 %.
>
> Two independent defects were both required:
> 1. **Trigger:** the CN built the GPU engine *before* binding any socket — ~6.9 s of refused
>    connections on a GB200 (RMM pool reservation), while the FE is up in ~4 s and immediately
>    heartbeats every node it remembers from persisted metadata.
> 2. **Why permanent:** the CN advertised `http_port` and never bound it, so
>    `checkAccessibleForAllPorts` could never succeed and blacklist eviction never ran.
>
> **Both fixed and verified live:**
> * `start_http_server` — bind the advertised port. FE now auto-evicts in ~1 s.
> * **Listeners before engine init + `EngineReadiness` gate** — bind first, build the engine second,
>   open the gate only once BRPC is up. While warming the heartbeat is *answered* (not blacklistable)
>   but *not OK* (not schedulable). Verified 05:10: **zero blacklist entries ever added**.
>
> **Result:** q14 across 4 CNs = **25.5 / 24.6 / 24.5 / 25.4 %** (1.04× spread), no intervention.
> 127 CN tests pass.
>
> **Two caveats that matter more than the fix:**
> * It did **not** make queries faster at SF100 — 2 CNs and 4 CNs are indistinguishable there.
>   At SF500 it does help (Q01 3.5×, others 1.4–1.6×).
> * **Confound:** a concurrent `cpu_affinity` change landed in the same binary (04:12 vs my 04:19
>   build), so **no latency delta measured on 2026-08-09 is attributable to either change alone.**
>   The *distribution* results are unaffected.
>
> **The question this leaves open** is now the important one: at SF100 four GB200s still lose to one
> (Q06 376 ms vs 241 ms) with distribution provably even. That gap is a per-query fixed cost, and
> attributing it is the top item in `OPEN-ISSUES.md`.

**Session ended:** 2026-08-09 ~02:45 UTC · **Branch:** `demo-multi-cn` (default branch is `dev`)
**Goal (still active):** distribute q14's work evenly across N compute nodes, and establish whether
the engine is NUMA-aware. Test only on q14, SF100, from `/raid`.

---

## 1. The one thing to read first

**The 2-of-4 imbalance is almost certainly an ARTIFACT, and chasing it uncovered a real defect that
is worth more than the original question.**

A Sirius CN that ever fails one RPC is **blacklisted by the FE for the rest of the FE's process
lifetime** — permanently, silently, and with no way back short of a manual SQL statement.

The mechanism, read from source (all cited in
`/home/prestouser/aocsa/benchmark-results/investigate-phase-results.json`):

1. A failed query adds the CN to the blacklist — `DefaultCoordinator.java:772` on
   `THRIFT_RPC_ERROR`, plus `FragmentInstanceExecState.java:410`, `ExecutionDAG.java:643`,
   `ResultReceiver.java:138/:147` — all funnelling into `SimpleScheduler.addToBlocklist` →
   `HostBlacklist.add` (`HostBlacklist.java:72-90`).
2. Removal (`HostBlacklist.java:208-214`) requires
   `NetUtils.checkAccessibleForAllPorts(host, [bePort, brpcPort, httpPort])` — a raw TCP connect to
   each of the three.
3. **The Sirius CN advertises `http_port` in its heartbeat (`src/lib.rs:462`) but never binds it.**
   It binds exactly three sockets: heartbeat thrift (`src/lib.rs:912`), BackendService thrift
   (`src/lib.rs:939`), brpc (`src/main.rs:532`). There is no HTTP server in `Cargo.toml`.
4. So the connect to `httpPort` always refuses → `checkAccessibleForAllPorts` always returns false
   → `HostBlacklist.remove()` is **never** called.

`Config.black_host_penalty_min_ms` (500 ms) and friends are irrelevant — the port gate sits upstream
of all of them. Only `DELETE COMPUTE NODE BLACKLIST <id>;` or restarting the FE clears it.

**Why this explains the original observation.** The known cold-start defect (M1.5, first query after
cluster start hangs to the 60 s FE timeout) blacklists whichever CNs it touched. They never return.
Every subsequent query in that FE lifetime runs on the survivors — which is exactly
`cn0=2 operators / cn2=2 operators` versus `cn1=16 / cn3=13`, and exactly why cn0/cn2 had **zero**
query and channel records.

**Consequence for past results:** any Engine A number taken after a failed query, without an FE
restart, was measured on a silently reduced cluster. That plausibly includes some of the CN-scaling
data in `benchmark-results/`.

### Two independent things that produce the same shape

- **Heartbeat death** (`ComputeNode.java:643-655`): 3 retries × 5 s timeout → `isAlive=false`. The
  CN's thrift server is a *single-threaded, strictly sequential* accept loop
  (`src/lib.rs:968-1013`) that blocks for the whole life of each connection, so a slow connection can
  starve heartbeats under load. **UNVERIFIED** that this happened. Distinguish it from the blacklist
  by the `Alive` column — and note it self-heals, unlike the blacklist.
- **Backup-worker re-homing** (`NormalBackendSelector.java:95-117` +
  `DefaultSharedDataWorkerProvider.selectBackupWorker:251-260`): a blacklisted node's ranges get
  silently re-homed onto a live buddy. So read **per-CN byte counts**, not just host counts — this
  path yields 4 "hosts used" while 2 do all the work.

---

## 2. Retractions — things I asserted this session that are WRONG

Do not carry these forward. Each was refuted by source reading, not opinion.

| Claimed | Actually |
|---|---|
| `FILES()` → `FileTableScanNode`, one scan range per file, so 6 files caps it at 6 | **Wrong node.** `FILES()` SELECT builds a `FileScanNode` (`PlanFragmentBuilder.java:4254`). It emits real sub-file byte ranges via `setStart_offset`/`setSize` (`FileScanNode.java:632-645`). Range count is an instance budget, not a file count. |
| `parallelInstanceNum` is hardcoded to 1 | The 6-arg ctor sets 1, but `PlanFragmentBuilder.java:4270-4271` immediately overrides it via the 7-arg `setLoadInfo` with `getSinkDegreeOfParallelism()`. Real default ≈ **9–18 instances per node**, so lineitem baseline is ~36–72 ranges, not 4. |
| `prefer_compute_node=false` blocks fan-out; `LOCALITY` pins downstream to the child's hosts | **H2 REFUTED.** FE runs `run_mode = shared_data`, so `DefaultSharedDataWorkerProvider.isPreferComputeNode()` returns a **literal `true`** (`:225-227`), ignoring the session variable. `RemoteFragmentAssignmentStrategy.java:156` always takes the fan-out branch; the `:171` LOCALITY branch is **unreachable** here. `use_compute_nodes` and `computation_fragment_scheduling_policy` are dead knobs in this deployment. |
| NUMA layer 1 (process pinning) is ACTIVE because `cluster4-numa.sh` uses `numactl` | The script is correct — **and never invoked.** The bench SOP launches `benchmarks/cluster8.sh` (no `numactl` anywhere) and `run-comparison.sh` hardcodes `pixi run cluster2`. **Every SF100 number to date ran unpinned.** |
| NUMA layer 2 is DORMANT because `engine_settings.rs` emits no `numa_id` | **Backwards.** `use_host_per_numa()` is called unconditionally (`sirius_config.cpp:322`) and `numa_id` is derived from `nvmlDeviceGetMemoryAffinity` (`reservation_manager_configurator.cpp:235`). Live proof: `.cn0` log shows `GPU 0 (numa=0)`, `GPU 2 (numa=1)`, `[host_pool] HOST:0`. **Emitting `numa_id` from `engine_settings.rs` would be actively harmful** — the only YAML path that accepts it is `sirius.space:`, and supplying any `space:` entry sets `using_configurator=false` (`sirius_config.cpp:479`), silently discarding `memory.gpu`. |
| The `host space count (1) != NUMA node count (34)` warning proves dormancy | **False alarm.** It compares against a raw `/sys/devices/system/node` count (34), of which only 0 and 1 have CPUs. **1 is the correct count for a 1-GPU CN.** The warning itself is the bug. |

---

## 3. NUMA — the actual answer

**Not one bit. Three layers, three different answers.**

| Layer | Status | Detail |
|---|---|---|
| **L1 process placement** | 🔴 **DORMANT** | `cluster4-numa.sh` is correct and tracked (commit `c3bfe660`) but nothing runs it. `cluster8.sh` / `cluster2` have no `numactl`. **This is the highest-value fix and it is a one-line SOP change.** |
| **L2 host-memory topology** | 🟢 **MOSTLY ACTIVE** | `use_host_per_numa` → per-GPU `numa_id` from NVML → `numa_alloc_onnode` (`numa_region_pinned_host_allocator.cpp:60`). Also active: `numa_small_pinned_mr` (but only for cuDF buffers **< 8 KB**), `topology_index`, per-GPU downgrade `preferred_numa_node`. Dormant: prefetch-cache per-NUMA arenas (gated off by `enable_prefetch_cache{false}`). |
| **L3 I/O + threads** | 🟡 **MIXED** | ACTIVE: `gpu_pipeline_executor` threads pinned to the GPU's `local_cpulist` (verified on box: `0008/0009:01:00.0 → 0-71`, `0010/0016:01:00.0 → 72-143`). DORMANT: every `thread_pool.cpu_affinity_list` defaults empty and the derived YAML never sets it; `uring_n_reactors{1}`. **ABSENT:** `uring_reactor.cpp` has `#include <numa.h>` and *zero* `numa_*` calls; `datasource_factory.cpp:67` picks `host_spaces.front()` with no NUMA selection. |

**Recommended topology (do not "improve" on this):** **one host space per CN, `numa_id` = the NUMA
node of that CN's single GPU** — which is what already happens. Node 0 for CN0/CN1, node 1 for
CN2/CN3.

- **Not 34.** `numa_alloc_onnode(bytes, 2)` would pin host staging *into GPU0's HBM*. The other 28
  nodes report `size: 0 MB` and would fail outright.
- **Not 2 per CN.** `set_per_host_capacity` is **per space**, so 2 spaces = 2 × 160 GiB per CN =
  1,280 GiB across 4 CNs against 956.82 GiB of real LPDDR, with swap = 0 → OOM-kill. And
  `datasource_factory.cpp:67` takes `front()` (= numa 0) anyway, so CN2/CN3 would stage everything
  cross-socket.

---

## 4. State of the box and the run

**Nightly CI owns the GPUs until ~03:47.** `0 2 * * *` → `/opt/sirius-ci/scripts/cron_benchmark.sh`,
sweeping SF1…SF1000. Six nights of history: **1h43m–1h49m**, extremely consistent. Started
02:00:01 today; was on the SF1000 4-iteration leg at 02:36.

- Watch armed: monitor `baw58p46x` fires when driver pid **2631662** exits.
- ⚠️ **Do not poll `nvidia-smi --query-compute-apps` to decide the GPUs are free** — I made that
  mistake and got a false "GPUs FREE" from a gap *between* runs. Watch the **driver process**.
- ⚠️ **Never** set `ALLOW_SHARED_GPUS=1` to get past the preflight. The RMM pool is reserved in full
  at startup, so sharing is an allocation failure or a zero-headroom cluster, not a slowdown.
- **`08-08` nightly has a start line but no finish line** — the only incomplete night in the log. I
  was running 4-CN clusters that day. Undiagnosed; worth a look.
- A peer interactive session (`aocsa-55`) shares this box. It emptied `.cn*/telemetry` at 01:44:59.

### Workflow

`wf_104e2feb-229` (task `wkqz30hih`), script at:
```
/home/prestouser/.claude/projects/-home-prestouser-aocsa-sirius-experimental-starrocks/838d0459-8adc-4090-955f-0b0244a401af/workflows/scripts/cn-distribution-and-numa-wf_7ee97c56-f40.js
```
- Investigate: **all 3 agents complete** (FE knobs, NUMA audit, telemetry analyzer) — full results
  saved to `benchmark-results/investigate-phase-results.json` (61 KB). The workflow was stopped
  immediately after, before the Measure phase could start during the nightly.
- `experimental/starrocks/scripts/cn-distribution.py` (30 KB) is self-tested against a synthetic
  telemetry tree but **has never seen real data**. Known structural limit: `channel` records carry
  only engine-scoped refs, so per-query channel counts are always 0 — use the **per-CN** channel
  totals for the participation signal (a CN with 0 query *and* 0 channel records never joined the
  query at all).
- Measure / Implement / Report: **not started.**
- The script on disk has since been edited to add a GPU-exclusivity wait gate. The *running*
  instance predates that edit — **stop it before it reaches the Measure phase.**
- Resume with `Workflow({scriptPath, resumeFromRunId: "wf_104e2feb-229"})`; completed agents return
  cached. **But note the FACTS block in that script is now stale** (see §2) — it still asserts
  `parallelInstanceNum=1` and the `prefer_compute_node` hypothesis. Fix FACTS before resuming, or
  the measurement agents inherit refuted premises.

---

## 5. What to do next, in order

1. **Settle H1 at zero box cost.** On a freshly restarted FE with 4 CNs alive:
   ```sql
   SHOW COMPUTE NODES;              -- Alive column, LastStartTime
   SHOW COMPUTE NODE BLACKLIST;     -- StarRocks.g4:2089; HostBlacklist.getShowData:127-145
   ```
   Run q14 once. Re-check both. If a CN appears in the blacklist → **H1 confirmed, the 2-of-4 is an
   artifact, and the original question dissolves.**

2. **Fix the permanent blacklist.** Either bind the advertised `http_port` in the CN (smallest
   change that restores `checkAccessibleForAllPorts`), or stop advertising a port that isn't bound.
   This is a genuine Sirius defect independent of the benchmark and should get its own issue.

3. **Make the SOP use the NUMA-pinned launcher.** Point `.claude/skills/tpch-bench/SKILL.md:83`,
   `:152` and `run-comparison.sh:50/:52` at `configs/gb200-4gpu/cluster4-numa.sh`. Then re-baseline —
   every prior number was unpinned. Verify with
   `grep Mems_allowed_list /proc/<cn-pid>/status` → must read `0` or `1`, never `0-2,10,18,26`.

4. **Only then** run the q14 distribution matrix. Candidates the FE agent produced, ranked:
   `baseline-with-blacklist-probe` · `skip-blacklist` (`SET skip_black_list = true` — a
   *discriminator*, not a fix: `LoadScanNode.java:119` consults the blacklist directly and bypasses
   the WorkerProvider) · `pipeline-dop-1` · `pipeline-dop-8` · `prefer-compute-node` and
   `choose-instances-auto` (both **expected to be no-ops** — keep them only as negative controls).

5. **Fix the false-alarm warning** at `src/sirius_context.cpp:365` — compare against CPU-bearing
   NUMA nodes, not the raw `/sys` count. It has already misled one investigation.

### Deliberately deprioritized

- **Row-group split strategy** (the "metadata scan + prune + global rebalance" question). Answer:
  largely already built — `scan_paths.rs` implements row-group ownership by start offset, refuses
  overlaps, coalesces adjacents. Measured: lineitem SF100 has **564 row groups at ~30 MB**, and
  **564/564 survive q14's `l_shipdate` min/max pruning** — the data is not clustered on shipdate, so
  pruning removes nothing and a rebalance would redistribute identical work. It would pay only where
  pruning is large (data clustered on the filter column), which is not this dataset.
- **Cold-start `warmup.rs`** — compiles, still **UNVERIFIED**, zero trials. Note it is likely the
  *upstream cause* of the blacklist, so item 2 may matter more.

---

## 6. Measurement protocol (learned the hard way)

- Data: `/raid/prestouser/kkristensen/tpch_parquet_sf100` — local NVMe. Never NFS: `use_odirect{true}`
  on NFS caps at 0.78 GB/s vs 25 GB/s on NVMe (see `benchmark-results/scan-defaults-sweep.md`).
- Clean telemetry before **every** measured run: `experimental/starrocks/scripts/clean-telemetry.sh`
  (sandbox-verified: `--dry-run` deletes nothing; the real run clears contents and keeps the mount
  dirs; refuses while CNs are live).
- **Telemetry is buffered in memory and flushed only at engine shutdown.** A CN killed mid-query
  flushes nothing — which reads downstream as "this CN did no work." That is how the original
  capture got contaminated.
- Warm-up run first, discarded, because of the cold-start hang — **but** warm-up and measured run
  share one CN process lifetime and land in the same telemetry uuid. Report distribution *shape*,
  not absolute counts, unless you can attribute by query id.
- The **FE query profile is authoritative** for placement (`SHOW PROFILELIST` →
  `ANALYZE PROFILE FOR '<id>'`); telemetry corroborates. If they disagree, report the disagreement.
- q14 at SF100 returns a single `promo_revenue` ≈ 16.38.

---

## 7. Files touched this session

**New (untracked):**
- `experimental/starrocks/scripts/clean-telemetry.sh` — telemetry cleaner, sandbox-tested
- `experimental/starrocks/scripts/cn-distribution.py` — per-CN telemetry distribution analyzer (30 KB,
  **unverified against real data**)
- `/home/prestouser/aocsa/benchmark-results/investigate-phase-results.json` — full findings, 61 KB

**Staged from earlier sessions, uncommitted, still unverified:** `nixl_transport/warmup.rs`,
`compute_node_service.rs`, `bench.sh`, `analyze.py`, `run-comparison.sh`, `OPEN-ISSUES.md`.

**`OPEN-ISSUES.md` needs updating** with §1 (the permanent blacklist), §2 (the retractions), and §3
(NUMA is mostly active at L2, absent at L1 in practice).

**Untracked debris to clean up:** `experimental/starrocks/scripts/__pycache__/`,
`experimental/starrocks/log/`. Also `.cn0/` and `.cn3/derived-sirius-config.yaml` are **staged for
only 2 of the 4 CNs** — probably unintentional.

⚠️ **A `PreToolUse` cleanliness hook is denying some Bash calls**, citing `OPEN-ISSUES.md` diary
framing and those half-staged configs, and demanding `/pre-commit-cleanup`. An agent hit this ~5
times and correctly declined to run it, since that would rewrite another session's staged work.
**Whoever owns that staged commit needs to resolve it** — it will keep blocking work otherwise.
