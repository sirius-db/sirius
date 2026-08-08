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



## M4. The scan datasource default is a 5–21× performance regression 🔴 **NEW — biggest single win found**

**What**: `scan_manager.use_sirius_datasource` defaults to **`true`**
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

## 🎯 What to focus on now (audit 2026-08-08, live 4× GB200 + SF100)

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