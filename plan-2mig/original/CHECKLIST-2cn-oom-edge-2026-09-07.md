# Checklist: TPC-H SF1000 on Sirius-as-StarRocks-CN, 2-CN OOM edge (q05/q08/q09/q17/q18/q21)

Box: 2x RTX PRO 6000 (95.6 GiB each), 499 GB RAM, `/opt/dlami/nvme` instance store (wiped on stop/start), no disk tier configured.
Everything below was measured on this box; the GB200 is the eventual target. Last updated 2026-09-07 (UTC). Owner of the runs: aocsa.

Where things are:
- Trees: `/home/ubuntu/sirius-wt/demo` = `all22/integration` 95bec853 · `/home/ubuntu/sirius-wt/perf` = `perf/exchange-04-07-06` 222b646a ·
  `/home/ubuntu/sirius-wt/mcn` = `perf/multi-cn-ingress-packing-transfer` 2e0cbf51 · `/home/ubuntu/sirius-wt/pin` = `feat/pin-table-cn` d24f02c4 (all aocsa/sirius).
- Plans (00-17): `experimental/starrocks/docs/performance/` on `codex/starrocks-performance-plans`, on the mcn branch, and on sirius-db/sirius `docs/sf1000-q08-memory-staging`.
- Our analysis docs: `/home/ubuntu/sirius-wt/all22/ANALYSIS-oom-edge-2cn-staging.md`, `all22/REVIEW-sf1000-q08-memory-staging-analysis.md`,
  `arms/R-status-20260906.md`, `arms/R-controls-20260906.md`, `arms/SA-vs-MCN.md`, `arms/MCN-compare.md`, experiment log `all22/LOG.md` (72 rows).
- Harness: `/home/ubuntu/sirius-wt/harness` (capture-arm.sh, compare.py, cnlog_extract.py, results_table.py); drivers and tools in `/home/ubuntu/sirius-wt/all22`
  (gpu-lock.sh, evidence.sh, pin_compare.py, arms-*.sh, rerun-all.sh, rerun2-2cn.sh, controls-20260906.sh, arms-SA.sh).

## 1. Established facts (do not re-measure)

- [x] Root cause of the 2-CN edge: the FE plans with no statistics (every node cardinality 1), hash-partitions the whole lineitem for the first join of
      q05/q08/q09, and the CN runs the shuffle as a barrier (park all sender output, dispatch the join after every EOS). Per-CN working set 84/96/120 GB
      vs an 81.6-90.2 GB pool. Exchange volume: q08 101 GB (95% lineitem), q05 88 GB, q09 120 GB. (ANALYSIS §1, §2.4)
- [x] q08 failure anatomy (R-MCN16-2cn): both CNs die 2.3-2.5 s into the join fragment; 99.96% of the pool held outside any task reservation; the failing
      request is the HOST->GPU materialization of one 987 MB lineitem frame (246,814,280 B = 30,851,785 rows x 8); downgrade freed 0 for eligibility
      reasons (subscribed/locked batches), host pool had 76-96 GiB free, disk absent but irrelevant. (ANALYSIS §2.1, REVIEW §3.1)
- [x] The staging arena is a bystander: never exhausted in any compared arm (peak live 4.49 of 8 GiB). All failure classes are GPU-pool failures:
      integration "failed to stage" = unreserved deep copy into a full pool; path-04 = sender scan-gate OOM; MCN = receiver upgrade flood. (ANALYSIS §3.1, §5)
- [x] Pool size is not the lever on the MCN tree: mode OFF at 76/16 = 16/22 with the same six failures as 84/8; the 5 extra passes of mode ON come from
      host-first owned ingress. (R-controls-20260906.md)
- [x] Everything fits one GPU without an exchange: integration ONE CN 84/8 = 22/22 (q05 21.5 s, q08 13.6 s, q09 15.8 s); standalone (DuckDB CLI,
      86 GiB) = 22/22, 178 s all-22 warm, streams lineitem as a dynamic-filtered probe; 1-CN fuses the lineitem leaf into the join fragment (plan 11).
- [x] The 2.24-2.25 GB frames that hit mcn's per-frame cap (arena/4 - 8 MiB) are the engine default batch = pool/40 (sirius_config.cpp:589); bounding
      batch bytes via `operator_params` (CN accepts `--sirius-config`) is a config-only way to restore 84/8 in optimized mode.
- [x] The instance moved to another physical host after the 2026-09-06 stop/start (GPU UUIDs changed): all arms 11-16% slower; compare same-day arms only.
- [x] Optimized MCN mode at 2 CNs produces silent wrong answers at ~2% of runs (q08 19%): see §3; any MCN pass count must be read with compare.py on every run.
- [x] Path-04 credits never engaged at 84/8 on this box (0 credit waits in R-P04); its gain (16->20) comes from the spill hook (P04s-off-2cn 20/22 with credits off).

## 2. What was tried, with outcome (chronological; details in all22/LOG.md)

| When | Tried | Outcome |
|---|---|---|
| 09-05 | Integration waves W1-W5 (kit + RIGHT_SEMI + async dispatch + decimal rounding + A1 receive-into-pool + B2 multicast + B3 distinct + C5/C7) | 1 CN 21-22/22; 2 CNs 14->16/22; six OOM queries remain at 2 CNs |
| 09-05 | A1 receive-into-pool (stage on arrival, ticketed) | Removed "arena exhausted" at 32 GiB staging; failures moved to pool OOM |
| 09-05 | A2 batch-major drain | Not implemented (export_packed already pulls per batch); dropped |
| 09-05 | Fix 3 FILES() cardinality knob (Gate A) | Knob ON changes plans (region/nation broadcast) but q03/q12 lineitem stays SHUFFLE; knob-ON plans do not fit at 1 CN (F3-1cn 15/22) |
| 09-05 | Forced CTE reuse (cbo_cte_reuse_rate 0 / -1) | Irrelevant at 2 CNs (I2b/I2 1.00x); multicast projection bugs fixed on the way |
| 09-05 | Memory split 60/32 vs 84/8 (I2s) | 0.99x, same failures: the split does not explain the edge |
| 09-05 | Pin branch feat/pin-table-cn, pinned and unpinned (PA/PB/PVI) | 1.9x on fitting queries with pins; same six OOM at 2 CNs; predates rounding fixes |
| 09-05 | Path 04 slice 1 (receive credits, pool/4) | 16/22, q17 passes for the first time; slice-1 deadlock (14 senders waiting 120 s) |
| 09-05 | Path 04 slice 2 (spill staged frames via downgrade TIER 0, credit bound to copying thread) | **20/22 at 2 CNs**, no speed cost vs integration; 1 CN 92 GiB 22/22 |
| 09-05 | MCN branch mode OFF 84/8 | 16/22 = integration |
| 09-05 | MCN mode ON window 1, 84/8 | 18/22; q07 q09 q16 q22 die with EXPORT_CAPACITY_EXCEEDED (2.24 GB frame > 2 GiB - 8 MiB) |
| 09-05 | MCN mode ON window 2, 76/16 | **21/22** (q05 r2 OOM after 100 retries); 4-10% slower on fitting queries |
| 09-05 | Standalone Sirius 1 GPU vs MCN-on16 | 22/22; 146 s vs 205 s over 21 common; scan-bound queries identical (NVMe floor) |
| 09-06 | Post-wipe rerun of the four arms (R-*) | Pass sets unchanged; MCN lost q08 instead of q05; all arms 11-16% slower (host change) |
| 09-06 | Control: MCN tree mode OFF 76/16 | 16/22, same six; q18/q21 stage-copy pool OOM even with 16 GiB arena |
| 09-06 | Control: integration ONE CN 84/8 | 22/22 |
| 09-07 | Second same-day 2-CN round (R2-*) for regression/stability/backpressure | in progress; R2-I2 16/22 with flips (q17 pass, q07 r2 fail) |
| 09-05 | Branch review of MCN (wf 119 agents) + design review of paths 04/07/06 | high findings listed in §3 |

## 3. perf/multi-cn-ingress-packing-transfer: what is on it and what is NOT

On the branch (Codex commit 7412ef4d + benchmarks/docs 2e0cbf51): opt-in `SIRIUS_EXCHANGE_OPTIMIZED=1`; owned ingress with pinned-HOST reservations
(every inbound frame lands in HOST, D2H then H2D on reload); independent pack worker (`[exchange_pack] independent=1`); fair transport with
`SIRIUS_CN_NIXL_TRANSFER_WINDOW` 1..8 and `_PEERS`; per-frame receive cap arena/4 (sender-side pack check, `EXPORT_CAPACITY_EXCEEDED`); replay ledgers;
quarantine on stalled WRITEs; plans 00-17 docs; GB10 SF500 scripts.

NOT on it (our work lives elsewhere): path-04 receive credits + staged-frame spill (`perf/exchange-04-07-06`), the integration-side A1 stage-on-arrival is
its base. Nothing from this session has been landed on the MCN branch yet.

Known defects on the branch (review wf_2467e875, confirmed high), none fixed yet:
- [ ] **SILENT WRONG ANSWERS in optimized mode** (found 2026-09-07, all22/INCIDENT-q08-wrong-answer-mcn.md): 7 wrong results in 385 exit-0 optimized-mode runs on
      2e0cbf51 (q08 4 of 21 completed = 19%, q09 3 of 21; shapes: one year low, uniform 0.998x, garbage 1e11), every harness case 1-2 queries after a
      restart-after-OOM; transport ledgers, frame multisets and declared cardinalities all clean, so the defect is at value level (leading candidate: the
      year-keyed HASH_GROUP_BY -> PARTITION[FULL] -> MERGE_GROUP_BY on the CN holding BRAZIL; alternatives: payload corruption, un-row-counted local relay).
      Mode-off path 0/98 wrong but never completed q08/q09 at 2 CNs. Blocks release of SIRIUS_EXCHANGE_OPTIMIZED=1; instrumentation plan in the incident note §5.
- [ ] Quarantine never returns receiver credits: a stalled WRITE poisons the CN pair until restart.
- [ ] Replay ledgers never freed; admission rejects at 262,144 identities (shorter session life once frames get smaller).
- [ ] Backpressure becomes a hard failure after `SIRIUS_CN_RPC_TIMEOUT_SECS` (60 s default).
- [ ] Every inbound byte takes a synchronous D2H + later H2D (structural cost, 4-10% on fitting queries; 4,912 HOST-tier ingress lines per CN on q08).
- [ ] Default (mode OFF) path is not byte-identical to integration: legacy lease/publish RPCs lost their reconnect retry; bounded transport queue.
- [ ] Per-frame cap arena/4 - 8 MiB vs pool/40 batches forces the 76/16 split (or bounded batches).
- [ ] Intermittent wrong q08 answers in optimized mode were noted by one investigator (ANALYSIS §7); unverified, must be checked with compare.py on every run.

## 4. To explore next (ordered; the top 3 are the work packages in PROMPT-top3-2cn-oom-edge.md)

- [ ] **0. Telemetry (prerequisite, days)**: promote the downgrade outcome to INFO with candidates / skipped_subscribed / lock_failed / host_reserve_failed
      per tier (`mcn/src/downgrade/downgrade_executor.cpp:366-400`), add `[gpu_pool]/[host_pool]` snapshots inside the `:367` warning, log per-query arena
      peak live. Arm `M-dbg-2cn` (76/16, opt on, q08 x3) to attribute the 81.57 GB.
- [ ] **1. Plan 10, bounded frames and batches (top 3)**: config-only first: `--sirius-config` hook in `benchmarks/cluster8.sh` (line 86 passes
      `--gpu-memory-limit`; the flag is mutually exclusive with a config file), 1 GiB `operator_params` (scan_task_batch_size, hash_partition_bytes,
      concat_batch_bytes; `pin/.../benchmarks/pinned/gen-config.sh` already writes them), arm `MCN-1g-84-2cn` (84/8, opt on, all 22; pass criterion:
      0 EXPORT_CAPACITY_EXCEEDED, >= 21/22) plus a 76/16 twin with the same operator_params. Then code: sender splits oversized batches into self-contained
      frames (`mcn/src/sirius_ffi.cpp:1104-1183` pack/cap; `exchange_protocol.rs:162-165,284-289` RX limits; `nixl_transport/pipeline.rs:602-668` TX).
- [ ] **2. Plan shape: filter/broadcast before shuffling lineitem (top 3)**: diagnostic first with hinted SQL (`JOIN [BROADCAST]` on filtered dimensions,
      `StarRocks.g4:2536-2567`, `JoinHelper.java:344-348`), arms `H-I2-2cn`/`H-MCN16-2cn` gated on EXPLAIN (criterion: q08 passes in demo at 84/8 with
      tx_bytes < 10 GB and MATCH). Durable: apply `PREDICATE_UNKNOWN_FILTER_COEFFICIENT` (0.25, `StatisticsEstimateCoefficient.java:31`, unused) in
      `BinaryPredicateStatisticCalculator.java:86-120` for FILES() columns AND raise `broadcast_row_limit` (0.25 x 200M = 50M > 15M today); real FILES row
      counts (fix-3 knob) as the estimate source. Optional engine side: admit stream-fed small builds as dynamic-filter evidence (`build_filter_evidence.cpp:23-48`).
- [ ] **3. Plan 12 + 05, stream the shuffle (top 3, structural)**: dispatch the receiver once local senders are parked and the first remote frame is
      readable, push frames during Run, close on EOS (`local_exchange.rs:573-611`, `engine.rs:640-745`, `compute_node_service.rs:1210-1243`; export during
      Run instead of `export_provider` after completion). Keep join-build completion semantics. Per-CN working set becomes scan + in-flight window.
- [ ] 4. Plan 01/02 hardening (release requirement): ledger retirement by acknowledged sequence/epoch, credit return on quarantine, nonblocking peer setup.
- [ ] 5. Sink-first scheduling scoped to exchange-fed fragments (`creator/config.hpp:106` priority_order::sink has zero callers; YAML key throws at
      `sirius_config.cpp:154-158`).
- [ ] 6. Exportable-pool canary (unified accounting): fabric arena needs nvidia-imex (`exchange_staging_arena.cpp:116-118`); test `cuda_async_memory_resource`
      with posix file-descriptor handles + UCX mempool IPC on this single host. Buys at most 8-16 GiB per CN; decide before committing a quarter.
- [ ] 7. One CN driving both GPUs (`topology.num_gpus: 2`, no `--gpu-device`): removes the exchange; bounded experiment, not a fix.
- [ ] 8. Disk tier on `/opt/dlami/nvme` for q09's HOST peak: cannot flip q08 (eligibility, not capacity), may help q09/q21 tails.
- [ ] 9. Land on the MCN branch: bounded frames (1), telemetry (0), plan-01 fixes (4); keep `SIRIUS_EXCHANGE_OPTIMIZED` opt-in until the default path is
      byte-identical to integration again.

## 5. Protocol reminders

Two CNs one per GPU, SF1000 kit at `demo/experimental/starrocks/benchmarks/tpch/queries` (q11 SF-scaled), FE defaults (`SET GLOBAL cbo_cte_reuse_rate = 1.15`),
1 cold + 2 warm, 300 s timeout, watchdog 240, `RESTART_ON_FAIL=1`, oracle `/home/ubuntu/starrocks-tools/oracle/tpch_sf1000` at rel 1e-6 on every run,
`ASYNC=1` recorded in every config.txt, everything GPU-side under `all22/gpu-lock.sh`, same-day comparisons only, never exclude failed queries from coverage.
