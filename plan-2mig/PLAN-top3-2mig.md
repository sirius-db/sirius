# PLAN (adapted): three general work packages for the N-CN exchange working-set edge, on 1 GPU split into 2 MIG instances

Adapted 2026-09-09 from `original/PLAN-top3-general.md` (2026-09-08, written for the 2x RTX PRO 6000 box at SF1000). The original stays
authoritative for every mechanism, file anchor, test list, principle and the generality rule (§0, §3-§5 "Principles/Files/Tests"). This document
replaces only what the box changes: sizes, bounds, arms, references, gates that quote numbers, host-memory handling and the sequencing for one
engineer on an 8-core machine. Numbers are derived in `BOX-TRANSLATION.md` (BT); layouts L-A = 40/4, L-B = 36/8, L-C = 36/4 GiB (pool/arena per CN),
HOST_MEM 40 GiB (44 ceiling), SF500, `GPU_DEVICES=0,1`. Trees: mcn 2e0cbf51, demo 95bec853, perf 222b646a as in the original.

## 0. Purpose, generality rule, and what the box changes

Root cause and the three-sided attack are unchanged (original §0): FE plans FILES() at cardinality 1, shuffles the largest input of the first join, the CN
runs the shuffle as a barrier, per-CN working set = rows x width. At SF500 that is 1.5e9 rows x 28/32/40 B = 42/48/60 GB against a 38.7-42.9 GB pool
(BT §2), the same edge the original had at SF1000 against 81.6-90.2 GB. Confirmed on this box: S500-I2-2cn-2mig (integration, L-A) fails exactly q05 q08 q09
q17 q18 q21 with GPU out_of_memory, 16/22 MATCH, like R-I2-2cn did at SF1000.

Generality rule unchanged: nothing keys on a table, column, constant, query or role; SF500 q05/q08/q09 at 2 CNs are acceptance instances only.

Three things the box changes that the original never had to handle:
1. **Host memory is now a binding constraint** (BT §4). mcn's HOST-first ingress needs ~65-70 GiB per CN at SF500 against a 44 GiB ceiling. The failure class
   "HOST evacuation unavailable" is expected to bind before "engine operator OOM" in optimized mode. Adaptation: a DISK tier on the nvme via the
   `CN_SIRIUS_CONFIG_DIR` launcher (WP1 P0 deliverable, no code). This is a capacity substitute for the missing 375 GiB of RAM, explicitly not a fix; every
   optimized-mode arm exists in a HOST-44 and a HOST-44+DISK variant until one is shown sufficient.
2. **The default engine batch violates the mcn frame cap at L-A by 8 MiB** (1024 MiB batch vs 1016 MiB cap, BT §3), reproducing the 84/8 structural
   violation, so WP1 P0 is exercised for real here; the bound B is 512 MiB (not 1 GiB).
3. **One engineer, cold builds, one GPU.** Every optimized-mode arm waits on one instrumented mcn rebuild (INCIDENT §6) that costs hours here; the sequencing
   (§6) puts all no-rebuild work and all mode-off reference arms before it, and starts the rebuild as early as the telemetry patch is complete.

## 1. Where these tasks are detailed today

Original §1 table unchanged. Add: BT (this box), `CHECKLIST-2mig.md` (facts established here), `arms/S500-*` and `arms/R5-*` (same-box references),
`/home/ubuntu/starrocks-tools-wt/sf500-2-mig-gpus` (drivers, oracle, evidence, sirius patches).

## 2. Cross-cutting prerequisites

**Telemetry (a)-(f)** and **Correctness (1)-(5)**: unchanged from the original §2 (same sites, same protocol revision 1 -> 2, same rule: land before any
optimized-mode 2-CN arm). Branch `wp0/exchange-telemetry` off mcn 2e0cbf51; one engine + CN rebuild.

**Harness knobs (a)-(f)**: unchanged in content; the box adds:
- (a) `CN_SIRIUS_CONFIG_DIR` + `gen-cn-config.sh` also emit `memory.disk` when `DISK_ROOT`/`DISK_BYTES` are set (the HOST substitute), keep the
  `GPU_DEVICES`/`MIG_DEVICES` placement, and derive `usage_limit_bytes` = GPU_MEM, `capacity_bytes` = HOST_MEM, operator_params = B, max_build = 2B; the CPU
  affinity list is the box's single NUMA node (0-15), taken from the CN's own derived YAML.
- (d) config.txt records parent GPU UUID + both MIG UUIDs, DISK tier (root, bytes) and `tier=DISK` counts from the engine log at arm end.
- The canonical harness stays `/home/ubuntu/sirius-wt/harness`; the mirror lives in `starrocks-tools-wt/sf500-2-mig-gpus/scripts/`.

**Measurement protocol**: unchanged except references and bounds (BT §5): controls are the same-box R5-* arms; regression bound is "no fitting query > 10%
slower than the same-day, same-tree reference on this box"; every pool comparison pins B; HOST layout is one more recorded variable (40 / 44 / 44+DISK)
held fixed within a comparison set.

## 3. WP1 -- Size-bounded exchange frames and engine batches

Goal, principles, files, tests, risks, dependencies: original §3. Phase table with the box's numbers:

| Phase | Steps (delta vs original) | Arms | Exit criterion |
|---|---|---|---|
| P0 config-only bound | launcher mode + gen-cn-config.sh; B = 512 MiB (screen {256, 512, 768}); DISK-tier emission | MCN-b512-LA-2mig (40/4, w2, HOST 44 or 44+DISK), MCN-b512-LB-2mig (36/8), MCN-b512-LC-2mig (36/4); all after the §2 rebuild | 0 EXPORT_CAPACITY_EXCEEDED at L-A (the default batch produces them there today by construction, BT §3); max `[exchange_pack] bytes` <= 1016 MiB; >= the same-box R5-MCN16-2mig pass count (not 21/22, which is a 2-GPU number); oracle MATCH every run; HOST layout identical across the three arms |
| P1 telemetry + protocol rev 2 | unchanged; smoke M-dbg-2mig (L-B, w2, 3 runs of a query with >= 1 HOST reload, >= 1 multi-frame stream, >= 1 relayed batch) | M-dbg-2mig | original P1 criterion verbatim |
| P2 explicit bound threaded | unchanged; CN batch clamp = min(pool/40, frame_bound - 8 MiB - slack) = min(1024, 1016 - slack) MiB at L-A, i.e. the clamp is active at L-A by default here | MCN-derived-LA-2mig; the "512 MiB / w4" arm becomes "256 MiB / w4" (in-flight per peer = min(4, floor(1 GiB / 256 MiB)) = 4 vs 2 at the default) | derived config at L-A: 0 capacity errors; in-flight per peer = 4 in the 256 MiB / w4 arm and 2 at default regardless of window; rest verbatim |
| P3 export cursor | unchanged | MCN-split-LA-2mig (clamp off so 1024 MiB batches exceed the 1016 MiB cap) and -clamped | verbatim (batches > bound through a 4 GiB arena) |
| P4 bounded reload + accounted scratch | unchanged | MCN-reload-LA-2mig | verbatim; baselines re-measured on this box in P1 |
| P5 identity budget, sweeps, decision | arena sweep {40/4, 42/2, 43/1} at w2/w4 with B pinned (45 GiB ceiling); frame sweep {128, 256, 512, 1016} MiB | all 22 + x5 boundary set | **WP1 PASS** = 0 frame-capacity / export-reload / HOST-evacuation failures across all 22 at >= 40 GiB pool incl. repeats; MATCH + checksum every run; no fitting query > 10% slower than R5-MCN16-2mig; arena peak live < 75%; identities bounded. Residual failures only "engine operator OOM at the receiver" (WP3) |

Note on HOST evacuation: on the 2-GPU box that class never bound (HOST was free). Here it can bind for a reason unrelated to WP1 (BT §4); a WP1 arm that fails
only with that class and passes with the DISK twin is attributed "host capacity (box)", not to WP1, and both results are reported.

## 4. WP2 -- Statistics-driven plan shape (FE) and runtime filtering (engine)

Goal, principles, files, tests, risks, dependencies: original §4. Box deltas:
- Byte budget: `broadcast_bytes_limit <= max_build_hash_table_bytes = 2 x pool/40 = 2 GiB` at L-A (1 GiB with B pinned) (BT §3).
- Fan-out: `getAliveExecutionNodesNumber` = 2 here as on the original box (two CNs); unchanged.
- FE rebuilds: two rebuilds on 8 cores; measure the first (`pixi run -e fe fe-build` in the wp2 tree) before scheduling the second; the demo FE keeps serving every
  other tree meanwhile (never rebuild in place).
- P0 translate-only EXPLAIN capture: `gate-a.sh` parameterized (tree, output, `GPU_DEVICES=0,1`, L-A sizes, `TPCH_SF=500`).
- P5 acceptance: 1-CN first on one MIG instance at 40/4 (S-MCN-1mig vs R5-MCN-1mig, S-I2-1mig vs R5-I2-1mig); then 2-CN one variable each with B pinned at both
  L-A and L-B: S-MCN-LB-2mig, S-MCN-LA-2mig, S-I2-LA-2mig. Gates verbatim (>= 10x wire-byte reduction on flipped edges, broadcast bytes within budget, MATCH every
  run, no fitting query > 10% slower than the same-box reference), with ">= 21/22 both splits" replaced by ">= the same-box reference pass count + every
  predicate-flipped query".

## 5. WP3 -- Bounded in-flight window first, then a streaming receiver

Goal, principles, files, tests, risks, dependencies: original §5. Box deltas:
- P0 arms: M-dbg-2mig (L-B, the same-box control's failing queries x3), M-chk-2mig (checksum on, x30 per query with any VALUES-DIFFER/failure in R5-MCN16-2mig;
  P(miss) = (1-p)^30 stated). At SF500 a x30 arm of one query is ~5-15 min; affordable.
- P2 HWM initial = 2 x hash_partition_bytes x remote senders = 2 GiB (default batch) or 1 GiB (B pinned) at L-A; sweep {1,2,4}x. Arms M-sink-2mig, M-hwm-2mig,
  M-hwm-sweep at L-B then all 22. Gate verbatim (0 futility, oom_reschedules 0, inequality holds, no query > 10% slower than R5-MCN16-2mig).
- P3/P4: verbatim; acceptance layouts L-B and L-A (were 76/16 and 84/8); S-MCN-LA-2mig uses WP1's pinned B.
- P5 trigger inequality `max_exchanges(P/N x row width) + W + build > pool` evaluated with P = SF500 rows and pool = 40 GiB: it holds for q05/q08/q09 at 2 CNs
  (42/48/60 GB > 42.9 GB) exactly as it did at SF1000/84 GiB, so the P5 decision gate is reached on this box if P4 passes.
- Host: WP3 P2's conditional GPU landing (skip the D2H/H2D) also reduces HOST occupancy, which matters here (BT §4); report HOST peak per arm.

## 6. Sequencing for one engineer on this box

Serial GPU, hours-long cold builds, so the order is chosen to keep the GPU busy with no-rebuild arms while code is written and built:

| Step | Work | GPU? | Rebuild? |
|---|---|---|---|
| 0 | this folder; worktrees wp0/exchange-telemetry (own build) and wp1/bounded-frames (shell-only, shares mcn build) off mcn 2e0cbf51 | no | no |
| 1 | WP1 P0 launcher (`CN_SIRIUS_CONFIG_DIR`, gen-cn-config.sh incl. DISK tier, GPU_DEVICES kept) + harness knobs (b) retries K=3, (c) per-CN dumps at any N, (d) config.txt fields, (f) per-run compare + per-query engine-log copy, never restart on VALUES-DIFFER; MIG standalone YAML | no | no |
| 2 | same-box references, mode-off only, under the lock: R5-P04-2mig (perf, L-A), R5-MCNoff-2mig (mcn off, L-B), R5-I2-1mig (demo 1 CN 40/4), R5-SA-1mig (standalone); smoke of the config-dir launcher on demo (mode-off) at L-A incl. a DISK-tier smoke | yes | no |
| 3 | wp0: telemetry (a)-(f) + correctness (1)-(5) + protocol rev 2; unit tests; `pixi run make` (hours) + CN build; start the build as soon as the engine side compiles, write the WP3 P1 design note and the WP2 P0 predicate while it builds | build only | **the one shared rebuild** |
| 4 | on the instrumented build: M-dbg-2mig, then R5-MCN16-2mig (L-B, w2, HOST 44 and 44+DISK) = the optimized reference; M-chk-2mig x30 on its failing/VALUES-DIFFER set | yes | no |
| 5 | WP1 P0 arms (B=512 MiB, L-A/L-B/L-C, fixed HOST layout) | yes | no |
| 6 | WP2 P1 engine evidence patch (on the wp0 build, same binary) + arms; WP3 P2 HWM/sink-first/conditional landing on wp3/streaming-shuffle | yes | engine rebuilds (incremental) |
| 7 | WP1 P2 bound threading; WP2 P2 FE rebuild 1; WP3 P2 arms | yes | FE + CN |
| 8+ | original weeks 4-8 with the box's arm names; stop rule applies at every gate | | |

Starts now with no rebuild: steps 0-2. Waits: every optimized-mode arm waits on step 3 (INCIDENT §6). Stop and report if a gate fails twice.

## 7. Deliberately not included (unchanged) plus box items

Original §7 unchanged. Added on this box: (a) un-splitting MIG mid-plan to get a 92 GiB 1-CN control (needs root, breaks same-day comparability; the 1-CN control is
one MIG instance at 40/4); (b) SF1000 on this box (working set 2x the pool, every 2-CN arm would fail the same way; nothing to learn); (c) HOST_MEM above 44 GiB
(pinned host memory; the OS has no swap); (d) treating a HOST-evacuation failure as a WP finding without its DISK twin.
