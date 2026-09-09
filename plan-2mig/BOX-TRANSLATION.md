# Box translation: 2x RTX PRO 6000 (SF1000) -> 1x RTX PRO 6000 split into 2 MIG 2g.48gb (SF500)

Date 2026-09-09. Every constant in the original plan documents (`original/`) was derived on the 2-GPU box. This table is the
one place that re-derives them for the g7e.4xlarge; the adapted PLAN / PROMPT / CHECKLIST cite it and never restate a number
without it. Measured values carry the arm that measured them.

## 1. Hardware

| Item | 2-GPU box (original) | This box (g7e.4xlarge) | Consequence |
|---|---|---|---|
| GPUs | 2x RTX PRO 6000, 97,887 MiB (95.6 GiB) each | 1x RTX PRO 6000 in MIG mode: 2x `MIG 2g.48gb`, 48,512 MiB (47.4 GiB) each, 94 SMs each; UUIDs MIG-b033b1f9-5271-5055-a47c-6ad1488f922d (CUDA ordinal 0), MIG-3ee9ed20-4cc9-50c9-990a-d67cc1c592dd (ordinal 1); parent GPU-d70bd34d-b789-3ee0-696a-2134418cb930 | "one CN per GPU" becomes one CN per MIG instance; CUDA 13 / driver 580.126.09 enumerates the instances as ordinals 0 and 1, so `--gpu-device 0/1` works and `CUDA_VISIBLE_DEVICES=MIG-<uuid>` does NOT (cucascade counts GPUs through NVML: "Requested number of GPUs exceeds available GPUs") |
| CN launcher | `cluster8.sh` one CN per visible GPU | `GPU_DEVICES=0,1` override (commit 92b7eed4 on `bench/sf500-single-gpu-2cn`, carried on `bench/sf500-2-mig-gpus` 59ab07c3) because `nvidia-smi --query-gpu=index` reports 1 GPU | every arm exports `GPU_DEVICES=0,1`; the harness (`/home/ubuntu/sirius-wt/harness/capture-arm.sh`) accepts it past the GPU-count check |
| Pool + arena per CN | <= 92 GiB of 95.6 (84/8 or 76/16); context + libs ~3.6 GiB | <= 45 GiB of 47.4: measured 45,530 MiB resident with 40 GiB pool + 4 GiB arena (S500-I2-2cn-2mig), i.e. 474 MiB overhead on top of pool + arena | layouts below |
| Host RAM | 499 GiB; HOST_MEM 160 GiB per CN (320 pinned) | 124 GiB total, no swap; FE heap 8 GiB (`-Xmx8192m`); the host arena is `numa_alloc_onnode` + `cudaHostRegister` (pinned at bring-up) | HOST_MEM 40 GiB per CN (80 GiB pinned) is the proven value; 44 GiB is the ceiling (88 pinned + 8 FE leaves 28 GiB for OS, mysql, harness, page cache). 160 GiB is impossible |
| CPU | 48 cores | Xeon 8559C, 8 cores / 16 threads | cold builds (sccache cache was on the wiped nvme): engine `pixi run make` is hours, not minutes; CN `cargo build --release` tens of minutes; FE build: measure once, budget an hour; DuckDB oracle at SF500 with 16 threads = 3.5 min for 22 queries (measured 2026-09-08) |
| Scratch | `/opt/dlami/nvme` 3.5 TB | `/opt/dlami/nvme` 1.7 TB, wiped on stop/start (unchanged) | dataset regen per `~/regen_tpch.sh 500` (232 s); DISK tier candidate (see §4) |
| Transport | nixl/UCX cuda_ipc between two GPUs; canary 52.5/51.0 | nixl canary between the two MIG instances 52.9 / 52.1 Gbps (S500-I2-2cn-2mig cluster.log 01:41:18) | transport substrate unchanged; plan-00 canary requirement satisfied |
| Instance identity | GPU UUIDs changed on 2026-09-06 (host move) | record parent GPU UUID + both MIG UUIDs in every config.txt | timings from the 2-GPU box are never comparable to this box |

## 2. Workload

| Item | Original | This box | Why |
|---|---|---|---|
| Dataset | TPC-H SF1000, 265 GiB, lineitem 6.0e9 rows, `/home/ubuntu/tpch_parquet_sf1000` | TPC-H SF500, 132 GiB, lineitem 3,000,028,242 rows (30 files), `/home/ubuntu/tpch_parquet_sf500` (tpchgen-rs cdcf74d, regenerated 2026-09-08 19:02 UTC; verifier's lineitem constant is off by +356,561, every other table OK) | pool halves, so the scale factor halves: the per-CN shuffle working set stays at the edge |
| Per-CN shuffle working set (local half + inbound half = rows x width) | 3.0e9 rows x 28/32/40 B = 84/96/120 GB vs 81.6-90.2 GB pool | 1.5e9 rows x 28/32/40 B = 42/48/60 GB vs 38.7 GB (36 GiB) or 42.9 GB (40 GiB) pool | same three queries (q05/q08/q09) sit past the edge; same failing set observed: S500-I2-2cn-1gpu and S500-I2-2cn-2mig both fail q05 q08 q09 q17 q18 q21 with GPU out_of_memory |
| Oracle | `/home/ubuntu/starrocks-tools/oracle/tpch_sf1000` | `/home/ubuntu/starrocks-tools-wt/sf500-2-mig-gpus/oracle/tpch_sf500` (identical copy in `starrocks-tools-wt/sf500-single-gpu/oracle/tpch_sf500`; branch `sf500-2-mig-gpus` on github.com/aocsa/starrocks-tools) | generated 2026-09-08 by `experimental/starrocks/tools/oracle.py`, 16 threads, 90 GB, spill on nvme |
| Kit | `demo/experimental/starrocks/benchmarks/tpch/queries`, q11 SF-scaled via `__TPCH_SF__` | same kit, `TPCH_SF=500` exported by every driver (cluster-env.sh defaults to 1000) | q11 constant |
| Standalone control | `all22/sirius-standalone-rtx6000.yaml` (86 GiB, operator_params 3/12/2/12 GB) | must be rewritten for one MIG instance: `usage_limit_bytes: 40GiB`, operator_params 1/1/1/2 GiB; not yet done | old YAML exceeds 47.4 GiB |

## 3. Memory layouts and the derived constants

Engine batch = `min(clamp(device_total/40, 512 MiB, 5 GiB), pool/40)` (`mcn/src/sirius_config.cpp:42-60, :573-591`); device_total on a MIG instance is 48,512 MiB.
mcn per-frame cap in optimized mode = `arena/4 - 8 MiB` (`mcn/src/sirius_ffi.cpp:1104, :1176-1182`); RX max_batch = arena/4 (`exchange_protocol.rs:162-165`).

| Layout name | Original (pool/arena) | This box (pool/arena) | batch = pool/40 | frame cap = arena/4 - 8 MiB | cap violated? | Role |
|---|---|---|---|---|---|---|
| L-A | 84/8 GiB | **40/4 GiB** | 1024 MiB | 1016 MiB | **yes** (as 84/8 was: 2.25 GB > 2.139 GB) | integration / path-04 / mcn-off reference; WP1 P0 target ("restore L-A in optimized mode") |
| L-B | 76/16 GiB | **36/8 GiB** | 921.6 MiB | 2040 MiB | no | mcn optimized reference (as 76/16 was) |
| L-C | 76/8 GiB | **36/4 GiB** | 921.6 MiB | 1016 MiB | no | "pool only" control (as 76/8 was) |
| 1-CN | 84/8 on one GPU | 40/4 on one MIG instance (a 92 GiB 1-CN has no analog: MIG cannot be un-split from inside a run) | | | | 1-CN control |

WP1 bound B (plan 10: `B + 8 MiB <= A/4` in both fan-out modes): at A = 4 GiB, B <= 1016 MiB. Chosen **B = 512 MiB** (`operator_params.{scan_task_batch_size,hash_partition_bytes,concat_batch_bytes} = 512 MiB`, `max_build_hash_table_bytes = 2B = 1 GiB`); P0 screening points {256, 512, 768} MiB. The original "1 GiB" was 84 GiB/40 rounded; here 1 GiB is the cap-violating default, not a bound.
Pinned-B pool comparisons: L-A vs L-B = "pool + arena layout", L-A vs L-C = "pool only" (original 84/8 vs 76/16 and 84/8 vs 76/8).
WP2 FE byte budget: `broadcast_bytes_limit <= max_build_hash_table_bytes = 2 x pool/40` = 2 GiB at L-A (was 4.2 GiB), 1 GiB with B pinned.
WP3 HWM initial value: `2 x hash_partition_bytes x remote senders` = 2 GiB at L-A default batch, 1 GiB with B pinned (was 4.5 GB).

## 4. Host memory: the new binding constraint

mcn optimized mode is HOST-first (every inbound frame lands pinned in HOST, D2H then H2D). On the 2-GPU box at SF1000 it consumed 128-133 GiB per CN with q09 peaking at 138-143 GB (ANALYSIS §2.1, CHECKLIST §3). Scaled to SF500 that is ~65-70 GiB per CN against a 40-44 GiB HOST_MEM ceiling. Expected outcome: a failure class the original plan lists but never saw bind, **HOST evacuation unavailable**, on q05/q08/q09 in optimized mode. Two adaptations, both launch-config only (no code):
1. `HOST_MEM=44GiB` for optimized-mode arms (ceiling), 40 GiB for everything else, recorded in config.txt.
2. **DISK tier on `/opt/dlami/nvme`** (`memory.disk.{capacity_bytes, downgrade_root_dirs}`, `mcn/src/sirius_config.cpp:540-559`), reachable only through a full `--sirius-config` YAML, i.e. through WP1 P0's `CN_SIRIUS_CONFIG_DIR` launcher. The original plan dropped DISK as *the q08 fix* (eligibility, not capacity, bound there); here it is a host-capacity substitute, not a fix, and every arm records whether it was configured and how many bytes went to DISK (`tier=DISK` counts). Validate with a smoke arm before use.
Every arm in the adapted plan therefore carries one more variable than the original: HOST layout (40 / 44 / 44+DISK). One variable per arm still applies: the DISK tier is fixed per comparison set.

## 5. References and gates

The original gates cite same-day arms on the 2-GPU box (R-MCN16-2cn 121.859 s common-16, R-I2-1cn, R-SA-1gpu, MCN-on-2cn w1). None is valid here. Same-box references, all SF500, 2 CNs on 2 MIG instances unless stated:

| Reference | Tree | Layout | Status | Result |
|---|---|---|---|---|
| S500-I2-2cn-2mig | demo (all22/integration 95bec853) | L-A, HOST 40 | **done 2026-09-09 01:41-01:48** | 16/22 MATCH; q05 q08 q09 q17 q18 q21 out_of_memory; warm sums identical to S500-I2-2cn-1gpu (shared un-split GPU) within 2% |
| R5-P04-2mig | perf (perf/exchange-04-07-06 222b646a) | L-A, HOST 40 | to run (no rebuild) | expected 20/22 by analogy (q08 q09 remain) |
| R5-MCNoff-2mig | mcn (2e0cbf51), `SIRIUS_EXCHANGE_OPTIMIZED` unset | L-B, HOST 40 | to run (no rebuild) | expected 16/22 |
| R5-I2-1mig | demo, 1 CN on ordinal 0 | 40/4, HOST 40 | to run (no rebuild) | 1-CN control; at SF500/40 GiB the 1-CN escape hatch may NOT hold (R-I2-1cn needed exactly 84 GiB for q05 at SF1000) |
| R5-MCN16-2mig | mcn optimized, window 2 | L-B, HOST 44 (+DISK arm twin) | **only after the shared instrumented rebuild** (INCIDENT §6) | the optimized-mode reference for every WP gate |
| R5-SA-1mig | standalone DuckDB CLI + integration extension, one MIG instance | 40 GiB YAML | to run after the YAML rewrite | standalone control |

Regression bound (original: no fitting query > 10% slower than R-MCN16 warm): here **no fitting query > 10% slower than the same-box, same-day reference arm of the same tree** (S500-I2 for demo, R5-P04 for perf, R5-MCN16 for mcn). Timing numbers from arms before 2026-09-09 are not comparable.

## 6. Unchanged

Mechanisms, file anchors, tests, generality rule, failure taxonomy, protocol (1 cold + 2 warm, TO 300, watchdog 240, RESTART_ON_FAIL=1, oracle every run at rel 1e-6, ASYNC=1), lock rule (`all22/gpu-lock.sh`), the INCIDENT ordering rule (no optimized-mode 2-CN arm before telemetry + correctness instrumentation land), the stop rule (gate fails twice -> stop and report).
