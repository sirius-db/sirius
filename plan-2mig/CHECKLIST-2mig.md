# Checklist: TPC-H SF500 on Sirius-as-StarRocks-CN, 2 CNs on 2 MIG instances of one RTX PRO 6000 (g7e.4xlarge)

Started 2026-09-09 (UTC). Owner: aocsa. Companion to `original/CHECKLIST-2cn-oom-edge-2026-09-07.md`, whose §1 facts (root cause, staging arena is a
bystander, pool size is not the lever on mcn, silent wrong answers in optimized mode, path-04 gain comes from the spill hook) carry over as mechanisms; none of
its numbers or arm names apply here (BOX-TRANSLATION §5).

Where things are: trees and tools in `PROMPT-top3-2mig.md`; arms under `/home/ubuntu/sirius-wt/arms/S500-*` and `R5-*`; drivers, oracle, evidence and the
sirius launcher patches in `/home/ubuntu/starrocks-tools-wt/sf500-2-mig-gpus` (branch `sf500-2-mig-gpus`, pushed); experiment log `all22/LOG.md`.

## 1. Established on this box (do not re-measure)

- [x] Two CNs on one un-split GPU (`GPU_DEVICES=0,0`, 40/4/40): S500-I2-2cn-1gpu 16/22 MATCH, q05 q08 q09 q17 q18 q21 GPU out_of_memory (2026-09-08).
- [x] Two CNs on two MIG 2g.48gb instances (`GPU_DEVICES=0,1`, 40/4/40): S500-I2-2cn-2mig 16/22 MATCH, same six, warm medians within 2% of the shared-GPU
      run on all 16 (2026-09-09 01:41-01:48). MIG neither helps nor hurts at this layout; the edge is the pool-to-scale ratio.
- [x] `CUDA_VISIBLE_DEVICES=MIG-<uuid>` does not work with this engine (cucascade NVML GPU count -> "Requested number of GPUs exceeds available GPUs"); CUDA
      ordinals 0/1 do. `MIG_DEVICES` mode exists in cluster8.sh (59ab07c3) for the day the engine resolves MIG UUIDs.
- [x] nixl/UCX canary between the two MIG instances 52.9 / 52.1 Gbps: transport substrate equals the 2-GPU box.
- [x] Resident GPU memory with pool 40 GiB + arena 4 GiB = 45,530 MiB per instance (474 MiB overhead); 45 GiB pool+arena is the ceiling.
- [x] Default engine batch at L-A = 1024 MiB; mcn frame cap at a 4 GiB arena = 1016 MiB: the cap violation is structural at L-A (BT §3).
- [x] Dataset SF500 regenerates in 232 s; the verifier's lineitem constant is wrong by +356,561 rows (generator is right); DuckDB oracle for all 22 in 3.5 min.
- [x] Failure classes seen so far: q05 "GPU pipeline task exceeded maximum retry limit (100) ... OOM" (200 GPU_SCAN reschedules); q08 q09 q17 engine
      out_of_memory; q18 q21 "failed to stage a 5.2e8 byte inbound frame: out_of_memory" (integration stage-copy into a full pool, same class as at SF1000).

## 2. Tried on this box

| When | Tried | Outcome |
|---|---|---|
| 09-08 | integration 2 CNs sharing the un-split GPU | 16/22 (see §1) |
| 09-09 | integration 2 CNs on 2 MIG instances | 16/22, timings equal to shared |
| 09-09 | MIG UUIDs through CUDA_VISIBLE_DEVICES | engine refuses; use ordinals |

## 3. Open / to do (in plan order, `PLAN-top3-2mig.md` §6)

- [ ] Step 1: config-dir launcher + gen-cn-config.sh (+ DISK tier), harness knobs (b)(c)(d)(f), MIG standalone YAML.
- [ ] Step 2: R5-P04-2mig, R5-MCNoff-2mig, R5-I2-1mig, R5-SA-1mig; config-dir launcher smoke (mode-off) incl. DISK tier.
- [ ] Step 3: wp0/exchange-telemetry instrumented rebuild (telemetry (a)-(f), correctness (1)-(5), protocol rev 2).
- [ ] Step 4: M-dbg-2mig, R5-MCN16-2mig (HOST 44 / 44+DISK), M-chk-2mig x30.
- [ ] Step 5: WP1 P0 arms at B = 512 MiB.
- [ ] Known defects from the original §3 (silent wrong answers, quarantine credits, ledger retirement, 60 s hard failure, default path not byte-identical) are
      all still open and all still apply.

## 4. Protocol reminders

Two CNs one per MIG instance (`GPU_DEVICES=0,1`), SF500 kit (`TPCH_SF=500`), `SET GLOBAL cbo_cte_reuse_rate = 1.15`, 1 cold + 2 warm, TO 300, watchdog 240,
`RESTART_ON_FAIL=1`, oracle `/home/ubuntu/starrocks-tools-wt/sf500-2-mig-gpus/oracle/tpch_sf500` at rel 1e-6 on every run, `ASYNC=1`, everything GPU-side under
`all22/gpu-lock.sh`, drivers detached with `setsid nohup`, same-box same-day comparisons only, HOST layout recorded per arm, never exclude failed queries.
