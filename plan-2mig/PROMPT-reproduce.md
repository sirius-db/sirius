# Prompt: reproduce the SF500 MIG experiments in a fresh Claude Code session

Start the session in `/home/ubuntu/sirius-wt` and paste everything below the line.

---

You are continuing a benchmark and diagnosis campaign on Sirius (GPU SQL engine, DuckDB extension,
cuDF/RMM/cuCascade) running as a StarRocks compute node (CN). **Everything below has already been
built, measured and committed. Do not redo it.** Read this, verify the preflight, then continue from
"Open questions".

## 1. The box

AWS **g7e.4xlarge**: ONE RTX PRO 6000 Blackwell in **MIG mode**, split into two `2g.48gb` instances
(47.4 GiB each), Xeon 8559C 8 cores / 16 threads, **124 GiB RAM, no swap**, 1 TB EBS root,
1.7 TB instance store at `/opt/dlami/nvme`. CUDA 13, driver 580.126.09.

This is NOT the 2x RTX PRO 6000 / 48-core / 499 GB box the older docs describe. Timings from before
2026-09-08 are not comparable. `/opt/dlami/nvme` is **wiped on every stop/start**.

Recovery after a restart:
```bash
for l in ~/.cache/rattler ~/.cache/sccache ~/.cargo ~/.rustup ~/.m2; do mkdir -p "$(readlink $l)"; done
mkdir -p /opt/dlami/nvme/tmp
bash ~/regen_tpch.sh 500      # 232 s; "RESULT: FAIL" on lineitem is a known verifier-constant bug
```

## 2. Read these first (10 minutes)

1. `/home/ubuntu/sirius-wt/s500-mig/plan-2mig/BOX-TRANSLATION.md` — every constant re-derived for this
   box. Wins over every other doc on numbers.
2. `/home/ubuntu/sirius-wt/s500-mig/plan-2mig/PLAN-top3-2mig.md` and `PROMPT-top3-2mig.md`.
3. `/home/ubuntu/sirius-wt/s500-mig/plan-2mig/DISK-TIER.md` — how to enable spilling and why.
4. `/home/ubuntu/starrocks-tools-wt/sf500-2-mig-gpus/evidence/estimator/FINDING.md` — **the most
   important result**: q13's 73.6 GB reservation traced to a hard-coded 16x heuristic.
5. `/home/ubuntu/sirius-wt/all22/INCIDENT-q08-wrong-answer-mcn.md` — the silent wrong-answer defect,
   reproduced again on 2026-09-09.

## 3. Trees (all built; do NOT rebuild unless you change code)

`source /home/ubuntu/sirius-wt/env.sh` first in every shell. Clone: `/home/ubuntu/sirius-wt/demo`
(remote `git@github.com:aocsa/sirius.git`), with worktrees under `/home/ubuntu/sirius-wt/`:

| Worktree | Branch | What |
|---|---|---|
| `demo` | all22/integration 95bec853 | baseline engine + CN + the FE every tree symlinks |
| `perf` | perf/exchange-04-07-06 | path-04 credits + staged-frame spill |
| `mcn` | perf/multi-cn-ingress-packing-transfer | optimized exchange, opt-in |
| `wp0` | **wp0/exchange-telemetry** | **the instrumented build used for every 2026-09-09 result** |
| `s500-mig` | bench/sf500-2-mig-gpus (**pushed**) | launcher + plan docs |
| `wp1` | wp1/bounded-frames | launcher only; redundant, cherry-picked into wp0 |

`perf` and `mcn` carry the `GPU_DEVICES` launcher patch **uncommitted** (cluster8.sh only) so they
show `dirty=1`. That is intentional.

Tools repo: `/home/ubuntu/starrocks-tools-wt/sf500-2-mig-gpus`, branch `sf500-2-mig-gpus`
(**pushed** to `aocsa/starrocks-tools`). Drivers, oracle, harness and all evidence live there.

## 4. Non-obvious rules that cost time to learn

- **MIG placement**: always `GPU_DEVICES=0,1`. CUDA enumerates the two MIG instances as ordinals 0
  and 1. `CUDA_VISIBLE_DEVICES=MIG-<uuid>` does **not** work: cucascade counts GPUs through NVML and
  fails with "Requested number of GPUs exceeds available GPUs".
- **Launch detached**: the interactive background shell is capped at 10 minutes. Every arm runs as
  `setsid nohup bash /home/ubuntu/sirius-wt/all22/gpu-lock.sh bash <driver> > logs/<tag>.log 2>&1 < /dev/null &`.
  Use the **absolute** path to `gpu-lock.sh`; a relative path breaks from any worktree.
- **Killing an arm**: also kill its `capture-arm.sh` / `run-queries.sh` children and run
  `harness/stop-cluster.sh <tree>`. A half-killed driver once ran queries against the next driver's cluster.
- **`run-queries.sh` stops a query at its first failed run**, so an intermittent failure yields ONE
  sample, not the requested count. An improved version that runs every repetition, compares each run
  against the oracle immediately and marks wrong answers `wrong`, lives in the tools repo at
  `scripts/sf500-2mig/harness/run-queries.sh` and is **not installed** into `/home/ubuntu/sirius-wt/harness`.
  Install it if you need intermittency statistics.
- **Reading quent telemetry**: session dirs are named `.cn0` / `.cn1`, so Python `glob` skips them
  (dotfiles). Use `os.walk`. The `Reserving` task record carries `requested_bytes`, `peak_estimate`,
  `input_basis` and `bytes_to_materialize` — this is how the estimator was caught.
- **Spill sizing is unobservable after the fact**: spill files are deleted on reload, so measuring the
  spill directory at arm end reports residue, not peak. Sample during the run.
- Layouts, per CN: **L-A 40/4**, **L-B 36/8**, **L-C 36/4** GiB (pool/arena), `HOST_MEM=40GiB`
  (44 is the ceiling; 160 as in the old docs is impossible on 124 GiB).

## 5. Already measured — do not repeat

SF500, 2 CNs on 2 MIG instances, oracle at rel 1e-6 on every run. Arms in `/home/ubuntu/sirius-wt/arms/`.

| Arm | Config | Result |
|---|---|---|
| `S500-I2-2cn-1gpu` | integration, both CNs sharing the un-split GPU | 16/22 |
| `S500-I2-2cn-2mig` | integration, one CN per MIG instance | 16/22, timings within 2% of shared |
| `R5-I2-1mig` | integration, 1 CN | 21/22 (only q05 fails) |
| `R5-P04-2mig` | perf/exchange-04 | 19/22 |
| `R5-MCNoff-2mig` | mcn, optimized OFF | 16/22 |
| `R5-SA-1mig` | standalone DuckDB CLI | **22/22** |
| `R5-MCN16-2mig` | mcn optimized w2, wp0 build | **17/22**, 0 wrong in 58 runs |
| `M-dbg-2mig` | wp0 telemetry smoke, q03+q08 | q03 3/3 MATCH; all telemetry families attributable |
| `D-MCN16-2mig` | + DISK tier, q05/q08/q09/q21 | q05/q09/q21 **flip to pass**; **q08 wrong answers** |
| `D-q13-2mig` | + DISK, q13 | unchanged (0 to_disk requests) |
| `D-q13-b512-2mig` | + `BATCH_BYTES=512MiB`, q13 | still fails; max pack 76 MB, cap never near |
| `G-q13-2mig` | + reservation-gate change, q13 | unchanged; `gate_waits=0` |

Conclusions already established:
- MIG neither helps nor hurts; the edge is the pool-to-scale ratio.
- The DISK tier converts three host-full failures into correct results. **It is a substitute for the
  375 GiB of RAM this box lacks, not a fix for the exchange working set.**
- Optimized mode nets 17/22 vs 16/22 mode-off: gains q17/q18, loses q13.
- **q08 returns silent wrong answers** (2 of 3 runs, with zero restarts, and `[exchange_reconcile]`
  clean across 223 lines — so it is value-level, not row loss). Optimized mode is not release-ready.
- **q13's cause is the estimator**: `DENSE_COUNT_JOIN` asks for `16 x input + 1 MiB` = 73.61 GB from a
  4.60 GB input, which is 1.90x the whole pool. Exact match to telemetry. The task then **succeeded on
  a 34 GB partial grant**, so the estimate is inflated. Frame size and waiting were both ruled out by
  measurement.

## 6. How to run an arm

```bash
cd /home/ubuntu/sirius-wt && source env.sh
S=/home/ubuntu/starrocks-tools-wt/sf500-2-mig-gpus/scripts/sf500-2mig
setsid nohup bash all22/gpu-lock.sh bash $S/arms-R5MCN-2mig.sh \
  > logs/R5MCN-$(date -u +%Y%m%dT%H%M).log 2>&1 < /dev/null &
```

Drivers in `$S` (all take `TAG`, `QUERIES`, `RUNS`, `CFG` via env where noted):
`arms-R5.sh` (mode-off references), `arms-R5MCN-2mig.sh` (optimized, all 22),
`arms-Mdbg-2mig.sh` (telemetry smoke), `arms-DISK-2mig.sh` (disk tier; `TAG=`/`QUERIES=`/`RUNS=`/`CFG=`),
`arms-S500-mig.sh`, `arms-SA-mig.sh` (standalone).

To enable spilling (only reachable through a full `--sirius-config`, which is clap-exclusive with the
memory flags — that is why the `CN_SIRIUS_CONFIG_DIR` launcher exists):

```bash
cd /home/ubuntu/sirius-wt/wp0/experimental/starrocks
DISK_ROOT=/opt/dlami/nvme/spill DISK_BYTES=600GiB NUM_CNS=2 GPU_MEM=36GiB HOST_MEM=40GiB \
  OUT_DIR=/home/ubuntu/sirius-wt/arms/DISK-cn-cfg bash benchmarks/gen-cn-config.sh
```
then pass `CN_SIRIUS_CONFIG_DIR=<that dir>`. Add `BATCH_BYTES=512MiB` to pin operator_params.
Reference outputs: `$S/cn-configs-reference/`.

## 7. Open questions — start here

1. **Fix the DENSE_COUNT_JOIN estimate** (`src/op/sirius_physical_dense_count_join.cpp`,
   `no_history_peak_memory_estimate`). It returns `max(dense_peak, sparse_peak, minmax_peak)` and
   `sparse_peak = 16 x input` dominates. The estimator cannot choose a mode because it has no key
   domain. Options in `evidence/estimator/FINDING.md`: compute the key domain (a min/max reduction)
   before reserving so the real mode is known; or reserve for dense and recover through the existing
   OOM reschedule path; or re-derive the 16.
2. **q08's wrong answers.** Reconciliation is clean, so it is value-level. The decisive experiment is
   the checksum arm, which needs the wire to carry the checksum — protocol revision 2, deferred. The
   sender-side checksum already appears in `[exchange_frame]` lines.
3. **Measure the full 22 with the DISK tier.** The current 20/22 figure is a *projection* splicing two
   arms, not a measurement.
4. WP1 P2+ and WP3 remain unstarted.

## 8. Rules

Every GPU arm goes through `all22/gpu-lock.sh`. Oracle-compare every run; a VALUES-DIFFER is a
failure, never a pass. Record the parent GPU UUID and both MIG UUIDs in every `config.txt`. Compare
same-box, same-day arms only. Never drop failed queries from a results table. Append one row per arm
to `/home/ubuntu/sirius-wt/all22/LOG.md`. Do not push unless asked.
