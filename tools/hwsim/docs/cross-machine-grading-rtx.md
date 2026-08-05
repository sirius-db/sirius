# Cross-machine grading — the frozen GB300→RTX predictions scored on the RTX box

**2026-08-05, RTX PRO 6000 Blackwell box (x86/PCIe).** Grades the pre-registered
predictions of [cross-machine-predictions-README.md](cross-machine-predictions-README.md)
(frozen at `ec5bd09d`, produced from the GB300 SF100 trace + this box's spec-sheet
descriptor, no RTX measurement) against the real SF100 baselines this box captured for
[external-validation-rtx-pro-6000.md](external-validation-rtx-pro-6000.md). Grader:
`experiments/bin/grade_cross_machine.py`, default `--iters 2,3` (the pre-registered
method); real session `019fd242-f189…` (22 q × 3 iters, config-matched per the README).
Grading-only — nothing on the prediction side was refit or re-derived. Also contains the
two follow-up validations run on the new branch head (§5, §6).

## 1. Headline

**The absolute predictions failed their pre-registered bands; the ordering survived; the
failure is one tier, and it is the tier the design flagged as unvalidated
(`cpu_compute`).** This box runs the suite 1.8–2.9× *faster* than the GB300
(real/GB300-source per query 0.34–0.55, median 0.42); the primary arm predicted 0.87×
(suite), the frozen arm 0.77×. Both arms over-predict every query. Rank correlation with
reality is 0.94 — the model knows *which* queries are expensive, but priced the host tier
(85–93% of span time on this lane) with a directionally inverted knob.

## 2. Per-query table (primary nominal + optimistic, cpufrozen nominal, real)

Real = median of iters 2,3. E = pred/real − 1.

| q | GB300 src ms | nominal ms | optimistic ms | cpufrozen ms | real ms | E_nom | E_froz |
|---|---|---|---|---|---|---|---|
| 1 | 1132.8 | 1051.1 | 982.0 | 836.1 | 597.4 | +75.9% | +39.9% |
| 2 | 454.7 | 415.8 | 414.0 | 383.7 | 154.7 | +168.7% | +148.0% |
| 3 | 1233.3 | 960.7 | 949.3 | 786.1 | 593.3 | +61.9% | +32.5% |
| 4 | 696.7 | 654.6 | 651.1 | 621.4 | 294.6 | +122.2% | +110.9% |
| 5 | 1538.3 | 1138.2 | 1129.5 | 885.2 | 571.2 | +99.3% | +55.0% |
| 6 | 803.6 | 560.1 | 548.0 | 411.3 | 413.8 | +35.4% | **−0.6%** |
| 7 | 1710.3 | 1647.2 | 1644.8 | 1597.0 | 661.7 | +148.9% | +141.3% |
| 8 | 1965.6 | 1450.0 | 1436.8 | 1112.4 | 758.2 | +91.3% | +46.7% |
| 9 | 2377.6 | 2158.6 | 2135.6 | 1953.4 | 966.3 | +123.4% | +102.2% |
| 10 | 1495.5 | 1543.1 | 1485.1 | 1495.9 | 601.1 | +156.7% | +148.8% |
| 11 | 334.0 | 283.7 | 281.0 | 255.5 | 126.8 | +123.8% | +101.5% |
| 12 | 833.4 | 806.5 | 802.7 | 782.1 | 454.4 | +77.5% | +72.1% |
| 13 | 1120.7 | 1178.4 | 1165.2 | 1162.7 | 504.4 | +133.6% | +130.5% |
| 14 | 1285.3 | 895.4 | 880.0 | 639.5 | 551.6 | +62.3% | +15.9% |
| 15 | 1280.1 | 909.1 | 897.2 | 655.6 | 529.2 | +71.8% | +23.9% |
| 16 | 251.9 | 268.7 | 259.9 | 242.0 | 85.2 | +215.2% | +184.0% |
| 17 | 1789.1 | 1582.0 | 1577.5 | 1446.5 | 733.5 | +115.7% | +97.2% |
| 18 | 1080.7 | 1000.8 | 994.7 | 945.3 | 596.1 | +67.9% | +58.6% |
| 19 | 1421.8 | 1086.3 | 1057.7 | 836.6 | 587.9 | +84.8% | +42.3% |
| 20 | 1547.1 | 1553.5 | 1543.9 | 1549.4 | 643.6 | +141.4% | +140.7% |
| 21 | 2879.0 | 2599.3 | 2585.3 | 2398.4 | 1276.9 | +103.6% | +87.8% |
| 22 | 328.1 | 296.8 | 293.4 | 268.6 | 129.7 | +128.8% | +107.1% |

## 3. Aggregates and the pre-registered bands

| arm / scope | med E | med \|E\| | suite TW | rank ρ | [nom,opt] brackets |
|---|---|---|---|---|---|
| primary, q11 in | +109.6% | 109.6% | +103.2% | 0.936 | 0/22 |
| primary, q11 out | +103.6% | 103.6% | +103.0% | 0.926 | 0/21 |
| cpufrozen, q11 in | +92.5% | 92.5% | +79.7% | 0.915 | 0/22 |
| cpufrozen, q11 out | +87.8% | 87.8% | +79.5% | 0.903 | 0/21 |

Against `cross-machine-experiment.md` §4 (primary arm, q11 in — q11-out changes no
verdict):

| band | result | tier |
|---|---|---|
| suite time-weighted E | +103.2% | **FAILURE** (beyond ±50%) |
| per-query median \|E\| | 109.6% | **FAILURE** (> 50%) |
| gpu_device/gpu_memory-bound queries | — | **vacuous**: the G4b gate stood down on all 44 GB300 windows (serialization 0.41–0.90), so no query was predicted device-bound. (Deviation note: the runbook says the CSVs carry a predicted-binding column; they don't — 9 columns, provenance only. The vacuousness is established from the README's own ingest diagnostics, not per-row data.) |
| rank ρ(real, nominal) | 0.936 | **OUTSTANDING** (≥ 0.9) |
| [nominal, optimistic] bracket coverage | 0/22 | informative-tier miss (band is degenerate: the two arms' knob vectors differ only in derate direction and sit within ~1–9% of each other, far tighter than the real error) |

Anomaly flags per the runbook (|E| > 50% → one-line hypothesis): **19 of 22 queries**
in the primary arm — a single common-mode hypothesis covers all of them (§4): the host
tier is charged at `cpu_compute=0.667` while reality is an effective ~2.2–2.9×. No
per-query binding hypothesis is checkable (no binding column; see above). The three
sub-50% queries (q6 +35%, q22 +41% — nearly, q16's opposite — and q3 +62% narrowly
misses) are the most scan-bound: the *validated* io knob carried them.

## 4. The cpu_compute question — first empirical read

**The frozen arm is closer on 22/22 queries** (median |E| 92.5% vs 109.6%; suite +79.7%
vs +103.2%) — and still fails by a wide margin. The pre-registered arm spread (13.1%
suite-level) bounded the knob's *modelled* influence; the real error attributable to the
host tier is ~80–100 pp. Both facts point the same way:

- **Reality**: real/GB300-source = 0.34–0.55 across the suite. On the four
  host-dominated flagships the primary arm predicted *slower-or-equal* walls than the
  GB300 source (q10 1543 vs 1496, q13 1178 vs 1121, q16 269 vs 252, q20 1554 vs 1547);
  reality ran them at 0.34–0.45× the source. E_nom +134% to +215% — the four worst
  cells in the table, exactly the queries the experiment pre-identified as the
  cpu_compute probes.
- **Implied effective knob**: holding the frozen arm's other tiers, matching the real
  walls on the host-heavy queries needs host time at ~0.35–0.45× GB300's — an effective
  `cpu_compute ≈ 2.2–2.9`, vs the assumed 0.667. The cores-only ratio is off by
  3.3–4.3× *and directionally inverted*.
- **Mechanism**: both configs run 4 pipeline executor threads (config-matched by
  design). Host work on this lane is per-core-latency-bound — the engine never uses
  more than ~8 threads, so a 48-vs-72 core-count ratio measures a resource the workload
  does not consume. What transfers is per-core throughput (≈5 GHz Zen x86 vs 3.4 GHz
  Neoverse-V2, plus ISA/decode differences on parquet/glue code), which favours this box
  ~2–3× — matching the implied knob.
- **Descriptor-schema consequence**: `cpu_compute` from spec sheets must be a
  *per-core-performance* ratio (boost clock × an arch-IPC class), optionally scaled by
  `min(engine_thread_footprint, cores)` — not a cores ratio. Until a per-core anchor
  exists (a trivial single-thread microbenchmark in the calibration kit would do — it
  runs on any box in seconds and does not touch the model), host-dominated lanes should
  carry a loud LOWER/UPPER-BOUND warning the way device-saturated lanes already do.

The failure is informative in precisely the way §4's rationale hoped: the per-query
decomposition isolates one tier; the validated tiers (io, and the split's GPU shares)
show through on the scan-bound queries (q6 −0.6%, q14 +15.9%, q15 +23.9% in the frozen
arm — the arm that removes the bad knob).

## 5. Follow-up 1 — q11 replay fix (`57b4dae6`) on this box: **does not fix RTX's q11**

Selfcheck on the RTX baseline (`019fd242-f189…`) at the new head, `--no-cache` verified:

| iteration | before (e38a9d84) | after (57b4dae6) |
|---|---|---|
| q11 iter1 | +14.78% | **+14.78%** |
| q11 iter2 | −0.17% | −0.17% |
| q11 iter3 | +10.31% | **+10.31%** |
| suite median / p90 / worst | 0.13 / 0.54 / 14.78 | 0.13 / 0.54 / 14.78 |

Bit-identical — the dispatch fix (which cured GB300's q11 iter3 +17.9% → −0.05%) does
not engage on this box's anomaly. The fix's data path is live here: all 61 q11 tasks
carry `t_queued_exec`, and the bad iterations do contain executor-vs-scheduler order
inversions (3 in iter1/iter3, 1 in iter2). Ruled out by direct probes: thread
over-subscription (busy/wall 3.52–3.62 on 4 threads, max concurrency 4, in good and bad
iterations alike) and attribution volume (`ambiguous_producer_batches` 18/17/17).
**RTX q11 is a second, distinct replay defect, still open** — same query shape
(61 tasks / 33 pipelines / ~130 ms), different mechanism from WS20 item 4.

## 6. Follow-up 2 — overlap-cap (WS20 item 3) on its decisive dataset: **clean null**

Re-ingested `hwsim_run/nsys/paired_base.sqlite` at the new head
(`physics_paired_ws20.json`, 17,066 span entries with `f_kernel_overlap`) and re-scored
the MPS-50 what-if exactly as in `hwsim_run/RESULTS.md`:

- Suite: physics median E +0.2%, median |E| 3.3%, time-weighted **+0.8%** (was +0.9%
  pre-WS20; small shifts also carry the 57b4dae6 dispatch change). 20/22 within ±10%,
  unchanged.
- **q17 +15.6%, q20 +19.7% — unchanged.** The cap stands down on them, and the
  diagnostic shows why: kernel-time share in spans at overlap ≥ 0.9 is **0.4% (q17) and
  0.0% (q20)** (suite-wide: span-median overlap 0.0, only 749/17,066 entries ≥ 0.9).
  Their launching threads sit in sync APIs over the kernels — "kernel time hidden under
  host work" is *not* this box's q17/q20 mechanism.
- Revised hypothesis for the residual pessimism: these kernels are
  **SM-partition-insensitive** (low-occupancy/latency-bound — the §8.7 population,
  here on the query scale instead of the lane scale). Distinguishing that needs
  per-kernel occupancy (GPU_METRICS / ncu), perf-counter-blocked on both boxes so far.
- The cap's gating guarantees held: knobs=1 identity within envelope, no disturbance of
  the other 20 queries. The mechanism is validated as *safe*; its motivating datum is
  answered "not this".

## 7. Artifacts

Grader outputs reproduced from
`experiments/analysis/predictions_sf100{,_cpufrozen}.csv` against
`hwsim_run/telemetry_data/019fd242-f189…` on the RTX worktree; WS20 re-score inputs
`hwsim_run/physics_paired_ws20.json` + per-query sim JSONs (session scratchpad); q11
probes inline (this doc §5). Real-wall extraction: the grader's own (Executing→Exit,
median iters 2,3).
