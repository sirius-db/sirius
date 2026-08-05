# Cross-machine predictions — GB300 → RTX PRO 6000 (WS21, frozen 2026-08-05)

The GB300 side of [cross-machine-experiment.md](cross-machine-experiment.md) is **executed
and frozen**. The two prediction CSVs below were produced BEFORE any grading and are the
pre-registration. No RTX measurement fed them.

## What the owner runs on the RTX box (one command per arm)

Pull this branch (`claude/sirius-hardware-simulation-25edbe`, PR #1424) on the RTX box, then:

```bash
python3 tools/hwsim/experiments/bin/grade_cross_machine.py \
    tools/hwsim/experiments/analysis/predictions_sf100.csv \
    <worktree>/hwsim_run/telemetry_data/<baseline-session-uuid>   # 019fd242-f189... x 3 iters
```

where `<worktree>` is the RTX box's
`.claude/worktrees/simulator-timing-output-prediction-e02735` (per the
[external report](external-validation-rtx-pro-6000.md) artifacts). Repeat with
`predictions_sf100_cpufrozen.csv` (the cpu_compute-frozen sensitivity arm). Grade q11 with
and without (known RTX replay outlier, confound #4). The grader defaults to the median of
iters 2,3 — the design.

## Pre-registered success bands (decided in advance, cross-machine-experiment.md §4)

| scope | success | outstanding | failure |
|---|---|---|---|
| suite time-weighted E (nominal arm) | within ±30% | within ±15% | beyond ±50% |
| per-query median \|E\| | ≤ 30% | ≤ 15% | > 50% |
| queries predicted `gpu_device`/`gpu_memory`-bound | \|E\| ≤ 20% | ≤ 10% | > 40% |
| rank ρ(real, nominal) across 22 queries | ≥ 0.8 | ≥ 0.9 | < 0.6 |
| band [nominal, optimistic] | brackets ≥ 1/3 of reals | ≥ 1/2 | informative |

## What was captured (GB300, this box)

- Engine commit `ec5bd09d` (branch head at capture; **zero `src/` changes since
  `e38a9d84`**, the commit whose build produced the RTX baselines — engines are
  timing-equivalent). Worktree binary sha256
  `522fdfa6…` (duckdb) / `a1977c2d…` (extension), recorded in
  `experiments/throttle_logs/XM-BASE/binary_sha256.txt`.
- Dataset: TPC-H **SF100** parquet, per-table dirs, freshly generated (34 GB,
  `/localhome/local-faramburu/tpch_parquet_sf100`), page-cache-evicted before each capture.
  iostat confirmed every iteration re-reads NVMe (434 GB total read during the baseline,
  ~6.4 GB/s peaks) — the io-knob basis holds; no warm-vs-cold wall split (iter1 ≈ iter2/3).
- Config (`experiments/configs/XM-BASE.yaml`, config-matched to the RTX capture per §1.1):
  ship defaults — **4** pipeline executor threads, default scan manager; `num_gpus: 1`;
  `usage_limit_fraction: 0.95`, `reservation_limit_fraction: 0.8`; quent **ndjson**;
  gates dark (no `SIRIUS_EXP_*` in env), no pins, single process, labeled.
- Sessions: baseline `XM-BASE` = `019fd2cc-c526-7f83-83ca-d172f3b4b28e` (22 q × 3 iters);
  paired quent+nsys `XM-NSYS` = `019fd2d1-07ae-7e23-a83e-098f97a476de` (22 q × 2 iters,
  19 reports; merges q10+q11 and q19+q20+q21 — all 44 windows matched, structure score
  1.00).
- Selfcheck gate (knobs=1 replay, 66 executions): **median 0.03%, p90 0.07%** —
  one outlier, q11 iter3 **+17.9%** (iters 1–2 clean). This reproduces the RTX box's q11
  replay anomaly (2/3 iters, +10–15%) on a different arch — same query, now on both boxes.
- Ingest diagnostics: kernel-ns attribution 99.8–100% (median 99.9%); per-window kernel
  serialization 0.41–0.90 (median 0.64) → the **G4b fluid gate stands down lane-wide**
  (1/44 windows at the 0.9 threshold), as pre-registered for this NVMe-class lane (the
  RTX twin measured 0.53–0.86).

## Prediction (frozen numbers)

`predict_cross_machine.py --iter 2`, target `rtx-pro-6000-blackwell.yaml`, source
`gb300.yaml`, physics `experiments/nsys/XM-SF100/` (all 22 queries on the physics path).

Derived nominal knob vector: `gpu_compute=1.672` (measured fma), `gpu_mem_bandwidth=0.262`
(measured CE), `io_bandwidth=2.238` (measured O_DIRECT), `c2c_bandwidth=0.394`
(RTX pinned 57.7 / source-trace wire peak 146.5 GB/s), `cpu_mem_bandwidth=0.397`,
`cpu_compute=0.667` (cores-only ratio, UNVALIDATED — hence the frozen arm),
`gpu_mem_capacity=0.31434`.

Capacity override note: the runbook's literal `80e9/<pool>` (=0.3929) and the
reservation-scaled convention `(80e9x0.8)/203602924666` (=0.31434, matching the packaged
script's documented example and target.py's same-pool-convention rule) produce
**identical walls on all 22 queries** (SF100 sits far above the spill knee); the CSVs use
0.31434.

| arm | suite (22q, iter-2 sum) | vs GB300 source 27 559.5 ms |
|---|---|---|
| primary nominal | 24 040.1 ms | −12.8% |
| primary optimistic | 23 714.7 ms | −14.0% |
| cpufrozen nominal | 21 264.9 ms | −22.8% |
| cpufrozen optimistic | 20 961.0 ms | −23.9% |

Arm spread (nominal): **13.1%** — the cpu_compute knob's total suite-level influence
(confound #1's bound).

Top-5 by predicted wall (primary nominal): q21 2599 ms, q9 2159 ms, q7 1647 ms,
q17 1582 ms, q10/q20 ≈1543/1554 ms.

## Files

- `tools/hwsim/experiments/analysis/predictions_sf100.csv` — primary (nominal +
  optimistic walls, per-query path + provenance)
- `tools/hwsim/experiments/analysis/predictions_sf100_cpufrozen.csv` — sensitivity arm
- `tools/hwsim/experiments/analysis/XM-BASE_selfcheck.csv` — the gate
- raw traces / nsys reports / physics JSONs stay local per convention
  (`experiments/traces/XM-*`, `experiments/nsys/XM-*`); runs.csv rows `XM-BASE`,
  `XM-NSYS`.

After grading, file the result table into
[validation-results.md](validation-results.md) §10 per the runbook.
