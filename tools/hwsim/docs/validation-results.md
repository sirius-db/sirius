# WS8 — Validation Results: does the v0 simulator predict physically-degraded reality?

**Status:** IN PROGRESS (rows land incrementally; this header is replaced by the final
verdict when the matrix completes). Executor: WS8, 2026-08-04/05, box pmgb300ws-0163.
Runbook: [validation-plan.md](validation-plan.md). Raw data: `tools/hwsim/experiments/`
(runs.csv = manifest; analysis/ = summary CSVs; traces/ + throttle_logs/ gitignored).

## 1. Executive verdict table

(filled at the end)

## 2. Noise floors (P1/P2/P3) — measured first, everything judged against them

**Binary**: `/localhome/local-faramburu/repos/sirius/build/release/duckdb`, dev@b77aa438
(sha256 in `experiments/env/binary_sha256.txt`). Dataset: SF1000 parquet (265 GB).
Config: plan's base.yaml (ndjson quent telemetry), defaults otherwise; pool
usage_limit ≈ 254.5 GB → traced pool capacity 229.05 GB; 4 executor threads.

### 2.1 P1 run-to-run variance (2 sessions × 3 iters, warm, quent on)

All 9 pool queries passed the ≤5% stability rule — none of the known SF1000
13–28% swingers appeared under this (default, untuned) config:

| q | med A (s) | med B (s) | spread A% | spread B% | shift% | σ_q% | quent ovh% |
|---|---|---|---|---|---|---|---|
| 1 | 7.836 | 7.867 | 0.26 | 0.51 | 0.39 | 0.25 | −0.4 |
| 3 | 7.716 | 7.741 | 1.30 | 0.90 | 0.32 | 0.65 | −1.3 |
| 5 | 9.380 | 9.420 | 0.11 | 0.32 | 0.43 | 0.21 | −0.3 |
| 6 | 5.236 | 5.245 | 0.57 | 1.35 | 0.18 | 0.68 | −0.6 |
| 9 | 12.545 | 12.585 | 0.00 | 0.33 | 0.32 | 0.16 | −0.2 |
| 13 | 5.470 | 5.449 | 0.53 | 0.28 | 0.38 | 0.27 | −0.1 |
| 18 | 5.212 | 5.130 | 1.11 | 2.73 | 1.59 | 1.36 | −0.4 |
| 19 | 8.449 | 8.448 | 0.24 | 0.67 | 0.02 | 0.34 | 0.0 |
| 21 | 17.492 | 17.449 | 0.76 | 0.29 | 0.25 | 0.38 | −0.5 |

- **σ_q = 0.16–1.36%** (half-width of the median band). 3σ_q < 4.1% for every query →
  every acceptance threshold in the matrix is governed by its stated floor, not noise.
- **Final throttle set: q1, q9, q13, q19, q21** (plan's target set — all stable).
  I/O set: q1, q6, q19 (subject to B2 cold-stability gate).
- **P2 quent overhead: −1.3%…0.0%** (quent-on marginally *faster*; i.e. overhead is below
  the noise floor on these multi-second walls — far under the +1–5% expectation; no
  absolute-wall caveats needed anywhere).

### 2.2 G-SELF gate (fresh binary + fresh traces)

`python3 -m hwsim selfcheck` on each traced session:

| session | queries | median err% | worst err% |
|---|---|---|---|
| P1a | 27 | 0.05 | 0.36 |
| P1b | 27 | 0.03 | 0.26 |
| B1 | 20 | 0.05 | 0.12 |
| P3 (under MPS pct=100) | 15 | 0.02 | 0.11 |
| E1-75 (under MPS pct=75) | 20 | 0.02 | 0.12 |

**PASS** (gate was median ≤ ~0.5%, worst ≤ 2%). The dirty-tree caveat on the binary is
irrelevant in practice: replay consistency on fresh traces is *better* than WS6's sample
(0.17%/1.2%), and stays intact under MPS and physical degradation.

### 2.3 P3 — MPS-daemon null effect (pct=100)

q1 +0.4%, q9 +0.5%, q19 −0.3%, q21 +0.9% → within/near the P1 band. q13 +2.9% (the
shortest query) shows a small real MPS-daemon overhead. Consequence: **all E1 ratios are
computed against P3 (the pct=100 anchor), not B1**, so the daemon overhead is common-mode.
Probe anchors at pct=100: fma 52 163 GFLOP/s, saxpy 4 787.9 GB/s.

### 2.4 Baseline trace T (B1, 5 queries × 4 iters)

Selfcheck 0.05%/0.12%. Per-query peak GPU pool (sim knobs=1 report, iter2):

| q | wall (med iters 2–4) | peak pool | H2D volume | membw proxy (H2D GB/s of wall) |
|---|---|---|---|---|
| 1 | 7.843 s | 13.7 GiB | 256.5 GB | 32.7 |
| 9 | 12.495 s | 54.7 GiB | 276.9 GB | 22.2 |
| 13 | 5.452 s | 21.2 GiB | 19.0 GB | 3.5 |
| 19 | 8.441 s | 21.5 GiB | 227.9 GB | 27.0 |
| 21 | 17.447 s | 76.3 GiB | 332.0 GB | 19.0 |

Peak pools match the WS6 sample trace within ~1 GiB (q21 76.3, q9 54.7 vs 55.7, q19 21.5
vs 19.2 GiB) → the matrix's E2 capacity points stand: hi=128 GB, mid=64 GB, lo=38 GB.

### 2.5 Discovery that reframes two knobs (found at the B1 analysis gate)

**Fresh traces on this box have ~zero-length transfer spans.** The B1 session records
27 913 SOURCE→GPU Preparing→Computing spans totalling **1.25 s across ~207 s** of query
wall (median 22 µs for ~455 MB payloads ⇒ nominal ~20 TB/s — unphysical). hwsim
accordingly derives a channel "capacity" of 165 280 GB/s (vs 709.6 GB/s in the WS6 sample
trace). Data movement on this coherent-C2C box happens *inside* Computing spans
(zero-copy / fully overlapped reads), invisible to the FSM (the nsys doc's §5.2 caveat,
gap G4). Consequences, pre-registered before running E4/E5:

- the `c2c_bandwidth` knob can only re-time ~0.6% of wall on these traces → the sim is
  structurally **c2c-blind** here; E5 measures the real cost of that blindness;
- the plan's E5 "trace-derived f_ach from Preparing bandwidth" statistic is degenerate
  (documented deviation: probe-derived factor is used instead);
- this differs from the WS6 sample trace (340 GB explicit H2D, c2c sweep +58.9% at
  0.25×) — same commit, so the sample trace was likely captured with a different
  transfer path/config. Walls differ too (q21 17.4 s here vs 4.38 s there; peak pools
  identical).

## 3. Per-knob results

### 3.1 E1 — `gpu_compute` via MPS ACTIVE_THREAD_PERCENTAGE (flagship)

Mechanism checks: in-session fma probe scaled exactly linearly (pct=75 → 0.7498;
gate ±1%) → `f_ach = pct/100` as calibrated by WS5. saxpy (HBM-bound) retained 0.868
at pct=75 — the conflation WS5 warned about, now visible in-session.

#### E1-75 (f_ach = 0.75, 4 iters, anchor = P3)

| q | real base (ms) | real (ms) | real × | sim (ms) | sim × | E_ratio | abs err |
|---|---|---|---|---|---|---|---|
| 1 | 7859 | 8462 | 1.077 | 10444 | 1.332 | **+23.7%** | +23.4% |
| 9 | 12592 | 13048 | 1.036 | 16593 | 1.328 | **+28.2%** | +27.2% |
| 13 | 5609 | 5955 | 1.062 | 7248 | 1.329 | **+25.2%** | +21.7% |
| 19 | 8404 | 8611 | 1.025 | 11208 | 1.328 | **+29.6%** | +30.2% |
| 21 | 17581 | 18579 | 1.057 | 23197 | 1.330 | **+25.8%** | +24.9% |

**FAIL vs the ≤10% floor — uniformly and structurally.** Cutting the SM ceiling by 25%
slows real queries only 2.5–7.7%; v0 predicts +33% for everything (Computing ÷ f). The
queries behave much closer to the saxpy probe (memory/latency-bound) than the fma probe.

#### E1-50 (f_ach = 0.4998 by fma probe; 4 iters)

| q | real × | sim × | E_ratio | abs err |
|---|---|---|---|---|
| 1 | 1.237 | 1.995 | **+61.3%** | +61.0% |
| 9 | 1.106 | 1.989 | **+79.8%** | +78.4% |
| 13 | 1.180 | 1.991 | **+68.8%** | +64.0% |
| 19 | 1.083 | 1.989 | **+83.6%** | +84.4% |
| 21 | 1.168 | 1.992 | **+70.6%** | +69.3% |

#### E1-25 (f_ach = 0.2499; 3 iters — deviation: 1 warmup + 2 measured, to bound session length)

| q | real × | sim × | E_ratio | abs err |
|---|---|---|---|---|
| 1 | 1.556 | 3.985 | **+156%** | +156% |
| 9 | 1.272 | 3.973 | **+212%** | +210% |
| 13 | 1.381 | 3.975 | **+188%** | +180% |
| 19 | 1.321 | 3.971 | **+201%** | +202% |
| 21 | 1.480 | 3.981 | **+169%** | +167% |

#### E1 knob verdict

- **FAIL at every factor** (floors were 10/10/15%): median E_ratio +26% @0.75, +71% @0.50,
  +188% @0.25. Direction uniformly pessimistic, exactly the G4 conflation caveat the
  design doc warns about — but ~4–8× larger than any "conflation" reading of the knob
  would suggest: at pct=25 the *entire query set* retains ≥64% of its speed while the
  fma ceiling drops to exactly 25%. Sirius Computing spans on this box are dominated by
  work that does not scale with the SM-thread ceiling (memory/latency/launch-bound
  kernels + host-side glue inside the spans).
- **Structure is intact everywhere**: binding constraint (executor_threads) matches
  prediction↔degraded-replay 15/15; per-pipeline busy-share error ≤1.2 pp (top-5
  pipelines); thread-busy ~99% on both sides; real walls perfectly monotone in f;
  per-query Spearman(real, sim) = 1.00. The simulator's *flow* model survives heavy
  compute degradation; only the span-scaling law is wrong.
- Probe pair per row (fma/saxpy retained): 0.750/0.868, 0.500/0.657, 0.250/0.364 —
  the two roofline extremes; every real query sits *above* even the saxpy line.
- **E6 (coupling)**: per-query E1-25 error vs the H2D-bytes-per-wall proxy shows **no
  usable correlation** (Pearson r = −0.23, slope −0.5%/GB·s⁻¹). The plan's hypothesis
  (error grows with membw share) is not supported at query granularity — the insensitivity
  to SM count is not predicted by transfer intensity; a per-kernel (nsys) classification
  (G4) is required, not a per-query proxy.

### 3.2 E2 — `gpu_mem_capacity` via Sirius pool YAML knob

f_ach read from `MemoryOperating.capacity_bytes` in each degraded trace ÷ B1's
229.053 GB. E2-hi: 115.2 GB → f_ach 0.5029 (=128 GB × 0.9).

#### E2-hi (pool 115.2 GB, above every working set) — null-match row

Real walls (q21 med 17.36 s vs baseline 17.45 s; all others within σ_q), **zero**
real `Downgrading` events, sim predicts Δ=0 with zero forced admissions. **PASS.**

#### E2-mid (usage_limit 64 GB → pool 57.6 GB, f_ach 0.2514; 3 iters) — split row

Trace-verified capacity 57 600 000 000 B. Real Downgrading events: **41 154, all three
q21 iterations (13.5–13.8 k each); zero for q1/q9/q13/q19** (timestamp attribution).

| q | real × | sim × | E_ratio | verdict |
|---|---|---|---|---|
| 1 | 1.005 | 1.000 | −0.5% | match ✓ (≤3σ_q) |
| 9 | 1.001 | 1.000 | −0.1% | match ✓ — the *marginal* query (peak 54.7 GiB vs pool 53.6 GiB): both sim and reality say it fits |
| 13 | 0.999 | 1.000 | +0.1% | match ✓ |
| 19 | 0.997 | 1.000 | +0.3% | match ✓ |
| 21 | **11.242** | 2.005 | **−82.2%** | boundary quantify: sim fa=448 vs real 13.7 k Downgrading/iter |

- **Structural agreement 5/5**: sim `forced_admissions>0 ⟺` real `Downgrading>0` for
  every query (target was ≥4/5).
- **The headline G5 datum**: the plan (and the sim design doc) expected v0's no-spill
  blocking cliff to be *pessimistic* — reality would spill gracefully and run faster
  than predicted. **The opposite is true at SF1000**: real downgrade-and-recompute cost
  ×11.2 where the blocking model says ×2.0. v0's capacity knob under-prices memory
  pressure by ~5.6× at this point. A capacity knob without a calibrated spill model is
  not just imprecise — it is misleading in the *optimistic* direction.
- Replay fidelity collapses under spill: selfcheck of the degraded trace's q21
  iterations is +2743% (vs ≤0.36% everywhere else in this campaign) — v0 cannot even
  *replay* a downgrade-heavy execution (downgraded tasks execute on paths whose spans
  the FSM does not capture the way the engine re-runs them). q1/q9/q13/q19 in the same
  trace still replay at ≤0.3%.

(E2-lo, E4/E5, E3 sections land as rows complete)

## Runs manifest snapshot

See `experiments/runs.csv`. Aborted rows: E1-50 attempt 1 (foreign process took 243 GB
VRAM mid-row → Sirius pool OOM at init; row re-run after idle gate).
