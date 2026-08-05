# WS8 — Validation Results: does the v0 simulator predict physically-degraded reality?

**Status: COMPLETE, 2026-08-05.** All 21 matrix rows of
[validation-plan.md](validation-plan.md) executed or explicitly accounted for
(one query dropped by the plan's runtime guard; zero rows silently skipped).
Box pmgb300ws-0163 (GB300); binary `/localhome/local-faramburu/repos/sirius/build/release/duckdb`
(dev@b77aa438, sha256 in `experiments/env/binary_sha256.txt`); dataset SF1000 parquet.
Raw data: `tools/hwsim/experiments/` (`runs.csv` = manifest; `analysis/` = summary
CSVs/metrics; `traces/`, `throttle_logs/`, `env/` gitignored). ~150 min GPU-busy total.

## 1. Executive verdict table

| Knob | Rows | Validated? | Slowdown-ratio error (E_ratio) | Recommendation |
|---|---|---|---|---|
| `gpu_compute` (MPS pct 75/50/25) | E1-75/50/25 | **NO — fails at every factor** (v0). **FIXED by the physics join — see §7**: median −1.5/−3.7/−6.4% | median **+26% / +71% / +188%**, uniformly pessimistic; structure (binding, pipeline shares ≤1.2 pp, monotonicity, per-query rank ρ=1.0) intact | Use `--physics` (§7) for quantitative what-ifs; the v0 path stays for structure-only questions |
| `gpu_mem_capacity` (pool YAML 0.50/0.25/0.15) | E2-hi/mid/lo | **YES above the spill threshold; NO below it — and in the *opposite* direction to design assumptions** | unpressured queries: ≤0.5% (12/12 null matches); pressured: **−82%** (q21@mid, real ×11.2 vs sim ×2.0), **−58%** (q9@lo, real ×55.2 vs sim ×23.1) — sim *under*-prices spill 2.4–5.6× | Capacity knob is trustworthy for "does it still fit" questions; forbid/flag predictions past the spill knee until a G5 spill model is calibrated (captures now exist) |
| `io_bandwidth` (O_DIRECT injector, cold) | B2, E3-84/60/36 | **YES (best knob in the matrix)** | scan-bound q6/q19: **±5% at every depth**; decode-heavy q1: +19–20% pessimistic at deep throttle | G1 instrumentation **deprioritized** per §7.4 rule (errors ≪ the 50% promotion trigger); optional cheap fix for decode-heavy queries (§5 #4) |
| `gpu_mem_bandwidth` (HBM CE eater) | E4-73/45 | **NO (quantify-G4 row)** (v0). **FIXED by the physics join — see §7**: median −2.8/−5.3% | median **+24% @0.76, +74% @0.48**, all-pessimistic; rank ρ = 0.9–1.0 | Same root cause and same fix as `gpu_compute` (they share the v0 scaling law) |
| `c2c_bandwidth` (C2C H2D eater + co-sets) | E5-70/49 | **UNTESTABLE on this box** (formally passes: ≤+7.6%). **Now TESTED and VALIDATED on a link-bound lane — see §7.3**: physics −9.1/−1.5% where real slowdown is ×1.83/×1.53 | null–null agreement: real ≤3% even at the physical floor (c2c 0.50 + dram 0.57); sim c2c knob touches ~0.6% of wall (zero-length Preparing spans) | Physics path + wire-capacity correction validated (§7.3); v0/physics on coherent-C2C traces now WARN the knob is inert (§7.4) |
| plan-delta under degradation | B3, E7 | **INCONCLUSIVE (null result)** | real cross-plan deltas ±1.0–1.6% ≈ noise; sim −0.6/−0.9%; sign 1/2, magnitudes within 15 pp trivially | `fuse_merge_pipelines=false` changes structure (4/5 queries +1–2 pipelines) but not walls; retest with a lever that moves walls |
| self-consistency (G-SELF) | P1/B1/B3 + every degraded trace | **YES, exceptional** | median 0.02–0.10%, worst 0.36% across 9 traced sessions — *including under MPS, eaters and injectors*; sole exception: spill-heavy executions (+2743%) | Replay core is solid; the fidelity frontier is span *scaling laws* and *spill*, not flow |

**Bottom line:** v0's discrete-event flow engine (threads, admission, dependencies,
structure) replays and degrades faithfully; its **per-span scaling laws are the problem**.
The two compute-side knobs over-predict degradation 4–8× because Sirius Computing spans
on GB300 are mostly *not* SM/HBM-throughput-bound; the capacity knob under-predicts
below the spill knee because real downgrade costs ~5× more than blocked waiting; and the
io knob — expected to be the weakest — turned out the most accurate because GPU_SCAN
spans genuinely contain device-read time on this box.

## 2. Noise floors (P1/P2/P3) — measured first; every error above is read against them

### 2.1 P1 run-to-run variance (2 sessions × 3 iters, warm, quent on)

All 9 pool queries passed the ≤5% stability rule (none of the known 13–28% SF1000
swingers appeared under this default, untuned config):

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

- **σ_q = 0.16–1.36%**; 3σ_q < 4.1% everywhere → every §3 verdict is governed by its
  stated floor, not by noise. Final throttle set: **q1, q9, q13, q19, q21**; io set
  q1/q6/q19 (B2 cold repeatability 0.1/1.3/2.6% ≤ the 8% gate).
- **P2 quent overhead: −1.3%…0.0%** (below noise on multi-second walls — far under the
  +1–5% expectation; no absolute-wall caveats required).
- **P3 MPS-daemon null (pct=100)**: +0.4/+0.5/−0.3/+0.9% for q1/q9/q19/q21; **q13 +2.9%**
  (real daemon overhead, visible on the shortest query) → all E1 ratios use P3, not B1,
  as the real baseline (common-mode cancellation). Probe anchors: fma 52 163 GFLOP/s,
  saxpy 4 788 GB/s.

### 2.2 G-SELF (fresh binary + fresh traces) — PASS

| session | n | median err% | worst err% |
|---|---|---|---|
| P1a / P1b | 27 / 27 | 0.05 / 0.03 | 0.36 / 0.26 |
| B1 / B3(variant plan) | 20 / 20 | 0.05 / 0.04 | 0.12 / 0.15 |
| P3(MPS100) / E1-75 / E1-50 / E1-25 | 15/20/20/15 | 0.02–0.06 | ≤0.70 |
| E4-73 / E4-45 / E5-70 / E5-49 | 20/20/20/8 | 0.03–0.05 | ≤0.20 |
| E2-mid (spill-free queries / q21 spill iters) | 15 | 0.10 | **+2743%** on q21 |

Better than WS6's sample-trace result (0.17%/1.2%) — the dirty-build caveat is moot.
The one systematic failure: **spill-heavy executions cannot be replayed** (see §3.2).

### 2.3 Baseline trace T (B1) and the discovery that reframed two knobs

B1 walls/peak pools (med iters 2–4): q1 7.843 s/13.7 GiB, q9 12.495 s/54.7 GiB,
q13 5.452 s/21.2 GiB, q19 8.441 s/21.5 GiB, q21 17.447 s/76.3 GiB — peak pools match the
WS6 sample trace within ~1 GiB, so the matrix's E2 capacity points stood as designed.

**Discovery 1 (pre-registered before E4/E5): fresh traces on this box have ~zero-length
transfer spans.** 27 913 SOURCE→GPU Preparing→Computing spans total **1.25 s of ~207 s**
of session wall (median 22 µs for ~455 MB ⇒ nominal ~20 TB/s). hwsim derives a channel
"capacity" of 165 280 GB/s (vs 709.6 GB/s in the WS6 sample trace). Data movement hides
inside Computing spans (coherent-C2C zero-copy / full overlap; nsys doc §5.2, G4) → the
sim is structurally **c2c-blind** on these traces. Same commit as the sample trace, so
that capture's explicit 340 GB H2D came from a different path/config; walls also differ
~3–4× (q21 17.4 s vs 4.38 s) while peak pools are identical.

**Discovery 2 (found in the cold block): Sirius streams most parquet bytes from NVMe on
*every* run here.** iostat shows ~5.1 GB/s device reads during "warm" runs; page cache
never retains the dataset (buff/cache peaked ~101 GB « 265 GB); cold ≈ warm walls at ±2%.
This is why `io_bandwidth` works (§3.5) — GPU_SCAN spans *contain* real device-read time
— and it removes the planned warm/cold confound entirely.

## 3. Per-knob results

### 3.1 E1 — `gpu_compute` via private-MPS ACTIVE_THREAD_PERCENTAGE (flagship)

`f_ach = pct/100` confirmed in-session every row (fma probe: 0.7498 / 0.4998 / 0.2499 —
exact). saxpy (HBM-bound) retained 0.868 / 0.657 / 0.364 — the two roofline extremes.
Iterations: 4 (75/50), 3 at pct=25 (documented deviation: bound session length; 1 warmup
+ 2 measured). Real baseline = P3.

| q | E1-75 real× | sim× | E_ratio | E1-50 real× | sim× | E_ratio | E1-25 real× | sim× | E_ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.077 | 1.332 | **+23.7%** | 1.237 | 1.995 | **+61.3%** | 1.556 | 3.985 | **+156%** |
| 9 | 1.036 | 1.328 | **+28.2%** | 1.106 | 1.989 | **+79.8%** | 1.272 | 3.973 | **+212%** |
| 13 | 1.062 | 1.329 | **+25.2%** | 1.180 | 1.991 | **+68.8%** | 1.381 | 3.975 | **+188%** |
| 19 | 1.025 | 1.328 | **+29.6%** | 1.083 | 1.989 | **+83.6%** | 1.321 | 3.971 | **+201%** |
| 21 | 1.057 | 1.330 | **+25.8%** | 1.168 | 1.992 | **+70.6%** | 1.480 | 3.981 | **+169%** |

**FAIL at every factor** (floors were 10/10/15%). Direction uniformly pessimistic — the
G4 conflation the design doc warns about, but ~4–8× larger than a "conflation" reading
suggests: with the SM-thread ceiling at exactly 25%, the *whole query set* retains
64–79% of its speed — above even the pure-HBM saxpy line (36%). Sirius Computing spans
on this box are dominated by work that does not scale with SM count (low-occupancy /
latency-bound kernels, launch overhead, host glue inside spans).

**Structure survives everywhere**: binding constraint matches (executor_threads) 15/15;
top-5 pipeline busy-share error ≤1.2 pp; thread-busy ~99% both sides; real walls
perfectly monotone in f; per-query Spearman(real, sim) = 1.00; cross-query wall ranking
ρ = 1.0/0.9/0.9 (q1↔q19 swap at deeper factors — real q1 degrades faster than its
baseline rank; sim keeps baseline order).

**E6 (coupling, no extra runs):** regressing per-query E1-25 error on the membw proxy
(B1 H2D GB per wall-second: q1 32.7, q19 27.0, q9 22.2, q21 19.0, q13 3.5) gives
**r = −0.23** — no usable correlation. The plan's hypothesis (error grows with membw
share) fails at query granularity; SM-insensitivity is a *kernel-level* property (G4
needs the nsys join, not a per-query proxy).

### 3.2 E2 — `gpu_mem_capacity` via Sirius pool YAML

f_ach from `MemoryOperating.capacity_bytes` ÷ B1's 229.053 GB (trace-verified per row:
115.2 / 57.6 / 34.2 GB → 0.5029 / 0.2514 / 0.1493).

| row | q | real × | sim × | E_ratio | real Downgrading | sim fa | verdict |
|---|---|---|---|---|---|---|---|
| hi (all ≥1.4× peaks) | all 5 | 0.995–1.005 | 1.000 | ≤0.5% | 0 | 0 | **null match 5/5** |
| mid | 1,9,13,19 | 0.997–1.005 | 1.000 | ≤0.5% | 0 | 0 | null match 4/4 (incl. *marginal* q9: peak 54.7 GiB vs pool 53.6 GiB — both sides say "fits") |
| mid | 21 | **11.242** | 2.005 | **−82.2%** | 13.5–13.8 k/iter | 448 | boundary quantify |
| lo | 1,13,19 | 0.999–1.004 | 1.000 | ≤0.4% | 0 | 0 | null match 3/3 at 15% capacity |
| lo | 9 | **55.19** (689.9 s) | 23.12 | **−58.1%** | 40 562 (iter1) | 1355 | boundary quantify |
| lo | 21 | guard-dropped | 3.03 | — | — | 1157 | see deviations |

- **Structural agreement 9/9** on executed queries: sim `forced_admissions>0 ⟺` real
  `Downgrading>0` (target ≥4/5). Downgrading attribution by query-window timestamps.
- **The headline G5 datum — the expected error direction is inverted.** The design doc
  calls the no-spill blocking cliff "honestly pessimistic" (real engine would spill and
  run *faster* than predicted). At SF1000 the opposite holds: real downgrade-and-recompute
  costs ×11.2 where blocking says ×2.0, and ×55 where it says ×23. During q9-lo the GPU
  sat at 8% util with the pool pegged — the engine thrashes rather than degrades. Per
  §7.4's G5 rule (framed for the other direction): the sim's cliff is not an upper bound;
  **v1 must not ship sub-threshold capacity predictions without a spill model**.
- Replay collapses under spill: q21-mid iterations self-check at **+2743%** (vs ≤0.36%
  everywhere else); unpressured queries in the same trace replay at ≤0.3%. v0 cannot
  even replay downgraded executions — G5 events are needed for replay, not only pricing.
- The E2-mid/lo traces (13.7 k & 40.6 k Downgrading events) are the calibration captures
  WS6 §10 requested — preserved under `experiments/traces/E2-*`.

### 3.3 E4 — `gpu_mem_bandwidth` via HBM copy-engine eater (quantify G4)

In-session victim triplets (authoritative): E4-73 → hbm **0.762**, dram 1.005, c2c 0.998;
E4-45 → hbm **0.480**, dram 1.011, c2c 0.996. Both co-sets ≈1.0 at sub-saturation →
single-knob sim runs. (Side finding: the CE eater degrades the SM streaming victim like
the SM engine at equal rate — WS4's unverified equivalence confirmed for memory-bound
victims; the provisional CE cross-talk cells did not need re-measurement, the in-session
probes measured ~1.0 directly.) Eater held targets exactly (500.3 @duty 0.17; 986.6
@duty 0.93).

| q | E4-73 real× | sim× | E_ratio | E4-45 real× | sim× | E_ratio |
|---|---|---|---|---|---|---|
| 1 | 1.198 | 1.311 | +9.4% | 1.232 | 2.078 | +68.7% |
| 9 | 1.047 | 1.307 | +24.8% | 1.112 | 2.071 | +86.3% |
| 13 | 1.107 | 1.308 | +18.2% | 1.193 | 2.073 | +73.8% |
| 19 | 1.052 | 1.307 | +24.3% | 1.189 | 2.071 | +74.2% |
| 21 | 1.045 | 1.309 | +25.2% | 1.101 | 2.075 | +88.5% |

Median E_ratio **+24% @0.76, +74% @0.48** — same failure law as E1 (v0 prices whole
spans at the degraded rate; only a fraction of real span time is HBM-bound). Per-query
spread at one factor: 16–20 pp (below the plan's 25 pp trigger, but E1+E4 jointly make
the per-kernel G4 split the top backlog item). q1 — the most scan/decode-intensive —
is the *most* HBM-sensitive in reality and the best-predicted. Rank ρ(sim, real):
1.0 per query across factors; 0.9–1.0 across queries; degraded traces self-check ≤0.20%.

### 3.4 E5 — `c2c_bandwidth` via C2C H2D eater (+ measured co-sets)

Probes: E5-70 → c2c **0.7105**, hbm 0.961 (co-set), dram 0.827 (v0 no-op — documented
optimistic residual; WS4's first-order C2C⇄DRAM coupling confirmed in-session).
E5-49 (floor) → c2c **0.5031**, hbm 0.914 (co-set), dram 0.572.

Deviation: the plan's trace-derived f_ach (Preparing bandwidth ratio) is **degenerate**
on this box (§2.3 discovery 1) — probe-derived factors used; that degeneracy is itself
the E5 finding.

| row | q | real × | sim × | E_ratio |
|---|---|---|---|---|
| E5-70 | 1/9/13/19/21 | 1.003–1.015 | 1.038–1.041 | **+2.4…+3.6%** |
| E5-49 | 9 | 1.030 | 1.091 | +6.0% |
| E5-49 | 21 | 1.016 | 1.093 | +7.6% |

Formally PASS (floor 15%), but it is a **null–null agreement**: even at the deepest
physically-achievable degradation (c2c 0.50 with dram dragged to 0.57), the heaviest
transfer queries lose ≤3% — Sirius's C2C demand (≈25–40 GB/s average) never approaches
the ~170–240 GB/s still available, so the resource binds neither reality nor the sim
(whose prediction is entirely the hbm co-set; the c2c knob itself touched ~0.6% of wall).
The v0 c2c mechanics were **not exercised** — do not extrapolate this PASS to boxes or
configs with explicit staged transfers (like the WS6 sample capture).

### 3.5 E3 — `io_bandwidth` via O_DIRECT read injector (cold; quantify G1)

Same-session re-probe: B = 6.411 GB/s, C_mix = 7.110 (WS3 law → nominal targets
1.72/3.26/4.80 GB/s); `f_ach = (C_mix − R_achieved)/B` per (row, query) from injector
SUMMARY (held exactly at 1.7; undershot 6% at 3.26, 13% at 4.8 — device-limited, folded
into f_ach). Sim input: **B2 cold trace** of the same query (median of both sessions),
never the warm B1.

| row | q | f_ach | real × | sim × | E_ratio |
|---|---|---|---|---|---|
| E3-84 | 1 | 0.841 | 1.111 | 1.162 | +4.6% |
| E3-84 | 6 | 0.844 | 1.231 | 1.179 | −4.2% |
| E3-84 | 19 | 0.847 | 1.216 | 1.162 | −4.5% |
| E3-60 | 1 | 0.631 | 1.264 | 1.499 | +18.7% |
| E3-60 | 6 | 0.670 | 1.475 | 1.477 | **+0.1%** |
| E3-60 | 19 | 0.658 | 1.496 | 1.473 | −1.6% |
| E3-36 | 1 | 0.459 | 1.666 | 2.006 | +20.4% |
| E3-36 | 6 | 0.493 | 1.944 | 1.998 | +2.8% |
| E3-36 | 19 | 0.464 | 2.026 | 2.055 | +1.4% |

**G1 decision (plan §7.4): |E_ratio| ≤ 19% at v≈0.6 on scan-bound queries → G1
split-level instrumentation DEPRIORITIZED below G2/G4.** The knob is the matrix's most
accurate: q6/q19 track ×≈1/f within ±5% at every depth (they are genuinely
device-throughput-bound — §2.3 discovery 2); q1 (≈20 s user-CPU decode per run) drifts
pessimistic to +20% because the knob scales the decode share of the span too — the
documented G1 mechanism, now bounded at ≤20%, not the feared ≥50%. Real walls perfectly
monotone in f, 3/3 queries.

### 3.6 B3/E7 — plan-change stretch (cross-plan delta under degradation)

B3 (`SET fuse_merge_pipelines=false`, 5×4 warm): lever verified — pipeline counts q1 6→7,
q9 36→37, q13 14→16, q21 52→54 (4/5 changed; ≥2 required); baseline walls within noise of
B1; selfcheck 0.04%/0.15%. E7 = q13,q21 ×4 under MPS pct=50 + preamble (fma probe 0.4999).

| q | real Δ (E7 vs E1-50) | sim Δ (B3@0.5 vs B1@0.5) | sign | |Δ| within 15 pp |
|---|---|---|---|---|
| 13 | **+0.98%** | −0.91% | ✗ | ✓ |
| 21 | **−1.61%** | −0.57% | ✓ | ✓ |

**Inconclusive-null**: the lever changes structure but moves walls by ~1% ≈ the
cross-session noise floor, so the sign test has no power here. The claim "v0 supports
per-plan traces and cross-plan deltas" is neither confirmed nor refuted; retest needs a
lever with ≥5% wall effect (e.g. `max_broadcast_join_size=0` on join-heavy queries).

## 4. Methodology deviations (complete list)

1. **E1-25 ran 3 iterations** (1 warmup + 2 measured) instead of 4 — session-length
   bound; σ_q ≤0.4% makes 2 measured iterations sufficient.
2. **E2-lo split into three sessions** (q1/q13/q19; q9; q21) so the runtime guard kills
   one query without destroying the row. **q21 dropped by the guard** (projected >15
   min/iter from the mid-row ×11.2); q9's guard extended 5→12 min under owner-granted
   exclusive access, iteration 2 stopped after iteration 1 completed (689.9 s wall
   recorded; 40 562 Downgrading events in the partial trace used only for event counts).
3. **E5 trace-derived f_ach abandoned** — statistic degenerate on this box (Preparing
   spans ≈22 µs for ~455 MB); probe-derived factors used and the degeneracy reported as
   the c2c-blindness finding.
4. **E1 ratios anchored to P3** (MPS pct=100) rather than B1, cancelling the measured
   +0.4–2.9% daemon overhead (plan already designated P3 as the E1 anchor).
5. **Shared-GPU protocol changed mid-campaign by the owner**: initial rows ran under the
   full idle-window gate (one E1-50 attempt aborted when a foreign process took 243 GB
   VRAM mid-row — row deleted and re-run); after exclusive access was granted the
   remaining rows ran back-to-back with sanity gates only.
6. **B3 first attempt failed on a wrapper path bug** (the plan's `run_with_preamble.sh`
   copy resolves `PROJECT_DIR` relative to its own location); fixed by pinning
   `PROJECT_DIR` to the main repo, re-run cleanly. Stock runner never modified.
7. **No re-warm pass before the warm stretch rows after the cold block** — justified by
   discovery 2 (cold ≈ warm on this box; B3 baseline walls confirm: within noise of B1).
8. **mid-campaign sim-code change by another agent** (new `physics/` subpackage +
   `cli.py`): engine/knobs/build untouched; bit-exact reproduction of a stored E1-50
   prediction verified post-change (`analysis/` spot-check), so all sim outputs are
   consistent.

## 5. What to fix next (ranked by measured cost)

1. **G4 per-kernel compute/HBM split (nsys structural join)** — the flagship
   `gpu_compute` knob over-predicts by +26%→+188% (f 0.75→0.25) and `gpu_mem_bandwidth`
   by +24%→+74%; E6 proves no query-level proxy rescues them (r = −0.23). Until the
   join lands, an *empirical stopgap* is justified by the data: per-operator (or even
   global) span-sensitivity coefficients — real slowdown ≈ 1 + s·(1/f − 1) with
   measured s ≈ 0.15–0.25 for this workload — would collapse the E1 error by ~4×.
2. **G5 spill model + downgrade telemetry** — capacity predictions below the spill knee
   are wrong in the *dangerous* direction (real ×11–55 vs predicted ×2–23; under-priced
   2.4–5.6×). Calibration material now exists (`traces/E2-mid`, `traces/E2-lo-q9`:
   13.7 k/40.6 k Downgrading events with walls). Also needed for *replay*: spill-heavy
   traces self-check at +2743%.
3. **Transfer-window instrumentation on coherent-C2C boxes** — Preparing spans of ~22 µs
   for ~455 MB make the c2c knob silently inert (touches 0.6% of wall). Either emit real
   copy/zero-copy windows (G4/nsys §5.2) or make the knob refuse/warn when channel
   "capacity" is unphysical (>10 TB/s), so users on GB300-class captures aren't handed
   a confident no-op prediction.
4. **(Deprioritized) G1 read/decode split inside GPU_SCAN** — the knob is already ±5% on
   scan-bound queries; a per-query decode-fraction constant (one cold calibration run)
   would fix the remaining +20% on decode-heavy q1. Below G2/G4 in priority per the
   plan's own decision rule.
5. **Small structural items**: cross-query rank inversions at deep factors (ρ 0.9 —
   q1/q19 swap; would be fixed by #1 automatically); MPS-daemon overhead (+2.9% on a
   5.5 s query) argues for matching-mechanism baselines in any future MPS rows (as done
   here via P3).

## 6. Runs manifest

`experiments/runs.csv` (one line per session; aborted/partial rows retained with
reasons). 21/21 matrix rows executed or accounted: E5-49/B3/E7 stretch rows all ran;
E2-lo-q21 is the single guard-sanctioned drop. Environment snapshots pre/post every row
in `experiments/env/`; mechanism evidence (probe outputs, eater/injector CSVs, iostat
sidecars, eviction transcripts) in `experiments/throttle_logs/<ROW>/`.

## 7. Physics re-score (WS12, 2026-08-05) — the failed knobs after the nsys join

**Status: COMPLETE.** First real nsys captures taken (WS10's join had only been
fixture-verified), the two failed knob families re-scored against WS8's *actual
measured* degraded rows, and the c2c knob validated on a purpose-built link-bound
lane. Rows R1/R2/R2n/R3 in `runs.csv`; sim CSVs `analysis/E*P_sim.csv`,
`analysis/R3{v0,phys}_sim.csv`. Same binary as the whole campaign
(main repo dev@b77aa438, sha verified).

### 7.1 First-real-capture checklist (nsys-join.md §7), R1 = paired quent+nsys of the 5 campaign queries

1. **Ingest clean** — `reader_notes: []` on both captures (R1 and R2n): no schema
   drift against nsys 2025.6.3.
2. **Kernel attribution** — R1: 99.9% by count, **100.0% of kernel time** (240,641
   kernels); R2n: 100/100. (R1 memcpy-time attribution is 41% — the NVMe lane's
   small staging copies often launch outside task NVTX ranges; their absolute time
   is trivial.)
3. **GPU_METRICS semantics** — **unverifiable on this box**: Tier B capture fails
   with `ERR_NVGPUCTRPERM` (GPU perf counters restricted to admin). Classification
   ran on name priors alone: 23.0% of R1 kernel time classified (6.2% R2n); the
   rest scales by the v0 conflated rule. ncu spot-checks are blocked by the same
   permission — not attempted (see gaps).
4. **Curve sanity** — R1 buckets monotone (H2D pinned pooled 54.6→75.2→113.1→138.8
   GB/s up to its 1 MB max — all R1 H2D copies are small staging chunks); R2n
   large buckets (64–256 MB): **338.9–347.4 GB/s pooled H2D** — inside the
   measured 340–380 C2C line; R1 D2D 256 MB bucket **464.7 GB/s** ≈ the
   flag-capped ~470. Wire-side peak aggregates: 214 GB/s H2D (R1), **382.3 GB/s**
   H2D (R2n).
5. **Clock fit** — same-run detected on all 9 windows; R1 slopes 1.000004
   (±4 ppm), rms 0.5–29.7 µs (< 100 µs everywhere).
6. **knobs=1 identity** — R1 trace, physics vs traced wall: −0.11/−0.04/−0.14/
   +0.01/−0.04% (5/5 within ±1.2%); R2n: −0.13/−1.34% (q14 marginally at the
   envelope edge).
7. **Split-prediction divergence** — see 7.2: physics predictions diverge from v0
   exactly as the kernel/host mix dictates.

**Capture overhead**: R1 walls vs B1 medians: q1 +6.3%, q9 +2.7%, q13 +2.9%,
q19 +6.4%, q21 +1.4% — Tier A in the expected +1–5% band (slightly above on the
launch-densest queries). Physics quantities are per-event device durations and
byte counts (overhead-clean); only span *fractions* inherit host-side skew.

**The one-number explanation of the E1/E4 failure** (from the R1 decomposition):
Sirius spans on this NVMe-streaming lane are **85–93% host time** (span minus
GPU-busy: q1 85.4%, q9 93.3%, q13 84.9%, q19 92.7%, q21 92.1%); GPU-kernel-busy
time is 5–15% of traced busy time. v0 scaled the whole span by 1/f; reality only
scales the small GPU share. No query-level proxy could see this (E6's r = −0.23);
the per-span nsys decomposition does.

### 7.2 Before/after: v0 error vs physics error per (query, factor)

Method identical to §3 (same P3/B1 real anchors, same iters, `row_metrics.py`);
physics rows simulate the same B1 traces with `--physics physics_R1.json`.
E_ratio = sim-slowdown / real-slowdown − 1; **v0 → physics**:

| q | E1-75 | E1-50 | E1-25 | E4-73 | E4-45 |
|---|---|---|---|---|---|
| 1 | +23.7% → **−2.6%** | +61.3% → **−7.0%** | +156% → **−6.4%** | +9.4% → **−12.6%** | +68.7% → **−5.3%** |
| 9 | +28.2% → **−1.5%** | +79.8% → **−3.7%** | +212% → **−5.4%** | +24.8% → **−2.4%** | +86.3% → **−3.4%** |
| 13 | +25.2% → **−0.4%** | +68.8% → **−0.8%** | +188% → **+9.6%** | +18.2% → **−6.2%** | +73.8% → **−5.5%** |
| 19 | +29.6% → **−0.4%** | +83.6% → **−1.4%** | +201% → **−8.0%** | +24.3% → **−2.8%** | +74.2% → **−9.3%** |
| 21 | +25.8% → **−3.4%** | +70.6% → **−8.0%** | +169% → **−16.5%** | +25.2% → **−2.1%** | +88.5% → **−1.6%** |
| **median** | **+25.8 → −1.5** | **+70.6 → −3.7** | **+188 → −6.4** | **+24.3 → −2.8** | **+74.2 → −5.3** |

**24/25 cells within ±15%** (sole exception q21@E1-25: −16.5%); direction flips
from uniformly pessimistic to mildly optimistic — consistent with the residual
name-prior share and MPS effects the split cannot see (launch/host overhead does
degrade slightly under a 25% SM ceiling). Join coverage 99.8–100% of busy time on
every row; the classification weakness (77% of kernel time unclassified) turned
out **not** to be the binding error term — the host/GPU-busy decomposition did
the work. The §5 #1 "empirical stopgap coefficient" is obsolete.

### 7.3 c2c re-test on a link-bound lane (the E5 non-test, done properly)

**Scenario construction** (option "make bytes cross the wire"): `pin_table` the
q6+q14 column union into pinned **HOST** memory (SF1000; lineitem 5 cols + part
2 cols, ~250 GB decoded) with `pipeline.num_threads: 8`, so every scan does
explicit staged H2D. Link-boundness verified from the paired capture's memcpy
table (R2n): **sustained aggregate H2D 335–339 GB/s while active** (≈ the 340–380
line), link active 56–85% of the query wall, individual copies at 381–382 GB/s.
Quent-side: real Preparing transfer spans reappear (171.0 / 152.8 GB per query,
byte-identical to nsys wire volume — uncompressed lane), channel peak 730 GB/s
(same order as the WS6 sample trace's 709.6, confirming that capture was this
kind of lane).

**Degraded run** (R3): WS4 c2c eater at its floor (`--engine h2d --chunk-mb 256
--buf-mb 1024 --gbps max`, achieved 271 GB/s); in-session co-sets measured
c2c **0.5038**, dram **0.5383**, hbm **0.8922**. Real traced slowdowns (iters
2–4 medians vs R2): **q6 ×1.834, q14 ×1.531** — the knob's territory is finally
exercised (E5 had ≤×1.03). Selfcheck on both traces ≤0.93%.

Sim (co-set knobs `c2c=0.5038, cpu_mem=0.5383, gpu_mem=0.8922`) vs real:

| q | real × | v0 sim × (E) | physics, uncorrected cap × (E) | physics + wire cap × (E) |
|---|---|---|---|---|
| 6 | 1.834 | 1.421 (**−22.5%**) | 1.148 (−37.4%) | **1.668 (−9.1%)** |
| 14 | 1.531 | 1.616 (+5.5%) | 1.083 (−29.3%) | **1.507 (−1.5%)** |

**Verdict: c2c knob slope VALIDATED** through the physics path with the
wire-capacity correction (both queries within ±10%; floor was 15%). The
correction was the missing piece: the physics transfer-only line-sweep
overlap-inflates channel capacity (derived **3,433 GB/s** vs **382.3 GB/s** of
true wire aggregate — sub-windows are placed contiguously at span starts while
real copies stagger), so an uncorrected knob under-binds. The v0 path on this
lane is usable but noisier (−22%/+6%).

### 7.4 Code changes (physics package; v0 engine untouched)

New `sim/hwsim/physics/sanity.py` + hooks in `physics/integrate.py` /
`physics/cli.py` (13 new unit tests; suite now 122, all green; B1 selfcheck and
a stored E1-50 v0 prediction reproduce bit-identically):

1. **Unphysical-capacity warning** — any host↔GPU channel whose trace-derived
   capacity exceeds 1 TB/s (B1-lane traces derive ~165,000 GB/s) now WARNs, on
   both the v0 and physics paths, whenever `c2c_bandwidth`/`cpu_mem_bandwidth`
   moves: the knob is inert on such traces (fixes §5 #3's "confident no-op").
2. **nsys wire-capacity correction** — `corrected = wire-side peak aggregate ×
   (quent bytes / nsys wire bytes over matched transfers)`, applied as an upper
   bound on the physics channel capacity **only when the link multiplier ≠ 1**
   (identity at knobs=1 stays by-construction; gated on ≥50% byte coverage and
   ≥1 GB matched wire volume, so degenerate coherent-C2C lanes are never
   "corrected", they warn instead).

### 7.5 Remaining gaps (updates to §5)

- §5 #1 (G4) — **CLOSED** by measurement for gpu_compute/gpu_mem_bandwidth at
  SF1000 on this box (7.2). Residual: q21 deep-throttle −16.5%; per-kernel
  classification still priors-only (Tier B + ncu both blocked by
  `ERR_NVGPUCTRPERM`; re-run classification once admin perf-counter access
  exists).
- §5 #3 — **partially closed**: the silent no-op is now loud (7.4), and the knob
  is validated where the link genuinely binds (7.3). Still open as a quent
  instrumentation ask (G-item): **wire-time transfer spans** — emit cudaEvent-
  timed copy windows (start/stop around the `cudaMemcpyAsync` batch in the
  host-pin/staging path, e.g. `cached_databatch_provider`'s H2D slice-copy and
  `native_h2d`) so Preparing spans carry (bytes, wire-ns) even on coherent
  lanes; that would let the v0 path price c2c without a paired nsys capture.
- Physics baseline identity on heavily link-bound traces: −1.3% (R2n q14) — the
  fluid-prep approximation's edge; acceptable but worth a regression test if
  future lanes go deeper.
- R1 memcpy-time attribution 41%: small staging copies launched outside task
  ranges; harmless here (trivial absolute time) but a candidate for a
  scan-manager-thread attribution pass if a lane ever stages large volumes
  outside tasks.
