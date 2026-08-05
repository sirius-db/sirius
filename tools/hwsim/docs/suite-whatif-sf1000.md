# Suite-level hardware what-if — TPC-H SF1000, ship config (WS16, 2026-08-05)

**What this is:** the first full-22-query what-if report from the hwsim simulator,
scored against a fresh suite-wide MPS-50 ground-truth arm before being asked anything
(validation-results.md §9). Every prediction below comes from the G4b physics path
(`simulate --physics`, commit `4f2c57a6`) on the S-BASE baseline trace with per-query
nsys profiles (S-NSYS), unless a row says otherwise.

**Lane:** the lit sf1000-repro ship config — PR #1409 gates on, all-8-table GPU-tier
grouped pins, simpatico plans, `ast_jit`, patched libcudf `LD_PRELOAD`ed, plus the
q10 workaround **`SET enable_compressed_materialization = false`** in every arm
(deviation from engine ship default; without it the GBR flagship silently dies —
validation-results.md §8.1). Binary `522fdfa6…`, box GB300, dataset
`/localhome/local-faramburu/tpch_parquet_sf1000`.

**Baseline anchor:** S-BASE per-query medians (iters 2–3 of 3) sum to **6.654 s —
exactly the PR's banked suite time**. All what-if deltas are relative to per-query
*traced* execution walls (pins excluded; runner walls run ~2–9% higher on the small
queries due to host-side result materialization outside the trace window).

## 1. How much to trust each column (measured, not asserted)

| Knob | Tier | Evidence |
|---|---|---|
| `gpu_compute` ≤ 1 | **VALIDATED (mildly pessimistic)** | Scored against the real 22-query MPS-50 arm: G4b median E_ratio **+6.8%**, suite-aggregate **+6.7%**, 17/22 within ±15% (§9.3). Errors are pessimistic — safe for capacity planning. Exception: **q1**, whose kernels co-run (serialization 0.29–0.33 < 0.9 gate) — its prediction is a **lower bound** (model self-gates off and warns; real q1 slowdown ×1.92 was the suite's worst). |
| `gpu_compute` > 1 | **BAND (order-of-magnitude)** | Emergent waits cannot shrink below traced demand rates (§8.7 limitation 3), and on saturated queries the fluid model produces small *wrong-direction* cells (q9 +4.9% at 1.25×). Report gives [G4b conservative, v0f optimistic] bands. |
| `gpu_mem_bandwidth` 0.5 | **ORDER-OF-MAGNITUDE (conflated)** | 80–96% of suite kernel time is unclassified (GPU_METRICS admin-locked, §8.5) and scales by the conflated min-rule — the 0.5 column is numerically ≈ the `gpu_compute` 0.5 column. Independent HBM validation (§7.2, E4) predates G4b and was on the un-pinned lane. |
| `gpu_mem_bandwidth` 2 | **≈0, artifact-flagged** | Physically capped by the achieved-bw law (×1.27 ceiling, laws.py §3) on the tiny classified-membw share; v0f says 0.0%. The G4b +4.2% suite cell is a **packing artifact** (faster phases re-contend a capacity-1.0 device; q17 +45.8% is spurious) — do not use the positive cells. |
| `c2c_bandwidth` | **VALIDATED-INERT on this lane** | Grid measures ≤+1.2% at 0.25× (suite +0.2%). Trace H2D/D2H volumes ~zero (all tables GPU-pinned); matches §3.4/§7.3: the knob only bites on link-bound (host-pinned/staged) lanes. Honoring the INERT warning: these zeros are *lane properties*, not hardware insensitivity. |
| `io_bandwidth` | **INERT at steady state — grid column is a G1 artifact; do not use** | The S-BASE trace's `io_request` stream is **empty** (zero events: pinned lane never touches NVMe inside query windows). v0 still scales whole GPU_SCAN spans by 1/io (gap G1), producing fictitious +21%/+8% suite cells at 0.25/0.5. The ±5% validated io verdict (§3.5) applies to **un-pinned/cold configs only**. |
| `gpu_mem_capacity` | **DIRECTION + ORDER-OF-MAGNITUDE below the knee (G5 bands ±40%)** | Above the spill knee ≤0.5% (validated §3.2). Below it the G5 model gives correct direction/order, over-warns at shallow pressure. **Extra suite caveat: pins are outside query windows** — the model prices query-time pool pressure only; a real capacity cut would also shrink the pinnable set (re-reads / evictions), so these are optimistic floors. |

## 2. The grid — predicted per-query wall delta vs baseline (%)

G4b physics path, S-BASE trace, per-query S-NSYS profiles. Columns are knob
multipliers (0.5 = half the resource). Cells in *(parens)* are artifact/lower-bound
per §1 and should not be quoted bare.

| q | base ms | gc 0.5 | gc 0.75 | gc 1.25 | gc 2.0 | membw 0.5 | membw 2 | c2c 0.25 | c2c 0.5 | c2c 2 | io 0.25 | io 0.5 | cap 0.5 | cap 0.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 469 | *(+39.7)* | *(+12.8)* | (−6.9) | (−18.3) | *(+39.9)* | (−0.1) | +0.1 | +0.0 | +0.0 | *(+0.1)* | *(+0.0)* | +2.7 | +34.5 |
| 2 | 51 | +18.2 | +4.1 | −1.8 | −3.9 | +28.1 | (−4.9) | +1.2 | +0.4 | +0.0 | *(+25.0)* | *(+8.4)* | +0.0 | +0.0 |
| 3 | 207 | +81.9 | +24.2 | −2.6 | −7.0 | +86.9 | (+1.3) | +0.2 | +0.0 | +0.0 | *(+22.9)* | *(+10.7)* | +0.0 | +5.9 |
| 4 | 168 | +97.9 | +32.8 | −1.4 | −3.6 | +99.0 | (+3.9) | +0.1 | +0.0 | +0.0 | *(+0.1)* | *(+0.0)* | +19.9 | +63.9 |
| 5 | 228 | +95.1 | +31.9 | −1.1 | −5.3 | +97.2 | (+4.8) | +0.1 | +0.0 | +0.0 | *(+0.1)* | *(+0.0)* | +0.0 | +6.2 |
| 6 | 49 | +93.5 | +30.2 | −0.2 | −5.1 | +97.2 | (+5.7) | +0.6 | +0.2 | +0.0 | *(+4.3)* | *(+0.8)* | +13.2 | +55.3 |
| 7 | 227 | +79.3 | +23.2 | −2.2 | −5.4 | +81.7 | (−0.8) | +0.1 | +0.0 | +0.0 | *(+10.8)* | *(+0.0)* | +0.0 | +0.0 |
| 8 | 182 | +103.5 | +36.3 | (+0.8) | −6.6 | +105.1 | (+9.7) | +0.1 | +0.1 | +0.0 | *(+0.3)* | *(+0.1)* | +0.0 | +9.8 |
| 9 | 841 | +98.7 | +37.8 | (+4.9) | −4.5 | +102.3 | (+11.4) | +0.0 | +0.0 | +0.0 | *(+1.6)* | *(+0.7)* | +0.0 | +275.0 |
| 10 | 234 | +86.6 | +26.0 | −4.9 | −12.2 | +87.8 | (−1.0) | +0.2 | +0.0 | +0.0 | *(+13.7)* | *(+4.6)* | +0.0 | +14.4 |
| 11 | 196 | +111.0 | +49.9 | (+8.7) | −2.7 | +112.5 | (+18.7) | +0.1 | +0.0 | +0.0 | *(+12.9)* | *(+4.3)* | +0.0 | +88.6 |
| 12 | 166 | +97.8 | +36.4 | −2.0 | −4.9 | +100.5 | (+1.3) | +0.1 | +0.0 | +0.0 | *(+0.1)* | *(+0.0)* | +0.0 | +25.2 |
| 13 | 766 | +98.8 | +32.7 | −11.9 | −29.8 | +101.6 | (+0.0) | +0.6 | +0.2 | +0.0 | *(+111.5)* | *(+37.0)* | +11.0 | +35.3 |
| 14 | 76 | +93.3 | +32.1 | −11.6 | −29.4 | +97.2 | (+1.8) | +0.4 | +0.1 | +0.0 | *(+4.7)* | *(+0.3)* | +0.0 | +7.8 |
| 15 | 73 | +109.2 | +42.8 | −4.0 | −13.7 | +111.9 | (+9.7) | +0.4 | +0.1 | +0.0 | *(+0.6)* | *(+0.3)* | +16.3 | +52.0 |
| 16 | 276 | +80.6 | +24.9 | −14.1 | −35.3 | +84.5 | (−1.9) | +0.1 | +0.0 | +0.0 | *(+31.9)* | *(+8.4)* | +0.0 | +0.0 |
| 17 | 156 | +137.7 | +75.9 | −1.0 | −2.4 | +138.3 | (+45.8) | +0.1 | +0.1 | +0.0 | *(+36.5)* | *(+34.3)* | +0.0 | +42.0 |
| 18 | 855 | +95.2 | +32.0 | −0.2 | −0.5 | +97.2 | (+0.7) | +0.0 | +0.0 | +0.0 | *(+0.0)* | *(+0.0)* | +13.9 | +374.5 |
| 19 | 185 | +80.6 | +21.5 | −1.5 | −3.8 | +82.6 | (−0.2) | +0.7 | +0.2 | +0.0 | *(+19.9)* | *(+15.1)* | +12.9 | +28.1 |
| 20 | 132 | +81.9 | +37.7 | −0.1 | −0.1 | +85.5 | (+0.2) | +0.4 | +0.0 | +0.0 | *(+68.2)* | *(+42.7)* | +0.0 | +25.6 |
| 21 | 619 | +89.8 | +30.9 | −1.6 | −4.2 | +86.0 | (+2.6) | +0.0 | +0.0 | +0.0 | *(+0.1)* | *(+0.0)* | +0.7 | +14.7 |
| 22 | 107 | +69.2 | +22.0 | −11.8 | −29.3 | +70.3 | (−0.9) | +0.2 | +0.1 | +0.0 | *(+29.7)* | *(+1.9)* | +0.0 | +0.0 |
| **SUITE** | **6267** | **+89.7** | **+31.6** | **−2.8** | **−10.5** | **+91.5** | *(+4.2)* | **+0.2** | **+0.1** | **+0.0** | *(+21.1)* | *(+8.0)* | **+4.7** | **+106.4** |

Raw CSVs: `experiments/analysis/S-WHATIF_matrix.csv`, per-scenario
`analysis/SW-*_sim.csv`, per-run JSONs under `experiments/sim/SW-*/`.

### Speedup bands (gpu_compute > 1: G4b conservative ↔ v0f optimistic)

| scenario | G4b suite | v0f suite | read as |
|---|---|---|---|
| gpu_compute 1.25 | −2.8% | −9.8% | "somewhere in −3…−10%" |
| gpu_compute 2.0 | −10.5% | −24.3% | "somewhere in −10…−24%" |

Even the optimistic bound says **doubling SM throughput buys at most ~24%** — the
suite's queue-wait/host floor (executor threads, dependencies) caps the win. v0f's
biggest 2× winners: q1 −49.9%, q6 −49.0%, q14 −47.4%, q15 −47.3%, q16 −41.9%.

## 3. What binds each query at baseline

From the paired capture (device-busy = kernel-ns ÷ steady-state window, §8.2
convention; serialization = union/sum, the G4b validity gate) and the knobs=1 replay
binding:

| q | dev-busy w2 | serial w1 | knobs=1 binding | reading |
|---|---|---|---|---|
| 1 | 343% (union 98%) | **0.33** | threads | device-saturated via **co-running** kernels — the one suite query outside the G4b regime |
| 2 | 40% | 0.99 | dependencies | host/dependency-bound (only 57 ms window); least compute-sensitive (real MPS-50 ×1.40) |
| 9, 13, 18, 21 | 97–100% | 0.98–0.99 | threads | the big four: serialized device-saturated; every wall-moving lever is GPU kernel throughput |
| 3–8, 10–12, 14–17 | 87–102% | 0.90–1.00 | threads | serialized device-saturated (mid weights) |
| 19, 20, 22 | 82–87% | 0.91–1.00 | threads/dependencies | moderately saturated; slightly milder MPS response (real ×1.71/×1.78/×1.74) |
| capacity-fragile at 0.25× | | | | q18 (+375%), q9 (+275%), q11 (+89%), q4 (+64%), q6 (+55%), q15 (+52%) go over the spill knee; q2/q7/q16/q22 stay flat |

## 4. Top findings

1. **The suite is a GPU-kernel-throughput machine.** 21/22 queries are
   device-saturated at baseline (kernel time 82–100% of wall) with serialized
   kernels; suite cost of halving SM throughput is **+90%** (validated against real
   MPS-50: real ×1.78, predicted ×1.90) and even the *optimistic* bound for doubling
   it is only **−24%** (−10.5% conservative). Scheduling/host knobs are not on the
   critical path — consistent with the PR's own profiling claims, now quantified
   per query.
2. **HBM capacity is the suite's cliff, not bandwidth.** At 0.5× capacity the suite
   pays only +4.7% (6 queries graze the knee), but at 0.25× it pays **+106%**,
   concentrated in q18/q9/q11 (order-of-magnitude tier, optimistic floor — pins
   unmodeled). Capacity planning question "can we run SF1000 in 64 GB?" → "only with
   ~2× wall on the heavy tail, likely worse".
3. **c2c and io are dead knobs on the ship lane** — by construction (everything
   GPU-pinned, zero steady-state NVMe/H2D traffic), and the io grid column is a
   known G1 artifact on pinned traces. Anyone costing NVMe/link upgrades for THIS
   config should spend nothing; the un-pinned/cold lane verdicts live in §3.5/§7.3.
4. **q17 is the most compute-elastic query** (+138% at half SMs — worst in suite;
   its real MPS-50 ×1.89 is also near-worst) and **q13/q16/q22/q14 are the most
   compute-*inelastic* big queries at speedups** (−30/−35/−29/−29% at 2× as their
   dependency chains release) — where to look first for any "buy faster GPUs"
   decision.
5. **q1 needs its own model.** It is the only query whose kernels co-run
   (serialization 0.33, kernel-sum 343% of wall) — the G4b gate correctly stands
   down and its `gpu_compute` cells are lower bounds (its real MPS-50 slowdown,
   ×1.92, was the suite's *largest* — co-running does not protect against an SM
   partition, §9.6 surprise 2).

## 5. Reproduction

```bash
# collection (GPU): S-BASE / S-P3 / S-M50 / S-NSYS rows in experiments/runs.csv
EXTRA_PRE_SQL='SET enable_compressed_materialization = false;' \
  bash tools/hwsim/experiments/bin/run_latemat_row.sh S-BASE S-BASE 3 plain $(seq 1 22)
# ... mps100 / mps50 / nsys arms identically (see runs.csv notes)

# per-query physics profiles (22 queries arrive in 19 reports — ranges MERGE
# when nsys misses a profiler_stop/start pair; nothing is lost, the ingest's
# structural matcher separates them):
python3 -m hwsim ingest-nsys traces/S-NSYS/<uuid> nsys/S-NSYS/S-NSYS.<i>.sqlite \
    -o nsys/S-NSYS/physics_SN<i>.json

# one what-if cell:
python3 -m hwsim simulate traces/S-BASE/<uuid> \
    --query-label S-BASE_tpch_q9_iter2_iter1 \
    --physics nsys/S-NSYS/physics_SN8.json --knob gpu_compute=0.5
```

Query→profile map (merged reports): q8,q9→SN8; q17,q18,q19→SN16; q10–16→SN9–15;
q20,q21,q22→SN17,SN18,SN19; else qN→SNN. S-NSYS2 (q20–22) is an unused replicate.
