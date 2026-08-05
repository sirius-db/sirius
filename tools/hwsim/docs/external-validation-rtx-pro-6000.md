# External validation — RTX PRO 6000 Blackwell (x86/PCIe), first non-GB300 box

> Independent validation run reported by the project owner on 2026-08-05, reproduced here
> verbatim (received as a report; artifacts live on that box, see the Artifacts section).
> Headline: **the physics split transfers across GPU arch (sm_120 vs GB300), CPU arch
> (x86 vs Grace), link class (PCIe vs C2C), and scale factor (SF100 vs SF1000) with zero
> recalibration** — MPS-50 what-if predicted at median E +0.2%, median |E| 3.1%, suite
> time-weighted +0.9%, 20/22 within ±10% (v0 contrast: +80.7%). Follow-ups extracted from
> this report are tracked as WS19 (spec-sheet target mode) and WS20 (physics-retimed quent
> export + kit paper-cuts).

**Branch tested:** `claude/sirius-hardware-simulation-25edbe` @ `e38a9d84` (PR #1424 head), built clean.
**Scope:** replay self-consistency, one physical what-if (MPS ACTIVE_THREAD_PERCENTAGE=50), and
quent-export fidelity. **Zero fitting/training:** every prediction consumed only the baseline
trace + baseline paired-nsys profile + the nominal knob (0.5, not the measured f_ach); the
degraded run was captured independently and used only as the grader, after predictions existed.
No spill-model engagement (0 downgrades), so no GB300-fitted constant contributed.

## Environment

- 1× RTX PRO 6000 Blackwell Workstation (sm_120, 188 SMs, 97.9 GB), x86 (48 cores), PCIe.
  Second identical GPU present but runs were `num_gpus: 1`.
- TPC-H **SF100** parquet (per-table dirs), NVMe at 14.6 GB/s O_DIRECT seq.
- Throttle-kit victims: fma 87.2 TFLOP/s; saxpy 1401 GB/s; HBM-CE 1471 GB/s; host DRAM
  (8-thr memcpy) 77.8 GB/s; PCIe H2D/D2H 57.7/57.3 GB/s.
- **No Grace co-limit** (hbm eater → dram victim: no effect), mild PCIe⇄DRAM coupling
  (dram eater drags h2d 57.7→33.9). Platform-law overrides behave as the docs predicted.
- MPS private daemon; fma f_ach at pct 75/50/25 = **0.746 / 0.507 / 0.250** (linear, like GB300).
- Capture config: `usage_limit_bytes: 80GB` (default 0.95 fraction fails to init on 97 GB cards),
  `reservation_limit_fraction: 0.8`, `exporter: ndjson`. 4 executor threads (default).
- Sessions (22 queries × 3 iters each, single process, labeled): baseline, mps100 anchor, mps50;
  plus a paired quent+nsys baseline (22 × 2 iters) via
  `nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi` around
  `CALL profiler_start()/profiler_stop()`. nsys **2025.5.2** (not 2025.6.3): ingest clean anyway.

## Result 1 — replay self-consistency (knobs=1), 66 executions

**median |err| 0.13%, p90 0.54%.** Matches GB300 G-SELF. One anomaly worth a look:
**q11 iter1 +14.78% and iter3 +10.31%, while iter2 is −0.17%** (130 ms query, 61 tasks,
33 pipes, binding=executor_threads). Not noise-shaped — same query, same session, two of three
iterations replay ~11–15% slow. Smells like a mis-attributed dependency or barrier on a short
many-pipeline query. Trace available (see artifacts).

## Result 2 — MPS-50 what-if (the E1-50 analogue), all 22 queries

Method identical to §3.1/§7.2: real× = med(mps50 i2,i3)/med(mps100 i2,i3); v0× from the
baseline trace (med of iters 2,3); physics× = sim_wall/sim_baseline_wall on the paired trace
(med of both iters) with `--physics` from `ingest-nsys` of the paired baseline capture.

Ingest diagnostics first: 44/44 windows matched (structure score 1.00), same_run clock fits
(iter-1 windows rms ≤ µs-scale; iter-2 windows show the skewed-slope artifact documented in
§8.5 — slopes 0.95–1.42, one q21 window fell back to structural join). **Kernel serialization
0.53–0.86 on every window → the G4b fluid gate correctly stood down everywhere**; §7 split
semantics applied. This lane is the NVMe-lane twin: spans host-dominated, Preparing spans
≈zero-length, derived SOURCE→GPU channel capacity 90–500 TB/s (unphysical-capacity class).

| q | real× | v0× | E_v0 | phys× | E_phys | phys ident% (knobs=1) |
|---|------|------|------|------|------|------|
| 1 | 1.189 | 1.996 | +67.9% | 1.134 | **−4.6%** | −0.42 |
| 2 | 1.070 | 1.709 | +59.7% | 1.052 | **−1.6%** | −0.23 |
| 3 | 1.091 | 1.941 | +77.9% | 1.063 | **−2.6%** | −0.07 |
| 4 | 1.083 | 1.951 | +80.1% | 1.184 | **+9.3%** | −0.27 |
| 5 | 1.012 | 1.961 | +93.8% | 1.044 | **+3.2%** | −0.15 |
| 6 | 1.018 | 1.994 | +95.8% | 1.053 | **+3.4%** | −0.34 |
| 7 | 1.018 | 1.939 | +90.5% | 1.031 | **+1.3%** | −0.04 |
| 8 | 1.033 | 1.958 | +89.6% | 1.038 | **+0.5%** | −0.11 |
| 9 | 1.076 | 1.890 | +75.6% | 1.051 | **−2.4%** | −0.12 |
| 10 | 1.150 | 1.958 | +70.3% | 1.078 | **−6.2%** | +0.10 |
| 11 | 1.176 | 2.121 | +80.3% | 1.143 | **−2.8%** | **+4.69** |
| 12 | 1.133 | 1.974 | +74.2% | 1.066 | **−5.9%** | +0.16 |
| 13 | 1.063 | 1.980 | +86.2% | 1.165 | **+9.6%** | −0.06 |
| 14 | 1.017 | 1.977 | +94.5% | 1.033 | **+1.6%** | −0.15 |
| 15 | 1.071 | 1.977 | +84.6% | 1.040 | **−2.9%** | +1.01 |
| 16 | 1.178 | 1.629 | +38.3% | 1.205 | **+2.3%** | −0.32 |
| 17 | 0.999 | 1.968 | +97.0% | 1.155 | **+15.6%** | −0.10 |
| 18 | 1.021 | 1.869 | +83.1% | 1.083 | **+6.1%** | −0.15 |
| 19 | 1.096 | 1.976 | +80.2% | 1.067 | **−2.7%** | −0.06 |
| 20 | 1.007 | 1.891 | +87.7% | 1.206 | **+19.7%** | −0.06 |
| 21 | 1.148 | 1.940 | +69.1% | 1.055 | **−8.1%** | −0.06 |
| 22 | 1.075 | 1.521 | +41.4% | 1.074 | **−0.1%** | −0.44 |

**Physics: median E +0.2%, median |E| 3.1%, 20/22 within ±10%, suite time-weighted +0.9%
(pred 12 860 ms vs real 12 741 ms). v0: median +80.3%, suite +80.7%.**

Cross-hardware read: §7.2 GB300 numbers (−1.5/−3.7/−6.4% medians, mildly optimistic) vs
here **+0.2% centered** — the physics split transfers to a different CPU arch, link class
(PCIe vs C2C), GPU arch (sm_120 vs GB300), and scale factor (SF100 vs SF1000) with no
recalibration. The two outliers are the two queries with real× ≈ 1.0 (q17 0.999, q20 1.007)
where physics still charges the kernel share: both fail *pessimistic* at +15.6/+19.7%.
Hypothesis for the backlog: on queries whose kernel time co-runs entirely with host work even
at half SMs, the frozen-host/scaled-kernel split over-charges; a per-span overlap fraction
(kernel-time hidden under host-time) from the same nsys join would cap those.

## Result 3 — quent-export fidelity

Exports: q1/q9/q21 `@baseline`, q9 `@gpu_compute=0.5` (v0 lane).

- `validate_quent_session.py --simulated`: **0 errors, 0 warnings** on all four.
- `ingest_check.sh` (real Rust analyzer): ingested; **132-node resource tree** with the true
  operator plan.
- q9 `@baseline` vs source trace: wall −0.3% vs traced; re-sim of the export **bit-exact**
  (985.742461 ms both); 36/36 pipelines, per-pipeline busy deltas 0.0000 ms; 13/13 operator
  aggregates identical (calls + sim_ms). The GB300 round-trip guarantees reproduce here.
- q9 `@gpu_compute=0.5` export vs the **real** mps50 trace: structure exact (192 tasks /
  36 pipes / 27.69 GB H2D) but wall 1852 ms vs real 1034 ms — i.e. the export carries v0's
  +79% on this lane while the physics number (1039 ms, +0.5%) exists and can't be exported.
  **This bumps the priority of the "export physics-retimed runs" follow-up**: on
  host-dominated lanes (both the GB300 NVMe lane and this whole machine class), v0-only
  export means every compute-what-if timeline a user opens in the quent UI is ~2× wrong.

## Defects / paper-cuts found on the way (none block the results)

1. `tools/hwsim/throttle/io/Makefile`: `LDFLAGS ?= -pthread` lands before the source in the
   nvcc-less link on this toolchain → `undefined reference to pthread_join`. Put `-lpthread`
   after `$<` (or use `LDLIBS`).
2. Compute/membw Makefiles hardcode sm_100. PTX fallback JITs fine on sm_120, but consider
   `-arch=native` or a doc note.
3. The membw kit's `hbm-sm` victim reads ~3 PB/s here (L2-resident on 128 MB-L2 parts) — only
   the CE number is meaningful on this class of GPU; maybe size the buffer off L2.
4. Bundled `tpch_telemetry_sirius.yaml` sets `exporter: postcard`; hwsim only parses ndjson.
   Runbook should say so explicitly (hit before first capture, cheap to warn in `info`).
5. q11 replay outlier (Result 1) — reproducible in-session, trace preserved.
6. Confirmed on a second platform: iteration-2 nsys windows get skewed clock fits (§8.5
   artifact) and one window fell to structural join; fractions unaffected.

## Artifacts (on the RTX PRO 6000 box, worktree `.claude/worktrees/simulator-timing-output-prediction-e02735`)

- `hwsim_run/RESULTS.md` — the write-up; `hwsim_run/base_selfcheck.csv`.
- Sessions under `hwsim_run/telemetry_data/`: `019fd242-f189…` baseline ×3,
  `019fd245-12c3…` mps100 ×3, `019fd245-a665…` mps50 ×3, `019fd246-a2e5…` paired ×2.
- `hwsim_run/nsys/paired_base.{nsys-rep,sqlite}` (177 MB sqlite),
  `hwsim_run/physics_paired.json`, `hwsim_run/quent_exports/` (4 sessions),
  `hwsim_run/sirius_hwsim_capture.yaml`.
- Not run there: io/membw/capacity knob families against physical throttles, cpu_compute
  (still no physical validation anywhere), spill regime, multi-GPU.
