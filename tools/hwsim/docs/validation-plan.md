# WS7 — Validation Experiment Matrix & Runbook

**Status:** design complete, 2026-08-04. No GPU runs were performed for this plan (design
only); every environment fact below was verified read-only this session. Executor: WS8,
which writes `docs/validation-results.md`.

**Question this plan answers:** when we physically degrade one hardware resource by a
measured factor `f` and feed the *baseline* trace + `knob=f` to the v0 simulator
([simulator-design.md](simulator-design.md)), how close are the predicted wall time and
execution structure to the physically degraded run — per knob, beyond the noise floors?

Two framing rules inherited from the completed workstreams:

1. **Feed achieved factors, never nominal ones** (io-throttle.md, membw-throttle.md).
   Every degraded run must record the mechanism's achieved rate (CSV) and, where possible,
   derive the factor from the degraded run's own telemetry.
2. **Not every row is pass/fail.** For `gpu_mem_capacity` below the spill threshold and for
   `io_bandwidth` (gap G1), v0 is *known* to be wrong in a specific direction; those rows
   exist to measure *how* wrong, so the result prioritizes (or kills) the v1 items. A row
   that quantifies a documented boundary is a success, not a failure.

---

## 0. Environment inventory (pinned 2026-08-04)

| Item | Fact |
|---|---|
| Box | pmgb300ws-0163 (GB300: 72-core Grace + Blackwell, 152 SMs, cc 10.3), 744 GB DRAM (`free -g`), GPU 256,703 MiB |
| Shared-GPU state at inventory time | one other compute app (`python3`, **243 GB GPU memory**) — WS8 must gate on this (§5.1) |
| Dataset SF1000 | `/localhome/local-faramburu/tpch_parquet_sf1000` — 265 GB total, `lineitem/` = 166 GB in 60 files; layout `<table>/part.N.parquet` |
| Dataset SF10 | `/localhome/local-faramburu/tpch_parquet_sf10` — 2.5 GB (rejected for validation, §1.2) |
| Disk | everything on `/dev/nvme0n1p2` (root fs), 2.7 TB free — same device the io injector was calibrated on |
| Main-repo binary | `/localhome/local-faramburu/repos/sirius/build/release/duckdb` (Jul 31 21:09), checkout `dev@b77aa438`, **no `src/*.cpp` newer than the binary** → build is current w.r.t. its checkout |
| Sample-trace provenance | `model.qmi` of the 654 MB trace: Sirius commit `b77aa438` (dirty), built 2026-07-31 → same commit as the main-repo binary |
| Telemetry exporter default | `src/include/sirius_config.hpp:140` → `ndjson` (hwsim parses ndjson only). **The bundled `test/tpch_performance/tpch_telemetry_sirius.yaml` overrides to `postcard` — do not use it as-is** |
| Memory-capacity config keys | `sirius.memory.gpu.usage_limit_bytes` (absolute, mutually exclusive with `usage_limit_fraction`; docs/super-sirius/configuration.md §GPU). Traced pool capacity = usage_limit × `reservation_limit_fraction` (0.9): sample trace 229.05 GB ≈ 269.2 GB × 0.95 × 0.9 |
| Plan levers (runtime `SET`) | `fuse_merge_pipelines` (default true — changes pipeline boundaries), `max_broadcast_join_size`, `enable_dynamic_filter_pushdown`, … (configuration.md §DuckDB SET Variables) |
| Throttle binaries | **already built in this worktree**: `tools/hwsim/throttle/io/io_load`, `throttle/membw/{membw_eater,membw_victim}`, `throttle/compute/{throttle_compute,victim_bench}` + `mps_probe.sh` |
| Simulator | `tools/hwsim/sim`, stdlib-only, system `python3` is 3.12.3 (no pixi needed); model cache `~/.cache/hwsim/` |
| Trace runner | `test/tpch_performance/run_tpch_parquet_and_generate_telemetry.sh` — labels each iteration `<note>_tpch_q<N>_iter<K>` (**no zero-padding**: `q9`, not `q09`) via `sirius_set_query_label`; uses `$PROJECT_DIR/build/release/duckdb` relative to its own location |
| ndjson volume | ~10 MB per query-iteration (654 MB / 66) |
| Sample-trace facts reused for sizing | q21 4.38 s / 340 GB H2D / peak pool 76.3 GiB; q09 2.34 s / 55.7 GiB; q19 1.52 s / 19.2 GiB; 4 executor threads; pool 229 GB |

### 0.1 Binary decision

**Use the main-repo binary `/localhome/local-faramburu/repos/sirius/build/release/duckdb`
(read-only) for every run in this plan.** Rationale:

- It is the same commit (`b77aa438`) the WS6 sample trace was produced from, and the build
  is current w.r.t. its checkout (verified: no source file newer than the binary).
- This worktree has no build and uninitialized submodules; building here costs ≥1 h
  (submodule init + `pixi run make`) of wall time with zero validation benefit — the
  validation compares *fresh* baseline traces against *fresh* degraded runs made with the
  *same* binary, so absolute binary vintage cancels.
- Other agent worktrees have builds, but their branch provenance is experiment-specific —
  worse, not better.

Caveats to record: the sample trace's `model.qmi` says the tree was *dirty* at build time,
so the binary may contain small uncommitted deltas vs `b77aa438`. Mitigation: WS8 records
`sha256sum` of the binary + extension in every environment snapshot, and gate **G-SELF**
(§2, P1) re-establishes simulator self-consistency on a *fresh* trace from this binary
before any degraded run. Fallback if G-SELF fails (median selfcheck error > 2%): build in
this worktree (`git submodule update --init --recursive && pixi run make`, ~1 h, no GPU)
and re-run P1.

### 0.2 Dataset decision

**SF1000 parquet for all experiments.** Justification per experiment class:

- *Capacity (E2)* needs real memory pressure: sample-trace working sets (76/56/19 GiB
  against a 229 GB pool) only exist at SF1000. SF10 peaks would be ~100× smaller — no
  experiment possible.
- *Compute/membw/C2C (E1/E4/E5)* need walls ≫ noise floor and ≫ per-task overheads:
  SF1000 heavyweights run 1.5–4.4 s; SF10 queries run tens of ms, where the −0.2%
  fixed-overhead bias and run-to-run jitter dominate.
- *I/O (E3)* needs reads that actually hit the device: only SF1000 has multi-GB column
  reads (and the io injector was calibrated on exactly this dataset's device).
- Variance risk at SF1000 is handled by empirical query selection (P1), not by shrinking
  the dataset. One baseline trace set then serves every experiment.

### 0.3 Output layout (WS8 creates; everything gitignored)

```
tools/hwsim/experiments/            # WS8: mkdir; add .gitignore containing "*"
  runs.csv                          # manifest = resume state (§5.4)
  configs/                          # generated YAMLs (base + per-row variants)
  bin/                              # run_with_preamble.sh wrapper (§4.6)
  env/<row>_<ts>.txt                # per-run environment snapshots
  traces/<ROW_ID>/<session-uuid>/   # quent output_directory per row
  throttle_logs/<ROW_ID>/           # injector/eater CSVs, iostat logs, probe outputs
  sim/<ROW_ID>/                     # simulator JSON/CSV outputs
  analysis/                         # final tables for validation-results.md
```

Disk budget: ~20 traced sessions × 100–250 MB ndjson ≈ **3–5 GB** traces + ~1 GB sim
caches/JSON + 16 GiB io scratch files ≈ **< 25 GB** total (2.7 TB free — no risk). The
io scratch files (`inject.dat`, `victim.dat`, 8 GiB each via `--mkfile 8`) must live on
the same filesystem as the dataset — put them in `experiments/scratch/` (root fs, same
NVMe).

The base validation config `configs/base.yaml` (defaults for everything except telemetry):

```yaml
sirius:
  telemetry:
    enable_quent: true
    exporter: ndjson            # hwsim parses ndjson only; bundled yaml says postcard!
    output_directory: /localhome/local-faramburu/repos/sirius/.claude/worktrees/sirius-hardware-simulation-25edbe/tools/hwsim/experiments/traces/<ROW_ID>
    engine_name: siriusDB
```

Per-row configs are copies with `<ROW_ID>` substituted (and, for E2 only, a
`memory.gpu.usage_limit_bytes` line added). Leaving every other key at default keeps the
resource model identical to the WS6-validated regime (4 executor threads, ~229–230 GB
pool).

---

## 1. Query selection

### 1.1 Candidate pool (pre-flight P1 measures these; final set is data-driven)

Pool of 9: **q1, q3, q5, q6, q9, q13, q18, q19, q21**.

- q21 / q9 / q19 are the WS6 sensitivity trio — transfer-heavy (340 GB H2D), memory-heavy
  (55.7 GiB working set), and compute-diverse respectively; they anchor "diverse binding
  constraint".
- q1, q6 are lineitem-scan-dominated → the E3 (I/O) candidates; q6 may be too fast warm
  but its *cold* wall is read-dominated, which is what E3 measures.
- q3, q5, q13, q18 are heavyweight join/aggregate alternates in case any anchor turns out
  variance-prone (the known SF1000 hazard: several queries swing 13–28% run-to-run).

### 1.2 Selection rule (applied after P1, before B1)

From P1's per-query walls (2 sessions × 3 warm iterations):

1. **Stability:** keep queries whose within-session spread `(max−min)/median` over iters
   2–3 AND across-session median shift are both ≤ 5%.
2. **Weight:** keep queries with median warm wall ≥ 0.8 s (so a ±3% noise floor is ≥ 10×
   smaller than the smallest effect we accept, and per-task fixed overheads stay ≪ wall).
3. **Diversity:** the final set must contain q21, q9, q19 (or a same-role replacement
   if one fails stability — replacements: q5 for transfer, q18 for memory, q1 for
   compute; document the swap).

**Final throttle set = 5 queries** (target: q1, q9, q13, q19, q21).
**I/O set = 3 queries** from {q1, q6, q19} that pass a *cold*-stability sanity in B2.
If fewer than 5 pass, run with 4 — do not admit an unstable query to hit a count.

---

## 2. Noise-floor pre-flights

Two floors must be established before any acceptance threshold means anything, plus one
gate on the fresh binary+trace pipeline.

| ID | What | Runs | Expected (to be replaced by measurement) |
|---|---|---|---|
| **P1** | Run-to-run variance + **G-SELF** gate | 2 sessions, pool of 9, 3 iterations, warm, quent ON (ndjson) | stable-query median-of-3 repeatability **±2–5%** (WS3 device-level repeats ~2%, WS5 GPU baselines <0.5%; the 13–28% swingers get excluded). G-SELF: `python3 -m hwsim selfcheck` on both P1 sessions — median |err| ≤ ~0.5%, worst ≤ 2% (WS6 got 0.17% / 1.2%) |
| **P2** | Quent-telemetry overhead (simplified A/B from nsys-extraction §4.3: quent-on vs quent-off, no nsys anywhere) | 1 session, pool of 9, 3 iterations, warm, config = base.yaml with `enable_quent: false`; walls from the runner's `.timer on` output | **+1–5%** on multi-second queries (~45 k events/query, ~10 MB ndjson/query). Interpretation rule below |
| **P3** | MPS-daemon null effect | 1 session, final 5-query set, 3 iterations, under private MPS daemon at `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100`, `CUDA_MODULE_LOADING=EAGER` | walls within the P1 noise band of the no-MPS baseline (WS5 measured pct=100 ≡ baseline for kernels); doubles as the E1 anchor point f=1.0 |

**How the noise floors feed acceptance:** let `σ_q` = P1's per-query repeatability
(half-width of the median-of-3 band, in %). Acceptance thresholds in §3 are expressed as
`max(threshold_floor, 3·σ_q)` — a knob prediction is only judged against effects that
exceed three times what identical runs disagree by. Queries with σ_q > 5% are excluded
(rule §1.2), so in practice thresholds are the stated floors.

**How P2 is used:** all validation arms (baseline trace AND degraded runs) run quent-on,
so quent overhead is *common mode* and cancels in the primary metric (slowdown ratios,
§7.1). P2 exists to (a) caveat external claims ("predicted walls are quent-on walls"),
and (b) verify the overhead is stable (if P2 shows > 10% on any selected query, that
query's absolute-wall comparisons get a caveat flag in the results; ratios remain valid).

**P1 doubles as cache warmer:** its first session's iter-1 walls will include cold reads
(265 GB dataset vs 744 GB RAM); discard iter 1 everywhere (standard) and note that P1
leaves the page cache warm for all subsequent warm-cache experiments.

---

## 3. The experiment matrix

Conventions for the table:

- **Baseline trace T** = B1's session (5 queries × 4 iterations; iters 2–4 are the
  measured ones). Every `simulate` call runs once per measured baseline iteration; the
  prediction is the median of the three.
- `f_ach` = achieved factor, measured per §4 for each mechanism; the simulator is invoked
  with `knob=f_ach`, never the nominal target.
- "Acceptance" states the wall-time criterion on the **slowdown-ratio error** (§7.1) at
  `max(floor, 3σ_q)`, plus structural checks (§7.2). Rows marked *quantify* have no
  pass/fail — their deliverable is a measured error with a decision rule.
- Iterations: 4 per query per degraded session (iter 1 = warmup, discarded), except E3
  (cold runs are single-iteration by construction) and E2's lowest point (2 measured
  iters, runtime guard §4.4).

| # | Row ID | Knob(s) → sim invocation | Mechanism (exact recipe in §) | Factor target → expected f_ach | Queries | Est. GPU-busy | Acceptance / purpose |
|---|---|---|---|---|---|---|---|
| 1 | P1a,P1b | — (selfcheck knobs=1) | plain runs, quent on (§4.1) | 1.0 | pool of 9 ×3 | 8 + 5 min | G-SELF ≤ 2% worst; σ_q table |
| 2 | P2 | — | quent OFF run (§4.1) | 1.0 | pool of 9 ×3 | 5 min | overhead table; expect 1–5% |
| 3 | P3 | `gpu_compute=1.0` | MPS pct=100 (§4.2) | 1.0 | final 5 ×3 | 3 min | null effect ≤ noise band |
| 4 | B1 | — (baseline trace T) | plain run, quent on (§4.1) | 1.0 | final 5 ×4 | 4 min | selfcheck ≤ 2%; peaks-pool read → finalize E2 capacities |
| 5 | E1-75 | `gpu_compute=0.75` | MPS pct=75 on Sirius (§4.2) | 0.75 → exact (WS5: linear ±1%) | final 5 ×4 | 5 min | ratio err ≤ **10%**; rank-corr 1.0 across E1 rows |
| 6 | E1-50 | `gpu_compute=0.50` | MPS pct=50 (§4.2) | 0.50 | final 5 ×4 | 6 min | ratio err ≤ **10%** |
| 7 | E1-25 | `gpu_compute=0.25` | MPS pct=25 (§4.2) | 0.25 | final 5 ×4 | 9 min | ratio err ≤ **15%** (coupling floor); feeds E6 |
| 8 | E2-hi | `gpu_mem_capacity=f_ach` (≈0.50) | YAML `usage_limit_bytes: 128000000000` (§4.4) | pool ≈115 GB, above all working sets | final 5 ×4 | 5 min | **match row**: sim Δ=0; real Δ ≤ 3σ_q; zero forced_admissions AND zero real `Downgrading` |
| 9 | E2-mid | `gpu_mem_capacity=f_ach` (≈0.25) | YAML `usage_limit_bytes: 64000000000` (§4.4) | pool ≈57.6 GB: q9 marginal, q21 pressured, q19 untouched | final 5 ×3 | 8 min | *split row*: unpressured queries = match (≤3σ_q); pressured queries = **boundary quantify** (see below) |
| 10 | E2-lo | `gpu_mem_capacity=f_ach` (≈0.15) | YAML `usage_limit_bytes: 38000000000` (§4.4) | pool ≈34 GB: q9+q21 spill | final 5 ×2 (runtime guard) | 10 min | *boundary quantify*: report real-vs-sim slowdown ratio; structural: `forced_admissions>0 ⟺ real Downgrading>0` per query (≥4/5 correct) |
| 11 | B2 | — (cold baseline traces) | evict + plain run (§4.3) | 1.0 cold | io set 3 ×(2 sessions ×1 iter) | 12 min | cold repeatability ≤ 8% → gates E3 |
| 12 | E3-84 | `io_bandwidth=f_ach` (≈0.84) | io_load inject R≈1.63 GB/s, 32 thr, evicted (§4.3) | v=0.84 | io set 3 ×1, session/query | 7 min | *quantify G1* (decision rule §7.4) |
| 13 | E3-60 | `io_bandwidth=f_ach` (≈0.60) | inject R≈3.26 GB/s (§4.3) | v=0.60 | io set 3 ×1 | 8 min | *quantify G1* |
| 14 | E3-36 | `io_bandwidth=f_ach` (≈0.36) | inject R≈4.89 GB/s (§4.3) | v=0.36 | io set 3 ×1 | 10 min | *quantify G1* + monotonicity of real walls |
| 15 | E4-73 | `gpu_mem_bandwidth=0.73` + co-set `c2c_bandwidth` from probe | membw_eater hbm/ce @500 GB/s (§4.5) | victim ratio 0.73 (WS4 table) | final 5 ×4 | 6 min | *quantify G4 placeholder*: expect sim **pessimistic** on compute-bound queries; rank-corr ≥ 0.9 across E4 rows+baseline |
| 16 | E4-45 | `gpu_mem_bandwidth=0.45` + co-set | membw_eater hbm/ce @1000 GB/s (§4.5) | victim ratio 0.45 | final 5 ×4 | 8 min | *quantify G4*; also tests WS4's unverified "CE eater vs compute-bound victim" |
| 17 | E5-70 | `c2c_bandwidth=0.70` + co-set `gpu_mem_bandwidth` from probe | membw_eater c2c/h2d @100 GB/s (§4.5) | victim ratio 0.70 (1-stream victim; Sirius-side f_ach from trace, §4.5) | final 5 ×4 | 6 min | ratio err ≤ **15%** on transfer-heavy queries (q21) using the **trace-derived f_ach**; others: quantify |
| 18 | E5-49 *(optional)* | `c2c_bandwidth≈0.49–0.65` | eater `--chunk-mb 256 --buf-mb 1024` (§4.5) | probe-measured | q21+q9 ×4 | 5 min | second c2c point for monotonicity; run only if budget allows |
| 19 | E6 | analysis-only (uses E1-25 + B1) | none — no new GPU time | — | — | 0 | quantify the `min(HBM, free_SMs×~40GB/s)` coupling: per-query E1-25 error vs query's membw-share proxy (§7.5) |
| 20 | B3 *(stretch)* | — (plan-variant trace) | `SET fuse_merge_pipelines=false` preamble (§4.6) | 1.0 | final 5 ×4 | 4 min | variant plan differs (pipeline count in `operator` entities); selfcheck ≤ 2% |
| 21 | E7 *(stretch)* | `gpu_compute=0.50` on **both** plan traces | MPS pct=50 × plan-variant (§4.6) | 0.50 | plan-sensitive subset (≥2 queries) ×4 | 5 min | sim predicts the **plan delta** under degradation: sign of (variant−default) wall correct; delta magnitude within 15 pp |

**Totals: 21 rows (17 with GPU time; E5-49/B3/E7 optional-stretch), estimated 105–135 min
GPU-busy** — inside the ~2.5 h budget with ~30 min contingency. Every session is ≤ ~10 min,
satisfying the ≤15-min idle-window chunking requirement.

Sim invocation template (per measured baseline iteration `K∈{2,3,4}`, per query `N`):

```bash
cd /localhome/local-faramburu/repos/sirius/.claude/worktrees/sirius-hardware-simulation-25edbe/tools/hwsim/sim
python3 -m hwsim simulate ../experiments/traces/B1/<session-uuid> \
    --query-label B1_tpch_q<N>_iter<K> \
    --knob gpu_compute=0.50 \
    --json ../experiments/sim/E1-50/q<N>_iter<K>.json
```

(Repeat `--knob` for co-sets. Remember: labels are **not zero-padded** — `q9_iter2`.
`cpu_mem_bandwidth` co-sets are pointless — the knob is a v0 no-op; the DRAM cross-talk
confound is handled in §6 instead.)

---

## 4. Mechanism recipes (exact commands)

Shared variables for all recipes:

```bash
MAIN=/localhome/local-faramburu/repos/sirius                     # binary + runner live here (READ-ONLY checkout)
WT=$MAIN/.claude/worktrees/sirius-hardware-simulation-25edbe     # this worktree (tools, plan, experiments)
EXP=$WT/tools/hwsim/experiments
DS=/localhome/local-faramburu/tpch_parquet_sf1000
RUN=$MAIN/test/tpch_performance/run_tpch_parquet_and_generate_telemetry.sh
IO=$WT/tools/hwsim/throttle/io;  MB=$WT/tools/hwsim/throttle/membw;  CT=$WT/tools/hwsim/throttle/compute
```

### 4.1 Plain traced run (P1, P2, B1)

```bash
cd $MAIN   # runner resolves the binary relative to its own tree
bash $RUN --config $EXP/configs/<ROW_ID>.yaml --parquet-dir $DS \
     --iterations 4 --note <ROW_ID> 1000 1 9 13 19 21 \
     2>&1 | tee $EXP/throttle_logs/<ROW_ID>/runner.log
```

- The runner's `.timer on` output in `runner.log` is the quent-independent wall record
  (this is all P2 has, since P2 runs quent-off).
- P1/P2 use the 9-query pool and `--iterations 3`; B1 uses the final set and 4.
- Session UUID = the new directory that appears under `$EXP/traces/<ROW_ID>/`; record it
  in `runs.csv`.

### 4.2 MPS compute throttle (P3, E1, E7-degraded)

```bash
export MPS_DIR=/tmp/hwsim-mps-$USER
export CUDA_MPS_PIPE_DIRECTORY=$MPS_DIR/pipe CUDA_MPS_LOG_DIRECTORY=$MPS_DIR/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d                                   # private daemon, no root

# in-session linearity probe (~5 s GPU): retained should equal pct/100 ±1%
CUDA_MODULE_LOADING=EAGER CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=<PCT> \
  $CT/victim_bench --victim fma --seconds 1.2 | grep RESULT \
  | tee $EXP/throttle_logs/<ROW_ID>/fma_probe.txt

CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=<PCT> CUDA_MODULE_LOADING=EAGER \
  bash $RUN --config $EXP/configs/<ROW_ID>.yaml --parquet-dir $DS \
       --iterations 4 --note <ROW_ID> 1000 1 9 13 19 21 \
  2>&1 | tee $EXP/throttle_logs/<ROW_ID>/runner.log

echo quit | nvidia-cuda-mps-control                          # ALWAYS tear down
rm -rf "$MPS_DIR"
```

Non-negotiables (WS5): `CUDA_MODULE_LOADING=EAGER` on the Sirius process (lazy-loading
trap); iteration 1 discarded as warmup; **daemon torn down after each E1 row** (E4/E5
calibrations assume *no* MPS — a leftover daemon changes cross-process semantics from
time-slicing to SM-sharing and invalidates the WS4 tables). `f_ach = PCT/100` (WS5
measured exactly linear); the fma probe is a recorded confirmation, not a correction.

### 4.3 I/O throttle (B2, E3) — cold-cache, sequential injection

One session per (query, fraction): eviction is per-run and cannot be done mid-session.
Single iteration per session (iteration 2 would be page-cache-warm and meaningless).

```bash
# one-time: scratch files with real extents on the SAME filesystem as the dataset
mkdir -p $EXP/scratch && $IO/io_load --file $EXP/scratch/inject.dat --mkfile 8

# same-session mixed-capacity re-probe (once per E3 block, ~30 s disk time, disk idle):
$IO/io_load --file $EXP/scratch/victim.dat --mkfile 8   # once
$IO/io_load --file $EXP/scratch/victim.dat --rate 0 --threads 8 --req-kb 1024 \
    --warmup 2 --duration 6            # → B (baseline, expect ≈6.5 GB/s)
# + one contended point at R=3.26 to confirm C_mix ≈ 6.9–7.1 (io-throttle.md law)

# per run:
for f in $DS/*/*.parquet $DS/*.parquet; do $IO/io_load --file "$f" --evict; done
iostat -x 1 > $EXP/throttle_logs/<ROW_ID>/iostat_q<N>.log 2>&1 &  IOSTAT=$!
$IO/io_load --file $EXP/scratch/inject.dat --rate <R> --threads 32 \
    --duration 3600 --csv $EXP/throttle_logs/<ROW_ID>/inject_q<N>.csv &  INJ=$!
bash $RUN --config $EXP/configs/<ROW_ID>.yaml --parquet-dir $DS \
     --iterations 1 --note <ROW_ID> 1000 <N> 2>&1 | tee .../runner_q<N>.log
kill -TERM $INJ && wait $INJ;  kill $IOSTAT
```

- B2 = the same recipe with **no injector** (2 sessions per io query → cold baseline
  traces + cold noise floor).
- Injector rates from the WS3 linear law `R = C_mix − v·B` with the re-probed `C_mix`:
  targets ≈ 1.63 / 3.26 / 4.89 GB/s for v = 0.84 / 0.60 / 0.36.
- `f_ach = (C_mix − R_achieved)/B` with `R_achieved` = injector CSV mean over the query's
  scan window; cross-check with iostat (device total − injector = Sirius's achieved read
  rate). Record both in `runs.csv`.
- Sim input: the **cold** B2 trace of the same query (never the warm B1 trace — a warm
  trace contains no disk time to scale at all).
- Gate before each E3 run: `iostat -x 1 2` shows the device otherwise idle (shared disk!).

### 4.4 GPU memory capacity (E2) — Sirius's own pool knob

Config variant (only delta vs base.yaml):

```yaml
sirius:
  memory:
    gpu:
      usage_limit_bytes: 64000000000     # E2-mid; pool = 0.9 × this = 57.6 GB
```

Run = plain recipe (§4.1), warm cache, `--iterations 3` (mid) / `2` (lo).

- **Recompute the three capacity points after B1**: read each query's peak pool from the
  B1 knobs=1 report (`python3 -m hwsim simulate … --json` → pool stats; WS6 sample values:
  q21 76.3 GiB, q9 55.7 GiB, q19 19.2 GiB). Choose: hi = comfortably above every peak
  (~1.4× max peak); mid = between q9's and q21's peaks (splits the set into
  matched/pressured); lo = below q9's peak. Defaults in the matrix assume B1 ≈ sample
  trace.
- `f_ach` = `MemoryOperating.capacity_bytes` (degraded trace) ÷ same (B1 trace) — read
  both from the traces (`grep -m1 MemoryOperating traces/<row>/<uuid>/memory/*.ndjson`),
  don't trust YAML arithmetic.
- Count real spills per query: `grep -c '"Downgrading"' traces/<row>/<uuid>/task/*.ndjson`
  attributed per query via task→pipeline→plan→query (or coarsely per session when only
  one query is pressured).
- **Runtime guard**: if any single query exceeds 5 min wall in E2-lo, Ctrl-C the session,
  drop that query from the row, note it in `runs.csv`, rerun without it. (Spill cost at
  SF1000 is unmeasured — this is exactly the data v1 needs, but not at any price.)
- These traces are the **memory-pressured captures** WS6 §10 asked for (G5 calibration
  material) — preserve them.

### 4.5 Memory-bandwidth eaters (E4 hbm, E5 c2c) — no MPS anywhere

```bash
# verify no MPS daemon survives from E1: test ! -d /tmp/hwsim-mps-$USER/pipe

# same-session cross-talk probe triplet, WITHOUT eater then WITH eater (~20 s GPU):
for d in hbm dram c2c; do $MB/membw_victim --domain $d --secs 3 --quiet; done \
    | tee $EXP/throttle_logs/<ROW_ID>/probe_base.txt
$MB/membw_eater --domain hbm --engine ce --gbps 500 --duration 3600 \
    --csv $EXP/throttle_logs/<ROW_ID>/eater.csv &  EATER=$!
sleep 3
for d in hbm dram c2c; do $MB/membw_victim --domain $d --secs 3 --quiet; done \
    | tee $EXP/throttle_logs/<ROW_ID>/probe_eater.txt

bash $RUN --config $EXP/configs/<ROW_ID>.yaml --parquet-dir $DS \
     --iterations 4 --note <ROW_ID> 1000 1 9 13 19 21 2>&1 | tee .../runner.log
kill -TERM $EATER && wait $EATER
```

E5 swaps the eater line for `--domain c2c --engine h2d --gbps 100` (and the optional
E5-49 adds `--chunk-mb 256 --buf-mb 1024`).

Factors and co-sets:

- Primary knob = same-session probe ratio (`probe_eater/probe_base` for the targeted
  domain). WS4 table values (0.73 / 0.45 hbm; 0.70 c2c) are the *expected* values; the
  probes are authoritative because host-side baselines drift 7–11% between quiet periods.
- Co-set the *other* GPU-side knob from the same probe triplet (WS4 §7: an eater run is
  never a single-knob experiment): E4 rows co-set `c2c_bandwidth=<probe c2c ratio>`
  (expected ~0.90–0.95 at sub-saturation); E5 rows co-set `gpu_mem_bandwidth=<probe hbm
  ratio>` (expected ~0.95–0.98 at 100 GB/s). The DRAM ratio is recorded but **cannot** be
  fed to the sim (`cpu_mem_bandwidth` is a v0 no-op) — that residual is a documented
  confound (§6), expected to make the sim *optimistic* on scan/host-heavy queries.
- **E5's Sirius-side achieved factor (headline)**: the 1-stream probe under-states what a
  4-stream Sirius loses (C2C share tracks copy sizes in flight). Derive the headline
  `f_ach` from telemetry: median per-task Preparing bandwidth
  (`Preparing.input_bytes ÷ Preparing→first-Computing span`, HOST→GPU tasks only) in the
  E5 trace ÷ same statistic in B1. Report predictions under both factors (probe-derived
  and trace-derived); the trace-derived one is the accuracy headline, the gap between
  them is itself a finding about eater-vs-victim shape sensitivity.
- Keep eater targets in the **controllable region** (hbm ≤ ~1300 GB/s consumed, c2c
  ≤ ~130 GB/s): saturated eaters additionally steal ~50% GPU time (WS4 §4) and would
  require co-setting `gpu_compute` with no calibrated value.
- Two cells of the WS4 cross-talk matrix (CE-engine related) are pending re-measurement —
  the in-session probe triplet supersedes them for our rates; treat doc values as
  provisional priors only.

### 4.6 Plan-change stretch (B3, E7)

Lever: `SET fuse_merge_pipelines = false;` (runtime SET, default true) — changes pipeline
boundaries (merge fusion, physical-plan-generation.md), i.e. a *structural* plan change
visible as a different pipeline/operator count in the trace's `operator` entities.
Fallback lever if the final set shows no fused merges: `SET max_broadcast_join_size = 0`
(forces partitioned builds).

The stock runner has no SQL-preamble hook. WS8 creates
`$EXP/bin/run_with_preamble.sh` — a copy of `run_tpch_parquet_and_generate_telemetry.sh`
with one addition (after the `.timer on` line of the temp-SQL heredoc):
`[ -n "${PREAMBLE_SQL:-}" ] && printf '%s\n' "$PREAMBLE_SQL"`. Do **not** modify the
original under `$MAIN/test/` (read-only checkout).

```bash
PREAMBLE_SQL='SET fuse_merge_pipelines = false;' \
  bash $EXP/bin/run_with_preamble.sh --config ... --iterations 4 --note B3 1000 1 9 13 19 21
```

Design: (1) B3 captures the variant baseline trace; verify ≥2 queries actually changed
pipeline count vs B1 (else switch lever); selfcheck it. (2) E7 runs the plan-sensitive
subset for real under MPS 50% with the preamble. (3) Simulate **both** B1 and B3 traces at
`gpu_compute=0.5`; the tested prediction is the *cross-plan delta under degradation*:
`(wall_B3 − wall_B1)/wall_B1` predicted vs real (E7 vs E1-50). v0 cannot simulate a plan
change from one trace (G3 — batch bytes are replayed); simulating each plan's own trace
and comparing deltas is exactly what v0 *should* support — this row tests that claim.

---

## 5. WS8 execution runbook

### 5.1 Before every session: the shared-box gate

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv
```

- Proceed only if **no other compute app** is listed (an idle-check pass ≥ 2 min apart is
  the norm from the shared-box memory notes; at inventory time a `python3` held 243 GB —
  expect to wait). Never kill others' processes.
- Capacity-sensitive rows (B1, P1, E2, and any row whose config expects the ~229 GB pool)
  additionally require `memory.used` < ~10 GiB — Sirius's `usage_limit_fraction` needs
  the VRAM actually free.
- E3/B2 rows additionally require `iostat -x 1 2 nvme0n1` near-idle (disk is shared too).
- If the gate fails, wait; if a session must be interrupted (other user appears
  mid-session), let the current query finish if < 2 min remains, else Ctrl-C, mark the
  row `aborted` in `runs.csv`, clean up throttlers (`pkill -TERM -u $USER io_load
  membw_eater; echo quit | nvidia-cuda-mps-control`), and redo the whole row later
  (partial sessions are never analyzed).

### 5.2 Ordering (dependency-driven; each block fits one idle window)

```
P0  (no GPU): mkdir experiments tree, .gitignore, configs, wrapper script,
     scratch files (--mkfile), sha256 of binary+extension, tool --help smoke tests
P1a → P1b → P2                        # noise floors; warms page cache; G-SELF gate
   [ANALYSIS GATE: freeze final query set + σ_q table; run selfcheck on P1 traces]
B1                                    # the baseline trace T
   [ANALYSIS GATE: selfcheck B1; read peak pools; finalize E2 capacity bytes]
P3 → E1-75 → E1-50 → E1-25            # flagship; one MPS daemon per row, torn down after
E2-hi → E2-mid → E2-lo                # warm cache; runtime guard armed
E4-73 → E4-45 → E5-70 (→ E5-49)       # verify MPS pipe dir absent first
B2 (6 short sessions) → E3-84 → E3-60 → E3-36   # cold block last: it evicts the cache
B3 → E7                               # stretch, only if ≥30 min budget remains
```

Rationale for the cold block last: every eviction destroys the warm-cache state all other
experiments rely on; re-warming costs a full dataset read (~40+ s disk plus query time).
If the schedule forces interleaving, re-warm by running one throwaway full pass of the
final set (not analyzed) before the next warm row.

### 5.3 Per-run logging (all under `$EXP`, named by ROW_ID)

1. **Environment snapshot** (`env/<row>_<ts>.txt`), before AND after: both nvidia-smi
   queries from §5.1, `free -g`, `uptime`, `date -u`, binary+extension sha256, MPS env
   vars if any, `iostat -x 1 2 nvme0n1` (io rows).
2. **Runner stdout** (`throttle_logs/<row>/runner.log`) — contains `.timer` walls and the
   telemetry banner.
3. **Mechanism evidence**: injector/eater `--csv` files, iostat sidecar (io rows), fma /
   victim-triplet probe outputs (MPS / eater rows), eviction command transcript.
4. **Trace dir**: session UUID under `traces/<row>/`; verify non-empty
   `task/`+`query/` subdirs before marking the row done.
5. **`runs.csv` row**: `row_id, status(pending|running|done|aborted), utc_start, utc_end,
   session_uuid, nominal_factor, f_ach(+how derived), queries_run, gpu_busy_est_s, notes`.

### 5.4 Resume protocol

`runs.csv` is the single source of truth. On resume: (1) re-run the §5.1 gate; (2) find
the first row not `done`; (3) if `aborted`/`running` (crash), delete that row's trace dir
and throttle logs, verify no stray `io_load`/`membw_eater`/`mps` processes remain
(`pgrep -u $USER -l 'io_load|membw_eater|throttle_comp|nvidia-cuda-mps'`), re-run the row
from scratch. Analysis gates (after P1/P2 and after B1) are CPU-only and can run any time,
including while the GPU is busy with other users' work.

### 5.5 Simulation & analysis passes (CPU-only, no gating needed)

After each block: run the row's `simulate` calls (§3 template), then the §7 metrics.
First parse of each session costs ~17 s/654 MB (ours are ~5× smaller → a few seconds);
the pickle cache makes reruns ~1 s. Also run `python3 -m hwsim simulate <degraded trace>
--query-label <label>` **with no knobs** on every degraded trace — that replays the
degraded run's own structure (traced wall, per-pipeline breakdown) and is the "actual"
side of the structural comparison, plus a free selfcheck that the degraded regime still
replays consistently (expect worse than 0.17% under heavy distortion; report it).

---

## 6. Confounds & mitigations

| Confound | Bites | Mitigation (already encoded above) |
|---|---|---|
| **Page cache hides the disk** (265 GB dataset vs 744 GB RAM) | E3/B2 | `io_load --evict` over every parquet file before *each* cold run; one session per (query, fraction); iostat sidecar proves device reads happened; cold block scheduled last |
| **MPS lazy-loading stall** looks like starvation | E1/P3/E7 | `CUDA_MODULE_LOADING=EAGER` on Sirius; iteration 1 discarded; fma probe confirms pct linearity in-session |
| **Leftover MPS daemon flips cross-process semantics** (time-slicing → SM sharing) | E4/E5 | daemon torn down at the end of every §4.2 row; §4.5 asserts pipe dir absent |
| **Eater cross-talk** (C2C⇄DRAM first-order; HBM→C2C ~0.83 at saturation) | E4/E5 | stay in the controllable (sub-saturation) region; same-session victim-triplet probes; co-set the measurable GPU-side knob; DRAM leg documented as an *unmodelable* v0 residual (cpu knob no-op) with expected sign (sim optimistic on host-heavy queries); two provisional WS4 CE cells superseded by in-session probes |
| **Eater/injector achieved ≠ target** under contention | E3/E4/E5 | achieved-rate CSVs mandatory; `f_ach` from CSV/probes/trace, never nominal |
| **Injector out-queued** at deep fractions | E3 | 32 threads (WS3 regime-2 fix), sequential access only (`--rand` collapses the device), scratch files via `--mkfile` (real extents) |
| **Quent overhead** | all | measured in P2; identical (quent-on) config in baseline and degraded arms → cancels in slowdown ratios (§7.1) |
| **Run-to-run variance / unstable queries** (13–28% swingers at SF1000) | all | P1 empirical selection; thresholds at `max(floor, 3σ_q)`; medians of ≥2–3 measured iterations |
| **Other users** (GPU, and disk for E3) | all | §5.1 gate before every session; ≤10-min sessions; resume protocol; never analyze a session that overlapped foreign GPU work (check the post-run snapshot) |
| **Thermal / clock drift** | E1/E4/E5 | sessions short; `clocks.sm`+`temperature.gpu` in pre/post snapshots; if post-run SM clock < 95% of pre-run, flag the row and re-run its last query set once |
| **Cold-vs-warm baseline mismatch** | E3 | E3 compares against **B2 cold traces** only; warm B1 never used for io knob |
| **Spill runtime unknown** at low capacity | E2-lo | runtime guard (5 min/query cap), fewer iterations, drop-query rule |
| **v0 queue-order anchoring** may misorder under extreme knobs (§2.2 of sim design) | E1-25, E2-lo | expected-structure degradation documented; wall metrics still reported; structural metric (§7.2) quantifies it rather than hiding it |
| **Binary built from dirty tree** | provenance | same binary everywhere; sha256 recorded; G-SELF gate on fresh traces |
| **Label pitfalls** | analysis | labels are `<note>_tpch_q<N>_iter<K>`, no zero-padding; hwsim only synthesizes `tpch_qNN_iterK` names for *unnamed* queries — always pass the literal label |

---

## 7. Analysis plan (what WS8 computes per knob)

### 7.1 Wall-time metrics (primary = slowdown-ratio error)

Per (query q, row r with factor f):

- `real_wall(q,r)` = median traced `Executing→Exit` over measured iterations of the
  degraded run (from the degraded trace; cross-checked against `.timer`).
- `sim_wall(q,r)` = median over baseline iterations K∈{2,3,4} of simulate(T_K, knob=f_ach).
- **Ratio error (headline):**
  `E_ratio = [sim_wall(q,r)/sim_wall(q,base)] / [real_wall(q,r)/real_wall(q,base)] − 1`
  where `sim_wall(q,base)` is the knobs=1 simulation of B1 (or B2 for io) and
  `real_wall(q,base)` the matching real baseline. Ratios cancel quent overhead, the −0.2%
  sim bias, and any constant binary/box offsets. Judged against §3 thresholds at
  `max(floor, 3σ_q)`.
- **Absolute error** `sim_wall/real_wall − 1` reported alongside (secondary, caveated by
  P2's overhead table).
- **Rank/monotonicity:** per query, Spearman ρ between f_ach and real_wall across a
  knob's rows + baseline (expect exactly monotone), and between sim_wall and real_wall
  (acceptance: ρ = 1.0 for E1; ≥ 0.9 for E4/E5 with ≥3 points).

### 7.2 Structural metrics (from `--json` reports vs knobs=1 replay of the degraded trace)

- **Binding constraint match**: reported binding resource (executor_threads /
  gpu_memory / channel) equal between prediction and degraded replay. For E2's pressured
  queries the interesting version: sim says `gpu_memory` ⟺ real trace shows `Downgrading`.
- **Per-pipeline share error**: for the top-5 pipelines by baseline wall share, mean
  |Δ share| in percentage points between predicted and degraded-replay breakdowns.
  Report; flag > 10 pp.
- **Spill agreement (E2)**: per query, `forced_admissions > 0` ⟺ real `Downgrading`
  count > 0 (target ≥ 4/5 queries per row); also report sim forced_admissions vs real
  Downgrading counts as a calibration pair for v1's spill model.
- **Thread-utilization direction (E1/E5)**: sim predicts thread-busy% falls as transfers/
  compute dominate shift; check the degraded replay agrees in direction.

### 7.3 Per-knob accuracy verdict format (for validation-results.md)

One table per knob: rows = (query × factor), columns = f_ach, real slowdown, sim
slowdown, E_ratio, |abs err|, binding-match, notes; then a knob-level summary: median /
worst E_ratio, rank-ρ, pass/fail vs threshold, and the *sign* of the error with its
mechanistic explanation (each knob has a predicted error direction: E1-25 sim-pessimistic
on membw-bound queries; E2-below sim-pessimistic (no spill pricing); E3 sim direction
unknown a priori — that's the G1 measurement; E4 sim-pessimistic on compute-bound
queries; E5 with trace-derived f_ach should be closest to neutral).

### 7.4 Decision rules the results must emit

- **G1 (io instrumentation)**: if E3's |E_ratio| > 50% at v≈0.60 on scan-bound queries
  (equivalently: sim captures < half the real degradation, or over-predicts > 2×),
  G1 split-level instrumentation is promoted to the top of the v1 backlog. If
  |E_ratio| ≤ 25%, G1 is deprioritized below G2/G4. In between: judgment call, report the
  curve.
- **G4 (compute/HBM split)**: quantified by E4's per-query error spread and E6. If the
  spread across queries at one factor exceeds 25 pp, the per-kernel classification (WS2
  §5.1) is necessary, not optional.
- **Spill pricing (G5)**: E2's `real slowdown vs sim cliff` ratio is the first calibration
  datum; if real slowdown at E2-mid/lo is < 20% of the sim's predicted blocking cliff, v1
  must not ship a capacity knob without a spill model (the pressured traces from E2 are
  the calibration input).

### 7.5 E6 — coupling quantification (no new runs)

Using E1-25 (f=0.25 → free-SM bandwidth cap ≈ 38×~43 ≈ 1.65 TB/s < HBM): regress
per-query E1-25 `E_ratio` against a membw-boundedness proxy (per-query H2D+batch bytes ÷
wall from B1, or WS2 name-priors if a Tier-B profile exists). Expected: errors grow
pessimistic with membw share (WS5: saxpy retained 0.344 vs FMA's 0.250 at pct=25 — a
~38% wall gap the v0 knob cannot express, because co-setting `gpu_mem_bandwidth=0.305`
is inert under the `min(gpu_compute, gpu_mem_bandwidth)` rule). Deliverable: the measured
error-vs-membw-share slope — the concrete cost of G4, feeding §7.4.

### 7.6 Noise-floor reporting

The results doc leads with the two floors (P1 σ_q table incl. G-SELF selfcheck numbers;
P2 overhead table) so every subsequent error is read against them.

---

## 8. Budget summary

| Block | GPU-busy est. |
|---|---|
| P1+P2+P3 (floors) | ~21 min |
| B1 + analysis gates | ~4 min |
| E1 (flagship, 3 rows) | ~20 min |
| E2 (3 rows) | ~23 min |
| E4+E5 (3–4 rows) | ~20–25 min |
| B2+E3 (cold block, 15 short sessions) | ~37 min |
| Stretch B3+E7 | ~9 min |
| **Total** | **~125–140 min** (≤ 2.5 h; every session ≤ ~10 min) |

Disk: < 25 GB under `$EXP` (traces ~3–5 GB, scratch 16 GiB, sim outputs ~1 GB).

---

## 9. Open risks

1. **Idle-window scarcity**: a foreign process held 243 GB of VRAM at inventory time;
   capacity-sensitive rows need a nearly-empty GPU. The chunked ordering (§5.2) tolerates
   arbitrarily long waits, but calendar time is unbounded.
2. **E2-lo runtime**: spill cost at SF1000 is unmeasured; the guard caps it, possibly at
   the cost of losing q21 from the lowest point.
3. **Cross-talk co-sets are probe-interpolated** at our eater rates (WS4 matrix was
   measured at other rates; two CE cells provisional) — mitigated by in-session probes,
   but the DRAM leg is fundamentally unrepresentable in v0 and will contaminate E5 error
   on host-heavy queries in the optimistic direction.
4. **Fresh baseline may differ from the sample trace** (dirty-build deltas, box drift):
   E2 capacity points and query-set assumptions are re-derived from B1 at the analysis
   gate — but if q21/q9/q19 walls or working sets shifted radically, parts of §3's
   expected values need recomputation (procedure provided, §4.4).
5. **Cold-run column subsets unknown**: E3 session lengths (hence its 37-min estimate)
   depend on how many bytes each query actually reads cold; B2 measures this before the
   injected rows commit GPU time.
6. **v0 queue-order anchoring** (sim design §2.2) may distort structure at f=0.25 and in
   deep-capacity rows; wall metrics remain valid, structural metrics may degrade —
   reported, not hidden.
7. **Plan-change lever** may not alter plans for the chosen queries
   (`fuse_merge_pipelines` affects only fusable merge patterns); fallback lever named
   (`max_broadcast_join_size=0`), verified cheaply in B3 before E7 spends GPU time.
