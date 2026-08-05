# The definitive cross-machine experiment — GB300 trace → RTX PRO 6000 prediction

**Status: GB300 side EXECUTED + predictions frozen 2026-08-05 (WS21)** — see
[cross-machine-predictions-README.md](cross-machine-predictions-README.md) for the frozen
CSVs, capture provenance, and the exact grading command. Only the grading half remains,
on the RTX PRO 6000 box (one command, the owner runs it).

**Claim under test:** spec-sheet target mode ([spec-sheet-mode.md](spec-sheet-mode.md))
can predict the *absolute per-query walls of a machine we never measured against*, from a
source trace + the target's advertised specs. The
[external validation](external-validation-rtx-pro-6000.md) already proved the physics
*ratios* transfer cross-platform (+0.2% median, zero recalibration); this experiment
grades the *derated absolute* transfer — the derating table, the capacity/link/io ratios,
and the (unvalidated) cpu_compute knob all at once.

**Design:** capture a TPC-H **SF100** baseline on GB300 (matching the RTX box's dataset
and config), predict all 22 queries with
`--target rtx-pro-6000-blackwell.yaml`, carry one CSV to the RTX box, grade against the
already-captured RTX baselines (its artifacts live there per the external report:
worktree `.claude/worktrees/simulator-timing-output-prediction-e02735`,
`hwsim_run/telemetry_data/`, baseline session `019fd242-f189…` × 3 iters). The RTX box
is the *grader only* — no RTX measurement feeds the prediction.

## 1. GB300 side (this box)

### 1.0 Dataset — SF100 parquet does NOT exist here (checked 2026-08-05)

`/localhome/local-faramburu` has `tpch_parquet_sf10` and `tpch_parquet_sf1000` only.
Generate SF100 in the same per-table-directory layout (the repo's `dataset-manager`
skill does this; raw fallback):

```bash
pixi run python - <<'EOF'
import duckdb, os
con = duckdb.connect()
con.execute("INSTALL tpch; LOAD tpch;")
con.execute("CALL dbgen(sf=100)")
base = "/localhome/local-faramburu/tpch_parquet_sf100"
for t in ["lineitem","orders","partsupp","part","customer","supplier","nation","region"]:
    os.makedirs(f"{base}/{t}", exist_ok=True)
    con.execute(f"COPY {t} TO '{base}/{t}' (FORMAT parquet, PER_THREAD_OUTPUT true)")
EOF
```

(~35 GB on disk; needs ~150 GB RAM headroom for in-memory dbgen — fine here. Check
`nvidia-smi` for foreign GPU users before any capture, per the shared-box protocol.)

### 1.1 Capture config — MATCH THE RTX BOX, not this box's tuned config

The RTX baselines were captured with (external report, Environment):

| setting | RTX value | GB300 action |
|---|---|---|
| executor threads | **4** (ship default) | capture with 4 — do NOT use the SF1000 tuned config's 8 (the sim replays source thread counts; this is not descriptor-driven) |
| `usage_limit_bytes` | **80 GB** absolute | GB300 uses a fraction — capture however is convenient, then override the derived capacity knob (§1.3) |
| `reservation_limit_fraction` | 0.8 | match |
| exporter | **ndjson** | the bundled `tpch_telemetry_sirius.yaml` says `postcard` — change it (external-report defect #4) |
| engine | branch `claude/sirius-hardware-simulation-25edbe` (PR #1424) | build the CURRENT head and record the commit; the RTX artifacts were made at `e38a9d84` — if walls must be compared against those existing sessions, the engine-side changes since then must be timing-neutral (they are hwsim/tooling-only so far; re-verify at run time) |
| gates | late-mat/fused dark (env unset), no pins | match (ship defaults) |

Sessions to capture (single process each, labeled, 22 queries):
1. **baseline** x 3 iters (quent ndjson on) — the prediction input;
2. **paired quent+nsys** x 2 iters — `nsys profile --trace=cuda,nvtx
   --capture-range=cudaProfilerApi --capture-range-end=repeat` around
   `CALL profiler_start()/profiler_stop()` — the physics profiles.
   Expect range-merging (fewer reports than queries — harmless, see
   validation-results.md §9.6.1).

Before the capture: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` and verify with
`iostat -x 1 /dev/nvme0n1` during iteration 1 that reads actually hit NVMe — SF100 may
fit in page cache on this box, and a cache-warm source trace breaks the io knob's basis
(the RTX box reads at 14.6 GB/s from NVMe; the knob assumes the source's GPU_SCAN spans
contain device-read time). If warm-vs-cold walls differ, use the cold session.

### 1.2 Ingest the physics profiles

```bash
cd tools/hwsim/sim
for rep in /path/to/nsys_reports/*.nsys-rep; do
  nsys export --type sqlite "$rep"
done
mkdir -p ../experiments/nsys/XM-SF100
i=0
for sq in /path/to/nsys_reports/*.sqlite; do
  i=$((i+1))
  python3 -m hwsim ingest-nsys <paired_trace_dir> "$sq" \
      -o ../experiments/nsys/XM-SF100/physics_XM$i.json
done
```

### 1.3 Predict (the packaged script)

```bash
# source pool bytes: the "memory space: ... tier=GPU ... limit=<BYTES>" line
python3 -m hwsim info <baseline_trace_dir> | grep 'tier=GPU'
# capacity override = RTX absolute pool / GB300 pool, e.g. 80e9 / 241.8e9:
CAP=$(python3 -c "print(80e9 / <SOURCE_POOL_BYTES>)")

cd tools/hwsim/experiments/bin
# primary prediction (derived vector, incl. the unvalidated cpu_compute=0.667)
python3 predict_cross_machine.py <baseline_trace_dir> \
    --target ../../hw-descriptors/rtx-pro-6000-blackwell.yaml \
    --source ../../hw-descriptors/gb300.yaml \
    --physics-dir ../nsys/XM-SF100 --iter 2 \
    --knob gpu_mem_capacity=$CAP \
    -o predictions_sf100.csv
# sensitivity arm: freeze host time (the known-weakest knob)
python3 predict_cross_machine.py <baseline_trace_dir> \
    --target ../../hw-descriptors/rtx-pro-6000-blackwell.yaml \
    --source ../../hw-descriptors/gb300.yaml \
    --physics-dir ../nsys/XM-SF100 --iter 2 \
    --knob gpu_mem_capacity=$CAP --knob cpu_compute=1 \
    -o predictions_sf100_cpufrozen.csv
```

Both CSVs are frozen BEFORE any grading (they contain nominal + optimistic walls,
per-query path and provenance). Commit them or hash them — this is the pre-registration.

## 2. RTX box side (the owner) — one command

Copy the two CSVs and this branch to the RTX box, then:

```bash
python3 tools/hwsim/experiments/bin/grade_cross_machine.py \
    predictions_sf100.csv \
    <worktree>/hwsim_run/telemetry_data/<baseline-session-uuid>   # 019fd242-f189…
```

(Repeat with `predictions_sf100_cpufrozen.csv`.) The script computes real walls as the
median of iters 2,3 from the measured baseline session, prints per-query E% for both
vectors, medians, the time-weighted suite aggregate, Spearman rank ρ, and band coverage.

## 3. Expected confounds (list them before looking at the grade)

1. **cpu_compute** (0.667, UNVALIDATED, host-arch-crossing): on this NVMe lane spans are
   85-93% host time — the single largest exposure. That is why the frozen arm exists:
   the spread between the two arms bounds the knob's influence; if |grade difference| >
   |error|, the experiment cannot attribute the residual.
2. **Config deltas**: executor threads (mitigated: capture at 4), `usage_limit`
   (mitigated: capacity-knob override), scan-manager thread counts (unmitigated —
   replayed from source).
3. **Engine version**: RTX baselines are `e38a9d84`; predictions replay a trace from the
   current head. Timing-relevant engine changes between them void the comparison.
4. **q11**: reproducible +10-15% replay outlier on the RTX box (2 of 3 iterations,
   external report Result 1) — grade q11 with and without.
5. **SF100 short queries**: several run <150 ms on the RTX box; nsys/quent overhead and
   run-to-run variance are proportionally larger (the suite campaign saw +12% Tier-A
   overhead on an 80 ms window).
6. **Page cache on the source side** (§1.1): a warm GB300 capture underweights the io
   share the knob then scales by 2.24x.
7. **L2 residency**: 128 MB L2 on the RTX card can make source-membw-bound kernels
   L2-resident (the kit itself hit 3 PB/s there) — membw-tier queries may beat the 0.26
   ratio; this error lands *pessimistic*.

## 4. What counts as success (per tier, decided in advance)

| scope | success | outstanding | failure |
|---|---|---|---|
| suite time-weighted E (nominal arm) | within **±30%** | within ±15% | beyond ±50% |
| per-query median \|E\| | ≤ 30% | ≤ 15% | > 50% |
| queries whose predicted binding is `gpu_device`/`gpu_memory` (validated tiers dominate) | \|E\| ≤ 20% | ≤ 10% | > 40% |
| rank ρ(real, nominal) across 22 queries | ≥ 0.8 | ≥ 0.9 | < 0.6 |
| band `[nominal, optimistic]` | brackets ≥ 1/3 of reals | ≥ 1/2 | — (informative) |

Rationale: the within-box ratio validation earned ±10-15% tiers; a cross-machine
*absolute* prediction stacks the derate table (~±10% per class), the fma-anchor spread
(~7%), and an unvalidated host knob on 85-93%-host spans — ±30% is the honest first bar.
Whatever the outcome, the per-query decomposition (which tier failed) feeds directly
back into §3's derate ranges — and a passing grade makes the RTX box the third derate
anchor for free.

## 5. Artifacts to keep

GB300: the SF100 baseline + paired traces (`experiments/traces/XM-*`), profiles
(`experiments/nsys/XM-SF100/`), both prediction CSVs, engine commit + config snapshot.
RTX: grader stdout for both arms. File the result table into
[validation-results.md](validation-results.md) as §10.
