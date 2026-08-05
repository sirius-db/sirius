# Late-mat lane — running the compute-bound Sirius for `gpu_compute` validation

**Status:** written 2026-08-05 at the PR #1409 merge (merge base `61a036ff`, PR head
`188e0c5a`). Nothing in this document has been re-measured on this branch — every number
is quoted from the PR's own bench notes (`bench/sf1000-repro/README.md`,
`bench/sf1000-repro/NEXT-STEPS.md`) and its commit messages.

## Why this lane exists

E1 (`gpu_compute` via MPS `ACTIVE_THREAD_PERCENTAGE`) failed on `dev@b77aa438`
([validation-results.md](validation-results.md) §3.1): cutting the SM ceiling 25% slowed
real queries only **2.5–7.7%**, so the v0 prediction (+33%) had nothing to bite on — the
engine was memory/latency-bound, like the saxpy probe. PR #1409's stack removes most of
that memory-bound work (decode-time filtering, deferred column materialization,
group-by-rowid): its own profiling states **GPU-busy is 91–97% of wall** and
"scheduling/parallelism knobs cannot help; only removing work or raising achieved
bandwidth moves the clock" (README §"What was measured and rejected", NEXT-STEPS §Session
3). A compute-bound engine is the baseline the `gpu_compute` knob needs: under an SM
throttle the real slowdown should now be far closer to `1/f`, giving E1 a signal to
validate (or refute) against.

That is a **hypothesis to be measured by WS8**, not a result: decompression is SM-bound
(it scales with the SM ceiling), but hash joins and shuffles still carry
bandwidth-sensitive phases. Expect E1 fidelity to improve substantially, not to become
exact.

## The gates (all env, all default OFF — the PR ships "dark")

Gate off ⇒ byte-identical classic behavior. Convention for every `SIRIUS_EXP_*` flag:
*set and not `"0"`* means on (single authoritative reader:
`src/include/late_mat/column_origin.hpp`, `src/scan_manager/sirius_scan_manager.cpp`).

### Primary gates (the ship config, banked 2026-08-04 at 6.654 s)

```bash
export SIRIUS_EXP_FUSED_SCAN_FILTER=1   # decode-time filtering + selection-consuming decompression
export SIRIUS_EXP_LATE_MAT=1            # v1: stop-port deferral via pin-order rowids
export SIRIUS_EXP_LATE_MAT_V2=1         # v2: planner lifetime pass + group-by-rowid (implies v1)
export SIRIUS_EXP_LATE_MAT_V3=1         # v3: FD/composite-key group-by-rowid riders (implies v2)
export SIRIUS_LATE_MAT_PIN_UNIQUE_COLS=all   # pin-time uniqueness probes (enables GBR proofs)
```

Sub-gates imply parents: `V3` does nothing without `V2`, `V2` nothing without
`SIRIUS_EXP_LATE_MAT`. The stack can never invert.

### Secondary knobs (engine defaults are the measured ship values — do not set unless sweeping)

| env | default | meaning |
|---|---|---|
| `SIRIUS_EXP_LATE_MAT_DEFER` | ON (under main gate) | the defer-policy walk itself |
| `SIRIUS_EXP_LATE_MAT_COMPRESSED` | OFF | compressed-origin scans defer too (killed on TPC-H) |
| `SIRIUS_LATE_MAT_COUNT_DEFER` | OFF | count-on-deferred (killed on TPC-H, q13 noise) |
| `SIRIUS_LATE_MAT_MIN_VALUE_BYTES` | 32 | min deferred-bundle B/row (uncompressed origins) |
| `SIRIUS_LATE_MAT_MIN_VALUE_COMPRESSED` | 32 | same, compressed origins |
| `SIRIUS_LATE_MAT_MASK_SEL` / `_DENSE_SEL` | 1.0 / 0.35 | selection-shape admits |
| `SIRIUS_LATE_MAT_MIN_BOUNDARIES` | 4 | min ride length (pipeline boundaries) |
| `SIRIUS_LATE_MAT_GBR_MIN_GROUP_ROWS` | 0 (inert) | GBR volume floor; ships disabled, fail-open |
| `SIRIUS_EXP_FUSED_SCAN_MAX_SEL` | 0.35 | RULE-2 write-skip bail threshold |
| `SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL` | 0.10 | tier-B re-admission |
| `SIRIUS_EXP_FUSED_SCAN_K4_MAX_SEL` | 0.15 | K3-vs-K4 payload pick crossover |
| `SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER` | 1 | dynamic membership sources kept per scan |
| `SIRIUS_EXP_FUSED_SCAN_DIAG`, `SIRIUS_FUSED_DIAG` | OFF | deterministic decision tracing |

## The gates alone are NOT enough

Both features key off **GPU-tier pinned tables**: late-mat origins are stamped by the
pinned-cache provider (`cached_databatch_provider`, needs the pin-time
`late_mat_handle`), and fused scan-filter needs **compressed** pinned chunks whose
simpatico plans can answer predicates (dictionary roots). A fresh-read (non-pinned)
parquet scan lights neither. The full compute-bound configuration is the repro kit's —
now in-tree at `bench/sf1000-repro/run.sh`:

```bash
# per-table GPU-tier pins (all eight)
for t in LINEITEM ORDERS PART CUSTOMER SUPPLIER NATION REGION PARTSUPP; do
  export "SIRIUS_PIN_TIER_$t=gpu"
done

# compression + plans + JIT evaluator (SIRIUS_PRE_SQL is honored by
# test/tpch_performance/performance_test.py as of this PR)
export SIRIUS_PRE_SQL="SET pin_table_compression = true; \
SET pin_table_input_compression_plan_dir = '<repo>/bench/sf1000-repro/plans'; \
SET expression_evaluator_strategy = 'ast_jit'"
```

Plus:

- **Config:** `bench/sf1000-repro/sirius-sf1000.yaml` (GB300-tuned:
  `scan_task_batch_size: 8GB` is the one knob that mattered). For the hwsim lane, flip
  `telemetry.enable_quent: true` — the kit ships it `false` — and keep the ndjson
  exporter default.
- **Patched libcudf** via `LD_PRELOAD` (`bench/sf1000-repro/build-libcudf.sh`;
  `run.sh` hard-fails without it). It carries the `strings::like` backtrack fix
  (q13 −36.5%) and the memcpy-flag threshold (q9 −5.8%). The gates work on stock
  libcudf, but the "6.654 s compute-bound" configuration is defined *with* the patched
  lib — and baseline vs throttled runs must use the **same** lib (the
  sm_103a-vs-sm_100 arch trap invalidates cross-lib comparisons).
- **Plans dir:** `l_shipinstruct` must stay `dictionary` in the lineitem plan or
  decode-time predicate pushdown silently no-ops (run.sh header comment).
- **Dataset:** SF1000 parquet, `DATA=/localhome/local-faramburu/tpch_parquet_sf1000`
  on the GB300 box.

Trace attribution: with this branch's follow-up commit, the engine `Init`
`custom_attributes` snapshot records `late_mat.enabled/v2/v3/defer/compressed`,
`late_mat.pin_unique_cols`, and `fused_scan_filter.enabled`
([ws9-new-fields.md](ws9-new-fields.md) §4) — a trace tells you itself which lane
produced it.

## Which queries benefit most (from the PR's own bench notes)

| feature | flagship wins (as measured by the PR) |
|---|---|
| late-mat v1 (stop-port deferral) | q10 −47..−49 ms in every measured pair (suite 6.918 → 6.860) |
| late-mat v2 (group-by-rowid) | **q10 −42.8%** (0.464 → 0.266 s, −198 ms, byte-identical ×4); suite 6.912 → 6.677 single-binary (−3.4%) |
| late-mat v3 (FD riders) | q10 −14 ms further (ship config 6.654 s); explicitly *no other* TPC-H claim — shipped as a generality capability |
| fused scan-filter (iterations 1–7) | suite 8.180 → 6.918 (−15.4%) cumulatively; notable: q12 −35.8% (K6 masked gather), q19 −21.4% (dict predicate pushdown + OR-branch derivation), campaign-wide dynamic-filter masks in it7 |

So for the E1 rerun, **q10** is the single most late-mat-shaped query, and **q12/q19**
the most fused-scan-shaped. Caveat from the kit's own variance notes (README §Variance,
consistent with [sf1000-query-variance](validation-results.md) experience): **q7/q8/q10/q19
swing 13–28% between runs** even at best-of-3 — pair baseline/throttled runs
back-to-back and prefer the stable big queries (q1/q9/q13/q21) for the E_ratio gates,
using q10/q12 as directional evidence only.

## What this changes for the validation lane

1. **New baseline required.** All E1 rows (and any other knob rows meant to be compared
   against them) must be re-anchored on a trace from THIS branch's binary with the lane
   config above — the old `dev@b77aa438` baseline is a different engine.
2. **Expected E1 outcome:** real slowdown under MPS 75% should move well above the old
   2.5–7.7% band (the work that stayed flat was the memory-bound part the PR removed).
   If it doesn't, that is itself a finding: the residual is bandwidth-bound and the v0
   `Computing ÷ f` model needs the roofline split regardless.
3. **`io_bandwidth` rows shrink.** With all eight tables pinned GPU-tier there are no
   fresh reads at steady state — `io_request` events (G1) will be near-absent in lane
   traces. I/O-knob validation stays on the un-pinned classic lane.
4. **`gpu_mem_capacity` rows get tighter.** The pins occupy most of HBM by design
   (253.9 GB of 256 GB at `scan_task_batch_size: 8GB`); capacity-knob experiments on
   this lane will hit the PR's new unsatisfiable-reservation fail-fast guard rather
   than livelocking — a behavior change to note when comparing against old runs.
5. **Gate provenance is in-trace** (`late_mat.*` attrs), so the parser can hard-fail a
   mixed-lane comparison instead of silently averaging two engines.
