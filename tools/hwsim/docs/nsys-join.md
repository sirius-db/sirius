# nsys Physics Join (WS10, v1 of gap G4)

Status: implemented + unit-tested against synthetic sqlite fixtures,
2026-08-04. **No real nsys capture existed on the box when this was built**
(shared-GPU rule: no profiling runs), so the ingestion is fixture-verified
against the schema documented in [`nsys-extraction.md`](nsys-extraction.md)
and defensively coded for first contact — see the runbook in §7.

Code: `tools/hwsim/sim/hwsim/physics/` (stdlib only). Spec inputs:
[`nsys-extraction.md`](nsys-extraction.md) (WS2),
[`simulator-design.md`](simulator-design.md) §4/§10 (WS6 knob semantics + G4),
[`compute-throttle.md`](compute-throttle.md) /
[`membw-throttle.md`](membw-throttle.md) (WS4/WS5 coupling measurements).

## 1. What it fixes (gap G4)

v0 spans are wall-clock: the `gpu_compute` knob scales *everything* inside a
`Computing` span (SM time, HBM time, launch overhead, host glue), and
`gpu_mem_bandwidth` is only a pessimistic `min()` alias. Preparing spans are
treated as pure link transfers although they contain SM-bound decompression.

The physics join decomposes every traced span into shares measured from a
paired nsys capture, giving **honest split semantics**:

| share (per span) | measured as | scales with |
|---|---|---|
| compute-bound kernel time | kernel-interval union, classified `compute` | `gpu_compute` |
| membw-bound kernel time | union share classified `membw` | `min(gpu_mem_bandwidth, gpu_compute × 1.27)` (§3) |
| unclassified kernel time | union share classified `unknown` | v0 conflated rule (`min(gpu_compute, gpu_mem_bandwidth)`) |
| explicit transfers | memcpy union (minus kernel overlap), per direction | per-size α+β curve × `min(c2c_bandwidth, cpu_mem_bandwidth)` (§3); D2D scales with `gpu_mem_bandwidth` |
| host remainder | span − GPU-busy | `cpu_compute` only |

Fractions (not absolute times) travel from the profiled run to the unprofiled
trace: **scheduling from Quent, physics from nsys, joined by structural key**
(query structure, pipeline id, task ordinal, operator position + op-id check)
— never by timestamp.

**Knob-semantics note (G4b, WS15 2026-08-05):** the "host remainder scales
with `cpu_compute` only" rule above is *per-span*; it is NOT a promise that
span host time is invariant under GPU knobs. On device-saturated lanes
(late-mat: device-level kernel-busy 82–100% of wall) most of the span "host"
remainder is emergent queue-wait for the shared device, and it is re-derived
by the engine's fluid device-compute resource: each task's knob-scaled
kernel time flows through a per-device capacity `max(1.0, baseline
device-busy fraction)`, so contention delay emerges from demand vs capacity
(simulator-design.md §3.4). The model engages when `gpu_compute` ≠ 1 or
`gpu_mem_bandwidth` is set AND the capture's **kernel-serialization
fraction** (device-timeline union/sum of kernel time, stored per query
window at ingest: `kernel_sum_ns` / `kernel_union_ns`) is ≥ 0.9 — i.e. the
lane's kernels demonstrably serialize. Lanes whose kernels co-run
(NVMe lane: 0.60–0.86) keep the per-span split semantics unchanged, and the
run warns that predictions are lower bounds if such a lane is also
device-saturated. Profiles ingested before G4b lack the diagnostic: the
model engages with a re-ingest warning.

## 2. Pipeline

```
nsys.sqlite ──reader.py──▶ rows (defensive: schema checks, enum fallbacks)
     │  NVTX 'Pipeline P Task T [...]' / 'Pipeline P: op (id=N)' ranges
     ▼
attribute.py: kernel/memcpy → correlationId → RUNTIME launch row →
  same-thread interval containment → (pipeline, task attempt, operator);
  per-window decomposition (interval unions; multi-stream overlap counted
  once; out-of-window kernel time counted as a diagnostic)
     │                       classify.py: overrides ▸ gpu-metrics ▸ name priors
     ▼
ingest.py: + curves.py (per-channel, per-log2-bucket α+β fits) + clock.py
  (utcEpochNs anchor + robust linear fit when trace and capture are the same
  run) + structure matching diagnostics + per-window kernel sum/union
  (the G4b serialization diagnostic)  ──▶  physics.json (schema.py)
     ▼
join.py: profile × QueryGraph → per-tid annotations (+ match-rate stats)
     ▼
integrate.py: retime spans, split Preparing, re-derive channel capacity,
  run the *unmodified* v0 engine with neutralized GPU knobs
```

### CLI

```bash
# once per capture (trace_dir is used for structural/clock validation):
python -m hwsim ingest-nsys <trace_dir> <nsys.sqlite> -o physics.json \
    [--overrides ncu_overrides.json]

# then on any simulate/sweep:
python -m hwsim simulate <trace_dir> --query-label tpch_q09_iter2 \
    --physics physics.json --knob gpu_mem_bandwidth=0.5
python -m hwsim sweep <trace_dir> --query-label tpch_q09_iter2 \
    --physics physics.json --sweep gpu_compute=0.5,1,2,4
```

Without `--physics`, simulate/sweep delegate to the v0 code paths unchanged.

## 3. Coupling laws (measured, encoded in `laws.py`)

1. **SM-issue cap on bandwidth** (`compute-throttle.md`, cross-talk section):
   a free SM sustains ~40 GB/s of streaming issue ⇒ achievable HBM bandwidth
   ≈ `free_SMs × 40 GB/s` (≈6.1 TB/s at 152 SMs vs ~4.8 TB/s HBM). Encoded:
   membw-bound time scales by `min(gpu_mem_bandwidth, gpu_compute ×
   SM_BW_HEADROOM)` with `SM_BW_HEADROOM = 1.27`. Consequences: raising
   `gpu_compute` alone never speeds membw-bound kernels; lowering it below
   ~0.79 starts throttling them (the knobs couple below the line).
2. **Grace C2C ⇄ host DRAM co-limit** (`membw-throttle.md` §5): C2C H2D reads
   host DRAM at line rate (D2H writes it) ⇒ link transfers scale by
   `min(c2c_bandwidth, cpu_mem_bandwidth)`. A C2C what-if is always a joint
   (c2c, cpu_mem) experiment on this machine.
3. **Decompress kernels are SM-bound on this box** (measured; memory note +
   `nsys-extraction.md` §5.1) ⇒ name-prior `compute`, so the Preparing kernel
   share follows `gpu_compute`, not the link.
4. **Per-size transfer regimes** (`nsys-extraction.md` §2.2c + memcpy-flag
   measurements): α+β fitted per log2(bytes) bucket per (direction, src, dst)
   channel; knobs scale β only, so small copies stop speeding up with the
   link (α floor) — `curves.py`.

## 4. Engine integration (what changed, what didn't)

`integrate.py` produces a transformed copy of the QueryGraph and runs the
engine with *neutralized* GPU knobs (`gpu_compute=1`,
`gpu_mem_bandwidth=None`, `c2c=1`), so nothing is scaled twice;
`gpu_mem_capacity` (pool admission) and `io_bandwidth` (G1) keep their v0
engine-level meaning. Since G4b the engine additionally hosts a fluid
device-compute resource (all knob logic stays in the physics layer: tasks
carry pre-scaled `dev_work_ns` and the derived capacity is knob-free):

- **Computing spans**: replaced by `span × Σ shareᵢ / multiplierᵢ` per §1.
- **Preparing spans** (the roadmap's "split Preparing into decompress (SM) +
  copy (link)"): the memcpy share stays a fluid-channel transfer (so
  contention remains emergent) re-timed via the α+β curve; the kernel + host
  shares become a `PHYS::PREP` pseudo-operator executed after the transfer
  (fluid approximation of the real interleaving; the sum is exact at
  knobs=1). Transfer shares under 0.5% of the window are folded into the
  pseudo-op (no channel service).
- **Channel capacity**: v0 used the peak aggregate of full-span rates. With
  the split, per-task demand rates rise by `1/f_xfer`, so capacity is
  re-derived by line-sweeping the *transfer-only* sub-windows at the traced
  positions (floored by the session-wide v0 peak, in quent-byte units —
  nsys-side achieved GB/s per channel is stored separately in the profile
  diagnostics because compressed payloads make the byte units differ), then
  scaled by the co-limited link multiplier.
- **Fallback**: any task/op without an annotation is re-timed with exactly
  the v0 conflated rule and counted; `retime` warns with the % of traced busy
  time that fell back. Zero-match joins warn loudly. Nothing is dropped.

At knobs = 1.0 the transformation is identity-preserving by construction
(shares sum to 1, curve factor = 1), verified by unit test
(`test_baseline_identity_with_v0`) and by an end-to-end check on the sample
trace: a structurally-unmatched physics profile reproduces the v0 selfcheck
wall exactly through the fallback path.

## 5. Profile format (`physics.json`, format_version 1)

Per capture: `source` (paths, utcEpochNs, overrides), `diagnostics`
(attribution match rates: % kernels attributed by count and by time, % kernel
time classified, unattributed kernel names, per-channel peak aggregate GB/s,
reader schema notes, per-window trace match + clock fit), `curves`
(per-channel per-bucket α+β), and `queries`: one entry per `sirius::query`
NVTX window with `pipelines: {pid: [task attempts in start order]}`, each
task carrying `prep` (span, f_xfer/f_comp/f_membw/f_unknown/f_host,
xfer_bytes, dominant channel, up to 64 (bytes, ns) copies) and `ops`
(op_id, span, the seven fractions, launch/sync ns diagnostics).

Downgraded/re-executed tasks appear as separate attempts (same nsys task id,
increasing `attempt`), matching Quent's convention of one TaskSpec per
attempt.

### 5.1 Per-span kernel-overlap fraction (WS20; the RTX q17/q20 cap)

Each `prep`/`ops` entry also carries **`f_kernel_overlap`**: the share of the
span's kernel-busy union NOT covered by a same-thread synchronization API
span — kernel time the launching thread ran *past* (hidden under concurrent
host work) instead of waiting on. Motivation: the RTX PRO 6000 external
validation's two pessimistic outliers (q17/q20: real ×≈1.0 at MPS-50, but
the split charged +15.6/+19.7%) are queries whose kernel time co-runs
entirely under host work — stretching those kernels does not stretch the
span.

At simulate time (`integrate.py`), spans whose overlap is **≥
`OVERLAP_CAP_MIN` (0.9)** charge the hidden kernel share as
`max(hidden_base, hidden_scaled − f_host)` instead of `hidden_scaled` — the
span pays max(host, kernels) in the hidden window, with the host residue as
absorbing headroom (deep throttles still pay the excess). Gates and
guarantees:

- **knobs=1 identity** and bit-identical behavior on pre-WS20 profiles
  (missing field ⇒ 0 ⇒ no capping) — B1 selfcheck + a stored E1-50P
  prediction reproduce exactly.
- **Section-7 split path only**: when the G4b fluid device engages
  (kernel-serialization ≥ 0.9) the cap stands down — serialized kernels are
  not hidden. Verified on the LN lane: re-ingested profiles predict
  identically to the stored ones (gate engages first).
- **Below the gate the full charge is kept**: on the GB300 R1 capture,
  kernel time clusters at overlap < 0.5 (the uncovered sliver is launch-side
  latency; the stretch lands in the sync wait). The gated cap re-scores the
  whole §7.2 E1/E4 grid within **0.70 pp** of the stored errors (24/25
  within ±15%, same sole exception q21@E1-25); an ungated linear discount
  moved the medians 1–3.5 pp more optimistic and was rejected.

**Validation status:** the GB300 captures contain no query where the cap
changes predictions materially (max 0.7 pp) — the decisive datum (q17/q20,
real ×≈1.0) lives on the RTX PRO 6000 box. To validate there (worktree
`.claude/worktrees/simulator-timing-output-prediction-e02735`), re-ingest the
paired baseline so the profile carries the new field, then re-score the
MPS-50 what-if exactly as in `hwsim_run/RESULTS.md`:

```bash
python -m hwsim ingest-nsys hwsim_run/telemetry_data/019fd246-a2e5-* \
    hwsim_run/nsys/paired_base.sqlite -o hwsim_run/physics_paired_ws20.json
python -m hwsim simulate hwsim_run/telemetry_data/019fd246-a2e5-* \
    --query-label <q17/q20 labels> --physics hwsim_run/physics_paired_ws20.json \
    --knob gpu_compute=0.5
```

Expected if the hypothesis holds: q17/q20 drop from +15.6/+19.7% toward 0
while the other 20 queries stay within their ±10% band.

## 6. Fixture methodology (how this is tested without a GPU)

`tests/nsys_fixture.py` builds sqlite files conforming to the documented
schema: `NVTX_EVENTS` (eventType 59/60/34, textId→StringIds), CUPTI kernel/
runtime/memcpy tables with correlationIds, enum tables, session start time,
optional `GPU_METRICS`. 66 new unit tests cover: happy-path attribution;
kernels with no enclosing NVTX range / no RUNTIME row / wrong thread
(reported, never dropped); multi-stream overlap unions; re-executed task
attempts; classification via priors, override table and Tier B metrics;
clock-skew alignment (ppm slope + outlier rejection at epoch-ns magnitudes);
α+β fitting and the small-copy α floor; the split-knob laws analytically
(compute-only scaling, membw coupling, host/cpu_compute, co-limited
transfers); schema-version and missing-table diagnostics; fallback identity
with v0; CLI wiring (v0 delegation when `--physics` is absent).

## 7. First real capture — runbook

1. Capture (Tier A, identical to `performance_test.py --mode nsys-profile`):

   ```bash
   nsys profile \
     --trace=cuda,nvtx \
     --sample=none --cpuctxsw=none --cudabacktrace=none \
     --capture-range=cudaProfilerApi --capture-range-end=stop \
     --stats=false --export=sqlite \
     --output=<out> --force-overwrite=true \
     build/release/duckdb -unsigned -f <query.sql>
   ```

   Follow the shared-box rule (idle `nvidia-smi` window). For kernel
   classification calibration add
   `--gpu-metrics-devices=all --gpu-metrics-frequency=10000` (Tier B) once
   per (dataset, config).

2. Schema sanity (the *(verify)* marks from `nsys-extraction.md`):

   ```bash
   sqlite3 cap.sqlite ".schema CUPTI_ACTIVITY_KIND_KERNEL"
   sqlite3 cap.sqlite "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME;"
   sqlite3 cap.sqlite "SELECT eventType, COUNT(*) FROM NVTX_EVENTS GROUP BY eventType;"  # expect 59,60,75
   sqlite3 cap.sqlite "SELECT text FROM NVTX_EVENTS WHERE text LIKE 'Pipeline % Task %' LIMIT 3;"
   ```

3. Ingest + validate. The run must check, in order:
   - `ingest-nsys` completes with **no reader notes** about missing
     tables/columns (any note = schema drift → fix `reader.py` first);
   - **kernel attribution ≥ ~95%** of kernel time (the NVTX chain is emitted
     unconditionally, so large gaps mean capture-range or thread-attribution
     problems, not missing labels);
   - `GPU_METRICS` value semantics: `classify.py` assumes percent-of-peak
     (`METRICS_MEMBW_PCT`) — check against a known-membw kernel (e.g. a cudf
     gather) and a known-SM-bound one (decompress) and recalibrate the
     thresholds if the values are raw counters;
   - the per-channel α+β curves are monotone in the bucket rates and the
     large-bucket pooled H2D rate is in the measured C2C range (~340–380
     GB/s per `membw-throttle.md` §2; D2D near the flag-capped ~470 GB/s);
   - if the capture was taken with Quent enabled on the same run: clock fit
     reports `same_run` with slope ≈ 1 ± 1e-5 and rms < 100 µs;
   - selfcheck-equivalence: `simulate --physics` at knobs=1 on the paired
     trace should land within the v0 self-consistency envelope (±1.2%); a
     bigger gap means the fraction decomposition or the Preparing split is
     mis-measuring;
   - sweep `gpu_compute × {0.5, 1, 2}` and `gpu_mem_bandwidth × {0.5, 1}`
     and compare against WS4/WS5 throttled runs (the WS7/WS8 loop) — the
     split predictions should now diverge from v0 exactly where the
     kernel-class mix says they should.

4. Known limits to keep in mind when reading results: SM-driven zero-copy
   C2C traffic is invisible to memcpy rows (`nsys-extraction.md` §5.2) and
   stays inside "kernel time"; host-side shares are not split into
   compute-vs-memstall (§5.4); the classification priors are priors — feed
   ncu spot-check results back via `--overrides`.
