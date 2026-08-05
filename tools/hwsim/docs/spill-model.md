# Spill / Downgrade Model (gap G5) — replay bookkeeping + calibrated prediction

**Status: implemented and calibrated, 2026-08-05** (WS13). Code:
`tools/hwsim/sim/hwsim/engine.py` (`SpillParams`, spill modes), parsing in
`build.py`/`model.py`. Fixes the two G5 failures from
[validation-results.md](validation-results.md):

1. **Replay**: spill-heavy traces self-checked at **+2743%** (v0 could not
   replay Downgrading events at all) → now **+0.25…+0.41%** on E2-mid q21.
2. **Prediction**: `gpu_mem_capacity` below the spill knee under-priced
   reality 2.4–5.6× *in the dangerous direction* (real ×11.2 vs predicted
   ×2.0; real ×55.2 vs ×23.1) → now within **+39% / −20%** at the shipped
   default (both pressure points), with an honest one-sided held-out result
   (see §5).

## 1. What the traces revealed the mechanism actually is

Everything below is measured from the E2 pressure captures
(`experiments/traces/E2-mid`, 64 GB pool; `experiments/traces/E2-lo-q9`,
38 GB pool) plus the engine code (`src/pipeline/gpu_pipeline_executor.cpp`,
`src/downgrade/downgrade_executor.cpp`, `docs/super-sirius/memory-management.md`):

- **The cost is OOM-reschedule thrash, not downgrade waits.** E2-mid q21:
  13.5–13.8 k `Downgrading` events per iteration, but their traced waits
  (Downgrading→Preparing) total **<1 s**. The wall (195 s vs 17.4 s
  baseline) is 13.6 k **failed task attempts** (`Finalizing{success:false}`,
  OOM-rescheduled) burning **639 s compute + 39 s prep** of thread time —
  threads 91–93% busy, mostly on waste. That is why "GPU at 8% util":
  executor threads are busy, SMs are not.
- **Thrash is stage-local.** All 13.6 k failures sit in ONE pipeline pair
  (the big `PARTITION` + the `GPU_SCAN` feeding it) at ~50 attempts per
  logical task; every other pipeline ran clean *through the same epoch*.
- **Failed attempts make partial progress.** `oom_reschedule_exception`
  carries `operator_input_output_data` and the retry resumes at the failing
  operator (`reschedule_intermediate` placements) — consumed inputs are
  freed, partial outputs survive.
- **Intermediates escape the pool by being born downgraded.** The E2-mid
  session moved only **22.9 GB** GPU→HOST by tier *conversion* (the 7.7 GB
  host pinned space is the window) but **1153 GB** HOST→GPU in retry
  Preparing spans at ~29 GB/s effective — i.e. under pressure new batches
  materialize host-side and consumers pay the re-upgrade, repeatedly.
- **Reservation-dominated pressure blocks, it does not downgrade.** The
  marginal q9@0.25 point (pool 99.6% peaked by *reservations*) shows **zero**
  Downgrading events — `make_reservation` waits when running tasks will free
  the bytes; the downgrade path fires only when **resident data** overshoots.

## 2. The model

Three engine modes (`spill_mode="auto"` resolves per graph/knobs):

| mode | when (auto) | behaviour |
|---|---|---|
| `off` | unpressured trace, `gpu_mem_capacity == 1` | v0 semantics, **byte-identical** (regression bar) |
| `replay` | the trace contains Downgrading / failed tasks | pure bookkeeping: memory-blocked heads evict idle resident batches at **zero time cost** — every real cost is already in the traced spans (downgrade waits inside `grant_ns`, re-upgrades inside retry Preparing spans, recompute waste = the traced failed attempts themselves, replayed as tasks) |
| `model` | capacity knob moved on an unpressured trace | predictive mechanism below |

**Model-mode admission** (mirrors `manager_loop` + downgrade executor):

1. Head fits (`reserved + resident + need + rematerialize ≤ cap`) → admit.
   Demoted inputs re-enter the pool and charge `bytes/upgrade_rate` to the
   task's prep (the ask is clamped to capacity, as the engine clamps to
   `get_max_memory()`).
2. Doesn't fit but **waiting suffices** (`resident + need ≤ cap`; the
   deficit is held by active reservations) → block, v0-style. No Downgrading
   is emitted in this regime in reality.
3. **Resident overshoot** → downgrade sweep: demote idle (unpinned,
   non-input) resident batches, LRU by publish order, down to
   `downgrade_stop_fraction × cap` (engine hysteresis), **bounded by free
   HOST-pool capacity**; the manager stalls
   `downgrade_base_ns + bytes/downgrade_rate` (admission serialized, running
   tasks continue).
4. Still short → **dispatch-and-OOM spin** (what the engine really does:
   partial grant → run → OOM → reschedule). One attempt: a thread slot for
   `oom_cycle_ns + f×compute`, then an off-thread `retry_backoff_ns` and
   re-queue at the tail. Each attempt **banks progress**
   `f = clamp(available/(need+rematerialize), min_progress, 1)`
   (resume-at-operator), **frees `f` of its input batches** and publishes
   `f` of its outputs, of which `spin_output_host_fraction` (β) are **born
   on HOST** (consumers pay re-upgrade) and `1−β` stay GPU-resident. A task
   whose banked progress reaches 1 completes directly; `max_oom_retries`
   (=100, the engine's budget) force-admits as a safety valve.

Slowdown is then **emergent**: attempts-per-task, thrash-epoch length, and
which pipelines thrash all come out of the flow model, not from a formula.

## 3. Parameters — measured vs calibrated

| param | value | provenance |
|---|---|---|
| `oom_cycle_ns` | 50 ms | **measured**: median failed-attempt span 47–50 ms on both captures |
| `min_progress` | 0.02 | **measured**: ~50 attempts per thrashed logical task on both captures (12 608 attempts / ~235 partition tasks at E2-mid; same floor at E2-lo) |
| `downgrade_rate` / `upgrade_rate` | 30 / 29 GB/s | **measured**: InTransit D2H conversions; retry-prep effective H2D |
| `retry_backoff_ns` | 50 ms | engine reschedule backoff (executor code comment) |
| `downgrade_stop_fraction` | 0.7 | engine config (`memory-management.md`) |
| `max_oom_retries` | 100 | engine `MAX_RETRIES` |
| HOST pool bound | from trace (`memory` space, ×`cpu_mem_capacity`) | conversion-window bound (7.7 GB on the E2 box) |
| `spin_output_host_fraction` (β) | **0.47 (CALIBRATED)** | the one free dial: how much intermediate output escapes the pool |

## 4. Calibration methodology

Baselines: **B1** graphs (unpressured, q21_iter3 = 17.45 s, q9_iter4 =
12.50 s — the medians the validation campaign used). Targets: real E2 walls
(q21@0.2514 → ×11.242; q9@0.1493 → ×55.19 = 689.9 s/12.5 s). All other
parameters frozen at their measured values; β fit by bisection on ONE point,
the other held out; protocol run in both directions.

## 5. Results

### Held-out prediction (the honest protocol)

| calibrated on | β | fit on calibration point | held-out point | predicted | real | held-out error |
|---|---|---|---|---|---|---|
| q9@0.1493 | 0.316 | ×55.7 (+1.0%) | q21@0.2514 | **×31.4** | ×11.24 | **+179%** ✗ |
| q21@0.2514 | 0.472 | ×15.6 (+38.8%, best reachable — see bistability below) | q9@0.1493 | **×43.9** | ×55.19 | **−20.4%** ✓ |

### Shipped default (β = 0.47, joint two-point fit)

| point | v0 predicted | new predicted | real | v0 error | new error |
|---|---|---|---|---|---|
| q21 @ 0.2514× | ×2.005 | **×15.6** | ×11.242 | **−82%** (dangerous under) | **+38.8%** (over-warns) |
| q9 @ 0.1493× | ×23.12 | **×43.9** | ×55.19 | **−58%** | **−20.4%** |
| q9 @ 0.2514× (marginal fit) | ×1.000 | ×1.000 | ×1.001 | ✓ | ✓ |
| q1/q13/q19 @ 0.1493× | ×1.000 | ×1.000 | ×0.999–1.004 | ✓ 3/3 | ✓ 3/3 |
| q21 @ 0.5029× | ×1.000 | ×1.000 | ×0.995–1.005 | ✓ | ✓ |
| q21 @ 0.1493× (guard-dropped) | ×3.03 | **×35.4** | >×51 projected | wrong order | right order |

Direction and order of magnitude are now correct everywhere, and the error
is no longer in the dangerous direction at the shallow point (v0 said "mild
×2" where reality melts down ×11; the model now over-warns slightly).

### Replay fidelity (Stage 1)

| trace | before | after |
|---|---|---|
| E2-mid q21 iter1/2/3 (13.5–13.8 k Downgrading, 14.4–14.7 k tasks) | **+2698 / +2761 / +2743%** | **+0.41 / +0.24 / +0.25%** |
| E2-mid unpressured rows (q1/q9/q13/q19) | ±0.13% | ±0.13% (unchanged, value-identical) |
| E2-lo-q9 main epoch (27 906 tasks, 27.5 k Downgrading, 671.4 s) | **unreplayable** (query/plan/memory ndjson lost when the session was killed) | **−0.00%** (queries synthesized from task extents + batch-dataflow clustering; walls exclude planning/collection tails) |
| E2-lo-q9 iter2 partial epoch (13 136 tasks) | unreplayable | −1.13% |

Note: E2-lo-q9's blocker was parse-level; its truncated trace also lost the
memory-space entities, so its replay runs with an unbounded pool and the
traced attempt structure carries all costs. The +2743% pathology (static
residency permanently over capacity → total admission serialization) is
fixed by replay-mode eviction, verified on E2-mid.

### Regression bar

- 109/109 unit tests (93 existing + 16 new analytic spill tests).
- Sample trace (66 queries): median 0.17% / p90 0.52% / worst 1.20%
  (q04_iter2) — exactly the documented v0 result.
- B1, P1a, E4-73 selfchecks value-identical to the stored campaign CSVs.
- All 5 null capacity points: ×1.000.

## 6. Honest limitations — the mechanism resists a clean two-point calibration

- **Bistability at mid pressure.** In the model, β is a phase dial: below
  ~0.46 the q21 pool pegs and every downstream stage floors at ~50
  attempts/task (×31+); above ~0.48 the pool drains and thrash collapses
  (×2). The real q21@0.25 sits *between*: the partition stage pegs while
  downstream stays clean on host-resident intermediates. A single global β
  cannot express that per-stage split, which is why calibrating on q9
  transfers badly to q21 (+179%) while the reverse direction lands (−20.4%).
- The calibration set is two points, one query each, one box, one scale
  factor. Treat sub-knee predictions as order-of-magnitude with ±40% bands
  *at best*, expect over-warning at shallow pressure, and do not transfer β
  to non-join-heavy workloads without re-calibrating.
- Replay-mode eviction ignores the host bound (real costs are in the spans);
  a replay under *additional* knobs (e.g. pressured trace + compute knob)
  scales spans but keeps the traced thrash structure — attempt counts do not
  re-emerge.
- `cpu_mem_capacity` now scales the host-pool spill bound (its only effect).

## 7. G5 telemetry that would nail it (concrete asks)

1. **Born-tier + allocation-policy events on new batches** — one field on
   `data_batch.Constructed` (tier it materialized on + whether pressure
   forced it). This *measures* β directly per stage and would replace the
   calibrated global constant with per-pipeline truth — the single
   highest-value item; it dissolves the bistability problem.
2. **Retry linkage + OOM point**: emit `original_task_id`, `retry_count`,
   `failed_operator_index`, `start_operator_index` and allocated-bytes-at-OOM
   on `Finalizing{success:false}` — gives attempts-per-logical-task and
   per-attempt progress (pins `min_progress` per operator type instead of a
   global 0.02, and validates `oom_cycle_ns` composition).
3. **Downgrade-executor FSM** (request → per-candidate convert{batch,
   bytes, tiers, duration} → done{freed}) — pins `downgrade_rate`/base and
   replaces the LRU-candidate guess with the executor's actual choices
   (repos back-to-front, then queued tasks).
4. **HOST pinned-pool occupancy series** — validates the conversion-window
   bound (7.7 GB here) and its recycle rate.
5. **Manager-loop wait state** around `make_reservation` — separates
   waiting-for-reservations from downgrade waits inside Reserving spans
   (fixes the block-vs-downgrade boundary, §2 step 2).
6. **Flush static entities eagerly** (query/plan/memory/executor ndjson at
   query start, not process exit) — killed sessions like E2-lo-q9 lose
   exactly the entities a replayer needs; the parser now synthesizes them,
   but capacity context stays lost.

## 8. How to use

```bash
# replay a pressured trace (auto-detects Downgrading -> replay bookkeeping)
python -m hwsim selfcheck <E2-mid-session>

# capacity what-if with the calibrated model (auto -> model mode)
python -m hwsim simulate <B1-session> --query-label tpch_q21_iter3 \
    --knob gpu_mem_capacity=0.25

# override mechanism parameters / disable
python -m hwsim simulate ... --spill-param spin_output_host_fraction=0.3 \
    --spill-param oom_cycle_ns=5e7
python -m hwsim simulate ... --spill-mode off     # v0 blocking cliff
```

Report fields: `spill.mode`, `downgrade_events`, `downgraded_gb`,
`reupgraded_gb`, `oom_retries`, `spin_s`, `retry_cap_forced`; sweep CSVs
carry the same columns.
