# Quent Session Export (WS17) — simulated executions in the real viewer

`hwsim` can export a simulated execution as a **Quent ndjson session
directory** with the same envelope, entity set and file layout as a real
Sirius capture (the inverse of the parser spec in
[quent-extraction.md](quent-extraction.md) §d). The Quent analyzer UI
(`pixi run quent`) then renders the *predicted* timeline exactly like a real
one — open the real baseline trace and the simulated what-if side by side.

## Usage

```bash
cd tools/hwsim/sim

# export one simulated query (works with any knobs / spill flags)
python -m hwsim simulate <trace_dir> --query-label <L> \
    --knob gpu_compute=0.5 --export-quent /path/to/outdir

# sweep: one exported session per knob point
python -m hwsim sweep <trace_dir> --query-label <L> \
    --sweep c2c_bandwidth=0.5,1,2 --export-quent /path/to/outdir
```

Each export creates `<outdir>/<session-uuid>/` (`model.qmi` + one ndjson file
per entity type). The session uuid is a deterministic uuid7-style id derived
from `(source session uuid, exported query label)` — re-exporting the same
point produces the same directory name, and the export refuses to overwrite an
existing one (remove it first). Programmatic API:
`hwsim.export_quent.export_session(model, graph, knobs, result, out_dir,
seed=None)`.

`--export-quent` is wired on the v0 simulate/sweep paths; it is **ignored
when `--physics` is passed** (the physics pipeline reports through its own
command functions — exporting a physics-retimed run is a follow-up).

## Session identity and alignment

- `engine.Init.implementation.name` = **`hwsim-sim`** (a simulated session is
  unmistakable in the UI and in scripts).
- `engine.Init.custom_attributes` (the machine-readable contract):
  - `hwsim.simulated` = 1 (I64)
  - `hwsim.source_session` = source session uuid (String)
  - `hwsim.source_query` = source query label (String)
  - `hwsim.knob.<name>` = value (F64), one per non-default knob
  - plus `hwsim.spill_mode` (String) and `hwsim.seed` (String)
- The exported query label is `<source label>@<knob suffix>`, e.g.
  `tpch_q21_iter1@gpu_compute=0.5` (`@baseline` when all knobs are 1.0).
- Timestamps are unix-epoch ns. Sim `t=0` is anchored at the **source query's
  traced `Executing` timestamp**, so the real and simulated sessions align on
  the UI time axis; query `Exit` lands at `t0 + simulated wall`.
- `model.qmi` copies the source session's quent/model provenance (when
  available) and adds an `hwsim` block (generator, source, knobs, seed,
  `sim_wall_ns`).

## What is emitted

| Entity | Contents (all timings are the SIMULATED ones) |
|---|---|
| `engine`, `worker`, `query_group` | one each; the hwsim identity above |
| `gpu_device`, `thread_group` | per device + shared/executor/manager groups (parent_group_id-driven resource tree, like a real session) |
| `memory`, `memory_tier` | **simulated capacities**: GPU = the engine's knob-scaled pool, HOST scaled by `cpu_mem_capacity`; names carry the scaled limit |
| `channel` | mesh mirrored from the source session (placeholder `u64::MAX` capacity, as in real traces) |
| `task_queue` | scheduler queue (parent=shared) + per-GPU executor queue |
| `executor_thread`, `task_manager_loop_thread` | the engine's thread counts; scheduler + per-GPU manager loops |
| `query` | Init/Planning (synthetic pad before t0) → Executing @ t0 → Exit @ t0+wall |
| `plan`, `operator`, `port` | pipeline-level DAG: one operator per pipeline (chain string + `Pipeline Id N`), one receiver+sender port per pipeline, plan edges from the sim's edge set |
| `task` | full FSM per replayed task: Created @ release, Queued (scheduler) / Routing / Queued (executor) @ enqueue, Reserving @ admit, Preparing @ prep-start (origin/target tier, input bytes, executor-thread + granted-reservation usages), one Computing per physical operator, Finalizing (success flag), Exit @ finish |
| `data_batch` | Constructed @ producer finish (producer pipeline uuid), Stationary (memory-space usage, bytes), Destructed/Exit @ last consumption or query end |
| `batch_placement` | per (batch × consumer): Registered/Queued @ publish, Packaged @ consumer admit, Processing @ consumer prep-end, Consumed(`processed`) @ consumer finish; consumer-less batches get a single placement drained with `query_end` |

Mechanical details:

- Per-entity `seq` is contiguous from 0 with a terminal `Exit`; timestamps
  are clamped monotone per entity (the parser orders by seq).
- All uuids are freshly synthesized, deterministic given the seed (48-bit
  ms timestamp | ver 7 | monotone counter | seeded-hash bits).
- Thread bindings are a greedy interval colouring of each task's
  [admit, finish] window over the device's executor threads — a feasible
  binding consistent with the engine's slot accounting (the engine tracks
  slot *counts*, not identities).
- Per-operator Computing boundaries are laid out proportionally to the
  knob-scaled base durations inside the simulated compute window (exact at
  knobs=1; proportional when the fluid device stretched the phase).
- Producer-less batches (scan-manager staging / orphans) are Constructed at
  `t0 − 10 ms` with a nil producer uuid, so a re-parse classifies them as
  externally available — exactly how the engine treated them (resident from
  t=0).

## What is omitted (and why)

The export never invents measurements: event types the simulator has no data
for are omitted, not fabricated.

- **`io_request`** — the v0 sim has no I/O sub-model (gap G1).
- **`data_batch.InTransit` / tier-change self-transitions** — the sim models
  transfers as task-level `Preparing` spans (which are exported, with bytes
  and origin/target tiers), not per-batch moves.
- **task `Downgrading` states** — spill-model downgrades are engine-level
  bookkeeping without per-task timestamps in `SimResult`; a replayed traced
  downgrade wait is folded into the Reserving→Preparing span.
- **OOM-rescheduled attempt FSMs** (spill model mode) — the sim tracks spin
  counts/time, not per-attempt windows, so no failed-task FSM instances are
  synthesized. Replayed *traced* failures are exported with
  `Finalizing.success=false` like any other replayed task.
- **`Computing.peak_allocated_bytes`** is emitted as 0 (schema-required
  field; the sim does not track per-op allocator peaks).
- **WS9 fields are EMITTED with the schema's unknown markers** (revised
  after WS18's verification, defect 1 in
  [quent-export-verification.md](quent-export-verification.md)): the Rust
  analyzer's serde types have **no field defaults** — a line missing any
  current-model field silently truncates that entity's whole stream. So the
  exporter emits `Computing.input_rows=0`, `Finalizing.output_rows=0` /
  `output_bytes=0`, `data_batch.Constructed.num_rows=0` / `num_columns=0`
  (0 = unknown, the engine's own convention), and
  `producer_task_uuid` on `Constructed` + `BatchRegistered` — the **real
  exported producer-task uuid** where the sim graph knows it, nil otherwise.
  Only `io_request` remains omitted as a whole entity (a missing entity
  subdir is legal; required fields bite per-line).
- `Reserving.input_basis` = the task's input bytes, `peak_estimate` =
  requested bytes, `bytes_to_materialize` = transfer bytes (all sim-held
  quantities; the source estimator internals are not copied).
- `batch_placement.origin` is always `operator_output` (the sim does not
  retain per-placement origin).

## Round-trip guarantees (verified)

The export is a *valid trace of the simulated execution*: the sim's own
parser accepts it, and re-simulating it at knobs=1 reproduces the exported
wall. Verified on B1 q21 (2339 tasks, 6849 batches):

| check | result |
|---|---|
| `python -m hwsim info <export>` / `selfcheck` | parses; 1 query, same task/pipeline/batch counts |
| re-sim @ knobs=1 of the `@baseline` export | +0.06% vs exported wall |
| re-sim @ knobs=1 of the `@gpu_compute=0.5` export | +0.05% vs exported wall |
| exported task spans (prep/compute/tail/pre-queue) vs the sim replay and vs the source traced spans | 0 mismatches > 2 ns over 2339 tasks |
| WS18 rig: `export-verify/validate_quent_session.py --simulated` (no `--allow-legacy`) | 0 errors, 0 warnings on both exports |
| WS18 rig: `export-verify/ingest_check.sh` (Rust analyzer) | PASS — 22,294/22,294 task, 27,396/27,396 data_batch, 41,874/41,874 batch_placement lines ingested; 180-node resource tree |

Caveat: a `gpu_mem_capacity`-knobbed export re-simulated at knobs=1 keeps the
scaled pool (the exported memory spaces ARE the what-if hardware) but cannot
re-derive the spin/banked-progress dynamics of spill *model* mode — the spin
thread-time is not representable as task states.

## Side-by-side viewing sketch

1. Run the real capture as usual (`telemetry_data/<session>/`).
2. Export the what-if: `python -m hwsim simulate telemetry_data/<session>
   --query-label <L> --knob gpu_compute=0.5 --export-quent telemetry_data`.
3. Point the analyzer at the output directory (`pixi run quent`) — both
   sessions appear; the simulated one is named `hwsim-sim` and its query
   label carries the knob suffix. Because the exported t0 matches the source
   query's traced start, the two timelines align on the time axis.

WS18 owns the schema validator, analyzer-ingest test rig and the full
viewing workflow doc; this file records only the format decisions of the
exporter side.
