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

`--export-quent` works on **both the v0 and the `--physics` paths** (WS20;
the RTX PRO 6000 external validation showed v0-only export makes every
compute-what-if timeline ~2x wrong on host-dominated lanes while the correct
physics number existed but could not be exported). With `--physics`, the
export carries the physics-retimed schedule — the exported wall IS the
physics-predicted wall:

```bash
python -m hwsim simulate <trace_dir> --query-label <L> \
    --physics physics.json --knob gpu_compute=0.5 --export-quent /path/to/outdir
```

A physics export differs from a v0 export in three visible ways: the query
label carries a trailing `,physics` token (`...@gpu_compute=0.5,physics`;
`@baseline,physics` at knobs=1), the engine Init grows `hwsim.physics=1` plus
profile provenance (see below), and per-task Computing spans are the
physics-split re-timed ones (including the synthesized `PHYS::PREP`
pseudo-operator carrying the non-transfer share of split Preparing phases,
exported with the `u32::MAX` no-operator-id placeholder).

## Session identity and alignment

- `engine.Init.implementation.name` = **`hwsim-sim`** (a simulated session is
  unmistakable in the UI and in scripts).
- `engine.Init.custom_attributes` (the machine-readable contract):
  - `hwsim.simulated` = 1 (I64)
  - `hwsim.source_session` = source session uuid (String)
  - `hwsim.source_query` = source query label (String)
  - `hwsim.knob.<name>` = value (F64), one per non-default knob
  - plus `hwsim.spill_mode` (String) and `hwsim.seed` (String)
  - physics-retimed exports additionally carry `hwsim.physics` = 1 (I64) and
    the profile provenance: `hwsim.physics_profile` /
    `hwsim.physics_nsys_sqlite` / `hwsim.physics_profile_created_utc`
    (String) and `hwsim.physics_pct_span_matched` /
    `hwsim.physics_kernel_serial_frac` (F64) and
    `hwsim.physics_device_model` (I64: G4b fluid device engaged?)
- The exported query label is `<source label>@<knob suffix>`, e.g.
  `tpch_q21_iter1@gpu_compute=0.5` (`@baseline` when all knobs are 1.0);
  physics-retimed exports append a `,physics` token
  (`@gpu_compute=0.5,physics`, `@baseline,physics`).
- Timestamps are unix-epoch ns. Sim `t=0` is anchored at the **source query's
  traced `Executing` timestamp**, so the real and simulated sessions align on
  the UI time axis; query `Exit` lands at `t0 + simulated wall`.
- `model.qmi` copies the source session's quent/model provenance (when
  available) and adds an `hwsim` block (generator, source, knobs, seed,
  `sim_wall_ns`; physics exports add an `hwsim.physics` sub-block mirroring
  the provenance attributes).

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
  knobs=1; proportional when the fluid device stretched the phase). On the
  physics path the retimed graph's spans are ALREADY knob-scaled (the engine
  ran with neutralized GPU knobs), so the layout weights are engine-neutral
  (only the G1 `io_bandwidth` divisor applies).
- The `Routing` state's free-form `instance_name` carries a
  **`qprio=<rank>` dispatch-order marker**: the engine dispatches released
  tasks in the source trace's queue-entry order (`queue_order="traced"`),
  which the simulated enqueue timestamps do NOT encode — without the marker,
  re-simulating an export repacks order-sensitive schedules (measured **+67%
  wall drift** on B1 q9's physics export). `build.py` parses it back into
  `TaskSpec.queue_prio`, which the engine prefers over `t_queued`; real
  traces have no marker and keep the old behavior.
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

Physics-retimed exports (WS20), verified on B1 q9 (1642 tasks, 6433 batches)
against `experiments/nsys/R1/physics_R1.json`:

| check | result |
|---|---|
| exported wall vs the physics-predicted wall (`sim_wall_ms`) | **identical to the ns** (13383.207038 ms at `gpu_compute=0.5`; 12612.606657 ms at baseline) |
| `export-verify/validate_quent_session.py --simulated` | 0 errors, 0 warnings on both exports |
| `export-verify/ingest_check.sh` (Rust analyzer) | PASS — 16,976/16,976 task lines, 132-node resource tree |
| re-sim @ knobs=1 of both exports | **0.00%** vs exported wall (requires the `qprio` marker; without it this query repacks +67%) |

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

## Follow-up: pre-WS9 REAL traces vs the current analyzer

WS18's flip-side finding (quent-export-verification.md): every pre-WS9 real
capture (the 654 MB sample, B1, all E1/E2/E3 campaign traces, model commit
`b77aa438`) is silently truncated by the **current** branch's Rust analyzer
for the same no-serde-defaults reason this exporter was fixed for.

The obvious remedy — `#[serde(default)]` on the four WS9-extended states
(`task.rs` Computing/Finalizing, `data_batch.rs` Constructed, `batch.rs`
BatchRegistered in `rust/crates/telemetry/model`) — is currently **blocked
by the pinned quent macro** (rev `2a5ca834`): `state!` expands user
attributes *before* its own `#[derive(Serialize, Deserialize)]`, so a
container-level `#[serde(default)]` trips the deny-by-default
`legacy_derive_helpers` future-incompat lint (rust-lang/rust#79202,
verified: 8 compile errors), and the macro's inline attribute grammar
accepts no per-field attributes. The alternative external-struct
`attributes:` form would mean hand-implementing `EventMetadata` /
`Extract*` for four states with a codegen-drift risk in the cxx bridge.

Proper fix (follow-up, not done here): patch quent's `state!` to emit user
attributes after its derive (or accept per-field attrs), then add
`#[derive(Default)] #[serde(default)]` to those four states and verify the
generated cxx bridge sources are byte-identical (they should be — codegen
reads `StateDef` field metadata, not serde attrs). Until then, view
pre-WS9 traces with an analyzer built at the matching model commit, and
use `export-verify/validate_quent_session.py --allow-legacy` to
distinguish legacy-shape warnings from real corruption.
