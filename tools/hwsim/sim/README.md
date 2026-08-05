# hwsim — hardware what-if simulator for Sirius traces (WS6, v0)

Discrete-event simulator that replays the task graph recorded by Quent
telemetry under a resource model (executor threads, task-queue admission,
GPU memory-pool reservations, transfer channels) and re-times / re-flows it
when hardware knobs change. Design and validation results:
[`../docs/simulator-design.md`](../docs/simulator-design.md).

Pure Python 3 stdlib — no dependencies. Runs in the repo pixi env or any
Python ≥ 3.10.

## Quick start (against the real sample trace)

```bash
cd tools/hwsim/sim
TRACE=/localhome/local-faramburu/repos/sirius/telemetry_data/019fbafc-e2cc-7bb3-b1ba-915fd149e7f5

# list queries, wall times, session facts (labels are synthesized
# tpch_qNN_iterK when the run is 22 unnamed queries x 3 iterations)
pixi run python -m hwsim info $TRACE

# replay every query with knobs=1.0 and compare to the traced wall time
pixi run python -m hwsim selfcheck $TRACE --csv selfcheck.csv

# what-if: halve the C2C link on TPC-H q9
pixi run python -m hwsim simulate $TRACE --query-label tpch_q09_iter2 \
    --knob c2c_bandwidth=0.5 --json q9_c2c05.json

# knob sweep (repeat --sweep for a cartesian product)
pixi run python -m hwsim sweep $TRACE --query-label tpch_q21_iter2 \
    --sweep c2c_bandwidth=0.25,0.5,1,2,4 --csv q21_c2c.csv
pixi run python -m hwsim sweep $TRACE --query-label tpch_q09_iter2 \
    --sweep gpu_mem_capacity=0.05,0.1,0.25,0.5,1,2

# unit tests (deterministic toy graphs with analytic answers)
pixi run python -m unittest discover -s tests
```

The first invocation streams and parses the full 654 MB session (~17 s) and
caches the built model under `~/.cache/hwsim/` (override with `--cache-dir`,
disable with `--no-cache`); subsequent runs load in ~1 s. A full 66-query
selfcheck runs in ~3 s warm.

## Knobs

Continuous multipliers, default 1.0 (= traced hardware); pass repeated
`--knob name=value`:

| knob | v0 effect | fidelity |
|---|---|---|
| `c2c_bandwidth` | re-times HOST→GPU/GPU→HOST Preparing materializations from bytes at scaled rate through a shared fluid channel | span includes SM-bound decompression (warned) |
| `gpu_mem_capacity` | scales the GPU pool; reservation admission waits **emerge** | real engine would spill instead of waiting (warned; sample trace has no Downgrading events to calibrate spills) |
| `gpu_compute` | scales operator Computing spans | conflates SM + HBM + launch overhead until the nsys join (gap G4, warned) |
| `gpu_mem_bandwidth` | placeholder: spans scale by `1/min(gpu_compute, gpu_mem_bandwidth)` | pessimistic roofline stand-in (warned) |
| `io_bandwidth` | scales `GPU_SCAN` Computing spans | **loud caveat**: no I/O events in the trace (gap G1); scales decode too |
| `cpu_*` | accepted, **no effect** in v0 (warned) | — |

Every degraded-fidelity knob prints a WARNING when moved off 1.0.

## Output

`simulate` prints a human-readable report (wall time vs traced and vs the
simulated baseline, binding constraint, per-device thread/pool stats, channel
stats, per-pipeline and per-operator breakdowns) and `--json` writes the full
machine-readable report including thread-busy and pool-occupancy timelines.
`selfcheck`/`sweep` print tables and write CSV with `--csv`.

Diagnostic counters to watch in reports:
- `forced_admissions` — the sim force-admitted a memory-blocked task to make
  progress (the real engine would downgrade/spill there; v0 cannot price it),
- `dep_cycle_breaks` — batch→producer attribution produced a cycle (never
  observed on the sample trace),
- `ambiguous_producer_batches` — batches whose producing task was inferred
  among several concurrent candidates (gap G2; ~98% on the sample trace
  because 4 executor threads run pipeline tasks concurrently).

## Layout

```
hwsim/
  trace.py    streaming ndjson parser (per-entity seq ordering, substring
              pre-filters for the 500 MB of batch events), pickle cache
  build.py    task FSM -> TaskSpec spans; batch->producer attribution (G2
              heuristic); dependency + barrier inference; channel capacity
  model.py    dataclasses (SessionModel / QueryGraph / TaskSpec / BatchSpec)
  knobs.py    knob definitions, scaling rules, fidelity warnings
  engine.py   discrete-event core: thread slots, head-of-line memory
              admission, fluid transfer channels, deadlock resolution
  report.py   text/JSON/CSV reports, breakdowns, timelines
  cli.py      info / simulate / selfcheck / sweep
tests/
  test_engine.py  19 analytic toy-graph tests (pool serialization, channel
                  sharing, head-of-line, knob scaling, cross-dependency
                  saturation, queue-order semantics)
```
