# hwsim — Hardware What-If Simulation for Sirius Queries

## Goal

Predict how a Sirius query's wall-clock time and execution behavior would change on
different hardware, using traces we already collect (Quent telemetry + nsys profiles)
from a real run on the current machine. The simulator answers questions like:

- What if C2C (Grace↔Blackwell NVLink-C2C) bandwidth halves or doubles?
- What if GPU or CPU memory **capacity** halves or doubles?
- What if I/O bandwidth slows down or speeds up?
- What if GPU (or CPU) memory **bandwidth** changes?
- What if compute capacity (SMs / clocks) doubles or halves?
- (Stretch) What if the plan changes — different operator order/placement?

Scaling factors are continuous multipliers, not just 2× / 0.5×.

## Core modeling requirement: cross-dependencies

This is **not** per-phase time scaling. The simulator must be a resource-constrained
replay (discrete-event simulation) of the query's task graph, so second-order effects
emerge naturally. Canonical example: if I/O gets faster, scan output accumulates sooner,
GPU memory fills earlier, and the scanner **blocks on memory back-pressure** — the query
becomes processing-bound and the extra I/O bandwidth buys nothing. The simulation must
capture how resource changes alter *scheduling* (queue waits, admission blocking,
spill/downgrade decisions), not just how they scale individual event durations.

## Simulated resources (knobs)

| Knob | What it scales |
|------|----------------|
| `io_bandwidth` | disk/S3 scan read rate |
| `cpu_mem_bandwidth` | host DRAM bandwidth |
| `gpu_mem_bandwidth` | HBM bandwidth |
| `c2c_bandwidth` | host↔device transfer rate (NVLink-C2C here; PCIe elsewhere) |
| `gpu_mem_capacity` | device memory pool size (drives back-pressure, spill, downgrade) |
| `cpu_mem_capacity` | host memory / pinned pool size |
| `gpu_compute` | SM throughput (kernel busy time) |
| `cpu_compute` | host thread throughput |

## Validation strategy

We validate the simulator on the machine we have, by *physically* degrading each
resource and checking that the simulator (fed the baseline trace + the measured
degradation factor) predicts the degraded run's behavior:

- **I/O**: competing reader that discards bytes at a closed-loop controlled rate
  (and/or cgroup v2 `io.max` for direct limiting).
- **Memory bandwidth**: background copy engines (GPU D2D, host memcpy, C2C H2D/D2H
  streams) rate-controlled to consume a fixed share (and/or `nvidia-smi -lmc`).
- **Compute**: occupancy/spin kernels at a duty cycle or SM share (and/or MPS
  `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`, `nvidia-smi -lgc`).
- **Memory capacity**: shrink Sirius's own memory-pool config (already a YAML knob).
- **Plan changes**: run the altered plan for real and compare.

## Workstreams & layout

| WS | Deliverable | Where |
|----|-------------|-------|
| 1 | Quent telemetry: inventory, mapping to simulator inputs, gap list | `docs/quent-extraction.md` |
| 2 | nsys: what to extract, correlation with quent, overhead budget | `docs/nsys-extraction.md` |
| 3 | I/O throttler design + prototype | `docs/io-throttle.md`, `throttle/io/` |
| 4 | Memory-bandwidth throttler (CPU/GPU/C2C) design + prototype | `docs/membw-throttle.md`, `throttle/membw/` |
| 5 | GPU compute throttler design + prototype | `docs/compute-throttle.md`, `throttle/compute/` |
| 6 | Discrete-event simulator over the task graph | `docs/simulator-design.md`, `sim/` |
| 7 | Validation experiment matrix | `docs/validation-plan.md` |
| 8 | Run experiments, quantify simulator accuracy | `docs/validation-results.md` |
| 19 | Spec-sheet target mode: `--target <descriptor.yaml>` derives the full knob vector for hardware we don't have, from advertised specs + a derating table | `docs/spec-sheet-mode.md`, `hw-descriptors/`, `docs/cross-machine-experiment.md` |

## Ground rules

- This box (GB300) is shared: check `nvidia-smi` for other users' runs before any GPU
  experiment; keep validation microbenchmarks short; never kill others' processes.
- Throttler prototypes are standalone tools (own small build), not part of the Sirius
  extension build.
- Do not modify `src/legacy/`.
