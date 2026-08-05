# WS5: GPU compute throttling — design, calibration, findings

Goal: reduce effective GPU compute capacity by a controlled factor so we can
physically emulate a smaller/slower GPU and calibrate the simulator's
`gpu_compute` knob (see `tools/hwsim/README.md`).

Hardware/software context for every number in this doc:

- NVIDIA GB300 (Grace + Blackwell, aarch64), **152 SMs**, cc 10.3,
  2048 threads/SM, max SM clock 2070 MHz, single memory P-state (3996 MHz).
- Driver 595.58.03, CUDA toolkit 13.2 (`/usr/local/cuda/bin/nvcc`), MIG disabled,
  compute mode Default, no system MPS daemon. Shared box — all runs were
  seconds-long and gated on `nvidia-smi` showing no other compute apps.

Tools: `tools/hwsim/throttle/compute/` (`throttle_compute`, `victim_bench`,
`probe_concurrency`, `run_calibration.sh`, `mps_probe.sh`).

## Victim workloads

Calibration measures the *victim's* throughput while a throttler runs:

- **Compute-bound victim** (`victim_bench --victim fma`): register-only FMA
  chains, 608 blocks x 256 threads, ~12.5 ms/launch, no DRAM traffic.
  Baseline: **52.2 TFLOP/s** FP32.
- **Memory-bound victim** (`victim_bench --victim saxpy`): grid-stride
  `y = a*x + y` over 2 x 1 GiB arrays, 8 passes/launch (~5.4 ms), pure HBM
  streaming, 1216 blocks by default. Baseline: **4.80 TB/s**.

Throughput is derived from the median per-launch wall time over a ~1.2 s window.
Baselines were reproducible to <0.5% across the session (52,135–52,162 GFLOP/s;
4,574–4,806 GB/s, one saxpy baseline dipped 4.8% right after the box freed up).

## Approaches

### 1. SM-stealing occupancy kernel (`throttle_compute --mode smsteal`) — implemented

A persistent spin kernel is launched as **exactly one full wave**
(152 SMs x 2 blocks/SM x 1024 threads, sized via
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`). Every block reads `%smid`
(inline PTX); blocks on SMs `>= N` exit immediately, freeing those SMs, while
blocks on SMs `< N` spin on register-only FMAs until a host-mapped stop flag is
set (SIGINT/SIGTERM stop cleanly). `N = round(fraction * 152)`.

Placement is verified at startup from a device-side `%smid` histogram; on this
box the wave lands exactly 2 blocks on each of 152 distinct SMs (`min=max=2`),
so a stolen SM is fully occupied (2048/2048 thread slots) and a free SM is
fully free. `%smid` values are dense (0..151).

### 2. Duty-cycle throttler (`throttle_compute --mode duty`) — implemented

A full-GPU burst kernel (one full wave) spins for `fraction * period` of wall
time (measured on-device with `%globaltimer`), then the host sleeps the rest of
the period; default period 10 ms. Throttling serializes with the victim at
kernel granularity.

### Bring-up trap worth knowing: lazy module loading

With the CUDA 12+ default `CUDA_MODULE_LOADING=LAZY`, the **first-ever launch
of a kernel blocks while a persistent kernel is resident in the same context**
(module upload appears to need the context quiescent). This looked exactly like
"SM stealing blocks the victim completely" — even a 1-block/32-thread spinner
"blocked" a 4-block victim — until `probe_concurrency` bisected it: warmed-up
kernels co-run fine, and `CUDA_MODULE_LOADING=EAGER` makes the probe pass.
Both tools now set `EAGER` at startup and `victim_bench` warms victims up
before starting a co-throttler. **Any real victim (Sirius/cudf load lazily by
default) that loads kernels mid-query while a spin throttler is resident in the
same MPS domain can hit long load stalls — run victims with
`CUDA_MODULE_LOADING=EAGER` during throttled experiments.**

### Cross-process semantics: no MPS means no SM sharing

Within one context (two streams), kernels space-share SMs, so
`victim_bench --co smsteal:F` gives true "smaller GPU" semantics. **Across
processes without MPS the driver time-slices the two contexts instead** — the
measured victim slowdown is flat and fraction-insensitive (table below). To
throttle a separate process (Sirius) with SM-share semantics, both processes
must be MPS clients — measured under MPS, cross-process smsteal exactly matches
the in-process numbers.

## Calibration tables

Reproduce with `run_calibration.sh` (plus `mps_probe.sh` for MPS rows and
`--saxpy-blocks` runs for the grid-matched cross-talk rows). "Retained" =
victim throughput / baseline; "ideal" for a spatial throttle is `1 - f`.

### Compute-bound victim (FMA), in-process SM-steal — the primary calibration

| requested f | SMs stolen/free | victim GFLOP/s | retained | ideal 1-f | slowdown x |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 0 / 152 | 52,162 | 1.000 | 1.000 | 1.00 |
| 0.125 | 19 / 133 | 42,481 | 0.814 | 0.875 | 1.23 |
| 0.250 | 38 / 114 | 35,937 | 0.689 | 0.750 | 1.45 |
| 0.375 | 57 / 95  | 31,160 | 0.597 | 0.625 | 1.67 |
| 0.500 | 76 / 76  | 27,486 | 0.527 | 0.500 | 1.90 |
| 0.625 | 95 / 57  | 20,349 | 0.390 | 0.375 | 2.56 |
| 0.750 | 114 / 38 | 14,159 | 0.271 | 0.250 | 3.68 |
| 0.875 | 133 / 19 | 7,188  | 0.138 | 0.125 | 7.26 |

Monotone and near-linear in `1-f`; deviations (-6 to +4 pp) come from the
victim's fixed 608-block grid quantizing into waves over the free SMs, i.e.
they are a property of the victim's launch shape, not of the throttler. Use
this table (not the ideal line) as the `gpu_compute` calibration map.

### Duty cycle (10 ms period), in-process

| duty f | FMA retained (ideal 1-f) | SAXPY retained |
|---:|---:|---:|
| 0.25 | 0.985 (0.75) | 0.680 |
| 0.50 | 0.822 (0.50) | 0.520 |
| 0.75 | 0.706 (0.25) | 0.428 |

Fidelity is poor and **victim-kernel-length dependent**: the 12.5 ms FMA victim
stays resident across bursts and mostly co-runs with them (only -1.5% at 25%
duty!), while the 5.4 ms saxpy kernels serialize into the idle gaps and land
closer to nominal. A duty throttle therefore imposes a slowdown that depends on
each kernel's duration relative to the period — exactly what a hardware
simulator does *not* want. Verdict: keep as comparison point only; SM-stealing
is strictly better on linearity and predictability.

### Cross-process, NO MPS (time-slicing)

| throttler | f | FMA retained |
|---|---:|---:|
| smsteal | 0.25 | 0.431 |
| smsteal | 0.50 | 0.431 |
| smsteal | 0.75 | 0.430 |
| duty    | 0.50 | 0.612 |

Confirms context time-slicing: the SM fraction has **zero effect** across
processes without MPS (~2.3x slowdown at any fraction). Never deploy the
throttler this way.

### Green contexts (victim restricted to N SMs; no throttler kernel at all)

| requested SMs | granted SMs | FMA retained | matching smsteal cell |
|---:|---:|---:|---:|
| 114 | 120 | 0.690 | 0.689 (f=0.25) |
| 76  | 80  | 0.527 | 0.527 (f=0.50) |
| 38  | 40  | 0.272 | 0.271 (f=0.75) |

Grants quantize to multiples of 8 SMs on this GPU. Green-context restriction
and SM-stealing agree to 3 decimal places — strong evidence both implement the
same physical semantics ("fewer SMs"), so the cheap external throttler is a
faithful stand-in for the clean partitioning mechanism.

### MPS `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` (applied to the victim)

| pct | FMA GFLOP/s | retained | ideal |
|---:|---:|---:|---:|
| 100 | 52,151 | 1.000 | 1.000 |
| 75  | 39,112 | 0.750 | 0.750 |
| 50  | 26,074 | 0.500 | 0.500 |
| 25  | 13,035 | 0.250 | 0.250 |

**Exactly linear.** Two reasons: MPS partitions SM occupancy, and it also
reduces the client-visible `cudaDevAttrMultiProcessorCount`, so a victim that
sizes its grids from the SM count (as `victim_bench` and most real engines do)
avoids wave-quantization tails entirely. Additionally, with the MPS daemon
running, **cross-process smsteal at f=0.50 gave 27,494 GFLOP/s (0.527
retained) — identical to the in-process 27,486**, i.e. MPS restores spatial
semantics for the standalone throttler.

## Cross-talk: compute throttle vs the memory-bound victim

The spin kernel itself consumes no DRAM bandwidth — proven by green contexts:
restricting saxpy to ~40 SMs with *no throttler kernel running* costs the same
as stealing 114 SMs with the spinner running (1,775 vs 1,762 GB/s). But taking
SMs away from a memory-bound kernel still costs bandwidth, because streaming
needs SMs to issue/latency-hide:

| mechanism | free SMs | SAXPY GB/s | retained |
|---|---:|---:|---:|
| baseline            | 152 | 4,804 | 1.000 |
| smsteal f=0.25 (fixed 1216-block grid)   | 114 | 3,169 | 0.660 |
| smsteal f=0.25 (grid matched, 912 blocks) | 114 | 4,161 | **0.866** |
| smsteal f=0.50 (fixed grid)   | 76 | 3,004 | 0.625 |
| smsteal f=0.50 (grid matched, 608 blocks) | 76 | 3,060 | **0.637** |
| smsteal f=0.75 (fixed grid)   | 38 | 1,762 | 0.367 |
| greenctx 40 SMs               | 40 | 1,775 | 0.369 |
| MPS pct=25 (grid auto-adapts) | 38 | 1,651 | 0.344 |

Reading: at f=0.25 most of the apparent cross-talk is the victim's grid shape
(recovers to -13% when the grid matches the free SMs); from f=0.50 up it is
intrinsic — a free SM sustains only ~40 GB/s of streaming issue, so achievable
bandwidth is capped at roughly `free_SMs x ~40 GB/s` (76 x 40 ~= 3.0 TB/s,
38 x 43 ~= 1.65-1.78 TB/s).

**Simulator implication:** `gpu_compute` and `gpu_mem_bandwidth` are separable
knobs only while `gpu_compute_fraction x 152 x ~40 GB/s` exceeds the bandwidth
a kernel needs. The replay model should treat a memory-bound task's achievable
bandwidth as `min(gpu_mem_bandwidth, k_sm x free_SM_share)` with
`k_sm ~= 6.1 TB/s` at full SM count (152 x 40 GB/s), rather than assuming
compute throttling never touches bandwidth.

## Availability verdicts

| mechanism | verdict on this box | notes |
|---|---|---|
| MPS + `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` | **Available, no root; recommended** | `/usr/bin/nvidia-cuda-mps-control` exists, no system daemon running. A private daemon (own `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY`, see `mps_probe.sh`) starts as a normal user in Default compute mode and only affects clients pointed at that pipe dir. Exactly linear (table above). |
| `nvidia-smi -lgc` | **Present but not exercised** | Supported graphics clocks 120–2070 MHz in 261 x ~7.5 MHz steps; memory clock has a single P-state (3996 MHz) so `-lmc` offers no range here. Clock changes need admin rights (user has sudo but this session's policy denied the write, correctly: it mutates *global* GPU state on a shared box). Caveat if ever used: SM clock scaling also shifts latencies relative to the fixed HBM clock (L2/interconnect domains), so it is not a pure "fewer SMs" knob, and it throttles every user on the machine. |
| CUDA green contexts (`cuGreenCtxCreate`, 12.4+) | **Available and prototyped** | Driver 595.58.03/CUDA 13.2. `victim_bench --greenctx N` works; grants quantize to multiples of 8 SMs. Cleanest semantics, but requires driver-API calls inside the victim process — for Sirius that means a code hook or an LD_PRELOAD shim, unlike MPS which is env-var only. |

## How to throttle Sirius (deployment recipe)

1. **Primary: MPS active thread percentage.** Start a private MPS daemon, run
   the Sirius process with `CUDA_MPS_PIPE_DIRECTORY=... 
   CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=<pct> CUDA_MODULE_LOADING=EAGER`. Linear,
   root-free, no code changes, and Sirius/cudf see a reduced SM count so their
   own launch heuristics adapt the way they would on a genuinely smaller GPU.
   Semantic note: this limits Sirius directly ("smaller GPU") rather than
   simulating a competing tenant — that is the cleaner match for the
   simulator's `gpu_compute` knob.
2. **Competing-workload semantics: `throttle_compute --mode smsteal` under the
   same MPS daemon.** Measured identical to in-process/green-context behavior.
   Use when the experiment wants "someone else is using N SMs" (throughput
   steal with intact per-SM latency).
3. **Never cross-process without MPS** (flat, fraction-insensitive ~2.3x).
4. **Duty-cycle: comparison only** — non-linear and victim-kernel-length
   dependent.

## CPU compute throttling (design only)

Not implemented in WS5; design options for the `cpu_compute` knob, in order of
preference:

1. **cgroup v2 `cpu.max` (recommended)** — `echo "50000 100000" >
   /sys/fs/cgroup/<grp>/cpu.max` caps the group at 50% of one CPU-equivalent
   (scale quota for more CPUs, e.g. `3600000 100000` = 36 cores' worth on the
   72-core Grace). Continuous fraction, applies to all threads of the process,
   kernel-enforced with 100 µs..100 ms periods.
   Pros: precise, continuous, matches the simulator's "slower CPU" semantics
   (threads run, just get less time); works per-process; `systemd-run --scope
   -p CPUQuota=3600%` gives it without root on delegated systemd user slices.
   Cons: needs a writable cgroup (user delegation or sudo); quota throttling
   arrives in bursts at period granularity (use 10 ms period for smoothness —
   same fidelity consideration as GPU duty-cycling).
2. **cpuset restriction (`taskset -c 0-35` or cgroup `cpuset.cpus`)** — halve
   the visible cores.
   Pros: no root for `taskset` on own processes; models "fewer cores" exactly,
   which is the right analog of SM-stealing/green contexts; interacts
   realistically with DuckDB/Sirius thread pools **only if** the thread count
   is also lowered (DuckDB sizes its pool from affinity/`threads` setting —
   set both).
   Cons: quantized to whole cores; changes NUMA/SMT topology seen by the app;
   doesn't model "same cores, lower clocks".
3. **Background spinner processes at competing priority** — N spinners pinned
   to the victim's cores.
   Pros: no privileges at all.
   Cons: CFS gives each runnable thread a fair share, so the achieved fraction
   depends on the victim's own thread count (the same non-linearity we measured
   for GPU duty-cycling); noisy; pollutes the box for other users. Last resort.
4. **`cpufreq` scaling / locking CPU clocks** — root-only, affects all users,
   limited OPP range on Grace. Same "changes everything at once" caveat as GPU
   `-lgc`. Not recommended on the shared box.

Recommendation: `taskset` (core-count scaling, matching the GPU "fewer
processors" semantic) as the primary mechanism, with cgroup `cpu.max` where a
continuous factor is needed and delegation is available.
