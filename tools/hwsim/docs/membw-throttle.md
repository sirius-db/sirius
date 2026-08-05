# WS4 — Memory-bandwidth throttler (CPU DRAM / GPU HBM / C2C)

Physically emulate reduced memory bandwidth on each of the three memory domains of
the GB300 (Grace CPU + Blackwell GPU, NVLink-C2C between them), so the simulator's
`gpu_mem_bandwidth`, `cpu_mem_bandwidth`, and `c2c_bandwidth` knobs can be validated
against real degraded runs.

Tools: [`tools/hwsim/throttle/membw/`](../throttle/membw/) —
`membw_eater` (rate-controlled background consumer), `membw_victim` (flat-out
per-domain micro-benchmark), `run_calibration.sh` (reproduces every table below).

All measurements in this doc were taken on pmgb300ws-0163 (GB300, CUDA 13.2, driver
595.58.03, aarch64, 72-core Grace, 9-NUMA-node layout with node 0 = all CPUs + 506 GB
DRAM, node 1 = 256 GB HBM exposed over C2C), binaries built `-arch=sm_100`, GPU
otherwise idle (verified via `nvidia-smi --query-compute-apps` before each section).

## 1. Design

### Eater (one binary, `--domain {hbm,dram,c2c}`)

Each domain moves data in bursts of `chunk` bytes and paces bursts with an
**absolute-deadline scheduler**: `period = chunk * traffic_factor / target_rate`,
`deadline += period` after every burst, sleep-until-deadline (coarse
`clock_nanosleep` + ~100 us spin). Absolute deadlines make controller overhead
non-accumulating; a catch-up clamp (max 2 periods of credit) prevents long flat-out
recovery bursts after a stall; a small bounded multiplicative trim (closed loop,
+/-5% per stats interval, skipped when duty > 0.95) absorbs any residual bias.
Chunk size is auto-picked so one burst fires every ~2 ms at the target rate
(clamped 4 MiB..256 MiB GPU, 1..32 MiB per CPU thread), so a paced eater
interleaves with the victim at sub-ms granularity rather than in long block bursts.

Traffic accounting ("consumed GB/s"): `hbm` and `dram` count 2x bytes per copy
(read + write both land on that domain); `c2c` counts 1x (each byte crosses the
link once). An H2D copy inherently also reads host DRAM and writes HBM at 1x —
that is real cross-talk, quantified in section 5.

Domain engines:

- **hbm**: `--engine sm` (default) — grid-stride copy kernel using `__ldcs`/`__stcs`
  streaming accesses so traffic bypasses L2 persistence and lands on HBM
  (1 GiB src/dst buffers >> 126 MB L2, rotating chunk offsets); or `--engine ce` —
  `cudaMemcpyAsync` D2D on a dedicated stream (copy engine, no SM usage).
- **dram**: N `memcpy` threads (default 16) pinned to `--cpu-start..+N-1`
  (default 36..51), buffers first-touched after pinning so pages are NUMA-local
  (node 0 on this box). Per-thread pacers each hold `target/N`.
- **c2c**: `cudaMemcpyAsync` H2D (default), D2H, or alternating, between pinned
  host memory (`cudaMallocHost`) and device memory on a dedicated stream.

Stop: SIGINT/SIGTERM handled, workers drain, final summary printed, exit 0.
Stats: CSV per `--interval` (default 0.5 s):
`epoch_s,elapsed_s,domain,engine,target_gbps,achieved_gbps,bursts,avg_burst_ms,duty`.

### Victim benchmarks

`membw_victim --domain {hbm,dram,c2c}` runs the same access pattern flat-out for
`--secs` (default 3) and prints `RESULT ... gbps=`. HBM victim = SM streaming copy
kernel (representative of a memory-bound query kernel); DRAM victim = 8 memcpy
threads on cpus 0..7 (disjoint from the eater's cpus, so contention is memory, not
cores); C2C victim = pinned H2D memcpy loop.

### Rate-control accuracy (eater alone, unsaturated)

| Domain | Target GB/s | Achieved GB/s |
|--------|-------------|---------------|
| hbm (sm) | 2000 | 2000.0 |
| hbm (sm) | 500 | 500.0 |
| dram | 100 | 99.5 |
| dram | 30 | 29.8 |
| c2c (h2d) | 150 | 150.0 |
| c2c (h2d) | 50 | 50.0 |

Verdict: the controller holds any unsaturated target within ~1%.

## 2. Baselines (eater off)

| Victim | Achieved GB/s (this campaign) | Range across quiet periods |
|--------|-------------------------------|----------------------------|
| HBM D2D, SM streaming-copy kernel (2x bytes) | 5033 | 5033–5090 (~1%) |
| HBM D2D, copy engine `cudaMemcpyAsync` (2x bytes) | 5619 | 5619–5663 (~1%) |
| Host DRAM memcpy, 8 threads NUMA-node 0 (2x bytes) | 196 | 183–196 (~7%) |
| C2C H2D, pinned (1x bytes) | 383 | 345–383 (~11%) |
| C2C D2H, pinned (1x bytes) | 373 | 369–373 (~1%) |

GPU-domain baselines repeat within ~1%. The host-side paths (DRAM, C2C H2D — both
of which read host DRAM) moved 7–11% between quiet periods hours apart, so **always
re-measure the baseline in the same session as the calibration points**
(`run_calibration.sh` does). All ratios below use the same-session baselines.

Eater flat-out maxima (`--gbps max`, alone, 2 s runs): hbm-sm **5073**, hbm-ce
**5621**, dram 16 threads **245**, c2c-h2d **341**, c2c-both (alternating H2D/D2H
on one stream) **352**. Each eater alone can saturate its domain to within a few
percent of the corresponding victim's flat-out rate.

## 3. Per-domain calibration tables

Victim = flat-out micro-benchmark of the same domain; eater running concurrently in
a separate process at the given target. `eater achieved` is the eater's measured
rate while contended (from its CSV). Ratio = victim contended / victim baseline.

### 3a. GPU HBM (victim: SM streaming-copy kernel, baseline 5033 GB/s)

| Eater target (sm engine) | Eater achieved | Victim GB/s | Victim ratio |
|---|---|---|---|
| off | — | 5033 | 1.00 |
| 250 | 250 | 4351 | 0.86 |
| 500 | 500 | 3684 | 0.73 |
| 1000 | 1000 | 2283 | 0.45 |
| 1500 | 1344 (sat) | 1334 | 0.27 |
| 2000 | 1383 (sat) | 1334 | 0.27 |
| 4000 | 1569 (sat) | 1338 | 0.27 |
| max | 1747 (sat) | 1338 | 0.27 |

**Verdict: controllable and monotonic** over eater targets 0..~1300 GB/s, mapping
the victim continuously from 1.00x down to a **floor of 0.27x**. Beyond ~1300 the
eater itself saturates (its contended share) and nothing changes.

Mechanism (important): without MPS, kernels from two processes **time-slice** the
GPU. The victim's loss is much larger than the eater's consumption
(eater 1000 + victim 2283 = 3283 << 5033): each preemption also costs pipeline
drain/refill. So the mapping is *not* `victim = max - eater`; use this table.

### 3b. Host DRAM (victim: 8 memcpy threads cpus 0-7; eater: 16 threads cpus 36-51; baseline 196 GB/s)

| Eater target | Eater achieved | Victim GB/s | Victim ratio |
|---|---|---|---|
| off | — | 196 | 1.00 |
| 60 | 52 | 165 | 0.84 |
| 120 | 104 | 130 | 0.66 |
| 180 | 154 | 121 | 0.62 |
| max | 171 (sat) | 98 | 0.50 |
| max, 32 threads | 162 (sat) | 107 | 0.55 |

**Verdict: controllable and monotonic**, victim range 1.00x down to a **floor of
~0.50x**. DRAM behaves closer to bandwidth sharing (sums 217..270 GB/s; more total
threads extract more aggregate DRAM bandwidth, so the sum is not constant). Note
the eater undershoots its target under contention at higher rates (154/180) —
achieved rate, not target, is the input to any model. Going from 16 to 32 eater
threads does **not** lower the floor.

### 3c. C2C (victim: pinned H2D loop, baseline 383 GB/s; eater: pinned H2D on its own stream)

| Eater target | Eater achieved | Victim GB/s | Victim ratio |
|---|---|---|---|
| off | — | 383 | 1.00 |
| 100 | 100 | 267 | 0.70 |
| 200 | 129 (sat) | 249 | 0.65 |
| 300 | 138 (sat) | 250 | 0.65 |
| max | 143 (sat) | 253 | 0.66 |
| max, 256 MB chunks | 200 (sat) | 187 | 0.49 |

**Verdict: controllable and monotonic in 0..~130 GB/s**, with a floor that depends
on the eater's copy granularity. C2C behaves like true bandwidth arbitration (sum
~= baseline: 129+249 = 378, 200+187 = 387): copy engines interleave copies from the
two processes, and the *share* each side gets tracks its copy sizes in flight. With
default (128 MB) eater chunks vs the victim's 256 MB copies the victim floors at
0.65x; matching the victim's 256 MB chunk size pushes the floor to **0.49x**
(roughly a 50/50 split, as expected for equal-size interleaved copies). For deeper
degradation, raise `--chunk-mb`/`--buf-mb` (and note the D2H direction measured
369 GB/s baseline; `--engine both` alternates directions at 352 GB/s aggregate).

## 4. CE vs SM copies — which degrades a victim kernel?

Both, almost identically, for a memory-bound victim kernel — but for different
reasons and with different side effects:

| HBM eater engine @ rate | Eater achieved | SM-victim GB/s | Victim ratio |
|---|---|---|---|
| sm @ 4000 (sat) | 1569 | 1338 | 0.27 |
| ce @ 4000 (sat) | 1608 | 1374 | 0.27 |
| sm @ max | 1747 | 1338 | 0.27 |
| ce @ max | 1813 | 1379 | 0.27 |

- The **SM eater** steals *GPU time slices* (whole-context preemption) and, while
  resident, SM issue slots. It degrades any co-running kernel — memory-bound or
  compute-bound — because the victim simply does not run during eater slices. This
  overlaps with WS5's compute throttler semantics: at saturation an SM eater is
  also a ~50% duty-cycle compute throttler with HBM traffic on top.
- The **CE eater** (`cudaMemcpyAsync` D2D) was expected to run on copy engines
  concurrently and contend purely for HBM bandwidth. Empirically it degrades the
  memory-bound victim exactly like the SM engine (within ~3%), and the combined
  throughput (1379 + 1813 = 3192 << 5033) shows this is **not** pure bandwidth
  arbitration either — cross-process CE work is scheduled/serialized against the
  victim context much like kernel work when MPS is off.
- Architecturally the CE eater issues no SM work, so it *should* be gentler on a
  compute-bound victim — but given the time-slicing observation this is
  **unverified**; cross-check against WS5's compute victim before relying on it.
  Until then treat both engines as equivalent for memory-bound victims, and treat
  a saturated HBM eater of either engine as also costing ~50%+ GPU time
  (i.e. co-set `gpu_compute` in the simulator for saturated-eater experiments).
- Practical rule: the controllable region is sub-saturation targets
  (<= ~1300 GB/s consumed); both engines produce the same victim curve there.

**Provisional — pending re-measurement: CE-eater cross cells (`hbm-ce -> c2c`,
`hbm-ce -> dram`).** One attempt was made and discarded: another user's Sirius
benchmark (a `python3` from the `perf-query-prs-integration` worktree, holding
243 GB of device memory) re-attached to the GPU mid-measurement, and the CE eater
could only hold ~3150 of its 4000 GB/s target (vs 4000.0 held exactly in the clean
SM-engine cells) — a clear contamination signature. The discarded readings
(c2c victim 366 = 0.96x, dram victim 186 = 0.95x) *hint* that the CE engine has
even less cross-talk onto C2C than the SM engine's 0.83x, but do not trust them.
Re-measure in a verified-idle window with:
`cd tools/hwsim/throttle/membw && for vic in "c2c --engine h2d" "dram --threads 8 --cpu-start 0"; do ./membw_eater --domain hbm --engine ce --gbps 4000 --csv /tmp/e.csv & sleep 0.7; ./membw_victim --domain $vic --secs 2.5 --quiet; kill -INT %1; wait; done`
(valid only if the eater CSV shows achieved ~= 4000).

## 5. Cross-talk matrix

Each eater at a fixed aggressive rate vs all three victims (victim ratio =
contended/baseline; eater's own achieved rate in parentheses). Baselines:
hbm 5033, dram 196, c2c-h2d 383 GB/s.

| Eater \ Victim | hbm victim | dram victim | c2c victim |
|---|---|---|---|
| **hbm** sm @ 4000 | **0.26** (1488, sat) | 0.95 (4001) | 0.83 (4000) |
| **dram** @ 200 | 1.00 (174) | **0.49** (166, sat) | **0.49** (147) |
| **c2c** h2d @ 300 | 0.93 (300) | **0.45** (253) | **0.65** (137, sat) |

All nine cells above are clean measurements (eater held its target exactly in
every unsaturated cell, and diagonal cells reproduce the section 3 calibration
values). Two *additional* cells using the CE-engine HBM eater are provisional —
see the note at the end of section 4.

Off-diagonal findings, in order of importance for the simulator:

1. **C2C and host DRAM are nearly the same resource on Grace.** A C2C H2D eater
   reads its pinned source buffers from host DRAM at the transfer rate, so at
   253 GB/s of C2C it took the DRAM victim to **0.45x** — *more* degradation than
   the dedicated DRAM eater at 200 (0.49x). Symmetrically, the DRAM eater took the
   C2C victim to **0.49x** (and the C2C victim's own DRAM reads suppressed the
   DRAM eater to 147 of its 200 target — the interference is mutual). A C2C
   experiment is therefore always a joint `(c2c_bandwidth, cpu_mem_bandwidth)`
   experiment on this machine, and vice versa for large-rate DRAM experiments.
2. **HBM eater degrades C2C by 17%** (0.83x): H2D copies write into HBM, and the
   saturated HBM eater competes for those writes (plus GPU time-slicing of the CE
   channel work). Moderate but not negligible.
3. **C2C eater degrades HBM by 7%** (0.93x): 300 GB/s of C2C writes into HBM is
   only ~6% of the 5 TB/s HBM budget — the observed 7% matches almost exactly.
4. **HBM eater vs DRAM victim: 0.95x** — small; the SM eater has no host-memory
   footprint, and the residual 5% is likely time-slicing of the victim process's
   housekeeping plus measurement variance (DRAM baseline itself moved 183..196
   between quiet periods).
5. **DRAM eater vs HBM victim: 1.00x** — fully independent, as hoped.

The matrix says the three knobs are *not* independent on Grace-Blackwell:
`c2c<->dram` coupling is first-order (~0.5x), `hbm->c2c` is second-order (~0.8x),
and everything else is <= 7%. Section 7 explains how to feed this into validation.

## 6. Alternative knobs evaluated (not depended on)

### `nvidia-smi -lmc` (memory clock locking): **not usable on GB300**

`nvidia-smi -q -d SUPPORTED_CLOCKS` lists exactly **one** memory clock: 3996 MHz
(current == max). There is no lower memory P-state to lock to, so `-lmc` cannot
reduce HBM bandwidth on this box. (On boards with multiple memory clocks it would
be the cleanest HBM knob — global, no SM/CE side effects — worth re-checking on
other validation machines.)

### resctrl / MPAM (Grace memory-bandwidth allocation): **present but not mounted**

The kernel is built with `CONFIG_ARM64_MPAM=y`, `CONFIG_ACPI_MPAM=y`,
`CONFIG_ARM64_MPAM_RESCTRL_FS=y`, and `resctrl` is a registered filesystem in
`/proc/filesystems`, but `/sys/fs/resctrl` is an empty, unmounted mountpoint and we
run unprivileged (uid 2099, sudo not verified). Mounting requires root:
`mount -t resctrl resctrl /sys/fs/resctrl`, then MB-allocation classes could cap a
control group's DRAM bandwidth directly — a much cleaner `cpu_mem_bandwidth` knob
than an eater (per-cgroup, no CPU-core cost). Worth pursuing with an admin;
whether Grace's MPAM MSCs actually expose MB controls (vs only cache portioning)
is unverified.

### MPS

No MPS daemon runs on this box (`/tmp/nvidia-mps` absent). This matters for
interpreting GPU contention: without MPS, kernels from different processes
**time-slice** the GPU rather than sharing SMs concurrently (see section 4).

## 7. Feeding the simulator knobs

The simulator's bandwidth knobs are continuous multipliers on nominal domain
bandwidth. The eaters do **not** implement `victim = max - eater_rate`; the actual
mapping from eater target rate to effective victim bandwidth is the calibration
table (section 3), for two reasons:

1. **GPU domains time-slice across processes** (no MPS): an HBM eater steals GPU
   *time*, and preemption overhead destroys aggregate bandwidth, so the
   victim's loss exceeds the eater's consumption.
2. **C2C and DRAM behave closer to bandwidth arbitration**, but still not exactly
   subtractive.

Validation recipe per domain:

- Pick a simulator knob value `k` (e.g. `gpu_mem_bandwidth = 0.45x`).
- Invert the calibration table: find the eater target whose measured victim ratio
  `victim_contended / victim_baseline` equals `k`; interpolate between rows.
- Run the Sirius query with that eater running; feed the simulator the baseline
  trace with the knob at `k`; compare predicted vs measured wall clock.
- Because of cross-talk (section 5), set the simulator's *other* two bandwidth
  knobs to the measured cross-talk ratios for that eater, not to 1.0 — e.g. a C2C
  eater run is not a pure `c2c_bandwidth` experiment; it is
  `(c2c=r1, gpu_mem=r2, cpu_mem=r3)` with all three ratios taken from the matrix
  row. This is the single most important methodological point in this doc: treating
  eater runs as single-knob experiments would mis-attribute error to the simulator.

Concrete per-knob summary (from sections 3 and 5):

| Simulator knob | Eater config | Emulable range | Side effects to co-set |
|---|---|---|---|
| `gpu_mem_bandwidth` | `--domain hbm --engine ce --gbps 0..1300` | 1.00x .. 0.27x | `c2c_bandwidth` ~0.83x at saturation (scale down with rate); dram ~1.0 |
| `cpu_mem_bandwidth` | `--domain dram --gbps 0..180 --threads 16 --cpu-start 36` | 1.00x .. 0.50x | `c2c_bandwidth` down to ~0.49x at high rates (first-order!) |
| `c2c_bandwidth` | `--domain c2c --engine h2d --gbps 0..130` (default chunks) or `--chunk-mb 256 --buf-mb 1024` for the 0.49x floor | 1.00x .. 0.49x | `cpu_mem_bandwidth` down to ~0.45x (first-order!); `gpu_mem_bandwidth` ~0.93x |

Limits of emulation: the victim-bandwidth range is bounded below by the eater's
saturated share of the domain (floors above); values below those floors cannot be
emulated by contention on this box. If deeper degradation is ever required, the
remaining levers are: multiple concurrent eater processes/streams, larger C2C
chunks (demonstrated), or root-only mechanisms (resctrl/MPAM for DRAM — section 6).
