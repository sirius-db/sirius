# GPU compute throttler (hwsim WS5)

Tools to reduce effective GPU compute capacity by a controlled factor, used to
physically emulate a smaller/slower GPU and calibrate the simulator's `gpu_compute`
knob. Standalone build — not part of the Sirius extension build.

See `tools/hwsim/docs/compute-throttle.md` for the design discussion, approach
comparison, calibration tables, and availability findings (MPS / `-lgc` / green
contexts).

## Build

```bash
make                # nvcc from PATH or /usr/local/cuda/bin; targets sm_100
                    # (+compute_100 PTX — the JIT fallback covers newer parts,
                    # e.g. cc 10.3 GB300 and sm_120 workstation Blackwell)
make ARCH=native    # compile for the GPU actually in this box
make ARCH=sm_103a   # explicit arch — keep ONE arch across every binary whose
                    # numbers you compare (arch changes codegen)
```

## Binaries

### `throttle_compute` — standalone throttler process

```bash
# Steal 50% of SMs with a persistent spin kernel (verifies %smid placement):
./throttle_compute --mode smsteal --fraction 0.5 [--duration 10]

# Duty-cycle the whole GPU: 50% busy at a 10 ms period:
./throttle_compute --mode duty --fraction 0.5 [--period-ms 10] [--duration 10]
```

Runs until SIGINT/SIGTERM (clean stop) or `--duration` seconds. The spin kernels
are register-only FMA loops — they consume no DRAM bandwidth by design.

**Cross-process caveat (important):** without MPS, kernels from a different
process do NOT space-share SMs with this throttler — the driver time-slices the
two contexts, so `smsteal --fraction F` does not produce a `1/(1-F)` slowdown of
the other process. To throttle a separate process (e.g. Sirius) by SM share, run
both the throttler and the victim as MPS clients. See the calibration doc for
measured cross-process behavior.

### `victim_bench` — calibration victims + in-process co-throttler

Victims:
- `--victim fma` — compute-bound: register-only FMA loop (reports GFLOP/s)
- `--victim saxpy` — memory-bound: streaming `y = a*x + y` over 1 GiB arrays
  (reports GB/s of HBM traffic)

```bash
./victim_bench --victim fma --seconds 1.5                 # baseline
./victim_bench --victim fma --co smsteal:0.5              # in-process SM steal
./victim_bench --victim saxpy --co duty:0.5:10            # in-process duty cycle
./victim_bench --victim fma --greenctx 76                 # green ctx, ~76 SMs
```

`--co` runs the throttler on a second stream in the same context, which
guarantees SM-level co-residency (the semantics the simulator's `gpu_compute`
knob models); this is the primary calibration mode. Output is one greppable
`RESULT ...` line with median per-launch time and throughput.

### `probe_concurrency` — diagnostic (kept as a regression probe)

Documents a real trap hit during bring-up: with default lazy module loading
(CUDA 12+), a kernel's **first-ever launch blocks while a persistent kernel is
resident in the same context**. Both tools therefore set
`CUDA_MODULE_LOADING=EAGER` at startup, and `victim_bench` warms victims up
before starting the co-throttler. If you throttle a real application under MPS,
launch the application with `CUDA_MODULE_LOADING=EAGER` too.

## Scripts

- `run_calibration.sh` — the full calibration matrix (baselines, in-process
  smsteal/duty, green contexts, cross-process no-MPS). ~90 s GPU-busy total;
  gates every section on GPU idleness.
- `mps_probe.sh` — starts a private MPS daemon (own pipe/log dirs, no root),
  measures `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` at 100/75/50/25 plus
  cross-process smsteal under MPS, then shuts the daemon down.

## Shared-box etiquette

The GB300 is shared. Before any run:

```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv
```

Wait for idle; keep runs seconds-long; never kill other users' processes.
