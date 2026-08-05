# membw — memory-bandwidth eaters + victim benchmarks (hwsim WS4)

Rate-controlled background bandwidth consumers ("eaters") for physically emulating
reduced memory bandwidth on three separately-controllable domains of the GB300
(Grace + Blackwell), plus tiny flat-out "victim" benchmarks to calibrate how much a
given eater rate degrades each domain. Design, calibration tables, and the 3x3
cross-talk matrix live in [`../../docs/membw-throttle.md`](../../docs/membw-throttle.md).

| Domain | What it stresses | Mechanism | Traffic accounting |
|--------|------------------|-----------|--------------------|
| `hbm`  | GPU HBM bandwidth | SM streaming-copy kernel (`--engine sm`, default) or copy-engine `cudaMemcpyAsync` D2D (`--engine ce`) | 2x bytes per copy (read + write) |
| `dram` | Host (Grace) DRAM bandwidth | pinned CPU `memcpy` threads | 2x bytes per copy |
| `c2c`  | NVLink-C2C host<->device link | `cudaMemcpyAsync` H2D / D2H with pinned host memory on a dedicated stream (`--engine h2d` default, `d2h`, or `both` alternating) | 1x bytes (each byte crosses the link once) |

## Build

```bash
make                 # nvcc (found on PATH or /usr/local/cuda/bin), -arch=sm_100
                     # (sm_100 SASS + compute_100 PTX; the PTX JIT covers newer
                     # parts, e.g. sm_120 workstation Blackwell)
make ARCH=native     # compile for the GPU actually in this box
make ARCH=sm_103a    # override arch — GB300 is CC 10.3; keep one arch for all comparisons
make nocuda          # g++-only fallback: membw_{eater,victim}_nocuda, dram domain only
```

## membw_eater

Holds a target *consumed* GB/s with a closed-loop controller until SIGINT/SIGTERM
(or `--duration`), emitting CSV stats of the achieved rate:

```bash
./membw_eater --domain hbm  --gbps 1000                       # eat 1000 GB/s of HBM
./membw_eater --domain dram --gbps 120 --threads 16 --cpu-start 36
./membw_eater --domain c2c  --gbps 200 --engine h2d --csv c2c.csv
./membw_eater --domain hbm  --gbps max --duration 3           # flat out (find the max)
```

CSV columns: `epoch_s,elapsed_s,domain,engine,target_gbps,achieved_gbps,bursts,avg_burst_ms,duty`.
`duty` is the fraction of wall time spent inside copy bursts — `duty ~ 1.0` with
`achieved < target` means the domain is saturated and the target is not reachable.

Controller: each burst moves a `--chunk-mb` payload (auto-sized so one burst fires
every `--pace` seconds, default 2 ms); bursts are scheduled on absolute deadlines
(`period = chunk * factor / target`), with a catch-up clamp of 2 periods and a small
bounded multiplicative trim updated each `--interval` from the measured rate.
Achieved rate typically holds within <1% of target when unsaturated.

dram notes: `--threads N` workers are pinned to `--cpu-start .. cpu-start+N-1` and
first-touch their buffers after pinning, so pages land on those CPUs' NUMA node
(node 0 holds all 72 cores + all host DRAM on this box; use `numactl -m` for other
layouts). Keep eater and victim/workload CPU ranges disjoint so you measure memory
contention, not core contention.

## membw_victim

Flat-out micro-benchmark of one domain; run it with and without an eater to build
calibration tables. Prints per-interval rates and a final grep-able line:

```bash
./membw_victim --domain hbm --secs 3 --quiet
RESULT domain=hbm engine=sm secs=3.000 gbps=5090.13
```

## run_calibration.sh

Reproduces everything in the doc: baselines, per-domain eater-rate sweeps, the CE-vs-SM
engine comparison, and the 3x3 cross-talk matrix (each eater at a fixed aggressive
rate vs all three victims). Waits for the GPU to be free of other users' processes
before each GPU section (this box is shared — see repo ground rules).

```bash
./run_calibration.sh                       # everything, ~2 min GPU-busy
SECTIONS="baseline xtalk" ./run_calibration.sh
```

## Caveats

- Cross-process GPU contention is **context time-slicing**, not proportional
  bandwidth arbitration (no MPS on this box): an HBM eater degrades a victim by
  stealing GPU time slices, and aggregate throughput under contention is well below
  the solo maximum. The victim-vs-eater-rate calibration table is therefore the
  operative mapping — do not assume `victim = max - eater_rate`.
- `dram` accounting counts memcpy as 2x bytes; with write-allocate the true DRAM
  traffic can be up to 3x. Calibrate against the victim, not against absolute
  numbers.
- An `h2d` C2C copy also reads host DRAM and writes HBM at the same rate — cross-talk
  onto the other two domains is physics, not a tool artifact (quantified in the doc).
- **Workstation-class GPUs (big L2): the CE number is authoritative for HBM.**
  On parts like the RTX PRO 6000 Blackwell (128 MB L2) the `hbm --engine sm`
  streaming victim measured absurd cache-resident rates (external-validation
  defect 3). The victim now sizes its default buffer off the device's L2
  (`max(1 GiB, 8x l2CacheSize)`), but when in doubt trust
  `--domain hbm --engine ce` (copy-engine memcpy cannot be L2-satisfied at
  those sizes) and treat the SM number as a cross-check only.
