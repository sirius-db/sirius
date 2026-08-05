# io_load — O_DIRECT read-load injector (hwsim WS3)

Physically emulates "slower disk" for Sirius by stealing a controlled share of the
NVMe device's read bandwidth. One small C binary, no dependencies beyond pthreads.
Design, calibration table, and pitfalls: [`../../docs/io-throttle.md`](../../docs/io-throttle.md).

## Build

```bash
make          # -> ./io_load
```

## Roles

| Role | Invocation | Behavior |
|------|-----------|----------|
| Injector | `--rate <GB/s>` or `--fraction F --baseline-gbps B` | Holds a target aggregate O_DIRECT read rate (token bucket + closed-loop queue-depth controller), discards bytes. |
| Victim / baseline | `--rate 0` (default) | Reads flat-out with N parallel sequential streams; reports achieved bandwidth. |
| Scratch-file maker | `--mkfile <GiB>` | Creates the file with O_DIRECT writes (real extents — do **not** use bare `fallocate`, ext4 serves unwritten extents as zero-fill without touching the media). |
| Cache evictor | `--evict` | Drops the file's pages from the page cache (`posix_fadvise DONTNEED`, owner permission, no root). Run over the parquet files before a validation run so Sirius's buffered reads start cold. |

Rates are decimal (1 GB/s = 1e9 B/s). All reads/writes are O_DIRECT with
4096-aligned buffers, request sizes that are multiples of 4 KiB, and offsets
aligned to the request size. SIGINT/SIGTERM stop any mode cleanly (summary and
CSV are still written; exit code 0).

## Usage examples

```bash
# 0. scratch files on the SAME filesystem/device the dataset lives on
#    (never point the injector at the dataset itself)
./io_load --file /path/on/same/fs/inject.dat --mkfile 8
./io_load --file /path/on/same/fs/victim.dat --mkfile 8

# 1. measure the baseline (victim alone, like a parquet scan: 8 seq streams, 1 MiB)
./io_load --file victim.dat --rate 0 --threads 8 --req-kb 1024 --warmup 2 --duration 6

# 2. run the injector at 3.2 GB/s with a rate timeline CSV
#    (use --threads 32: it must out-queue the victim to hold high rates, see docs)
./io_load --file inject.dat --rate 3.2 --threads 32 --duration 60 --csv inject.csv

# 2b. same thing as a fraction of a measured baseline
./io_load --file inject.dat --fraction 0.5 --baseline-gbps 6.525 --threads 32 --duration 60

# 3. run it behind a Sirius benchmark, stop it cleanly afterwards
./io_load --file inject.dat --rate 3.2 --threads 32 --duration 3600 --csv inject.csv &
INJ=$!
# ... evict dataset pages, run the Sirius query ...
kill -TERM $INJ && wait $INJ    # prints the SUMMARY line

# full calibration protocol (~90 s of disk time, writes 2x 8 GiB scratch files)
./calibrate.sh /path/on/same/fs/scratch ./results
```

## Options

```
--file PATH        file to read (O_DIRECT)                       [required]
--rate GBPS        target rate, decimal GB/s; 0 = unlimited      [0]
--fraction F       alternative to --rate: F * --baseline-gbps
--baseline-gbps B  baseline for --fraction
--threads N        max queue depth / parallel streams            [8]
--req-kb K         request size in KiB (multiple of 4)           [1024]
--duration S       measured seconds (after warmup)               [10]
--warmup S         seconds excluded from the summary             [0]
--rand             random offsets (DON'T use for injection — see docs)
--csv PATH         per-100ms CSV: time_s,target_mbps,achieved_mbps,active_qd,req_kb,phase
--mkfile GIB       create PATH as scratch file and exit
--evict            drop PATH from the page cache and exit
```

Output ends with a single greppable line:

```
SUMMARY file=... mode=throttled pattern=seq req_kb=1024 threads=32 \
  target_mbps=3262.5 achieved_mbps=3224.4 achieved_gbps=3.224 measured_s=12.01 bytes=... final_qd=7
```

## Rules of thumb (measured on pmgb300ws-0163, nvme0n1)

- Injector `--threads` must exceed the victim's concurrency for fractions ≥ 0.5
  (with the `none` scheduler, device share is proportional to queued requests).
  32 threads holds 0.75 x baseline against an 8-stream victim.
- Keep the injector **sequential** (default). `--rand` collapses total device
  throughput and makes the degradation unpredictable.
- Rate-hold accuracy when not device-limited: within ~1% of target.
