# I/O throttling for hwsim validation (Workstream 3)

Goal: reduce the effective disk-read bandwidth available to Sirius by a controlled,
measurable factor, so real degraded runs can validate the simulator's `io_bandwidth`
knob. Tool and usage: [`../throttle/io/`](../throttle/io/README.md).

Everything below was measured on **pmgb300ws-0163** (GB300 box): dataset device is
`/dev/nvme0n1` (KIOXIA XG10d SED, 3.7 TB, ext4 root partition `nvme0n1p2` = 259:5,
whole disk = **259:2**), scheduler `none`, `max_sectors_kb=256`, all 72 Grace cores on
NUMA node 0. The SF1000 TPC-H parquet set lives on this device
(`/localhome/local-faramburu/tpch_parquet_sf1000`). `/raid` (3x NVMe md RAID0) is
root-owned and not where benchmark data lives.

## Approach 1 (primary): competing-load injector — `io_load`

A background process issues O_DIRECT reads against its **own scratch file on the same
device**, discards the bytes, and holds a target aggregate rate:

- **Token bucket** (100 ms refill, 200 ms burst cap) enforces the rate ceiling;
  bucket is pre-filled one tick so there is no startup hole.
- **Closed-loop queue-depth controller**: N worker threads are parallel sequential
  streams; the controller parks/unparks them (grows QD when a 500 ms moving average
  falls >3% below target, decays it slowly when comfortably at target). This is what
  keeps the rate held *while the victim is also hammering the device* — a static QD
  either undershoots under contention or over-pressures when alone.
- Blocking `pread` at 1 MiB per request (the block layer splits these into
  `max_sectors_kb`=256 KiB device commands; observed `rareq-sz` ≈ 226 KiB — no read
  amplification for aligned reads).
- SIGINT/SIGTERM → clean stop, summary line + per-100 ms CSV of achieved rate.

Rate-hold accuracy when not device-limited: **within ~1%** (e.g. 999.4 MB/s at a
1000 MB/s target; 1630.5 at 1631.2; 3224 at 3262 under a saturating victim).

## Calibration (measured 2026-08-04, disk otherwise idle)

Protocol = `calibrate.sh`: victim is 8 parallel sequential O_DIRECT streams, 1 MiB
requests (a stand-in for Sirius parquet scans); injector runs in the background at a
fraction of the victim-alone baseline; victim measured for 8 s after 2 s warmup.
Total disk-busy time ≈ 90 s.

**Baseline** (victim alone, 8 streams): **6.525 GB/s**.
Device ceiling at QD 32: **7.167 GB/s** (the 8-stream victim reaches ~91% of it).

| injector target (frac x baseline) | injector achieved GB/s | victim GB/s | victim / baseline |
|---|---|---|---|
| 0 | — | 6.525 | 1.000 |
| 0.25 (1.631 GB/s), 8 thr | 1.631 | 5.452 | 0.836 |
| 0.50 (3.263 GB/s), 8 thr | 3.224 | 3.892 | 0.596 |
| 0.75 (4.894 GB/s), 8 thr | 3.462 (saturated) | 3.805 | 0.583 |
| 0.75 (4.894 GB/s), **32 thr** | 4.583 | 2.314 | **0.355** |

Two regimes, both well-behaved:

1. **Injector holds target** (enough QD headroom): the mapping is linear with
   slope −1 in the injector rate:
   `victim_GB/s ≈ C_mix − R_inj`, with mixed-load aggregate capacity
   **C_mix ≈ 6.9–7.1 GB/s** (slightly above the 8-stream baseline because total
   outstanding QD is higher under mixed load). Fit residuals < 2% on all rows.
2. **Injector out-queued** (8 thr vs 8-thr victim at the 0.75 point): with the `none`
   scheduler, device share is proportional to queued requests, so equal thread counts
   fair-share ~50/50 and the injector can't exceed ~C_mix/2. Fix: give the injector
   more threads than the victim (32 vs 8 holds 94% of the 0.75 target).

So the practical recipe for a target victim fraction `v` of baseline `B`:
run the injector at `R_inj = C_mix − v·B` with `--threads 32`. Monotonic and
predictable across the measured range (1.000 → 0.836 → 0.596 → 0.355).

### Feeding the simulator's `io_bandwidth` knob

For a degraded validation run, the knob value is the **victim's achieved bandwidth**,
not the injector's nominal fraction: measure `victim_GB/s` from the table (or re-measure
with a 5 s `io_load --rate 0` probe while the injector runs), and set
`io_bandwidth = victim_GB/s / baseline_GB/s`. The simulator's predicted wall/behavior
under that factor is then compared against the real run executed alongside the same
injector settings. The linear `C_mix − R_inj` model lets us dial arbitrary factors in
[0.35, 1.0] without re-calibrating; below ~0.35 add injector threads and re-measure
(expect the same linear law until the injector itself hits device limits).

### Sensitivity notes

- **Access pattern (big one)**: a *random*-offset injector (1 MiB, QD 32) at a nominal
  0.50 target achieved only 2.43 GB/s yet crushed the victim to **0.757 GB/s (0.116 of
  baseline)** — total device throughput collapsed to ~3.2 GB/s. Random competition
  destroys the drive's sequential efficiency and makes the mapping non-linear and
  target-unreachable. **Always inject sequential streams** (the default).
- **Injector thread count**: irrelevant below ~0.4·B, decisive above (regime 2). Use 32.
- **Request size**: 1 MiB chosen to match parquet-scan-sized reads; the block layer
  splits to 256 KiB anyway, so 256 KiB–4 MiB behaves the same on this device.
- **Run-to-run**: repeated victim-alone probes were within ~2%; combined-run rows are
  averages over 8 s and stable at that horizon.

## Approach 2 (cross-check): cgroup v2 `io.max`

Direct limiting: cap the Sirius process's read bytes/s at the block layer.

**Not usable without root on this box.** Findings (2026-08-04):

- cgroup v2 is mounted; the root `cgroup.subtree_control` enables
  `cpuset cpu io memory pids` — the `io` controller exists and works.
- But `user.slice` and `user-2099.slice` propagate only `cpu memory pids`, so the
  user manager (`user@2099.service`) has **no `io` controller delegated** →
  `systemd-run --user -p IOReadBandwidthMax=...` silently doesn't enforce, and no
  user-writable cgroup can carry `io.max`. Root is required.

Exact commands for an admin (throttling keys on the **whole disk** 259:2, not the
partition):

```bash
# one-off scope around a Sirius run (systemd sets up and tears down the cgroup):
sudo systemd-run --scope -p "IOReadBandwidthMax=/dev/nvme0n1 3200M" \
    -p "User=local-faramburu" <sirius/duckdb command...>

# or manually, attaching an already-running pid:
sudo mkdir /sys/fs/cgroup/hwsim-io
echo "259:2 rbps=3200000000" | sudo tee /sys/fs/cgroup/hwsim-io/io.max
echo <sirius_pid>            | sudo tee /sys/fs/cgroup/hwsim-io/cgroup.procs
# revert:
echo "259:2 rbps=max"        | sudo tee /sys/fs/cgroup/hwsim-io/io.max
sudo rmdir /sys/fs/cgroup/hwsim-io    # after moving the pid back out

# permanent alternative: delegate io to user slices (then no sudo per run):
sudo systemctl set-property user.slice IOAccounting=yes   # plus DelegateControllers
```

Trade-offs vs the injector: `io.max` is exact and per-process (no competing traffic,
no C_mix model needed) but throttles at the target cgroup only — it models "this
process has a slower disk" whereas the injector models "the disk is busier/slower for
everyone". For simulator validation the two should agree on scan-bound queries; when
root access is available, run one `io.max` point (e.g. rbps = 0.5·B) as an independent
check of the injector-derived mapping. Note `io.max` throttles buffered reads too
(charged on page-cache miss), so the cold-cache requirement below still applies.

## Alternatives considered

- **fio `--rate`** (fio 3.36 installed): legitimate building block
  (`fio --name=inj --filename=inject.dat --direct=1 --rw=read --bs=1M --numjobs=8
  --rate=400m --time_based --runtime=60`), but `--rate` is per-job and open-loop —
  under victim contention jobs individually undershoot and nothing rebalances; no
  aggregate closed loop, CSV timeline requires log post-processing. Kept as a sanity
  cross-check, not the tool.
- **dm-delay / dm-flakey**: device-mapper latency/error injection — needs root *and* a
  remap of the block device; impossible for the mounted root filesystem the dataset
  sits on. Would be the right tool for a dedicated data disk on a sacrificial mount.
- **ionice / BFQ weights**: the NVMe scheduler here is `none`; ionice classes have no
  effect without BFQ, and io.weight (proportional) doesn't give an absolute cap anyway.
- **blkio (cgroup v1)**: box is pure v2; not applicable.

## Pitfalls (handled / to keep handling)

- **Page cache**: both injector and victim use O_DIRECT (4096-aligned buffers,
  request size multiple of 4 KiB, offsets aligned to request size — valid for 512e
  and 4Kn). But *Sirius/DuckDB parquet reads are buffered*: the injector slows the
  device, not cache hits. With 506 GB RAM vs a 265 GB dataset, a warm run would see
  almost no I/O at all. Before each validation run, evict the dataset:
  `for f in /path/to/tpch_parquet_sf1000/*.parquet; do io_load --file "$f" --evict; done`
  (`posix_fadvise DONTNEED`, owner permission — no root needed).
- **Same file vs same device**: the injector reads its **own scratch file** on the same
  filesystem. Contention happens at the block layer, which is what we want; sharing the
  victim's file adds inode/readahead interactions and risks touching real datasets.
  Never point it at the dataset.
- **Scratch files must have real extents**: `fallocate` alone leaves unwritten extents
  that ext4 answers as zero-fill *without any device I/O* — benchmarks against such a
  file measure nothing. `--mkfile` does O_DIRECT writes. Zero-filled contents are fine
  on this drive: it's a SED (encrypts at rest), so media reads are incompressible
  regardless of logical content.
- **Read amplification / request splitting**: reads are split to `max_sectors_kb`
  (256 KiB) device commands; aligned 1 MiB requests have zero amplification.
- **I/O scheduler**: `none` on all NVMe here → sharing is proportional to submitted
  queue slots; this is why injector QD must exceed the victim's for deep throttling,
  and why ionice is inert. If a box uses `mq-deadline`/`bfq`, re-run `calibrate.sh` —
  the C_mix model will differ.
- **NUMA**: all 72 cores and all CPU DRAM are node 0 on this box, so placement is
  trivially right. On multi-node hosts, pin injector threads to the node owning the
  NVMe's PCIe root complex (`numactl --cpunodebind=N --membind=N`), else the injector
  measures partly its own cross-socket latency.
- **Shared box**: check `iostat -x 1 2 /dev/nvme0n1` is idle before measuring; the
  full calibration keeps total disk-busy time ≈ 90 s.

## Status / follow-ups

- Injector prototype done and calibrated (table above); `calibrate.sh` reproduces the
  table in ~90 s of disk time.
- Cross-check `io.max` run pending root access (commands above); one point at
  rbps = 0.5·B would independently validate the mapping.
- For WS7/WS8: a degraded-run recipe is
  (1) evict dataset pages, (2) start injector at `R_inj = C_mix − v·B`, 32 threads,
  (3) run the query, (4) record the injector CSV alongside Quent telemetry so the
  achieved (not nominal) degradation factor is stored with the run.
