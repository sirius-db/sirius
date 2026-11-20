# G7e.8xlarge benchmarks

The [current configs](configs/) target one GPU on AWS `g7e.8xlarge`. They were
saved against Sirius main `6c53654c0498026d8b48daa7e1f19edd3c5ff988`. No timings
have been selected for these updated presets.

| Preset | Eviction | Pipeline threads | REST reactors | Read-ahead |
| --- | --- | ---: | ---: | ---: |
| `cold.yaml` | idle | 4 | 4 | 16 |
| `hot.yaml` | lru | 4 | 4 | 16 |
| `pinned.yaml` | lru | 4 | 4 | 16 |

All use `backend: native`, `cache.mode: cucs`, eviction threshold 0.8, 225 GiB
host capacity, and 1 TiB compressed spill. Pinned is byte-for-byte identical to
hot. The runner selects actual input pinning separately. The old three
`SIRIUS_EXP_*` master flags are retired; their replacement optimization settings
default to true, so no environment sidecar is needed.

## Replay

```bash
bash bench/g7e/replay.sh 1000 cold --iterations 2 --dry-run
bash bench/g7e/replay.sh 3000 hot --iterations 2 --dry-run
bash bench/g7e/replay.sh 10000 cold --iterations 1 --dry-run
bash bench/g7e/replay.sh 1000 pinned --pin-tier host --dry-run
```

G7e cold starts a fresh Sirius process and clears the local dataset-file OS
cache and Sirius cache before **every query iteration**, in suite
order: Q1(1), ..., Q22(1), Q1(2), ..., Q22(2). Hot starts a fresh process for
**each query** and reuses that process for its iterations: Q1(1), Q1(2), restart,
Q2(1), Q2(2), and so on. Pinned uses hot's process lifetime with explicit host
or GPU pinning. No extra warmup is hidden in these measurements.

For an actual run, pass a complete local Parquet directory or a same-region S3
prefix using `--input`, then remove `--dry-run`. The script verifies the actual
EC2 region, bucket location and table presence, and requires suitable AWS read
credentials. It does not launch an instance or generate a dataset.

`/mnt/datasets/spill` must already exist on writable local NVMe instance storage
with space for the configured spill tier. The runner follows the backing
devices and rejects EBS by default, including EBS devices named `nvme*`.
`--allow-ebs-spill` is an explicit opt-in exception. The runner never formats
disks or silently changes spill paths. Recheck mounts after EC2 stop/start.

## Historical results

The [November 11 baseline](results/baseline-20261111/README.md) contains the
previously frozen SF1000/SF3000 cold/hot and SF10000 cold/incomplete-hot results.
Its measured configs and flags are archived beside the timings. These are
older configs and source commits, not results from the current presets above.
No pinned result is available. Raw Quent traces remain in the separately
retained archive identified by the baseline manifest.
