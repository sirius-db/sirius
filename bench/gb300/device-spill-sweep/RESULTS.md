# GB300 GPU memory limit sweep

Measured on GB300 on 2026-11-20 with Sirius main
`26b5caba3e2c96290fdc63d4cf40ac42572eaab8`. The extension SHA-256 was
`25a9c16319571926356c813bfbf145a3b2ed0d61c76771e83ee48f9e24389fa6`.
The commands were `bash bench/gb300/device-spill-sweep/run_sf.sh <scale>` for
scales 1000, 3000, and 10000. For SF10000, the runner
skipped the already completed 32 GB results and then ran 64, 128, and 256 GB.
Each of the 22 TPC-H queries ran once at each limit, unpinned, in a fresh Sirius
process against the same Parquet datasets used by the host-memory sweep.
All 264 queries finished with GPU plans.
The [all-scale reproduction script](reproduce_all_sf.sh) runs SF1000, SF3000,
and SF10000 sequentially into a fresh output directory.

The base was `bench/gb300/configs/cold.yaml`, with
`memory.gpu.usage_limit_bytes` set to the indicated decimal GB limit, host cap
fixed at 384 GiB, and initial host pools at 192. Spill compression stayed on,
max readahead scans stayed at 8, and no `--profile` override or query-specific
config was used. Quent telemetry was on.

SF1000 and SF3000 recorded no disk movement at any GPU limit with this host
budget. Their GPU→HOST movement fell as the GPU limit increased, while the
runtime improvement was modest compared with SF10000.

## SF1000

| GPU limit | Queries | Sum of query runtimes | GPU→HOST | HOST→DISK | GPU→DISK | Total into disk |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 GB | 22/22 | 139.302 s | 108.179 GiB | 0 GiB | 0 GiB | 0 GiB |
| 64 GB | 22/22 | 137.559 s | 62.012 GiB | 0 GiB | 0 GiB | 0 GiB |
| 128 GB | 22/22 | 136.304 s | 35.228 GiB | 0 GiB | 0 GiB | 0 GiB |
| 256 GB | 22/22 | 137.464 s | 27.611 GiB | 0 GiB | 0 GiB | 0 GiB |

Exact per-query results:
[32 GB](results/sf1000-gpu32gb-host384gib-per-query.csv),
[64 GB](results/sf1000-gpu64gb-host384gib-per-query.csv),
[128 GB](results/sf1000-gpu128gb-host384gib-per-query.csv), and
[256 GB](results/sf1000-gpu256gb-host384gib-per-query.csv). Matching exact-byte
totals files are alongside them. The runner exited with code 0.

## SF3000

| GPU limit | Queries | Sum of query runtimes | GPU→HOST | HOST→DISK | GPU→DISK | Total into disk |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 GB | 22/22 | 235.793 s | 529.137 GiB | 0 GiB | 0 GiB | 0 GiB |
| 64 GB | 22/22 | 215.665 s | 330.011 GiB | 0 GiB | 0 GiB | 0 GiB |
| 128 GB | 22/22 | 206.210 s | 170.784 GiB | 0 GiB | 0 GiB | 0 GiB |
| 256 GB | 22/22 | 206.495 s | 139.537 GiB | 0 GiB | 0 GiB | 0 GiB |

Exact per-query results:
[32 GB](results/sf3000-gpu32gb-host384gib-per-query.csv),
[64 GB](results/sf3000-gpu64gb-host384gib-per-query.csv),
[128 GB](results/sf3000-gpu128gb-host384gib-per-query.csv), and
[256 GB](results/sf3000-gpu256gb-host384gib-per-query.csv). Matching exact-byte
totals files are alongside them. The runner exited with code 0.

## SF10000

| GPU limit | Queries | Sum of query runtimes | GPU→HOST | HOST→DISK | GPU→DISK | Total into disk |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 GB | 22/22 | 1,588.921 s | 2,848.782 GiB | 993.122 GiB | 37.765 GiB | 1,030.887 GiB |
| 64 GB | 22/22 | 982.456 s | 2,209.570 GiB | 454.757 GiB | 7.881 GiB | 462.638 GiB |
| 128 GB | 22/22 | 811.486 s | 1,483.199 GiB | 190.755 GiB | 2.633 GiB | 193.388 GiB |
| 256 GB | 22/22 | 777.086 s | 772.163 GiB | 164.343 GiB | 4.682 GiB | 169.025 GiB |

All traffic columns sum one-way DataBatch conversions during query execution
and use the destination representation size. They measure cumulative movement,
not unique data volume or peak occupancy. Some individual queries do not
improve monotonically; for example, Q9 moved more from host to disk at 256 GB
than at 128 GB. These are single-iteration measurements. The GPU limits apply
to Sirius-managed memory, not all device allocations reported by the GPU.
The benchmark requests that the OS discard cached dataset pages, but a fully
cold page cache is not guaranteed.

The exact per-query and byte-count totals for SF10000 are saved for
[32 GB](results/sf10000-gpu32gb-host384gib-per-query.csv),
[64 GB](results/sf10000-gpu64gb-host384gib-per-query.csv),
[128 GB](results/sf10000-gpu128gb-host384gib-per-query.csv), and
[256 GB](results/sf10000-gpu256gb-host384gib-per-query.csv). Matching totals
files are alongside them. The effective per-query YAML, logs, and raw Quent
traces remain on the GB300 under
`test_datasets/gb300-device-spill-sweep/host384Gi/sf<scale>/` for each scale
factor. The SF10000 runner exited with code 0. Earlier incomplete GPU 16 GB
attempts are archived separately and are not part of these totals.
