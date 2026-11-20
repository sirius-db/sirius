# GB300 host-memory spill sweep

Measured on GB300 on 2026-11-19 at Sirius main `26b5caba`. Each of the 22
TPC-H queries ran once, unpinned and in a fresh Sirius process, at each of
the 32, 64, 128, 256, and 450 GiB host caps. The base was
`bench/gb300-cold/config.yaml` with spill compression enabled. Only the host
capacity, proportional initial pool count, and Quent output path changed.
All 330 final query runs completed with confirmed GPU plans; no query-specific
config was needed. SF10000 needed a 65,536 soft open-file limit because of
its many Parquet parts. An initial Q1 attempt at the OS default 1,024-file
limit failed, was archived, and was rerun successfully.

The table separates **three logical DataBatch transitions**. Values are GiB
of destination representation size summed over the 22 queries at each cap.
Exact byte counts and per-query values are in the linked CSVs.

| Scale factor | Host cap | Initial allocation | Queries | GPU→HOST | HOST→DISK | GPU→DISK |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SF1000 | 32 GiB | 8 GiB | 22/22 | 12.324 | 10.820 | 16.764 |
| SF1000 | 64 GiB | 16 GiB | 22/22 | 26.125 | 16.998 | 0.000 |
| SF1000 | 128 GiB | 32 GiB | 22/22 | 29.520 | 0.000 | 0.000 |
| SF1000 | 256 GiB | 64 GiB | 22/22 | 27.106 | 0.000 | 0.000 |
| SF1000 | 450 GiB | 112.5 GiB | 22/22 | 29.548 | 0.000 | 0.000 |
| SF3000 | 32 GiB | 8 GiB | 22/22 | 62.758 | 38.565 | 67.280 |
| SF3000 | 64 GiB | 16 GiB | 22/22 | 99.005 | 39.598 | 0.000 |
| SF3000 | 128 GiB | 32 GiB | 22/22 | 103.153 | 32.329 | 0.000 |
| SF3000 | 256 GiB | 64 GiB | 22/22 | 87.193 | 0.000 | 0.000 |
| SF3000 | 450 GiB | 112.5 GiB | 22/22 | 118.695 | 0.000 | 0.000 |
| SF10000 | 32 GiB | 8 GiB | 22/22 | 533.031 | 411.147 | 349.675 |
| SF10000 | 64 GiB | 16 GiB | 22/22 | 633.883 | 441.663 | 269.134 |
| SF10000 | 128 GiB | 32 GiB | 22/22 | 739.763 | 481.461 | 59.417 |
| SF10000 | 256 GiB | 64 GiB | 22/22 | 771.949 | 260.845 | 8.693 |
| SF10000 | 450 GiB | 112.5 GiB | 22/22 | 809.886 | 63.135 | 0.000 |

GPU→DISK is a direct *logical* transition, although its I/O path stages data
physically through temporary pinned host buffers. Those buffers are not
HOST-tier placement and therefore do not increment GPU→HOST. The counters
are cumulative completed moves, not unique data volume, peak occupancy, or
physical GPU/host/disk bandwidth. Compression can change the representation
size at each destination, so the three columns should not be added as if
they were the same bytes moving through consecutive tiers.

At SF1000, disk movement stops at the 128 GiB host cap. At SF3000 it stops
at 256 GiB. At SF10000, disk movement remains at 450 GiB; all 63.135 GiB of
HOST→DISK at that cap is from Q9. The runner did not clear the OS page cache,
since this sweep measures spilling rather than cold-I/O timing.

Exact results: [SF1000](results/sf1000/per-query.csv),
[SF3000](results/sf3000/per-query.csv), and
[SF10000](results/sf10000/per-query.csv). The raw Quent traces remain under
`test_datasets/gb300-spill-sweep/` on the GB300 machine. The saved runner can
recount the three transitions from these traces without rerunning queries.
