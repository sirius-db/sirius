# GB300 host-memory spill sweep

## SF1000

Measured on GB300 on 2026-11-19 at Sirius main `26b5caba`. Each of the 22
TPC-H queries ran once, unpinned and in a fresh Sirius process, at every host
cap. The base config was `bench/gb300-cold/config.yaml` with spill compression
enabled; only `memory.host.capacity_bytes`, its proportional
`initial_number_pools`, and the Quent output path changed. All 110 queries
completed and their Sirius logs confirmed GPU plans. No query-specific config
was needed.

| Host cap | Initial allocation | Queries | Into host (bytes) | Into disk (bytes) | Into host (GiB) | Into disk (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 GiB | 8 GiB | 22/22 | 13,232,503,880 | 29,617,979,175 | 12.324 | 27.584 |
| 64 GiB | 16 GiB | 22/22 | 28,051,668,126 | 18,251,699,318 | 26.125 | 16.998 |
| 128 GiB | 32 GiB | 22/22 | 31,696,764,483 | 0 | 29.520 | 0 |
| 256 GiB | 64 GiB | 22/22 | 29,104,534,637 | 0 | 27.106 | 0 |
| 450 GiB | 112.5 GiB | 22/22 | 31,726,857,130 | 0 | 29.548 | 0 |

At 64 GiB, Q9 alone accounts for all 18,251,699,318 bytes moved into disk.
At 32 GiB, Q9 accounts for 23,426,274,114 of the 29,617,979,175 disk bytes;
Q3, Q5, Q8, Q10, Q15, Q19, Q20, and Q21 account for the remainder.

These figures sum completed GPU→HOST, HOST→DISK, and GPU→DISK batch moves during
query execution. Each move is counted by its destination representation size,
so compression affects the byte count. They are cumulative one-way movements,
not peak memory use. The runner does not force an OS page-cache state; these
figures are intended for spilling analysis, not a cold-I/O timing comparison.
Exact per-query values are in [SF1000 results](results/sf1000/per-query.csv).
Preserved Quent traces remain under
`test_datasets/gb300-spill-sweep/sf1000/` on the GB300 machine.

## SF3000

Measured with the same runner and Sirius main `26b5caba` on 2026-11-19.
All 110 query runs completed and their Sirius logs confirmed GPU plans.
No query-specific config was needed.

| Host cap | Initial allocation | Queries | Into host (bytes) | Into disk (bytes) | Into host (GiB) | Into disk (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 GiB | 8 GiB | 22/22 | 67,386,149,951 | 113,649,828,593 | 62.758 | 105.845 |
| 64 GiB | 16 GiB | 22/22 | 106,305,587,697 | 42,517,848,782 | 99.005 | 39.598 |
| 128 GiB | 32 GiB | 22/22 | 110,759,374,211 | 34,712,981,196 | 103.153 | 32.329 |
| 256 GiB | 64 GiB | 22/22 | 93,622,288,647 | 0 | 87.193 | 0 |
| 450 GiB | 112.5 GiB | 22/22 | 127,447,551,261 | 0 | 118.695 | 0 |

At 128 GiB, only Q9 and Q14 moved data into disk: 15,152,979,652 and
19,560,001,544 bytes respectively. Of the five caps tested, 256 GiB was the
smallest with zero disk movement across all 22 queries. The counts use the
same cumulative, compressed-destination definition described above. Exact
per-query values are in [SF3000 results](results/sf3000/per-query.csv).
Quent traces remain under `test_datasets/gb300-spill-sweep/sf3000/` on the
GB300 machine.

## SF10000

Measured with the same runner and Sirius main `26b5caba` on 2026-11-19.
All 110 final query runs completed successfully and their Sirius logs confirmed GPU plans.
The SF10000 runner raised its soft open-file limit to 65,536 because the
dataset has many Parquet parts. An initial Q1 attempt at the OS default
1,024-file limit failed; it was archived and rerun successfully. No
query-specific config was needed, including for Q10.

| Host cap | Initial allocation | Queries | Into host (bytes) | Into disk (bytes) | Into host (GiB) | Into disk (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 GiB | 8 GiB | 22/22 | 572,337,199,793 | 816,925,822,226 | 533.031 | 760.821 |
| 64 GiB | 16 GiB | 22/22 | 680,627,177,203 | 763,212,533,298 | 633.883 | 710.797 |
| 128 GiB | 32 GiB | 22/22 | 794,314,793,977 | 580,763,606,458 | 739.763 | 540.878 |
| 256 GiB | 64 GiB | 22/22 | 828,873,670,832 | 289,413,628,548 | 771.949 | 269.537 |
| 450 GiB | 112.5 GiB | 22/22 | 869,608,914,928 | 67,790,995,998 | 809.886 | 63.135 |

At 450 GiB, Q9 alone accounts for all 67,790,995,998 bytes moved into disk;
every other query recorded zero disk movement. These are cumulative one-way
batch moves using compressed destination sizes, as described above, rather
than peak memory use. Exact per-query values are in
[SF10000 results](results/sf10000/per-query.csv). Quent traces remain under
`test_datasets/gb300-spill-sweep/sf10000/` on the GB300 machine.
