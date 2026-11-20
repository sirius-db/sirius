# SF10000 at a 32 GB Sirius GPU limit

Measured on GB300 on 2026-11-20 with Sirius main `26b5caba3e2c96290fdc63d4cf40ac42572eaab8`.
The extension SHA-256 was
`25a9c16319571926356c813bfbf145a3b2ed0d61c76771e83ee48f9e24389fa6`.
The command was
`bash bench/gb300-device-spill-sweep/run_sf.sh 10000 --caps 32`.
Each of the 22 TPC-H queries ran once, unpinned, in a fresh Sirius process
against the same SF10000 Parquet dataset used by the host-memory sweep.
The base was `bench/gb300-cold/config.yaml`, with `memory.gpu.usage_limit_bytes`
set to 32,000,000,000, host cap fixed at 384 GiB and initial pools at 192.
Spill compression stayed on, max readahead scans stayed at 8, and no
`--profile` override or query-specific config was used. Quent telemetry was on.

| Result | Value |
| --- | ---: |
| Successful GPU query plans | 22/22 |
| Sum of query runtimes | 1,588.921 s (26m 28.9s) |
| GPU→HOST | 3,058,856,018,757 bytes (2,848.782 GiB) |
| HOST→DISK | 1,066,356,907,973 bytes (993.122 GiB) |
| GPU→DISK | 40,549,759,928 bytes (37.765 GiB) |
| Total moved into disk | 1,106,906,667,901 bytes (1,030.887 GiB) |

Q9 took 243.903 s and Q18 took 386.959 s, the longest query. The counters
sum successful one-way DataBatch conversions during query execution and use
the destination representation size. They measure cumulative movement, not
unique data volume or peak occupancy. The 32 GB setting limits Sirius-managed
GPU memory; it is not a bound on all device allocations reported by the GPU.
The benchmark requests that the OS discard cached dataset pages, but a fully
cold page cache is not guaranteed.

Exact [per-query values](results/sf10000-gpu32gb-host384gib-per-query.csv) and
[totals](results/sf10000-gpu32gb-host384gib-totals.csv) are saved here. The
effective per-query YAML, logs, and raw Quent traces remain on the GB300 at
`test_datasets/gb300-device-spill-sweep/host384Gi/sf10000/`; the persistent
runner exited with code 0. The earlier incomplete GPU 16 GB attempts are
archived separately and are not part of these totals.
