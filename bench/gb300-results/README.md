# GB300 TPC-H benchmark snapshots

These are the selected GB300 measurements at Sirius commit
`1dbb366baaf7bd26325ede685346ecb6d5fe7692`. The cold/hot/per-query-pin
snapshot uses one GPU and TPC-H Parquet at SF1000, SF3000, and SF10000; the
concurrency snapshot below covers SF100, SF300, and SF1000. The configs are in
`bench/gb300-cold/`,
`bench/gb300-hot/`, `bench/gb300-pin/`, and the two SF10k Q10 directories. The
`--profile` option now preserves those YAML cache settings.

| Scale | Cold, iterations 0 / 1 (s) | Hot, iterations 0 / 1 (s) | Host pin (s) |
| ---: | ---: | ---: | ---: |
| SF1000 | 65.080 / 65.099 | 47.432 / 40.771 | 9.376 |
| SF3000 | 195.736 / 195.548 | 149.854 / 119.086 | 28.740 |
| SF10000 | 767.324 / 783.834 | 777.722 / 740.332 | Unavailable: Q13 host pin failed |

Times sum the 22 query executions; setup, cache flushing, and pin/unpin time are
excluded. Cold and hot use two iterations, spill compression, RA8, and no pinning.
Cold drops the local dataset-file cache and resets the Sirius cache before each
query. Hot groups iterations by query and uses LRU retention from its YAML.

SF10000 cold and hot totals combine a 21-query run with a separate Q10 run using
the saved 8 GiB build-hash-table limit. They are composite query-time sums, not
uninterrupted 22-query session times. The regular cold and hot configs use the
same settings at all three scales; the Q10 configs differ only by that hash
limit.

The SF1000 and SF3000 pin results use `bench/gb300-pin/config.yaml`, one hot-profile
iteration, per-query host pinning, and Simpatico input compression. The pinned
`lineitem` and `orders` plans are the repository's
`src/compression/simpatico_codegen/plans/tpch_sf1000/lineitem.txt` and
`orders.txt`. All 22 queries completed on Sirius without CPU fallback in those
two selected runs. At SF10000, per-query host pinning failed before Q13 executed:
the 450 GiB host pool ran out of blocks after a 448.12 GiB lifetime peak. There
is no selected SF10000 pin timing to report.

Each `sf*/` directory contains the selected per-query CSVs. `manifest.json`
records the exact totals, config paths, checksums, and the SF10000 pin failure.
CSV line endings were normalized from CRLF to LF without changing the values;
the manifest retains both the original and repository checksums.
The later SF10000 fresh-instance cold rerun remains outside this snapshot.

## Host-pinned Parquet concurrency

The [concurrency results](concurrency/summary.csv) add SF100, SF300, and SF1000
with 1, 2, 4, 8, and 10 simultaneous query streams. Each stream executes all 22
queries in its TPC-H throughput permutation with qgen-generated, stream-specific
predicate values. This read-only test has no refresh stream and is not an
official TPC-H throughput score. One run was measured per case.

| Scale | 1 stream | 2 streams | 4 streams | 8 streams | 10 streams |
| ---: | ---: | ---: | ---: | ---: | ---: |
| SF100 | 12.416 | 18.096 | 22.509 | 25.371 | 25.473 |
| SF300 | 5.725 | 7.415 | 8.508 | 8.807 | 8.619 |
| SF1000 | 1.955 | 2.421 | 2.465 | 2.459 | 2.451 |

Values are completed queries per second (`22 × streams / wall_seconds`), with
upfront host pinning excluded. Before each case, a fresh Sirius process pins the
union of all query columns in host memory using the Simpatico plans in
`bench/sf1000-repro/plans/`, then runs its streams concurrently. Approximate pin
times were 3.4 s at SF100, 10.0 s at SF300, and 39–40 s at SF1000. Throughput
flattens by 8 streams in this single-pass sample.

The [concurrency manifest](concurrency/manifest.json) records seed `1111221257`,
the Sirius source commit and binary checksum, config and plan checksums, and the
generated stream SQL checksums. Streams 1–8 in the 10-stream case were
byte-identical to those in the earlier 1/2/4/8 cases. Each scale directory
contains five per-query timing CSVs, one for each stream-count case.
The initial run used a runner limited to 1/2/4/8 choices; the later 10-stream
run used the same workload logic with 10 added as an option. Both runner hashes
are recorded in the manifest.
