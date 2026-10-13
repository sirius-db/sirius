# Narrowed `.hpln` reads: format options

Status: draft, for discussion. Context: TPC-H SF1000 from S3 on an RTX PRO 4500, ratio
`.hpln` datasets, branch `hpln-prefetch-with-io`.

## Problem

A `.hpln` scan narrows a chunk to the 8,192-row zone-map groups that survive the scan's filter.
To do that it must know where each surviving group's bytes start inside every buffer of every
column it reads. For most buffers that is arithmetic (fixed-width values, dictionary keys read
whole). For **bit-packed** buffers it is not: each 1,024-row block has its own bit width, so a
block's byte offset is a prefix sum over two small per-block arrays, `chunk_count` (int32) and
`chunk_bits` (uint8). Those arrays live **in the payload**, beside the packed data they describe,
so they must be fetched before the narrowed read can even be planned.

Measured on a selective lineitem scan (`l_shipdate <= '1995-08-19'`, 78 chunks, 7 columns, 47% of
groups pruned):

| | before | after the interim fix (below) |
|---|---|---|
| ranges to fetch | 1,092 (205 MB) | same |
| CPU to discover them | 1.56 s (3 rounds of trial subset headers) | 1.9 ms (one header parse) |
| I/O | 0.55 s in 2 batches | 0.55 s in 1 batch on the first scan; 0 s on later scans of the file |
| query (best of 2) | 5.0 s | 3.6 s |

The interim fix (`list_chunk_sizing_ranges` + a per-file `hpln_sizing_store` in the io_context
metadata store) removes the CPU and the repeat I/O. **The first scan of a file still pays one
batched fetch of every narrowed chunk's sizing arrays before its first task can run**, and that
cost grows with the number of chunks, columns and bit-packed leaves. Only a format change removes
it.

## End-to-end results

TPC-H SF1000, 22 queries, one process per query with 2 iterations, data read from S3 on a
100 Gbit/s NIC, with the I/O fixes PR (`fix/io-s3-readahead-and-fused-reads`, which adds the
readahead byte budget and the cache cap) applied: 16 GB readahead budget, prefetching cache capped
at 0.4 of host memory with `eviction: idle`. The `.hpln` arms read the `ratio` datasets. "Before" is the cache-to-device
direct fetch alone; "after" adds batch-wide reads, the 0.9 narrowing threshold, the one-pass
sizing listing and the sizing store. Every query's answer matches parquet.

Dataset sizes on S3:

| Dataset | Total | lineitem | orders | partsupp | customer | part |
|---|---|---|---|---|---|---|
| parquet | 284.0 GB | 177.2 | 52.4 | 37.9 | 10.8 | 5.1 |
| parquet, clustered | 275.2 GB | 168.0 | 52.8 | 37.9 | 10.8 | 5.1 |
| `.hpln` ratio | 272.4 GB | 175.2 | 44.5 | 35.9 | 9.7 | 6.3 |
| `.hpln` ratio, clustered | 262.0 GB | 165.2 | 44.2 | 35.9 | 9.7 | 6.3 |

Suite totals. Cold is the first iteration, hot the second in the same process. Bytes are what the
REST reactors fetched from S3, summed over the 22 queries:

| Arm | Cold | Bytes, cold | Hot | Bytes, hot |
|---|---|---|---|---|
| parquet | 159.6 s | 882 GB | 135.7 s | 878 GB |
| parquet, clustered | 132.6 s | 620 GB | 110.0 s | 615 GB |
| `.hpln` clustered, before | 140.3 s | 612 GB | 99.0 s | 431 GB |
| `.hpln` clustered, after | 121.8 s | 612 GB | 83.8 s | 401 GB |
| `.hpln` unclustered, before | 143.7 s | 872 GB | 107.6 s | 625 GB |
| `.hpln` unclustered, after | 144.7 s | 872 GB | 106.5 s | 633 GB |

- **Cold is the format comparison.** Clustered `.hpln` is 8% faster than clustered parquet and
  unclustered `.hpln` 9% faster than parquet, each reading about the same bytes (612 vs 620 GB, 872
  vs 882 GB). Clustering cuts both formats' bytes by 30%: `.hpln`'s 8,192-row zone maps prune
  about as much as parquet's row groups on this data.
- **The sizing changes do not change what is read** (cold bytes are equal). Clustered they cut
  13% cold and 15% hot by removing latency in front of the first read; unclustered, which
  narrows little, is unchanged.
- **Hot favors `.hpln` for a reason unrelated to the format.** Parquet's hot iteration rereads
  everything, so its 15-17% gain is warm-up alone. `.hpln`'s hot iteration also finds a third of
  its bytes still in the cache (q18: 35.6 GB cold, 2.2 GB hot; q21: 42.9 GB and 4.0 GB). Both
  leave the same data resident after a query; the difference is the evictor. Under `idle` it
  drops a chunk once no live request names it, both formats release their requests at query end,
  and the evictor runs only when a new request is enqueued -- so it runs at the next query's first
  request. A `.hpln` scan creates all of its requests (one per batch) in one burst at scan start,
  so by then about 167 of q18's 180 have already named the previous iteration's chunks again and
  only 129 chunks are evicted. Parquet creates its 370 requests one or two at a time over
  ~0.5 s; only one exists when the evictor runs, and it evicts 31,488 chunks that the rest then
  fetch again (counts from temporary evictor logging on q18).
- **With `eviction: lru` both formats reuse the cache.** `lru` (the code default) frees chunks
  only when the cache needs the room. Same settings otherwise:

  | Arm | Cold, idle / lru | Hot, idle / lru | Bytes hot, idle / lru |
  |---|---|---|---|
  | parquet | 159.6 / 150.7 s | 135.7 / 114.6 s | 878 / 650 GB |
  | parquet, clustered | 132.6 / 130.8 s | 110.0 / 89.7 s | 615 / 364 GB |
  | `.hpln` clustered, after | 121.8 / 122.1 s | 83.8 / 72.1 s | 401 / 250 GB |

  Cold does not move (the cache starts empty in each process; the 6% on unclustered parquet is
  within the build and S3 variation between those runs). Hot, clustered `.hpln` leads clustered
  parquet by 20% instead of 24%, the rest of the old gap having been the eviction race. Queries
  too large to cache (q9, q8, q19, q5) still refetch most of their bytes.

Selective queries, clustered, cold bytes and time (hot time in brackets):

| Query | parquet | `.hpln` before | `.hpln` after |
|---|---|---|---|
| q1 | 16.1 GB, 3.99 s (3.51) | 19.0 GB, 6.18 s (5.03) | 19.0 GB, 4.74 s (3.60) |
| q6 | 5.3 GB, 1.56 s (1.05) | 5.1 GB, 2.63 s (1.65) | 5.1 GB, 1.60 s (0.99) |
| q12 | 8.5 GB, 2.63 s (1.88) | 8.3 GB, 3.90 s (2.63) | 8.3 GB, 2.52 s (1.40) |
| q14 | 2.2 GB, 1.63 s (0.92) | 1.2 GB, 3.98 s (1.52) | 1.2 GB, 1.68 s (0.77) |
| q15 | 3.3 GB, 1.81 s (0.96) | 2.4 GB, 2.28 s (1.46) | 2.4 GB, 1.63 s (0.59) |
| q20 | 14.4 GB, 3.48 s (2.86) | 12.4 GB, 4.11 s (2.82) | 12.4 GB, 3.34 s (1.56) |

Where clustered `.hpln` still trails clustered parquet cold:

| Query | parquet | `.hpln` after |
|---|---|---|
| q1 | 16.1 GB, 3.99 s | 19.0 GB, 4.74 s |
| q9 | 89.0 GB, 20.68 s | 89.5 GB, 22.48 s |
| q10 | 30.1 GB, 5.77 s | 43.2 GB, 7.23 s |
| q13 | 36.3 GB, 5.43 s | 34.8 GB, 6.34 s |

q6, q14 and q16 trail by under 0.1 s. q1 and q10 read 18% and 44% more than parquet (not
investigated yet); q9 and q13 read about the same.

## How other formats avoid it

- **Parquet**: narrowing stops at the row group (about 1M rows here) and, with a page index, at
  the page. The OffsetIndex stores each page's file offset and first row, in the footer area, so
  page-level narrowing needs no payload reads. Sirius does not use the page index today.
- **Vortex** (to verify against current sources): small, independently decodable chunks with every
  segment's offset in the footer's segment map, and FastLanes bit-packing with one bit width per
  array (outliers as patches), so a block's offset inside an array is arithmetic. Pruning skips
  whole segments; nothing narrows inside one.

Both keep everything a planner needs in metadata that is read once, up front.

## Option A: a group offset index beside the zone maps

The writer emits, for every row-indexed buffer of every leaf, the byte offset at which each
zone-map group starts (one offset per group: 9,478 per lineitem chunk), stored with the existing
per-group bounds that bind already reads (`group_bounds`).

- Planning becomes arithmetic: no payload reads, no rounds, no first-scan fetch.
- Size: 4-8 B per group per row-indexed buffer, about 5-10x smaller than `chunk_count` +
  `chunk_bits`, and read together with bounds that are of the same order already.
- Generalizes past bit-packing only if every codec can resume decoding at a group boundary. Codecs
  that need running state (delta: the previous value; RLE: run index and offset within the run)
  must also store that state per group. A codec that cannot resume is marked whole-column, as
  today.
- Cost: writer change per codec family; the decoder's resume path must exist for each codec that
  claims it; bind reads a somewhat larger zone-map segment.

## Option B: independently decodable sub-chunks

Encode each group of rows (a zone-map group or a small run of them; 64K-256K rows suggested) as
its own unit with its own header and buffers, listed in the chunk directory with its offset.

- Narrowing becomes selecting units. No offsets inside buffers, no resume state, and it works for
  any codec and any plan without codec involvement.
- Adjacent surviving units are contiguous, so the REST reactor still fuses them into large GETs.
- Cost: some compression ratio (statistics and dictionaries reset per unit), per-unit header bytes,
  a larger directory, and more decode launches per batch. Unit size trades these against pruning
  resolution.

## Comparison

| | A: offset index | B: sub-chunks |
|---|---|---|
| First-scan planning I/O | none | none |
| Compression ratio | unchanged | slightly lower (unit-size dependent) |
| Works for every codec | only codecs with resume points | yes |
| Writer complexity | per codec | uniform |
| Reader complexity | planner arithmetic + per-codec resume | unit selection only |
| Pruning resolution | 8,192 rows | the unit size |
| Risk | per-codec resume bugs decode wrong data silently unless self-checked | low |

## Recommendation

**Option B** if the goal is "narrowing for any compression plan": it removes codec involvement
entirely and is the more robust design. Choose the unit size by measuring compression ratio and
TPC-H selective-scan time at 64K, 128K and 256K rows on the ratio datasets.

**Option A** if compression ratio must not move and narrowing below the unit size matters; start
with bit-pack only (the case that costs today) and add codecs as they gain resume points.

Either way, keep `list_chunk_sizing_ranges` and the sizing store for files written in the current
format.

## Open questions

1. Unit size for B, and whether units should align with zone-map groups or with batch sizes.
2. Whether per-unit dictionaries in B cost too much ratio on low-cardinality string columns, or
   whether a shared per-chunk dictionary segment is worth adding.
3. For A: which codecs in the current plan set can provide resume points cheaply.
4. Dataset regeneration and version gating of the reader.
