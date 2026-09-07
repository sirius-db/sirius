# Chunk-skipping metadata for Simpatico

**Status:** investigation complete, nothing implemented.
**Branch:** `feat/simpatico-zone-maps` — worktree `/home/nvidia/joost/sirius-zonemap`.
**Goal:** attach per-chunk metadata to Simpatico-compressed pinned data so a scan can skip
*fetching and decoding* chunks that cannot contain a match, instead of decoding everything and
filtering afterwards.

This file is the living record for the project. Keep the *Measurements* numbers and the
*Open questions / log* section current as work lands.

---

## 1. Where we start from

### 1.1 What already exists (verified in `dev` @ `5201d9c3`)

Three of the four pieces are already built. The project is mostly about connecting them.

**a) Simpatico already has a 1024-row chunk, and it is a real structural unit.**
`kChunkSize = 1024` (`src/compression/simpatico_codegen/include/codegen/jit/fused_tree.hpp:32`)
is the decode grid stride, the bitpack block, the selection-mask stride and the CSR bucket key.
`selection.hpp:41-47` states the identity explicitly: the selection chunk *is* the bitpack chunk.

**b) Selective decode already exists at exactly that granularity.** The decode renderer is a
product of an *enumerator* × a *consumer*
(`src/compression/simpatico_codegen/include/codegen/decode/jit/renderer.hpp:63-108`). The
`chunk_csr` enumerator launches **only touched chunks** — "empty chunks are absent rather than
launched-and-skipped" (`.../codegen/selection/chunk_row_set.hpp:22-23`). Entry points:
`decompress_column_rows` (sparse walk) and `decompress_column_compacted` (mask route),
`.../api/simpatico_codegen.hpp:261-308`. What is missing is any *producer* of "this chunk cannot
match" that does not first evaluate the predicate over the decoded values.

**c) Per-chunk min is already stored, for free, on every bitpacked column.** `Bitpack` emits
`{chunk_min, chunk_count, chunk_bits, packed}` (`src/compression/simpatico_codegen/src/plan/operator_registry.cpp:45`,
written at `.../src/encode/jit/renderer.cpp:983-988`, buffers allocated `:1053-1055`). `For` emits
`references[chunk_id]` = per-chunk minimum (`.../src/encode/jit/renderer.cpp:486,526-529`).
So for every bitpacked or FOR'd column we already have, device-resident, per 1024 rows:

- an **exact minimum** (`chunk_min[c]` / `references[c]`), and
- a **tight maximum bound**: `chunk_min[c] + (2^chunk_bits[c] − 1)`.

That is a zone map at zero storage cost and zero pin-time cost. Nothing reads it as one today.

**d) Coarse zone maps exist for *uncompressed* pins — and are switched off for the configuration
that wins.** `compute_pinned_chunk_stats` (`src/include/scan_manager/pinned_chunk_stats.hpp:43`)
computes per-pin-chunk min/max via `cudf::minmax`; `pinned_zone_maps` (`:57`) is the sidecar;
`build_cached_scan_plan` prunes with it (`src/scan_manager/sirius_scan_manager.cpp:2699-2783`,
`chunk_provably_empty`), gated by `enable_pinned_zone_map_pruning` (default true,
`src/include/sirius_config.hpp:181`).
**But the GPU-tier compressed pin path forces capture off** — `options.capture_chunk_stats = false`
at `src/pin_table.cpp:715`, and `device_pin_result` (`src/include/pin_table.hpp:258-270`) has no
field to carry stats. GPU-tier compressed is the configuration that wins the suite by 17.8–19.7%
(`docs/super-sirius/compressed-pinning.md:44-48`), so **the fastest compressed configuration
currently has no zone maps at all.**

### 1.2 The two chunk sizes, and why the gap matters

| Unit | Size | Set by |
|---|---|---|
| **Pin chunk** (= scan batch = one `simpatico::compressed_table`) | `scan_task_batch_size`, default `clamp(gpu_mem/40, 512 MiB, 5 GiB)`, packed as whole 122,880-row DuckDB row groups | `src/include/sirius_config.hpp:38-48`, `src/pin_table.cpp:126` |
| **Simpatico decode chunk** | **1024 rows, fixed** | `fused_tree.hpp:32` |

Existing pruning acts only on the pin chunk — millions of rows, all-or-nothing
(`DECODE_PUSHDOWN_PLAN.md:722-723`: "a chunk is all-or-nothing today"). All the interesting
skipping lives at 1024 rows, where the metadata already is.

### 1.3 What has been tried and did not work

`enable_dynamic_zone_map_filter` (default **false**, `src/include/sirius_config.hpp:163`) publishes
join-build `[min,max]` and applies it as a *row mask during decode*, never as a chunk skip. It was
measured at **+131 ms on GB300** and not kept. `DECODE_PUSHDOWN_PLAN.md:554-576` records the
diagnosis, which is the thesis of this project:

> The one exact test — does the published `[min, max]` exclude any of THIS chunk's values — needs
> **device-resident per-chunk metadata**, and reading it to decide whether to launch a kernel is
> its own cost.

`DECODE_PUSHDOWN_PLAN.md:689` lists `zone_map` as an unbuilt consumer candidate.

---

## 2. What is needed — overview

Five workstreams, roughly in dependency order. (1) and (2) are independently shippable.

### W1 — The in-payload metadata: useful, but not a foundation

`chunk_min` is a value-domain minimum **only when `bitpack` consumes the raw input**. Auditing the
shipped plans (`src/compression/simpatico_codegen/plans/tpch_sf1000/{lineitem,orders}.txt`):

| Plan root | Columns | `chunk_min` is a min of… | Usable as a zone map? |
|---|---|---|---|
| `input -> bitpack` | `l_{partkey,suppkey,linenumber,quantity,extendedprice,discount,tax,shipdate,commitdate,receiptdate}`, `o_{custkey,totalprice,orderdate,shippriority}` | the values | **yes, directly** |
| `dictionary -> bitpack(indices)` | `l_{returnflag,linestatus,shipinstruct}`, `o_{orderstatus,orderpriority,clerk}` | dictionary codes | **yes** — cuDF dictionary keys are "a sorted set of unique values" (`cudf/dictionary/dictionary_column_view.hpp:24`), so codes are order-preserving. Needs a per-pin-chunk lookup of the predicate constant into that chunk's key set |
| `delta -> bitpack` | `l_orderkey`, `o_orderkey` | the **deltas** | **no** |
| `str_split -> bitpack(offsets)` | `l_shipmode` | string **lengths** | **no** |
| `str_split -> delta -> ans` / `snappy` | `l_comment` | — | **no** |
| `identity` | `o_comment` | — | **no** |

So today the free metadata happens to cover most filterable columns. It is still the wrong thing to
build on, for two reasons:

1. **It is contingent on a plan chosen for ratio and throughput, not for prunability.** The
   compression explorer (`src/compression/simpatico_codegen/src/explore/compression_explorer.cpp`)
   picks max ratio subject to a decode-throughput floor. Nothing keeps `chunk_min` in the value
   domain.
2. **Clustering the data is likely to destroy it.** §3.3 says the whole scheme only pays once a
   column is clustered — and a clustered date column becomes monotone, which makes
   `delta -> bitpack` the ratio winner on exactly the column that was going to do the pruning.
   *Unverified prediction; cheap to check by re-running the explorer on sorted input, and worth
   doing early (Phase 0 below).*

**Therefore: min/max is stored out-of-band, unconditionally, for every column of a supported
type, independent of the plan.** The in-payload arrays remain valuable as a *second-level*
optimisation once the out-of-band index has already selected a group — the in-kernel early-out
below costs nothing and needs no new storage:

- **In-kernel early-out.** In the ballot emitters
  (`src/compression/simpatico_codegen/src/decode/jit/renderer.cpp:885+`, `emit_generic_mask_out`)
  the block already loads `chunk_min` and `chunk_bits` and already has `pred_lo`/`pred_hi` as
  kernel parameters. Add: if `[chunk_min, chunk_min + 2^bits − 1]` is disjoint from the predicate,
  write 32 zero mask words and return; if fully contained, write all-ones and return. Applies only
  to the `input -> bitpack` rows of the table above; sound to skip elsewhere.

### W2 — Plumb the existing sidecar into the GPU compressed pin path

Purely mechanical, unblocks pin-chunk pruning for the winning configuration:
`src/pin_table.cpp:715` (remove the forced-off), `src/include/pin_table.hpp:258-270` (add
`chunk_stats` to `device_pin_result`), `src/include/scan_manager/sirius_scan_manager.hpp:602`
(`insert_pinned_entry_device` takes them), `src/sirius_extension.cpp:1539` (pass `true`).
Note `docs/super-sirius/scan.md:283-288`: statless entries are **not** retrofittable, so this must
be captured at pin time.

### W3 — Out-of-band per-group min/max (the core of the project — see §4.3 for granularity)

A sidecar of exact per-group `{min, max, null_count}`, computed at pin time from the
*uncompressed* column before it enters the compressor — so it is exact, plan-independent, and
covers `delta`, `str_split`, `ans` and `identity` columns alike. `compute_pinned_chunk_stats`
(`src/include/scan_manager/pinned_chunk_stats.hpp:43`) already does exactly this with
`cudf::minmax`; the work is to run it per *group* rather than per pin chunk, widen its type
support, and give it a device-resident representation. Granularity and cost: §4.3.

Optionally also add an exact `chunk_max` channel next to `chunk_min` in the encoder (same
`BlockReduce` already running in `emit_bitpack`/`emit_for`, one `add_buffer` at
`.../src/encode/jit/renderer.cpp:1053`, one registry entry; the `.hpln` format is self-describing
per leaf so this is additive). Only worth it if the in-kernel early-out (W1) proves to be
selectivity-limited by the `2^bits` bound.

### W4 — Metadata for the operators that have none

- **Dictionary columns**: a per-1024-chunk bitmap of "which key ids occur" over the bitpacked
  `indices` channel. `bool8_filter_directive::equals_any`
  (`.../codegen/selection/selection.hpp:165-168`) is the ready-made predicate side. This is the
  only thing that helps low-cardinality string equality (`l_returnflag='R'`,
  `l_shipmode IN (…)`), where min/max is useless — see §3.4.
- **Per-chunk null counts**: no null model exists for the fused numeric ops, so the cheap route is
  a pin-time capture. This is also the single fact blocking compressed late-materialization
  (`src/include/late_mat/materialize.hpp:93-99`, `src/scan_manager/sirius_scan_manager.cpp:638-643`
  returns `nullopt` for compressed chunks; `docs/super-sirius/late-materialization.md:162-173`).
- **ALP-RD `right_bw`** is column-wide today; the cwida reference keeps it per 1024-value group
  (`.../codegen/plan/leaf_desc.hpp:21-28`).

### W5 — Skip decisions at the right sites

| Level | Site | Saves |
|---|---|---|
| Pin chunk, prepare | `build_cached_scan_plan`, `sirius_scan_manager.cpp:2762-2774` | task + H2D + decode. **Already implemented**; preserve the all-pruned sentinel at `:2777-2782` |
| Pin chunk, serve | `get_next_batch`, `sirius_scan_manager.cpp:241-256` | same, but can use post-prepare (join) filters |
| Pin chunk, decode | `prepare_for_processing`, `src/op/scan/sirius_gpu_scan_operator_data.cpp:118+` | H2D + decode; sees the fresh dynamic-filter snapshot (`:169-199`) |
| **1024-row chunk, in-kernel** | ballot emitters, `src/compression/simpatico_codegen/src/decode/jit/renderer.cpp:885+` (`emit_generic_mask_out`) | unpack + ballot. **Zero new metadata** |
| **1024-row chunk, sparse walk** | drop entries from `chunk_ids`/`block_offsets` before `decompress_column_rows` (`src/late_mat/materialize.cpp:219-224`) | whole blocks |

**Soundness rules to inherit** (all already established in the codebase):
missing stats never prune (`pinned_chunk_stats.hpp:36`); dropping conjuncts can only *retain* extra
chunks (`docs/super-sirius/scan.md:364`); an all-pruned scan must keep one chunk or the pipeline
never completes (`sirius_scan_manager.cpp:2777-2782`); parquet null semantics to mirror at
`scan.md:366`.

---

## 3. Measurement: how often can we actually prune?

All scripts in `/tmp/claude-1000/.../scratchpad` (`prune.py`, `sweep.py`, `synth.py`, `idxsize.py`);
re-derive with the recipe in §3.6.

### 3.1 Parquet row-group pruning on TPC-H SF1000, as the data sits today

Method: `EXPLAIN (FORMAT json)` each of the 22 queries in
`test/tpch_performance/tpch_queries/orig/` against views over `/datasets/tpch_sf1000` (SF1000, so
DuckDB's statistics-derived join filters carry the right constants — using SF1 views produces a
bogus `c_custkey<=149999` and a fake 99% prune on `customer`). Extract every scan-level filter,
attribute it to a table by column prefix, and evaluate it against all 106,000 row-group column
statistics read from the parquet footers.

> **0.00% of row groups are prunable. 0.0 GB of 851.5 GB.**

Every one of lineitem's 5,640 row groups has `l_shipdate` min/max = `1992-01-02 … 1998-12-01`. The
data is generated in `orderkey` order and every date column is effectively uniform-random within
each ~1M-row group, so no date, quantity or discount predicate excludes anything. String columns
(`l_returnflag`, `l_shipmode`, `l_shipinstruct`, …) *do* carry min/max statistics — verified,
0 of 94 row groups missing them — but a 1M-row group of a 3-to-7-value column contains every
value, so `[min, max]` spans the whole domain and excludes nothing either. There are **zero bloom
filters**, which is the only structural gap in the footers.

This is not a Sirius bug — Sirius's cuDF stats pruning
(`src/op/scan/parquet_gpu_ingestible.cpp:801-810`) is doing exactly the right thing and finding
nothing to do. It is a property of unclustered TPC-H.

### 3.2 The ceiling: how much volume is even *under* a prunable predicate

Same run, measuring which scans carry at least one min/max-evaluable predicate on the scanned table:

| Table | Scanned across 22 queries | Under a min/max-evaluable predicate |
|---|---|---|
| lineitem | 709.8 GB | 60.7% |
| orders | 82.3 GB | 51.9% |
| partsupp | 25.7 GB | 0.0% |
| customer | 23.8 GB | 1.9% |
| part | 8.6 GB | 92.4% |
| supplier | 1.3 GB | 0.0% |
| **total** | **851.5 GB** | **56.6%** |

So 56.6% of scan volume is *addressable*. Everything below is the question of how much of that
56.6% clustering lets us actually capture.

### 3.3 What clustering buys — SF10 lineitem, real predicates, real data

Method: materialize all filterable SF10 lineitem columns (60M rows), reorder under several sort
keys, compute per-chunk min/max at each granularity, and evaluate the actual TPC-H scan predicates
from §3.1. Fraction of **chunks pruned**:

| Layout | 8K rows/chunk | 64K | 256K | 1M |
|---|---|---|---|---|
| natural (as generated) | 0.0% | 0.0% | 0.0% | 0.0% |
| sort by `l_shipdate` | 60.1% | 60.1% | 59.8% | 59.2% |
| sort by `(l_returnflag, l_shipdate)` | 62.4% | 62.2% | 61.7% | 59.4% |
| sort by `(l_shipmode, l_shipdate)` | 73.0% | 72.2% | 69.1% | 57.7% |

Per-query at 64K/`sort:shipdate`: q14 98.6%, q15 96.1%, q6 84.7%, q20 84.7%, q12 83.5%, q7 69.5%,
q10 50.0%, q1 47.4%, q3 46.4%; q19 (`l_shipinstruct=`/`l_shipmode IN`) 0.0% under min/max.

Projecting the per-query chunk-prune fractions onto the real SF1000 per-query scan byte volumes:

| Layout, 64K chunks | lineitem bytes skipped | of all 851.5 GB scanned |
|---|---|---|
| sort `l_shipdate` | 288.1 / 709.8 GB = **40.6%** | **33.8%** |
| sort `(l_returnflag, l_shipdate)` | 294.2 GB = 41.4% | 34.6% |
| sort `(l_shipmode, l_shipdate)` | 320.9 GB = 45.2% | **37.7%** |

**Headline: 0% today, ~34–38% of all scanned bytes with one sort key.** The metadata is the cheap
part; the clustering is the expensive part.

### 3.4 Min/max is not enough — the dictionary case

q19 (`l_shipinstruct = 'DELIVER IN PERSON'`, `l_shipmode IN ('AIR','REG AIR')`) and q12's
`l_shipmode IN ('TRUCK','REG AIR')` prune **0%** under min/max at every granularity and every sort
key that does not sort on that column: a 1024-row chunk of a 7-value column contains all 7 values,
so `[min,max]` spans everything. A per-chunk **present-values bitmap** (7 bits for `l_shipmode`)
turns those into exact tests. In the sweep, the `minmax+bitset` mode is identical to `minmax`
*except* it makes those queries prunable whenever the column is a sort prefix — which is why
`sort:(shipmode, shipdate)` reaches 73% where `sort:shipdate` reaches 60%.

This is W4. Note the reason min/max fails here is *distributional*, not a missing-statistics
problem — the footers do carry string min/max (§3.1); they just span the whole domain. What
parquet lacks is the *shape* of metadata that can answer set membership. Parquet's own answer is
the bloom filter, which these files do not have and which DuckDB only consults as a fallback
(`parquet_reader.cpp:1243-1249`); a per-chunk present-values bitmap is both smaller and exact for
the cardinalities involved.

### 3.5 How this translates to Simpatico

- **The addressable volume is the same 56.6%** — the predicates come from the same
  `TableFilterSet`, already available at Sirius scan time (`src/planner/sirius_plan_get.cpp:142-197,458`,
  `src/include/op/scan/parquet_gpu_ingestible.hpp:73`).
- **The unit is 1024 rows instead of ~1M.** §3.3 shows that buys almost nothing on *sorted* data
  (60.1% at 8K vs 59.2% at 1M) — see §4 for why — but it is what makes skipping composable with the
  existing `chunk_csr` decode path, and it is where the free metadata already lives.
- **The saving is different in kind.** Parquet pruning saves file IO. Simpatico chunk skipping on a
  GPU-tier pin saves **decode SM time** — the payload is already resident. On the winning config the
  seven hottest lineitem columns are 93 GB compressed decoded at 500–1700 GB/s
  (`docs/super-sirius/compressed-pinning.md:56-60`); ~40% of that decode is what is on the table.
  On a **host**-tier pin it additionally saves the payload H2D fetch, which is the dominant cost
  there (`compressed-pinning.md:52-56`).
- **Simpatico can prune where parquet cannot**: string/dictionary columns (§3.4), and any table
  reordered at pin time regardless of how the parquet file is laid out.
- **Caveat to carry:** these are *opportunity* numbers (bytes not decoded), not measured query time.
  Nothing here has been run end-to-end. See the log.

### 3.7 Phase 0 result — measured on a clustered dataset (2026-09-07)

`/datasets/tpch_sf100_sorted` was built from `/datasets/tpch_sf100` with lineitem ordered by
`l_shipdate` and orders by `o_orderdate`, everything else symlinked
(`tools/chunk-skipping-study/mksorted.py`). Row-group spans went from **2,525 days to 4.4 days**
(lineitem) and 2,405 → 20.7 days (orders); the six lineitem files are non-overlapping in date.

*Methodology note 1:* DuckDB's `FILE_SIZE_BYTES` rotation writes files in parallel and does **not**
preserve a global `ORDER BY` across them — the first attempt produced overlapping files and only a
197-day average span. The dataset must be written to one file and then split by `LIMIT`/`OFFSET`.

*Methodology note 2 — writer provenance.* Every other tree under `/datasets/` was written by
**parquet-rs 57.3.1**; the clustered tree is written by **DuckDB 1.5.5**, and the two writers do not
agree on encoding:

| | parquet-rs | DuckDB |
|---|---|---|
| `l_orderkey`, `l_partkey`, `l_suppkey`, `l_extendedprice` | `DELTA_BINARY_PACKED` | `PLAIN` |
| `l_comment` | `DELTA_LENGTH_BYTE_ARRAY` | `PLAIN` |
| dates, decimals, low-card strings | `RLE_DICTIONARY`, several **uncompressed** | `PLAIN_DICTIONARY`, all snappy |
| `l_orderkey` type | `INT64` | `INT64` + `converted_type=INT_64` |

Decimals are INT64-backed on both, so the FLBA-decimal probe that disables pushdown
(`src/op/scan/parquet_gpu_ingestible.cpp:700-724`) does not fire either way. But the encodings
drive different cuDF decode paths, so **absolute scan times are not comparable across the two
trees.** Consequences:

- The **pruning** figures above are safe: they are computed from footer statistics, which both
  writers emit correctly, and the headline is row-weighted (row counts are identical at 12.69e9).
  The byte figures are not comparable, which is why they are reported separately.
- A sorted-vs-unsorted **timing** comparison is not safe against `/datasets/tpch_sf100`. For that,
  use `/datasets/tpch_sf100_duckdb_natural` — the same tables through the same DuckDB writer with
  the `ORDER BY` removed (`SORT=0 python mksorted.py`), so the only variable is row order.
- Phase 1's primary experiment (pruning on vs off on one dataset) is unaffected by any of this.

Row-group pruning, same 22 queries, same script, both datasets at SF100 (identical row counts, so
the row-weighted figure is a clean comparison; byte figures are not, because the two datasets were
written by different parquet writers with different encodings):

| | unsorted | clustered |
|---|---|---|
| rows scanned | 12.69e9 | 12.69e9 |
| **rows prunable** | **0.00%** | **36.45%** |
| bytes prunable | 0.00% | 33.85% |

Per-query rows pruned: q14 98%, q15 96%, q6 85%, q20 85%, q12 83%, q7 69%, q10 50%, q1 47%,
q3 46% on lineitem; q4 96%, q10 96%, q5 83%, q8 69%, q3 50%, q21 47% on orders. The orders
pruning was not predicted by §3.3 (which only modelled lineitem) and is a free addition.

This lands inside the 34–38% band predicted in §3.3 from the SF10 sweep, which validates the
projection method.

**The W1 delta-flip risk did not materialise — clustering *strengthens* the free metadata.**
Running the explorer (`simpatico explore`) on sorted vs unsorted SF100 columns:

| column | unsorted pick | ratio | clustered pick | ratio |
|---|---|---|---|---|
| `l_shipdate` | `input -> bitpack` | 2.651x | `input -> bitpack` | **426.5x** |
| `l_commitdate` | `input -> bitpack` | 2.651x | `input -> bitpack` | 3.992x |
| `l_receiptdate` | `input -> bitpack` | 2.651x | `input -> bitpack` | 6.311x |
| `l_quantity` | `input -> bitpack` | 4.885x | `input -> bitpack` | 4.885x |
| `l_discount` | `input -> bitpack` | 15.604x | `input -> bitpack` | 15.604x |
| `l_orderkey` | `input -> bitpack` | 6.032x | `input -> bitpack` | 2.174x |
| `l_returnflag` | `input -> str_split` | 1.000x | `dictionary -> bitpack` | 37.372x |

Every column keeps a bitpack root. Inside a 1024-row chunk of clustered data the values are nearly
identical, so `chunk_bits` collapses toward zero and plain bitpack already wins — `delta` buys
nothing. **The prediction in §2/W1 that clustering would flip the date columns to `delta -> bitpack`
and destroy `chunk_min` is refuted.** (Caveat: these are SF100 single-file columns; the shipped
SF1000 plan does pick `delta -> bitpack` for `l_orderkey`, so delta roots do occur in practice —
just not on the columns that do the pruning, and not as a consequence of clustering.)

Clustering also improves compression on the pruning columns (`l_shipdate` 2.651x → 426x), so the
pin gets smaller as well as more skippable. The net across all 16 columns is not yet measured —
`l_orderkey` got worse (6.03x → 2.17x) — and needs a full re-explore.

### 3.6 Reproducing

```bash
# 1. per-query scan filters, at SF1000 so join-derived constants are right
pixi run python explain.py      # EXPLAIN (FORMAT json) over views on /datasets/tpch_sf1000
# 2. footer stats for all 8 tables (~0.2 s total, metadata only)
pixi run python dumpstats.py
# 3. row-group prune evaluation + addressable-volume coverage
pixi run python prune.py
# 4. granularity x layout sweep on real SF10 lineitem
pixi run python mkcols.py && pixi run python gran.py && pixi run python sweep.py
# 5. synthetic clustering-window sweep
pixi run python synth.py
```

---

## 4. Granularity, and whether to compress the index

### 4.1 The governing rule

A synthetic sweep (16.7M sorted rows, then shuffled within a window `W` to control clustering
strength) gives the shape cleanly. Rows pruned, 10%-selective range predicate:

| clustering window `W` | 1K chunks | 4K | 16K | 64K | 256K | 1M |
|---|---|---|---|---|---|---|
| 1 (fully sorted) | 90.0% | 90.0% | 89.9% | 89.5% | 89.1% | 81.2% |
| 8K | 89.9% | 89.9% | 89.9% | 89.5% | 89.1% | 81.2% |
| 64K | 89.5% | 89.5% | 89.5% | 89.5% | 89.1% | 81.2% |
| 1M | 81.2% | 81.2% | 81.2% | 81.2% | 81.2% | 81.2% |
| random | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

> **Effective granularity = max(chunk size, clustering window).** Shrinking chunks below the
> clustering window buys exactly nothing. Above the window, pruning degrades roughly like
> `1 − chunk/window`. And no granularity rescues randomly-ordered data — the curve is flat at zero.

The real SF10 sweep (§3.3) agrees: flat from 1K to 64K, first real loss at 256K–1M.

### 4.2 Index cost

Per-column min+max+null-count, as a fraction of the column's own bytes (compressed = ÷2.84, the
measured lineitem ratio, `compressed-pinning.md:59`):

| chunk rows | 4-byte col: % of raw / % of compressed | 8-byte col: % of raw / % of compressed |
|---|---|---|
| 1,024 | 0.293% / **0.83%** | 0.244% / **0.69%** |
| 8,192 | 0.037% / 0.10% | 0.031% / 0.09% |
| 65,536 | 0.005% / 0.013% | 0.004% / 0.011% |
| 1,048,576 | 0.000% / 0.001% | 0.000% / 0.001% |

Whole-table sidecar for SF1000 lineitem (6.0e9 rows), against 93 GB of compressed pinned data:

| scope | 1K chunks | 8K | 64K | 1M |
|---|---|---|---|---|
| all 16 columns | 1.36 GB | 0.17 GB | 0.02 GB | 0.00 GB |
| 10 filterable columns | 0.84 GB | 0.11 GB | 0.01 GB | 0.00 GB |
| 3 date columns | 0.21 GB | 0.03 GB | 0.00 GB | 0.00 GB |

### 4.3 Recommendation: a configurable metadata group of G simpatico chunks, default **G = 8**

**Shape: one metadata entry per group of G consecutive 1024-row simpatico chunks.** Not an
independent stride, and not 1024.

Why a *bundle of chunks* rather than an independent stride: group `g` covers chunk ids
`[g·G, (g+1)·G)` exactly, so a surviving group expands into a chunk-id list for `chunk_csr` with a
shift, no modular arithmetic and no partial-chunk case. It also composes with the free in-payload
metadata (W1), which is defined at exactly 1024.

Why **G = 8 (8,192 rows)** as the default:

- **It costs nothing in pruning power.** On real SF10 lineitem under every sort key tested, 8,192
  is indistinguishable from 1,024 — mean 73.5% vs 73.5% (`sort:shipdate`), 73.0% vs 73.1%
  (`sort:(shipmode,shipdate)`). §4.1 explains why: effective granularity is
  `max(group, clustering window)`, and any data with a clustering window under ~8K rows is close
  enough to random that pruning is ~0 at every granularity.
- **It costs 8× less than G=1.** SF1000 lineitem, all 16 columns: **0.17 GB at G=8 vs 1.36 GB at
  G=1**, against a 93 GB compressed pin — 0.18% versus 1.5%.
- **It tiles the pin chunk exactly.** DuckDB-native pin chunks are whole 122,880-row row groups
  (`src/pin_table.cpp:126`) and 122,880 = 120 × 1024, so the well-behaved G are the divisors of
  120: **8**, 10, 12, 15, 20, 24, 30, 40, 60. G = 8 gives 15 groups per row group. (G = 16 does
  not divide 120 — avoid it despite being the obvious power of two.) The format must tolerate a
  short final group regardless, since the parquet path packs differently.
- **Going coarser buys nothing.** At G=8 the index is already 0.18% of the pin, so there is no
  space left to win, and 262K/1M strides start to cost real selectivity on multi-key sorts
  (73.0% → 69.1% → 57.7% for `sort:(shipmode,shipdate)`).

Why **configurable**: the right G tracks the data's clustering window, which is not knowable at
build time. G is a pure space-vs-selectivity knob with no correctness consequence — coarsening
never produces a wrong answer, only fewer skips. Expose it as a `SET` option alongside
`enable_pinned_zone_map_pruning`.

**Scope the sidecar to columns of a supported type, all of them.** Restricting to "filterable"
columns is a 1.5–4× further saving but needs a filter workload to know; at 0.18% it is not worth
the coupling. Start with all supported types, revisit only if a wide table makes it hurt.

### 4.3.0 What the index costs to *build* (measured)

Storage (§4.2) was never the worry; compute might have been. A G=8 index over a 189 M-row pin chunk
needs 23,072 min/max pairs per column instead of one. Measured with `cudf::segmented_reduce` over a
fixed-stride offsets column, all 16 lineitem columns, GB300
(`tools/chunk-skipping-study/gpusort/statsbench.cu`):

| G | stride | groups per 189 M-row chunk | segmented min+max | vs the whole-chunk `minmax` W2 already pays |
|---|---|---|---|---|
| 1 | 1,024 | 184,571 | 0.0209 s | 5.8× |
| **8** | **8,192** | **23,072** | **0.0061 s** | **1.7×** |
| 64 | 65,536 | 2,884 | 0.0061 s | 1.7× |

(whole-chunk `cudf::minmax` × 16 columns = 0.0036 s; segmented at G≥8 runs at 2,730 GB/s, i.e.
bandwidth-bound.)

**Building the entire G=8 index for SF1000 lineitem costs 32 × 0.0061 s ≈ 0.20 s** against a
~151 s pin — 0.13%. Negligible, and only 1.7× what the existing coarse capture already costs.

This is a third independent argument for G=8, and the three converge: G=1 buys no extra pruning
(§4.1), costs 8× the storage (§4.2), and costs **3.4× the compute**. Going coarser than 8 buys
nothing either — G=64 is the same 0.0061 s, because the reduction is already bandwidth-bound.

### 4.3.1 Why pin-chunk granularity is not enough

The existing sidecar is per *pin chunk*, and at the configured `scan_task_batch_size: 8GB`
(`bench/sf1000-repro/sirius-sf1000.yaml:47`) a pin chunk is a large fraction of most SF1000 tables:

| table | rows | pin-chunk rows @8 GB | % of table | chunks in the whole table |
|---|---|---|---|---|
| lineitem | 6.0e9 | 189 M | 3.1% | ~32 |
| orders | 1.5e9 | 109 M | 7.3% | ~14 |
| partsupp | 800 M | 64 M | 8.1% | ~12 |
| customer | 150 M | 63 M | **42%** | ~2 |
| part | 200 M | 161 M | **81%** | ~1 |
| supplier | 10 M | 10 M | **100%** | 1 |

(Rows are computed from raw bytes/row; a narrow pinned subset has fewer bytes/row and so *more*
rows per chunk, making this worse.)

Pin-chunk pruning is therefore only meaningful for lineitem and marginally orders — everything
else has too few chunks for any prune rate to exist. On perfectly sorted lineitem the fine index
buys only ~1–2.5 points over pin-chunk granularity (mean 73.5% at 8K vs 70.9–72.4% at 2–3.5% of
table); on the other five tables it is the difference between a working mechanism and none.

### 4.4 Should the index be compressed?

**Numeric min/max arrays: no. Present-value bitmaps: no. Nothing: no.**

Delta + FOR + bitpack over the per-chunk min arrays compresses well — measured ~9–11% of raw at
1K–8K chunks on real SF10 lineitem (chunk minima of a clustered column are monotone, so deltas are
tiny). That would take the 8K/16-column sidecar from 0.17 GB to ~0.016 GB.

It is still the wrong trade:

1. **The absolute size is already negligible** at the recommended stride — 0.18% of the pin. Saving
   90% of 0.18% is not worth a decode step.
2. **The index is on the critical path of the skip decision.** The whole point is to answer "can I
   skip this chunk" *before* launching work. A compressed index has to be decompressed first, which
   is precisely the cost `DECODE_PUSHDOWN_PLAN.md:554-576` blames for the dynamic-zone-map
   regression. An uncompressed index is a coalesced strided read of a few MB that a pruning kernel
   can scan at bandwidth.
3. **Random access matters.** Pruning at serve time and during late-mat needs `cell(col, chunk)`
   for scattered chunks. A bitpacked index makes that a per-access unpack.

Compression becomes worth revisiting only if we ever want 1024-row *new* metadata on wide tables
(the 1.36 GB row), or for the disk-spilled copy — see §5.4.

---

## 5. Clustering: what it costs, and how much of it is enough

§3.1 and §3.7 together say the metadata is the cheap part and the clustering is the whole game
(0.00% → 36.45%). So the cost of clustering is the project's real budget question. The premise
here is that it runs as a **preprocessing step at pin time / table registration**, not as an
offline dataset rewrite.

### 5.1 Measured cost of sorting

Two harnesses, both in `tools/chunk-skipping-study/`: `cpusort.py` (DuckDB, in-memory table, 72
threads) and `gpusort/sortbench.cu` (a standalone libcudf `sorted_order` + `gather` on a
lineitem-shaped table; build line in the file header). GB300, 256 GB HBM, 494 GB host.

| method | measured | extrapolated to SF1000 lineitem (6.0e9 rows, 16 cols, 88 B/row) |
|---|---|---|
| **CPU** DuckDB in-memory global sort, 72 threads | 600 M rows in **7.6 s** = 79 Mrows/s | **76 s** — and the 600 GB working set exceeds the 494 GB host, so it would spill |
| **CPU** DuckDB sort + parquet rewrite (what Phase 0 did) | SF100 lineitem: 44 s to one file, **393 s** including the 6-way split | ~65 min. Not viable as preprocessing |
| **GPU** `sorted_order` + `gather`, one 189 M-row pin chunk | **0.071 s** (0.004 s order + 0.067 s gather) = 2,659 Mrows/s, 234 GB/s | — |
| **GPU** local sort of all 32 pin chunks | — | **≈ 2.3 s** |
| **GPU** global sort | does not fit: 528 GB in+out vs 256 GB HBM | needs partition + per-partition sort, ≈ 5 s of GPU work **plus a full-table shuffle** |

**The GPU is ~34× faster than 72 CPU threads** (2,659 vs 79 Mrows/s) and the answer to "would the
GPU help" is unambiguously yes.

Two structural facts fall out of the split timings:

- **The cost is the gather, not the comparison.** At 800 M rows, `sorted_order` is 0.016 s and the
  gather is 0.304 s. Sorting is bandwidth-bound on the payload, so its cost scales with *pinned
  bytes*, not with key cardinality or sortedness.
- **Which means clustering during pinning is close to free**, because the pin already moves those
  bytes. In principle the gather can be fused into the existing materialize→compress path rather
  than run as a separate pass.

**Budget anchor.** The SF1000 hot suite is ~5.8 s of query time, but the whole-process wall is
~151 s, dominated by pinning. A 2.3 s GPU local sort is **~1.5% of the existing pin cost**; even a
two-pass global sort is ~3%. Against a 36% reduction in scanned rows, that is not a close call —
*provided* the skip mechanism can actually exploit the clustering the cheap strategy produces,
which is the subject of §5.2.

### 5.2 Full sort vs weaker clustering — the tradeoff

You do not need a global sort. Zone maps only care about each group's `[min, max]`, not about
order within the group, so there are strictly cheaper strategies. Measured on SF10 lineitem with
the real TPC-H scan predicates (`tools/chunk-skipping-study/cluster_modes.py`), pin chunk sized to
SF1000's 3.1% of table; mean % of chunks pruned across q1/q3/q6/q7/q10/q12/q14/q15/q20:

| clustering strategy | cost | G=8 (8,192 rows) | G=64 (65,536) | pin chunk |
|---|---|---|---|---|
| 0. none (as generated) | — | 0.0% | 0.0% | 0.0% |
| **1. local sort inside each pin chunk** | one GPU sort per batch, **no shuffle**, streaming | **72.7%** | 67.6% | **0.0%** |
| 2. range-partition to pin chunks, unsorted within | full-table shuffle, **no comparison sort** | 71.2% | 71.1% | 69.0% |
| 3. both (= global sort) | shuffle + sort | 73.5% | 73.4% | 71.7% |

The two rows that matter:

- **Local sort is the cheapest possible strategy and gets within 0.8 points of a global sort — but
  only if the index is fine-grained.** It leaves every pin chunk spanning the full key range, so at
  pin-chunk granularity it prunes **exactly nothing**. It needs no shuffle, fits on one GPU, is
  embarrassingly parallel across batches, and drops straight into the streaming pin path.
- **Range-partitioning is the strategy that works with today's coarse sidecar** (69.0% at pin-chunk
  granularity) but requires a full-table shuffle, so it cannot be done streaming — it needs the
  whole table before any chunk is final.

> **This is the strongest argument in the project for the fine index (W3): it is what makes the
> cheapest clustering strategy viable.** Local sort + a G=8 index ≈ a global sort, with no shuffle
> and ~2.3 s of GPU time at SF1000.

Note also that G=64 costs 5 points under local sort (72.7% → 67.6%) while costing nothing under the
other two. Fine granularity matters *more* the weaker the clustering — consistent with §4.1's
`max(group, clustering window)` rule, where local sort makes the effective window the group size
itself.

### 5.3 Where it would go, and what it breaks

- **(a) Local sort per pin batch** — insert a `cudf::sort_by_key` in `materialize_pin_batches`
  (`src/pin_table.cpp`) before compression. Streaming, no extra full-table memory, ~2.3 s at
  SF1000. Pays only with the fine index.
- **(b) Range-partition at registration** — two passes over the table, works with the existing
  coarse sidecar, but cannot be folded into the streaming pin.

Open in both cases: **choosing the sort key.** Needs a workload hint, a `SET` option, or a
heuristic; out of scope for Phase 0/1.

Reordering rows is not free of consequences — flag before implementing:

1. Late-materialization row addressing is positional within a batch
   (`src/include/late_mat/column_origin.hpp:26-44`); sorting must happen *before* those ids are
   handed out, not after.
2. MVCC deleted-row keep-masks are positional against the pinned order
   (`src/scan_manager/sirius_scan_manager.cpp:312-318` already special-cases them).
3. Any consumer relying on scan order matching file order. There should be none, but it is worth
   an explicit check.

## 6. Where the index should live: device, host, spilled

### 6.1 Keep the index in device memory even when the data is pinned to host — yes

This is the highest-leverage placement decision, and the asymmetry is stark. Take SF1000 lineitem,
8K stride, 10 filterable columns: **0.11 GB of index against 264 GB of raw / 93 GB of compressed
data** — a ratio of roughly **1 : 850**. Reasons to hold the index on device unconditionally:

- **It is what makes a host pin cheap to skip.** On a host-tier pin the dominant cost is the payload
  H2D fetch on the task's critical path (`compressed-pinning.md:52-56`). A device-resident index
  lets us decide *not to issue the fetch*. A host-resident index would need its own H2D round trip
  to make that decision — reintroducing exactly the latency we are trying to avoid, and the
  measured failure mode of `enable_dynamic_zone_map_filter`.
- **The decision is a GPU decision.** Survivor-chunk lists feed `chunk_csr` and the ballot kernels;
  producing them on device avoids a host sync. `build_chunk_row_set` already costs one host sync
  (`chunk_row_set.hpp` (`build_chunk_row_set`)); we should not add another per chunk.
- **The cost is bounded and predictable.** Index bytes scale with `rows / stride`, independent of
  column width and of compression ratio, so a device-memory budget for indexes is a simple,
  stable number the pin planner can reserve up front.

**Concretely:** a `pinned_zone_maps`-shaped sidecar should gain a device mirror
(`rmm::device_buffer` of the packed min/max/nullcount arrays, column-major, positional with
`cache_info.column_ids`) that is populated for *both* tiers, while `pinned_entry` keeps the host
`BaseStatistics` form for the existing DuckDB `CheckStatistics` path.

**Where it stops being free:** at 1024 stride over all columns of every pinned table the index is
~1.5% of the pinned footprint, which on a GPU that is already the binding constraint is a real
cost. The §4.3 recommendation (8K stride, filterable columns) keeps device residency
unconditionally affordable; if a configuration ever exceeds a budget, drop the *stride* (coarsen)
before dropping *residency*, because coarsening costs ~0.5 pp of pruning while evicting the index
to host costs the entire skip mechanism.

### 6.2 When data spills to host

Nothing changes for the index: it stays on device. The data moving to host makes the index *more*
valuable, not less, because the cost of a wrong "do not skip" decision goes up by the H2D transfer.
The downgrade executor (`src/downgrade/`, `src/creator/`,
`docs/super-sirius/memory-management.md`) should treat the index as **non-spillable** at its natural
size — it is in the same class as plan metadata, not payload.

Implication for the spill policy: when a table is demoted GPU→host, its index must survive the
demotion. Since `docs/super-sirius/scan.md:283-288` says statless entries are not retrofittable,
losing the index on demotion would be permanent for that entry.

### 6.3 When data spills to disk

Here the index should be **kept in device memory and additionally persisted**. Two distinct roles:

- **Resident copy (device):** answers "do I need to read this chunk back at all". This is the whole
  value — a disk round trip is orders of magnitude more expensive than the ~1:850 index, so the
  break-even is not close.
- **Persisted copy (with the payload):** the `.hpln` format is self-describing per leaf
  (`.../api/compressed_table_io.hpp:14-39`) and `payload_offset` is already per-buffer, so index
  buffers serialize alongside the data with no format work. For W1 metadata this is automatic —
  `chunk_min`/`chunk_bits` are already part of the serialized payload.

The persisted copy is also what enables **partial re-read**: `read_compressed_table_subset_from_memory`
(`compressed_table_io.hpp:137-143`) already does column-granular partial fetch via `payload_offset`;
adding chunk-range granularity would let a spilled table read back only surviving chunks. That is
the natural follow-on and the one place `range_slice` (`DECODE_PUSHDOWN_PLAN.md:722-723`) becomes
necessary.

### 6.4 Summary table

| Data location | Index location | Compressed? | Why |
|---|---|---|---|
| GPU pin (raw or compressed) | device | no | decision is on-device, cost ~0.1% of payload |
| Host pin | **device** | no | lets us skip the H2D fetch itself — the dominant host-tier cost |
| Spilled to host | **device**, non-spillable | no | wrong decisions now cost a transfer; not retrofittable if lost |
| Spilled to disk | device + persisted with payload | persisted copy: optional | disk round trip dwarfs the index; persisted copy enables partial re-read |

---

## 7. Proposed order of work — prove value before committing

The governing constraint: **on TPC-H as it sits, every implementation measures exactly zero**
(§3.1). Any proof of value must first create clustered data. And the cheapest honest proof does
*not* require building the out-of-band index at all.

### Phase 0 — data prep and two cheap facts (no C++)

1. **Build a clustered SF100.** `COPY (SELECT * FROM lineitem ORDER BY l_shipdate) TO …` and the
   same for orders on `o_orderdate`, into `/datasets/tpch_sf100_sorted`. Point
   `tools/chunk-skipping-study/explain.py` and `prune.py` at it and confirm parquet row-group
   pruning goes from 0% to the predicted ~40–60%. This validates both the dataset and the
   §3 methodology, and it is the artifact every later phase needs.
2. **Re-run the compression explorer on the sorted columns.** Answers the W1 delta-flip question:
   does a monotone `l_shipdate` make `delta -> bitpack` the ratio winner and remove the free
   `chunk_min`? Either answer is useful — it either confirms the out-of-band store is mandatory or
   removes a stated risk.

**Status: done, 2026-09-07.** Results in §3.7 (0.00% → 36.45% of rows) and §5 (sort costs and the
clustering-strategy tradeoff). The delta-flip risk was refuted. Actual cost: about half a day, no
build required.

### Phase 1 — DONE (2026-09-07). W2 landed; the sweep answers the granularity question.

**W2 implemented (`f9ca10ad`).** The four plumbing sites landed exactly as scoped —
`device_pin_result::chunk_stats`, `materialize_all_batches_compressed` no longer forcing capture
off, `insert_pinned_entry_device` building the sidecar, the extension passing
`capture_chunk_stats` through. 189 unit tests pass, and the full suite is **22/22 byte-exact**
against DuckDB CPU with pruning on.

**Result of the batch-size sweep** (`bench/chunk-skipping/run-sweep.sh`, clustered SF100, all eight
tables pinned GPU-tier compressed, best-of-3, sum of per-query bests):

| `scan_task_batch_size` | chunks pruned | suite ON | suite OFF | ON − OFF |
|---|---|---|---|---|
| 8 GB | 25% | 0.9551 s | 0.9544 s | +0.1% (noise) |
| 2 GB | 48% | **0.9348 s** | 0.9493 s | **−1.5%** |
| 512 MB | 66% | 1.0292 s | 1.0720 s | **−4.0%** |
| 128 MB | 71% | 1.5801 s | 1.8085 s | **−12.6%** |

This is the third of the three outcomes anticipated below: **the delta grows monotonically as
granularity refines.** The prune rate climbs 25 → 48 → 66 → 71%, approaching the 73.5% ceiling
§3.3 predicted for this data and these predicates.

**Run-to-run variance (3 independent processes, 8 GB and 2 GB arms).** Suite totals spread ~0.8%
process-to-process, so single-arm absolutes are worth about one digit. The deltas are stable:

| arm | run 1 | run 2 | run 3 | mean |
|---|---|---|---|---|
| 8 GB, ON − OFF | +0.1% | −0.0% | +0.4% | **+0.2% (noise)** |
| 2 GB, ON − OFF | −1.5% | −1.5% | −2.4% | **−1.8%** |
| 2 GB-ON vs 8 GB-OFF (absolute) | 0.9348 / 0.9544 | 0.9332 / 0.9484 | 0.9294 / 0.9465 | **−1.8%** |

The 2 GB delta is outside the noise floor in all three runs; the 8 GB delta is not. The 512 MB and
128 MB arms were run once each.

**A free, actionable finding: the optimal batch size shifts once pruning exists.** `scan_task_batch_size`
was tuned to 8 GB on *unclustered* data with no pruning, where bigger was strictly better
(`bench/sf1000-repro/sirius-sf1000.yaml:10` records 5 GB → 8 GB as −1.85%). With W2 on clustered
data the trade reverses at the margin: **2 GB-ON beats 8 GB-OFF by 1.8%**, reproducibly, with no
code beyond W2. Worth re-testing at SF1000, where 8 GB also peaks at 253.9 GB of 256 GB HBM.

**Mapping SF100 batch sizes onto SF1000.** What governs pruning is the chunk's *fraction of the
table*, not its byte size (§4.1). SF1000 lineitem at the production 8 GB batch is 189 M of 6.0e9
rows = **3.1%**; SF100 lineitem at 512 MB is 1/25…1/57 = **1.8–4.0%**. So:

> **SF100 @ 512 MB is the SF1000 @ 8 GB proxy, and it says W2 alone is worth ~4% of suite time**
> on clustered data — from a change that is pure plumbing and adds no new metadata.

**The sweep also proves you cannot buy granularity with batch size.** 128 MB prunes the most (71%)
yet is the *slowest* arm in absolute terms (1.58 s vs 0.955 s at 8 GB) because batching overhead
grows faster than the pruning saves. The best whole configuration is 2 GB-ON at 0.9348 s. Shrinking
batches to chase granularity is a losing trade — **granularity has to be bought with an index,
which is exactly what Phase 2 is.**

**What Phase 2 is now worth, stated honestly.** On SF1000 lineitem the group index adds only about
7 points of prune rate over the coarse sidecar (66% → ~73%), because a 3.1%-of-table chunk is
already fine enough for these predicates. Its value is elsewhere, and both parts are already
measured:

1. **The other tables.** §4.3.1: at 8 GB a pin chunk is 42% of customer and 81% of part. Coarse
   pruning is not weak there, it is structurally impossible — 1 or 2 chunks exist.
2. **It is what makes cheap clustering viable.** §5.2: a local sort inside each pin chunk — no
   shuffle, streaming, ~2.3 s at SF1000 — yields 72.7% at G=8 and **0.0%** at pin-chunk
   granularity. Without the fine index the only clustering strategy that pays is the one that
   needs a full-table shuffle.

*Caveat carried forward:* the −12.6% at 128 MB is measured against an inflated baseline (that arm's
own batching overhead), so it is **not** a prediction of the group index's value. The defensible
numbers are the ~4% for W2 at SF1000-equivalent granularity and the prune-rate curve.

*Process note:* the first run of this sweep was silently invalid — a `$( ... && ... )` inlined into
a `sed` replacement, with `&` escaped for sed, broke the shell's AND operator so **both** arms got
`enable_pinned_zone_map_pruning: false`. It reported a flat ±0.5% and would have been read as "the
feature does not help". The script now computes the flag before the `sed`, verifies the injection,
and fails the run if a prune-on arm logs no pruning.

### Phase 2 — NEXT: the out-of-band group index (W3)

Phase 1 says build it, for the two reasons above rather than for lineitem prune rate. §4.3.0 says
it is nearly free to build (1.7× the capture W2 already pays, ~0.20 s for all of SF1000 lineitem).

**The hard part is not the statistics, it is serving a partial chunk.** Today
`cached_scan_plan::survivor_chunk_indices` selects whole pin chunks, and
`DECODE_PUSHDOWN_PLAN.md:722-723` records that "a chunk is all-or-nothing today". Three pieces, in
dependency order:

**2a. Capture (self-contained, testable in isolation).**
`compute_pinned_group_stats(chunk, column_types, group_rows, …)` alongside the existing per-chunk
function, using `cudf::segmented_reduce` over a fixed-stride offsets column (the exact shape
benchmarked in §4.3.0). Extend `pinned_zone_maps` from `cell(pos, chunk)` to
`cell(pos, chunk, group)` plus a `group_rows` field, keeping the existing all-or-nothing
normalization and merge-degradation invariants. `group_rows = G * 1024` with G configurable,
default 8; a short final group per chunk must be legal (the parquet pin path does not pack whole
122,880-row units).

**2b. Plan (mechanical).** `build_cached_scan_plan` gains a per-surviving-chunk list of surviving
*group* ids. A chunk with every group pruned drops out exactly as today; a chunk with every group
surviving carries no list (the common case — keep it free). Preserve the all-pruned sentinel
(`sirius_scan_manager.cpp:2777-2782`).

**2c. Serve (the real work), by chunk kind:**

| chunk kind | route | notes |
|---|---|---|
| uncompressed GPU | `cudf::slice` the surviving group ranges + concatenate | no simpatico involvement; the easy case, good for landing 2a/2b end-to-end first |
| compressed, `input -> bitpack` root | `simpatico::decompress_column_rows` with a `chunk_row_set` built from the surviving groups | the path already exists and already skips untouched 1024-row chunks (`chunk_row_set.hpp:22-23`). Group g maps to chunk ids `[g·G, (g+1)·G)` by construction — a shift, no modular arithmetic |
| compressed, `dictionary`/`str_split`/`delta` root | full decode + gather | `decompress_column_rows` refuses these (nullptr + `error_out`, `simpatico_codegen.hpp:261-308`). Per the shipped plans that is `l_orderkey`, `l_shipmode`, `l_comment`, and the dictionary strings — so a mixed table skips what it can and full-decodes the rest |

**Suggested landing order:** 2a with unit tests → 2b → 2c for uncompressed chunks (proves the whole
path end to end against the existing zone-map tests) → 2c for bitpack-rooted compressed columns
(where the suite time actually is).

**Measurement that closes it:** re-run `bench/chunk-skipping/run-sweep.sh` at the *production* 8 GB
batch with the group index on. The target is the 128 MB arm's 71% prune rate at the 8 GB arm's
batching cost — which the Phase 1 table shows no batch size can deliver.

### Phase 3+ — in priority order, each independently justifiable

1. W1 in-kernel early-out — free, no storage, applies to the `input -> bitpack` columns.
2. W4 dictionary present-value bitmaps — the only path to pruning on `l_shipmode`-style
   predicates (§3.4), where min/max spans the whole domain regardless of granularity.
3. W4 per-group null counts — unblocks compressed late-materialization, a separate win.
4. **Clustering as a first-class pin option.** None of the above pays on TPC-H as it sits (§3.1);
   this is what converts 0% into ~34–38%. Arguably the real project, with the metadata as what
   makes it usable.

## 8. Open questions / log

- **2026-09-07** — project opened. Investigation and all measurements in §3–§6 done; nothing
  implemented, nothing benchmarked end-to-end.
- **2026-09-07** — plan audit corrected the W1 story: `chunk_min` is a value-domain minimum only
  for `input -> bitpack` roots. `delta -> bitpack` (`l_orderkey`, `o_orderkey`) and
  `str_split -> bitpack(offsets)` (`l_shipmode`) give minima of deltas and of string lengths
  respectively. `dictionary -> bitpack` *is* usable (cuDF dictionary keys are sorted). Min/max is
  therefore stored **out-of-band**, unconditionally. Granularity settled at a configurable group of
  G simpatico chunks, default G = 8 (8,192 rows); see §4.3.
- **2026-09-07** — **Phase 1 done.** W2 landed (`f9ca10ad`), 22/22 byte-exact. The batch-size sweep
  on clustered SF100 gives prune rates 25/48/66/71% at 8GB/2GB/512MB/128MB with suite deltas
  +0.1/−1.5/−4.0/−12.6%. SF100@512MB is the SF1000@8GB proxy (both ~3% of table per chunk), so
  **W2 alone is worth ~4%**. Shrinking batches to chase granularity loses on net — granularity must
  come from an index. Phase 2 is justified by the small tables (§4.3.1) and by enabling local-sort
  clustering (§5.2), not by lineitem prune rate.
- **2026-09-07** — **Phase 0 done.** Built `/datasets/tpch_sf100_sorted`; row-group spans 2525 → 4.4
  days; pruning 0.00% → **36.45% of rows**, inside the predicted band. The explorer keeps a bitpack
  root on every clustered column (`l_shipdate` ratio 2.651x → 426x), refuting the delta-flip risk.
  Added §5: GPU sort is ~34× faster than 72 CPU threads and a per-pin-chunk local sort of SF1000
  lineitem costs ~2.3 s; local sort reaches 72.7% of the global sort's 73.5% at G=8 but **0.0%** at
  pin-chunk granularity, which is the strongest argument yet for the fine index.
- **Open:** the §3/§5 pruning numbers are still *rows/bytes not decoded*, not query time. Need one end-to-end datapoint —
  cheapest is W2 on a GPU-compressed lineitem pin with an artificially clustered SF100.
- **Answered (§5.1):** sorting 6e9 rows at pin time costs ~2.3 s on the GPU for a per-chunk local
  sort — ~1.5% of the existing ~151 s pin. **Open:** where exactly the sort goes in
  `materialize_pin_batches`, and how the sort key gets chosen.
- **Open:** does clustering help or hurt the *net* compression ratio across all 16 columns?
  `l_shipdate` improves 2.651x → 426x but `l_orderkey` degrades 6.03x → 2.17x; needs a full
  re-explore and a re-picked plan file.
- **Open:** how much of the suite's time is scan/decode on the winning GPU-compressed config?
  Without that, §3's 40% "of scanned bytes" cannot be converted into an expected speedup.
  `docs/super-sirius/compressed-pinning.md:93` has the closest existing measurement.
- **Open:** the `2^chunk_bits − 1` max bound over-estimates the true range by up to 2×. Unmeasured
  how much selectivity that costs versus an exact `chunk_max` (W3).
- **Open:** interaction with the existing decompression-pushdown path
  (`src/compression/compressed_scan.{hpp,cpp}`, `pushdown_request`). Chunk skipping and
  filter-during-decode overlap; the skip should run first and narrow what pushdown sees, but the
  bookkeeping (`for_chunk`, `sirius_scan_manager.cpp:312-318`) needs checking.
- **Open:** benchmarks must run under `/home/nvidia/joost/bench-lock.sh` with
  `DATA=/datasets/tpch_sf1000`.
