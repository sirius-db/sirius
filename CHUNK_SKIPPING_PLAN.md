# Chunk-skipping metadata for Simpatico

**Status:** investigation complete, nothing implemented.
**Branch:** `feat/simpatico-zone-maps` — worktree `/home/nvidia/joost/sirius-zonemap`.
**Goal:** attach per-chunk metadata to Simpatico-compressed pinned data so a scan can skip
*fetching and decoding* chunks that cannot contain a match, instead of decoding everything and
filtering afterwards.

This file is the living record for the project. Keep the *Measurements* numbers and the
*Open questions / log* section current as work lands.

---

## 0. Where this stands (2026-09-09)

Read this first; the rest of the document is the reasoning and the measurements behind it.

### Shipped and measured

| | |
|---|---|
| **Best host-tier configuration** | 10.44 s → **9.48 s** at SF1000 (~−9.2%), 22/22 byte-exact |
| whole-chunk pruning on compressed pins | **−8.1%** (host, 2 GB batches) |
| the per-group index on top | **−2.20%** (44.6e9 further rows dropped) |
| **the whole chain on UNSORTED SF1000** | 9.6544 s → **8.7972 s, −8.88%**, for +1.6% pin cost (§3.13) |
| — of which pin-time clustering | −2.2% on its own (§5.4) |
| — of which the per-group index | +0.2% on its own; it locates rows, it does not save work |
| — of which range-skipped fetch | **−7.0%**, and only once the other two exist (§3.13) |
| without clustering, the same machinery | **+1.39%** — it prunes 0.00%, so it is pure overhead |
| GPU-tier pins | −0.4%, and the ceiling there is ~2.4% — the suite is join-bound |

Two findings that stand on their own, independent of the remaining work:

1. **Zone-map capture was off for GPU-tier compressed pins.** The path that wins the suite was the
   one configuration with no zone maps at all. Fixed; it is what makes every number above possible.
2. **Host-tier pins should use a 2 GB batch, not the tuned 8 GB.** Reproducible across runs. The
   8 GB value was tuned on unclustered data with no pruning, and the trade reverses at the margin.

### Working end to end

Per-group min/max capture at pin time → contiguous column-major arena → filter lowered once and
evaluated over packed bounds (24× faster than the `BaseStatistics` path) → surviving row ranges in
the scan plan → the scan serves those ranges: an uncompressed chunk by narrowing its column views
(zero-copy `cudf::slice`), a **host-tier compressed chunk by fetching and decoding only the
surviving 1024-row decode chunks** (`SIRIUS_EXP_CHUNK_SUBSET_FETCH`, on by default; `0` is the A/B
arm and the kill switch). 22/22 byte-exact on clustered SF100, host and GPU tier.

### What the fetch skip is worth today, and what caps it

**−1.33%** at SF1000 / host / 2 GB (9.6680 → 9.5395 s, best of two independent A/B pairs) against
the same build with the gate off. The scan-bound queries are what move: q3 −0.050, q1 −0.041,
q7 −0.035, q20 −0.032, q6 −0.015 (−22% of q6 itself). Each pair on its own reads −0.68% / −0.43%,
so the suite number sits at the noise floor while the per-query pattern is consistent.

**Dictionary-encoded strings are addressable (§6.5.6); `str_split` ones are not.** After the
dictionary work, the only column still forcing a whole-chunk serve in TPC-H is `l_shipmode`
(`input -> str_split`), which blocks q12. Extending this to `str_split` means rewriting offset
VALUES rather than gathering byte ranges — see §6.5.6 for why that is a different kind of problem.

### Known gaps, in the order they will bite

- **`str_split` string columns block the fetch skip** for any query that reads one (`l_shipmode`,
  `l_comment`). Dictionary-encoded ones no longer do.
- Only a **plain bitpack** plan is exercised end to end through the header synthesis. `delta ->
  bitpack`, `for` and `zigzag` should work per the registry but have no fixture.
- `plan_bitpack_packed_subset` computes 0 words for a `chunk_count == 0` chunk where the decode's
  own scan computes 1. Unreachable at top level (every column chunk has ≥1 row) and caught by a
  self-check that demotes the column to a whole fetch, but **worked around rather than fixed** — it
  will matter if a nested channel ever becomes subsettable.
- The **uncompressed GPU pin path has no group bounds**: it inserts through the merge-capable
  `insert_pinned_entry`, which would need the merge and degradation handling
  `pinned_zone_maps::append_column_from` provides. It keeps whole-chunk pruning.
- `max_gap_bytes` (the coalescing knob the network path needs) is plumbed but never tested
  non-zero, and there is no policy for choosing it.
- ~~Every pruning number here is on **clustered** data.~~ **Fixed (§5.4):** `pin_table(...,
  cluster_by=['l_shipdate'])` sorts each chunk as it is pinned, and on the UNSORTED SF1000 dataset
  that is worth −8.57% for +1.6% pin cost. The remaining gap is choosing the key — today it is an
  explicit argument, not a heuristic.

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

### 3.8 SF1000 validation — and the row-order finding that dominates everything (2026-09-08)

Built `/datasets/tpch_sf1000_sorted` (global sort, verified: row counts and checksums match the
source, lineitem row-group spans 2525 → **0.44 days**, files monotone) and ran the same on/off
sweep with the tuned `bench/sf1000-repro` config.

| batch | chunks pruned | suite ON | suite OFF | ON − OFF |
|---|---|---|---|---|
| 8 GB (production) | 17% | 6.9625 s | 6.9903 s | **−0.4%** |
| 2 GB | 45% | 7.2858 s | 7.4372 s | −2.0% |

**The ~4% projected in §7 Phase 1 did not materialise: the real number is −0.4%.** As at SF100,
2 GB is worse in absolute terms (7.29 s vs 6.96 s), so 8 GB-ON remains the best configuration and
W2's benefit there is at or below the noise floor.

### Why: pin chunks are not clustered, even when the file is

The prune *rate* is the tell. At SF100@512 MB q14 pruned 28/30 chunks; at SF1000@8 GB it prunes
3/19. q14 is a **one-month** predicate over seven years, so a clustered layout should prune ~98%.

A direct probe settles it. On the globally sorted SF1000 lineitem, a **single-day** predicate
(`l_shipdate = DATE '1995-06-15'`) prunes only **13/37 chunks** — 24 of 37 pin chunks contain one
particular day. The pin chunks span most of the seven-year range regardless of how the file is
sorted.

Two hypotheses tested and eliminated first: scan thread count (18 vs 3: identical, 41/75) and
lexicographic file ordering, where glob order `part.0, part.1, part.10, …` makes consecutive files
jump a year (zero-padding the names to restore numeric order moved q14 only 41→44 of 75).

The cause is documented in the pin path itself, `src/pin_table.cpp:260-262`:

> "chunk ranges overlap, because **the coalescer interleaves row groups rather than partitioning
> the key space**"

The scan's row-group coalescer builds each batch from row groups spread across the file set — good
for IO parallelism, fatal for zone maps. **Sorting the data on disk does not produce clustered pin
chunks.**

### What this changes

1. **Clustering must happen at pin time, not in the dataset.** §5.3's "(a) local sort per pin
   batch" is not an optimisation over the dataset sort — it is the only thing that works, because
   the pin discards file order regardless. That also makes the §5.1 measurement the relevant one:
   a GPU sort inside `materialize_pin_batches`, ~2.3 s for SF1000 lineitem.
2. **Every §3 pruning number is an upper bound that the current pin path cannot reach.** They were
   computed against parquet row groups and against synthetic reorderings, both of which assume the
   scan preserves row order into chunks. It does not.
3. **It does not invalidate Phase 2**, and arguably strengthens it: a coalescer-interleaved chunk
   is exactly the case where one min/max per chunk is useless and finer groups could still isolate
   runs — *if* the interleaving is at row-group granularity (≈1 M rows) rather than finer. Whether
   G = 8 (8,192 rows) is fine enough to see through it is now the open question, and it is
   measurable against the existing sorted dataset.
4. **The SF100 result was not wrong, it was lucky.** Six files and 512 MB batches meant a batch
   rarely spanned much of the key space. The effect scales with the file count, which is why it
   appeared at SF1000 and not at SF100.

### 3.9 The ceiling: this suite is join-bound, not scan-bound (2026-09-08)

Two follow-up measurements reframe how much chunk skipping can ever be worth on the winning
configuration.

**(a) A fine index would see straight through the interleaving — no pin-time sorting needed.**
The coalescer interleaves at *row-group* granularity: sorted SF1000 lineitem row groups are
1,048,576 rows, so a G = 8 group (8,192 rows) sits entirely inside one row group and inherits its
**0.44-day** span. The chunk spans seven years; the groups inside it do not. Measured at row-group
granularity on the sorted SF1000, **36.54% of scanned rows are prunable** (0.00% unsorted) — that
is the floor for what a G = 8 index recovers, against the 17% of chunks W2 prunes today at the
production batch size. So §3.8's "clustering must happen at pin time" is true for *chunk*-level
pruning but **not** a prerequisite for Phase 2: the fine index is an alternative to pin-time
sorting, not a complement to it.

**(b) But scan is a tiny share of the suite, so the ceiling is low.** Per-query bests from the
8 GB-ON run (6.963 s total):

| query | time | share |
|---|---|---|
| q18 | 1.7277 s | 24.8% |
| q9 | 0.9366 s | 13.5% |
| q21 | 0.8573 s | 12.3% |
| q1 | 0.5489 s | 7.9% |
| **q6** — the most scan-dominated query in TPC-H (filter + aggregate, no joins) | **0.0413 s** | **0.6%** |
| q14 | 0.0745 s | 1.1% |

The suite is dominated by join-heavy queries. Extrapolating the measured deltas linearly to a
hypothetical 100% prune rate gives **≈ −2.4% at 8 GB** (−0.028 s for 17%) and **≈ −4.8% at 2 GB**
(−0.151 s for 45%). That is the whole envelope for chunk skipping on a GPU-resident compressed pin
at SF1000 — and Phase 2's realistic 17% → 36% would be worth roughly **−0.9%**.

**Why the "34–38% of scanned bytes" headline does not translate.** On a GPU-tier compressed pin
the payload is already device-resident and decodes at 500–1700 GB/s, so skipped bytes are cheap
bytes. Scanned bytes are the right metric for a *fetch*-bound tier, not for this one.

### 3.10 What this means for the project

The mechanism works, is correct, and is nearly free — but on the configuration Sirius currently
wins with, there is little for it to win. Its value is concentrated where scan cost is not already
near zero:

1. **Host-tier pins**, where payload H2D is on the critical path and compressed host-tier *loses*
   7.0% today (`docs/super-sirius/compressed-pinning.md:52-56`). This is where §6's range-skipped
   fetch has real headroom, and it is the case that matters when data does not fit on the GPU.
2. **Spilling / larger-than-memory configurations**, for the same reason.
3. **Scan-heavy workloads**, which TPC-H at this scale is not.

None of that is visible in a benchmark where everything fits in GPU memory. Measuring the host tier
is therefore not just the next experiment — it is the one that decides whether the project has a
target at all. **Measured in §3.11: it does.**

### 3.11 Host tier: the project has a target after all (2026-09-08)

Same sweep, same clustered SF1000, `--pin host` instead of `--pin gpu`. This is the tier where a
skipped chunk saves the payload H2D transfer, not just the decode.

| batch | chunks pruned | suite ON | suite OFF | ON − OFF |
|---|---|---|---|---|
| 8 GB | 23% | 11.0396 s | 11.2354 s | **−1.7%** |
| 2 GB | 50% | **9.6778 s** | 10.4428 s | **−7.3%** |

**−7.3%, against −2.0% for the same arm on the GPU tier — and −0.4% at the GPU tier's production
batch.** Every one of the deltas is far outside the ~0.8% run-to-run noise floor. The per-query
pattern is the expected one: the scan-bound queries move most (q12 −0.124 s, q6 −0.101 s,
q15 −0.083 s, q20 −0.086 s, q14 −0.076 s).

**Extended and reproduced** (second independent process, 512 MB added):

| batch | chunks pruned | suite ON | suite OFF | ON − OFF |
|---|---|---|---|---|
| 8 GB | 23% | 11.0396 s | 11.2354 s | −1.7% |
| **2 GB** | 50% | **9.6250 / 9.6778 s** | 10.4733 / 10.4428 s | **−8.1% / −7.3%** |
| 512 MB | 69% | 10.4173 s | 11.9232 s | −12.6% |

The 2 GB arm reproduces across two processes (−7.3%, −8.1%), so it is solid. And the curve has the
same shape as the GPU tier, just shifted: **512 MB prunes far more (69%) and shows the largest
delta (−12.6%), yet is slower in absolute terms** (10.42 s vs 9.63 s) because batching overhead
overtakes the saving. **2 GB-ON at 9.625 s is the optimum.**

Three things follow.

**(a) The fetch hypothesis (§6) is confirmed empirically, before any implementation.** q6 costs
0.0413 s on a GPU pin and 0.1143 s on a host pin — 2.8×, and that difference is payload H2D. That
is the headroom chunk skipping is eating into, and it is why the same mechanism is worth 4× more
here.

**(b) The batch-size conclusion inverts on this tier.** On the GPU tier smaller batches always lost
in absolute terms (§3.8: 2 GB 7.29 s vs 8 GB 6.96 s). On the host tier **2 GB-ON (9.678 s) is the
best configuration measured**, beating 8 GB-OFF (11.235 s) by 13.9% and 2 GB-OFF by 7.3%. Fetch
granularity matters more than batching overhead once the transfer is on the critical path.

**(c) Phase 2 has a quantified target, and it is the same argument as §7 Phase 1: you cannot buy
granularity with batch size.** 512 MB reaches a 69% prune rate but pays for it in batching
overhead. The index's job is to deliver 512 MB's prune rate at 2 GB's batching cost. Scaling the
2 GB arm's saving (50% → 0.85 s) up to a 69% rate predicts ≈1.17 s off 10.47 s, i.e. **≈9.3 s
against today's best of 9.625 s — roughly a further −3%** on the host tier.

**This does not rescue the GPU tier**, and the two should not be conflated. Host-tier compressed is
still slower in absolute terms (9.68 s vs 6.96 s), so this is not "host tier now wins" — it is
"chunk skipping recovers a meaningful part of the fetch penalty that makes host tier lose", which
is what matters for the configurations where data does not fit on the GPU.

### 3.12 The group index, measured end to end (2026-09-08)

The capture is now wired into the pin, so the chain runs: per-group min/max at pin time → packed
arena → row ranges in the scan plan → the scan serves them. SF1000 clustered, host pin, 2 GB
batches (the best configuration from §3.11), pruning on in both arms so the only variable is the
finer capture:

| `pinned_zone_map_group_rows` | sub-chunk rows dropped | suite |
|---|---|---|
| 8192 (8 decode chunks) | **44.6e9** | **9.4813 s** |
| 0 (whole-chunk pruning only) | 0 | 9.6943 s |

**−2.20%** on top of the −8.1% whole-chunk pruning already delivered. It moves the scan-bound
queries and nothing else: q12 −0.043 s, q14 −0.043 s, q8 −0.037 s, q4 −0.036 s, q15 −0.020 s,
q20 −0.019 s. Two queries move the other way by less than the ~0.8% noise floor.

Worth being precise about what this is and is not:

- **It is not the fetch saving.** These chunks are still fetched whole; only the served row ranges
  narrow, so what is saved is downstream work, not the H2D transfer. The fetch skip (§6.5) is a
  separate and larger prize, and this number does not include it.
- **44.6e9 rows dropped for 0.21 s** says the same thing §3.9 did: on this suite, rows are cheap
  once resident. The value of skipping is dominated by what it lets you *not move*.
- The prediction that a G = 8 index sees through the coalescer's row-group interleaving (§3.9) is
  confirmed — the pruning materialises without any pin-time clustering.

### 3.13 The full ladder, on unsorted SF1000 (2026-09-09)

Every mechanism this project built, added one at a time, on the dataset a user actually has
(`/datasets/tpch_sf1000`, unsorted), all eight tables pinned HOST tier, 2 GB batches, best-of-3:

| arm | what it adds | suite | vs prev | vs none |
|---|---|---|---|---|
| `none` | nothing — zone-map pruning off | 9.6544 | — | — |
| `coarse` | whole-chunk zone maps | 9.7090 | +0.56% | +0.56% |
| `group` | + the per-group index (G = 8) | 9.7179 | +0.09% | +0.66% |
| `subset` | + range-skipped fetch | 9.7885 | +0.73% | +1.39% |
| `cluster` | + pin-time clustering | **8.7972** | **−10.13%** | **−8.88%** |
| `allplans` | + compression for the other four tables | 8.7739 | −0.27% | −9.12% |

**Read the top half first: on TPC-H as generated, every pruning mechanism is a small net LOSS.**
Zone maps, the group index and the fetch skip together cost **+1.39%**, because they prune 0.00%
(§3.1) and the machinery is not free. This is a sharper statement than §3.1's: the metadata is not
merely useless without clustering, it is mildly negative, and clustering is the switch that turns
all of it on.

**And the three are a package.** Holding clustering fixed and removing the other two:

| arm | suite | vs `cluster` |
|---|---|---|
| clustering + whole-chunk pruning only (`group_rows = 0`) | 9.4441 | +7.35% |
| clustering + the per-group index, fetch skip OFF | 9.4617 | +7.55% |
| clustering + index + fetch skip | **8.7972** | — |

So **clustering alone is worth −2.2%; the group index on top of it is worth nothing (+0.2%); the
fetch skip is worth −7.0%** — and only once the other two exist. The index locates the surviving
rows but changes nothing about what is moved; the fetch skip is what converts that knowledge into
work not done. This supersedes §3.12's "the group index is worth −2.20%", which was measured on the
pre-sorted dataset with the §6.5.5 duplicate-rows bug live.

### 3.14 Plan selection: ratio beats decode throughput on this path (2026-09-09)

`src/compression/simpatico_codegen/plans/tpch_sf1000/` carries `*_disabled.txt` plans for `part`,
`partsupp`, `customer` and `supplier` — the four tables that pin UNCOMPRESSED today. Three
candidate plan sets, each pinned end to end (clustered, fetch skip on, mean of two runs):

| plan set | selection rule | suite |
|---|---|---|
| explored | max ratio with decode ≥ 250 GB/s (the documented policy) | **8.8211** |
| the `_disabled` originals | same policy, explored earlier | 8.9089 (+1.00%) |
| cost-model | minimise `1/(ratio×370 GB/s) + 1/decode` | 9.0023 (+2.05%) |

The third row is the interesting one, because it is a **refuted hypothesis**. On the host tier a
column plausibly costs transfer plus decode per byte, so weighting decode throughput over ratio
looks obviously right: the max-ratio pick for `p_partkey` is 193x at 136 GB/s, which that model
scores at 7.36 ms/GB against 2.70 for not compressing at all. The model predicted its picks would
be 34–40% cheaper per table. They measured **2.05% slower**, consistently across both runs.

The error is the model's serial assumption. The decode has 18 scan threads and many concurrent
batches to hide behind; the host→device copy is a shared resource on the critical path. So ratio
buys more than the model credits and decode costs less, and the committed **"max ratio with decode
≥ 250 GB/s" floor is closer to right than a throughput-weighted rule**. Worth writing down because
the throughput argument is intuitively compelling and wrong.

Note also what the top two rows say: enabling those four tables' plans at all is worth −0.27%
against leaving them uncompressed, i.e. **nothing measurable**. That is a reasonable explanation
for why they were disabled, and there is no case for re-enabling them on this workload.

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

### 5.4 Built and measured (2026-09-09): strategy (a), and it is what makes the project pay

`CALL pin_table(..., cluster_by=['l_shipdate'])` sorts each chunk in
`materialize_pin_batches` — a `cudf::sort_by_key` immediately after materialization and before
anything observes the chunk, since the zone maps must describe the order the rows are stored in and
every id handed out downstream is positional against it.

**On the UNSORTED SF1000 dataset — the one a user actually has — host tier, 2 GB batches:**

| | suite (best-of-3) |
|---|---|
| no clustering | 9.7569 s |
| `cluster_by` lineitem/`l_shipdate`, orders/`o_orderdate` | **8.9210 s** |
| | **−8.57%** |

Pin cost: whole-process wall 127 s → 129 s, **+1.6%**, against §5.1's ~2.3 s / ~1.5% prediction.

The movers are the scan-bound queries, and several roughly halve: q6 0.2350 → 0.1183, q15 0.2562 →
0.1293, q14 0.2590 → 0.1491, q20 0.3522 → 0.2153, q1 1.0119 → 0.8315. 22/22 byte-exact at SF100.

Three things worth recording:

- **§5.2's central claim holds end to end.** A local sort prunes nothing at chunk granularity and
  everything through the group index. At SF10 a one-month predicate drops **59.1M of 60.0M rows**
  with `cluster_by`, in **19 row ranges over 19 surviving chunks** — one contiguous run per chunk,
  and *zero* chunks pruned coarsely. Without `cluster_by` the same query drops nothing at all.
- **It beats the pre-sorted dataset** (8.92 s vs 9.54 s on `tpch_sf1000_sorted`), which is not a
  paradox: §3.8 showed the coalescer interleaves row groups, so a globally sorted FILE still yields
  pin chunks spanning the key range. Sorting after coalescing is strictly better than sorting
  before it, and it needs no dataset rewrite.
- **One query really regresses: q12, +0.027 / +0.038 s** across two runs (~+7% of q12). q17 looked
  like a second one at +0.080 s, but a same-config repeat put it at +0.002 s — it was noise, and
  the per-query verdicts below are taken from two independent clustered runs for exactly that
  reason. Everything else moves the right way.

### 5.5 What clustering costs the compression, and why re-exploring does not help (2026-09-09)

Clustering reorders rows, so every column's compressibility changes. Measured directly: one 16.8M-row
pin chunk of SF1000, compressed with the committed plans in file order and again sorted by the
cluster key (`simpatico benchmark --mode per-column`).

**The damage is one column per table, and it is the key the data used to be ordered by:**

| column | plan | unclustered | clustered | bytes |
|---|---|---|---|---|
| `l_orderkey` | `delta -> bitpack` | 12.393x | **2.739x** | 10.8 MB → **49.0 MB** |
| `o_orderkey` | `delta -> bitpack` | 12.393x | **2.521x** | 10.8 MB → **53.2 MB** |
| `l_shipdate` | `bitpack` | 2.651x | 142.932x | 25.3 MB → 0.5 MB |
| `l_linestatus` | `dictionary -> bitpack` | 37.372x | 568.295x | 2.2 MB → 0.1 MB |
| `o_orderstatus` | `dictionary -> bitpack` | 19.321x | 235.044x | 4.3 MB → 0.4 MB |

Every other column is unchanged to three decimals — they were uncorrelated with the sort key before
and after. **Net per table: lineitem 613.7 → 598.9 MB (−2.4%), orders 1091.3 → 1104.8 MB (+1.2%).**
So clustering is roughly footprint-neutral overall, and strictly better for lineitem; it just moves
the bytes from the date columns to the order keys.

**Re-exploring does not recover the orderkey.** `simpatico explore --score pareto --rerank-top 16`
on a clustered chunk returns plain `bitpack` for both keys — 2.706x for `l_orderkey` against the
committed cascade's 2.739x. The loss is intrinsic: sorting on shipdate scatters the order key, and
no cascade compresses a scattered 64-bit key. What the re-explore does find is that the `delta` is
now **dead weight** — it costs 0.1% more bytes and buys nothing, while removing it takes decode from
1126 → 1435 GB/s (`l_orderkey`) and 1135 → 1496 GB/s (`o_orderkey`).

That decode win does not reach the suite: clustered SF1000 with the re-explored plans measures
8.9547 s against 8.8539 s with the committed ones, i.e. **+1.14%, inside the ~0.8% noise floor**.
**Conclusion: keep the committed plans.** The follow-up is closed as a negative result rather than
left open — worth knowing, because "the plans were tuned on unclustered data" is an obvious
objection to clustering and it turns out not to matter.

Refused by design: **duckdb-native pins**, whose rows stay addressable by DuckDB row id — the
deleted-row keep-masks are positional against the pinned order, so reordering would misapply them
silently. `cluster_by` on `format='duckdb'` is an error rather than a silent no-op.

Still open: **choosing the key automatically.** Today it is an explicit argument
(`SIRIUS_PIN_CLUSTER_<TABLE>` in the benchmark harness), which is honest but leaves the win to
whoever knows the workload.

## 6. Skipping the *fetch*, not just the decode

Chunk skipping saves different things at different tiers. On a GPU-tier pin the payload is already
device-resident, so a skip saves only decode. On a **host** pin, and on any future **disk** read, a
skip could save the transfer itself — which on the host tier is the dominant cost
(`docs/super-sirius/compressed-pinning.md:52-56`: host-tier compressed *loses* 7.0% today precisely
because every scan batch pays payload H2D → sync → decode on the critical path).

The question is whether a row range maps to a computable byte range. For simpatico it does, exactly.

### 6.1 Byte-range addressing is a SECOND out-of-band table, not a derived bitpack fact

The index that decides *which groups survive* is out-of-band min/max at G = 8 groups (§4.3). Fetch
skipping needs a second, separate thing: a map from **group id → byte range** in each bulk leaf
buffer. It is tempting to derive that from what bitpack already stores, and for a bitpack root you
can — `bp_offsets[c]` is an exclusive scan of `(chunk_count[c]*chunk_bits[c]+31)>>5`
(`src/compression/simpatico_codegen/src/bridge/offsets_cumsum.cu:9-11`), so 5 B/chunk of existing
metadata locates every chunk exactly.

**But that is a bitpack channel, and depending on it repeats the mistake §2/W1 exists to avoid.**
The plan is chosen by an explorer optimising ratio and decode throughput
(`src/compression/simpatico_codegen/src/explore/compression_explorer.cpp`); nothing keeps a column
on a bitpack root, and a plan change would silently remove the addressing. The min/max index is
stored out-of-band for exactly this reason, and the byte-range map has to be too.

**So: the encoder emits an explicit per-group byte-offset table per bulk leaf buffer, at the same
G = 8 granularity as the index.** Stored, not derived. Properties:

- **Granularity must match the index.** We skip at group granularity, so byte ranges are needed at
  group granularity — 23,072 entries for a 189 M-row pin chunk at G = 8, not the 184,571 that
  per-1024-chunk addressing would need.
- **Cost is negligible.** 8 B per group per bulk buffer. Only *bulk* buffers need one — the small
  per-chunk metadata channels are fetched whole. For 16 lineitem columns (~20 bulk buffers) that is
  23,072 × 8 B × 20 ≈ **3.7 MB against ~8 GB of payload, 0.046%**. Larger than the 5 B/chunk
  bitpack derivation, and worth it to be plan-independent.
- **It degrades gracefully, like a missing zone-map cell.** An operator that cannot be
  group-addressed simply emits no table, and its buffers are fetched whole. No correctness
  consequence, just no fetch saving for that column.

Which operators *can* emit one, for the shipped TPC-H plans:

| root | can emit a group→byte table? | why |
|---|---|---|
| `input -> bitpack` | yes | variable width per chunk, but the encoder knows every boundary as it writes |
| `delta -> bitpack` | yes | `delta_first[c]` is a *per-chunk anchor*, not a running global prefix (`encode/jit/renderer.cpp:403,443`), so groups decode independently |
| `dictionary -> bitpack(indices)` | yes for the indices | the key set is per-pin-chunk and small — fetch it whole |
| `str_split -> bitpack(offsets)` | offsets yes, `chars` two-step | `chars` is data-dependent: you need `offsets[start..end]` before you know the char range. One extra dependent round trip |
| `ans` / `snappy` / `lz4` / `bitcomp` | **no** | codec-internal chunking we do not control; would need groups aligned to codec blocks |
| `identity` | trivial | fixed stride, computable without a table |

**This does put a third axis on plan choice.** Whether a column can skip fetch at all depends on
its plan, and the explorer currently optimises only ratio and decode throughput — so a
ratio-optimal plan can silently cost fetch skippability, the same way §5.1 notes clustering changes
what the ratio-optimal plan is. Worth surfacing to the explorer eventually; not a blocker, since
the degradation is per-column and graceful.

### 6.2 The plumbing already has the right shape

The host→GPU path drives every fetch through a byte-range callback —
`simpatico::payload_fetch_fn = void(offset, size, dst, stream)`
(`src/compression/compression_converters.cpp:167-171`, declared at `compressed_table_io.hpp:111`) —
and `read_compressed_table_subset_from_memory` already does **column**-granular partial fetch via
each buffer's `payload_offset`. Going from "which buffers" to "which byte ranges within a buffer"
is the same mechanism one level finer, not a new one.

### 6.3 Caveats to measure before believing it

- **Guard words.** Decode reads a few words past a chunk's end ("dense Compact words + decode
  gather guard words", `tests/test_bitpack_layout_contract.cpp:5,141`), so every fetched range
  needs slop. Cheap, but it must be in the range arithmetic.
- **Many small transfers may beat one large one only if they coalesce.** On clustered data
  surviving groups are *runs*, so ranges should merge into a few large copies — but that is an
  assumption, not a measurement. Unclustered survivors would scatter and could easily be slower
  than one bulk copy. Any implementation needs a coalescing pass and a "just fetch it all"
  fallback above some fragmentation threshold.
- **Disk amplifies both effects**: bigger win per skipped byte, worse penalty for scattered reads.

### 6.4 Where the fetch win lands

| tier | is there a fetch to skip? | value |
|---|---|---|
| GPU pin | no — payload already device-resident | decode only |
| **host pin** | **yes, and it is the dominant cost** | host-tier compressed loses 7.0% today entirely on fetch; see §8 for why this is the first thing to measure |
| spilled to disk | yes | biggest per skipped byte, worst penalty for scattered reads |
| **simpatico as an ingestion format** | yes | its own section — see §7 |

## 6.5 What range-skipped fetch actually needs

This is the piece worth building: it targets the measured host-tier **−8.1%**, where every scan
batch pays payload H2D on the critical path, rather than the ≈2.4% envelope of a GPU-resident pin.

### 6.5.1 The key structural fact: a compacted subset of chunks is a valid column

`bp_offsets` — where each 1024-row chunk's bits start inside `packed` — is **not stored**. It is
computed at decode time by an exclusive scan over the `chunk_count` and `chunk_bits` arrays that
were loaded (`src/bridge/offsets_cumsum.cu:78-95`).

That has a consequence worth stating plainly:

> If we load only the surviving chunks' metadata entries and only their `packed` bytes,
> concatenated in order, the decode-time scan produces correct offsets **into the compacted
> buffer**. A normal, unmodified full decode of that smaller table then yields exactly the
> surviving rows.

**So the fetch path needs no new decode enumerator** — the thing §6.6 says a GPU-resident chunk
needs. Skipping transfer and skipping launches turn out to be separate problems with separate
solutions, and the transfer one is both more valuable and cheaper.

### 6.5.1a Two ways to get the byte ranges — and why the host path needs no format change

Where the ranges come from differs by backend, but the *result* is the same type, so the consumer
does not care:

| backend | how ranges are obtained | cost | needs the stored table? |
|---|---|---|---|
| **host pin** | derive from the column's own per-chunk metadata (`chunk_count`/`chunk_bits`, 5 B/chunk), read from pinned host memory | a memcpy, no round trip | **no** |
| disk / S3 | read the group→byte table out of the segregated metadata region (§7.5) | one sequential read, already amortised over the query | **yes** |

The host path can therefore ship with **no format change at all** — deriving is cheap precisely
because the metadata is already local. Deriving is the wrong answer over a network, where those
same 5 B/chunk arrays are scattered through the payload region and cost a request each; that is
what the stored table buys.

**Keep both behind one interface.** `plan_bitpack_packed_subset` (derived) and a future
`plan_from_stored_table` both return `buffer_subset`, so the fetch machinery, the coalescing policy
and the header synthesis are written once and shared. Disk and S3 remain the eventual target; the
host path is the cheap way to prove the mechanism first, not a different mechanism.

### 6.5.2 The pieces, in dependency order

1. **Byte ranges per surviving chunk.** Derived from per-chunk metadata for a host pin, or read
   from the stored group→byte table (§6.1) for disk and S3 — see §6.5.1a. **Done** for the derived
   case (`073dd5e1`: `plan_bitpack_packed_subset`, guard words and zero-bit chunks included).

2. **A per-operator "extract these groups" capability. Done** (`d14b9161`). To hand the decoder a self-consistent
   smaller table, an operator must say which of its channels are *per-chunk metadata* (compact by
   selecting the surviving entries) versus *bulk* (fetch the surviving byte ranges). For `bitpack`
   that is `{chunk_min, chunk_count, chunk_bits}` metadata and `packed` bulk. An operator that
   cannot answer refuses, and its column is fetched whole — the same graceful degradation as a
   missing zone-map cell. Per the shipped plans this covers the bitpack- and delta-rooted columns;
   `ans`/`snappy`/`lz4`/`bitcomp` refuse.

3. **Header synthesis — DONE (`804aa986`).**

   `read_compressed_table_subset_from_memory` allocates each leaf buffer at its *declared* size and
   fills it via `payload_fetch_fn(offset, size, dst, stream)`. So serving a compacted table means
   handing it a header whose declared sizes are the compacted ones, plus a fetch that gathers the
   ranges. The reader itself then needs no change at all — that is what makes this approach
   attractive.

   Shape, and why it is two-phase:

   ```
   // Phase 1: read the small per-chunk metadata buffers (5 B/chunk for bitpack) — a memcpy on a
   //          host pin. Their VALUES are what the byte ranges are derived from.
   // Phase 2: compute per-buffer subsets, re-lay-out the payload densely, emit a new header.
   std::string plan_chunk_subset(std::span<const std::uint8_t> header,
                                 std::span<const std::uint32_t> surviving_chunks,
                                 metadata_read_fn const& read_metadata,   // phase 1
                                 std::vector<std::uint8_t>& out_header,
                                 std::vector<gather_range>& out_gather);  // {src_offset, size, dst}
   ```

   Per leaf buffer, classified by `buffer_layout(leaf.kind, buffer.name)`:
   `per_chunk_metadata` → `plan_metadata_subset`; `bulk_chunked` → `plan_bitpack_packed_subset`
   (or the stored-table equivalent); `whole_column` → the whole range, and its presence means the
   column is not subsettable at all.

   **Exactly three numeric fields need to change**, which is fewer than it first looks. Per buffer
   the writer emits `name, type_tag, size_bytes, payload_offset`
   (`src/api/compressed_table_io.cpp:798-801`) — `leaf_buffer_desc::num_rows` is **not serialized**,
   it is derived on read. So:

   | field | new value |
   |---|---|
   | buffer `size_bytes` | `buffer_subset::compacted_size` |
   | buffer `payload_offset` | dense re-layout of the compacted payload |
   | leaf `num_rows` (`:789`) | the node's own output length over surviving chunks |
   | column `num_rows` | sum of surviving chunks' rows (the last chunk is short) |

   `leaf_desc::num_rows` is explicitly the *node's own* output length rather than the column's
   (`leaf_desc.hpp:106-110`), so the last two are different quantities and must be recomputed
   separately.

   **Patch in place rather than re-emitting.** All four are fixed-width fields at computable
   offsets, so recording their byte positions during the parse and overwriting them keeps the
   output structurally identical to what the writer produces. Re-emitting from parsed records
   would mean a second writer that can drift from `build_compressed_table_header` — the kind of
   duplication that fails silently years later. `parse_hpln_header` is static in that file, so an
   optional offset-recording out-parameter is the natural place.

   **Why this deserves its own careful pass rather than being appended to a long session:** it is
   binary-format rewriting where a wrong `size_bytes` does not fault — the reader allocates what
   the header says and the decode reads whatever landed there, so the failure is wrong values, not
   a crash. It wants the same treatment `plan_bitpack_packed_subset` got: a test asserting that
   "every chunk surviving" reproduces the original header byte-for-byte, so the subset path
   provably degrades to the existing one.

   Implemented as `build_chunk_subset_header` with `parse_hpln_header` gaining an optional
   offset-recording out-parameter; the four fields are patched, nothing is re-emitted. Two
   requirements emerged that this design had not anticipated, both of which would have produced
   silently wrong data:

   - **A leaf whose `num_rows` differs from its column's disables subsetting for that column.**
     Survivor ids name the *column's* chunk grid; a nested node (an rle `values` channel, say)
     chunks on a different grid, so chunk *c* there is not the same rows.
   - **A per-buffer self-check**: with every chunk surviving, the layout model must reproduce the
     `size_bytes` the writer declared, else the column is demoted to a whole fetch. This makes the
     byte-for-byte property hold *by construction* rather than by test, and it is what caught the
     `chunk_count == 0` divergence noted in §0.

   A column that cannot be subsetted is emitted whole (only its payload offsets shift to keep the
   payload dense), so a mixed table degrades per column rather than failing.

4. **Row-count bookkeeping at the scan level. Done.** The projected representation reports the
   rows the subset will produce and scales its byte footprints to match, so a reservation sized
   off it fits what the decode returns. Late-mat stays gated off for a narrowed batch for the
   reason in `a482f283`; on the host tier it is never stamped anyway.

   The integration also had to answer a question the design did not ask: **how a chunk's row
   ranges become BATCHES depends on what the chunk can serve.** An uncompressed device chunk
   slices, so a range is a batch. A compressed chunk cannot be sliced, so all its ranges must
   arrive as ONE batch naming the decode chunks to fetch — and a pinned HOST column, which
   `host_data_representation::slice` narrows by COLUMN only, serves whole. Getting that wrong is
   not a missed optimization; see §6.5.5.

5. **Coalescing policy (§7.3).** Merge adjacent surviving ranges, tolerate reading pruned bytes in
   a small gap, and fall back to a bulk fetch past a fragmentation threshold. `align_and_coalesce`
   already exists per backend and takes a caller-supplied alignment; what is missing is the policy,
   not the mechanism. **Less urgent than it looked**: on clustered data the surviving groups
   already merge into a handful of large copies — a measured q6 batch gathered 6,152 surviving
   decode chunks into **16 ranges**, so `max_gap_bytes = 0` costs almost nothing today.

### 6.5.4 A compacted column and a whole one cannot be the same table

`build_chunk_subset_header` emits a column it cannot address per chunk WHOLE, which §6.5.2 step 3
described as degrading "per column rather than failing". That is true of the column and false of
the **table**: a compacted column has the survivors' row count, a whole one has the chunk's, and
`cudf::table` rejects the pair (`Column size mismatch: 30433664 != 8192`).

The failure was loud only by luck — the mismatch throws inside `prepare_for_processing`, the query
falls back to DuckDB, and the *results stay correct* while the query gets ~10× slower. The first
SF1000-shaped measurement of this path read `+48%` and validated 22/22, which is exactly what a
silent fallback looks like.

So the builder now reports, per column, whether it was compacted, and the converter uses the
subset only when **every column the scan reads** was. Unread columns may be whole — the reader
never fetches their buffers.

### 6.5.6 Rows versus column state: what makes a dictionary column addressable

A dictionary column is `input -> dictionary -> keys_offsets, keys_chars, indices`, usually with
`dictionary.indices -> bitpack` and sometimes `dictionary.keys_offsets -> bitpack` on top. Serving
a subset of its chunks looks impossible under the original rule — the keys are `whole_column`, and
one of those refuses the column — but the rule was conflating two different things:

| | fetched whole | depends on WHICH rows are served |
|---|---|---|
| an lz4 / snappy / ans payload | yes | **yes** — so the column cannot be subsetted |
| a dictionary's keys | yes | **no** — they stay valid for any subset of the indices |

So `ChannelLayout` gained `column_state` beside `whole_column`, and `supports_chunk_subset`
accepts it. The keys are emitted whole (their `num_rows` and `size_bytes` untouched, only their
payload offset moved), the indices compact on the column's 1024-row grid, and an ordinary decode
gathers the surviving indices against the full keys.

**The mark has to follow the tree, not just the buffer.** When the keys are themselves compressed
(`dictionary.keys_offsets -> bitpack`, as `o_orderpriority` and `l_shipinstruct` do), that bitpack
leaf's buffers look perfectly row-indexed on their own grid, and its `num_rows` is the key count —
which the old code read as "a leaf whose length is not the column's", i.e. as a refusal. The
builder now walks the edges from the root and marks every node reached through a column-state
channel, so the whole keys subtree is fetched whole and its length is not evidence about the
column. `o_clerk`'s `keys_chars -> ans` and `keys_offsets -> delta -> rle` fall out of the same
rule for free.

`null_mask` is deliberately left unclassified: it is one BIT per row, which no layout here
describes, so a nullable dictionary column refuses rather than being addressed with byte-per-row
arithmetic. (str_split's `null_mask` is classified `bulk_fixed_stride` today, which is wrong for
the same reason; it is unreachable because str_split refuses anyway, and the per-buffer self-check
would demote the column if it ever became reachable.)

**Why `str_split` is a different problem.** Its `chars` are addressable only through the `offsets`
VALUES — which is the `bulk_variable` pattern bitpack already uses — but two things break the
byte-gather model: the offsets are cumulative over the whole column, so a compacted `chars` needs
its offsets REBASED rather than copied, and in the plans that matter the offsets are themselves
bitpacked, so the values are not readable host-side without decoding them. That is a value
rewrite, not a range gather, and it needs its own design (either a stored per-chunk chars offset
table, §6.1, or a post-fetch rebase on the GPU).

### 6.5.5 Serving row ranges duplicated rows on two of the four serve paths

Wiring the fetch skip surfaced a **correctness bug in what `a482f283` already shipped**. The
provider walked `survivor_row_ranges` and served one batch per range, but only ONE of the four
serve paths narrowed anything: a `device_pin_chunk`'s uncompressed columns. The other three —
per-column device storage (`data_batches_by_column`), a compressed chunk (either tier), and an
uncompressed HOST chunk — ignored the range and served the WHOLE chunk, once per range. A chunk
with two surviving groups was therefore emitted twice.

It reproduces as wrong query results: TPC-H q5 on clustered SF100, host tier, `group_rows = 8192`
returns revenue ~6% high, and is correct with `group_rows = 0`. It is fixed by deciding the batch
shape per chunk before serving (step 4 above) rather than assuming every chunk can slice.

Two lessons worth keeping:

- **The §3.12 −2.20% was measured with this bug live.** The timing is not invalidated (duplicated
  rows cost time, they do not save it) but nothing about that run's results was checked.
- A serve path that silently ignores a narrowing instruction is indistinguishable from one that
  cannot narrow — until the row count is wrong. The provider now decides `chunk_slices_per_range`
  explicitly for every chunk, so a new storage form has to answer the question.

### 6.5.3 Does the existing plan still hold?

Mostly, with one correction and one thing now done.

| claim | status |
|---|---|
| §6.1 — byte addressing must be a stored table, not derived from bitpack channels | **holds**, and step 2 above is the part §6.1 did not spell out |
| §6.2 — `payload_fetch_fn(offset, size, dst, stream)` is already the right shape | **holds** |
| §6.3 — guard words need slop; coalescing is unmeasured | **holds**, still the main risk |
| §6.6 — compressed chunks need a new decode enumerator | **corrected**: true for a GPU-resident chunk, **not** for the fetch path (§6.5.1) |
| §7.5.5 — the in-memory sidecar should move to an arena before it has a consumer | **done** (`56db39e9`) |
| §8.3 — the persisted copy enables partial re-read | **holds**, and is the same mechanism as this |

The one genuinely new requirement is step 2: an operator-level notion of "metadata channel vs bulk
channel". Nothing in the plan anticipated it, and it is what makes compaction expressible without
special-casing bitpack everywhere.

## 6.6 Skipping decode inside a GPU-resident compressed chunk

**Scope: this section is about a chunk whose payload is already device-resident**, where the only
thing to save is SM time. The *fetch* case is different and does **not** need what follows — see
§6.5, which can skip transfer without any new decode primitive.

Serving surviving row ranges works today for **uncompressed** chunks — `cudf::slice` narrows the
views, ownership stays whole, no copy. A GPU-resident compressed chunk still decodes whole, and
closing that gap needs one thing simpatico does not have.

### 6.6.1 Why none of the three existing enumerators fits

The decode is a product of an *enumerator* × a *consumer*
(`codegen/decode/jit/renderer.hpp:92-101`). Three enumerators can express a selection:

| enumerator | grid | selection storage | skips decode work? |
|---|---|---|---|
| `all_rows` | every chunk | — | no |
| `mask_bits` | **every chunk** (dense grid, compacted by rank) | 1 bit/row | **no** — it compacts the output, it does not skip launches |
| `index_list` | every chunk | 4 B/survivor row | no |
| `chunk_csr` | **only touched chunks** | `in_chunk_rows`, **2 B/survivor row** | **yes** |

So `chunk_csr` is the only one that actually avoids work — "the grid covers only TOUCHED chunks,
and block b serves `chunk_ids[b]`" (`renderer.hpp:96-100`), with empty chunks "absent rather than
launched-and-skipped" (`chunk_row_set.hpp:22-23`).

But its cost is wrong for this use. `chunk_row_set` stores `in_chunk_rows` — a `uint16` in-chunk
position per surviving row (`chunk_row_set.hpp:78`). For a 189 M-row pin chunk with half its rows
surviving that is **~190 MB of device memory for the selection alone**, to express a selection
whose content is "all 1024 rows of these chunks". Every one of those `uint16`s is the sequence
0…1023 repeated.

### 6.6.2 What is actually needed

A **dense chunk-list enumerator**: block *b* decodes all of `chunk_ids[b]` and writes it at output
offset `b * 1024`. Selection storage is one `uint32` per surviving *chunk* — for the same 189 M-row
chunk, **~370 KB instead of ~190 MB**, a 500× reduction — and there is no per-row bookkeeping to
build.

It fits the group index exactly: groups are a whole number of 1024-row decode chunks by
construction, so a surviving group *is* a run of chunk ids, and a surviving row range converts to
one with a shift.

Concretely that means a fourth `Enumerator` alongside `all_rows` / `mask_bits` / `index_list` /
`chunk_csr`, its renderer support, and a launcher — the same surface the existing sparse shapes
already occupy (`kShapeSparseConsume` and friends, `renderer.hpp:118-127`). Not conceptually new,
but real work inside the decode JIT.

### 6.6.3 Relative priority

This is the smaller of the two remaining pieces. It saves SM time on a GPU-resident pin, where
§3.9 measured the whole envelope for chunk skipping at ≈2.4% of suite time. The fetch path (§6.5)
saves the H2D on a host pin, where the measured benefit is −8.1%. They are independent, and the
fetch path does not depend on this one.

A per-plan-root caveat carries over from §6.1: `dictionary`, `str_split` and `delta` roots are
refused by `decompress_column_rows` today (`simpatico_codegen.hpp:261-308`), so a mixed table skips
what it can and full-decodes the rest. That is the same graceful-degradation shape as a missing
zone-map cell.

## 7. Simpatico as an ingestion format

Today simpatico is a *pin-time* representation. `.hpln` exists as a serialization
(`api/compressed_table_io.hpp:14-39`) but nothing ingests it: a table is read from parquet (or
DuckDB storage), materialized, then compressed on the way into the cache. The question is whether
being able to *read* simpatico directly is worth the format work.

The argument is the same one as §6.1, applied one tier further out — and it is stronger against
parquet than it is against a host pin, for a reason specific to object storage.

### 7.1 What it would buy over parquet

Parquet's skip granularity here is a **~1 M-row row group**, and it is the only granularity that
exists on that path: `PageIndex` / `ColumnIndex` / `OffsetIndex` are present in these files
(parquet-rs writes them by default) but are **never read** by either Sirius or DuckDB (§1). So
sub-row-group skipping is not merely coarse on the parquet path, it is absent.

A simpatico file would carry exact **group-granular** addressability (G = 8, i.e. 8,192 rows) from
the byte-offset table of §6.1 — ~0.046% of the payload, and still two orders of magnitude finer
than a parquet row group. Note this is a table the writer must emit, not something derivable from
any particular compression plan; §6.1 explains why deriving it from bitpack's channels would be a
mistake.

### 7.2 The S3 case, where the gap is widest

On object storage the mismatch is not just granularity, it is that **parquet's row-group size
fights the storage layer's own striping.** S3 stripes and replicates objects internally for
durability, with a layout we do not control and cannot see. A ranged `GET` that lands inside a
~1 M-row row group still forces the backend to reconstruct whatever internal unit that range falls
in, so a "pruned" parquet read plausibly costs the backend the same work as an unpruned one — the
saving shows up in bytes-on-the-wire but not necessarily in backend cost or latency. We get to skip
a row group only when the *whole* row group is prunable, which §3.1 says is essentially never on
unclustered data and, even clustered, is a 1 M-row all-or-nothing bet.

Finer granularity plausibly helps here precisely because it decouples "what we skip" from "what the
backend has to restore": many small surviving ranges can be re-packed into request sizes that suit
the backend, rather than being forced to the row-group boundary. **This is a hypothesis about
storage-backend behaviour that we have not measured, and it is the interesting thing to experiment
with** — it may well turn out that the backend cost is dominated by object-level effects that
neither granularity escapes.

### 7.3 The tension: fine skipping vs. large transfers

Fine granularity is worthless over a network if it turns one 8 MB `GET` into two hundred 40 KB
`GET`s. Network round-trips dominate; the REST reactor already says so —
"Network round-trips are high-latency; read ahead on demand rather than eagerly prefilling"
(`src/include/io/rest/rest_reactor.hpp:330-332`). So range-skipped ingestion needs **read
coalescing as a first-class part of the design, not an afterthought**: merge surviving ranges,
tolerate reading pruned bytes in the gaps when the gap is cheaper than an extra request, and fall
back to a bulk read once fragmentation passes a threshold.

The good news is the mechanism already exists and is already per-backend.
`io_context::align_and_coalesce` is on the virtual interface (`src/include/io/io_context.hpp:209`)
with backend-specific implementations: the uring one floors alignment at `IO_BLOCK_SIZE` for
O_DIRECT (`src/io/uring/uring_reactor.cpp:753-795`), while the REST one has no physical alignment
and honours a caller-supplied lower bound as a pure coalescing knob
(`src/io/rest/rest_reactor.cpp:1103-1108`). That caller-supplied alignment is exactly the dial this
needs — set it to a network-sensible minimum request size and the existing pass does the merging.

What is missing is the *policy*: today the alignment is chosen for physical constraints, not for a
skip-vs-request-count trade-off. Something like "merge ranges whose gap is under G bytes; if the
merged set still exceeds N requests, widen G until it doesn't" — with G and N measured, not
guessed.

### 7.4 Recommendation

Worth building, but **not first**. Sequence it behind the host-tier fetch experiment (§6, §8):
that exercises the identical mechanism — skip byte ranges of a compressed payload before decoding —
needs no new file format, and has a concrete existing target. If range-skipped fetch does not pay
over a ~370 GB/s C2C link where round-trips are nearly free, it will not pay over S3 where they are
not.

Two things to settle before committing to the format, both experiments rather than design work:

1. **Does coalescing hold up?** On clustered data surviving groups should be runs that merge into a
   few large requests. Measure the request-count and bytes-amplification curve as a function of
   selectivity and clustering strength.
2. **Does S3 actually reward finer ranges?** The §7.2 argument is a hypothesis about backend
   striping. A standalone probe — ranged GETs of varying size and scatter against a real bucket,
   measuring latency and throughput — settles it without touching Sirius at all, and should be run
   before any format work.

### 7.6 MEASURED (2026-09-10): S3 charges for requests, not for skipped bytes

§7.2 asked whether S3's internal striping means a pruned ranged GET costs the backend the same as
an unpruned one. Measured directly: `g7e.2xlarge` in **us-east-2b**, bucket in **us-east-2** (same
region, so this is S3's own behaviour and not an inter-region link), 4 GB object, stdlib HTTPS
ranged GETs with keep-alive, one connection per worker.

**Does reading less take proportionally less time?** (16 MB ranges, concurrency 64)

| read | time vs full read | ideal | throughput |
|---|---|---|---|
| 100% | 1.000x | 1.00 | 994 MB/s |
| 50% | **0.500x** | 0.50 | 994 MB/s |
| 25% | **0.259x** | 0.25 | 957 MB/s |
| 5% | 0.185x | 0.05 | 250 MB/s |

Proportional down to 25%. The 5% point falls short for a reason that is **ours, not S3's**: 5% of
1 GB in 16 MB ranges is only 3 requests, which cannot fill 64 threads. Keeping the pipe full is a
concurrency problem, not a granularity one.

**Fragmentation at equal volume** (268 MB every row, concurrency 64) — the §7.2 question:

| pattern | requests | throughput |
|---|---|---|
| 16 x 16 MB | 16 | 0.96 GB/s |
| 64 x 4 MB | 64 | 0.77 GB/s |
| 256 x 1 MB | 256 | 0.70 GB/s |
| 1024 x 256 KB | 1024 | 0.49 GB/s |
| 4096 x 64 KB | 4096 | 0.16 GB/s |

Fragmentation costs 6x from 16 MB down to 64 KB — but **entirely through request count, with no
striping term**. The model `throughput = min(NIC cap, concurrency x range / RTT)` predicts every
row to within 4–23% with no term for the number of distinct extents:

| range | predicted | measured |
|---|---|---|
| 64 KB | 0.17 | 0.16 |
| 256 KB | 0.60 | 0.49 |
| 1 MB | 0.89 | 0.70 |
| 4 MB | 1.00 | 0.77 |
| 16 MB | 1.00 | 0.96 |

Note the p50 latency FALLS as ranges shrink (254 ms → 24 ms): small ranges pay a fixed ~25 ms
floor, not a per-byte penalty. **So the §7.2 hypothesis is refuted in the direction that favours
the project: the bytes you skip really are free, and what you pay for is asking.**

### 7.7 The coalescing policy, as a number (answers §7.3)

The cost of a request is a LATENCY fact; the bytes it is worth dragging along to avoid one scale
with achievable BANDWIDTH. So the policy is a formula, not a constant:

> **`max_gap_bytes ≈ per_request_cost × achievable_bandwidth`**, both measurable at runtime.

Measured with curl (§7.8) at equal volume and equal 256-way parallelism, 4 MB ranges against
16 MB, on a **busy** and then an **idle** instance:

| instance | per extra request | max_gap_bytes | 4 MB vs 16 MB |
|---|---|---|---|
| busy | 0.755 ms | 1.99 MB | +36% |
| **idle** | **0.286 ms** | **0.75 MB** | **+13%** |

Take the idle row as the estimate and the spread as the error bar: **~0.75 MB, and load-sensitive
by 2.6x**. Coalesce to ~16 MB runs and keep ~1 GB in flight (64 x 16 MB) or the pipe starves
regardless of how well the ranges are merged. Note the fragmentation penalty at 4 MB is modest
(+13% idle) — it is the sub-MB ranges that collapse.

**Do not hardcode it.** The same formula gives 293 KB from the interpreter-bound python numbers,
0.75 MB idle and 1.99 MB busy — a 7x span, and exactly the kind of constant that gets baked in once
and never revisited. It moves with the link AND with instantaneous load.

That is `max_gap_bytes`, which §7.3 has carried as "measured, not guessed" since the project
opened. It also says a **G = 8 group is far too fine to address individually over S3** — 8192 rows
of a 4-byte column is 32 KB, deep in the 0.16 GB/s regime — while being exactly the right
granularity to *decide* with and then coalesce. The index picks the rows; the coalescer picks the
requests; they are different questions.

**What this means for the project.** §6.5 measured what our fetch skip actually produces on
clustered data: roughly one contiguous run per buffer covering ~32% of it, tens of MB at a 512 MB
batch. That is far above the ~4 MB knee, so **range-skipped ingestion from S3 would pay close to
its byte fraction.** Clustering matters here for the same reason it matters everywhere else in
this project (§3.13): it is what turns scattered survivors into long runs.

**Caveats.** (a) **The ~0.99 GB/s plateau was the PYTHON PROBE, not the network** — see §7.8;
curl on the same instance reached 2.85 GB/s. Every absolute throughput above is therefore a floor,
and the *relative* comparisons (all arms equally interpreter-bound at equal concurrency) are what
carry. (b) The 64 MB and "1 contiguous run" rows in the raw output are concurrency-starved
artifacts of a fixed byte budget (4 and 1 requests respectively), not findings. (c) One instance,
one region, one object, one sample.

### 7.8 The first ceiling was the measuring instrument (2026-09-10)

The python probe flattened at 0.99 GB/s regardless of concurrency (64 and 128 threads identical) and
§7.6 first attributed that to the instance NIC. It was the **GIL**: every response body is copied
through the interpreter. `curl --parallel` on the same instance, same object, same ranges:

| parallelism | 16 MB ranges |
|---|---|
| 32 | 2.21 GB/s (17.7 Gb/s) |
| 64 | **2.84 GB/s (22.7 Gb/s)** |
| 128 | 2.85 GB/s |
| 256 | 2.63 GB/s |

**2.9x the python ceiling**, saturating at 64-way. A flat ceiling that ignores added concurrency is
the signature to watch for — it looks exactly like a network cap.

**Repeated on an idle instance**, and the ceiling did not move: 2.65 / 2.70 / 2.51 / 2.63 GB/s at
32 / 64 / 128 / 256-way, against 2.21 / 2.84 / 2.85 / 2.63 busy. Same plateau either way, and it
saturates by 32-way. **So contention was not the limit** — ~21 Gb/s is the real behaviour of this
path, and the ~2.4x gap to the instance's rated 50 Gb/s is structural.

The prime suspect is that **one object is one S3 prefix**: per-prefix throughput is the classic way
to leave object-store bandwidth unclaimed, and a real scan spreads reads over many files. Until
someone runs the multi-object variant, treat 2.6 GB/s as this path's number rather than the
machine's.

What DID move with load is the per-request cost (0.29 ms idle vs 0.76 ms busy), which is why §7.7
records a range rather than a constant.

What this does NOT change: fragmentation costs request count, not skipped bytes (§7.6). That came
from p50 latency FALLING as ranges shrink and from the model needing no extents term — both
latency facts, independent of where the bandwidth ceiling sits. curl reproduces the same shape at
2.9x the bandwidth: 4 MB ranges cost +36% against 16 MB at equal volume and parallelism.

## 7.5 Physical layout: keep the metadata segregated and scannable on its own

The index is only useful if it can be read *without* reading the data it describes. That is the
whole premise of skipping a fetch: read metadata, decide, then fetch only what survived. Neither
of the two layouts we have today satisfies it.

### 7.5.1 What is wrong today

**On disk, metadata is interleaved with bulk data.** The `.hpln` payload is "all buffer bytes
concatenated in write order" (`api/compressed_table_io.hpp:40`), and `payload_offset` is a running
cursor assigned leaf by leaf (`src/api/compressed_table_io.cpp:755`). So a column's small metadata
channels sit immediately before/after its multi-gigabyte `packed` buffer, and the next column's
metadata is a payload-length away. Reading just the metadata for a table means one small scattered
read per column per channel — the worst possible access pattern over a network, and not much
better on disk.

**In memory, the sidecar is fragmented.** `pinned_entry::group_bounds` is
`vector<vector<packed_column_bounds>>`, and each `packed_column_bounds` owns three more vectors —
so roughly `3 × n_columns × n_chunks` separate allocations, chunk-major. Evaluating one column's
filter walks a pointer chase across the whole table, and there is no single buffer to hand to a
GPU kernel or to copy H2D in one go.

### 7.5.2 The layout we want

**One contiguous metadata region per table, column-major over groups.**

- **Segregated**: metadata occupies its own region, addressed by a small directory, with zero bulk
  data interleaved. A reader can fetch the entire region in **one** sequential read before touching
  any payload — the property everything else here depends on.
- **Column-major**: all groups of column *c* contiguous, then column *c+1*. A query filters on one
  to three columns; column-major means those are a few large sequential runs and the other columns
  are never touched at all. Chunk-major (today's shape) interleaves columns and forces a strided
  walk over the whole thing.
- **Parallel typed arrays within a column**: `mins[]`, then `maxs[]`, then the validity bitmap —
  the `packed_column_bounds` shape, but as spans into one arena rather than owning vectors. This is
  also exactly the shape a GPU evaluator wants.
- **Small directory**: per (column, chunk), the offset and group count into the region, plus the
  column's type, signedness and `group_rows`. Kilobytes, read first, tells you which slice to read
  or which device pointer to hand a kernel.

Concretely at SF1000 lineitem, G = 8: 732k groups × 16 B = **11.7 MB contiguous per column**, ~170 MB
for all sixteen. One column's bounds are a single 11.7 MB sequential read — one `GET`, not 732k
lookups.

### 7.5.3 Why segregation buys more than tidiness

- **It is what makes the fetch skip possible at all.** Read metadata → decide → fetch surviving
  ranges. If metadata is interleaved, "read the metadata" already means touching the payload
  region, and the ordering the whole scheme needs collapses.
- **It survives spilling independently.** A contiguous region can stay device-resident while the
  payload spills to host or disk. A fragmented sidecar interleaved with payload cannot be pinned
  separately from what it describes.
- **It is one H2D copy and one kernel launch.** ~700k cells is GPU-shaped work; a pointer chase is
  not.
- **It is independently cacheable.** Over object storage the metadata region is small, hot, and
  read by every query against the table — exactly the thing a cache should hold and the payload
  should not evict.

### 7.5.4 File-format choice: separate section, or separate object?

Two shapes, and the answer differs by backend:

| | one object, separate section | separate sidecar object/file |
|---|---|---|
| reads to get metadata | 2 (footer, then the region) — or 1 if the directory rides in the footer | 1 |
| atomicity | trivially consistent with the data | needs a version/identity check against the payload |
| cacheable independently | by byte range | naturally, as its own object |
| works for a pinned entry | n/a | n/a |

**Recommendation: a separate, contiguous section within the file, with its directory in the footer
so the region is reachable in one read after the footer.** It keeps data and index atomically
consistent — which matters because a stale index is a *correctness* bug, not a performance one —
and byte-range caching is enough to get the independent-caching benefit. A separate object is worth
revisiting only if metadata turns out to be re-read far more often than footers.

The `.hpln` header is self-describing per leaf, so this is additive: a new section plus a directory,
old files still readable, and the existing `payload_fetch_fn(offset, size, dst, stream)` callback
already expresses "fetch this byte range" without change.

### 7.5.5 What this implies for the code as it stands

**Done in memory (`56db39e9`):** `group_bounds_arena` holds every (column, chunk, group) in one
allocation, column-major, with `packed_column_bounds` a non-owning view of a slice. The layout
assertions caught a heap overflow in the first packing (`total_groups` summed over chunks but not
columns), which is the kind of bug only a contiguity test finds.

**Still to do:** a `rmm::device_buffer` mirror of the same bytes, so the layout is identical host
and device and a GPU evaluator needs no repacking; and the persisted form (§7.5.4).

## 8. Where the index should live: device, host, spilled

### 8.1 Keep the index in device memory even when the data is pinned to host — yes

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

### 8.2 When data spills to host

Nothing changes for the index: it stays on device. The data moving to host makes the index *more*
valuable, not less, because the cost of a wrong "do not skip" decision goes up by the H2D transfer.
The downgrade executor (`src/downgrade/`, `src/creator/`,
`docs/super-sirius/memory-management.md`) should treat the index as **non-spillable** at its natural
size — it is in the same class as plan metadata, not payload.

Implication for the spill policy: when a table is demoted GPU→host, its index must survive the
demotion. Since `docs/super-sirius/scan.md:283-288` says statless entries are not retrofittable,
losing the index on demotion would be permanent for that entry.

### 8.3 When data spills to disk

Here the index should be **kept in device memory and additionally persisted**. Two distinct roles:

- **Resident copy (device):** answers "do I need to read this chunk back at all". This is the whole
  value — a disk round trip is orders of magnitude more expensive than the ~1:850 index, so the
  break-even is not close.
- **Persisted copy (with the payload):** stored as its own contiguous section, not interleaved —
  see the layout section above for why that is load-bearing rather than cosmetic. The `.hpln`
  format is self-describing per leaf
  (`.../api/compressed_table_io.hpp:14-39`) and `payload_offset` is already per-buffer, so both the
  min/max index and the §6.1 group→byte table serialize alongside the data as ordinary buffers,
  with no format surgery.

The persisted copy is also what enables **partial re-read**: `read_compressed_table_subset_from_memory`
(`compressed_table_io.hpp:137-143`) already does column-granular partial fetch via `payload_offset`;
adding group-range granularity — using the §6.1 table — would let a spilled table read back only
surviving groups. That is
the natural follow-on and the one place `range_slice` (`DECODE_PUSHDOWN_PLAN.md:722-723`) becomes
necessary.

### 8.4 Summary table

| Data location | Index location | Compressed? | Why |
|---|---|---|---|
| GPU pin (raw or compressed) | device | no | decision is on-device, cost ~0.1% of payload |
| Host pin | **device** | no | lets us skip the H2D fetch itself — the dominant host-tier cost |
| Spilled to host | **device**, non-spillable | no | wrong decisions now cost a transfer; not retrofittable if lost |
| Spilled to disk | device + persisted with payload | persisted copy: optional | disk round trip dwarfs the index; persisted copy enables partial re-read |

---

## 9. Proposed order of work — prove value before committing

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

**⚠ This projection was measured at SF1000 on 2026-09-08 and is wrong: the real number is −0.4%
(§3.8).** The proxy failed not because the fraction-of-table reasoning was wrong but because it
assumed pin chunks inherit the file's row order, and they do not — the scan's row-group coalescer
interleaves across the file set, so SF1000 pin chunks span most of the key range. SF100 escaped
this because six files and 512 MB batches rarely spanned much of the key space. Read §3.8 before
trusting anything below.

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

**2a. Capture — DONE (`b672cb62`).**
`compute_pinned_group_stats(chunk, column_types, group_rows, …)` landed alongside the per-chunk
function, using one `cudf::segmented_reduce` per column over a fixed-stride offsets column (the
exact shape benchmarked in §4.3.0), returning a group-major `chunk_group_stats`. Same allowlist,
same "null cell never prunes" contract, and deliberately the same null precision as the coarse
capture. 25 cases / 243 assertions cover the edge cases that matter: short final group, all-null
group, partly-null group, off-allowlist type, `group_rows == 0`, shape mismatch, and
"one group per chunk reproduces the whole-chunk capture".

**The storage shape 2a emits does not scale — measured, and now fixed in 2b (`56295d4e`).** `chunk_group_stats`
holds one `duckdb::unique_ptr<duckdb::BaseStatistics>` per (group, column), mirroring the coarse
sidecar. Timed on this box (`[.][pinned_chunk_stats][bench]`, 8 M-row column):

- capture including host-side `BaseStatistics` construction: **~147 ns per group cell**
- `chunk_provably_empty` (the plan-time probe): **83 ns per group cell**

At 32 chunks those are irrelevant (32 × 83 ns = 2.7 µs). At SF1000 lineitem with G=8 there are
**732,000 groups**, so one filter column costs ~61 ms of plan time *per query*, and a three-column
predicate ~180 ms — against a 5.8 s suite. Pruning chunk-first and descending only into survivors
helps but does not fix an 83 ns primitive.

> **So 2b must not store one `BaseStatistics` per group cell.** It needs a packed columnar form —
> typed parallel min/max arrays per column, evaluated vectorized or on the GPU — with the DuckDB
> `TableFilter` lowered once per query into a typed bound rather than re-dispatched per cell.
> The 2a function stays useful as the *capture* (it is correct, tested, and GPU-side cheap); its
> output type is what has to change.

**Done (`56295d4e`):** `packed_column_bounds` (parallel min/max/valid arrays) plus
`lowered_bound_filter` (a `TableFilter` lowered once into a flat node array, evaluated with bounds
arithmetic). **79.9 → 3.4 ns/cell, 23×** — ~2.5 ms per filter column for SF1000 lineitem instead of
~61 ms. Correctness is defended two ways: `lower()` gates on the same `filter_safe_for_stats`
allowlist the `BaseStatistics` path uses, so the two cannot disagree about *which* filters are
evaluable; and a cross-check test runs both evaluators over 336 (filter, range) pairs — every
admitted shape, ranges straddling each constant, with and without nulls — requiring zero
disagreements. Still host-side; a GPU evaluator remains possible if 3.4 ns/cell ever matters.

This also revises §8.1: keeping the index device-resident is right, but the load-bearing reason is
**evaluation throughput**, not avoiding an H2D round trip. 732,000 group cells is a GPU-shaped
problem, not a host-loop-shaped one.

Still to do in 2a: the packed representation above, and making G configurable (default 8).

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
4. **Range-skipped fetch on a host-tier pin (§6).** The highest-value untested idea: host-tier
   compressed loses 7.0% today entirely on fetch cost, and the byte ranges are exactly computable.
   Gates the simpatico ingestion format (§7) — same mechanism, no new format.
5. **Clustering as a first-class pin option.** None of the above pays on TPC-H as it sits (§3.1);
   this is what converts 0% into ~34–38%. Arguably the real project, with the metadata as what
   makes it usable.

## 10. Open questions / log

- **2026-09-09** — **Range-skipped fetch runs end to end on a host-tier pin (§6.5).** The scan
  hands a compressed chunk its surviving 1024-row decode chunks, `build_chunk_subset_header`
  synthesizes a header for them, and the converter gathers only those bytes out of the pinned
  payload. **−1.03%** at SF1000 / host / 2 GB (9.6845 → 9.5848 s, best-of-3), 22/22 byte-exact on
  clustered SF100 at both tiers. Gate: `SIRIUS_EXP_CHUNK_SUBSET_FETCH` (on by default, `0` = off).
  Two things had to be fixed to get there, both in §6.5: a table cannot mix a compacted column
  with a whole one (§6.5.4), and the shipped row-range serving duplicated rows on three of the
  four serve paths (§6.5.5, a real wrong-answer bug).
- **2026-09-09** — **Full ladder measured on unsorted SF1000 (§3.13), and it reorders the
  project's own story.** Without clustering, zone maps + group index + fetch skip cost **+1.39%**
  — they prune nothing, so they are overhead. With clustering the same chain is **−8.88%**. The
  attribution is not what the project assumed: clustering alone −2.2%, the group index +0.2% (it
  locates surviving rows but moves nothing), the fetch skip **−7.0%**. All three are required; none
  pays alone.
- **2026-09-09** — **Ratio beats decode throughput when picking plans on this path (§3.14).** A
  host-tier cost model (`1/(ratio×link) + 1/decode`) predicted throughput-weighted picks would be
  34–40% cheaper per table; they measured **2.05% slower**, twice. The decode hides behind 18 scan
  threads while the host→device copy does not, so the committed "max ratio with decode ≥ 250 GB/s"
  floor is closer to right. Also: enabling the four `*_disabled.txt` plans is worth −0.27%, i.e.
  nothing — no case for re-enabling them.
- **2026-09-09** — **The clustered layout needs no new compression plans (§5.5).** Clustering is
  footprint-neutral overall (lineitem −2.4%, orders +1.2%); it only moves bytes from the date
  columns to the order keys, whose `delta -> bitpack` collapses 12.39x → 2.74x. Re-exploring
  returns plain `bitpack` — the loss is intrinsic — and the re-picked plans measure **+1.14%** on
  the suite, inside noise. Keep the committed plans. Also: **q17's +0.080 s was noise** (a
  same-config repeat gives +0.002 s); q12's +0.03 s is the one real regression.
- **2026-09-09** — **Pin-time clustering landed, and it is the result that makes the project pay
  in production (§5.4).** `pin_table(..., cluster_by=[...])` sorts each chunk as it is pinned:
  **−8.57%** at SF1000 host/2 GB on the UNSORTED dataset for **+1.6%** pin cost, 22/22 byte-exact.
  It beats the pre-sorted dataset (8.92 s vs 9.54 s) because sorting AFTER the coalescer is
  strictly better than sorting the files before it. Confirms §5.2 exactly: at SF10 a one-month
  predicate drops 59.1M of 60.0M rows through the group index while pruning zero chunks coarsely.
- **2026-09-09** — **Dictionary-encoded strings are now chunk-addressable (§6.5.6).** The blocker
  was a conflation: a dictionary's keys are fetched whole like an lz4 payload, but unlike it they
  do not depend on WHICH rows are served. `ChannelLayout::column_state` names that, and the
  builder marks whole subtrees reached through such a channel so compressed keys
  (`dictionary.keys_offsets -> bitpack`) stop looking like a refusal. q1 and q4 now engage; the
  suite goes to **−1.33%** at SF1000/host/2 GB (pooled over two A/B pairs), 22/22 byte-exact.
  Only `str_split` columns (`l_shipmode`) still force a whole-chunk serve, and that is a value
  rewrite rather than a byte gather — §6.5.6 says why.
- **2026-09-09** — **The simpatico host tests are NOT built by `pixi run make`.** Three runs of
  `test_chunk_subset_header` reported PASS from a stale binary while the new cases had never been
  compiled. Build them explicitly: `pixi run cmake --build build/release --target all` (or
  `--target test_chunk_subset_header`). A negative control — revert the change, expect the test to
  fail — is what caught it.
- **2026-09-09** — **The fetch skip is capped by string columns, not by the mechanism.** It
  engages on q6/q14 and refuses on q1/q4/q12 because each reads one `dictionary`- or
  `str_split`-rooted column, and one unaddressable column in the projection sends the whole chunk.
  Coalescing, meanwhile, is a non-problem on clustered data: 6,152 surviving decode chunks
  gathered into 16 ranges. **Next: teach `supports_chunk_subset` the string trees.**
- **2026-09-07** — project opened. Investigation and all measurements in §3–§6 done; nothing
  implemented, nothing benchmarked end-to-end.
- **2026-09-07** — plan audit corrected the W1 story: `chunk_min` is a value-domain minimum only
  for `input -> bitpack` roots. `delta -> bitpack` (`l_orderkey`, `o_orderkey`) and
  `str_split -> bitpack(offsets)` (`l_shipmode`) give minima of deltas and of string lengths
  respectively. `dictionary -> bitpack` *is* usable (cuDF dictionary keys are sorted). Min/max is
  therefore stored **out-of-band**, unconditionally. Granularity settled at a configurable group of
  G simpatico chunks, default G = 8 (8,192 rows); see §4.3.
- **2026-09-08** — **The group index runs end to end (§3.12).** The capture was inert (nothing
  populated `pinned_entry::group_bounds`); wiring it in gives **−2.20%** at SF1000 host tier on top
  of whole-chunk pruning, dropping 44.6e9 rows. Confirms a G = 8 index sees through the
  coalescer's interleaving with no pin-time clustering. This is decode/downstream saving only —
  the chunks are still fetched whole.
- **2026-09-08** — **Host tier measured (§3.11): the project has a target.** Same sweep with
  `--pin host` gives **−7.3% at 2 GB** and −1.7% at 8 GB, against −2.0%/−0.4% on the GPU tier.
  2 GB-ON (9.678 s) is the best host configuration measured. Confirms the §6 fetch hypothesis
  empirically — q6 is 0.041 s GPU-pinned vs 0.114 s host-pinned, and that 2.8× is what skipping
  eats into. Phase 2 at the production 8 GB batch is plausibly worth −3% to −5% here.
- **2026-09-08** — **Ceiling measured (§3.9/§3.10).** A G=8 index sees through the interleaving
  for free (groups are 1/128 of a row group, so they inherit a 0.44-day span; 36.54% of rows
  prunable at row-group granularity) — so Phase 2 is an *alternative* to pin-time sorting, not
  gated on it. But q6, the most scan-dominated TPC-H query, is 0.6% of the suite, and the whole
  envelope for chunk skipping on a GPU-resident compressed pin is ≈2.4% at 8 GB. Phase 2 is worth
  ≈0.9%. The project's real target is the host tier / spilling, not the GPU-resident config.
- **2026-09-08** — **SF1000 validation done, and it overturns the Phase 1 projection.** Real number
  is **−0.4%** at the production 8 GB batch, not the projected ~4%. Root cause found (§3.8): the
  scan's row-group coalescer interleaves row groups across the file set, so pin chunks span most of
  the key range even on a globally sorted table — a one-day predicate prunes only 13/37 chunks.
  Sorting the dataset does not cluster the pin; clustering has to happen at pin time.
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
- **Open (§6.5):** the operator-level "metadata channel vs bulk channel" split — the one
  requirement of range-skipped fetch that the earlier plan did not anticipate.
- **Answered (§7.6, 2026-09-10):** S3 rewards reading less, proportionally — 50% of an object
  takes 0.500x the time, 25% takes 0.259x. There is no striping penalty: fragmentation costs
  request count only, and `throughput = min(cap, concurrency x range / RTT)` explains every point.
  **Open:** everything above the ~1 GB/s NIC cap of the test instance.
- **Open (§6.1):** the group→byte table is new format surface. Confirm it can be added as an
  ordinary per-leaf buffer (the `.hpln` header is self-describing, so this should be additive and
  leave old files readable), and decide whether the encoder emits it always or only when asked.
- **Open (§6.1):** should the compression explorer learn about fetch-skippability as a third
  objective alongside ratio and decode throughput? Today a ratio-optimal plan can silently pick a
  codec whose buffers cannot be group-addressed.
- **Answered (§7.7, 2026-09-10):** bridge gaps under ~256 KB (one request amortises to 200–460 KB
  of transfer at concurrency 64); coalesce to ~4–16 MB runs; keep `concurrency x range >= cap x
  RTT`. **Open:** wiring those constants into `align_and_coalesce`'s caller.
- **Answered (§6):** range-skipped *fetch* on a host-tier pin is built, measured (−1.03% at
  SF1000/2 GB) and gated. Surviving groups DO merge into few large copies on clustered data (16
  ranges for 6,152 chunks). **Open:** the string-column ceiling above is what it now costs.
- **Open:** the §3/§5 pruning numbers are still *rows/bytes not decoded*, not query time. Need one end-to-end datapoint —
  cheapest is W2 on a GPU-compressed lineitem pin with an artificially clustered SF100.
- **Answered (§5.1, §5.4):** sorting 6e9 rows at pin time costs ~2.3 s on the GPU (measured
  +1.6% of the pin wall), and the sort goes in `materialize_pin_batches` immediately after
  materialization, before the zone-map capture. **Open:** how the sort key gets chosen — it is an
  explicit `cluster_by` argument today.
- **Answered (§5.5):** clustering is roughly footprint-neutral — lineitem −2.4%, orders +1.2% —
  because the only columns that change are those correlated with the sort key. The orderkey loss
  (12.39x → 2.74x) is intrinsic and no re-explored plan recovers it; the re-picked plan file
  measures +1.14% on the suite, i.e. no better. Keep the committed plans.
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
