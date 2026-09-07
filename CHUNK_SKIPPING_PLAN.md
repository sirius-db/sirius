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

### W1 — Read the metadata that is already there (no format change)

Expose `chunk_min` / `chunk_bits` / `references` as a *zone map* rather than only as decode inputs.
Two consumers, cheapest first:

- **In-kernel early-out.** In the ballot emitters
  (`src/compression/simpatico_codegen/src/decode/jit/renderer.cpp:~532-544`, `emit_generic_mask_out` `:885+`) the block already loads `chunk_min` and `chunk_bits` and already has `pred_lo`/`pred_hi`
  as kernel parameters. Add: if `[chunk_min, chunk_min + 2^bits − 1]` is disjoint from the
  predicate, write 32 zero mask words and return; if fully contained, write all-ones and return.
  Saves the per-chunk unpack + ballot. **No new metadata, no format change, no host round trip.**
- **Host-side chunk pre-selection.** Walk the plan tree (`describe()` /
  `PlanNode::channels`, `leaf_desc::buffers`) to get device pointers to the `chunk_min`/`chunk_bits`
  arrays, run a small kernel producing a survivor chunk list, and feed that list to
  `chunk_csr` — which already skips absent chunks. This is the version that saves the
  *decode launch* rather than just the ballot.

Limitation: the max is a bound, not exact — `chunk_min + 2^bits − 1` over-estimates by up to
2× the true range. Sound (never prunes a matching chunk), just less selective.

### W2 — Plumb the existing sidecar into the GPU compressed pin path

Purely mechanical, unblocks pin-chunk pruning for the winning configuration:
`src/pin_table.cpp:715` (remove the forced-off), `src/include/pin_table.hpp:258-270` (add
`chunk_stats` to `device_pin_result`), `src/include/scan_manager/sirius_scan_manager.hpp:602`
(`insert_pinned_entry_device` takes them), `src/sirius_extension.cpp:1539` (pass `true`).
Note `docs/super-sirius/scan.md:283-288`: statless entries are **not** retrofittable, so this must
be captured at pin time.

### W3 — Exact per-chunk max (cheap format addition)

Add a `chunk_max` channel next to `chunk_min`: same `BlockReduce` already running in
`emit_bitpack`/`emit_for`, one `add_buffer` at `.../src/encode/jit/renderer.cpp:1053-1055`, one
registry entry. The `.hpln` format is self-describing per leaf
(`.../api/compressed_table_io.hpp:14-39`) so this is additive and old files stay readable.
Cost: `elem_size × num_chunks` — see §4.

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
(`l_returnflag`, `l_shipmode`, `l_shipinstruct`, `c_mktsegment`, `p_brand`, …) carry **no min/max
statistics at all** in these files and there are **zero bloom filters**.

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

This is W4, and it is the piece with no parquet equivalent: these columns have no parquet
statistics at all (§3.1), so Simpatico can prune where the parquet reader structurally cannot.

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

### 4.3 Recommendation

- **Reuse 1024 for the free metadata (W1).** `chunk_min`/`chunk_bits` cost *nothing extra* — they
  are already in the payload and already loaded by the decode kernel. At that price the granularity
  question does not arise, and 1024 is the only granularity the decode path can act on anyway.
- **Build any *new* sidecar (W3/W4) at a coarser stride — 8K or 16K rows, i.e. one entry per 8/16
  simpatico chunks.** §4.1 says this loses ~0.5 percentage points of pruning; §4.2 says it costs
  8–16× less. At 8K a full 16-column lineitem sidecar is 0.17 GB against 93 GB of payload (0.18%),
  which is small enough not to argue about. 1024-row *new* metadata would be 1.36 GB — 1.5% of the
  pin — for no measured gain.
  A coarse sidecar still composes with the fine path: a surviving 8K super-chunk hands its 8
  constituent chunk ids to `chunk_csr`, which then applies the free in-kernel test.
- **Scope the sidecar to filterable columns.** Columns that never appear in a `TableFilterSet`
  do not need entries. This is a 1.5–4× further reduction and is knowable at pin time only
  heuristically — start with "all columns of a supported type", revisit if the size matters.

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

## 5. Where the index should live: device, host, spilled

### 5.1 Keep the index in device memory even when the data is pinned to host — yes

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

### 5.2 When data spills to host

Nothing changes for the index: it stays on device. The data moving to host makes the index *more*
valuable, not less, because the cost of a wrong "do not skip" decision goes up by the H2D transfer.
The downgrade executor (`src/downgrade/`, `src/creator/`,
`docs/super-sirius/memory-management.md`) should treat the index as **non-spillable** at its natural
size — it is in the same class as plan metadata, not payload.

Implication for the spill policy: when a table is demoted GPU→host, its index must survive the
demotion. Since `docs/super-sirius/scan.md:283-288` says statless entries are not retrofittable,
losing the index on demotion would be permanent for that entry.

### 5.3 When data spills to disk

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

### 5.4 Summary table

| Data location | Index location | Compressed? | Why |
|---|---|---|---|
| GPU pin (raw or compressed) | device | no | decision is on-device, cost ~0.1% of payload |
| Host pin | **device** | no | lets us skip the H2D fetch itself — the dominant host-tier cost |
| Spilled to host | **device**, non-spillable | no | wrong decisions now cost a transfer; not retrofittable if lost |
| Spilled to disk | device + persisted with payload | persisted copy: optional | disk round trip dwarfs the index; persisted copy enables partial re-read |

---

## 6. Proposed order of work

1. **W2** — plumb `capture_chunk_stats` through the device pin path. Mechanical, no format change,
   immediately gives GPU-tier compressed pins the pin-chunk pruning that uncompressed pins already
   have. Measurable on its own.
2. **W1 in-kernel early-out** — the free bitpack zone map as a ballot short-circuit. No format
   change, no new memory, self-contained in the decode renderer.
3. **W1 host-side chunk pre-selection** — survivor chunk lists into `chunk_csr`. This is the first
   version that skips *launches*.
4. **W3** — exact `chunk_max`, if (2)/(3) show the `2^bits` bound is costing real selectivity.
5. **W4 dictionary present-value bitmaps** — the only path to pruning on low-cardinality string
   equality (§3.4), and the capability parquet cannot match.
6. **W4 null counts** — unblocks compressed late-materialization, which is a separate win.
7. **Clustering.** None of the above pays on TPC-H as it sits (§3.1). A pin-time sort/clustering
   option is what converts 0% into ~34–38% (§3.3). This is arguably the real project; the metadata
   is what makes it *usable*.

---

## 7. Open questions / log

- **2026-09-07** — project opened. Investigation and all measurements in §3–§5 done; nothing
  implemented, nothing benchmarked end-to-end.
- **Open:** all §3 numbers are *bytes-not-decoded*, not query time. Need one end-to-end datapoint —
  cheapest is W2 on a GPU-compressed lineitem pin with an artificially clustered SF100.
- **Open:** is there a pin-time clustering hook at all? `pin_table` packs whole 122,880-row DuckDB
  row groups (`src/pin_table.cpp:126`); a sort would have to happen before or during that.
  Cost of sorting 6e9 rows at pin time vs. the 34–38% it unlocks is unmeasured.
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
