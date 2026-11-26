# Compressed pinning: putting hot columns on the GPU

`pin_table` attempts Simpatico compression by default on either host or GPU tiers,
without requiring a plan file:

```sql
CALL pin_table('/data/lineitem/*.parquet', tier => 'gpu', name => 'lineitem');
-- Opt out for one pin:
-- CALL pin_table(..., compression => false);
```

An optional table plan overrides automatic selection:

```sql
SET pin_table_input_compression_plan_dir = '/path/to/plans';  -- <table>.txt per table
CALL pin_table('/data/lineitem/*.parquet',
               tier => 'gpu',
               name => 'lineitem',
               cols => [...],
               compression => true);
```

When no readable, nonempty matching plan is found, a GPU-tier pin smaller than
`pin_table_compression_min_table_device_fraction` (default 0.05) of the memory of the GPUs it
pins to is stored uncompressed. Such a pin frees little GPU memory, and its scans would still
pay for decoding. The size is the pin's uncompressed bytes: the first materialized chunk's
bytes per row, after narrowing, times the table's row count. If the row count can't be read,
the pin is not gated. Host-tier pins are never gated, because every scan copies them over the
host-to-device link and fewer bytes always shorten that copy. On SF1000 with GB300, the default
compresses the `lineitem` and `orders` pins of most queries and leaves `part`, `partsupp`,
`customer` and `supplier` raw. Compressing those as well slowed q2, q11, q14, q16, q19 and q20
by 4 to 27 ms each on the GPU tier. On the host tier, compressing every table is 0.26 s faster
over the suite than keeping the small ones raw.

Otherwise each materialized chunk selects its plans automatically. The policy is close to
[spill compression](memory-management.md#spill-compression)'s, but it favors plans a filtered
scan can decode in place. It samples up to 65,536 values. Integer-valued columns choose
between bitpacking and delta-bitpacking, and delta-bitpacking needs over 10% estimated size
savings. Delta-RLE-bitpacking, which every scan decodes in full, is chosen only when it saves
at least a quarter of the column's uncompressed size more than the better of those two. A
sorted key whose differences already pack to almost nothing does not qualify. Strings with
at least 4,096 rows use a dictionary with bitpacked indices when at most 10% of the sample is
distinct. Otherwise they keep their characters as is beside delta-bitpacked offsets, which a
filtered scan gathers for surviving rows only. Other eligible fixed-width columns use
bitpacking; short strings and unsupported types are stored as is. Selection runs after
carrier narrowing and in pinned-column order.

Each explicit plan file carries one `---`-separated DSL
block per full-table column in schema order; a pin of a column subset selects the matching
blocks (`select_plan_blocks`). Each pin reads its plan from the calling connection's
`pin_table_input_compression_plan_dir`; plans are not stored in a process-wide registry.
Scans of a compressed entry decompress the projected columns at task-prepare time via the
`compressed_host_representation` / `compressed_device_blob` → `gpu_table_representation`
converters. With per-thread reservation tracking, multi-column decodes use up to 4
column-parallel streams (the limit is not configurable); otherwise they use the task stream.
Simpatico leases the parallel streams exclusively from its per-device cuCascade pool.
The scan orders their input reads after writes on the task stream, waits for the decode
before returning, and rebinds the decoded columns to the task stream (`decompress_chunk`
in `src/compression/compressed_scan.cpp`).

A GPU-tier compressed chunk is decoded in place on the GPU that holds it. When query admission
(`topology.gpus_per_query` or `admission_bytes_per_gpu`) assigns its scan to another GPU, the
converter copies the compressed buffers of the projected columns to that GPU and decodes them
there, as it does for a host-tier compressed chunk. The copy is a peer copy where the two GPUs'
pools have peer access and is staged through pinned host memory otherwise
(`memory::enqueue_device_copy`). The pinned chunk stays on its GPU and is only read. Such a
scan pays a transfer of the compressed bytes per query, and its reservation covers the copied
compressed bytes together with the decoded output. When that pinned host memory runs out, the
task is retried without growing its GPU reservation (see
[Pipeline Execution](pipeline-execution.md#reschedule-handling-oom-and-cuda-launch-failures)).

## Decode-time equality pushdown

A string equality filter over a compressed pinned column can be evaluated *inside* the
decode instead of re-expressed as a post-decode comparison. At `prepare_for_query`, the
scan manager derives a `sirius::decode_equality_pushdown` from the query's pushed-down
filters and attaches it to the entry's scan (`set_equality_pushdown()`); the converter
translates it into a `simpatico::decode_predicate`, so the decoder emits the match result
directly. Interfaces: `src/compression/compressed_representation.hpp`
(`decode_equality_pushdown`), `src/compression/compression_converters.cpp` (predicate
translation), threaded through `sirius_scan_manager::prepare_for_query`.

The `pin_table_compression` setting defaults to true; an explicit `compression` argument
always wins. Chunks below 1 MiB remain raw by default. After encoding, compression is
kept by default only when header plus payload is at most 75% of the original device footprint.
All-identity automatic plans skip encoding. Encoding failures leave subsequent chunks raw;
automatic GPU pins retain their source until staging succeeds so staging failures can
also fall back. An explicit plan that does not cover the pinned columns is rejected with
a warning and the pin stays raw; it is not silently replaced by an automatic plan.
The `parquet` pin tier continues to cache file bytes without this decode/compress step.

When pinned zone-map capture is enabled, supported native min/max statistics are
retained for compressed chunks on both tiers, so entire chunks can be pruned before decode.

Fully raw automatic GPU results use ordinary column-major storage and can merge compatible
raw columns. If the previous GPU pin stored chunks in compressed form, a fully raw re-pin
replaces that entry instead of attempting a column merge. GPU pins containing compressed
chunks also replace the previous entry on re-pin: pin the full intended column set in one
call, or opt out of compression when incremental column merging is required.

The current late-materialization installer declines compressed origins because their
null counts cannot be checked without decoding. Those queries use ordinary decoding and
materialization instead. Use `compression => false` when evaluating raw-pin deferral;
compression does not imply that late materialization will be admitted.

## Which tier to compress on (GB300, TPC-H SF1000, 22-query hot suite)

The tier decides whether compression helps at all. Measured against a 20.74 s all-host
uncompressed baseline (main `c4e8a10b`, pipeline 4, host-pinned, hot = min of iters 1–2 —
note the shipped TPC-H plans were rewritten onto `bitpack` after this measurement, so
per-column throughput/ratio figures below predate the current plans):

| Config | Hot total | vs baseline |
|---|---|---|
| All host-pinned, compressed | 22.20 s | **+7.0%** |
| Mixed tier, no compression (dims GPU raw) | 20.48 s | -1.2% |
| Dims GPU raw + **lineitem GPU compressed** | 17.05 s | **-17.8%** |
| … + **orders GPU compressed** (`o_comment` → identity) | **16.66 s** | **-19.7%** |

Why the sign flips between tiers:

- **Host-tier compressed loses.** Every scan batch pays payload H2D fetch → stream sync →
  decode on the task's critical path. Short queries cannot hide the added prepare latency,
  even for columns that decode at 1700 GB/s. Raw host pins stream at link rate
  (~370 GB/s on GB300 C2C) with no SM work and pipeline cleanly.
- **GPU-tier compressed wins.** The payload is already device-resident, so only the decode
  remains — 500–1700 GB/s for well-chosen plans, replacing the 370 GB/s link entirely.
  Compression is also what makes residency *fit*: the seven hottest lineitem columns are
  264 GB raw but ~93 GB compressed, carrying ~2.4 TB of the suite's 2.9 TB of pin traffic.

Rules of thumb:

- GPU-pin the highest-traffic tables compressed (lineitem, orders); leave tables whose
  bytes are dominated by an incompressible column (partsupp / `ps_comment`) host-raw.
- Pick plans whose **decompress throughput exceeds the host→GPU link** (~370 GB/s on
  GB300) so decode is never slower than the copy it replaces. Numerics, dates and keys
  clear this easily (bitpack/ans/delta at 580–1700 GB/s); low-cardinality string
  dictionaries are gather-bound (~250 GB/s for 8-char values) and long-text snappy sits
  at ~230–270 GB/s.
- Plan choice is tier-dependent: a slow-decoding, high-ratio plan (e.g. `o_comment`
  snappy, 2.6x @ 260 GB/s) is right for a **host** pin (fewer bytes over the link) but
  wrong for a **GPU** pin — there is no link to save, so use `input -> identity` and let
  the column sit raw on the GPU.
- `ans` materializes byte channels as cudf columns and fails above 2 GiB per batch; use
  nvcomp-backed codecs (snappy) or identity for large text columns. Pin batches are
  `scan_task_batch_size` of the *pinned subset*, so a narrow pin concentrates a column's
  chars in each batch.

All newly written `.hpln` files use version 14, with either nvCOMP 5 or 6. Version 13
introduced five-word bitpack decode guards; version 14 additionally records a payload-format
identifier in each ANS and Cascaded leaf's metadata. These identifiers distinguish pre-v6
and v6 nvCOMP payloads independently of future file-layout versions. Incompatible codec
metadata and older file versions are rejected before fetching or decoding payloads.

With nvCOMP 6, `nvcomp_cascaded` plans remain unavailable because the old configurable Cascaded
options were removed. Its metadata field reserves the distinction needed to support the new
Cascaded API later. nvCOMP 5 builds continue to support the existing Cascaded plans.

## Filtering while decompressing

A scan can hand its row filter to the decompressor instead of evaluating it afterwards. Where
a column's plan allows it the filter is answered from the compressed form, and the surviving
rows come back already compacted — so rows the filter rejects are never fully decoded, and the
scan skips its own filter pass when the decode carried the whole predicate.

When the decode carries the scan's whole filter it returns only the projected columns: a column
the scan reads only for its filter is tested while the selection is built and is never decoded or
gathered. When the projected columns are also the scan's output in order (no reordering, duplicate
or synthesized column), the scan then takes ownership of the batch without copying it. A join
filter the decode applies alongside does not change this: it is optional, so whether it ran on a
batch does not bear on the scan's own filter. With any conjunct of the scan's own filter left to
the scan (an unsupported shape) or on any decline, every column comes back as before.

It is enabled by default. Disable it per connection with
`SET enable_fused_scan_filter = false`; YAML accepts the same name under
`sirius.operator_params`. Enable the decision trace for a connection with:

```sql
SET fused_scan_diagnostics = true;
```

Measured on GB300, TPC-H SF1000, on top of a GPU-pinned compressed configuration: the suite
went 7.866 s -> 6.918 s. The wins are concentrated where a filter is selective and the column
is wide — q12 -35.8% from the string route.

**This is a property of the plan, not of the query.** What a column can do while decoding
follows from the compressor at its root:

| Plan root | What the decode can do |
|---|---|
| `bitpack` (leaf) | evaluates ranges into a selection mask; rejected rows are never unpacked |
| `delta -> bitpack` | the chunk is still reconstructed (a prefix sum is sequential), but only survivors are STORED — saves the full-width write and the downstream compaction |
| `dictionary` with bitpack codes | answers string equality/IN off the key set, and gathers only surviving keys. Wins 2.1-2.6x at EVERY selectivity, since it skips the string materialization round trip |
| `str_split` | reconstructs offsets under the mask and gathers only the survivors' chars |
| anything else (`identity`, entropy-coded, ...) | decodes full width and is compacted by a gather, which is admitted only when few rows survive |

So a plan chosen purely for ratio can cost row-skipping. `input -> identity` is the right GPU-tier
choice for a column nothing filters on (see above), but on a column the workload filters on, a
bitpack-rooted plan additionally buys the mask walk.

Selectivity governs whether compaction is attempted at all, because compacting is only cheaper
than decoding when enough rows are dropped:

- a batch with any full-width column proceeds only below `fused_scan_full_route_max_selectivity`
  (0.10), or below `fused_scan_max_selectivity` when it also has a dictionary output, whose
  savings pay for the full-width columns' gather;
- otherwise compaction is given up above `fused_scan_max_selectivity` (0.35), unless a
  dictionary output is present — that route is exempt, since it wins regardless;
- below `fused_scan_index_walk_max_selectivity` (0.15) the decode walks a survivor index list
  instead of the mask bits.

These thresholds and `fused_scan_max_membership_sources` (default 1) accept connection-local
`SET` overrides and YAML startup defaults under `sirius.operator_params`. A query snapshots its
policy, including diagnostics, so later settings changes do not affect work already submitted.

One batch measured unselective predicts the rest of that scan, so the scan stops attempting it
rather than paying per batch.

Use `SET <name> = <value>` to change these settings for subsequent queries on the current
connection, and `RESET <name>` to restore the startup default, including any YAML override.
Inspect their current values in `duckdb_settings()`. Each query keeps the policy captured when
it starts; later changes do not alter work already submitted.

The decision trace shows what happened: the plan per column (route, bounds, what was dropped
and why), then per batch the sources in the order they ran, the enumeration chosen, the
survivor count and whether the batch came back needing no further filtering. Enable
`fused_scan_diagnostics` to emit these decisions at INFO level.

### Reporting the rows the decode kept

A decode that compacts can hand back the positions it kept, as an ascending INT32 index list over
the pre-filter batch (`pushdown_outcome::survivor_rows`). It is produced only when the request
asks (`pushdown_request::report_survivors`, set through
`decompression_pushdown_scan::with_survivor_reporting()`), which only a scan carrying a
late-materialization deferral does — that rowid addresses the pinned chunk by position, and a
compacted batch no longer holds the chunk's rows in order.

The answer is not re-derived: the tier-B route already materialized the index list to gather by,
and every other route left the counted mask it balloted, so this either copies a buffer that
exists or expands one (`mask_to_row_indices`). `pushdown_outcome::compacted` reports that rows
were dropped at all — separately from `row_filtered`, which additionally claims the whole filter
was carried — so a consumer can tell "no positions asked for" apart from "no rows dropped".

## Interaction with the downgrade executor

GPU pins permanently occupy part of the downgrade budget, so heavy queries ride closer to
`downgrade_trigger_fraction`. At the defaults (trigger 0.8, stop 0.6) q9 crossed the
trigger and the executor evicted ~42 GB of ~5 GB concat intermediates in one episode —
re-uploaded ~450 ms later, an ~84 GB round trip per iteration (visible in quent as
GPU→HOST `InTransit` bursts; downgrade activity is debug-level, so info logs show nothing).

Measured on q9 (GB300, hot mean): episode **depth** is the cost driver, not triggering
per se — `0.8/0.75` (same trigger, shallow episodes) recovered 60% of the loss, and the
plateau is reached at `0.9/0.85` (1.872 s → 1.817 s, -3%; suite-neutral). Pre-staging the
re-upload (a downgraded-task prefetcher) was measured flat across a 25x staged-bytes
range: with 8 pipeline threads the re-upload is already overlapped at task-prepare, so
only avoiding the eviction helps. When GPU-pinning large tables, raise
`downgrade_stop_fraction` toward the trigger (and optionally the trigger itself) rather
than reaching for recovery-side prefetching.

## Benchmarking mixed configs

The TPC-H harness supports these experiments without edits: `SIRIUS_PRE_SQL` runs session
`SET`s after `LOAD`, and `SIRIUS_PIN_TIER_<TABLE>` overrides the global pin tier per table:

```bash
SIRIUS_PRE_SQL="SET pin_table_compression = true; SET pin_table_input_compression_plan_dir = '$PWD/plans'" \
SIRIUS_PIN_TIER_LINEITEM=gpu SIRIUS_PIN_TIER_ORDERS=gpu \
SIRIUS_PIN_TIER_PART=gpu SIRIUS_PIN_TIER_CUSTOMER=gpu SIRIUS_PIN_TIER_SUPPLIER=gpu \
SIRIUS_PIN_TIER_NATION=gpu SIRIUS_PIN_TIER_REGION=gpu \
python test/tpch_performance/performance_test.py --engine gpu --mode grouped \
  --iterations 3 --pin host --input /data/tpch_parquet_sf1000 --data-source parquet
```

Checked-in SF1000 plans live in `src/compression/simpatico_codegen/plans/tpch_sf1000/`;
per-column ratio and throughput annotations are in each file's comments. Re-measure with
`simpatico_cli benchmark` when hardware changes.

## Compression streams

In Sirius builds, Simpatico leases its column-parallel CUDA streams from
`cucascade::memory::exclusive_stream_pool`. Both scan decoding's four-stream
thread/device cache and the compression/decompression APIs' temporary leases use
this backend. Leases are exclusive and grow the per-device pool when needed, so
concurrent callers cannot deadlock while acquiring several streams.

These pools use the same implementation as pipeline execution, but have separate,
process-lifetime owners: compressed and decoded buffers can retain their allocation
stream after a call or submitting thread exits. Releasing a lease synchronizes its
streams before returning them. Input-ready events and decode completion preserve
the ordering needed for the downstream pipeline stream.
Allocations still run on the calling thread for cuCascade reservation accounting.
Standalone Simpatico builds retain their CUDA-only stream backend.
