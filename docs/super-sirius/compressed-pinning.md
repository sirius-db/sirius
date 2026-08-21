# Compressed pinning: putting hot columns on the GPU

`pin_table` can store pinned chunks Simpatico-compressed on either tier:

```sql
SET pin_table_compression = true;
SET pin_table_input_compression_plan_dir = '/path/to/plans';  -- <table>.txt per table
CALL pin_table('/data/lineitem/*.parquet', tier => 'gpu', name => 'lineitem', cols => [...]);
```

Tables with no plan file pin uncompressed. Each plan file carries one `---`-separated DSL
block per full-table column in schema order; a pin of a column subset selects the matching
blocks (`select_plan_blocks`). Scans of a compressed entry decompress the projected columns
at task-prepare time via the `compressed_host_representation` /
`compressed_device_blob` → `gpu_table_representation` converters, column-parallel across
4 fixed streams (measured strictly better than serial; the stream count is deliberately
not configurable).

## Decode-time equality pushdown

A string equality filter over a compressed pinned column can be evaluated *inside* the
decode instead of re-expressed as a post-decode comparison. At `prepare_for_query`, the
scan manager derives a `sirius::decode_equality_pushdown` from the query's pushed-down
filters and attaches it to the entry's scan (`set_equality_pushdown()`); the converter
translates it into a `simpatico::decode_predicate`, so the decoder emits the match result
directly. Interfaces: `src/compression/compressed_representation.hpp`
(`decode_equality_pushdown`), `src/compression/compression_converters.cpp` (predicate
translation), threaded through `sirius_scan_manager::prepare_for_query`.

The compression settings can be staged in either order before a table is pinned. They become
active only when `pin_table_compression` is true, the plan-directory setting is non-empty, and a
matching table plan exists. `pin_table` warns and pins uncompressed when the enable flag is true
but the plan directory is empty; a missing table plan likewise warns and falls back. The batch-size
and compressed-fraction settings are inert until a matching plan activates compression.

## Which tier to compress on (GB300, TPC-H SF1000, 22-query hot suite)

The tier decides whether compression helps at all. Measured against a 20.74 s all-host
uncompressed baseline (dev `c4e8a10b`, pipeline 4, host-pinned, hot = min of iters 1–2 —
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

## Filtering while decompressing (experimental)

A scan can hand its row filter to the decompressor instead of evaluating it afterwards. Where
a column's plan allows it the filter is answered from the compressed form, and the surviving
rows come back already compacted — so rows the filter rejects are never fully decoded, and the
scan skips its own filter pass when the decode carried the whole predicate.

It is off by default and inert when off (byte-identical to an ordinary decompress):

```bash
SIRIUS_EXP_FUSED_SCAN_FILTER=1     # the gate
SIRIUS_EXP_FUSED_SCAN_DIAG=1       # the decision trace — reach for this first
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

- a batch with any full-width column proceeds only below `SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL`
  (0.10);
- otherwise compaction is given up above `SIRIUS_EXP_FUSED_SCAN_MAX_SEL` (0.35), unless a
  dictionary output is present — that route is exempt, since it wins regardless;
- below `SIRIUS_EXP_FUSED_SCAN_K4_MAX_SEL` (0.15) the decode walks a survivor index list
  instead of the mask bits.

One batch measured unselective predicts the rest of that scan, so the scan stops attempting it
rather than paying per batch.

Two operational caveats. These are environment variables, not `SET` options: they are
process-wide, read once and cached, so they cannot be changed per session and do not appear in
`duckdb_settings()`. And the trace is the only way to see what happened — the plan per column (route,
bounds, what was dropped and why), then per batch the sources in the order they ran, the
enumeration chosen, the survivor count and whether the batch came back needing no further
filtering.

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
