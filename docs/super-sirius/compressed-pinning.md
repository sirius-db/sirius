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
