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
`compression_column_threads` streams (default 4 — measured strictly better than serial).

## Which tier to compress on (GB300, TPC-H SF1000, 22-query hot suite)

The tier decides whether compression helps at all. Measured against a 20.74 s all-host
uncompressed baseline (dev `c4e8a10b`, pipeline 4, host-pinned, hot = min of iters 1–2):

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
