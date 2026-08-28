# Reproducing the pinned-table tests: GPU, host, and compressed

Step-by-step reproduction of the `pin_table` hot-run tests: **Pinned GPU**,
**Pinned host**, and **Pinned (Host/GPU) Compressed**. Each mode runs the same query;
only where (and how) the table is materialized changes.

Prerequisites: a release build (`pixi run make`) and a TPC-H parquet dataset. The paths
below use SF1000 at `/opt/dlami/nvme/tpch/tpch_parquet_sf1000` — substitute your own
scale factor and path. To generate one:

```bash
cd test/tpch_performance
pixi run bash generate_tpch_data.sh 1000 --format parquet --output /opt/dlami/nvme/tpch/tpch_parquet_sf1000
```

## 0. Setup

Start the repo's DuckDB shell (auto-loads the Sirius extension) or `LOAD` it explicitly:

```bash
./build/release/duckdb
```

```sql
LOAD 'build/release/extension/sirius/sirius.duckdb_extension';
```

The test query is TPC-H Q1's scan shape over `lineitem`:

```sql
SELECT l_returnflag, l_linestatus,
       sum(l_quantity), sum(l_extendedprice),
       sum(l_extendedprice * (1 - l_discount)),
       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)),
       count(*)
FROM read_parquet('/opt/dlami/nvme/tpch/tpch_parquet_sf1000/lineitem/*.parquet')
WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
```

Baseline: run it twice with **no pin** and record the times (`.timer on`). Every run pays
file I/O + decode.

## 1. Pinned GPU

Materialize the queried columns into GPU memory once; subsequent queries over the same
source skip file I/O and decode entirely. Queries don't change — pinned tables are
matched automatically.

```sql
CALL pin_table('/opt/dlami/nvme/tpch/tpch_parquet_sf1000/lineitem/*.parquet',
               tier = 'gpu', name = 'lineitem',
               cols = ['l_returnflag', 'l_linestatus', 'l_quantity', 'l_extendedprice',
                       'l_discount', 'l_tax', 'l_shipdate']);
```

Sizing: the pin must fit in free VRAM. At SF1000 these seven lineitem columns are
~200 GB uncompressed — beyond a single 96 GB card. For the GPU-tier step use a smaller
scale factor (SF100 fits comfortably), fewer columns, or skip ahead to host/compressed
pinning.

Re-run the query. Verify:

- The hot run is much faster than the unpinned baseline.
- `nvidia-smi --query-gpu=memory.used --format=csv` jumped by roughly the pinned
  columns' uncompressed size.
- The result values are identical to the unpinned run.

Release it before switching modes:

```sql
CALL unpin_table('lineitem');
```

## 2. Pinned host

Same call with `tier = 'host'` — columns land in pinned host memory instead. Use this
when the table doesn't fit in GPU memory; scans stream over PCIe/NVLink but still skip
file I/O and decode.

```sql
CALL pin_table('/opt/dlami/nvme/tpch/tpch_parquet_sf1000/lineitem/*.parquet',
               tier = 'host', name = 'lineitem',
               cols = ['l_returnflag', 'l_linestatus', 'l_quantity', 'l_extendedprice',
                       'l_discount', 'l_tax', 'l_shipdate']);
```

Re-run the query, then `CALL unpin_table('lineitem');`.

## 3. Pinned (Host/GPU) Compressed

Compressed pinning fits much more data in the same tier. Enable Simpatico compression
and point Sirius at a per-table plan directory **before** calling `pin_table`. Plans for
TPC-H SF1000 ship with the repository:

```sql
SET pin_table_compression = true;
SET pin_table_input_compression_plan_dir =
  '/home/ubuntu/sirius/src/compression/simpatico_codegen/plans/tpch_sf1000';

CALL pin_table('/opt/dlami/nvme/tpch/tpch_parquet_sf1000/lineitem/*.parquet',
               tier = 'host', name = 'lineitem',
               cols = ['l_returnflag', 'l_linestatus', 'l_quantity', 'l_extendedprice',
                       'l_discount', 'l_tax', 'l_shipdate']);
```

`tier = 'gpu'` works the same way for compressed-in-GPU pinning.

How plan resolution works:

- The plan dir holds one file per table, named `<table_name>.<ext>` (the `name` argument
  must match). In `plans/tpch_sf1000/` only `lineitem.txt` and `orders.txt` are active;
  the others are suffixed `_disabled`.
- A table with no matching plan file is pinned **uncompressed** — the call still
  succeeds. So is the whole call if `pin_table_input_compression_plan_dir` is empty
  (a warning is logged). Compression silently not engaging is the failure mode to check
  for, not an error.

Verify compression actually engaged:

- Memory delta is far below the uncompressed size (the shipped `lineitem` plan reaches
  ~1.1–12× per column, e.g. `l_orderkey` 12.4×, `l_linenumber` 10.4×).
- With `SET sirius_log_level = 'debug';` the log shows the per-table plan being resolved;
  a missing plan logs `pinning uncompressed`.
- Query results still match the unpinned baseline.

Optional knobs (defaults are sensible):

```sql
SET pin_table_compression_min_batch_size_bytes = 0;      -- compress even small batches
SET pin_table_compression_max_compressed_fraction = 1.0; -- discard compressed form if bigger than this fraction of original
```

## 4. Expected outcome summary

| Mode | Tier | Memory cost | Hot-run speed |
|---|---|---|---|
| Unpinned | — | none | slowest (I/O + decode every run) |
| Pinned GPU | `gpu` | full column size in VRAM | fastest |
| Pinned host | `host` | full column size in pinned host RAM | fast (PCIe/NVLink bound) |
| Pinned compressed | `gpu` or `host` | ÷ compression ratio | fast; decompress ≥250 GB/s per column by plan construction |

## Measured reference (SF100, RTX PRO 6000 Blackwell, 2026-08-28)

TPC-H Q1 shape over lineitem (600M rows), warm runs, single GPU. Results were
byte-identical across all modes.

| Mode | Pin time | Warm query | vs unpinned |
|---|---|---|---|
| Unpinned | — | 0.72 s | 1× |
| Pinned GPU | 0.9 s | 0.116 s | 6.2× |
| Pinned GPU compressed | 1.1 s | 0.118 s | 6.1× |
| Pinned host compressed | 2.1 s | 0.141 s | 5.1× |
| Pinned host | 5.5 s | 0.337 s | 2.1× |

Compressed host pinning beats uncompressed host on both pin and query time: less data
crosses PCIe and GPU decompress outruns the transfer it replaces. Note `nvidia-smi`
cannot show the pin footprint — Sirius pre-claims 95% of VRAM into its memory pool, so
`memory.used` is near-constant; use the debug log to confirm compression engaged.

## Pinning from StarRocks (Sirius compute nodes)

The same pinning works on the Sirius-as-StarRocks-CN stack, from the StarRocks SQL
prompt. Three pieces make it work end to end:

1. Whole-file scan assignment — pinned tables never serve byte-range splits, and
   StarRocks normally cuts them for every distributed `FILES()` scan:

   ```sql
   ADMIN SET FRONTEND CONFIG ("files_query_whole_file_ranges" = "true");
   ```

2. Pin on every compute node (one statement per node id from `SHOW COMPUTE NODES`):

   ```sql
   ADMIN EXECUTE ON <node_id>
   'pin_table path=/data/tpch/lineitem/*.parquet tier=gpu name=lineitem
    cols=l_extendedprice,l_discount,l_shipdate,l_quantity';
   -- and later: ADMIN EXECUTE ON <node_id> 'unpin_table lineitem';
   ```

3. Query with `FILES()` as usual. Each CN is assigned a per-query subset of the
   files and serves exactly that subset's chunks from its pin — the CN log line to
   look for is `serves operator ... as a file subset: N/M files`.

Compressed pinning works the same way: set the `sirius.compression.*` keys in the
CN's sirius.yaml (`--sirius-config`); no SQL involved. Measured on the 2× RTX PRO
6000 box (SF100 lineitem, 2 CNs, Q6 shape): 0.63 s unpinned → 0.09 s pinned, values
byte-identical to the DuckDB oracle.

Layout note: whole-file assignment trades load balance for cacheability — keep many
files per table, each well under `totalBytes / (nodes × dop)`, or one CN ends up
with disproportionate work.

Timeout warning: `ADMIN EXECUTE` has a hard 600 s ceiling on the FE side (statement
default and brpc stub cap, neither raisable from SQL). A pin that materializes
longer than that — think host-tier at SF1000 with `cols` omitted, which pins every
column — makes the FE report `executeCommand RPC failed`, while the CN keeps
materializing to completion and queries queue behind it; there is no cancel. Do
NOT retry on timeout (that queues a second full materialization) — watch the CN
log for `pin_table finished` instead. Keep pins under the ceiling by pinning only
the queried columns, splitting large tables into one pin per name, or using the
compression plans.

## Notes and caveats

- `cols` omitted pins **all** columns — fine at small SF, wasteful at SF1000.
- DuckDB-native tables pin with `CALL pin_table(format = 'duckdb', name = 'my_table', tier = 'gpu');`
  (no path argument). `UPDATE`-family statements are rejected while a table is pinned;
  `unpin_table` first.
- Plans are dataset-specific: a plan generated for one scale factor's value distribution
  applies to another SF only approximately. Regenerate plans with the simpatico codegen
  under `src/compression/simpatico_codegen/` for new datasets.
- Repeat runs in one session reuse the pin; `unpin_table` (or closing the shell)
  releases the memory. Confirm release with
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
