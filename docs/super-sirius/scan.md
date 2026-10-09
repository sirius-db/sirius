#Scan Subsystem

This document covers the scan subsystem end-to-end: how data enters Super Sirius from storage through the unified GPU scan operator and its per-format `gpu_ingestible` sources, the scan manager that produces and balances scan splits, pinned-table caching, GPU decode of DuckDB-native storage, and the cuCascade IO layer underneath.

## Overview

The GPU scan path is a single unified source operator, `sirius_gpu_scan_operator` (physical type `GPU_SCAN`). It carries no format-specific code: it pulls pre-built splits off a `split_connector` and delegates per-split materialization to an installed **`gpu_ingestible`**. One `gpu_ingestible` implementation exists per source format:

| Format | Ingestible | Per-table bind data | Source |
|--------|-----------|---------------------|--------|
| Parquet (local or object-store) | `parquet_gpu_ingestible` | `parquet_ingestible_table_info` | `cudf::io::read_parquet` over row-group slices |
| DuckDB-native `.duckdb` tables | `duckdb_native_gpu_ingestible` | `duckdb_native_ingestible_table_info` | GPU decode of per-row-group storage segments |

The pipeline converter rewrites a DuckDB table scan into a `GPU_SCAN` source: it lowers the bind data into the appropriate `ingestible_table_info`, calls the free `make_ingestible(...)` factory to build the `gpu_ingestible`, constructs the operator carrying it, and inserts it at `operators[0]` of the pipeline. When a dynamic-filter channel is wired to the scan, a `DYNAMIC_FILTER` operator sits directly above it (see [Dynamic Filters](dynamic-filters.md)). No separate metadata pipeline is created.

Before a query runs, `sirius_scan_manager::prepare_for_query` walks the plan's `GPU_SCAN` operators. For each it either (a) matches a pinned-table cache entry and serves the scan from cached batches, or (b) builds a `split_provider` over the operator's ingestible. A single per-query sequencer (`load_balancing_scan_batch_coalescer`) drives metadata production, coalesces the output into right-sized data batches, balances each batch onto a GPU, and pushes the resulting splits onto each operator's `split_connector`.

Data reaches the GPU through cuCascade's IO layer (`cucascade::io::ioctx` / `cucascade::io::datasource`, with the pinned-memory `fs_cache` under `cache.mode: cucs`) — see [IO Layer (cuCascade io)](#io-layer-cucascade-io). The scan path consumes that layer: each split carries prefetch hints, and the read for a split goes through the `ioctx` its backend resolves to. The target device travels with the request.

## Scan contracts

Before transparent GPU execution, Sirius checks that its scans read the same bound inputs as DuckDB's original plan. Each fresh split is checked against the scan consuming it. Native scans hold checkpoint protection until cleanup, and CPU fallback requires permitted sources and successful cleanup.

See [Scan Contracts](scan-contracts-design.md) for the checks, fallback rules, and limitations.

## Scan Operator

### `sirius_gpu_scan_operator` — `GPU_SCAN`
**File:** `src/op/scan/sirius_gpu_scan_operator.hpp`, `src/op/scan/sirius_gpu_scan_operator.cpp`

The single GPU scan source operator. It owns:

- a `std::shared_ptr<gpu_ingestible>` — the installed per-format source, built by the pipeline converter and parked on the operator;
- a `std::shared_ptr<split_connector>` — the blocking queue the scan manager pushes splits into.

        **Source interface.*
      *As a pipeline source the operator exposes `get_next_task_hint()` / `all_ports_empty()` (
  both keyed off `split_connector::is_closed()`)and `get_next_task_input_data()`,
  which blocks inside `split_connector::get_next_split()` until a split arrives
    or the connector is closed and drained.Each pulled split is a `scan_operator_input`; on dequeue the operator issues an immediate prefetch hint for the split's byte ranges.

**Execution.** `execute(input_data, stream)` runs one split:

1. `gpu_ingestible::materialize_table(split, stream)` produces a `filtered_table` — the materialized `cudf::table` (wrapped in an `owning_table_view`) plus a `filter_state` tag describing how much filter/projection the materialize step already applied.
2. If the tag is `ROW_FILTERED_AND_PROJECTED` the table is already in final output layout and is released directly; otherwise `gpu_ingestible::post_filter_and_project(...)` applies any pending row filter and projection.
3. If the plan carries a compressed-materialization physical sidecar, exact runtime bounds verify wider-to-narrower casts for columns containing non-null values before the table is normalized to that complete physical schema. Such a sidecar exists only for scans the pinned cache serves with pin-time-narrowed columns (the plan-time residency gate); fresh unpinned scans carry no sidecar and skip normalization entirely. A resident cached input without an override is restored to the native schema when its stored numeric carrier is narrow. Fresh feature-off scans retain their existing natural shape and types. See [Compressed Materialization](compressed-materialization.md) for the narrowing/restoration rules and [Compressed Pinning](compressed-pinning.md) for Simpatico-compressed pinned chunks.
4. The result is wrapped in a `data_batch` tagged with the split's target `memory_space` and returned as a `pipelineable_operator_data` for the downstream pipeline.

The operator handles two split shapes transparently, both delivered as `scan_operator_input`: a **fresh read** (the input carries a `scan_info`, materialized via the ingestible) and a **pinned-cache hit** (the input carries a resident `data_batch`, filtered when required and normalized to the planned carrier schema). The operator never sees the source format directly.

Before execution, a resident split is converted or decompressed to a plain GPU table when necessary. An unmasked split with no filtering ahead of it — unfiltered, or decode-row-filtered by the pushdown decode — may detach that per-query table: a split needing no carrier cast transfers it directly to the output, while a carrier-casting split keeps the wrapper as the retained source of a transactional steal in which execute builds every replacement column before committing the take, so an OOM mid-cast leaves the split rematerializable. Because the transactional steal bypasses `post_filter_and_project`, a decode-row-filtered split takes it only when the ingestible reports its output assembly as a leading identity (no partition synthesis, no reordering); a width match alone cannot prove that, since trailing pure-filter columns can offset synthesized output columns. Masked or still-to-be-filtered splits retain the wrapper view, which preserves the source for copy-based filtering.

`no_history_peak_memory_estimate()` uses the larger of stored and working-set bytes for resident inputs that need no carrier conversion. For a known conversion it adds the exact destination bytes to the working set. The named maximum carrier expansion (8) is used only when a converting destination cannot be sized. A resident input with a physical sidecar but no reported conversion reserves the working set plus stored bytes as a source-bounded floor. For fresh reads the estimate remains 8 times the projected-column estimate plus decoded filter-only column buffers. The projected-column estimate remains the execution-history basis, and history-based reservations are clamped to the known decoded column-buffer footprint.

> `sirius_physical_table_scan` (`TABLE_SCAN`) remains only as the plan-time carrier that `wrap_table_scan_source` consumes; the read path for parquet and DuckDB-native tables runs entirely through `GPU_SCAN`.

## gpu_ingestible

**Files:** `src/op/scan/gpu_ingestible.hpp`, `gpu_ingestible_types.hpp`, `src/op/scan/gpu_ingestible.cpp`;
implementations `parquet_gpu_ingestible.cpp`, `duckdb_native_gpu_ingestible.cpp`
                                                  .

`gpu_ingestible` is the abstract source of cuDF tables — one implementation per data format.It is**
                                                    composed twice** : by the `split_provider` (
                                                                         metadata - side,
                                                                         to enumerate work) and
                                                by the `sirius_gpu_scan_operator` (
                                                  execution - side, to materialize each split)
                                                      .It inherits `enable_shared_from_this` so the
                                                    provider can borrow it non
                                                    - owningly while the operator holds the single
                                                      owning `shared_ptr`.

                                                      ## #Interface

                                                  | Method | Role | | -- -- -- --| -- -- --|
                                                  | `has_processed_all_metadata()` |
                                                  Thread - safe snapshot
  : is all metadata enumerated
  ? Typically an atomic cursor vs.a precomputed total.| | `next_split_provider(resolve)` |
      Atomically claim the next metadata unit and return a callable that produces its `scan_info`(
        s); `resolve` maps each file path to its ioctx. Null when nothing left to claim. |
| `create_batch_coalescer()` | Build the format's `batch_coalescer`, which bundles per-unit metadata into right-sized data-batch splits. |
| `materialize_table(split, stream)` | Produce the `filtered_table` for one split (dispatches to `materialize_metadata_to_table` for a fresh read, or wraps the resident batch for a cache hit). |
| `materialize_metadata_to_table(info, mem_space, stream)` | Issue the read/decode for one split into a `cudf::table`. `mem_space` names the destination: its allocator is where the decoded columns land. It does not select an ioctx. |
| `post_filter_and_project(table, mem_space, stream)` | Apply a pending post-decode filter and/or projection to output layout. |
| `table_info()` | The per-table bind data (`ingestible_table_info`). |
| `materialized_column_order()` | Storage indices in the exact order `materialize_table` emits columns (output columns first, then pure-filter columns). The pinned-cache path serves columns in this order so a cached batch is laid out identically to a fresh read. |

### Bind data vs. split descriptors

Two polymorphic carriers separate per-table from per-split state (`gpu_ingestible_types.hpp`):

- **`ingestible_table_info`** — built once by the pipeline converter from the DuckDB binding, parked on the operator. Exposes `column_names()` and `file_paths()` (used for pinned-cache matching). Implementations: `parquet_ingestible_table_info`, `duckdb_native_ingestible_table_info`.
- **`scan_info`** — one per emitted split. Carries the per-split read description and optional prefetch `fadvise_entries()` (acted on only when the bound datasource's context has an `fs_cache`; `fadvise` does nothing otherwise), projected-column estimate, and decoded column-buffer estimate. Implementations: `parquet_split_info` (the data-batch split), `parquet_file_scan_info` (the per-file metadata unit), `duckdb_native_scan_info`.

### `filtered_table` / `filter_state`

`materialize_table` returns a `filtered_table` = an `owning_table_view` plus a `filter_state` tag recording how much filter+projection the materialize step already absorbed:

| State | Meaning |
|-------|---------|
| `UNFILTERED` | No filter applied (e.g. a pinned table, or duckdb-native which always filters post-decode). |
| `ROWGROUP_FILTERED` | Row groups pruned by statistics only. |
| `ROW_FILTERED` | The reader applied the row-level filter (parquet reader-side pushdown). |
| `ROW_FILTERED_AND_PROJECTED` | Fully assembled to output layout (parquet hive-partition path, or a per-query-cached table). |

The operator skips `post_filter_and_project` only when the state is already `ROW_FILTERED_AND_PROJECTED`.

Required scan predicates, including `IS NOT NULL` introduced by DuckDB's string-predicate
rewrites, belong to the scan's row-level evaluator. Native reads and pinned batches retain
them through materialization and evaluate them before dropping filter-only columns. An empty
`projection_ids` means all `column_ids` are read; an explicit projection is extended with
filter-only columns while preserving the original output arity.

Parquet (and its Iceberg subclass) also retains NULL predicates in the residual. Null-count
statistics can prune whole row groups, but mixed groups still need row filtering. NULL tests
never enter cuDF's min/max statistics expression; a partial reader pushdown cannot claim the
whole predicate was applied. Numeric decode-range analysis likewise retains the residual when
it cannot represent required NULL rejection. Advisory `OPTIONAL_FILTER`s and hive partition
predicates already enforced by DuckDB's file selection remain omitted.

### Factories

There is no factory class. Each implementation provides a free `make_ingestible(std::unique_ptr<...table_info>)` overload (defined in its `.cpp`); the pipeline converter calls the right overload by the concrete `table_info` type it built.

### Parquet ingestible

`parquet_gpu_ingestible` (`parquet_gpu_ingestible.{
  hpp, cpp}`) builds the canonical `scan_plan` and shared `parquet_reader_options` (column projection only) once in its constructor, and pre-coalesces the DuckDB filter into a stored expression (partition-column filters dropped — DuckDB already prunes the file list by hive value). `next_split_provider` hands out **one file at a time**: each metadata task opens the file's `cucascade::io::datasource`, reuses or parses+caches the footer, runs the FLBA-decimal pushdown-safety probe, translates the filter to a cuDF AST and prunes row groups by statistics, estimates each surviving row group's projected data columns and all decoded column buffers (plus the partition columns the split will synthesize, which count toward both estimates), and emits one `parquet_file_scan_info`. A column-less scan's row-count carrier (see `scan_plan` below) is resolved per file: a file that lacks the carrier column, or has no row groups to resolve it against, keeps natural-batch reader options and is sized as the full-width read it is. The coalescer caps batches on decoded column-buffer bytes, while preserving projected-column bytes separately for memory history, and never puts files with different reader options in one split. `materialize_metadata_to_table` reads the bundled row-group slices via `cudf::io::read_parquet` (re-translating the filter on the task-local stream for reader-side pushdown unless the per-file probe disabled it), and assembles hive-partition output inline. Reader-side filter pushdown is a per-split decision.

### DuckDB-native ingestible

`duckdb_native_gpu_ingestible` (`duckdb_native_gpu_ingestible.{
  hpp, cpp}`) prepares its serial walk plan during execution preparation under a shared checkpoint lease (`prepare_duckdb_native_walk`: partition statistics, projected-type viability gate, and filter-stat row-group pruning — a non-viable query throws through the runtime fallback policy before any per-segment IO). See [Native checkpoint lease](scan-contracts-design.md#native-checkpoint-lease) for its lifetime and effect on checkpoints. It slices the table's row groups into fixed internal ranges of eight groups. `next_split_provider` hands out one range per claim; each metadata task walks that range and emits a `duckdb_native_scan_info`. `materialize_metadata_to_table` decodes the range's storage segments into a `cudf::table` (always `UNFILTERED`); filter evaluation and projection to output arity happen in `post_filter_and_project`.

### Iceberg ingestible

`iceberg_gpu_ingestible` (`iceberg_gpu_ingestible.{
  hpp, cpp}`) **extends** `parquet_gpu_ingestible` rather than reimplementing it: an Iceberg table's data files are parquet, and `iceberg_scan` resolves its manifests into the same `MultiFileBindData` file list `read_parquet` produces, so the parquet ingestible reads them unchanged. Only two behaviours differ. `create_batch_coalescer` wraps the parquet coalescer and stamps `disable_filter_pushdown` on every emitted split — but **only when the table has deletes** (per-split stamping is required; `reader_options` is one shared object handed to every split). `materialize_metadata_to_table` decodes through the base, then applies the delete pipeline to each decoded batch.

Suppressing pushdown is load-bearing, not conservative. Positional deletes and deletion vectors are keyed on a row's position **within its data file**; if cuDF drops rows during decode, decoded positions no longer identify file positions and the mapping is unrecoverable. Materialize therefore returns `UNFILTERED` and `post_filter_and_project` applies the predicate *after* deletes — which is also Iceberg's required order. A non-`UNFILTERED` state from the base throws. Row-group pruning stays on and is safe: it only removes rows the predicate could not have matched, and offsets come from the footer, which lists pruned groups.

Row mapping is a **list of runs, not one offset**. `build_batch_layout` emits one `batch_row_run` per (file, row group), because splits span files and pruning leaves gaps; the decoded row count must equal the sum of run rows or it throws.

**Delete discovery** (`iceberg_metadata_reader.{
  hpp, cpp}`) delegates manifest parsing to DuckDB's `iceberg` and `avro` extensions. `iceberg_metadata()` covers everything except the three V3 deletion-vector fields it does not expose (`content_offset`, `content_size_in_bytes`, `referenced_data_file`), which a `read_avro` query over the containing manifest supplies. Results are memoized per query, keyed on transaction id plus table path plus snapshot, because one query reads delete data more than once — `iceberg_scan` is not serializable, so the plan is generated twice. `read_deletion_vector` (`puffin_reader.cpp`) validates the Puffin container's leading and trailing magic before seeking to a blob offset inside it, then checks the deletion-vector blob's own magic and CRC-32.

**Failures throw; they never degrade to empty delete data.** An empty result is indistinguishable from "this table has no deletes", so swallowing a read error would turn *could not read the deletes* into *there are none* and return rows the table logically removed.

Tables the path cannot answer correctly **decline at plan time** (`sirius_plan_get.cpp`) rather than producing wrong rows — a `NotImplementedException` is the established CPU-fallback signal. Declining at plan time is deliberate: a runtime fallback wastes the plan, the GPU reservation and the decode, and it poisons the connection (the next GPU query on it deadlocks).

Currently declined:

- **An unpinned `iceberg_scan(path)`** — i.e. anything without an explicit `snapshot_from_id`. Sirius plans every query twice (the iceberg bind data is not serializable, so the plan is re-bound) and discovers delete files in a further pass; each resolves "current" independently, so a commit landing between any two pairs one snapshot's data files with another's deletes. The snapshot DuckDB actually bound is not reachable — it lives in the extension's own `IcebergMultiFileList` / `MultiFileBindData::bind_data`, `MultiFileBindData` exposes no snapshot, `OpenFileInfo::extended_info` carries only `file_size`/`sequence_number`, and duckdb-iceberg overrides no `GetBindInfo`. **Do not "fix" this by deriving the current snapshot here**: the highest sequence number is not the current snapshot under rollback, branches or staged WAP commits, and comparing data-file sets cannot tell two snapshots apart when only their deletes differ. Lifting the decline needs serializable iceberg bind data upstream, or Sirius not planning twice.
- **Equality deletes.** The filter, the GPU anti-join mask and the group build all exist, but key projection is not wired, and two pieces do not implement the spec: the applicability test compares the *manifest's* sequence number rather than the entry's inherited data sequence number, and keys are taken from every column of the delete file rather than the entry's declared `equality_ids`. `read_iceberg_delete_data` refuses live equality entries behind `kEqualityDeleteRouteImplementedToSpec`, which is the single switch that turns the route on; the components are covered directly by `test/cpp/scan/test_iceberg_equality_delete.cpp`, because no SQL test can reach them.
- **`snapshot_from_timestamp` and `version`.** The delete path resolves only by `snapshot_from_id`, so these would read one snapshot's data against another's deletes.
- **`allow_moved_paths = true`.** It rewrites every data-file path to `<table_path>/data/<name>`, while delete discovery calls `iceberg_metadata()` without the flag and keeps the manifests' original paths; the two would not match and the deletes would be dropped.
- **Any data file whose schema differs from the table's current one** — a dropped field id (gap test, bind data only), a rename or an addition (name+id pair comparison against every data file's footer), a promoted type (`int` -> `long` keeps the name and id while the old file stays INT32), or a file carrying **no** field ids at all (a name-mapped `add_files` migration, which the probe must decline rather than skip).
- **A deletion vector whose Puffin footer descriptor contradicts its manifest entry**, or whose decoded position count contradicts the entry's `record_count`.

Any code opening its own `Connection` on this path must bracket it with `SiriusContext::InternalQueryGuard` on **that connection's** context — the bracket is per-connection, so a caller's guard does not cover a connection it opens.

**Known limitation.** Projected columns are resolved by **name** (`parquet_gpu_ingestible.cpp`), while Iceberg's data model is field-ID keyed. A column dropped and re-added under the same name is a *new* field ID absent from older data files and must read NULL; name resolution would find the old column and return stale data. The plan-time gates above close that off by declining every evolved table, so the limitation costs performance rather than correctness — but it is why the gates are as broad as they are. Resolving by field ID is the real fix and removes all of them: DuckDB's `MultiFileColumnDefinition` already carries the field ID (`identifier`, `GetIdentifierFieldId()`, `GetDefaultValue()`) and `MultiFileColumnMapper` implements the spec rule, including the name-mapping fallback for files without field IDs.

## owning_table_view

**File:** `src/op/scan/owning_table_view.hpp`

`owning_table_view` is the handle the scan path threads tables through. It exposes a `cudf::table_view` while owning the data behind it, regardless of whether that data is a fully-materialized `cudf::table` or a view into some other type-erased owner. It is in one of three states: an owned `cudf::table`, a type-erased owner + column selection, or empty.

Column manipulations — `reorder_columns`, `drop_columns`, `select_columns` — are pure index manipulations over a stored selection: they never allocate device memory. An owned table is first demoted into a view (a `unique_ptr` move, not a copy) and the selection permuted/subset in place. Only `materialize` and `release` (the view -> table transition) may allocate, and even then, when the underlying owner can surrender its columns (the `no_alloc_materializable` concept — e.g. a `cudf::table`), the surviving column buffers are *moved* out rather than copied.

This is what lets the dominant scan paths — `SELECT *`, identity layouts, reader-side-pushdown reads — flow the reader output through projection/reorder to the output batch with no extra GPU copy.

## scan_plan

**File:** `src/op/scan/scan_plan.hpp`, `src/op/scan/scan_plan.cpp`

`scan_plan` is the canonical description of what a parquet scan reads, how it assembles output, and how filters map between index spaces. The `parquet_gpu_ingestible` builds it once in its constructor and shares it (immutably)
with every emitted split.

```cpp struct scan_plan {
  std::vector<data_column> data_columns;  // columns read from parquet, in batch order (D)
  std::vector<partition_column>
    partition_columns;                      // hive-injected columns (name, type, primary index)
  std::vector<output_entry> output_layout;  // one entry per output column, in DuckDB order
  std::vector<std::optional<size_t>> batch_position_by_column_id;  // C -> D map
  std::unordered_set<size_t> partition_primary_indices;            // for filter-skip
  std::optional<size_t> carrier_batch_index;  // D index of a column-less scan's row-count carrier
};
```

Three index spaces appear in the parquet path:

- **P (primary index)** — DuckDB schema position
- **C (column-ids position)** — index into the scan's `column_ids` list
- **D (batch position)** — column position in the cuDF reader output (post-hive-removal)

`output_layout` is walked once during materialization to produce the final table: `DATA(k)` entries `std::move` from the read batch at position k, `PARTITION(k)` entries synthesize a scalar-backed column from the hive partition value. Pure-filter data columns (read but not output) fall out of scope and free.

For `SELECT *` with no partitions and no pure-filter columns, the plan is a trivial identity and the reader output is forwarded unchanged — no permute, no copy. `SELECT count(*)` also has an empty `output_layout`;
that short circuit leaves the read batch unchanged rather than synthesizing a 0 -
  column table(a zero - column cudf table carries no row count),
  and the downstream count aggregation uses the batch row count.So that this batch does not decode every file column, `build_scan_plan` gives
                                                                                                                        a
                                                                                                                        column
                                                                                                                        -
                                                                                                                        less
                                                                                                                        scan(
                                                                                                                          count(
                                                                                                                              *),
                                                                                                                          virtual -
                                                                                                                            only,
                                                                                                                          or
                                                                                                                            partition -
                                                                                                                              only)
                                                                                                                          a *
                                                                                                                            *row
                                                                                                                        -
                                                                                                                        count
                                                                                                                        carrier *
                                                                                                                          * : the
                                                                                                                              narrowest
                                                                                                                              fixed
                                                                                                                              -
                                                                                                                              width
                                                                                                                              non
                                                                                                                              -
                                                                                                                              partition
                                                                                                                              column
  ,
  read as a data column with no output entry — the same shape as a pure
    - filter column — and recorded in `carrier_batch_index`,
  so the reader projects to that one column.Schemas with no fixed
    - width column keep the natural batch; `set_column_names({})` is never passed to cuDF. Partition columns synthesized for the output count toward both size estimates, so a partition-only scan is not sized by its carrier alone.

Sirius keeps `file_index` as the zero-based index in the original bound file list. When `file_index` is projected, DuckDB may instead renumber files after a runtime join filter on a Hive partition column or the legacy `filename=true` column; this known difference is tracked in [duckdb/duckdb#26044](https://github.com/duckdb/duckdb/issues/26044).

## Column Mapping

Parquet column-chunk order is not guaranteed to match DuckDB's logical column order. `parquet_gpu_ingestible` builds a name-based DuckDB->parquet mapping via `parquet_schema_mapping::leaf_indices_for_column(schema, column_name)`, which walks the parquet schema's `path_in_schema` (case-insensitive, mirroring DuckDB).

For nested types (`STRUCT`, `LIST`, `MAP`), one DuckDB column maps to multiple parquet leaf chunks; the mapping returns all leaves under the top-level column name. The cuDF parquet reader, given a top-level column name, materializes the nested `cudf::column` natively without post-read reassembly. MAP-annotated groups are recognized in the schema walk (`parquet_helpers.cpp`), consuming the `key_value` repeated group and mapping to `duckdb::LogicalType::MAP`. Nested columns can be scanned and projected through to the result, but not *operated on* — a nested column in WHERE, GROUP BY, or JOIN ON is rejected during plan generation (see [Physical Plan Generation](physical-plan-generation.md)).

## Scan Manager

**Files:** `src/scan_manager/sirius_scan_manager.hpp`, `src/scan_manager/sirius_scan_manager.cpp`; `split_provider.{
  hpp, cpp}`, `split_connector.{
  hpp, cpp}`, `load_balancing_scan_batch_coalescer.{
  hpp, cpp}`, `balancing_strategy.hpp`, `round_robin_strategy.{
  hpp, cpp}`, `config.hpp`.

`sirius_scan_manager` prepares scan-side state before a query runs and drives metadata production for every `GPU_SCAN` source. It owns a configurable thread pool, the cuCascade io registry and the contexts built from it — the default `cucascade::io::ioctx` (uring for `backend: native`; kvikIO for `backend: kvikio`, which is single-GPU only) plus contexts built on first use for other schemes, such as REST for `s3://` — the `fs_cache` of each context that has one, and the registry of pinned-table entries. It runs alongside the GPU pipeline executors and is independent of the data-repository machinery used between intermediate operators.

### Components

| Component | File | Role |
|-----------|------|------|
| `sirius_scan_manager` | `scan_manager/sirius_scan_manager.{
  hpp, cpp}` | Owns thread pool + io contexts (and their caches) + pinned-table registry; `prepare_for_query` wires providers and starts the sequencer |
| `split_provider` | `scan_manager/split_provider.{
  hpp, cpp}` | Concrete driver that composes a `gpu_ingestible`; `run()` dispatches one metadata task per claimed unit onto the dispatcher |
| `load_balancing_scan_batch_coalescer` | `scan_manager/load_balancing_scan_batch_coalescer.{
  hpp, cpp}` | Per-query sequencer: drains each provider's metadata output through the format's `batch_coalescer`, balances each batch onto a GPU, fires prefetch hints, pushes splits onto the connector |
| `batch_coalescer` | `op/scan/batch_coalescer.hpp` (impls in the ingestibles) | Bundles per-unit `scan_info`s into right-sized data-batch splits |
| `balancing_strategy` / `round_robin_strategy` | `scan_manager/balancing_strategy.hpp`, `round_robin_strategy.{
  hpp, cpp}` | Picks the target GPU for each split and stamps `preferred_device_id` on it |
| `split_connector` | `scan_manager/split_connector.{
  hpp, cpp}` | Lock-protected blocking queue between the producer (sequencer) and the operator |
| `cache_entry_info` / `pinned_entry` | `scan_manager/sirius_scan_manager.hpp` | Pinned-table identity + column layout, and the cached batches (see [Pinned Tables](#pinned-tables)) |

### Lifecycle

1. **Plan stage.** During pipeline conversion the bind data is lowered into an `ingestible_table_info`, `make_ingestible(...)` builds the `gpu_ingestible`, and a `sirius_gpu_scan_operator` carrying it is inserted as the pipeline source. The operator constructs its own (empty) `split_connector`.
2. **Per-query preparation.** `prepare_for_query(query, pruning, allocated_gpu_ids)` resets prior state, builds a `round_robin_strategy` over the query's admitted GPU ids and a fresh `load_balancing_scan_batch_coalescer`, then walks the query's `GPU_SCAN` operators in order. The admitted set is passed in by `SiriusContext::create_query`, which reads it back off `task_creator` after `sirius_engine::initialize_internal` installed it — so scan placement, partition pins and `pipeline_build_context` all draw on one list (see [configuration.md](configuration.md) for `topology.gpus_per_query`). For each it `register_pipeline`s a sequencer slot (which builds the ingestible's `batch_coalescer` and captures the operator's connector). Then either:
   - **Cache hit** — `try_assign_cached_entries` finds a pinned entry whose identity and columns can serve the scan; it attaches a `databatch_provider` to the slot and skips disk reading entirely; or
   - **Cache miss** — a `split_provider` is built over the operator's ingestible and stored in `_providers_by_op`.
3. **Execution.** `start_metadata_processing` spawns the single sequencer worker on the dispatcher, then calls `split_provider::run` for each non-cached operator. `run` iterates `has_more_splits` / `next_split_provider` and hands each claimed metadata task to the dispatcher; each task enqueues its `scan_info` onto the operator's sequencer slot queue. The sequencer worker walks slots in registration order, coalescing each slot's metadata into data-batch splits, balancing each onto a GPU, firing `fadvise`/`opportunistic` prefetch, and pushing a `scan_operator_input` onto the operator's connector. Consumers (the scan operators) block in `split_connector::get_next_split` until splits arrive.
4. **Teardown.** The sequencer closes each connector when its slot is drained (forwarding any worker-captured exception, first-writer-wins). Once closed and drained, the operator's `all_ports_empty()` returns true and `get_next_task_hint()` returns `nullopt`. `reset()` requests the dispatcher stop and rebuilds it for the next query.

### split_provider

`split_provider` is concrete and composes a `gpu_ingestible` non-owningly (the operator owns the lifetime; the provider is always torn down first by `reset()`). Its `has_more_splits` / `next_split_provider` delegate to the ingestible. `run(scheduler, on_split)` enqueues one task per claimed metadata unit; the connector is closed (with any captured exception) when the last enqueued task drops its reference to the shared completion state.

### load_balancing_scan_batch_coalescer

This is the per-query sequencer. Each `GPU_SCAN` operator gets a `metadata_processing_state` slot holding a blocking queue (fed by the provider's metadata tasks), the ingestible's `batch_coalescer`, the shared `balancing_strategy`, and the operator's `split_connector`. A single worker walks the slots in registration order. For a live scan it dequeues per-unit `scan_info`s, pushes them through the coalescer, and for each coalesced batch: picks a GPU via the balancer (stamping `preferred_device_id`), fires `fadvise` and an `opportunistic` prefetch for the batch's byte ranges, and pushes a `scan_operator_input` onto the connector. For a cached pipeline it replays the attached `databatch_provider`, pushing one resident `scan_operator_input` per cached chunk. Serialising the opportunistic prefetch tier across pipelines in execution order gives the prefetching cache its longest lead time for the head-of-line pipeline.

### Device balancing

`balancing_strategy` decouples *which GPU a split runs on* from *how splits are produced*. `round_robin_strategy` hands out devices from the query's admitted GPU id set via a single shared atomic cursor, spreading splits evenly across those GPUs and continuously across the whole scan stage. The chosen device is recorded on the split via `set_preferred_device_id`; the task creator reads it back when it builds the `gpu_pipeline_task` so the scheduler dispatches to that GPU. (Cached/resident inputs carry no balancer-assigned device; the task creator derives their device from the chunk's resident `memory_space` for NUMA/host-pin locality — see the memory-management doc.)

### split_connector

A lock-protected queue of pre-built splits. The producer (sequencer) enqueues via a friended `push_split` and calls `close(exception?)` when done. The consumer pulls via `get_next_split()`, which blocks until a split is available or the connector is closed and drained: returns `nullopt` when drained, the next split otherwise, or rethrows the producer's stored exception once the queue is empty.

### Configuration

`scan_manager_config` (`config.hpp`) tunes the thread pool, the IO backend selector (`backend`), the uring/REST reactor counts (`uring_n_reactors` / `rest_n_reactors`, one worker thread each), cuCascade's per-backend and cache configs, and object-store credentials. `to_io_config()` converts it into the `cucascade::io::io_config` the io registry is built from. See [Configuration](configuration.md#scan-manager--io-configuration).

## Pinned Tables

**Files:** `src/pin_table.hpp`, `src/pin_table.cpp`; pinned-entry storage + cache matching in `src/scan_manager/sirius_scan_manager.hpp` and `src/scan_manager/sirius_scan_manager.cpp`; zone-map capture and pruning in `src/scan_manager/pinned_chunk_stats.hpp` and `src/scan_manager/pinned_chunk_stats.cpp`; MVCC reconciliation in `src/op/scan/duckdb_mvcc_visibility.cpp`, `src/scan_manager/mvcc_mask_job.cpp`, `src/op/scan/duckdb_insert_delta.cpp`, and `src/scan_manager/insert_delta_job.cpp` (with their headers).

The `pin_table` table function preloads data for later scans. It supports parquet and DuckDB sources, with the tiers described below.

```sql
CALL pin_table('/path/to/lineitem.parquet',
               name = 'lineitem',
               tier = 'gpu',
               cols = ['l_orderkey', 'l_quantity', 'l_extendedprice', 'l_shipdate']);

-- A duckdb-native base table can be pinned too:
CALL pin_table('my_table', name = 'my_table', format = 'duckdb', tier = 'host');

SELECT SUM(l_extendedprice * l_quantity)
  FROM read_parquet('/path/to/lineitem.parquet')
  WHERE l_shipdate >= DATE '1994-01-01';

CALL unpin_table('lineitem');
```

`format` is `parquet` or `duckdb`, resolved at bind time from an explicit parameter or inferred from the path extension. `tier` is `gpu` (columns in GPU device memory), `host` (columns in pinned host memory) or `parquet` (parquet sources only: caches undecoded column-chunk ranges for the selected columns, or all columns when `cols` is omitted; cache hits avoid fetching those ranges, but scans still decode them; requires `scan_manager.cache.mode: cucs`).

### Materializing a pin

Pinning drives the source's `gpu_ingestible` to completion (`materialize_all_batches` / `materialize_pin_to_host` in `pin_table.cpp`):

- **Shared range step:** zone-map statistics, when enabled, are captured from the unnarrowed decoded
  table first. With `enable_compressed_materialization`, a separate exact min/max reduction then
  selects the narrowest signed, unsigned, same-scale DECIMAL, or DATE epoch-day carrier
  independently for each eligible column in that batch. The logical column-type vector is retained
  whenever either zone-map capture or narrowing needs it; it is cleared only when both features are
  disabled. See [Pin-time narrowing](compressed-materialization.md#pin-time-narrowing) for the full
  narrowing pass and its pin-cache invariants.
- **Simpatico compression:** with `CALL pin_table(..., compression => true)`, pinned chunks can additionally be stored compressed on either tier instead of (or after) narrowing — see [Compressed Pinning](compressed-pinning.md) for tier choice, plan selection, and measured results.
- **GPU tier** (`materialize_all_batches`): each resulting batch is stored as a GPU-resident `cudf::table`, round-robining placement across the GPU memory spaces. Placement is deterministic so re-pinning the same source yields identical per-chunk placement (required by the merge path below).
- **HOST tier** (`materialize_pin_to_host`): each resulting batch is converted on its round-robin GPU to a `host_data_representation` that preserves carrier type and decimal scale on that GPU's NUMA-local host space, then the GPU table is freed before the next batch. Peak GPU residency is therefore ~one batch, so a host pin never needs the whole table to fit in GPU memory.

### Storage and matching

Each pinned table is a `pinned_entry` keyed by name in the scan manager. It holds a
`cache_entry_info` (the cache identity + column layout) plus the cached batches:
`data_batches_by_column` (one chunk vector per column) for the GPU tier, or `host_chunks` (one
`host_data_representation` per batch, sliced by column at scan time) for the HOST tier. Its
`column_storage` sidecar records each chunk-column's pin-time native mapping, stored carrier, and
narrowing marker. It also carries a `pinned_zone_maps` sidecar — the pin-time per-column, per-chunk
min/max statistics, absent when capture was disabled (see [Zone maps](#zone-maps)).

`cache_entry_info` captures format identity — the resolved parquet **file set**, or the DuckDB
**catalog.schema.table plus the table's catalog object id and row-group collection identity** — plus the cached columns (by storage
index) and their names. `can_serve_with_columns(other)` returns a gather projection when this entry
can serve a scan: same format, same identity, and a **column superset** of the scan's request. A
Parquet pin never serves a DuckDB scan or vice versa. DuckDB cache serving additionally requires
every projected column's current native cuDF mapping to equal the mapping recorded at pin time; a
mismatch is a clean cache miss rather than a conversion of stale data under a new type.

The object id is the DuckDB identity's **incarnation** half. `DROP TABLE t; CREATE TABLE t (...)`
rebuilds a different table under the same qualified name, and neither the name, the column layout
nor the chunk shape tells the two apart. DuckDB gives newly created tables distinct object ids,
but preserves the id across `ALTER TABLE`. An ALTER that rewrites column values or layout replaces
the table's row-group collection, even for `ALTER COLUMN a TYPE INTEGER USING a + 100`. The cache
therefore also records a weak reference to that collection and requires it to match. An expired
reference cannot match newly allocated storage at the same address, and does not retain the old
table's memory. INSERT and DELETE preserve the collection; existing MVCC checks still govern them.
A scan of a recreated or storage-rewritten table misses the pin, as does a re-pin merge or ANN index
lookup against the old storage. Metadata-only ALTER operations that preserve storage are not
invalidated by this storage check. Whether a miss may fall through to a fresh disk-native read
depends on the table itself: that path
is MVCC-blind and reads only the checkpointed image, so the scan declines at plan time into the
transparent CPU fallback whenever the table has diverged from that image — uncheckpointed rows (the
pin's checkpoint suppression keeps them that way, leaving the dropped table's image on disk),
deleted rows the image still carries, or in-memory update chains. A `CHECKPOINT` after the recreate
folds appends and deletes into the image, and the superseded pin becomes an ordinary clean miss
served by a fresh read; a checkpoint taken *before* the recreate proves nothing. Either way
`CALL unpin_table(...)` then `pin_table` again to cache the new table.

Prepared `sirius_knn_search` statements also resolve the qualified table name in the executing
transaction's catalog before accessing the pin or ANN cache. If its catalog/storage identity
differs from the bound identity, execution fails with an instruction to re-pin and prepare again.
Refreshing only the cache key would leave the prepared output types and vector dimension stale.
A dropped table fails catalog lookup. This check applies to both exact and ANN searches.

ANN indexes additionally record a weak reference to the pin snapshot used for their build.
Unpinning, replacing a pin, or merging a re-pin invalidates that match. This matters even when
DML preserves the table's catalog/storage identity: new vectors or row positions in a new pin
must not be searched with the old index. ANN searches reject the stale index and request a rebuild
with `sirius_create_ann_index`; `use_index => false` can search the current pin immediately.
The stale index remains cached until it is rebuilt, explicitly dropped, or the cache is cleared.

During `prepare_for_query`, `try_assign_cached_entries` matches each `GPU_SCAN` operator's `table_info` against the pinned entries. On a hit it builds a `cached_databatch_provider` over the matched entry, ordering columns by the ingestible's `materialized_column_order()` so a cached batch is laid out identically to a fresh disk read and `post_filter_and_project` resolves the same columns on both paths. The provider emits one cached chunk as one resident split and bypasses the fresh-read coalescer. Each split carries whether its selected columns are actually narrow, and `GPU_SCAN` normalizes that chunk to the query's planned physical schema (or the native logical schema when there is no override) before downstream operators can combine batches.

### Re-pin semantics

For the GPU tier, `insert_pinned_entry` merges into an existing entry when that entry reads the **same source** (`same_source_as`: same table incarnation and storage, or same file set) and the row count matches — adding only columns not already cached, with per-chunk memory-space placement required to match — and replaces it otherwise. The identity half of that test is what stops a pin name reused across a `DROP`/`CREATE` from fusing two unrelated tables into one entry: an equally-sized different table passes every chunk-shape guard the merge applies. The HOST tier always replaces, since each host chunk already holds every column.

### MVCC under concurrent DML (duckdb pins)

A duckdb-format pin caches the table's **checkpointed prefix**: physical rows `[0, n_cache)` in on-disk order. `pin_table` refuses tables where that prefix is not stable — rows with in-flight update chains, or committed-but-uncheckpointed appends ("run CHECKPOINT before pinning"). After the pin, inserts and deletes remain supported; each query reconciles the cached prefix with the table's current state during `prepare_for_query`, against that query's own transaction snapshot:

- **Deletes** — a per-entry mask job walks the row groups covering the prefix with the query's transaction and builds a bit-packed keep-mask (cuDF validity convention) in pinned host memory, fanned out across scan-manager threads. Chunks with no invisible rows skip masking entirely; masked chunks apply the bitmask on device right after the resident batch materializes. A checkpoint that reshapes the row groups under a live pin makes the capture throw rather than serve drifted positions.
- **Inserts** — physical rows `[n_cache, N_total)` form the query's **insert delta**. Membership is positional; reading branches per segment: `TRANSIENT` segments (small appends, always `UNCOMPRESSED`) are copied into cuda-pinned staging at prepare time, releasing the DuckDB block pins before serving starts, while `PERSISTENT` segments (bulk appends flushed by `MergeStorage`, compressed like a checkpoint) keep their block references and decode through the normal file-read lane. Row groups the snapshot proves all-invisible are skipped whole; partially visible ones stage fully and carry a keep-mask like deleted base chunks. The delta is bundled to roughly the scan batch size, captured once per entry per query (a self-join stages one copy), and served as ordinary duckdb-native scan splits after the resident chunks — decoded on the GPU even for host-tier entries. Fixed-size `ARRAY` columns are served too: the capture walks the array-level validity plus the child element data and validity trees, staging their transient child bytes like any other column.
- **Updates** — DuckDB update chains version values in place, which the pinned cache cannot represent. Before executing an `UPDATE`, `MERGE ... UPDATE`, or `INSERT ... ON CONFLICT DO UPDATE`, Sirius checks the target against the shared pin registry and returns an error if it is pinned. This check runs for every connection and for prepared statements at execution time. Run `CALL unpin_table(...)` before updating the table.

States the delta cannot represent decline at plan time into the transparent CPU fallback: the querying transaction's own uncommitted rows (`LocalStorage`), rowid-only projections, and string columns whose statistics no longer fit the GPU string-decode limits. A manual checkpoint while a pin is live changes the database checkpoint generation; the next query refuses that pin and falls back or errors until it is unpinned and pinned again. The delta is re-captured and re-staged on every query, so a pinned table under sustained insert churn pays that cost until the next checkpoint + re-pin; delta splits also carry no zone maps, so filter pruning never drops them.

### Zone maps

By default, `pin_table` automatically captures per-chunk min/max statistics and cached scans use
them to skip chunks that cannot match a pushed-down filter. An advanced YAML escape hatch,
`sirius.operator_params.enable_pinned_zone_map_pruning`, can disable both capture and pruning for
a benchmark or diagnosis envelope. The direct DuckDB session override is test-only.

**Capture and types.** Both pin tiers run one `cudf::minmax` reduction per supported column and
chunk before the data is stored on the GPU or converted to HOST memory. Supported types are
signed and unsigned integers through 64 bits, `DATE` decoded as days, and `TIMESTAMP` decoded as
microseconds. Other or physically mismatched types, empty or all-null column chunks, and
results without valid bounds have no usable statistics and are never pruned. CUDA failures
abort the pin.

**Pruning.** During query preparation, static pushed-down table filters are checked against the
statistics with DuckDB's `CheckStatistics`. Supported filters include typed comparisons, `IN`,
`IS NULL`, `IS NOT NULL`, and safe `AND`/`OR` combinations. Missing statistics, unsupported
filters, type mismatches, and runtime dynamic filters keep the chunk. Surviving chunks still pass
through the normal GPU filter; pruning a HOST-tier chunk also avoids its H2D copy.

**Sentinel chunk.** If every chunk is proven empty, chunk 0 is still served so the scan can signal
pipeline completion; the normal GPU filter then removes its rows.

**Statless entries and re-pinning.** Disabling the option before pinning skips the extra GPU work
and creates an entry without zone maps. Turning the option on later does not retrofit statistics.
A HOST re-pin replaces the entry. For an in-place GPU merge, the re-pin must cover every cached
column and pass the normal merge-alignment checks; a strict subset cannot restore statistics.
Unpinning and then re-pinning all required columns is the safest recovery. Merging a new GPU
column while capture is disabled also drops the entry's existing zone maps.

## Batch Coalescing

When many small files (or row-group ranges) each yield a tiny GPU batch, per-task scheduling and kernel-launch overhead dominates. Coalescing is a responsibility of each `gpu_ingestible`, exposed through the `batch_coalescer` interface (`op/scan/batch_coalescer.hpp`): as the metadata side emits per-unit `scan_info`s, the sequencer feeds each one to the coalescer via `push()` (which may buffer and return zero or more ready batches) and `flush()` (remaining buffered batches at end of input).

- **`parquet_batch_coalescer`** (in `parquet_gpu_ingestible.cpp`): accumulates each file's pruned row groups into `parquet_split_info` batches sized to `approximate_batch_size` decoded column-buffer bytes, including filter-only columns. A single large file fills multiple batches; several small files bundle into one batch — but only when they share identical hive-partition values and the same pushdown decision (a mismatch on either forces a flush). It also seals a batch before it would exceed `cudf::size_type` rows. The downstream `cudf::io::read_parquet` reads all bundled slices in one invocation. Virtual-column scans are currently an exception: they read selected row groups separately and concatenate the results to preserve file and file-row provenance.
- **`duckdb_native_batch_coalescer`** (in `duckdb_native_gpu_ingestible.cpp`): accumulates row-group ranges up to `approximate_batch_size` decoded bytes, and additionally seals a batch before any VARCHAR column's accumulated bytes would cross the cuDF int32 string-offset threshold.

The coalescer runs inside the per-query sequencer (`load_balancing_scan_batch_coalescer`), so coalescing, device balancing, and prefetch hinting happen at one place per emitted batch.

## DuckDB-Native Decode

**Files:** `src/op/scan/duckdb_native_decoder.hpp`, `src/op/scan/duckdb_native_decoder.cpp`, `src/cuda/scan/gpu_native_decode.cuh`, `src/cuda/scan/gpu_decode_strings.cuh`, `src/cuda/scan/*.cu`, `src/cuda/scan/strings/*.cu`

The GPU DuckDB-native scan reads a table stored in DuckDB's own `.db` block format and decodes each projected column's on-disk segments directly on the GPU into a `cudf::table`, without going through Parquet. `decode_duckdb_native_split()` takes the row-group metadata for a split, stages the segment bytes on device, and dispatches per-column decode.

Fixed-size DuckDB `ARRAY` columns with supported fixed-width children decode as cuDF `LIST` columns. An empty or fully pruned split preserves that nested schema with an empty `LIST` column; variable-length `LIST` and `STRUCT` remain unsupported.

### Segment runs and codec dispatch

A DuckDB column inside a row group is a sequence of *segments*, each compressed with one codec. The decoder groups a column's segments into **codec runs** — maximal spans of segments sharing one codec — and a per-codec kernel consumes a whole run in one launch (the run is the batching unit). A column with mixed codecs produces multiple runs. Codec metadata (bitpacking width, dictionary references, FSST symbol tables, ALP parameters) lives inside the segment bytes; each codec kernel parses its own headers on device, so the dispatcher itself does no parsing and no I/O.

Decode splits into two entry points sharing the same run/segment descriptors:

- **Fixed-width columns** — `gpu_decode_table()` (`gpu_native_decode.cuh`). Codecs: `UNCOMPRESSED`, `CONSTANT`, `RLE`, `BITPACKING`. Floating-point columns additionally decode DuckDB's `ALP`/`ALPRD` adaptive-lossless codecs. The dispatcher synchronizes the stream once before returning so columns come back with `null_count` populated.
- **Varchar columns** — `gpu_decode_strings_column()` (`gpu_decode_strings.cuh`). Codecs: `UNCOMPRESSED`, `DICTIONARY`, `FSST`, `DICT_FSST`. The per-segment max-string-length stat captured during the metadata walk sizes the cuDF chars buffer up front, so the string path runs async modulo at most one host sync (the chars-buffer read-back, which only fires when that upper bound is unknown or pathological).

Each string codec lives in its own translation unit under `src/cuda/scan/strings/` (`uncompressed.cu`, `dictionary.cu`, `fsst.cu`, `dict_fsst.cu`) over shared device primitives in `src/cuda/scan/detail/` and `src/cuda/scan/strings/`; the fixed-width codecs live in `src/cuda/scan/` (`gpu_decode_bitpacking.cu`, `gpu_decode_rle.cu`, `gpu_decode_alp.cu`, `gpu_native_decode.cu`).

### Validity and rowid

Validity (null) masks decode from `UNCOMPRESSED`, `EMPTY` (all-null), and `CONSTANT` (all-valid) codecs on device. `ROARING`-compressed validity is host-decoded to a plain bitmap before the GPU sees it (DuckDB's roaring scan state drives the reads), then staged like any other host-produced segment. `CONSTANT` data segments likewise materialize their single value on the host. Synthetic `rowid` columns carry no on-disk storage; the decoder fills them from each row group's absolute first-row index.

### Viability gate

The metadata walk (below) rejects any codec or type the decoder cannot handle and falls the query back to DuckDB CPU before staging. The decoder's own `throw`s on unsupported codecs/types are a defensive backstop, not the primary gate. Unsupported cases include 128-bit (`HUGEINT`/`DECIMAL128`) and nested (`STRUCT`/`LIST`) types — with the exception of `ARRAY` (fixed-size lists, mapped to cuDF `LIST`), which the native path decodes when the element type is fixed-width (a VARCHAR or nested element falls back) — plus two independent varchar refusals: a column that may contain a *single* string at or above DuckDB's overflow-block limit (`StringUncompressed::GetStringBlockLimit`, 4 KB at the default block size — such strings live in overflow blocks behind a `BIG_STRING_MARKER` the GPU decoder cannot follow), and a column whose *summed* max-string-length upper bound would overflow cuDF's int32 string-offset limit. An absent max-string-length statistic is itself a refusal, since overflow strings cannot be ruled out. The overflow-block check is applied at three layers — plan time (`sirius_plan_get.cpp`, from `StringStats::MaxStringLength`, giving the cleanest CPU fallback), the serial prepare phase, and per-segment during the range walk.

## DuckDB-Native Metadata Walk

**Files:** `src/op/scan/duckdb_native_metadata.hpp`, `src/op/scan/duckdb_native_metadata.cpp`, `src/op/scan/duckdb_native_decoder.hpp`

Before decode, the scan walks DuckDB's storage metadata to learn, per row group, which segments each projected column occupies, what codec each uses, and how many decoded bytes the row group will cost. The walk is two-phase so the thread-unsafe parts run once and the expensive per-segment parsing runs concurrently.

### Phase 1 — `prepare_duckdb_native_walk()` (serial)

Runs once during execution preparation under a shared checkpoint lease. This work stays on the query thread because `ClientContext`/`LocalStorage` are not thread-safe:

- Reads `PartitionStatistics` for every row group (the source of each row group's absolute first-row index and row count, used both for rowid synthesis and decoded-byte budgeting).
- Gates the projected types: an exhaustive type switch refuses 128-bit and nested types up front — except fixed-size `ARRAY` with a supported fixed-width element, which is admitted and decodes as cuDF `LIST` — so an unsupported projection becomes a clean CPU fallback before any per-segment IO.
- Marks row groups that pushed-down filter statistics prove empty (see **Row Group Pruning**).

The result is a `duckdb_native_walk_plan` carrying per-row-group row starts/counts, the block size, the pruned-row-group bitmap, and the inputs the range walks need. A non-viable plan (unsupported type, invalid partition `row_start`, or the varchar overflow-block refusal) refuses the whole native-scan path.

### Phase 2 — `walk_duckdb_native_row_group_range()` (concurrent)

The row-group range `[0, n_row_groups)` is sliced into fixed internal chunks of eight row groups, and each chunk is walked independently on a scan-manager thread. For each surviving row group in its range, a range walk:

- Walks each projected column's **typed segment trees** directly — reading `block_id`, block offset, compression enum, per-segment row counts, the validity child's segments, and (for varchar) the per-segment max-string-length stat as typed fields. It does not build or re-parse the per-segment string blobs that DuckDB's generic `GetColumnSegmentInfo` would produce.
- Refuses on the first unsupported segment codec or an absent/over-threshold varchar stat, partially filling the range (which the caller then discards in favor of CPU fallback).
- Derives each segment's on-disk byte size from the sorted `(block_id, block_offset)` delta to the next segment — an upper bound, since codec headers self-bound the actual reads, so any overshoot only inflates staging/H2D bytes, never correctness.
- Sorts each column's segments by row start (so codec runs coalesce) and computes the row group's decoded-byte budget and per-column varchar char count.

Stats-pruned row groups are skipped before any segment metadata is requested. The per-row-group results (`duckdb_row_group_metadata`) feed the batch coalescer, which bundles row groups into decode-sized splits.

## Row Group Pruning

Both GPU scan formats drop row groups that a pushed-down filter proves cannot contain a matching row, before those row groups are read or decoded. Pinned tables get the same treatment at chunk granularity — see [Zone maps](#zone-maps).

### Parquet path

When filter pushdown is enabled and the `gpu_expression_translator` successfully converts DuckDB `TableFilterSet` filters into a cuDF AST, three mechanisms activate inside `parquet_gpu_ingestible`:

1. **Row group statistics pruning:** during the per-file metadata task, `filter_row_groups_with_stats()` runs against each fetched footer; row groups whose Parquet column min/max statistics cannot match the filter are dropped before any read is scheduled. Pure hive-partition filters are dropped during plan construction since hive columns aren't in the parquet file.

   Only the stats-safe part of the predicate is handed to the stats filter. `is_unsafe_for_stats_filter()` rejects any expression containing a null test, and a bare column reference used directly as the predicate (`WHERE flag`) — both fault inside cuDF's statistics rewrite rather than merely mis-pruning. `conjuncts_without()` keeps only the safe top-level AND conjuncts (so `v IS NULL AND id > 3000` still prunes on `id`); a compound conjunct containing anything unsafe is dropped whole. Dropping conjuncts is sound: it can only retain extra row groups, and the full predicate is still applied at read/post-decode time.

2. **Null-count pruning:** a second statistics pass prunes on `null_count`: an `IS NULL` predicate drops row groups with `null_count == 0`, and `IS NOT NULL` drops row groups where every row is null (`null_count == rg.num_rows`); an absent `null_count` stat keeps the row group. The predicates come straight from the `TableFilterSet` (not the converted AST — see the translation-path note below), only from conjunctive positions, and only for scalar leaf columns (nested/legacy-repeated schemas are skipped).

3. **Reader-level filter pushdown:** the cuDF AST is set on `parquet_reader_options` via `set_filter()`, so cuDF applies the filter inside `read_parquet`. When the reader applies the row filter, `materialize_table` reports `ROW_FILTERED` (or `ROW_FILTERED_AND_PROJECTED` once hive partitions are assembled), so the scan operator skips a redundant post-decode filter.

   Before cuDF 26.12, the reader's bloom filter probe is wrong for some column types (rapidsai/cudf#24319). The reader filter (`_reader_pushdown_expression`) therefore leaves out equalities on the types listed by `has_unreliable_bloom_filter_probe()`, and they are applied post-decode. From cuDF 26.12 the list is empty.

Reader-side pushdown is a per-split decision: an FLBA-decimal safety probe can disable it for a file, in which case the cached DuckDB filter expression is evaluated through `expression_evaluator` on the decoded batch in `post_filter_and_project`.

Virtual-column scans currently disable reader-side row filtering, including dynamic-filter AST merging. Row-group statistics pruning remains enabled, but all rows in selected row groups are decoded before residual predicates and membership filters run—even for a selective `WHERE filename = ...`.

Follow-up work can use cuDF 26.08.01's existing `enable_prepend_source_index_column()` and `enable_prepend_row_index_column()` APIs to preserve file and row identity during whole-split reads with filtering. Source selection for `filename`/`file_index` predicates and Iceberg positional-delete handling need separate integration.

**Filter translation path:** `TableFilterSet` -> `convert_table_filters_to_expression()` (skips `OPTIONAL_FILTER` and partition-column filters) -> `gpu_expression_translator` -> cuDF AST tree. Ordinary scans also skip top-level `IS_NOT_NULL`; virtual-column scans retain it in the residual predicate evaluated after decoding. Null-count pruning collects null-test predicates directly from the original `TableFilterSet`, independently of expression translation.

### DuckDB-native path

The DuckDB-native scan prunes row groups using DuckDB's own statistics machinery rather than Parquet footer stats. During the serial prepare phase of the metadata walk (`mark_row_groups_pruned_by_filter_stats`), for each row group and each pushed-down `TableFilter`, the scan calls DuckDB's `TableFilter::CheckStatistics` against that row group's per-column statistics (obtained from its `PartitionRowGroup` handle). A row group is pruned the moment any filter returns `FILTER_ALWAYS_FALSE`.

This runs entirely from `PartitionRowGroup` statistics — no segment metadata is needed — so a pruned row group is skipped **before its segments are walked, staged, copied to the GPU, or decoded**. If every row group is pruned, the scan stays on the GPU path: the coalescer emits one schema-correct 0-row split, so the pipeline completes with an empty result.

Only statically-known DuckDB `TableFilter`s participate in this DuckDB-native metadata walk. DuckDB `DYNAMIC_FILTER` entries are excluded because Sirius runtime dynamic filters use a separate `sirius_dynamic_filter_set` channel and their own scan-consumer paths — the parquet reader's `set_filter` and the post-decode `DYNAMIC_FILTER` operator (the duckdb-native scan consumes post-decode only) — described in [Dynamic Filters](dynamic-filters.md); they are not translated through this static `TableFilterSet` path. This metadata separation does not imply one universal execution order: the producing join's immediate probe scan starts after build-port publication, while a base scan reached transitively through an intervening join may materialize early splits before publication and samples the channel at its per-split checkpoints. See [Transitive scan targets and publication timing](dynamic-filters.md#transitive-scan-targets-and-publication-timing). The payoff from the static statistics walk is data-clustering-dependent — it costs almost nothing when statistics cannot help and is multiplicative when the table is ordered such that a filter eliminates most row groups.

## IO Layer (cuCascade io)

**Files:** Sirius glue in `src/io/` (`path_utils.{hpp,cpp}`, `ioctx_resolver.hpp`, `parquet_helpers.{hpp,cpp}`, `s3/sirius_httpfs.{hpp,cpp}`) and `src/scan_manager/` (`sirius_scan_manager.cpp`, `uring_gauges_sampler.{hpp,cpp}`); the io library itself is cuCascade's, under `cucascade/include/cucascade/io/` plus `cucascade/include/cucascade/cudf/datasource.hpp`

Sirius does not build its own IO stack. The `cudf::io::datasource` implementation, the backends (io_uring, REST/S3, kvikIO), the pinned file cache and the path→backend registry come from the io library of the cuCascade submodule (`cucascade::io`, built in-tree with `CUCASCADE_BUILD_IO=ON` and linked as `cuCascade::cucascade_io`; the `cudf::io::datasource` bridge itself lives in the cudf layer, `cuCascade::cucascade_cudf`). Sirius code names these types explicitly (`cucascade::io::ioctx`, `cucascade::io::datasource`, ...); `sirius::io` holds only the glue listed above. This section covers how Sirius uses the layer. For its internals — the reactors, per-backend behaviour, the cache's state machine — the reference is the doc comments in cuCascade's headers under `cucascade/include/cucascade/io/` (in the `cucascade/` submodule). The layer is read-only: it has no write API.

### Architecture

Header paths below are relative to `cucascade/include/cucascade/`.

| Component | Header | Role |
|-----------|--------|------|
| `cucascade::io::datasource` | `cudf/datasource.hpp` | `cudf::io::datasource` implementation bound to one context and one opened `io_object`. Routes each read through the context's `fs_cache` when it has one, otherwise straight to the context. Also carries the per-scan `cache_handle` that `fadvise` registers. |
| `ioctx` | `io/io_context.hpp` | Abstract backend context, shared by every scan and every GPU. Owns the optional `fs_cache` (`initialize_cache`) and an always-present `metadata_store`; exposes host and device reads over one prepared-slice backend hook (`mixed_readv_async_io`); `start()` / `shutdown()` start and stop its reactors. |
| `templated_ioctx<Reactor>` | `io/templated_ioctx.hpp` | The context over a backend reactor (uring, REST). Owns the pool of reactors. `next_reactor` picks at most two of them for a read; the read's slices are split by bytes into one grouped request per picked reactor, all sharing one `grouped_coordinator`, and each is pushed onto its reactor's queue. |
| `io_context_registry` | `io/datasource_factory.hpp` | Path→backend registry, built from `scan_manager_config::to_io_config()`. Registers uring (claims existing regular files), REST (`s3://`) and the kvikIO catch-all itself. `lookup_path` resolves a path to a backend type (an explicit backend wins over the catch-all; under `backend: kvikio`, kvikIO also takes local and `s3://` reads); `make_ioctx(type)` builds a context and returns null when the backend's factory fails or declines (for example REST with an unconfigured object store). |
| `cache::fs_cache` | `io/cache/fs_cache.hpp` | Pinned-memory chunk cache that replaced Sirius's `prefetching_cache`: lock-free per-chunk state machine, caller-driven prefetch and a background evictor. Serves partial reads and populates itself on read. Built only under `cache.mode: cucs`. |
| `cache::metadata_store` | `io/cache/metadata_store.hpp` | Per-object metadata cache keyed by `io_object::raw_file_cache_id()` (the path for local files; path plus strong ETag for REST objects, see [Object identity](#s3--object-store-backend)). It keeps one generation per path. Always present, independent of the `fs_cache`; Sirius parks parsed parquet footers here so a later scan of the same object generation skips the parse. |
| `semi_future` / `try_t` / `completion_controller` | `exec/` | Async primitives the io layer is built on. Sirius's `src/exec/` headers of the same names are alias headers that re-export `cucascade::exec`. |

### Reactor model

The uring and REST contexts are `templated_ioctx`s over a pool of *reactors*. A reactor is one worker thread plus its own backend state and request queue: for uring, one io_uring ring, 64 MiB of pinned staging in whole host blocks (at most 64 slots, at least one block) and a two-tier priority queue; for REST, a libcurl multi handle on an epoll loop with up to 64 connections and a single FIFO queue. There is no shared queue across reactors.

- `ioctx::start()` starts each reactor in turn — allocating its staging and launching its worker thread — so a failure such as a pinned staging allocation propagates out of `start()`. `uring_n_reactors` reactors serve uring (Sirius default 4) and `rest_n_reactors` serve REST (default 2).
- **Routing.** `next_reactor` takes the next starting point of a rotating counter, ranks the reactors by `queued_bytes()` (approximate bytes not yet expanded into physical operations) and returns the least backlogged two; rotation breaks ties so synchronous reads do not stick to reactor zero. A multi-slice read is balanced by bytes across the reactors returned (whole slices), a single-slice read goes to the first.
- **Per-reactor loop.** A uring reactor takes one request off its queue at a time (high tier first, see below) and works through it in order. Each loop pass it expands at most `slices_per_pass` slices of that request into physical operations, submits them, then waits for completions (a 20 ms poll tick when nothing completes). A future resolves, and its callbacks run, on the reactor thread that completed the request's last operation.
- **Failure.** A reactor whose loop hits a fatal error closes its admission and fails its in-flight, active and queued requests with that error; it is not restarted, and later requests routed to it are failed as canceled. The other reactors keep serving, but `next_reactor` does not skip the dead one (its near-zero backlog makes it a likely pick), so a share of reads keeps failing until the process restarts. Sirius neither detects nor restarts a dead reactor.

**Read priority.** Every read API (`ioctx`, `fs_cache`, and the datasource's `*_read_ranges_async`) takes an `io_priority` — `automatic` (the default), `high` or `low` — that picks the uring reactor queue tier. `automatic` resolves by call shape: device reads and single-range host reads (what an executor thread is blocked on) go `high`; multi-range host reads (`host_readv_async_io`, `host_read_ranges_async`, so the DuckDB-native decoder's column-chunk reads) and the `fs_cache` readahead prefetches go `low`. A uring reactor's queue is two lock-free moodycamel queues sharing one lightweight semaphore (`cucascade::io::tiered_blocking_queue`): the worker always takes a high request before a low one, and when a high request arrives while a low one is active it parks the low request at the next slice boundary, runs the high one, then resumes the parked request before taking any other low request. Physical reads already planned for the parked slice still complete first, so a high read waits behind at most one slice of low work plus what is already in flight. REST reactors carry the priority but keep a single FIFO queue.

**Range batching.** Because a local read does not prefer bulk I/O (`prefers_bulk_io() == false`), `mixed_readv_async_io` splits a host-only multi-slice read into queue entries of at most `uring.range_batch_slices` slices (default 8) after balancing it across the two selected reactors; all entries share one coordinator, so the caller still sees one future. Reads with a device slice are never split.

### Sirius wiring

- **Configuration.** `scan_manager_config` embeds cuCascade's config structs (`uring`, `rest`, `kvikio`, `cache`, `object_store`); `to_io_config()` copies them and the reactor counts into the `cucascade::io::io_config` the registry is built from, maps `backend` (`native` / `kvikio`) onto cuCascade's enum, and applies the cache mode. `uring_n_reactors` defaults to 4 and `uring.slices_per_pass` to 4 in both; `uring.n_max_concurrent_scans` is 0 in cuCascade, and Sirius re-defaults it to the pipeline pool size unless the config names it (REST: max(8, 2x the pool)).
- **Contexts.** The scan manager owns the registry. Its constructor builds and starts the default context — uring for `backend: native`, kvikIO for `backend: kvikio` (single-GPU only) — and throws if that fails. Contexts for other backends, such as REST for `s3://`, are built on first use (`ioctx_for_type` / `ioctx_for_path`), started, and then shared by every query and GPU. Each context gets its `fs_cache` when it is built (`init_cache_for`) when `cache.mode` is `cucs` and the backend can use one.
- **Backend failures.** cuCascade's logging is compiled out and its registry reports a throwing factory only as a null context, so Sirius derives the likely cause itself (`explain_ioctx_failure`): for REST, the empty `object_store` fields (a WARN, since REST is then disabled by configuration); for uring, a missing HOST-tier memory space or an invalid `uring.*` setting. A context built on first use whose factory fails is reported once and its backend then resolves to no context; a `start()` failure on such a context propagates to the caller and the next use retries. For the default context the cause goes into the thrown exception. A requested `fs_cache` that cuCascade declined to build is reported with a WARN.
- **Opening files.** Every Sirius open goes through `sirius::io::open_datasource(io_ctx, path[, hint])` (`src/io/path_utils.hpp`). It normalizes the path with Sirius's own `strip_file_scheme`, which parses a `file:` URI with DuckDB's `Path` (strips the scheme, folds `.`, `..` and empty segments, percent-decodes; any other path comes back byte-identical), and then calls `cucascade::io::open_datasource`. The `fs_cache` and the `metadata_store` key on the path (plus the ETag for REST objects), so this keeps one file under one key. cuCascade's own `strip_file_scheme` strips the scheme and percent-decodes but does not fold segments.
- **`[uring_gauges]` sampler.** When the default context is a `uring_ioctx`, the scan manager owns a `uring_gauges_sampler` (`src/scan_manager/uring_gauges_sampler.hpp`): a thread named `uring_gauges` that wakes every 250 ms and, only while the log sink accepts DEBUG, polls `uring_ioctx::reactor_gauges()` and logs one line per non-idle reactor — `[uring_gauges] reactor=… inflight=… max_inflight=… pending_ops=… active_slices=… queued_requests=… queued_low=… queued_MiB=… started=… preemptions=… MiB_s=…`, then the high-tier (`hq_*`) and low-tier (`lq_*`) queue-delay windows: count, sum and max in ms, and the non-empty log2-µs histogram buckets. Above DEBUG it costs a timed wakeup and does not reset the reactors' gauge windows. The scan manager stops it in `stop()`, before the contexts are released. REST contexts have no gauges. The default context's startup DEBUG line reports `n_reactors`, `slices_per_pass` and `range_batch_slices`.

### Read path

A scan resolves each file to a context — `split_provider` receives `ioctx_for_path` as a `sirius::io::ioctx_resolver` (`src/io/ioctx_resolver.hpp`) — and opens it with `sirius::io::open_datasource`. The backend creates an `io_object` (a local file's descriptors; for S3 a HEAD, or a suffix GET under the footer-probe hint) and `cucascade::io::open_datasource` wraps it in a `cucascade::io::datasource` bound to the context. Each cuDF read on the datasource forwards to the context:

- With an `fs_cache`, `host_read` / `device_read` ask the cache to classify every requested chunk. Resident pieces are copied from pinned host chunks, loadable pieces become cache-backed `prepared_io_slice`s, and gaps become ordinary prepared slices; the mixed batch is then dispatched through the context's asynchronous backend hook.
- Without one, the read becomes prepared slices directly. That is the case under `cache.mode: none` (uring reads with `O_DIRECT`), under `cache.mode: os` (uring reads buffered through the OS page cache), and always for kvikIO. Under `os` the readahead stays enabled by configuration, but it has no pinned cache to fill: `fadvise` does nothing on a context without an `fs_cache`.

One `grouped_coordinator` owns the future of a logical read across all its grouped requests and physical operations. The future resolves, or reports the first error, only after every published operation has settled. A cache fill publishes its chunks when its own operation completes, so a later failure does not discard completed cache data.

### Backends

| Backend | Context | Scheme | Notes |
|---------|---------|--------|-------|
| io_uring | `uring::uring_ioctx` (a `templated_ioctx<uring_reactor>`) | local files | `uring_n_reactors` reactors, each one worker thread with one ring and a 64 MiB pinned staging budget. A reactor chooses 256 KiB–16 MiB operations; compatible operations use `O_DIRECT`, with buffered fallback for unsupported or misaligned remainders (and buffered reads throughout under `cache.mode: os`). |
| REST / object store | `rest::rest_ioctx` (a `templated_ioctx<rest_reactor>`) | `s3://` | `rest_n_reactors` reactors, each one worker thread with a libcurl multi handle and up to 64 connections; 4–16 MiB GETs. See [S3 / Object-Store Backend](#s3--object-store-backend). |
| kvikio fallback | `kvikio_context` | any (catch-all) | Wraps kvikIO local/remote handles (GDS-capable for local files). It has no reactors and no `fs_cache`, and serves the prepared-slice hook eagerly on the calling thread, one slice after another. |

A new backend is added in cuCascade (a reactor satisfying its `io_reactor_c` concept, plus a registry entry); on the Sirius side it then needs a case in the switches over `io_context_type`.

### S3 / Object-Store Backend

**Files:** Sirius: `src/io/s3/sirius_httpfs.{hpp,cpp}`, `src/scan_manager/sirius_scan_manager.cpp` (routing, `describe_parquet`), `src/op/scan/parquet_gpu_ingestible.cpp`; cuCascade: `cucascade/include/cucascade/io/rest/`

cuCascade's `rest::rest_ioctx` handles `s3://` paths for AWS S3 and compatible stores such as MinIO. The `gs://` and `azure://` schemes that cuCascade's `uri_parser` parses have no dedicated backend (only the kvikIO catch-all would claim them). DuckDB uses Sirius's read-only `sirius_httpfs` to bind transparent `read_parquet('s3://...')` queries, while scan-manager callers open the same path directly through the registry. S3 scans require GPU execution and have no DuckDB CPU fallback.

**Key semantics.** The key portion of an `s3://` URI is literal text. `uri_parser` does not percent-decode it or split it at `?` or `#`; SigV4 applies RFC 3986 encoding when it builds the request, matching AWS CLI behavior. So `s3://bucket/my%20file.parquet` addresses the key `my%20file.parquet`; use an actual space to address `my file.parquet`.

DuckDB still decodes Hive partition values, so `col=a%20b/` yields `a b`. Glob syntax is unchanged: `?` in a pattern is still a wildcard. A concrete key whose directory segment contains both `=` and a literal `?` is rejected, because DuckDB would drop that partition column; encode it as `%3F`. A `?` in the final filename is allowed.

**Opening objects.** A generic open issues a blocking HEAD to obtain the object size. A `parquet_footer_probe` open uses a suffix-range GET to obtain both the size and the footer bytes; those bytes stay on the resulting `rest_io_object` and serve the binder's footer reads. If the suffix response cannot be used, the open falls back to HEAD. Parsed footer metadata is stored separately in the context's `metadata_store`.

**Object identity.** The open's strong ETag (RFC 7232 §2.3) is its cache-version validator: the `rest_io_object`'s cache id is `path + 0x1F + ETag`, so an overwrite that changes the validator gets a different cache *generation* and shares neither cached bytes nor parsed metadata with its predecessor, while an identical strong tag maps to the same generation. An open whose ETag is missing, weak (`W/`), `*`, or otherwise malformed gets an id unique to that open: it caches within itself only. So does every open with a known size, which is how list-driven (globbed) parquet files are opened: LIST does not carry the ETag into the open, so those opens share no cached bytes or parsed footers across opens and their GETs are unconditional. `rest_io_object::is_strong_tag()` and `generation_key()` expose the rule.

Every range GET for a strong-validator open carries `If-Match: <ETag>`. On an accepted data response (`206`, or a full-object `200`) the response ETag must equal the open's; a `412`, or a missing, weak, or different tag, fails the read with `cucascade::io::object_changed_error` (`object_path()`, `expected_tag()`, `observed_tag()`). Nothing from such a response is published, the failure is terminal rather than retried, and a transient retry (`503`) resends the same condition. The parquet scan reports it as a `reader_io` failure, like a `credential_error`. Opens without a strong validator read unconditionally and offer no snapshot consistency across their GETs. A prefetch that fails this way keeps its exception on the request; `cucascade::io::datasource::prefetch_failure()` returns it, already set when the completion callback runs. Cache hits and footer-stash hits do not revalidate against the store.

**Reads.** Each REST reactor runs its own worker thread over a libcurl multi handle on an epoll loop with a pool of easy handles, up to 64 connections (`rest::config::max_connections`, not a YAML key). The reactor claims free connections before expanding a logical slice. It chooses a 4–16 MiB physical target from the queued backlog and its free connections; contiguous slices are balanced under the 16 MiB ceiling, while fragmented cache fills are grouped only at whole cache-chunk boundaries. The response's `Content-Range` and exact byte count are checked before an operation completes. HEAD, LIST, and footer-probe requests are blocking requests on the caller thread; they share DNS and TLS-session caches with the reactors.

S3 has no direct-to-device transport in this backend. Device reads allocate enough page-aligned, pinned CuCascade blocks for each physical operation, scatter the GET into them, then issue `cudaMemcpyAsync`. The staging owner remains alive through retries and until a CUDA event reports completion; only then are cache chunks published and the coordinator credit settled. Reads into caller-provided contiguous host memory do not allocate staging.

**LIST and glob expansion.** `sirius_httpfs` expands S3 globs with paginated `ListObjectsV2` requests. It sends the longest static directory prefix to S3, then applies the remaining pattern locally. `*`, `?`, and `[...]` match within one path segment; a segment equal to `**` can cross directories. Bucket wildcards are rejected.

Pages are processed as they arrive, so listing memory is bounded by one page plus the matches retained for DuckDB. `list_max_scanned` limits inspected objects and `list_max_matches` limits retained matches. Reaching either limit raises an error instead of returning an incomplete file list. Results are sorted by URI, and LIST metadata lets globbed parquet files use the footer-probe open without a separate HEAD.

**Authorization.** cuCascade's `request_authorizer` signs each request attempt and returns the request URL and headers. LIST uses the separate `authorize_list` entry point, which custom authorizers must implement if they support glob expansion.

Sirius registers its own `CREATE SECRET (TYPE SIRIUS_S3, ...)` type, so this workflow does not require DuckDB's httpfs extension. It also accepts httpfs `TYPE S3` secrets when that extension is available. For each S3 bind, open, and glob, Sirius selects the best path-scoped `SIRIUS_S3` secret first, then an `S3` secret if no `SIRIUS_S3` secret matches, then the programmatically set `object_store_config` if neither matches. An invalid matching secret or failed S3 request is an error, not a reason to try another credential source. A selected secret must use `PROVIDER CONFIG` and contain non-empty `KEY_ID` and `SECRET`; partial, refresh-enabled, and non-static secrets fail without falling back to the programmatic config. `SESSION_TOKEN`, `REGION`, and `ENDPOINT` also come only from that selected secret: missing session token means no token, missing region defaults to `us-east-1`, and missing endpoint derives the regional AWS endpoint. `USE_SSL` and `VERIFY_SSL` map to endpoint transport and TLS certificate verification. Secret replacement therefore affects subsequent S3 operations on the same connection. Resolved config snapshots retain the credential fields needed by REST signing; Sirius does not retain the DuckDB secret object or its catalog name in bind data. Sirius currently supports only `URL_STYLE 'path'`; request-changing options such as requester-pays, proxies, extra headers, SSE/KMS, and URL compatibility mode are rejected when enabled. Other explicit URL styles fail clearly. Secret values are not included in Sirius diagnostics.

For example, a scoped Sirius secret supplies credentials directly to a normal Parquet scan:

```sql
CREATE SECRET project_s3 (
  TYPE SIRIUS_S3,
  PROVIDER CONFIG,
  SCOPE 's3://analytics-bucket/curated/',
  KEY_ID 'ACCESS_KEY',
  SECRET 'SECRET_KEY',
  REGION 'us-west-2'
);

SELECT count(*)
FROM read_parquet('s3://analytics-bucket/curated/events.parquet');
```

| Authorizer | Mechanism |
|------------|-----------|
| `sigv4_presigned_authorizer` | SigV4 credentials in the query string. This is the default. |
| `sigv4_header_authorizer` | A plain URL with signed `Authorization` and `x-amz-*` headers. |

Both authorizers use path-style URLs and support temporary credentials. The session token is signed as a header in header mode and as a query parameter in presigned mode. Custom authorizers can use another credential source or return broker-issued URLs.

**Configuration.** C++ callers can set `cucascade::io::object_store_config` through `sirius_config::set_object_store_config()` before scan-manager initialization. It supplies an in-memory endpoint, region, static credentials, optional session token, signing mode, and TLS settings when no scoped secret matches. This configuration cannot be loaded from YAML. `TYPE SIRIUS_S3, PROVIDER CONFIG` secrets supply a static key pair and optional session token without httpfs; httpfs `TYPE S3, PROVIDER CONFIG` secrets work too when installed. Credential-chain providers, SSO, automatic refresh, environment variables, AWS profiles, and IMDS are not consumed by Sirius. A custom authorizer can implement those sources. If the selected secret is incomplete, Sirius reports an error; without a secret, missing fallback endpoint, region, or static keys leaves no REST context, the scan manager logs which `object_store` fields are empty, and the S3 read fails.

**Scoped configs.** A resolved secret is installed into the scan manager's `sirius::io::scoped_object_store_configs` (`src/io/scoped_object_store_configs.hpp`) under the path's scope, as an immutable snapshot with its own id. Routed contexts are keyed by backend type and snapshot id, so paths with different credentials get different REST (or kvikIO) contexts. The registry holds only the default config, so a scoped context is built through a throwaway `io_context_registry` over a copy of the scan-manager config with the scoped `object_store`; the registry's constructor only registers the backend factories. Replacing a scope's secret publishes a new id and retires the superseded context, which in-flight users keep alive. A scoped build that fails is reported once per snapshot.

Connection limits, the logical merge-gap hint, footer-probe size, retry budgets, keepalive, and LIST caps live in cuCascade's `rest::config` (defaults in `cucascade/include/cucascade/io/rest/config.hpp`). Physical request sizing belongs to the reactor rather than the configuration. `request_timeout_s` is also used as the lifetime of a presigned URL. Async data requests retry transient curl and HTTP failures, with a separate bounded retry for HTTP 403. Control requests treat HTTP 403 as terminal.

### Cache Seam

Under `cache.mode: cucs`, each context whose backend can use it gets a `cache::fs_cache`, and `cucascade::io::datasource` forwards reads to it. The cache resolves hits immediately, claims missing chunks, and builds `prepared_io_slice`s that describe the caller's logical range plus contiguous host memory, cache fragments, and/or a device destination. It dispatches misses through the context's asynchronous backend hook; reactors see only this prepared buffer shape and a per-slice completion callback, not cache lookup policy.

A cache is attached only when the backend can benefit from it — `ioctx::can_use_fs_cache()` is true iff the backend supports vectored host reads or staged host-to-device reads (uring and REST do, kvikIO does not). The backend's staging block size must also equal the cache chunk size; otherwise cuCascade runs the context without a cache, and Sirius logs a WARN.

The cache does two things beyond classic prefetch:

- **Partial reads.** A `device_read` over a range whose chunks are only partially cached copies the cached chunks straight to device and completes the rest from the backend in the same call, instead of treating a partial overlap as a miss.
- **Populate-on-read.** On a backend that supports staged host-to-device reads, an uncached chunk being read for the device can be loaded into a cache buffer (file → cache chunk → device) and published to the cache, so the next read of the same chunk is a hit — caching as a side effect of reading, like an OS page cache. A piece the cache cannot load is read through the backend's own staging and left uncached.

Separately from the `fs_cache`, every context exposes a `metadata_store`, so parsed file metadata (e.g. a parquet footer) survives across scans of the same object generation whatever the cache mode. The store keeps one generation per path — a registration replaces whatever the path held — and `has_path()` is only an *open hint* (Sirius skips the footer probe when some generation's footer is known); the exact-key lookup (`get_metadata(io_object)`, keyed by `raw_file_cache_id()`) decides whether the footer is reused. A lookup by bare path misses strong-ETag REST objects.

**fadvise protocol.** `cucascade::io::datasource::fadvise(ranges, dev_id)` registers a scan range set and stashes its `cache_handle`. The readahead manager later drives that handle through allocation and asynchronous prefetch, which the cache issues as `prefetch`-class reads; the consumer stage records when reading begins so stale work can be refused or awaited instead of issuing duplicate I/O.

### Cache Internals

These are cuCascade's; see `include/cucascade/io/cache/` and `src/io/cache/fs_cache.cpp` in the submodule.

- **Chunked, pinned buffer pool.** The cache caches fixed-size *chunks* of pinned host memory drawn from a `buffer_pool`, which allocates per-NUMA arenas from `fixed_size_host_memory_resource`s. The chunk size is the resource's block size (not a compile-time constant). Staging buffers for a prefetch are placed on the NUMA node closest to the target GPU, derived from the shared `topology_index`.
- **Packed atomic state machine.** Each `cached_chunk` carries a `chunk_state` that packs its 4-bit lifecycle, reader pins, populated extent, and live-subscriber count into one `atomic<uint64_t>`. Every transition is a single CAS, closing the TOCTOU gap between checking coverage and pinning or claiming a load. Before planning a read, the cache considers only handle chunks that overlap that read. It waits when an overlapping `loading` chunk belongs to the active handle prefetch; otherwise a device read uses the reactor's own staging, so unrelated and demand-owned loads do not serialize behind the prefetch.
- **Request fan-in.** A `grouped_coordinator` (`io/io_request.hpp`) tracks the physical operations for one logical request and fulfills one `semi_future` after they settle, reporting the first error safely. A physical operation publishes only its completed cache chunks, so successful segments remain cached after a later segment fails.
- **Execution and teardown.** Preparation and prefetch dispatch are driven by the readahead scheduler; the cache owns one background evictor. Cache-backed I/O and cached H2D completion retain teardown credits. Cache-hit H2D batches use one stream-ordered ticket in `cuda_event_completion_poll` instead of allocating one CUDA event per read; bounded completion waiters drain those tickets and settle futures outside CUDA callback context. Destruction closes admission and drains those credits before cache generations or pinned buffers are released.
- **No admission control in the cache.** Issuing a prefetch never blocks the calling thread: how much read-ahead is in flight is decided by `readahead_scan_manager`'s scan budget (`scan_manager.max_readahead_scans`, see [Configuration](configuration.md#scan-manager--io-configuration)), not throttled a second time here. The cache takes an `exec::completion_controller` slot per issued IO purely so teardown can wait those completions out — they write through raw `cached_chunk*` into cache generations the cache owns.
- **Evictor.** Each prefetch request is handed to the evictor thread when it is created, and its chunks become eviction candidates once its consumer is done; candidates are swept in the order their requests were created. Under `eviction: idle` the evictor reclaims every candidate that no live request still subscribes to. Under `lru` it runs only under pressure — total allocations in the pinned host memory resources (by any user) above `eviction_threshold_fraction` of their capacity, while the cache holds more than its `min_prefetching_budget_fraction` floor — or when something asked for memory back (`evict` / `evict_sync`), and if sparing subscribed chunks cannot free enough it takes those too. A chunk a reader has pinned is never reclaimed. Preparing a prefetch can wait for the evictor instead of failing on a momentarily empty pool; the readahead does.
- **Multi-GPU safe.** Device reads carry the caller's device id; the reactor sets that device before the H2D copy, and pinned chunks are portable across CUDA contexts.
- **Generations.** Cache entries are keyed by the io_object's cache id (path plus strong ETag for REST objects, path for local files) and shared-owned (`cache_generation`): the map holds one owner while a generation is current, and a `cache_handle`, an in-flight fill, a queued completion, or a cached-copy retirement holds another for as long as it may touch the generation's chunks. Admitting a newer generation of a path retires the older ones — they leave the map, keep serving their holders, and are reclaimed when the last owner releases. The evictor only borrows. `generation_count(path)` and `retired_generation_count()` expose the state. Datasources and handles must not outlive the `memory_reservation_manager`.
- **Pin saturation.** A chunk at its maximum reader-pin count is treated as a miss rather than read unpinned, so the read falls back to a direct backend read and the chunk can never be evicted under a reader.

### Constants

Header paths are relative to `cucascade/include/cucascade/`.

| Name | Location | Role |
|------|----------|------|
| `IO_BLOCK_SIZE` (4096) | `io/types.hpp` | `O_DIRECT` alignment for local-disk reads. |
| `dispatch_fanout` (2) | `io/templated_ioctx.hpp` (`next_reactor`) | A read is split across at most this many reactors: the two least-backlogged reactors of the read's class partition (rotation only breaks ties). |
| `min_dynamic_io_size` / `max_dynamic_io_size` (256 KiB / 16 MiB) | `io/uring/types.hpp` | Bounds of the physical operation size a uring reactor picks from its backlog and free staging slots. |
| chunk size | `buffer_pool::chunk_size()` (FSMR block size) | Cache / staging chunk granularity; sourced from the pinned `fixed_size_host_memory_resource`'s block size rather than a compile-time constant. |
| `eviction_threshold_fraction` / `min_prefetching_budget_fraction` | `io/cache/config.hpp` | When the pool starts evicting and the floor reserved for prefetching. |
| `use_odirect` | `io/uring/config.hpp` | Buffered-vs-`O_DIRECT` toggle, derived from `scan_manager.cache.mode`; operation size and `readv` fusion are selected dynamically from queue pressure, free slots, and the FSMR block size. |
| `merge_max_gap` / retry policy | `io/rest/config.hpp` | REST planner hint and retry tunables (see [S3 / Object-Store Backend](#s3--object-store-backend)). The reactor selects 4–16 MiB physical GETs dynamically from backlog and free connections. |

## Complete Scan Flow

```mermaid
graph TD
    CONV[pipeline converter] -->|make_ingestible| ING["gpu_ingestible<br/>(parquet / duckdb-native)"]
    ING -->|carried by| OP[sirius_gpu_scan_operator GPU_SCAN]

    SM[sirius_scan_manager.prepare_for_query] -->|cache miss| SP["split_provider<br/>(composes ingestible)"]
    SM -->|pinned-cache hit| DBP[cached_databatch_provider]

    SP -->|one task per file/range| DISP[dispatcher thread pool]
    DISP -->|metadata scan_info| SEQ[load_balancing_scan_batch_coalescer]
    DBP -->|resident batches| SEQ

    SEQ -->|coalesce + balance + prefetch| SC["split_connector<br/>(per operator)"]
    SC -->|get_next_split| OP

    OP -->|get_next_task_input_data| TC[task_creator]
    TC -->|preferred_device_id| GPT[gpu_pipeline_task]
    GPT -->|execute| EX["materialize_table -> post_filter_and_project<br/>-> data_batch"]
    EX -->|pipelineable_operator_data| NEXT[downstream pipeline]
```

The converter builds a `gpu_ingestible` and parks it on the `GPU_SCAN` operator. At query prep the scan manager either serves the operator from a pinned cache (`cached_databatch_provider`) or builds a `split_provider` over the ingestible. The provider dispatches one metadata task per file (parquet) or row-group range (duckdb-native); the per-query sequencer coalesces, balances, and prefetches each batch, then pushes splits onto the operator's connector. The task creator turns each pulled split into a `gpu_pipeline_task` on the split's preferred GPU; `execute()` materializes, optionally post-filters/projects, and emits a `data_batch` to the downstream pipeline.

## Key Files

| File | Purpose |
|------|---------|
| `src/op/scan/sirius_gpu_scan_operator.hpp` / `src/op/scan/sirius_gpu_scan_operator.cpp` | Unified `GPU_SCAN` source operator |
| `src/op/scan/sirius_gpu_scan_operator_data.hpp` | `scan_operator_input` (fresh-read or resident-batch split) |
| `src/op/scan/gpu_ingestible.hpp` / `src/op/scan/gpu_ingestible.cpp` | `gpu_ingestible` abstraction + `materialize_table` dispatch |
| `src/op/scan/gpu_ingestible_types.hpp` | `ingestible_table_info`, `scan_info`, `filtered_table` / `filter_state` |
| `src/op/scan/parquet_gpu_ingestible.hpp` / `src/op/scan/parquet_gpu_ingestible.cpp` | Parquet ingestible + `parquet_batch_coalescer` + `make_ingestible` |
| `src/op/scan/duckdb_native_gpu_ingestible.cpp` / `src/op/scan/duckdb_native_gpu_ingestible.hpp` | DuckDB-native ingestible + `duckdb_native_batch_coalescer` |
| `src/op/scan/batch_coalescer.hpp` | Coalescer interface |
| `src/op/scan/owning_table_view.hpp` | View-or-table handle with no-alloc reorder/drop/select |
| `src/op/scan/scan_plan.hpp` / `src/op/scan/scan_plan.cpp` | Index-space mapping (P/C/D), output layout, partition injection |
| `src/op/scan/parquet_schema_mapping.hpp` | Name-based DuckDB->parquet column resolution |
| `src/op/scan/row_group_metadata.hpp` | `row_group_slice` + `hybrid_scan_reader` |
| `src/op/scan/duckdb_native_metadata.hpp` / `duckdb_native_decoder.hpp` | DuckDB-native row-group walk + GPU decode |
| `src/scan_manager/sirius_scan_manager.hpp` / `.cpp` | Scan manager, `cache_entry_info`, `pinned_entry` |
| `src/scan_manager/split_provider.hpp` / `.cpp` | Concrete provider composing a `gpu_ingestible` |
| `src/scan_manager/split_connector.hpp` / `.cpp` | Blocking queue between sequencer and operator |
| `src/scan_manager/load_balancing_scan_batch_coalescer.hpp` / `.cpp` | Per-query sequencer: coalesce + balance + prefetch + push |
| `src/scan_manager/balancing_strategy.hpp` | Device-placement policy interface |
| `src/scan_manager/round_robin_strategy.hpp` / `.cpp` | Round-robin GPU placement |
| `src/scan_manager/config.hpp` | `scan_manager_config` |
| `src/scan_manager/pinned_chunk_stats.hpp` / `.cpp` | Pin-time per-chunk min/max capture, filter-safety check + prune probe |
| `src/op/scan/duckdb_mvcc_visibility.hpp` / `src/op/scan/duckdb_mvcc_visibility.cpp` | Snapshot visibility walk: delete keep-masks + pin-time DML guards |
| `src/scan_manager/mvcc_mask_job.hpp` / `src/scan_manager/mvcc_mask_job.cpp` | Per-query delete-mask jobs over pinned entries |
| `src/op/scan/duckdb_insert_delta.hpp` / `src/op/scan/duckdb_insert_delta.cpp` | Insert-delta capture + visibility count / staging-copy task bodies |
| `src/scan_manager/insert_delta_job.hpp` / `src/scan_manager/insert_delta_job.cpp` | Per-query insert-delta jobs: bundling, staging carve, per-op split cuts |
| `src/pin_table.hpp` / `src/pin_table.cpp` | `pin_table` / `unpin_table` + pin materialization |
| `src/op/scan/cached_ranges.hpp` / `src/op/scan/cached_ranges.cpp` | Sorted byte-range coalescing/lookup |
| `src/op/sirius_physical_gpu_values.hpp` / `src/op/sirius_physical_gpu_values.cpp` | `GPU_VALUES` source for `ColumnDataCollection`, empty-result, and dummy-scan inputs |
| `src/op/scan/scan_utils.cpp` | Row group pruning, filter expression conversion |
