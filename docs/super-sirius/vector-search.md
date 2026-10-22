# Vector Similarity Search (VSS)

**Files:** `src/vss/`, `src/cuda/vss/`, `src/vss/`, table functions in
`src/sirius_extension.cpp:1740-2571`

GPU-native k-nearest-neighbor search over the vector column of a GPU-pinned table, built on
cuVS/RAFT. Three Sirius-owned table functions expose it: `sirius_knn_search` runs the search
(exact brute-force — ENN — or approximate via a prebuilt index — ANN), and
`sirius_create_ann_index` / `sirius_drop_ann_index` manage the session's GPU-resident IVF-Flat
indexes. Unlike the transparent SQL interception path, VSS is an explicit surface: the search
runs directly in the table-function body, outside the Super Sirius pipeline executor.

```sql
CREATE TABLE items (id INTEGER, vec FLOAT[3]);
INSERT INTO items SELECT i, [i::float, (i+1)::float, (i+2)::float] FROM range(5000) t(i);
CHECKPOINT;  -- pin_table(format => 'duckdb') reads on-disk blocks; WAL-resident rows are invisible

-- The table must be pinned on the GPU tier first.
SELECT * FROM pin_table(name => 'items', tier => 'gpu', format => 'duckdb');

-- Optional: build an ANN index (IVF-Flat) on the vector column.
SELECT * FROM sirius_create_ann_index('items', 'vec', metric => 'l2', n_lists => 64);

-- Top-20 nearest neighbors of a query vector. use_index defaults to true (ANN);
-- pass use_index => false for the exact brute-force path.
SELECT * FROM sirius_knn_search('items', 'vec', [4980, 4981, 4982]::FLOAT[3], k => 20);
```

## The SQL Surface

All three are DuckDB table functions registered in `SiriusExtension`
(`src/sirius_extension.cpp:2532-2571`). The vector column must be a fixed-size `FLOAT[N]`
array column of a DuckDB-native catalog table; parquet pins do not carry fixed-width float
arrays, so only the DuckDB-native pin path is supported. The table argument accepts a
qualified name (`catalog.schema.table`); an unqualified schema defaults to the `schema_name`
parameter (default `'main'`).

### `sirius_knn_search(table, column, query, ...)`

Bind at `src/sirius_extension.cpp:2203`, execution via `SiriusVectorSearchInit` →
`sirius::vss::run_vector_search` (`src/vss/vector_search.cpp:82`).

| Argument | Kind | Type | Default | Meaning |
|---|---|---|---|---|
| `table` | positional 1 | VARCHAR | — | GPU-pinned base table |
| `column` | positional 2 | VARCHAR | — | `FLOAT[N]` vector column (must be pinned) |
| `query` | positional 3 | ANY | — | one query vector: a `FLOAT[N]` array (`[..]::FLOAT[3]`) or a numeric list; length must equal N; NULL elements rejected |
| `k` | named | BIGINT | `10` | neighbors to return (≥ 1); clamped to the table's row count |
| `output_columns` | named | VARCHAR[] | all pinned columns, in catalog order | base-table columns to return; each must exist and be pinned; an explicitly empty list is an error |
| `metric` | named | VARCHAR | `'l2'` | `'l2'` or `'cosine'` |
| `use_index` | named | BOOLEAN | `true` | `true` → ANN over a cached index; `false` → exact ENN |
| `n_probes` | named | BIGINT | `0` | ANN only; IVF lists to probe. `0` → `min(n_lists, 32)`; values above `n_lists` are clamped |
| `index_type` | named | VARCHAR | `'ivf_flat'` | only `'ivf_flat'` |
| `schema_name` | named | VARCHAR | `'main'` | schema of an unqualified table name |

**Return shape:** one row per neighbor, nearest first — the `output_columns` in order (catalog
types), plus a trailing FLOAT `distance` column. Up to `k` rows; fewer when the table holds
fewer rows, when the ENN path compacted away null vector rows, or when the probed ANN lists
held fewer than `k` vectors.

Guards worth knowing (`src/sirius_extension.cpp:2253-2274`): `k >= 1`, `n_probes >= 0`, and
cosine ANN search is capped at `k <= 256` (cuVS returns wrong results for cosine over IVF-Flat
beyond that; run `use_index => false` or lower `k`). Pin changes after a prepared statement was
bound are re-checked at execution (`src/vss/vector_search.cpp:108-121`), with an error telling
the user to re-pin or re-run.

### `sirius_create_ann_index(table, column, ...)`

Bind at `src/sirius_extension.cpp:1760`, build at `:1824`. Returns a single BOOLEAN column
named `Success`.

| Argument | Kind | Type | Default | Meaning |
|---|---|---|---|---|
| `table`, `column` | positional | VARCHAR | — | GPU-pinned table and its `FLOAT[N]` column |
| `metric` | named | VARCHAR | `'l2'` | `'l2'` or `'cosine'`; one index per (column, metric) |
| `index_type` | named | VARCHAR | `'ivf_flat'` | only `'ivf_flat'` |
| `n_lists` | named | BIGINT | `0` | IVF inverted-list count; `0` → `min(floor(sqrt(n_rows)), 1024)` (`default_ivf_n_lists`, `src/sirius_extension.cpp:1817`), then capped at `n_rows` |
| `schema_name` | named | VARCHAR | `'main'` | schema of an unqualified table name |

Rebuilding an existing (column, metric) index replaces it. Each execution of a prepared
`sirius_create_ann_index` statement rebuilds.

### `sirius_drop_ann_index(table, column, ...)`

Bind at `src/sirius_extension.cpp:2090`, execution at `:2129`. Takes the same positional pair
plus named `metric` and `schema_name`. With a `metric`, only that index is dropped; without
one, every index on the column is dropped. Returns a BOOLEAN column named `Dropped` — `true`
iff at least one entry was removed.

## Shared Search Entry Point

`run_vector_search` (`src/vss/vector_search.cpp:82`) does the work common to both paths:

1. Picks the first GPU memory space and sets the device. Vector search is **single-GPU**: a
   pin whose chunks span devices is rejected (`src/vss/pinned_column.cpp:44-49`).
2. Requires the table pinned on the GPU tier, and re-validates that the vector and output
   columns are still pinned (pin/unpin does not rebind a prepared query).
3. Clamps `k` to `min(num_rows, k)`; an empty table or `k == 0` returns the empty result
   schema immediately.
4. Uploads the query vector once to the device; both impls read it there.
5. Builds a `vector_search_context` (`src/vss/vector_search_internal.hpp:51`) — the
   resolved spaces, pin, allocator, stream, device query pointer — and dispatches on
   `use_index` to `run_vector_search_ann` or `run_vector_search_enn`.

Both paths produce a GPU `cudf::table`, which `vss_result_to_host`
(`src/vss/vector_search.cpp:72`) converts to a `host_data_representation`; the table function
streams it out through a `host_table_chunk_reader` (`src/sirius_extension.cpp:2387-2400`).
Transient allocations go through the pinned table's GPU memory-space allocator, so they are
tracked by the same memory manager as everything else and freed when the query ends.

Pinned chunks may hold compressed-materialization narrow carriers; gathered output columns are
widened back to their native catalog types before leaving the search
(`restore_native_carriers`, `src/vss/vector_search.cpp:54` — see
[Compressed Materialization](compressed-materialization.md)).

## ENN: Exact Brute-Force Path

`run_vector_search_enn` (`src/vss/vector_search_enn.cpp:37`) computes exact top-k per pinned
chunk, then merges — the chunks are never concatenated, which both avoids a full extra copy of
the dataset and sidesteps cudf's 2^31-element per-column limit in the LIST child.

**Per-chunk search** (`compute_enn_top_k`, `src/vss/enn_top_k.cpp:59`): the chunk's columns are
assembled as `[vector, out0, out1, ...]`. If the vector column is sliced or has null rows, the
whole chunk is first compacted with a validity mask so the passthrough columns stay row-aligned
— this is why a search over data with null vectors can return fewer than `k` rows.
Element-level nulls are rejected. The compacted LIST column is then reinterpreted zero-copy as
a row-major `[n_rows, dim]` RAFT matrix view (`list_column_as_dataset_view`,
`src/cuda/vss/cudf_raft_interop.cu:27` — a non-owning borrow of the values child buffer, which
is exactly the flat `n_rows*dim` FLOAT32 blob for a gap-free fixed-size list). `brute_force_knn`
(`src/cuda/vss/brute_force_search.cu:32`) builds a cuVS brute-force index over the view (a
reference plus precomputed norms, not a copy) and searches it, returning flattened INT64
neighbor indices and FLOAT32 distances, nearest first. The neighbors gather the passthrough
columns; the chunk's candidate table is `[out0, ..., distance]`.

One `raft::device_resources` handle, bound to the search stream, is reused across all chunks so
its workspace setup is paid once; the per-chunk searches run async and are synchronized once
before the host copy.

**Merge** (`merge_enn_top_k`, `src/vss/enn_top_k.cpp:111`): each candidate is already sorted by
distance (cuVS `select_k` output), so a `cudf::merge` k-way merge on the trailing distance
column yields one globally sorted table without concatenating, and the nearest `k` rows are
sliced off.

## IVF-Flat Index Creation and the Index Cache

### Build (`SiriusCreateAnnIndexFunction`, `src/sirius_extension.cpp:1824`)

The vector column's pinned chunks are collected as zero-copy views in pin order
(`pinned_column_chunk_views`, `src/vss/pinned_column.cpp:32`), and the index is built
incrementally, one batch at a time (`build_ivf_flat_index_from_batches`,
`src/cuda/vss/ivf_flat_index.cu:68`):

- k-means centroids are trained on the first batch with at least `n_lists` rows (cuVS
  subsamples it internally); if no batch is large enough the build is rejected *before* any
  existing index is touched (`src/sirius_extension.cpp:1920-1932`).
- The dataset is then added batch by batch via `ivf_flat::extend`, tagging each vector with its
  global row id (offset by rows already added) so search returns indices into the whole dataset
  in pin order.
- `conservative_memory_allocation` is set so each inverted list is sized to what it holds; the
  default padded-growth policy would re-grow every list per batch and inflate the build-time
  peak well past the final index size.
- The tradeoff of incremental building is recall: centroids trained on one batch degrade recall
  at low `n_probes`.

**Memory admission:** the build's GPU footprint is estimated up front
(`ivf_flat_reservation_bytes`, `src/vss/ivf_flat_index.hpp:38` — roughly twice the
vector+id payload plus per-list rounding and centroids) and reserved through the cache
(`reserve_index_memory`, `src/vss/cuvs_index_cache.cpp:57` — non-blocking, pinned to the GPU
that holds the table). The reservation is attached to a dedicated build stream so every cuVS
allocation during the build is charged and bounded by it, and released when the build finishes
(success or failure); the finished index remains as ordinary capacity-accounted GPU memory. If
the reservation cannot be satisfied, the existing index for that (column, metric) is dropped
first and the reservation retried (the *destructive* path); if it still fails, the error
reports the needed vs. free MiB and lists other-metric indexes on the column that could be
dropped. On the non-destructive path the old index is only erased after the new one is built.

### The cache (`cuvs_index_cache`, `src/vss/cuvs_index_cache.hpp:166`)

A session-lifetime, mutex-guarded registry of GPU-resident cuVS indexes — the ANN analogue of
the scan manager's pin-table cache, deliberately standalone so future index kinds can add their
own search operators without touching shared scan code.

- **Key:** the length-prefixed routing identity `(catalog, schema, table, column, metric)`
  (`build_ann_index_cache_key`, `src/vss/cuvs_index_cache.hpp:71`). One index per
  (column, metric); inserting under an existing key replaces it.
- **Entries** (`pinned_index_entry`): copyable metadata (kind, identity, `dim`, `num_rows`,
  `n_lists`, metric, `resident_bytes`), the type-erased index payload (`any_cuvs_index` /
  `cuvs_index_holder<Index>`, recovered with `index_as<Index>()`), and the build stream — the
  index's device buffers are bound to that stream and freed on it when the entry is destroyed
  (the stream member is declared before the index so it is destroyed after it).
- **Lookups return `shared_ptr`:** a caller mid-search keeps the whole entry (and its GPU
  memory) alive even if the entry is erased or replaced concurrently; the mutex only guards the
  map.
- **Ownership:** the cache lives on `SiriusContext` (`src/sirius_context.hpp:687`),
  constructed with the memory reservation manager during initialization
  (`src/sirius_context.cpp:701`) and reset in `terminate()` after a device sync, before the
  memory manager is torn down (`src/sirius_context.cpp:974`); the member is also declared after
  `memory_manager_` so destruction order matches.
- **Lifetime and invalidation are user-managed:** an index is a point-in-time snapshot of the
  pinned table at build time. Nothing tracks base-table changes — after any INSERT/DELETE or an
  unpin/re-pin the user must rebuild with `sirius_create_ann_index` (which replaces the entry).
  Entries live until `sirius_drop_ann_index`, replacement, or session teardown; there is no
  eviction or spilling.

## ANN Search Path

`run_vector_search_ann` (`src/vss/vector_search_ann.cpp:106`):

1. **Auto-routing:** the index is found by identity — `find_by_column(catalog, schema, table,
   column, metric)` (`src/vss/cuvs_index_cache.cpp:90`). No index for that column under that
   metric is an error that points at `sirius_create_ann_index` and `use_index => false`.
2. **Probe count:** an explicit `n_probes` is clamped to `n_lists`; `0` means
   `min(n_lists, 32)` (`src/vss/vector_search_ann.cpp:121-125`).
3. **Search:** `search_ivf_flat_index` (`src/cuda/vss/ivf_flat_index.cu:160`) down-casts the
   type-erased index, searches the device-resident query, and returns `k` INT64 global row ids
   plus FLOAT32 distances (synchronous).
4. **Padding removal:** when the probed lists hold fewer than `k` vectors, IVF-Flat pads the
   result with sentinel-distance slots; rows with `distance >= FLT_MAX` are filtered out
   (`src/vss/vector_search_ann.cpp:130-147`), which is why ANN can return fewer than `k` rows.
5. **Chunked gather:** cuVS global row ids are mapped back onto the pinned chunks without
   concatenating them (`gather_pinned_by_global_index`, `src/vss/vector_search_ann.cpp:54`):
   chunk row-count prefix sums give each chunk's base, each chunk is gathered with
   out-of-range-→-null (`NULLIFY`) local indices, and the per-chunk partial results are
   null-coalesced — every output position is filled by exactly the one chunk that owns its
   index. This relies on the invariant that the index was built over the same chunks in the
   same pin order (another reason a re-pin requires a rebuild).
6. **Sort:** ANN output order is not guaranteed, so the result is sorted ascending by the
   trailing distance column before conversion to host.

## Distance Metrics

Two user-facing metrics, `'l2'` and `'cosine'` (`src/vss/distance_metric.cpp`):

| metric | ENN (`enn_distance_type_from_metric`) | ANN (`ann_distance_type_from_metric`) |
|---|---|---|
| `l2` | `L2SqrtUnexpanded` | `L2SqrtExpanded` |
| `cosine` | `CosineExpanded` | `CosineExpanded` |

Both L2 forms report true (square-rooted) Euclidean distance. The brute-force path uses the
*unexpanded* form to avoid catastrophic cancellation on large-magnitude vectors; an IVF-Flat
index is built *expanded* for correct coarse cluster selection, while its fine scan is
unexpanded either way — the reported distances are identical. The cache's metric matching
folds the two L2 variants into one canonical value (`canonical_metric`,
`src/vss/cuvs_index_cache.cpp:37`) so routing matches on distance semantics rather than the
exact cuVS enum.

## Limits

- Single query vector per call; single GPU (a pin spanning multiple GPUs is rejected).
- DuckDB-native tables only; the column must be `FLOAT[N]`.
- Cosine ANN search: `k <= 256`.
- ENN compacts null vector rows (result may have fewer than `k` rows); the ANN index build
  rejects null or sliced vector batches outright — build on clean data or use
  `use_index => false`. Element-level nulls are rejected on both paths.
- Only `ivf_flat` indexes; no incremental index updates, eviction, or spilling.

## Key Code Paths

| What | Where |
|---|---|
| Table-function registration | `src/sirius_extension.cpp:2532-2571` |
| `sirius_knn_search` bind / init / scan | `src/sirius_extension.cpp:2203`, `:2372`, `:2394` |
| `sirius_create_ann_index` bind / build | `src/sirius_extension.cpp:1760`, `:1824` |
| `sirius_drop_ann_index` bind / drop | `src/sirius_extension.cpp:2090`, `:2129` |
| Shared search entry, result-to-host | `src/vss/vector_search.cpp:82`, `:72` |
| ENN per-chunk loop | `src/vss/vector_search_enn.cpp:37` |
| ENN per-chunk top-k, k-way merge | `src/vss/enn_top_k.cpp:59`, `:111` |
| cuVS brute-force search | `src/cuda/vss/brute_force_search.cu:32` |
| LIST-column → RAFT matrix view | `src/cuda/vss/cudf_raft_interop.cu:27` |
| IVF-Flat batched build, search kernel | `src/cuda/vss/ivf_flat_index.cu:68`, `:160` |
| Build footprint estimate | `src/vss/ivf_flat_index.hpp:38` |
| Index cache (key, entries, API) | `src/vss/cuvs_index_cache.hpp:71`, `:123`, `:166` |
| Metric canonicalization, identity lookup | `src/vss/cuvs_index_cache.cpp:37`, `:90` |
| ANN search impl, global-id gather | `src/vss/vector_search_ann.cpp:106`, `:54` |
| Pinned chunk views (single-GPU check) | `src/vss/pinned_column.cpp:32` |
| Metric mapping | `src/vss/distance_metric.cpp:24-47` |
| Cache ownership on `SiriusContext` | `src/sirius_context.hpp:687`, `src/sirius_context.cpp:701`, `:974` |
| End-to-end SQL tests | `test/cpp/integration/test_gpu_execution_vector_search.cpp` |
| Unit tests | `test/cpp/vss/` |
