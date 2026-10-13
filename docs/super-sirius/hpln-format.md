# The `.hpln` file format

`.hpln` is Sirius's own columnar file format. A file holds a table already in the representation a
pinned entry uses, so loading one is a byte copy rather than a decode: `pin_table` over `.hpln`
stages the file's bytes straight into host memory, where the same pin over parquet must decode the
file and re-compress it. Its other property is metadata locality — one tail read tells a reader
where everything lives, which is what makes range-skipped fetch possible over a network.

It is a GPU-only format. DuckDB has no CPU reader for it, so a `read_simpatico` query that falls
back to the CPU errors rather than returning rows.

## SQL surface

```sql
-- Write a table (or any query) as .hpln
COPY (SELECT * FROM lineitem) TO '/data/lineitem.hpln' (FORMAT 'simpatico');

-- Read it; queries are GPU-only
SELECT count(*) FROM read_simpatico('/data/lineitem.hpln') WHERE l_shipdate >= DATE '1995-01-01';

-- Pin it. Host tier only: the file's chunks are already the host representation
CALL pin_table('/data/lineitem.hpln', format => 'simpatico', tier => 'host', name => 'lineitem',
               cols => ['l_orderkey', 'l_shipdate']);
```

`read_simpatico` takes a single path, local or `s3://`.

### `COPY ... (FORMAT 'simpatico')` options

| Option | Default | Meaning |
|---|---|---|
| `chunk_rows` | writer default | Rows per chunk. A chunk is the unit of independent compression, of the chunk directory, and of what a reader may skip. |
| `group_rows` | `pinned_zone_map_group_rows` | Rows per zone-map group inside a chunk. `0` writes chunk-level statistics only. |
| `cluster_by` | none | Sort each chunk on these columns as it is written, so its zone-map groups describe narrow ranges. The analogue of `pin_table`'s `cluster_by`, done once at write time. |
| `plan` | `identity` | Compression plan DSL applied to every column. |
| `plan_table` | none | Name of a table whose registered per-column plans to use. |

`cluster_by` belongs here rather than at pin time: a `.hpln` pin never decodes, which is exactly why
it is fast, and sorting would mean decode → sort → recompress. `pin_table` rejects `cluster_by` for
this format for that reason.

## Physical layout

```
[chunk 0 header][chunk 1 header]...[chunk 0 payload][chunk 1 payload]...[segments][postscript][trailer]
```

A fixed 16-byte trailer points at a postscript, which is a locator table for segments. Read the last
few KB and you know where every chunk and every piece of metadata lives — one round trip, no
speculative prefix read and no re-read. (An earlier layout was `[header][payload]` with nothing
saying where the header ended, so a reader could not locate the header without already holding it.)

Chunk headers are written **contiguously ahead of every payload**, so a bind fetches all of a file's
structural metadata in one sequential read rather than one seek per chunk.

### Segments

| Kind | Contents |
|---|---|
| `header` | The structural header per chunk: column names, decoded types, row counts, validity kind, the serialized plan tree, and each buffer's size and payload offset. |
| `payload` | Every leaf buffer, concatenated, at the offsets the header declares. |
| `zone_maps` | Per-column, per-group min/max bounds, in the packed form `group_bounds_arena` uses, so reading them is a copy rather than a rebuild. |
| `logical_types` | The engine's logical schema. The header carries cuDF physical types, which cannot express DECIMAL precision, nullability or a timestamp's time zone; a pin gets those from memory, a file has nowhere else to get them. |
| `chunk_directory` | Where each chunk's header and payload live, plus its row count, so a reader can report split sizes without parsing headers. |
| `checksums` | CRC32C over the other segments — per chunk for `header` and `payload`, once for each metadata segment. |

Segment kinds are **additive**: a reader skips kinds it does not know, so new metadata can be added
without a format break, and a file written before a kind existed simply carries none. A file with no
chunk directory is treated as a single chunk spanning the `header` and `payload` segments, which is
what every file written before that segment existed is.

Payload buffers are padded to 4096 bytes, the O_DIRECT block size, so a read's offset and length stay
aligned and the unbuffered path is usable. Recorded sizes are *not* padded — a buffer still declares
the bytes it holds, and the slack after it is never read.

## Reading

Reads go through `hpln_source`, which is either the local filesystem or a Sirius `io_context` (uring
for local paths, REST for `s3://`). Extents are turned into requests by `plan_hpln_reads`, which
sorts by offset, merges runs while the gap is worth bridging rather than paying for another request,
and cuts long runs to a target request size. The bridged bytes are read into scratch and discarded —
the trade is that a request costs more than the bytes it moves over an object store.

Because the zone maps are their own segment, a scan reads them at **bind** time and decides a chunk's
fate before emitting a split for it, so a pruned chunk is never fetched.

Payload chunks are verified against the file's CRC32C after staging and before the blob is handed to
anyone. A narrowed read — a column or row subset — cannot be verified, because the recorded CRC
covers a chunk's whole payload.

## Trade-offs

A `.hpln` pin chooses neither its row order nor its compression plans: both are properties of the
file. That is the price of the pin being pure I/O, and it cuts both ways — the file's plans are
picked for size, while a pin's are picked for decode speed, so an `.hpln` pin holds a smaller but
slower-decoding representation than the same columns pinned from parquet.

High-cardinality text is the format's weak spot. Columns like TPC-H's `o_comment` have no non-LZ plan
that beats storing them raw, so they dominate a file's size on disk and a query that scans one reads
more bytes than parquet would.

**Files:** `src/compression/hpln_io.{hpp,cpp}` (transport and read planning),
`src/compression/simpatico_file_ingest.{hpp,cpp}` (bind, staging, verification),
`src/compression/simpatico_copy_function.{hpp,cpp}` (the writer),
`src/op/scan/simpatico_gpu_ingestible.{hpp,cpp}` (the scan source),
`src/compression/simpatico_codegen/include/api/compressed_table_io.hpp` (the on-disk layout).
