# Iceberg conformance corpus

Tables written by **pyiceberg** (`apache/iceberg-python`, the ASF reference
implementation), with expected results recorded from **pyiceberg's own scan** — never
from Sirius and never from DuckDB.

That independence is the point. The hand-built iceberg fixtures elsewhere under
`test/cpp/integration/data/` were green at 27/27 cases and 580 assertions while the GPU
scan path was returning **silently wrong rows** for a dropped-and-re-added column. They
could not catch it, because a fixture built by hand encodes the same assumption the
implementation makes — here, that Iceberg columns resolve by name. They resolve by
**field ID**.

## Regenerating

Run from the **repo root**. A relative output path is what makes pyiceberg record
relative paths in both the metadata JSON and inside the binary `.avro` manifests, so the
corpus resolves against the repo root like every other fixture here:

```bash
pip install pyiceberg pyarrow sqlalchemy
python3 test/cpp/integration/data/iceberg_conformance/gen_corpus.py \
        test/cpp/integration/data/iceberg_conformance
```

Regenerating rewrites `expectations.json`. Those expectations come from pyiceberg, so a
diff in them means the reference implementation changed its mind — investigate it, do
not paper over it by re-recording from our own engine.

## Running

```bash
python3 test/cpp/integration/data/iceberg_conformance/run_conformance.py \
        test/cpp/integration/data/iceberg_conformance \
        --duckdb build/release/duckdb
```

Each case runs in its **own process behind a timeout**, and each issues a *second* query
on the same connection. Both matter: a runtime GPU fallback poisons its connection, so
the next GPU query on it hangs forever. Process isolation means one hung case cannot
stall or mask the others — which is also why the deadlocking cases below cannot yet live
in the Catch2 suite, where a hang has no timeout and would stall the whole run.

## Cases

| case | field IDs | what it catches | status |
|---|---|---|---|
| `append_only` | id=1, name=2 | baseline — no evolution, no deletes. If this ever declines, the gate is over-refusing | runs on GPU, matches |
| `drop_readd` | id=1, x=2, **y=4** | `y` dropped and re-added under the same name, so it is a NEW field id. The old data file holds the ORIGINAL `y` at field id 3, so the new column must read NULL. Resolving by name returns the old values | **returns wrong rows** |
| `rename_col` | id=1, value=2 | renamed column keeps its field id; the old data file carries the ORIGINAL name. Name resolution fails loudly, takes the runtime fallback | **deadlocks** |
| `add_column` | id=1, a=2, **b=3** | column added after the first data file was written; absent from that file, must read NULL. This is the case a `max(field_id) > column_count` pre-filter MISSES | **deadlocks** |

`file_uri` — a table whose manifests carry absolute `file://` URIs, the shape Java and
Spark writers emit — is deliberately **not** in this corpus: it cannot be expressed with
relative paths. Generate it with an absolute output path (`gen_corpus.py /var/tmp/...`),
which adds the case. The underlying normalization rule is covered in-tree by the
`strip_file_scheme` unit tests.

## Limitation: this corpus cannot cover delete files

**pyiceberg 0.11.1 cannot write delete files.** Asking it to delete rows — even on a
format-version-2 table with `write.delete.mode=merge-on-read` — emits:

```
UserWarning: Merge on read is not yet supported, falling back to copy-on-write
```

and it rewrites the data file instead. So every merge-on-read construct — V2 positional
deletes, V2 equality deletes, V3 deletion vectors — is **out of reach of this generator**.

What that means in practice:

- This corpus protects **schema evolution, field-ID resolution, and path handling**. That is
  where the three silent failures were, so it is worth having.
- The delete fixtures elsewhere under `test/cpp/integration/data/` must stay **hand-built**,
  and they therefore keep the weakness this corpus exists to remove: they were built by the
  same people who wrote the implementation, so they encode its assumptions. Treat a green
  delete test as weaker evidence than a green conformance test.
- One known bug currently has **no faithful regression test** for this reason: `iceberg_scan`
  honours three snapshot selectors (`snapshot_from_id`, `snapshot_from_timestamp`, `version`)
  but the delete path resolves only the first, so a time-travel query reads one snapshot's
  data and another snapshot's deletes. Reproducing it requires the current snapshot to carry
  delete files. The GPU path declines those queries instead (see `sirius_plan_get.cpp`), which
  is conservative and safe, but the decline rests on reasoning rather than a red-then-green
  test.

Closing that gap needs reference-written delete files from Spark + `iceberg-runtime` or the
Java Iceberg API — a JVM dependency, not a `pip install`. Worth doing; not free.

## Adding a case

Add it to `gen_corpus.py` so the table and its expectation are produced together. Do not
hand-build a table here and do not hand-write an expectation — a corpus that is not
generated by the reference implementation is just another fixture that agrees with us.
