# R4 flat evolution with position deletes

Regenerate both R4 fixtures from the repository root:

```bash
pixi run python test/cpp/integration/data/generate_r4_evolution_fixture.py
```

This fixture upgrades the existing V2 position-delete table to V3 metadata. Its
physical columns are `fruit#1`, `count#2` (INT32), and `old_score#4` (FLOAT32).
The snapshot's logical schema is `score#4` (DOUBLE), `count#2` (BIGINT),
`fruit#3` (STRING), and `added#5` (BIGINT). Thus the old `fruit#1` is retired,
and the reused name has no matching physical field. `added#5` has
`initial-default = 7` and `write-default = 9`.

Original file positions 1 and 3 are deleted. The remaining rows are:

| score | count | fruit | added |
| ----- | ----- | ----- | ----- |
| 1.5   | 1     | NULL  | 7     |
| NULL  | 3     | NULL  | 7     |
| 5.5   | 5     | NULL  | 7     |

The SQL tests require GPU execution and independently compare DuckDB's answer.
They also cover predicates on promoted/defaulted fields, all-absent projections,
NULL predicates and count-only outputs.

The companion `iceberg_r4_partition_missing` fixture omits `count#2` physically,
but supplies 42 from identity-partition metadata. It must retain runtime fallback
until the adapter exports per-file partition constants: replacing that field with
its NULL default would silently change the result.
