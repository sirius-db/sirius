# Probe-Accumulate Join

> **Experimental.** Off by default (`enable_probe_accumulate`). The operator is complete and
> tested, but nothing produces its input shape automatically yet, so today it accelerates queries
> written in the aggregate form described in [What triggers it](#what-triggers-it). TPC-H q21 as
> published is not one of them; see [Status and limits](#status-and-limits).

A probe-accumulate join computes

```sql
SELECT key, MIN(a), MAX(b), COUNT(*), COUNT(c) ...
FROM probe JOIN build ON probe.key = build.key      -- build side unique on key
GROUP BY key
```

without ever materializing the join output and without a group-by hash table. It builds one
accumulator record per build row, streams the probe side through the join's existing hash table,
folds each matching probe row into its build row's record with atomics, and emits one row per build
row that matched at least once.

## Why

The ordinary plan for that query has three expensive steps after the probe scan: the join probe
emits (probe row, build row) pairs, a gather materializes the joined rows, and a hash group-by
creates a group per distinct key and aggregates into it. When the build side is a large set of
distinct keys (TPC-H q21's candidate orders: 144M keys at SF1000), the group-by alone was 113 ms
of kernel time per run and the tail after the scan (join, group-by, merge) was about 200 ms.

The build side already *is* the set of groups, so nothing needs to be created: each build row
owns a slot, and the probe side only updates slots that exist.

## What triggers it

`try_plan_probe_accumulate` (`src/planner/sirius_plan_probe_accumulate.cpp`), called from the
aggregate planner right after the dense-count-join check, matches a `LogicalAggregate` whose:

- group key is a single column that is the join key (from either side; equal under INNER);
- input is one INNER hash-joinable equality `probe.k = build.k` on `INTEGER` or `BIGINT`, with no
  other condition, optionally through one projection;
- build side is **provably unique** on its key (`prove_unique_columns`: a primary key, or a
  `GROUP BY` on that key);
- aggregates are all `MIN`, `MAX`, `COUNT(col)` or `COUNT(*)`, with no `DISTINCT`, `FILTER` or
  `ORDER BY`, at most `probe_accumulate_max_specs` (8);
- each aggregate argument is any expression over **probe-side columns only** that the GPU
  expression evaluator accepts, and a `MIN`/`MAX` argument is a 32- or 64-bit integer carrier
  (`INTEGER`, `BIGINT`, `DATE`, `DECIMAL` up to 18 digits, timestamps). Floating point, narrow
  integers and strings have no device atomic.

It also declines when the build estimate cannot fit: the build keys must be under
`max_build_hash_table_bytes` (the join only runs single-partition `BUILD_PROBE` below it) and the
accumulator records must fit within 25% of the smallest GPU. When it declines after the shape
matched, an INFO line `not fusing: <reason>` says why.

The shape the planner recognises is what an `EXISTS` over a correlated `<>` becomes when written as
a pre-aggregation, for example TPC-H q21 rewritten so that the existence tests share one scan:

```sql
WITH cand AS MATERIALIZED (            -- lineitem rows of Brazilian suppliers, late
  SELECT l1.l_orderkey, l1.l_suppkey, s.s_name
  FROM lineitem l1, supplier s, nation n
  WHERE s.s_suppkey = l1.l_suppkey AND s.s_nationkey = n.n_nationkey
    AND n.n_name = 'BRAZIL' AND l1.l_receiptdate > l1.l_commitdate),
ck AS (SELECT l_orderkey FROM cand GROUP BY l_orderkey),   -- unique build side
agg AS (
  SELECT l.l_orderkey,
         min(l.l_suppkey) mn, max(l.l_suppkey) mx,
         min(CASE WHEN l.l_receiptdate > l.l_commitdate THEN l.l_suppkey END) lmn,
         max(CASE WHEN l.l_receiptdate > l.l_commitdate THEN l.l_suppkey END) lmx
  FROM lineitem l, ck WHERE l.l_orderkey = ck.l_orderkey
  GROUP BY l.l_orderkey)                                   -- fused here
SELECT c.s_name, count(*) AS numwait
FROM cand c, agg a, orders o
WHERE c.l_orderkey = a.l_orderkey AND o.o_orderkey = c.l_orderkey AND o.o_orderstatus = 'F'
  AND (a.mn <> c.l_suppkey OR a.mx <> c.l_suppkey)         -- EXISTS another supplier
  AND a.lmn = c.l_suppkey AND a.lmx = c.l_suppkey          -- NOT EXISTS another late supplier
GROUP BY c.s_name ORDER BY numwait DESC, c.s_name LIMIT 100;
```

`EXISTS (... l2.k = l1.k AND l2.x <> l1.x)` is equivalent to
`min(l2.x) <> l1.x OR max(l2.x) <> l1.x` over the rows with that key, which is why per-key
`MIN`/`MAX` is enough; the late-only variant puts the predicate in a `CASE` so one scan serves both.
Use `GROUP BY`, not `SELECT DISTINCT`, for the key set: the planner does not support `DISTINCT` and
the whole query would fall back to the CPU.

## Plan shape

```
EMIT (PROBE_ACCUMULATE_EMIT)          source + sink, own pipeline, runs once
  └─ PASSTHROUGH_SINK                 ends the join's pipeline
       └─ HASH_JOIN (accumulate mode) INNER, BUILD_PROBE, single partition
            ├─ probe child: PROJECTION [key, argument...]  over the original probe subtree
            └─ build child: unique key set (CONCAT → PARTITION wrapped as usual)
```

The planner rewrites the logical join before planning it: it inserts a projection under the probe
side that carries exactly `[join key, aggregate arguments...]`, rebases the join condition onto
that key, and slims the join's outputs to one key column per side (it only ever emits empty
batches). It then plans the join normally with `plan_comparison_join`, which means **dynamic
filters, sizing and the build pipeline are unchanged**: the build keys still publish their Bloom or
IN-list filter to the probe-side scan, which is what keeps the probe volume at the candidate rows
rather than every row of the table.

Two details matter and were both bugs during development:

- **`PASSTHROUGH_SINK` between the join and the emit operator.** A sink operator runs as the last
  operator of its pipeline, once per task. Without a sink boundary the emit operator executed
  inside each probe task, emitting whatever the other tasks had accumulated so far.
- **Native carriers.** The compressed-schema and narrowing passes treat an accumulate join as a
  boundary (they restore native carriers on its inputs), because the kernels read probe values as
  their planned type. `accumulate()` additionally refuses a probe column of any other type.

## Runtime

`sirius_physical_hash_join::accumulate` (a `probe_accumulate_context`) switches the join's
`BUILD_PROBE` path:

1. **Build.** The first probe task builds the build table as usual (`cudf::distinct_hash_join`,
   which is why uniqueness is required) and calls `probe_accumulate_state::initialize`, allocating
   one record per build row. The context also keeps a read lock on the build batch so `emit` can
   read the key column after the join's own state is finalized.
2. **Probe.** Each probe batch runs `distinct_hash_join::inner_join`, which yields
   (probe row, build row) pairs. Instead of gathering them, the join calls
   `accumulate(probe, probe_rows, build_rows)` and returns an empty batch. The stream is
   synchronized so the emit task, which runs on another stream, sees finished atomics.
3. **Emit.** When the join's pipeline has finished, `sirius_physical_probe_accumulate_emit` runs
   once: it compacts the records with at least one match, gathers the build key, and writes each
   aggregate into its output column. The result is cached in the context so a rescheduled task
   copies it rather than calling `emit` again on state that was already consumed.

The join is required to run as a single-partition `BUILD_PROBE` join. If sizing picks anything
else, `get_partition_strategy` throws and the query falls back to the CPU; the planner-side
capacity check exists to avoid ever getting there. Multi-GPU is not supported (the planner declines
with more than one GPU).

## Accumulator state

`probe_accumulate_state` (`src/cuda/probe_accumulate_impl.cu`) lays one record out per build row,
widest field first so every field is naturally aligned:

```
[ hits u64 ][ COUNT u64 ... ][ 8-byte MIN/MAX ... ][ 4-byte MIN/MAX ... ][ seen flags u8 ... ] pad
```

The record is padded to a power of two up to 64 bytes so it never straddles a sector. A match
touches one record: one `atomic_bump` on `hits`, then per aggregate an atomic fold (cuda::atomic_ref
`fetch_min`/`fetch_max`) or a count increment. `MIN`/`MAX` keep a one-byte `seen` flag instead of a
count, so an all-NULL group emits NULL without a second atomic. All aggregates are folded by a
single kernel whose per-aggregate descriptors travel in one by-value argument; that bounds the
count at 8.

For q21 the record is 32 bytes: 144M build rows, 4.6 GB.

## Results

TPC-H SF1000 on one GB300, `bench/sf1000-repro` settings, best of three iterations:

| Variant | q21 |
|---|---|
| Original SQL (DuckDB's decorrelated plan) | 0.603 s |
| Aggregate form above, ordinary join + group-by | 0.542 s |
| Aggregate form above, probe-accumulate | **0.425 s** |

Every variant returns byte-identical output. Per run, the 113 ms hash group-by kernel is replaced by
31 ms of accumulation plus about 13 ms for initializing and emitting the records, total kernel time
falls from 510 ms to 399 ms, and the tail after the probe scan (join, group-by, merge) drops from
about 200 ms to about 115 ms. With the setting on, the 22-query suite is unchanged (6.855 s against 6.815 s,
within run-to-run noise; no query other than the hand-written form is fused), and every result
file matches the baseline apart from tie order in q11.

## Status and limits

- **Original q21 does not benefit.** DuckDB decorrelates its two subqueries into two nested
  `DELIM_JOIN`s, each with its own lineitem scan. A per-subquery rewrite into
  `LEFT DELIM_JOIN` + aggregate + filter is possible (the delim side is a one-column `DELIM_GET`,
  which is a unique build side), but each subquery would still scan lineitem and join the full
  candidate set, so it is unlikely to beat today's plan. The gain above comes from sharing one scan
  and one aggregate between `EXISTS` and `NOT EXISTS`, so an automatic rewrite has to merge sibling
  subqueries that scan the same table on the same key. That is not implemented here.
- `SUM` and `AVG` are not maintained, and neither are composite keys, floating-point or string
  extrema.
- Single GPU, single partition, unique build side on the probe's left join input.
- Subqueries in a merged chain must correlate on the same key column of the same outer rows and be
  directly nested; a subquery with a different key or a scan of another table is rewritten
  separately.
- The accumulators are allocated up front and are not covered by the join task's reservation;
  `emit` does size its reservation from the recorded build size.

## Code map

| Piece | File |
|---|---|
| Accumulator state, kernels | `src/cuda/probe_accumulate_impl.cu`, `src/op/aggregate/probe_accumulate_impl.hpp` |
| Join/emit shared context | `src/op/aggregate/probe_accumulate_context.hpp` |
| Join accumulate mode | `src/op/sirius_physical_hash_join.cpp` (`accumulate`) |
| Emit operator | `src/op/sirius_physical_probe_accumulate_emit.{hpp,cpp}` |
| Detection and plan rewrite | `src/planner/sirius_plan_probe_accumulate.cpp` |
| Dynamic-filter evidence for a DISTINCT over an opaque relation | `src/planner/dynamic_filter/build_filter_evidence.cpp` |
| Config | `enable_probe_accumulate` in `sirius_config.hpp` |

Tests: `test/cpp/operator/test_probe_accumulate_state.cpp` (kernel against a host reference),
`test/cpp/planner/test_probe_accumulate_detection.cpp` (what fuses and what declines),
`test/cpp/integration/test_gpu_execution_probe_accumulate.cpp` (fused result equals CPU result).

## Related change: DISTINCT build sides and dynamic filters

`build_relation_is_opaque` now treats a pure `DISTINCT` group-by (no aggregate expressions) over a
`CTE_REF` or `DELIM_GET` as opaque, so a join whose build side is `GROUP BY key` over a CTE still
publishes a dynamic filter. Without it the aggregate form above scans every lineitem row (0.651 s
instead of 0.542 s) because nothing prunes the probe scan.
