# Multi-fragment demo cluster

A runnable StarRocks front end + Rust compute node + Sirius engine on one host, used to exercise
multi-fragment execution end to end against a real TPC-H Q6 query.

This worktree is `dev` plus the repository-backed streaming work
([`docs/super-sirius/streaming-sessions.md`](../../docs/super-sirius/streaming-sessions.md))
plus the `multi-fragment-execution` compute node.

## What this exercises

**A fragment's output crosses the exchange boundary as native `cucascade::data_batch` handles.**
It is never converted to Arrow, never written to a file, and never copied: the sender's rows stay
in GPU memory, parked in the engine's output repository, and the receiver's `STREAMING_SOURCE`
takes them from there.

| Was | Is |
|---|---|
| `ExchangeFile` temp parquet + `local_files` re-scan | `Fragment::relay_from` — a pointer move between two live fragments |
| `ExchangeOutput` (a whole materialized Arrow result) | native `data_batch` handles, parked on the GPU |
| The receiver's `EXCHANGE_NODE` lowered to a file read | lowered to a read of `sirius_stream_<node_id>` |
| `FragmentExecutor::execute → FragmentResult` per fragment | `FragmentExecutor::run`; only a *result* fragment produces Arrow |
| `LocalExchange` buffering sender data | `LocalExchange` tracking only *which* senders have produced |

`ExchangeFile` is deleted, which is what makes a correct answer evidence rather than coincidence:
with the temp-parquet path physically gone, the rows can only have crossed as native batches.
`grep -r ExchangeFile experimental/` finds nothing but this file, and no file appears under
`$TMPDIR/sirius-starrocks-cn` during a run.

The CN logs each boundary it crosses:

```
INFO sirius_starrocks_cn::engine: relayed native batches across a fragment boundary
     stream_id=2 sender_id=0 batches=1
```

### What it does not exercise yet

- **Sequential fragments.** The engine serializes queries, so a sender runs to completion before
  its receiver starts. Concurrency needs per-query lifecycle isolation.
- **A pre-filled stream.** Senders finish before the receiver is built, so the live producer path
  — a push arriving while the receiver runs — is still untested here.

## Running it

Bring the cluster up (front end + compute node + engine), and leave it in the foreground:

```bash
cd experimental/starrocks
pixi run cluster
```

In a second terminal, open a MySQL client against the front end:

```bash
cd experimental/starrocks
pixi run client        # mysql --host 127.0.0.1 --port 9030 --user root
```

Then:

```sql
SHOW COMPUTE NODES;

-- The FE's default plan is two-phase: a partial aggregate ("update serialize") in the scan
-- fragment and a merge aggregate ("merge finalize") behind a gather exchange. Both phases
-- translate; only the partial states cross the fragment boundary.
WITH lineitem AS (
  SELECT *
  FROM FILES(
    "path"="file:///home/ubuntu/git/sirius/scratch/tpch_sf1/lineitem/part.0.parquet",
    "format"="parquet"
  )
)
SELECT
  sum(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= date '1997-01-01'
  AND l_shipdate < date '1998-01-01'
  AND l_discount BETWEEN 0.03 - 0.01 AND 0.03 + 0.01
  AND l_quantity < 24;
```

Point `path` at whatever TPC-H `lineitem` parquet you have. On TPC-H SF1 that query returns

```
revenue
61567694.95019999
```

which matches DuckDB on CPU over the same file (`61567694.9502`).

A `GROUP BY` is worth running too — grouped two-phase aggregation runs at the FE's default
settings now that the hash-partitioned sink exists (#838): the partial fragment shuffles each
group-key partition to its merge instance, so the CN logs several boundary crossings for one
query:

```sql
WITH lineitem AS (SELECT * FROM FILES(
  "path"="file:///home/ubuntu/git/sirius/scratch/tpch_sf1/lineitem/part.0.parquet",
  "format"="parquet"))
SELECT l_returnflag, count(*) AS n, sum(l_quantity) AS qty
FROM lineitem GROUP BY l_returnflag ORDER BY l_returnflag;
```
