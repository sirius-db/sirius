# Multi-fragment demo cluster

A runnable StarRocks front end + Rust compute node + Sirius engine on one host, used to exercise
multi-fragment execution end to end against a real TPC-H Q6 query.

This worktree is `dev` plus the repository-backed streaming work
([`docs/super-sirius/streaming-sessions.md`](../../docs/super-sirius/streaming-sessions.md))
plus the `multi-fragment-execution` compute node.

## What this does and does not exercise

The compute node coordinates fragments the way it does today: `LocalExchange` is a
receiver-first rendezvous that buffers each sender's **fully materialized** Arrow result and
writes it to a temporary parquet file the receiver re-scans as a `local_files` scan.

The streaming primitives in this branch — `STREAMING_SOURCE`, `STREAMING_SINK`,
`exec::stream_lifecycle`, `exec::stream_session` — are **built into the engine and unit-tested,
but not yet wired into this compute node**. Wiring them is Milestone 2 (the cxx-FFI boundary and
plan launcher), and it is a seam swap rather than a rewrite: each primitive replaces exactly one
mechanism here.

| Compute-node mechanism today | Streaming replacement |
|---|---|
| `ExchangeFile` temp parquet + `local_files` re-scan | `push(stream_id, batch)` → source repository, native |
| `ExchangeOutput` (the whole result) | incremental native `data_batch` push |
| `LocalExchange` rendezvous (expected sender **count**) | `stream_lifecycle` expected sender **set** + a session registry |
| `ExchangeKey{fragment_instance_id, node_id}` | the direction-separated input `stream_id` |
| `per_exch_num_senders` + `exec.sender_id` | the source's expected sender set |
| `exec.destinations` | one sink partition per destination + the wrapper's routing table |
| `FragmentExecutor::execute → FragmentResult` (sync) | build session + submit; `push`/`pull` over cxx-FFI |
| `ResultStore` + `fetch_data` | root `STREAMING_SINK` + `pull`/`wait`/`drained` |

So: this demo proves the cluster runs the query correctly on the engine that carries the new
primitives. It does not yet prove the query flows *through* them.

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

-- The simplest two-fragment shape: a scan+filter+project sender and a single aggregate
-- receiver, joined by a gather exchange.
SET new_planner_agg_stage = 1;

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

Point `path` at whatever TPC-H `lineitem` parquet you have.

## The pre-packaged front end

`cluster` depends on `fe-check`, not `fe-build`: this demo ships the front end already packaged
under `starrocks/output/fe` so bringing the cluster up does not require a multi-hour Maven build
of the whole StarRocks front end. `fe-check` just asserts the package is present and tells you
what to run if it is not:

```bash
git submodule update --init --recursive experimental/starrocks/starrocks
pixi run fe-build    # long
```

Everything else — the compute node and the Sirius engine — is built from this worktree by
`cn-build` → `engine-build`.
