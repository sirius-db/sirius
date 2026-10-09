# Plan: CPU fallback and deployment model

## Why

Two problems:
- **Unsupported operators fail the query.** The translator works off an allowlist, so anything outside it is refused.
- **A CN and a native BE can't share an exchange,** because the CN rejects `ChunkPB`.

For a customer with an existing cluster, that leaves three shapes:
- all-GPU, with unsupported queries failing;
- routing whole queries to one kind of node or the other;
- operator-level fallback in the middle of a query, as Gluten does on Spark.

A related question: how Sirius relates to spark-rapids, which already has CPU fallback and a qualification tool.

Starburst's GPU work never fails a query for coverage: an unsupported operator runs on the CPU.

## Today

- **All-GPU.** An unsupported plan fails the query with the translator's reason.
- **`ChunkPB` is rejected** on purpose (`nixl_chunk.rs`), so a stock BE can't be mistaken for a CN.
- **Sirius has a DuckDB CPU fallback** (`enable_duckdb_fallback`, `src/sirius_context.hpp:955`, on by default in Sirius). It's turned off in StarRocks testing, so GPU gaps stay visible.

## Steps

### Step 1: DuckDB fallback inside the CN

- Turn the fallback on behind a CN setting, off in coverage runs.
- Check that it works for fragments whose inputs are exchange streams, not just tables.
- A fragment the translator refuses outright, such as a missing plan-node type, still fails. This step covers only what the translator accepts but Sirius can't run on the GPU.
- **Tests:** a fragment with an operator that has no GPU implementation runs on the CPU and matches DuckDB, and the CN logs that it fell back.

### Step 2: query-level routing

- The FE sends whole queries the CN can't translate to native BEs in the same cluster. This needs FE support, for example a check by the CN at planning time or a session hint.
- **Open question for StarRocks:** whether the FE can ask a CN to "pre-check" a plan.

### Step 3: operator-level fallback (mixed exchange)

- Let CNs and BEs exchange data: accept `ChunkPB` on the CN, or convert at the boundary.
- Then the FE could place a fragment the CN refuses on a BE.
- This is the largest change, so do it only if Steps 1–2 aren't enough for customers.

## Open questions

- Which shape customers expect first. This decides how far past Step 1 to go.
- How Sirius on StarRocks relates to spark-rapids, and whether a qualification tool (which queries would run on the GPU) is worth building from the translator's refusal reasons.

## Out of scope

- Moving a running fragment from the GPU to the CPU mid-flight.
