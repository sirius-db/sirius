# Scan Contracts

Scan contracts validate GPU scan inputs, split ownership, checkpoint protection, and CPU fallback. Format support and row visibility remain the responsibility of the [scan implementations](scan.md).

## Architecture

| Part | Responsibility |
|---|---|
| Read identity | Source, schema, reader options, and input used for equality checks. |
| Read view | Read identity plus captured evidence and statement context. |
| Scan contract | Read view, requested columns, predicates, and materialization requirements. |
| Query registry | Scan contracts and original/candidate comparison results. |
| Split certificate and dependencies | Split ownership and metadata needed for decoding. |

### Execution path

1. Verify sources and capture the original logical bindings before copying the plan.
2. Compare GPU inputs with the original logical and physical bindings.
3. Acquire native checkpoint leases before preparing storage metadata.
4. Validate fresh splits before materialization.
5. Drain scan work and release leases before any CPU fallback.

## Source verification

Sirius verifies supported scan functions against trusted definitions and the current catalog. A matching function name is insufficient.

Loadable builds require compatible DuckDB native and Parquet factory exports. Without them, affected GPU scans are declined; CPU fallback remains subject to source policy. Iceberg definitions are established when its extension loads.

## Matching the bound input

Transparent execution compares source, bound schema, reader options, and input:

| Source | Input compared |
|---|---|
| DuckDB-native table | Database and table identity. |
| Parquet and Iceberg | Bound file paths, preserving duplicates. |
| Streaming input | Stream identity. |

Projection and predicates are outside read identity. Iceberg also requires matching snapshot selectors: identical file paths can belong to snapshots with different deletes.

| Candidate plan | Required correspondence |
|---|---|
| Logical plan copy | Each scan matches its logical original; physical input sets also agree. |
| Single-scan SQL replan | Physical inputs match; Iceberg additionally needs original logical selector evidence. |
| Multi-scan SQL replan | Declined because scan order cannot prove correspondence. |

Missing correspondence or changed inputs prevent GPU admission. Rebuilds and repeated prepared executions validate again. File metadata is observational, not part of identity equality; overwrites at the same path may go undetected.

## Split ownership

Each fresh split must belong to its consuming scan, even when another scan reads the same files. Parquet checks ownership before coalescing and validates each slice's certificate and footer before materialization.

Cached batches use pin identity and visibility checks. Streaming inputs do not produce storage splits.

## Native checkpoint lease

Native scans and pinning acquire a shared checkpoint lease before inspecting storage layouts. Stored ranges are revalidated before decoding. The lease lasts through cleanup; idle prepared statements hold none.

An active lease makes `CHECKPOINT` fail and `FORCE CHECKPOINT` wait. A waiting forced checkpoint can also delay new writes. Failed cleanup may retain the lease and prevents CPU fallback.

## CPU replay policy

CPU replay runs a failed GPU query on DuckDB. It requires `enable_duckdb_fallback`, source permission, and successful cleanup of any entered execution window. Cancellation is not replayable.

| Source or discovery result | CPU replay |
|---|---|
| Local files and DuckDB-native tables | Permitted. |
| Sirius-owned S3 data | Forbidden. |
| Streaming inputs | Forbidden. |
| Incomplete source discovery | Forbidden. |
| Other unclassified sources | Entry-point default. |

The policy checks bound sources, including those hidden behind views. Transparent execution retains the original CPU plan. Explicit `gpu_execution()` checks its bind-time policy and validates the newly bound CPU plan before replay, so a replaced view cannot inherit stale permission. SQL-text and filesystem checks also block S3 replay.

Iceberg metadata queries use a separate read-only connection with recursive GPU execution disabled and complete planning before native leases are acquired.

Related documentation: [Scan](scan.md), [Physical Plan Generation](physical-plan-generation.md), and [Streaming Fragments](streaming-fragments.md).
