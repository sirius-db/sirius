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
3. During physical tree construction, acquire native checkpoint leases before preparing storage metadata and retain any early scan refusal.
4. Once the tree completes, record one verdict for each registered scan in the planning attempt before lowering any scan.
5. Produce physical evidence for each input unit as its metadata becomes available, then validate it before GPU consumption.
6. Publish only ready, validated inputs; drain preparation and GPU work and release leases before any CPU replay.

## Verdict at planning

For a completed physical tree, each registered scan receives `supported`, `unsupported`, or `incomplete` at finalize and at an execution-time rebuild. A supported verdict declares the physical checks that must still pass. `unsupported` records a known incompatibility and its reason; `incomplete` records missing required evidence or an unavailable interface. An unread future Parquet footer is neither a refusal nor incomplete evidence. An established unsupported verdict takes precedence over a later incomplete result.

The planner first records all registered scans in the completed tree, then decides whether to lower them. A refusal starts no dependent GPU work. An Iceberg or native refusal found while building the tree is retained so its original message wins if later planning also fails.

The certification budget measures only added in-memory `certify` work. Its 50 ms and 8 MiB thresholds count exceedances in production; only a latched test option can turn an exceedance into a refusal. The scan record reports `added`, `inherited_capture`, `borrowed` file count, and Iceberg `delete_preparation` separately. Provider-retained bytes are unknown at this pin.

A source rejected before its scan node exists still gets a classified lookup verdict. The reason distinguishes an unknown function, bind data or catalog mismatch, missing trusted reference, and callback mismatch. The existing planning error is returned unchanged.

## Source verification

Sirius verifies supported scan functions against trusted definitions and the current catalog. A matching function name is insufficient.

Loadable builds require ABI-compatible exports of `duckdb::TableScanFunction::GetFunction()` and `duckdb::ParquetScanFunction::GetFunctionSet()` from the host DuckDB module. Sirius checks that each resolved factory belongs to that module. Both `RTLD_LOCAL` and `RTLD_GLOBAL` loading are supported.

If a trusted definition is unavailable, Sirius warns once per source and declines its GPU scans. CPU fallback still depends on fallback settings and source policy. Iceberg definitions are established when its extension loads and also require the host Parquet factory.

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

Physical checks run per input unit. A Parquet file is checked from its retained footer for codec, encryption, and type compatibility before cuDF reads it. The codec table follows the pinned libcudf 26.08.01 support set: UNCOMPRESSED, SNAPPY, GZIP, ZSTD, LZ4_RAW, and BROTLI; level encodings and an empty encoding list do not cause a refusal. Iceberg checks its file schema in the metadata worker before the profile check; its former planning-time full-table footer sweep is gone. A pruned row group does not require a profile check. Type drift is allowed only for an export-only leaf when the actual export path performs the conversion; other semantic uses are refused before decode.

Each Parquet footer producer stores the original schema record and encryption evidence before reader normalization. Coalescing ends a run when the next file's original schema differs, including its `ARROW:schema` entry. This keeps an admitted export-only type drift out of a mixed-schema multi-file reader.

Native ranges carry their actual storage version and data and validity codec evidence. An insert delta combines the checks required by its persistent and transient segments and carries a checkpoint-key witness belonging to this query and database. A fresh or delta split with missing checks, a mismatched identity, or a non-empty payload without certificates is refused before materialization. An empty payload may have an empty certificate list.

Cached batches use pin identity, layout, structure, and applicable iteration and visibility checks. The query token and all applicable results are validated before publication to the split connector, so the prefetcher cannot consume an unadmitted batch. Streaming inputs do not produce storage splits.

In tests, the scan statistics retain each query's readahead registrations by file and its successful memory-prefetch conversions after the query state is drained. These observations do not change scan admission or publication.

## Preparation and publication

Each execution attempt owns its preparation units. A unit identifies one scan input and stays pending until all required evidence is ready: footer approval, native segment checks, a delete set, or checkpoint state. Readiness alone does not authorize publication. The gate rechecks cancellation and validates the split before prefetch initialization or connector insertion.

Eligible Iceberg deletion-vector scans share one host-memory admission for the statement. The reservation covers retained descriptors and delete positions plus bounded temporary decode work, including allocation granularity. Buffers use the reserved backing, and immutable per-file delete sets retain it until their last consumer releases it. A prepared execution consumes its admission once; a later execution needs a fresh admission.

Unbounded inputs, the statement deletion-vector limit, or insufficient host capacity select eager delete loading before deferred execution starts. Positional-delete files keep their existing loading path. This route decision does not itself select CPU fallback. Once deferred work starts, read, validation, and allocation failures use the normal failure and replay rules; they never become an empty delete set.

Cancellation closes admission and publication and wakes blocked work. Completion waits for workers, queued results, callbacks, and published consumers to drain. GPU completion alone cannot report success while preparation is still open.

### Limits and diagnostics

Preparation limits are internal C++ settings derived from the scan worker count. They bound active jobs, units, pending results, and coalescing work. By default, coordinator waits check interruption every 10 ms; draining an active read can take longer. Timed flushing of underfilled batches is disabled by default. These controls add no SQL, YAML, or environment configuration keys.

Optional test observations record planning completion, first ready input, first publication, queue peaks, admission bytes and permits, and legacy/deferred route reasons. Datasource counters separate preparation reads from execution reads; Puffin counters include opens, requested and returned bytes, and failures.

The hidden `[preparation_cost]` test accepts one SELECT through `SIRIUS_TEST_PREPARATION_COST_SQL_FILE`, checks results against CPU execution, and reports warm-query samples with observations enabled and disabled. `test/scripts/compare_query_cost.py` alternates baseline and candidate runs and reports median and P95 latency. `test/scripts/trace_query_io.py` saves file-read syscall evidence separately from timing runs. Missing observations are reported explicitly; datasource bytes and file syscalls are not a complete measure of physical storage traffic.

## Native checkpoint lease

Native scans and pinning acquire a shared checkpoint lease before inspecting storage layouts. Stored ranges are revalidated before decoding. The lease lasts through cleanup; idle prepared statements hold none.

An active lease makes `CHECKPOINT` fail and `FORCE CHECKPOINT` wait. A waiting forced checkpoint blocks new transactions except read-only ones until it is interrupted or the checkpoint keys are released. Sirius starts internal metadata transactions read-only and rejects an internal read-write start while its execution window holds checkpoint leases, avoiding a wait on its own protection.

Failed cleanup retains checkpoint keys until the Sirius runtime is destroyed and prevents CPU fallback. Interrupting `FORCE CHECKPOINT` ends its wait but does not release Sirius's keys.

## CPU replay policy

CPU replay runs a failed GPU query on DuckDB. It requires `enable_duckdb_fallback`, source permission, and successful cleanup of any entered execution window. Cancellation is not replayable.

A late GPU, physical-input, certificate, or exhausted-retry failure stops and drains the execution window before this decision. Transparent execution replays the retained CPU plan on its captured transaction only if that transaction remains valid, no result was emitted, and cleanup and query state permit replay. A non-read-only statement increments a counter but does not alone refuse replay. Explicit `gpu_execution()` re-executes on a fresh transaction. Failure causes and replay outcomes are counted separately; the returned terminal error determines the recorded cause.

| Source or discovery result | CPU replay |
|---|---|
| Local files and DuckDB-native tables | Permitted. |
| Sirius-owned S3 data | Forbidden. |
| Streaming inputs | Forbidden. |
| Incomplete source discovery | Forbidden. |
| Other unclassified sources | Entry-point default. |

The policy checks bound sources, including those hidden behind views. Transparent execution retains the original CPU plan. Explicit `gpu_execution()` checks its bind-time policy and validates the newly bound CPU plan before replay, so a replaced view cannot inherit stale permission. SQL-text and filesystem checks also block S3 replay.

On the explicit path, the rebound validator returns a policy refusal separately from the CPU result. Such a refusal counts as no replay; an admitted CPU execution counts as replay even if it returns an ordinary error or no rows.

Iceberg metadata queries use a separate read-only connection with recursive GPU execution disabled and complete planning before native leases are acquired.

Related documentation: [Scan](scan.md), [Physical Plan Generation](physical-plan-generation.md), and [Streaming Fragments](streaming-fragments.md).
