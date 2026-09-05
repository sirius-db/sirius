**15 · Concurrent schema reads and metadata caching**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: reduce CN-side `FILES()` schema inference latency for many-file and repeated-query workloads while validating every file. Prerequisite: [00 · Trustworthy measurements and benchmark coverage](00-measurement-and-benchmarks.md). Independent of exchange/runtime changes.

**Current behavior and code map**

`parquet_files_schema` awaits the first file and then every remaining file sequentially. Each call opens files and reads footer metadata. It deliberately requires agreement across all files; replacing this with sampling would change the correctness contract.

| Source | Planned change |
|---|---|
| [file_schema.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/file_schema.rs#L62) | Bounded concurrent metadata reads, deterministic validation, optional cache. |
| [compute_node_service.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/compute_node_service.rs#L1945) | Request path normalization/deduplication and request-level timing. |
| [tunable.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/tunable.rs) | Validated concurrency/cache budget settings, names to be chosen. |
| [bench.sh](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/benchmarks/tpch/bench.sh) | Separate schema/planning and end-to-end latency. |

**Proposed design**

Assign each input file a stable position in the request, schedule a bounded number of asynchronous footer reads, and validate results in deterministic input order. Completion order must not choose the canonical schema or which mismatch is reported. Preserve the first file's spelling and existing case-insensitive name/type agreement rules.

Bound concurrency per request and globally across simultaneous requests so N requests cannot each open an unlimited set of files. Deduplicate repeated paths within a request when they refer to the same validated identity, while preserving expected schema/result behavior.

Add caching only after measuring repeated work. Key entries by a reliable local-file version contract, relevant parsing options, and resolved identity. Detect changes during a read where supported; immutable datasets can use an explicit immutable-generation contract. Path plus TTL is not enough for correctness when files can be replaced in place. Coalesce concurrent misses for one identity and bound cache bytes/entries. Error caching, if any, must be brief and must not hide a fixed file.

**Implementation slices**

1. **Measurement:** instrument requested/distinct file counts, footer opens/bytes, per-file duration, queue wait, and total schema validation time. Establish whether FE caching already removes repeated calls in the target workload.
2. **Bounded reads:** add an async task queue or equivalent supported by the pinned Tokio version; preserve deterministic canonical schema and mismatch errors. Clean up pending work on request cancellation.
3. **Shared admission:** add a CN-wide limit and fairness between requests. Avoid one enormous file list starving small schema requests.
4. **Optional versioned cache:** implement reliable invalidation and coalesced misses, plus hit/miss/eviction counters. Keep complete validation of all file identities even when footer contents are cached.

**Tests**

Reuse the existing schema-only Parquet fixtures. Delay reads to force different completion orders and verify identical canonical schema/errors. Test name-case variants, type/count mismatch, duplicate paths, missing files, cancellation, many simultaneous requests, cache eviction, replace/modify during read, and repaired errors.

Acceptance: every distinct file is validated; concurrency never exceeds declared limits; no descriptor/task leak after cancellation; cached metadata cannot make a changed file silently pass. Invalid schema still fails before query execution.

**Benchmark**

Use 1/10/100/1000-file sets where practical, several footer widths, warm/cold page cache states, and simultaneous clients. Sweep concurrency 1/2/4/8/16 as experiment points. Report schema RPC latency, FE planning latency, file opens, cache hit rates, CPU, and total query time. Keep GPU execution constant to identify the actual contribution.

Adopt concurrency when it improves schema latency without saturating storage or harming concurrent requests. Adopt caching only if repeated validations remain material and invalidation has a concrete contract. A few-file workload may justify the simpler uncached implementation.

**Rollout and fallback**

Default to concurrency one for a controlled baseline, then choose a measured bound. Cache is independently switchable for new requests. On identity uncertainty, bypass the cache and reread; never skip validation. Draining existing reads is sufficient to lower concurrency safely.

**Open decisions**

Choose the file version contract, global scheduler placement, and deterministic error precedence before coding. Remote URI support remains outside this path: the current schema reader supports local paths, and this optimization should not implicitly add new storage backends.
