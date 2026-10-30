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
3. During physical tree construction, retain early scan refusals; defer native storage metadata walks until execution preparation.
4. Once the tree completes, record one verdict for each registered scan in the planning attempt before lowering any scan.
5. At execution preparation, preflight all bound providers on the query thread before acquiring native checkpoint leases and walking storage metadata. Produce physical evidence for each input unit as its metadata becomes available, then validate it before GPU consumption.
6. Publish only ready, validated inputs; drain preparation and GPU work and release leases before any CPU replay.

## Verdict at planning

For a completed physical tree, each registered scan receives `supported`, `unsupported`, or `incomplete` at finalize and at an execution-time rebuild. A supported verdict declares the physical checks that must still pass. `unsupported` records a known incompatibility and its reason; `incomplete` records missing required evidence or an unavailable interface. An unread future Parquet footer is neither a refusal nor incomplete evidence. An established unsupported verdict takes precedence over a later incomplete result.

The planner first records all registered scans in the completed tree, then decides whether to lower them. A refusal starts no dependent GPU work. An Iceberg or native refusal found while building the tree is retained so its original message wins if later planning also fails.

The certification budget measures only added in-memory `certify` work. Its 50 ms and 8 MiB thresholds count exceedances in production; only a latched test option can turn an exceedance into a refusal. The scan record reports `added`, `inherited_capture`, `borrowed` file count, and Iceberg `delete_preparation` separately. Provider-retained bytes are unknown at this pin.

A source rejected before its scan node exists still gets a classified lookup verdict. The reason distinguishes an unknown function, bind data or catalog mismatch, missing trusted reference, and callback mismatch. The existing planning error is returned unchanged.

## Source verification

Sirius verifies supported scan functions against trusted definitions and the current catalog. A matching function name is insufficient.

Loadable builds require ABI-compatible exports of `duckdb::TableScanFunction::GetFunction()` and `duckdb_extension_parquet_describe` from the host DuckDB module. Sirius checks that each resolved entry point belongs to that module. Both `RTLD_LOCAL` and `RTLD_GLOBAL` loading are supported.

If a trusted definition is unavailable, Sirius warns once per source and declines its GPU scans. CPU fallback still depends on fallback settings and source policy. Iceberg definitions are established when its extension loads and also require the host Parquet descriptor.

### DuckLake private binding boundary

`SIRIUS_BUILD_DUCKLAKE_BRIDGE` enables the optional connector and is off by default. It requires the pinned, host-patched DuckLake source and matching library. Private provider headers stay inside `binding_bridge.cpp`; installed packages include the matching provider library.

The connector verifies the provider callbacks and reads the original binding's snapshot, schema and transaction-local state. Each execution owns its readers and cursors; the saved CPU reader remains available for replay. Parquet and inline units share the existing preparation coordinator, split validation and query recovery. Discovery is lazy and must cover every unit in the binding. Inline readers supply detached batches to the shared Arrow driver; DuckLake retains transaction and provenance checks, while the driver handles admission, import and batch retirement.

| Input or condition | Behavior |
|---|---|
| Local Parquet, inline or mixed data | GPU scan through existing SQL and bound-plan entrypoints. |
| Selected columns | Up to 32 flat Boolean, integer, FLOAT/DOUBLE, VARCHAR, DATE or microsecond TIMESTAMP columns, including NULLs. |
| Schema evolution | Match stable field IDs; use typed `initial_default` for missing fields. INTEGER-to-BIGINT and FLOAT-to-DOUBLE widening are supported. |
| Ordinary Parquet deletes, local DVs and inline deletes | Combine visibility in the original file-row domain before residual filtering and projection. |
| Nested selections, external name/Hive mappings or other conversions | Reject GPU execution. Unselected nested columns do not prevent scalar reads. |
| Remote or encrypted inputs; snapshot-bearing deletes or per-row snapshot-filtered data | Reject GPU execution. |
| Missing statistics, defaults or conversions without usable statistics | Keep affected row groups and evaluate residual filters. |
| Malformed delete payload or invalid positions | Fail the scan; never treat invalid data as an empty delete set. |
| Insufficient preparation capacity or an oversized row | Return a resource error without truncation or unbounded allocation. |
| Cancellation or failure to drain GPU work | Do not replay on CPU. Retain owners needed by unfinished work. |
| Scan result caching | Disabled. |

Preparation reservations cover detached Arrow storage and retained delete data until their last consumer releases them. Native provider metadata, reader and chunk allocations remain outside that bound. The existing fallback settings and source policy govern SQL recovery; fragment errors propagate without CPU replay.

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

Physical checks run per input unit. A Parquet file is checked from its retained footer for codec, encryption, and type compatibility before cuDF reads it. The codec table follows the pinned libcudf 26.10.00 support set: UNCOMPRESSED, SNAPPY, GZIP, ZSTD, LZ4_RAW, and BROTLI; level encodings and an empty encoding list do not cause a refusal. Iceberg checks its file schema in the metadata worker before the profile check; qualified flat evolution compiles an immutable field-ID mapping and validates its physical read types before logical normalization; its former planning-time full-table footer sweep is gone. A pruned row group does not require a profile check. Type drift is allowed only for an export-only leaf when the actual export path performs the conversion; other semantic uses are refused before decode.

Each Parquet footer producer stores the original schema record and encryption evidence before reader normalization. Coalescing ends a run when the next file's original schema differs, including its `ARROW:schema` entry. This keeps an admitted export-only type drift out of a mixed-schema multi-file reader.

Native ranges carry their actual storage version and data and validity codec evidence. An insert delta combines the checks required by its persistent and transient segments and carries a checkpoint-key witness belonging to this query and database. A fresh or delta split with missing checks, a mismatched identity, or a non-empty payload without certificates is refused before materialization. An empty payload may have an empty certificate list.

Cached batches use pin identity, layout, structure, and applicable iteration and visibility checks. The query token and all applicable results are validated before publication to the split connector, so the prefetcher cannot consume an unadmitted batch. Streaming inputs do not produce storage splits.

### Bound split validators

Trusted planning binds a validator before publishing the scan contract. Execution uses that registered validator; an input split cannot supply its own validator. Adding a reader does not require type dispatch in the shared executor.

| Layer | Checks |
|---|---|
| `validate_split_for_gpu` | Contract, generation, certificate coverage, dependencies and registry ownership. |
| Bound materializer validator | Format-specific input identity, layout, visibility and retained owners. |
| Composite validator | Each input belongs to exactly one child materializer. |

Providers supplying `owned_arrow_scan_info` reuse the shared Arrow validator and retain batch storage through GPU completion. Coalescing must preserve certificates and dependencies.

## Preparation and publication

Each execution attempt owns its preparation units. A unit identifies one scan input and stays pending until all required evidence is ready: footer approval, native segment checks, row visibility, or checkpoint state. Readiness alone does not authorize publication. The coordinator checks cancellation before preparing prefetch buffers outside the gate. The gate checks again before readahead registration and connector insertion.

Providers declare bounded temporary allocations and retained storage through `preparation_request`. `prepare_statement_resources()` aggregates them into one statement reservation, prepares each provider's descriptors, then binds the shared admission. Failed preparation or binding rolls back all requests; runtime keeps the backing owner, not planning callbacks. Format-specific sizing stays in the adapter.

Shared DV helpers separate container parsing from provider visibility rules:

| Helper | Responsibility |
|---|---|
| `read_puffin_metadata` | Validate container framing and metadata; retain blob descriptors and properties without reading payloads. |
| `decode_deletion_vector_payload` | Validate one immutable deletion-vector-v1 payload and return owned positions within caller limits. |
| Provider adapter | Check file association, cardinality, snapshot applicability and complete delete coverage before certifying visibility. |

Metadata and decoded positions retain their charged backing. Neither helper grants scan eligibility; Iceberg and DuckLake apply their own provider checks.

Eligible Iceberg deletion-vector scans use this admission. The reservation covers retained descriptors and delete positions plus bounded temporary decode work, including allocation granularity. Buffers use the reserved backing, and immutable per-file delete sets retain it until their last consumer releases it. A prepared execution consumes its admission once; a later execution needs a fresh admission.

Temporary capacity is requested for multiple DV jobs, bounded by the DV unit count, scan workers, and preparation job, active-unit, and result-slot limits. If a request fails, admission halves the target concurrency down to one job. A valid partial grant is used only when it covers all retained results and at least one job's temporary bound; its size limits the admitted concurrency. Only one successful reservation is retained.

Unbounded inputs, the statement deletion-vector limit, or insufficient host capacity select eager delete loading before deferred execution starts. Positional-delete files keep their existing loading path. This route decision does not itself select CPU fallback. Once deferred work starts, read, validation, and allocation failures use the normal failure and replay rules; they never become an empty delete set.

Cancellation closes admission and publication and wakes blocked work. Completion waits for workers, queued results, query-owned callbacks, and published consumers to drain. GPU completion alone cannot report success while preparation is still open.

### Host Arrow batch source contract

The shared host Arrow path has two explicit boundaries. `import_arrow_host()` consumes one owned `ArrowSchema` plus `ArrowArray` record batch and retains its input capacity through the retirement lane. `materialize_arrow_stream()` consumes a sequential internal `arrow_batch_source`: each item carries an independently owned batch, a contiguous sequence number, and declared row and byte bounds; `nullopt` is the only EOS marker. The verifier compares the declared row count with the imported batch, sends results directly to a bounded sink, and calls `cancel()` on source, validation, import, or sink failure. Producers acquire preparation/output capacity and split oversized input before returning an item. An adapter may bridge an `ArrowArrayStream` one batch at a time, but this interface does not add a public Fragment Arrow-input API or a native Paimon/DuckLake adapter.

### Shared Arrow scan driver

`arrow_import_ingestible` drives one `arrow_batch_producer` on its declared execution domain. Trusted planning binds the schema, materialization program, original read identity and optional provider validator. Claims inspect cached readiness and acquire a preparation permit before entering the producer. The accepted root retains its charged backing and permit through GPU completion. Import waits on the borrowed task stream; it does not create a stream or synchronize the whole device.

| Input or lifecycle event | Behavior |
|---|---|
| Open push queue without input | Pending; an offer or terminal transition wakes the coordinator. |
| Non-null zero-row input | A real batch; validated and imported normally. |
| Normal EOS without any accepted input | One rootless empty descriptor; no Arrow import or input credit. A mixed outer scan can supply its own empty carrier instead. |
| Failure or cancellation | Propagates through the attempt; never treated as EOS. |
| Synchronous pin or cached rows | Unsupported throughout the source lifecycle. |

The push adapter accepts one offered root at a time. Concurrent offers return false without retaining another root. The external submitting thread waits until its root is copied or the queue terminates; it must hold no engine resource needed for progress. Offered views stay immutable and their owner covers all referenced buffers, including retained parent extents. Schema, row, copy and staging bounds are checked before ownership is accepted. The admitted job copies the selected window through the charged allocator; foreign backing is not adopted as reservation backing. Admission includes offered-input/copy overlap and normalization storage. `finish()` drains queued input before EOS; failure, cancellation and close release blocked offers.

### Execution and ownership

The coordinator claims work, coalesces results and publishes batches after GPU consumers are ready. Metadata-only attempts advance on the query owner without another thread. An attempt with query-owner or native-reader sources runs coordination independently of reader calls: on the manager's shared pool by default, or on a dedicated thread when `coordinator_workers` is `0`. All drivers use the same bounded control steps, publication gate and retirement path. Sources choose where their jobs run:

| Execution domain | Work and cleanup | Use |
|---|---|---|
| Metadata | Work runs in the metadata pool; source close runs on the control driver. | Independent file and metadata preparation. |
| Query owner | Provider operations, accepted-job disposal and close run on the statement thread. | Readers that require execution on the query's owning thread. |
| Native reader | Operations, accepted-job disposal and close run on one assigned native worker. | Readers with thread affinity. |

`gpu_ingestible::make_preparation_sources()` returns one or more preparation lanes for a scan. The default wraps `make_preparation_source()` for existing adapters. Each lane has its own execution domain, cursor and coalescer; all lanes publish to the scan's connector. Multi-lane factories supply an independent coalescer for every lane. Register the complete group before arming; an empty scan still supplies a source that reports exhausted.

Registration creates control state only. Readers initialize in their execution domain. Register and arm query-owner lanes on the statement thread. Control state must be safe to destroy without reader close if registration or native admission fails. The ingestible must outlive drain.

| Boundary | Contract |
|---|---|
| Cross-lane dependency | The adapter owns dependency state. Claim returns pending while inputs are unavailable; the adapter wakes the coordinator when they become ready. Never block a worker waiting for another lane. |
| Backpressure | Each lane has bounded lookahead and completion credits; the scan window is the sum of its lane windows. Consumption returns credit to the producing lane. Shared backing is charged once. |
| End of input | Close the scan output only after every lane has published or discarded its tail. Queued batches may still be consumed afterwards. |
| Reader close | Release only that lane's thread-affine resources. Other lanes may still be using shared adapter state. |
| Failure or cancellation | Stop all lanes, drain accepted work and close the output even when native admission failed before readers opened. |

The manager consumes the shared statement admission once; each ingestible installs its own execution state. Cleanup has two boundaries, separate from reader close:

| Hook | When it runs | What it releases |
|---|---|---|
| `stop_preparation()` | After preparation jobs and source close drain. | Producer state; retain resources still used by GPU consumers. |
| `finish_preparation()` | After GPU consumers retire. | Remaining scan-owned results and descriptors. |

Both hooks must be safe to repeat and to call after failed startup. Thread-affine reader cleanup stays in the source's `close` callback.

`try_claim` runs on the coordinator and must not enter SQL or a reader. It returns:

| Result | Meaning |
|---|---|
| Owning job | Resources are acquired; the job executes at most once. |
| Pending | No cursor advances and no permit is consumed. Other sources can progress until this source signals readiness. |
| Exhausted | No more jobs will be produced. An empty result batch alone does not mean exhaustion. |

Jobs that use a preparation unit carry its key and required inputs, derived by the provider from the bound scan. The shared registration path does not inspect format-specific bindings. Required evidence must be present before the result can enter coalescing; missing evidence fails the job.

The control driver reserves a claim before calling the adapter outside the gate. The returned job enters its execution domain before admission is checked. If stop wins that check, the domain disposes the job without executing it. Jobs retain their descriptors and resources through execution or disposal; their destructors do not stop source enumeration. The completion ticket reports work dropped without a stop request. Resource release and provider callbacks run outside the publication lock.

Query-owner and native sources allow one outstanding operation or unprocessed result per reader. They share the existing coalescing and publication path and do not consume metadata-worker permits. The query owner calls `advance()` while waiting for query completion: it advances metadata-only coordination or services one provider request. It remains available through drain; an external reset may request stop and wait, but cannot run provider operations itself.

The coordinator pool selects runnable or expired registrations by rotating through registration slots, advancing its cursor after each selection. This is round-robin slot selection, not FIFO wake order or earliest-deadline-first scheduling. Each attempt executes at most one step at a time; consecutive steps may run on different workers.

Steps return runnable, parked with an optional deadline, or finished. Racing wakes are retained, and workers service deadlines without owner ticks or extra timer threads. Control hooks must not retain thread-local state, require a fixed worker, or wait for I/O, capacity or dependencies. Provider work and disposal remain on the query owner; native work and disposal remain on the assigned native worker.

Native reader slots are shared across queries in a scan manager. An attempt reserves all its slots before any reader starts; insufficient capacity fails immediately as `late_failure_cause::resource`, without holding a partial group. CPU replay still requires the existing fallback, source-policy, transaction and cleanup checks; this classification grants no replay permission. Several readers may share a worker. A blocked native call can delay those readers, but not the coordinator or metadata pool. Worker and slot limits are documented in [configuration](configuration.md#siriusexecutorscan_managerpreparation).

### Control steps and driver waits

`state::control_step()` performs one bounded turn and returns `runnable`, `parked` with a wake generation and optional deadline, or `retire`. `retire_step()` advances cleanup without waiting for workers: cancel units, stop and drain metadata, close readers in their execution domains, then release remaining state. Pending stages park until a completion notification arrives. Query-owner, dedicated-thread and pooled drivers use these steps; the driver owns waits and serializes advancement.

Before sleeping, the query-owner and dedicated-thread drivers recheck the returned wake generation under the gate; stop also advances that generation. A wake between the step and the wait therefore cannot be lost. A timeout requests another step, which rechecks eligibility; it does not publish directly. Partial-batch deadlines use the coalescer clock; the driver also bounds waits for interrupt checks and the caller's idle limit.

Lock order is publication gate before pool or executor bookkeeping. Pool callbacks run without the pool mutex; executor completion must release executor bookkeeping locks before waking the gate. Adapter claims, input construction, capture destruction and completion notifications run outside the gate. Interruption checks, metadata inspection and the explicitly gated publication operations below may run under it; they must not reenter the gate. They must not wait for another attempt or synchronously drain their own execution domain.

The action limit bounds work count, not hook duration. Adapters used with the shared pool must preserve these control-path boundaries:

| Call on the control path | Current boundary |
|---|---|
| `try_claim` | Returns work, pending or exhausted without entering a reader. Parquet/Iceberg resolve and read inside the job. DuckDB-native storage walks must finish during manager setup; its legacy claim fallback can still perform the walk if setup is bypassed. |
| Job submission | Metadata enqueue does not wait for a worker slot; query-owner/native handoff uses the admitted mailbox or lease. Submission may allocate and take locks. |
| Coalescer `advance`, `partial_emit`, `flush`; input construction | Manipulate prepared metadata and allocate descriptors. Datasource duplication shares existing handles. Allocation, locks and capture destruction still run inline. |
| `prepare_prefetch` | Does not wait for eviction or device progress. It takes cache locks, allocates buffers and invokes ready completion callbacks inline. |
| Final publication | Readahead registration and connector insertion run under the gate; no arbitrary adapter hooks are allowed there. Their locks and allocations remain part of the step. |
| Source/output close and released captures | Metadata close and output notifications can run inline. Query-owner/native reader close and accepted-job disposal stay in their assigned execution domain. |
| Error reporting and retirement | Error observers and stop callbacks run inline. Unit cancellation, queue disposal and final cleanup yield between bounded steps; waiting for dispatcher drain or reader close parks the attempt. |

Stop requests immediately close admission and publication and signal running jobs; they do not walk the unit registry. Retirement keeps one control owner and cancels at most `max_control_work` units per step before proceeding to queue disposal and final cleanup, which use the same action limit. A pending unit removed during cancellation is cancelled by its retiring owner before release. Pending unit records may remain pending until their cancellation step, but cannot accept new inputs after the gate closes. Waiting for worker completion parks the attempt; unfinished cleanup keeps it runnable. The dispatcher sends one notification after stopped work and its captures drain; reader completion wakes the same driver. The query owner keeps servicing provider operations. `drain()` returns only after final cleanup and completion notifications have returned and control ownership has ended.

### Publication and cancellation

Publication uses typed `scan_operator_input` objects. Input construction and prefetch preparation run outside the publication lock. Only cancellation validation, readahead registration and connector insertion run inside it; adapters cannot add arbitrary callbacks there. Cache preparation may create requests, attach buffers and queue eviction before the final check. These buffers belong to the bounded cache pool, not the preparation job budget; rejecting publication does not immediately free them.

| Situation | Required behavior |
|---|---|
| External stop or drain | Close admission and publication, signal running jobs and wake the control driver. The driver retires queues after the active claim returns; drain callers wait for retirement. |
| Stop during claim | Keep the source alive through ownership transfer. Dispose of a rejected job in its execution domain before closing the reader. |
| Final gate rejects publication | Do not register readahead or insert into the connector. Destroy the input outside the gate and return its output credit once. |
| Cache preparation is abandoned | Input disposal marks its requests disposed; repeated disposal is harmless. The cache evictor drops each request's subscriptions once and reclaims buffers according to policy, respecting active reads. |
| Cache maintenance after drain | Preparation drain finishes input disposal, not cache eviction. Cache-owned generations, buffers and eviction requests may outlive the query; they must not depend on its reader or transaction. |
| Failure during preparation or commit | Dispose the surviving input. Any readahead work already registered follows normal query teardown; cache maintenance retains its own ownership. |
| Queued provider work | Dispose of it in its assigned execution domain without invoking the reader. |
| Active provider work | Use a supported interrupt or wait for return; discard late results before closing. The framework cannot forcibly interrupt SQL or native calls. |
| Between provider operations | Check the attempt's stop state before re-entering the reader. A DuckDB connection interrupt flag may be reset by a later query. |
| Reader close | Close without waiting for output capacity, and release thread-affine resources even when close reports an error. Remaining state must be safe to destroy on other threads. |

`preparation_stop_token` reaches the adapter task unchanged and distinguishes normal completion, execution failure and user cancellation. Normal completion or failure cleanup must not blindly interrupt the original transaction or replace the first query error. Logical completion does not finish cleanup: the query owner remains available through provider close. After physical work drains, a close failure retains checkpoint keys and query resources until runtime shutdown and prevents CPU replay. See [query retirement](query-lifecycle.md#retirement) for shutdown and close-failure handling.

### Provider read view

| Boundary | Requirement |
|---|---|
| Reader creation | Preserve the original binding, snapshot, schema and transaction; use a separate mutable cursor from the CPU reader. |
| Repeated execution | Create fresh reader state from that execution's binding. Never reuse a cursor across transaction generations. |
| Provider SQL | Use the provider's internal SQL path on CPU; never re-enter the active outer `ClientContext::Query`. Ordinary user SQL keeps its normal GPU eligibility. |
| Session lifetime | Retain context, transaction and plan owners through reader close and their last dependent use. A weak prepared handle is insufficient. |
| Provider failure | Preserve the error and close safely. Do not reopen an invalid transaction to continue the read. |

### Output ownership and limits

Published batches must own their storage independently of the reader and transaction. Detach reusable reader buffers before publication. Multiple descriptors or GPU consumers sharing an allocation share its owner and charge it once.

| Resource | Lifetime or limit |
|---|---|
| Preparation jobs and results | Bounded slots limit outstanding work, not total memory. |
| Output lookahead | Counts results, coalescer input and queued batches. Taking a batch returns lookahead capacity, not storage; submitted work may overshoot the target. |
| Reserved backing | The last physical owner releases the allocation and its reservation. Provider SQL, SDK allocations and decoder working memory are not automatically covered. |
| Arrow input | Use the shared importer with an explicit staging PMR kept alive by the capacity owner. Retain host backing, staging and capacity through the last asynchronous read, including cancellation and failure. Release callbacks and staging deallocation must be safe outside the statement thread. |
| GPU output | May outlive the source and imported host batch. After destroying imported outputs, call `lane.quiesce()` even if the lane is idle; retain the stream, device allocator and backing through that barrier. |

Partial publication follows the configured fill and residence limits. Residence starts at the first retained input; later arrivals do not reset it. Backpressure may delay publication. See [preparation settings](configuration.md#siriusexecutorscan_managerpreparation) for defaults.

## Native checkpoint lease

Native scans and pinning acquire a shared checkpoint lease before inspecting storage layouts. Stored ranges are revalidated before decoding. The lease lasts through cleanup; idle prepared statements hold none.

An active lease makes `CHECKPOINT` fail and `FORCE CHECKPOINT` wait. A waiting forced checkpoint blocks new transactions except read-only ones until it is interrupted or the checkpoint keys are released. Sirius starts internal metadata transactions read-only and rejects an internal read-write start while its execution window holds checkpoint leases, avoiding a wait on its own protection.

Failed cleanup retains checkpoint keys until the Sirius runtime is destroyed and prevents CPU fallback. Interrupting `FORCE CHECKPOINT` ends its wait but does not release Sirius's keys.

Provider metadata connections need the same ordering. Starting a non-read-only DuckDB backend transaction while holding a checkpoint lease can deadlock with a waiting forced checkpoint. `BEGIN` or an existing connection alone does not prove that the backend transaction has started.

The scan manager calls `gpu_ingestible::preflight_for_query()` on the query thread for all bound providers, including cache hits, before scan preparation or checkpoint-lease acquisition. Each adapter is responsible for discovering its metadata targets and validating the original transaction.

| Boundary | Requirement |
|---|---|
| Preflight | Establish or validate the bound backend transaction. Do not rebind, acquire checkpoint leases, start readers or launch background work. Repeat for every execution attempt. |
| Preflight failure or cancellation | Stop before scan preparation starts. Release temporary resources without replacing the original error or rolling back another provider's bound transaction. |
| Reader initialization and pulls | Validate owner thread, target, transaction generation and validity before using the retained backend witness. Use lookup that cannot start or reopen a backend while checkpoint leases are held; never dereference a stale witness. |
| Unknown or replaced backend | Refuse provider work. Preflight does not authorize reconnecting under a checkpoint lease. |
| Metadata writes | Require their own lock-order policy during checkpoint contention. |

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

Source discovery retains its first failure while continuing to check siblings. Known S3 and stream restrictions take precedence. If discovery failure prevents replay, the query error includes both the discovery failure and the underlying GPU error. Non-standard C++ exceptions are converted to ordinary query errors; cancellation still propagates immediately.

On the explicit path, the rebound validator returns a policy refusal separately from the CPU result. Such a refusal counts as no replay; an admitted CPU execution counts as replay even if it returns an ordinary error or no rows.

Iceberg metadata queries use a separate read-only connection with recursive GPU execution disabled and complete planning before native leases are acquired.

Related documentation: [Scan](scan.md), [Physical Plan Generation](physical-plan-generation.md), and [Streaming Fragments](streaming-fragments.md).
