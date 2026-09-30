# Concurrent SQL queries

## Configuration and scope

Use separate DuckDB connections sharing one DatabaseInstance. Configure the admission limit at
startup, then run normal SQL:

```yaml
sirius:
  max_concurrent_queries: 4
  executor:
    scan_manager:
      memory_prefetcher:
        enable: true
```

The default is **1**. A permit covers query initialization, execution and resource retirement.
Additional queries wait in a cancellable FIFO queue. A materialized result owns its returned
buffers and does not keep the permit after execution finishes.

The old `sirius.executor.scan_manager.max_concurrent_queries` key remains a deprecated alias.
Conflicting values are rejected. This is a per-runtime limit, not a process-wide limit shared
between independent DatabaseInstances.

Keep the default per-thread reservation tracking mode. `per_stream_reservation: true` is rejected
when the limit exceeds one: conversions may change streams, and the underlying per-stream tracker
reset cannot safely race allocation on another user's stream.

Concurrent FFI/streaming fragments on one Context are outside this contract. Nested execution
windows fail explicitly; a dormant fragment must not retain a permit needed by its producer.
Independent DatabaseInstances sharing the process GPU allocator also require a separate ownership
contract. Multiple connections to one DatabaseInstance use the shared runtime described here.

## Admission, planning and maintenance

Queries receive scheduling IDs from their arrival tickets. Tasks favor the oldest runnable query
that can use a ready GPU. Running kernels are not preempted. A task waiting for memory releases its
worker slot, letting another task run and potentially free memory. Equal GPU shares are not promised.
IDs fail explicitly at the 31-bit scheduling limit rather than wrapping.

Planning takes shared access and stable configuration/pin snapshots. Pin/unpin, cache reset and
resource-changing index operations take exclusive maintenance access. ANN operations remain
conservative maintenance users. Once maintenance is waiting, new query/planning admissions stop;
existing queries retire before maintenance begins. Maintenance never waits while retaining an
execution permit that it needs to drain. Internal pin queries reuse the outer maintenance ownership.

Operator/expression/compression SET values are connection-local overrides of YAML defaults.
Execution snapshots operator options once before final physical planning. RESET restores the
registered default. GLOBAL writes to query settings are rejected. Hardware decompression is
startup-only; shared logging updates serialize separately.

## Lifetime, errors and memory progress

See [Query lifetime and retirement](query-lifecycle.md) for the publication/lease protocol.
A submission guard covers permission through queue insertion. A work lease follows queued,
in-hand and running work through callbacks and CUDA completion. Spill workers borrow the actual
victim query; cancellation does not clear another query's repositories or downgrade requests.

Ordinary operator errors, cancellation, initialization failures and exhausted recoverable OOM
retries retire only the affected query. Queued requests poll interruption every 20 ms. Executing
queries poll completion/interruption/health every 25 ms. Already-submitted kernels must finish
before their buffers can be released; cancellation does not preempt them.

Reservations use nonblocking attempts. Waiting tasks remain visible to scheduling/spilling and
retry after a short delay. A reservation that makes no progress for 30 seconds fails its query.
Operator OOM/batch-contention retries are separately bounded. Sort caps derive from configured
GPU capacity divided by the admission limit, avoiding the assumption that current free memory
belongs to one query. HOST result transfer must obtain a real reservation.

Illegal device access, device assertion, launch failure/timeout, uncorrectable ECC and destroyed
CUDA contexts latch shared GPU-runtime unavailability. Existing owners stop/retire and later GPU
admission is refused. CPU execution can continue where normal fallback rules permit. A failed
mandatory cleanup likewise latches unavailability and retains possibly borrowed resources.
Clearing a sticky CUDA error does not restore health; restart is required. Physical device-fault
recovery is not established by ordinary error-injection tests.

## Pins, cache and prefetch

Pin publication installs a complete generation: data, placement, compression description, MVCC
facts, uniqueness proofs and late-materialization handle. Re-pin builds privately; failure leaves
the previous entry usable. Readers hold entry snapshots and query-local visibility masks. SQL pin
replacement and cache reset remain exclusive maintenance operations.

Metadata producers and blocking scan coalescers have separate pools, so coalescers cannot occupy
all capacity needed to produce their inputs. I/O readahead shares runtime backend budgets across
queries. Demand reads borrow immediately and block further speculation until their debt is repaid.
Canceling one subscriber releases its interest without invalidating another subscriber's bytes.
Cache pressure permits a bounded eviction attempt before abandoning speculative work.

GPU memory prefetch selects from the query's admitted GPU set, checks each device's headroom and
uses real reservations. Runtime-owned exclusive stream leases cover prefetch, pin materialization
and dynamic-filter publication/replication. Idle prefetch workers hold no stream. Batch locks and
conversion fences publish residency before task creation uses it for device affinity.

Queries may share GPUs and span their configured GPU prefix. Dedicated/disjoint device allocation,
equal-share scheduling and multi-GPU late materialization are separate optimizations. Actual
multi-GPU concurrency, peer-copy fallback and spill qualification must run on a multi-GPU host.
The implementation journal records which hardware and tests were available for this branch.

## Diagnostics

C++ runtime diagnostics are available without enabling telemetry:

- `SiriusContext::admission_counts()` reports queued/admitted queries, planning and maintenance
  occupancy, completion count and the most recent queue/admission/completion timestamps.
- `get_query_lifecycle_registry().diagnostics()` returns live query IDs with submission/work and
  memory-wait counts, admission/retirement/first-memory-wait timestamps and first-error text. It
  also retains the last 128 retired records, with completion timestamps and no resource owners.
- `completed_count()` is the cumulative retired-owner count, independent of bounded history.
- Window logs include instance, connection, window and query identity. Batch telemetry retires
  placements and port mappings by query ID; `records_for_query()` exposes their current counts.

Timestamps use `steady_clock` for interval measurement. These are diagnostic snapshots, not
permission to publish work or release resources.
