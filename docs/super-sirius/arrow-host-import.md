# Host Arrow Import

This is the Arrow-only component tracked in #2210 under #1839 and #2181. It does
not register a Paimon source or complete the other phase-5 deliveries. Its starting
point is the helper and tests in [PR #1965](https://github.com/sirius-db/sirius/pull/1965)
at `c58579d09d86c3fb890fccc0a18ebdc9a933e8d4`; fragment/FFI orchestration is not
copied here. The shared component can be consumed independently of that interface.

## API and Ownership

`inspect_arrow_host_batch` validates a complete struct batch against positional
declared types, computes checked windows/counts, and prepares CPU normalization
buffers with a required caller-supplied `std::pmr::memory_resource&`. It performs
no GPU work. There is no default-resource staging fallback on quota exhaustion.

`import_arrow_host_table` consumes the checked plan with an explicit stream and
device memory resource. The caller selects the device and ensures the stream and
resource belong to that device. Every device buffer request uses that resource;
the helper never switches the current global resource or chooses a reservation.
It synchronizes its stream before successful return, including before final table
construction. Recoverable post-enqueue failures drain the stream before propagating the
original exception.

The caller retains complete ArrowArray/ArrowSchema roots, their pointer arrays,
format strings and backing allocations from inspection through import completion.
The helper never invokes a producer release callback. Raw C pointers cannot prove
buffer capacities or UTF-8 value conformance: accessible, stable, valid allocations
and conforming UTF-8 data remain producer preconditions. The helper checks supported
structure/layout and decimal values, not arbitrary malformed pointers.

The plan owns its normalized bitmaps, boolean values, decimal carriers and string
offsets. Its PMR must outlive it. Plans are movable, not copyable; moving does not
silently select a default allocator. Importing a moved-from plan is refused before
GPU work, including for empty batches. A live plan may be reused or read concurrently
only while producer storage stays immutable, each import has its own valid stream,
and supplied resources support that usage.

If draining itself fails, `arrow_host_import_drain_error` retains the original
exception and CUDA status without allocating a diagnostic string. Its fixed
`what()` message cannot replace this terminal classification with a formatting
allocation failure. A failed synchronization does not prove source release
is safe. The caller must retain the plan/roots and use its terminal CUDA-error
handling until independent quiescence; it must not retry or CPU-replay that failure.
The helper does not reset a device or establish a new global recovery policy.
The returned table continues to require its device resource/admission backing and
stream to remain valid through device-buffer destruction, even after host roots
may be released.

## Input Contract

Supported scalar formats are signed/unsigned 8/16/32/64-bit integers, float32/64,
bit-packed booleans, UTF-8 with int32 offsets, date32, naive second/millisecond/
microsecond/nanosecond timestamps, and same-scale decimal128 input matching a
supported declared decimal precision (5 through 38). Decimal precision may not
exceed the target's; every selected non-null value is checked before narrowing.
Existing Sirius exclusions for smaller decimal declarations remain unchanged.
The CPU normalization/copy representation assumes the supported little-endian
CUDA hosts. It does not rescale decimals, round timestamps or coerce unknown types.

Root and child offsets are combined with checked bounds. Null counts describe the
selected window, not an unsliced child's entire extent. Selected string offsets
must be nonnegative and monotone, including null slots and the terminal offset;
string row count must leave room for its extra offset. Root null rows, dictionaries,
nested layouts, binary/large offsets, timezone-aware timestamps, decimal256 and
unknown formats are explicitly refused before device operations. Empty typed
batches remain typed. Positive-row zero-column batches require a physical carrier
and are refused until the entire downstream cardinality path is qualified.

Field IDs, name matching, row-kind/visibility semantics and residual filters belong
to the adapter. A positional type match is not proof of bound-field identity or
native merge correctness.

## Allocation Boundary

Pinned cuDF direct Arrow import and decimal-width cast paths do not consistently
route all work through the supplied MR and have problematic shifted bitmap
boundaries. This helper uses ordinary owning fixed-width/string column factories
and explicit copies instead, with CPU-normalized bits/offsets/carriers. It adds no
custom GPU kernels, GPU null counting or GPU decimal rescaling/casting.

`estimated_device_bytes()` estimates final carrier buffers; it is advisory rather
than a hard bound on allocator/library overhead. `host_staging_bytes()` describes
the normalized data buffers. The supplied PMR also accounts for the plan's column
container; diagnostics, factory metadata, provider buffers, CUDA driver staging
and allocator bookkeeping are not thereby charged or hard-bounded. Callers enforce
their actual admission policies through the PMR/MR and retain the associated owners.
These estimates are not a bound on native split planning or whole-process memory.

## Tests

The unit tests use real DuckDB Arrow exports plus isolated structural mutations.
They cover scalar values, slices/null masks, decimal constraints, invalid final
columns before device allocation, allocation refusal/cleanup, and PMR limits.
Run the focused suite after building through the project's Pixi environment:

```sh
pixi run build/release/test/cpp/sirius_unittest '[arrow_host_import]'
```

The helper tests do not qualify Paimon GPU execution. That requires the controlled
native bridge, preparation/admission, authenticated binding and end-to-end adapter
acceptance tracked in polardb/duckdb-paimon#139, #2211, #2212 and #2213.
