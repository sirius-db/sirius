# Rust FFI and DuckDB lowering

**Module / language:** Rust compute-node-facing API, C++ FFI bridge, DuckDB planner
**Scope:** This guide describes the reviewed `7610840c` snapshot. It is not a
description of later development work or a claim that the code has been run.

[Back to the guide](../README.md)

## The job of this layer

The StarRocks-facing code is Rust, while the GPU query engine and DuckDB
integration are C++. **FFI** (foreign-function interface) is the bridge that
lets Rust use the C++ engine through a Rust API with explicit lifetime rules. The key Rust
objects are `SiriusContext`, which owns an engine context, and `Fragment<'ctx>`,
which represents one planned piece of a distributed query.

The lifetime marker in `Fragment<'ctx>` means the Rust compiler will not let a
fragment outlive the context that created it. A caller declares inputs and
outputs, builds a plan, supplies any input batches, runs the fragment, and then
drains its output. That order is documented by the wrapper
([fragment overview](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/rust/crates/sirius/src/lib.rs#L201)).

An input is named `sirius_stream_<id>`. The Rust helper and C++ implementation
share this one naming convention, so a translated plan can read a stream as
though it were a table ([name helper](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/rust/crates/sirius/src/lib.rs#L190)).
At DuckDB bind time, the table function gets that stream's declared column names
and types. It is only a marker: the physical Sirius planner replaces it with a
streaming source, and DuckDB must never execute the function itself
([bind implementation](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/exec/stream_plan_bindings.cpp#L32)).

## Planning and the two FFI drafts

The original draft inventory described #1697 as **FFI transaction scope** and
#1702 as **Rust Fragment bindings**. They solve adjacent problems. #1697 makes
`lower_substrait` obtain a DuckDB transaction if the caller has not already
opened one. A transaction is DuckDB's scoped view of catalog changes; it is
needed while the code looks up objects and optimizes the Substrait plan
([lowering transaction](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L91)).

#1702 exposes the C++ `Fragment` through a `cxx` bridge and gives Rust callers
the `Fragment<'ctx>` wrapper. The wrapper makes important declarations explicit:
input columns, expected senders and output routing. Later branch work in
[P02](../pr-packages/p02.md) adds packed transfer APIs and input row-count
declarations; those additions are not part of the original #1702 change.
An optional cardinality is particularly useful because a stream has no file for
DuckDB to inspect. It is an estimate of the total input rows, not a promise that
the transport has delivered every row already
([`declare_input_cardinality`](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/rust/crates/sirius/src/lib.rs#L249)).

For a concrete local hand-off, a producer fragment can park native GPU batches
on output stream 7. After it has run, a receiver can call `relay_from` to move
those batches into its own declared input stream and close that sender. The API
rejects an early relay because an empty output before completion would otherwise
look like a finished stream and silently truncate the result
([relay safeguards](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/rust/crates/sirius/src/lib.rs#L304)).

## The staging hand-off

Remote exchange needs a stable device address that a transport can register.
The C++ staging arena provides it. It is a plain `cudaMalloc` region outside the
RMM pool (RMM is the GPU allocation library used for that pool), divided into explicitly released
leases. The arena's address-ordered free list coalesces neighboring free blocks,
so capacity measures current live transfers rather than all transfers ever made
([arena design](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/include/exec/exchange_staging_arena.hpp#L26)).

`export_packed` takes one parked output batch, packs it into an arena lease, and
returns host metadata plus an offset, length, and row count. A zero-row batch is
metadata-only and has no lease. Packing reserves payload bytes plus an 8 MiB
chunk of slack; its error path releases a lease it has just acquired
([packing path](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L863)).

On receipt, `push_packed` validates the declared schema, unpacks the bytes, and
deep-copies the table into ordinary pool memory. The caller can then release the
arena lease immediately. This distinction matters: staging holds in-flight wire
frames; the stream queue holds batches waiting to be consumed
([copy-out-on-arrival](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L949)).

`InboundStore` is a later, thread-safe receive-side convenience API: it stages a
frame into pool memory under a ticket, then `push_inbound` moves that ticket into
a fragment without another copy. It can outlive `SiriusContext` at the Rust type
level. The reviewed snapshot documents that late calls should return an error,
but the review found a separate teardown risk: an in-progress `stage()` may use
a raw GPU-memory-space pointer after concurrent context destruction. Treat that
as an unresolved review finding, not as a property of the core `Fragment<'ctx>`
lifetime guarantee ([InboundStore contract](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/rust/crates/sirius/src/lib.rs#L529)).

## Where to continue

Read the C++ bridge in `src/sirius_ffi.cpp`, then the Rust wrapper in
`rust/crates/sirius/src/lib.rs`, and then the C++ stream primitives in
[C++ streaming primitives](cpp-streaming.md). For file reads before a fragment
starts, continue with [Parquet scans and the pin cache](parquet-scans-and-cache.md).
