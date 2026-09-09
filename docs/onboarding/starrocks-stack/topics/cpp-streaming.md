# C++ streaming primitives

**Module / language:** Super Sirius execution engine, C++ and CUDA
**Scope:** This guide describes the reviewed `7610840c` snapshot. It is not a
description of later development work or a claim that the code has been run.

[Back to the guide](../README.md)

## What this part of the engine is for

A distributed query is broken into **fragments**: independent pieces of a plan
that can run separately. One fragment may produce rows that another fragment
needs. The C++ streaming primitives provide the in-process hand-off for those
rows. They keep rows as native GPU batches rather than converting them to files
or an Arrow result first.

The central type is `batch_stream`. It wraps a cuCascade repository, which is
the queue that holds batches. A producer pushes a batch; a consumer pulls one.
The wrapper adds the rules that a plain queue does not express: which producers
are expected, when the stream is finished, how a failure reaches the consumer,
and how a waiting GPU pipeline becomes runnable again. The public contract and
three observable states—data available, waiting, and end of stream—are defined
in [the header](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/include/exec/batch_stream.hpp#L31).

## How a stream ends and fails

Each input stream has an expected set of sender IDs. "End of stream" (EOS)
means that every expected sender has closed *and* no queued batch remains. A
set matters here: sender 0 closing twice cannot accidentally stand in for
senders 0 and 1. The implementation records the IDs in a set and marks the
stream terminal only after all expected IDs have appeared
([`close`](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/exec/batch_stream.cpp#L55)).

For example, imagine a join receiver that expects partitions from workers 10
and 11. If worker 10 sends its final batch and closes, the receiver must keep
waiting for 11. Once 11 closes, the receiver still consumes any already queued
batches before it sees EOS. If either worker reports an error, the stream is
**poisoned**: the first error is retained and later `try_pull()` calls rethrow
it. The failure also wakes waiters, so a peer does not wait forever
([`fail` and `try_pull`](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/exec/batch_stream.cpp#L78)).

The scheduler hook is intentionally separate from `wait()`. A source with no
batch does not block a GPU worker. It reports that it is waiting; a successful
push or failure calls its data hook after releasing the stream lock. That hook
can nominate the source for work again. The hook fires for every push, rather
than only once, so data cannot be missed between a notification and re-arming
the source ([hook contract](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/include/exec/batch_stream.hpp#L93)).

## Building a fragment around streams

`streaming_fragment` turns declarations into a runnable fragment. Before
planning, it registers each input stream's columns, expected senders,
repository, and optional row estimate. It then lowers the plan, adds a
`STREAMING_SINK` at the output, and connects each built source to the session
([build sequence](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/exec/streaming_fragment.cpp#L137)).

The declarations have a real lifetime rule: repositories must outlive the
engine that owns the plan, and the session borrowing operators is destroyed
first. The member order documents and enforces that ordering
([lifetime comment](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/include/exec/streaming_fragment.hpp#L111)).
If execution throws, `run()` marks every output stream failed before it lets the
error escape. That ensures downstream fragments learn the cause instead of
remaining parked ([failure propagation](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/exec/streaming_fragment.cpp#L201)).

## Features represented in the original PR inventory

The original draft inventory listed #1694 as **stream input cardinality**. It
lets a caller give DuckDB an estimated row count for a stream source. Without
one, DuckDB retains its old one-row assumption; with one, it can make a better
join-order decision. The callback deliberately returns no statistic whenever
the declaration is unavailable, rather than throwing during planning
([cardinality callback](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/exec/stream_plan_bindings.cpp#L68)).

The broader streaming foundations (#1094, #1320, #1479, #1480, and #1481) were
already merged upstream at the time captured by the onboarding research. They
are foundations, not evidence that every later exchange feature had landed in
`dev`.

## Detecting stalled work

Draft [#1699](https://github.com/sirius-db/sirius/pull/1699) adds an optional
watchdog: a check for queries that stop making scheduling progress.
`SIRIUS_QUERY_WATCHDOG_SECS` sets the allowed interval with no progress; the
feature is off by default. This is a scheduling-stall check, not a general
wall-clock deadline for every long-running query. The check is in
[sirius_engine.cpp](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_engine.cpp#L132).

A separate branch commit detects out-of-memory retries that cannot make useful
progress. It is proposed as [P07](../pr-packages/p07.md). Repeating an allocation
request does not help if no eligible data can leave memory, so the intended
behavior is to fail with the cause instead of repeatedly rescheduling the same
work. Its tests still need to pass on the actual extracted PR.

## Boundaries to keep in mind

This stream is not a network protocol and it has no channel-level byte-credit
or backpressure mechanism. Memory pressure is handled by cuCascade's memory
tiers and spilling, plus admission control elsewhere. The staging arena used by
remote transport is also not this queue: it is a short-lived, registered device
region. Read [Rust FFI and DuckDB lowering](rust-ffi-duckdb.md) for the API
boundary and staging hand-off, and [Parquet scans and the pin cache](parquet-scans-and-cache.md)
for the file-input path.

For a code-first reading path, start with `batch_stream.hpp`, then
`batch_stream.cpp`, then `streaming_fragment.cpp`. The full pressure contract
is in [streaming-sessions.md](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/docs/super-sirius/streaming-sessions.md#L356).
