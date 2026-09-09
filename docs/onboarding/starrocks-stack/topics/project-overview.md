# How the project fits together

[Back to the guide](../README.md)

**Topic:** architecture. **Features:** planning, execution and exchange.
**Modules:** StarRocks frontend, Rust compute node, C++ Sirius engine.
This document describes the reviewed source at `7610840c`.

## What Sirius does

Sirius is a SQL engine that uses GPUs for supported operations. It is also a
DuckDB extension: an application can load the extension and send ordinary SQL
through DuckDB. The integration reviewed here adds another entry path.
StarRocks plans a distributed query and asks Sirius compute nodes to run its
parts.

A **compute node**, or CN, is a process that accepts and runs work. A
**fragment** is one planned piece of a larger query. It might scan files, compute
a partial result, or combine results received from other fragments. The existing
[fragment overview](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/docs/super-sirius/streaming-fragments.md#L3)
describes the execution boundary.

## Follow one query

Consider a query that adds sales amounts stored in several Parquet files.
Parquet is a column-oriented file format: values from the same column are stored
together. The exact distributed plan depends on the frontend, but this example
shows the responsibilities at each boundary.

1. The StarRocks **frontend**, or FE, plans the SQL. It decides which fragments
   should read files and which fragment should combine their results.
2. The FE sends each CN a planned fragment. The CN receives it through its
   [compute-node service](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/compute_node_service.rs).
3. The Rust translator converts the StarRocks plan into **Substrait**, a common
   format for describing operations on tables. It preserves the meaning and
   positions of columns. It does not parse the original SQL again.
4. The C++ interface gives the Substrait plan to DuckDB. DuckDB resolves names
   and types and performs local planning. Sirius converts the result into GPU
   operators and work units called **pipelines**.
5. A sender fragment computes its result. When another CN needs that result,
   the sender packs it into temporary staging memory and transfers it.
6. The receiving CN copies the arriving data into its GPU pool. Once its
   required senders have finished, it runs the receiving fragment. The result
   path makes the final rows available to the FE.

The implementation boundaries are the
[translator](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/crates/starrocks-plan-translator/src/lib.rs),
[C++ interface](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp),
[physical planner](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/planner/sirius_physical_plan_generator.cpp),
and [result store](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/result_store.rs).

## The boundaries explain the modules

The translator must preserve query meaning. The **foreign-function interface**,
or FFI, must let Rust call C++ while keeping objects alive for as long as callers
need them. The scan module must return only the rows assigned to a fragment.
The CN must know when a fragment is ready and where its result belongs. The
transport must move the right bytes without releasing their memory too early.

These responsibilities meet at their interfaces. A wrong column order may look
like a transport problem even when the transfer is correct. A receiver that runs
out of memory may need a limit on incoming data rather than a larger staging
buffer.

## Streaming has a specific meaning here

The C++ engine has streams that let producers publish batches and consumers read
them. A **batch** is a group of rows handled together. The stream API does not
itself slow producers when queues grow; this is an explicit
[design boundary](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/docs/super-sirius/streaming-sessions.md#L356).

The distributed path also waits for completed inputs before running a receiver.
It can therefore retain a large amount of received data. Having streaming
operators does not mean the entire distributed query already executes as a
continuously flowing pipeline.

Continue with [C++ streaming](cpp-streaming.md),
[plan translation](plan-translation.md), and the [staging area](staging-area.md).
