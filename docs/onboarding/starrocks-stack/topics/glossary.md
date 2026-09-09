# Shared vocabulary

[Back to the guide](../README.md)

**Topic:** terms used in the onboarding documents. **Modules:** all layers.
These definitions describe their use in the reviewed Sirius integration.

| Term | Plain-English meaning |
| --- | --- |
| FE, frontend | The StarRocks component that plans SQL, schedules fragments and coordinates results. |
| CN, compute node | The Rust process that receives planned work and runs it through Sirius. |
| Fragment | One runnable piece of a distributed query, with declared inputs and an output boundary. |
| Operator | One operation in a plan, such as a scan, join, filter or aggregate. |
| Pipeline | A connected unit of engine work that the scheduler turns into executable tasks. |
| Batch | A group of rows handled or transferred together. |
| Stream | The engine interface through which one part publishes batches and another reads them. |
| Source / sink | The input and output ends of an operator pipeline. |
| EOF / EOS | End of input or end of stream. A receiver needs completion from every expected sender, not just one. |
| Exchange | A query-plan boundary where rows move from sender fragments to receiver fragments. |
| Shuffle | An exchange that distributes rows among destinations, often using a key. |
| Descriptor | Plan metadata describing columns, their types and how they are arranged. |
| Slot ID | A number the StarRocks plan uses to identify a column or expression result. |
| Column or wire order | The positions of values in the transferred row. Sender and receiver must agree on these positions. |
| Substrait | The serialized plan format used between the Rust translator and the C++/DuckDB planning path. |
| Binding | Resolving a plan's names and references to concrete types and inputs. |
| Transaction | The DuckDB scope required for relevant catalog and planning operations; the standalone FFI path must supply one when needed. |
| FFI | Foreign-function interface: the boundary through which Rust calls C++ code. |
| Context | The FFI owner of the embedded DuckDB connection and initialized engine resources. |
| Parked output | Completed fragment data retained until its consumers have received it. |
| Staging arena | A stable device allocation used temporarily for packed transfers, separate from the engine's normal GPU pool. |
| Lease | A temporary range within the staging arena. This implementation identifies it by its offset. |
| Inbound ticket | An ID for a copied received batch waiting to be moved into a receiver input. It no longer represents an arena lease. |
| RMM | The GPU allocation library used by the engine's ordinary memory resources. |
| cuDF | The GPU column/table library used for data operations and representations. |
| cuCascade | The reservation, data-representation and memory-tier machinery used by Sirius. |
| Spill / downgrade | Moving eligible data to a lower memory tier, such as GPU to host memory or host memory to disk. |
| Backpressure | A mechanism that makes a producer slow down when a consumer cannot keep up. The stream layer here has no queue-level mechanism of that kind. |
| Receive credits | A proposed way to limit how much received data may be outstanding. They are not implemented in the reviewed branch's receive path. |
| NIXL | The transfer layer used by the CN's remote GPU exchange implementation. |
| PRPC / BRPC | The RPC protocol and service machinery used for coordination messages; the bulk GPU WRITE is a separate operation. |
| Canary | A small transfer used during connection establishment to check the link before normal work. |
| MIG | GPU instances created by partitioning a physical GPU. Placement and memory limits must be checked for the actual engine and device setup. |
| Cardinality | An estimate or count of rows. An FE file estimate and a DuckDB stream estimate are separate planning inputs. |
| Partial aggregate | An intermediate result sent for later combination, such as the sum and count used to compute an average. |
| Oracle | The reference result used to judge another engine's answer. |
| Cold / warm run | A first-use run and a later repeat after initialization has occurred. |
| NaN | A floating-point value meaning "not a number." It must not silently pass as a finite expected result. |
| Draft PR | A proposed change that has not yet been marked ready for review. It is not an upstream landing. |

The definitions are grounded in [C++ streaming](cpp-streaming.md),
[Rust FFI and DuckDB](rust-ffi-duckdb.md), [plan translation](plan-translation.md),
[staging area](staging-area.md), [memory and MIG](memory-and-mig.md), and
[benchmarks and correctness](benchmarks-and-correctness.md). Those guides include
the source links and the limits that apply to each term.
