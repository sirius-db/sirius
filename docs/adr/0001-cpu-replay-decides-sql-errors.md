# CPU Replay decides SQL errors

A SQL error that Sirius detects on the GPU (an overflow, a conversion out of range, an aggregate result out of range) is a hint, not a verdict: transparent execution always answers it with CPU Replay, and DuckDB's result or error is final. Only a user interrupt, and the existing conditions under which CPU Replay is unavailable (for example Sirius-owned S3 data or `enable_duckdb_fallback = false`), surface the GPU's error directly. No Sirius exception class and no DuckDB `ExceptionType` selects the replay decision.

We chose this because a GPU-detected error cannot in general prove that DuckDB would raise:

- The materialize and cuDF AST strategies compute every CASE branch over every row, and an NVRTC kernel compiled for a subexpression of such a branch does the same, so they detect overflow on rows that a Row-exact Guard excludes.
- DuckDB's own raising depends on where an expression sits, on run-time adaptation and on LIMIT early stop. Its `ExpressionExecutor` evaluates every AND and OR operand on every row outside a filter predicate or CASE WHEN condition, so `SELECT (x <> MIN) AND abs(x) > 0` raises. A filter predicate or CASE WHEN condition goes through `Select`, which runs a later operand only on rows the earlier ones leave undecided, so `WHERE (x <> MIN) AND abs(x) > 0` returns rows; `Select` also reorders those operands by their observed cost as it runs, so whether such a filter raises can change between chunks. A LIMIT can stop the query before it reaches the row that would raise.
- DuckDB's error type and message are the user-facing contract. A Sirius-raised error reports, for example, an out-of-range DATE to TIMESTAMP cast where DuckDB reports a Conversion Error.
- A check can be stricter than DuckDB: Sirius rejects a non-finite STDDEV_SAMP, which DuckDB returns as infinity.

## Considered Options

- **Surface proven-exact errors directly.** Only an error from a fused NVRTC kernel at the root of an expression would skip replay. Every throw site would then have to know where it sits in evaluation, and the result would still diverge where DuckDB's raising depends on run-time operand reordering or LIMIT early stop, and in the error type.
- **Surface errors of DuckDB's OUT_OF_RANGE type directly** (the first revision of the NVRTC evaluator). This turned guarded overflow into false errors under every strategy, stopped replaying the pre-existing STDDEV_SAMP check, and changed the DATE to TIMESTAMP error type.

## Consequences

- A query that really errors runs twice: once on the GPU, then on the CPU, which raises. Successful queries are unaffected; the error path is not a performance target.
- Sirius error messages appear only where CPU Replay is unavailable, so they need not match DuckDB's text.
- The error contract in [`docs/super-sirius/expression-executor.md`](../super-sirius/expression-executor.md#sql-errors) is stated relative to DuckDB, with CPU Replay as the recovery wherever Sirius detects an error that DuckDB does not raise.
