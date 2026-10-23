# libcudf target

Generate an external project:

```bash
python .agents/skills/sirius-udf/scripts/scaffold.py --target cudf --output /tmp/example-udf-cudf
```

Read the installed `sirius/udf.hpp` before editing. Register an ordinary DuckDB
`ScalarFunction` and its GPU binder with `sirius::register_gpu_scalar_function`.
The binder receives the resolved `BoundFunctionExpression`; validate supported
types and constant requirements there, and return `nullptr` to decline GPU
planning. Capture immutable host configuration in the returned
`gpu_scalar_function`; no DuckDB execution objects survive into GPU execution.

The callback receives borrowed GPU columns already broadcast to the batch size,
row count, stream, and RMM resource. Return an owning cuDF column of the declared
type and size. Never retain borrowed views after invocation. Use the provided
stream and memory resource for every operation and intermediate allocation;
avoid device-to-host transfers for computation. GPU code may execute
concurrently, including on different devices.

DuckDB `DEFAULT_NULL_HANDLING` means the runtime applies input NULL propagation.
For `SPECIAL_HANDLING`, the callback owns the complete NULL semantics. Use only
API-supported fixed signatures and deterministic callbacks; the initial
interface rejects custom CPU bind/resolve/serialization callbacks and preexisting
`function_info`. Confirm the current header if the SDK evolves.
Initial native signatures support Booleans, integers up to 64 bits, FLOAT/DOUBLE,
VARCHAR, DATE, and timestamps without time zones. Decimal, nested, and variadic
signatures are outside the initial interface.

Use the pinned libcudf headers and local `/module-context` documentation when
available to choose exact APIs. Do not copy Java cuDF API names from Spark.
The scaffold compiles a standalone plugin against `sirius::udf_sdk` and a generic
embedding host. A provider can call its exported registration hook from its
existing DuckDB extension load callback; consumers do not rebuild Sirius.
