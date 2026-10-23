# Aggregate-to-SQL conversion

Use this workflow for a deterministic aggregate UDF whose behavior can be
expressed by existing DuckDB and Sirius SQL operators. Native cuDF/CUDA callbacks
remain scalar-only; no native UDAF registration API is introduced.

## Preserve the source state contract

Inspect initialization, per-row update, partial-state merge, and finalization.
Record accepted types, skipped rows, NULL/error behavior, empty-input output,
overflow, ordering sensitivity, and any allowed floating-point error. Use the
original implementation in its existing runtime as the reference. The supplied
`original.py` classes are independently synthetic examples, not a substitute for
customer source or proof of JVM/Spark compatibility.

Prefer an aggregate macro when expansion preserves invocation semantics. A
weighted mean must preserve which rows contribute to both numerator and
denominator, including zero-denominator behavior. Generate an entire query when
multiple stages or a relational transformation are needed; do not replace calls
through textual search/substitution. Built-in aggregate availability differs from
Spark. Inspect current Sirius aggregate planning and test every expanded operator
with fallback disabled rather than adding new engine operators to this workflow.

Do not claim conversion for stateful/external-side-effecting sources or ordering
contracts the rewrite cannot preserve. Keep the original function/deployment path
when equivalence or GPU support is unproven. A passing source-runtime adapter
proves only the source domain tested; partial-state merge correctness needs
independent cases with varied partitions and input order.

## Generated project and case suite

```bash
pixi run --locked -e cuda12 python .agents/skills/sirius-udf/scripts/scaffold.py --kind aggregate --target sql \
  --output /absolute/new/aggregate-project
```

Each external project contains `registration.sql`, `cases.json`, source originals,
a reference adapter, per-case SQL, `run.py`, and exact commands in `USAGE.md`.
The suite's version-1 structure is:

```json
{
  "version": 1,
  "oracle": {"argv": ["{python}", "oracle.py"], "timeout_seconds": 60},
  "cases": [{
    "id": "grouped_example",
    "setup": "cases/grouped/setup.sql",
    "input": "cases/grouped/input.sql",
    "candidate": "cases/grouped/candidate.sql",
    "parameters": {"source_specific": "opaque adapter parameters"},
    "gpu_required": true
  }]
}
```

Paths are relative to the external project. Setup SQL creates case-specific tables;
`@ROWS@` is replaced with the requested row count. Input SQL exports all source
rows in deterministic order. Candidate SQL emits matching result types and rows;
explicitly order grouped results, including NULL placement. Each case uses an
isolated database and original-runtime call. Empty/optimizer-folded cases may set
`gpu_required` false, but the suite must also contain required nonempty GPU cases.
No native plugin is needed for these SQL rewrites.

The runner invokes the adapter as an argument array without a shell, substituting
`{python}` with the runner's interpreter. Other runtimes can use their own explicit
argv (for example a provider-installed JVM adapter); Sirius gains no runtime
dependency. The adapter receives one request on stdin:

```json
{"version": 1, "case_id": "grouped_example",
 "input": {"types": ["BIGINT", "DOUBLE"],
           "rows": [["7", {"ieee_bits": "4000000000000000"}]]},
 "parameters": {"source_specific": "opaque adapter parameters"}}
```

It returns exactly one JSON object on stdout:

```json
{"version": 1, "result": {"types": ["BIGINT", "DOUBLE"],
 "rows": [["7", {"ieee_bits": "4010000000000000"}]]}}
```

NULL is JSON `null`; other scalar values use the existing host's canonical string
encoding. FLOAT/DOUBLE use network-order IEEE-754 hexadecimal `ieee_bits` to
preserve signed zeros and payloads. Adapters should send diagnostics to stderr.
A source error returns `{"version":1,"error":{"message":"..."}}`; failed,
malformed, timed-out, or mismatched references prevent successful validation and
benchmarking. Source errors establish a limit of the converted domain rather than
a passing GPU conversion.

Exact types, shape, NULLs, and values are required by default. A source-permitted
floating-point tolerance can be specified on a case as
`"comparison":{"float_tolerances":{"2":{"atol":1e-12,"rtol":1e-12}}}`.
Column indexes are zero-based. Record each tolerance and its justification; do
not infer tolerance from a failed test. Do not apply it to NULLs, logical types,
non-floating values, NaNs, or infinities. Exact mode retains IEEE-bit comparisons.

## Proof and measurement

Cover global/grouped aggregation; NULL/multiple keys; empty, singleton, all-NULL,
and duplicate rows; denominator/overflow boundaries; skew; shuffled order; and
groups spanning batches. Test source partial-state merges independently. Validate
source versus CPU SQL versus GPU SQL before benchmarking. Include deliberately
wrong finalization and NULL-handling rewrites to prove the gate rejects them.

Use `run.py --validate-only` for correctness without timing. Required nonempty
cases must show positive GPU execution counters and zero plan/runtime fallbacks.
Report optimizer-folded cases separately. Rerun scalar SQL/cuDF/CUDA tests after
changing shared tooling. Independently exercise the updated skill on fresh
conversion requests with held-out cases; scaffold smoke tests alone do not test
AI conversion.

For stage benchmarking use the repository's existing operator profiler: global
and 1,024-group uniform/skewed workloads, 100K and 1M rows, three warmups and seven
measured repetitions. Defaults are one CPU thread, one GPU executor, 8 MiB batches,
512 MiB tracked GPU budget, and no GPU table pinning. The original UDAF gates
correctness; equivalent DuckDB CPU SQL is the performance baseline. Preserve raw
samples, generated SQL, configuration, versions, counters, and pressure evidence.
Report operator work and regressions; exclude ratios affected by reservation
pressure, absent evidence, or insufficient timer precision. These sums may overlap
and do not measure query-latency speedup or kernel-only time.
