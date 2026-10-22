---
name: sirius-udf
description: Rewrite deterministic scalar UDFs for Sirius using SQL macros, libcudf, or CUDA. Generate reference tests, an external native package when needed, GPU execution evidence, and benchmarks; use for AI-assisted UDF conversion rather than installing SaaS infrastructure.
license: CC-BY-4.0
---

# Sirius GPU UDF Conversion

Perform the conversion for the user, including generated code, registration, build,
validation, and loading instructions. A skill supplies the development workflow;
Sirius supplies execution. Do not assume a remote AI server or a SaaS control plane.

## Establish behavior and workspace

- Inspect the source and its dependencies. Prefer an executable original CPU oracle;
  otherwise obtain an explicit behavior contract and label that weaker reference.
- Keep customer source, adapters, generated business logic, data, logs, and reports
  outside the Sirius repository and CI. Never embed confidential identifiers or
  translated business rules in repository examples. Use independently authored
  synthetic cases when a core change needs a reproducer.
- Discover the pinned DuckDB/cuDF/CUDA versions, installed Sirius SDK, matching
  extension, Pixi commands, and available GPU. Read the repository instructions
  and expression documentation before changing engine code.
- Create an external project using `scripts/scaffold.py --target sql|cudf|cuda
  --output /absolute/new/directory`. The scaffold contains synthetic examples;
  replace them with the conversion and independent reference tests externally.

## Convert and prove

1. Write a passing reference suite before implementing the candidate. Cover NULLs,
   types, boundary values, invalid-input and error behavior, and relevant branches.
2. Choose SQL when supported expressions preserve semantics; otherwise compose
   libcudf operations, or use CUDA for missing operations or demonstrated benefit.
   Follow [SQL](references/sql.md), [cuDF](references/cudf.md), or
   [CUDA](references/cuda.md) for the selected target only.
3. Generate the rewrite, CPU fallback, registration, build files, and comparison
   cases. Do not make the user manually write native code. Table-reading,
   recursive, side-effecting, and aggregate functions need a different contract;
   explain the limitation and use relational SQL when appropriate.
   Update the external `project.json` reference basis and synthetic flag to
   describe the actual conversion; a passing scaffold is only a smoke test.
4. Build and test; repair the implementation while retaining the reference
   behavior. Never remove failing cases or change NULL/error semantics to pass.
   An intentional semantic change requires the user's decision.
5. Run a GPU-required comparison with `enable_duckdb_fallback=false`; record
   execution-counter deltas proving GPU execution and zero fallbacks. Missing
   GPU/build prerequisites mean GPU validation is unperformed, not passed.
6. After correctness passes, benchmark representative generated data. Report warm
   query wall time separately from compilation/setup and device-only timing.
   The scaffold runner measures wall time; use CUDA events or profiling for a
   device-only measurement rather than relabeling wall time.

When asked to review a conversion, independently check reference coverage,
semantic equivalence, and execution evidence. When asked to optimize, profile a
passing implementation, change one measured bottleneck, and rerun correctness
and timing. Keep regressions visible rather than treating a rewrite as a promised
speedup.

Return paths to the external artifact, exact build/load commands, reference basis,
tested domain, execution evidence, measurements, and real limitations. Native
functions compile as their own package against the installed SDK; adding a UDF
must not require rebuilding Sirius. Deploy/upload only through a provider's
existing authorized tooling when requested.

## Provenance

Workflow adapted for Sirius from NVIDIA's
[cuDF Spark UDF skills](https://github.com/NVIDIA/cudf-spark/tree/main/skills).
See [attribution and license](references/upstream.md). The supplied synthetic
code and scripts are independently written; no Spark Java/JNI templates are used.
