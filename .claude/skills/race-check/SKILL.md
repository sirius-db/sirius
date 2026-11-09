---
name: race-check
description: >
  Diagnose suspected Sirius data races, deadlocks, thread safety issues, or
  nondeterministic query and test results using CPU and GPU diagnostics.
argument-hint: [sql-query-or-test-name]
disable-model-invocation: true
---

# Race Condition Analyzer

Use the supplied query or test and input data. Ask only for missing information
needed to reproduce. Avoid rerunning mutating SQL against user data without
authorization. See `.claude/skills/_shared/build-and-query.md` for build modes,
query execution, and logging. Repeated runs can identify inconsistent results;
account for unspecified row order and other nondeterministic query semantics
before attributing differences to a race.

Choose diagnostics based on the evidence and user resource constraints. Sanitizer
builds and repeated runs can be expensive. Preserve existing user changes and
remove temporary diagnostic instrumentation introduced during the investigation.

## CPU races

ThreadSanitizer reports CPU data races and lock-order issues. TSan and ASan cannot
be combined; the `clang-tsan` preset is separate. TSan may produce false positives for GPU
memory operations. It can impose 5–15x slowdown.

```bash
pixi run make clang-tsan
```

Run the supplied test using `build/clang-tsan/test/cpp/sirius_unittest` and the
TSan options documented in `docs/super-sirius/debugging.md`. These commands do not
verify instrumentation of the SQL wrapper; report that limit if the issue cannot
be reproduced as a unit test. For each report, inspect both access locations, the
shared data, and the synchronization protecting it.

## GPU diagnostics

Build with debug symbols (`relwithdebinfo` or `clang-debug`). Compute Sanitizer
`memcheck` detects invalid GPU memory accesses; `racecheck` detects shared-memory
hazards. Choose the tool suited to the evidence, and run the supplied query or test:

```bash
pixi run compute-sanitizer --tool racecheck sirius-duckdb/build/<preset>/duckdb -unsigned -cmd "LOAD 'sirius-duckdb/build/<preset>/extension/sirius/sirius.duckdb_extension';" <database_path> <<'EOF'
<QUERY>
EOF
```

Use the actual wrapper directory; to use a non-default Sirius preset, install it
and rebuild the wrapper as described in the shared reference. For a unit test:

```bash
pixi run compute-sanitizer --tool racecheck build/clang-debug/test/cpp/sirius_unittest "<name-or-tag>"
```

Inspect reported kernel locations and relevant stream synchronization. Sirius uses
one CUDA stream per GPU thread. Other possible CPU hotspots include the GPU thread
pool and task queue (`src/pipeline/`), Data Repository (`src/data/`), memory
reservation, and task creation.

Explain the findings and potential fixes. Apply fixes and rerun the relevant
reproduction and diagnostics only when the request includes fixing the issue.
Report results and remaining uncertainty.

## Scope

Analyze active code in both `namespace sirius` and `namespace duckdb`; see shared build-and-query.md.
