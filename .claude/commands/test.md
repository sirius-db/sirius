---
description: Build Sirius with pixi and run the requested tests.
---

# Build and Run Tests

Use the test name, tag, or scope supplied by the user. If no test is specified,
select tests relevant to the request and changes, or ask if the intended scope
cannot be inferred. Run commands in the Pixi environment:

- All C++ unit tests: `pixi run make test` (builds first).
- A specific test: build with `pixi run make`, then run
  `pixi run build/release/test/cpp/sirius_unittest '<name-or-tag>'`.
  Set `SIRIUS_EXTENSION_PATH` to the built
  `sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension`
  to include extension-loading checks; report those skips when it is unset.

For an inferred normal-suite tag selection, exclude hidden tests with `~[.]`, as
in [test/README.md](../../test/README.md). Honor explicitly requested hidden tests,
tags, or names without adding that exclusion.

If a build runs out of memory, retry with lower parallelism, for example
`CMAKE_BUILD_PARALLEL_LEVEL=8 pixi run make`. Report failures and their likely
cause. Apply fixes and rerun relevant tests when the user's request includes fixes;
otherwise report the findings. Summarize the results and any tests not run.
