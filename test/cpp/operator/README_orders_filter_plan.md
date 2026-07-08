# Tutorial: build & run the `orders` filter-plan test

A quick, hands-on walkthrough for building and running
[`test_orders_filter_plan.cpp`](test_orders_filter_plan.cpp) — the operator-level test that
hand-builds the physical plan for `SELECT * FROM orders WHERE amount > 100` (a `TABLE_SCAN`
feeding a `FILTER`) and runs it directly on the GPU, with no SQL parser and no scheduler.

- **File:** `test/cpp/operator/test_orders_filter_plan.cpp`
- **Catch2 tag:** `[orders_filter_plan]`

## Prerequisites

- The repo's **pixi** environment. Run every command through `pixi run <cmd>` so it executes in
  the activated toolchain (don't drop into `pixi shell`).
- A **CUDA GPU** — the test allocates a GPU memory space via `initialize_memory_manager()`.

## 1. Build

```bash
pixi run make        # builds the sirius extension + the sirius_unittest binary (uses all cores)
```

- The **first** build is slow; incremental rebuilds after editing a single test file are fast.
- If a build fails partway through, wipe and retry: `pixi run make clean && pixi run make`.

## 2. Run just this test

The C++ tests are all compiled into one binary. Filter it by this test's Catch2 tag:

```bash
pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[orders_filter_plan]"
```

Expected output:

```
All tests passed (6 assertions in 1 test case)
```

Useful Catch2 flags (append after the tag):

| Flag | Effect |
|------|--------|
| `-s` | show **s**uccessful assertions too (see every `REQUIRE`) |
| `-b` | **b**reak into the debugger on the first failure |
| `--list-tests` / `--list-tags` | list what's in the binary |

## 3. The edit → build → run loop

1. Edit `test/cpp/operator/test_orders_filter_plan.cpp`
2. `pixi run make`
3. `pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[orders_filter_plan]"`

## 4. How the test is wired into the build

There's no glob — every test `.cpp` is listed explicitly in the root
[`CMakeLists.txt`](../../../CMakeLists.txt) inside `set(TEST_SOURCES ...)`. This test was added
with one line next to its siblings:

```cmake
test/cpp/operator/test_orders_filter_plan.cpp
```

A pre-commit check (`scripts/check_orphan_tests.py`) fails if a `.cpp` under `test/cpp/` is not in
that list, so **add your new test there and re-run `pixi run make`**.

## Related commands

```bash
pixi run make test                 # build + run the FULL C++ unit suite (what CI runs)
pixi run make test_debug           # same, debug build (asserts, symbols)
```

## Where to look next

- The test itself is heavily commented and cites the reference docs it mirrors:
  [`docs/super-sirius/physical-plan-generation.md`](../../../docs/super-sirius/physical-plan-generation.md)
  (the `LOGICAL_GET → TABLE_SCAN`, `LOGICAL_FILTER → FILTER` mapping) and
  [`docs/super-sirius/pipeline-execution.md`](../../../docs/super-sirius/pipeline-execution.md)
  (each operator's `execute()` is called in sequence).
- Sibling operator tests using the same direct-`execute()` pattern:
  [`test_physical_filter.cpp`](test_physical_filter.cpp),
  [`test_physical_table_scan.cpp`](test_physical_table_scan.cpp).
