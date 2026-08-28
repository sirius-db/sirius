# CLAUDE.md

Sirius is a GPU-native SQL engine that runs as a DuckDB extension, routing supported SQL
operations to the GPU (via cuDF/RMM/cuCascade) and falling back to DuckDB's CPU execution
otherwise. Once the extension is loaded it **transparently intercepts** normal SQL and runs it
on the GPU — no special syntax needed.

## Contributions & PRs

**The default/main branch is `dev`** (not `main`/`master`) — branch and open PRs against it.
Before opening a PR, read `CONTRIBUTING.md`'s "PR branching strategy" section to determine which
of the three approved paths applies — most work is **Self-contained** (push to a personal fork,
not `origin`); dependent changes use **Stacked PRs**; CI/critical changes that need same-repo
write permissions are the third, narrower exception.

**Doing a chain of dependent changes (Stacked PRs)?** Read `CONTRIBUTING.md`'s "Stacked PRs" and
"Merging a stack" sections first. One thing worth knowing without opening that doc: never use
"Enqueue stack" or `gh stack merge`, they don't reliably work with this repo's merge queue;
stacks merge bottom-up instead.

## Build & test

Run commands through `pixi run <cmd>` (don't drop into the interactive `pixi shell`) so each
command runs in the activated environment:

```bash
pixi run make                              # full build (uses all cores)
pixi run make clean                        # wipe the build dir (after a failed build, before rebuilding)

pixi run make test                         # build + run the C++ unit tests (Catch2, what CI runs); make test_debug for debug

pixi run pre-commit run -a                 # all formatting/lint hooks
```

Running tests directly (non-obvious invocations):
```bash
pixi run build/release/test/unittest --test-dir . test/sql/tpch-sirius.test    # one SQLLogic file
pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"  # by Catch2 tag/test name
```

**Python API** (links against the repo's `duckdb/` submodule via `DUCKDB_SOURCE_PATH`):
```bash
pixi run -e duckdb-python build-duckdb-python
```

**Worktrees**: submodules are not auto-initialized — after creating one, run
`git submodule update --init --recursive`.

## Architecture

**Super Sirius** is the live engine: namespace `sirius`, source under `src/op/` (operators),
`src/planner/` (plan builders + `sirius_physical_plan_generator.cpp`), `src/pipeline/`,
`src/cuda/` (GPU kernels). **Read `docs/super-sirius/` before modifying Super Sirius code** —
see its [README](../docs/super-sirius/README.md) for reading order.

**Everything under `src/legacy/` is the dead `gpu_processing` path — do not modify it.** All new
work targets Super Sirius. Memory spilling / CPU fallback is handled by the downgrade executor
(`src/downgrade/`, `src/creator/`); see `docs/super-sirius/memory-management.md`.

Before implementing operators / memory / expression / I/O work, run `/module-context <task>` to
load accurate cudf/rmm/duckdb/cucascade API docs.

## Usage

Load the extension and run normal SQL — Sirius intercepts it transparently and runs supported
queries on the GPU (controlled by the `gpu_execution` setting, on by default):

```sql
LOAD 'build/release/extension/sirius/sirius.duckdb_extension';
SELECT ...;                  -- transparently routed to the GPU
-- SET gpu_execution = false;  -- to disable interception
```
