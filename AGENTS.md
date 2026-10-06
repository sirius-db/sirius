# AGENTS.md

Sirius is a GPU-native SQL engine that runs as a DuckDB extension, routing supported SQL
operations to the GPU (via cuDF/RMM/cuCascade) and falling back to DuckDB's CPU execution
otherwise. Once the extension is loaded it **transparently intercepts** normal SQL and runs it
on the GPU — no special syntax needed.

## Contributions & PRs

**The default branch is `main`** — branch and open PRs against it. If your local clone or a
stack still references `dev`, see `CONTRIBUTING.md`'s "Migrating your local clone" section.

Before opening a PR, read `CONTRIBUTING.md`'s "PR branching strategy" section to determine which
of the three approved paths applies — most work is **Self-contained** (push to a personal fork,
not `origin`); dependent changes use **Stacked PRs**; CI/critical changes that need same-repo
write permissions are the third, narrower exception.

**Doing a chain of dependent changes (Stacked PRs)?** Read `CONTRIBUTING.md`'s "Stacked PRs" and
"Merging a stack" sections first. One thing worth knowing without opening that doc: never use
"Enqueue stack" or `gh stack merge`, they don't reliably work with this repo's merge queue;
stacks merge bottom-up instead.

Before marking **any** PR ready for review (self-contained or stacked), check it against
`CONTRIBUTING.md`'s "PR reviewability" checklist. Open as a Draft until it meets that bar, then
convert to "Ready for review"; converting too early pings a reviewer before there's anything to
review, which is the exact problem that checklist exists to prevent.

## Build & test

Run commands through `pixi run <cmd>` (don't drop into the interactive `pixi shell`) so each
command runs in the activated environment:

```bash
pixi run make                              # library, C++ tests, and DuckDB extension (uses all cores)
pixi run make clean                        # wipe the library and wrapper build dirs

pixi run make test                         # build + run the C++ unit tests (scripts/run_unit_tests.py); make test_debug for debug

pixi run pre-commit run -a                 # all formatting/lint hooks
```

Running tests directly (non-obvious invocations):
```bash
pixi run build/release/test/cpp/sirius_unittest "[path_utils]"  # by Catch2 tag/test name
```

`scripts/run_unit_tests.py` runs the unit tests in parallel shards (2 per GPU), then the
`[multi_gpu]` and late-materialization tests. Arguments after `--` go to every Catch2 process.
Logs go to `build/release/test/cpp/log/<process>/`.

**Python API**: the default Pixi environment includes DuckDB's Python package. Load the built
Sirius extension from Python as shown in [docs/README.md](docs/README.md#python-api).

**Worktrees**: submodules are not auto-initialized — after creating one, run
`git submodule update --init --recursive`.

## Architecture

**Super Sirius** is the live engine: namespace `sirius`, source under `src/op/` (operators),
`src/planner/` (plan builders + `sirius_physical_plan_generator.cpp`), `src/pipeline/`,
`src/cuda/` (GPU kernels). **Read `docs/super-sirius/` before modifying Super Sirius code** —
see its [README](docs/super-sirius/README.md) for reading order.

The I/O layer (`cucascade::io`: io_uring / REST / kvikIO backends, the pinned `fs_cache`,
`cucascade::io::datasource`) comes from the cuCascade submodule, linked as
`cuCascade::cucascade_io` (the `cudf::io::datasource` bridge is in `cuCascade::cucascade_cudf`); `src/io/` keeps only Sirius glue (`path_utils`, `ioctx_resolver`,
`parquet_helpers`, `s3/sirius_httpfs`). The `src/exec/` headers shared with cuCascade
(`semi_future`, `thread_pool`, ...) are aliases of `cucascade::exec`. The io internals are
documented in the doc comments of cuCascade's headers under `cucascade/include/cucascade/io/`.

All new work targets Super Sirius. Memory spilling / CPU fallback is handled by the downgrade executor
(`src/downgrade/`, `src/creator/`); see `docs/super-sirius/memory-management.md`.

Before implementing operators / memory / expression / I/O work, run `/module-context <task>` to
load accurate cudf/rmm/duckdb/cucascade API docs.

## Usage

Load the extension and run normal SQL — Sirius intercepts it transparently and runs supported
queries on the GPU (controlled by the `gpu_execution` setting, on by default):

```sql
LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension';
SELECT ...;                  -- transparently routed to the GPU
-- SET gpu_execution = false;  -- to disable interception
```
