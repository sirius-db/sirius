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

### Choosing self-contained vs. Stacked

Decide this before starting work, not partway through — it's possible to split already-started
work into a stack later, but retrofitting costs more than planning it upfront. Default to
self-contained. Choose a stack only when reviewability is genuinely better served by splitting
into ordered layers: each layer should stand on its own as something a reviewer can fully
understand and approve without reading ahead, even though later layers depend on earlier ones
merging first. Size alone isn't a reason to stack — a large self-contained PR is still
self-contained if it can be reviewed as one coherent unit. If unsure, ask the user before deciding
to stack: the setup has real, sticky side effects (see "Remote gotcha" below) and requires
maintainer push access (check `MAINTAINERS.md`), so it isn't something to default into
unilaterally. If a stack is chosen, aim for every layer to be reviewable on its own merits, not
just the stack as a whole. If a stack turns out to be the wrong call after starting, `gh stack
unstack` converts the PRs back to self-contained targeting `dev`; see `CONTRIBUTING.md`'s
"Managing a stack" section.

### Stacked PR tips

Use `stacked/`-prefixed branches with the `gh-stack` CLI extension (`gh stack add`,
`gh stack submit`) instead of a single branch off `dev`, and push `stacked/` branches to `origin`
— see `CONTRIBUTING.md`'s "Stacked PRs" section for the full convention (requires push access to
`sirius-db/sirius`).

**Remote gotcha**: setting up for Stacked PRs (per `CONTRIBUTING.md`) involves renaming a fork
clone's `origin` remote to the main `sirius-db/sirius` repo — that change is local and sticky, so
in a clone that's been set up this way, `origin` no longer points at your fork. Always run `git
remote -v` before pushing rather than assuming `origin` = your fork; a self-contained PR from
such a clone needs to push to the renamed fork remote instead.

**Merging a stack**: never use "Enqueue stack" (Web UI) or `gh stack merge` (CLI). Tested against
this repo's actual merge queue settings and confirmed neither reliably walks a whole stack
through; the bottom PR merges, then the rest stalls with a stale base and nothing continues
automatically. Always merge bottom-up instead: `gh stack bottom` to find the bottom PR, confirm
it's approved and passing, `gh pr merge <number> --auto` to merge just that one PR (ask before
running this, since it's a real merge), then `gh stack sync --prune` once it lands, then repeat
for the next layer. See `CONTRIBUTING.md`'s "Merging a stack" section for the full steps.

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
