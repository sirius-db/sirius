# CLion: indexing Sirius via a compilation database

This tutorial sets up CLion so it fully indexes the Sirius C++/CUDA sources —
code completion, navigation, find-usages — **without** CLion needing to run
CMake itself. It also fixes the
`This file does not belong to any project target` banner on `src/` and
`test/cpp/` files.

If you instead want to **build and debug from inside CLion**, use the CMake
toolchain workflow in [DEVELOPMENT.md](DEVELOPMENT.md#using-clion-for-development)
— it is more setup, but gives you run/debug configurations. The two approaches
fix the banner equally; this one is the fast path for reading and editing code.

## Why CLion fails out of the box

Opening the repo as a CMake project fails for two reasons:

1. **The repo-root `CMakeLists.txt` is not the CMake entry point.** Sirius
   builds as a DuckDB extension: the real CMake source directory is the
   `duckdb/` submodule, which pulls the root `CMakeLists.txt` in via
   `DUCKDB_EXTENSION_CONFIGS`. CLion assumes the root file is the entry point
   and the configure step errors out.
2. **The compilers and dependencies live in the pixi environment**
   (`.pixi/envs/default/`), which CLion's default toolchain knows nothing
   about.

With zero CMake targets loaded, *every* file shows the
"does not belong to any project target" banner.

The fix: let the pixi build produce `compile_commands.json` (it already does —
`CMAKE_EXPORT_COMPILE_COMMANDS` is on in every preset) and open **that** in
CLion as a [Compilation Database project](https://www.jetbrains.com/help/clion/compilation-database.html).
Every translation unit is indexed with the exact flags and include paths the
real build uses, pixi paths included.

## Prerequisites

A configured release build, so `build/release/compile_commands.json` exists:

```bash
pixi run make
```

(Only the CMake *configure* step generates the database, so if a full build
fails partway you likely still have a usable `compile_commands.json`.)

## Setup

### Step 1 — close the project in CLion

**File → Close Project**, back to the Welcome screen.

This matters: a running CLion keeps its project model in memory and flushes it
back into `.idea/` — any edits made underneath it are silently overwritten.
On remote development, closing the project also stops the backend; give it a
few seconds.

### Step 2 — run the switch script

```bash
bash scripts/clion_use_compiledb.sh
```

The script refuses to run while CLion is still up. It does two things:

- symlinks `compile_commands.json` at the repo root →
  `build/release/compile_commands.json` (gitignored; because it is a symlink,
  every rebuild refreshes it automatically), and
- strips the `CMake*` project-model components from `.idea/` so CLion stops
  loading the project as CMake.

### Step 3 — reopen as a Compilation Database project

On the Welcome screen: **Open** → select the repo's `compile_commands.json`
(the *file*, not the folder) → **Open as Project**.

CLion loads it as a Compilation Database project and starts indexing. Symbol
navigation and completion work across `src/`, `test/cpp/`, and the `duckdb/`
submodule sources.

## Keeping the index fresh

- **Rebuilds**: nothing to do — the root symlink always points at the latest
  database.
- **New source files** (added to `CMakeLists.txt`): the database is regenerated
  at CMake configure time, which the next `pixi run make` triggers. If CLion
  does not pick the change up, use **Tools → Compilation Database → Reload
  Compilation Database Project** (or close/reopen the project).

## Troubleshooting

**CLion still opens the repo as a CMake project.**
Close it, move the IDE state aside (`mv .idea .idea.bak`), and reopen via
`compile_commands.json`. You lose only local editor state, not code.

**The script says CLion is still running.**
The backend takes a few seconds to exit after File → Close Project. Wait and
retry; if it persists, check `pgrep -af clion`.

**Some `test/cpp` files still show a banner.**
A compilation database can only index files that something compiles. A handful
of `test/cpp` files are excluded from the `sirius_unittest` target — see the
commented-out entries in `TEST_SOURCES` in the root `CMakeLists.txt`, each
annotated with the reason (references removed APIs, fails at runtime, hangs,
or legacy-only). No IDE setting can index those; they need to be repaired and
re-added to `TEST_SOURCES` first.

**Red squiggles inside CUDA kernel launches.**
`.cu` files are indexed with their real build flags, but CLion's engine does
not fully parse the `<<<grid, block>>>` launch syntax. Navigation and
completion still work; the squiggles on launch lines are cosmetic.

**Switching back to the full CMake workflow.**
Follow [DEVELOPMENT.md](DEVELOPMENT.md#using-clion-for-development). Delete
the root `compile_commands.json` symlink if you want CLion to stop offering
the compilation-database project type.
