# Contributing to SiriusDB

## Building in debug mode

Since SiriusDB is an extension to DuckDB, the CMake project source is actually the `duckdb` directory (a submodule of this project), which then pulls in SiriusDB as an extension, and because a `CMakePrests.json` must live along side the CMake source of the project (to gets its idiomatic benefits like IDE integrations), it is a bit of a hack (involving symlinks) to version control the equivalent of a presets file which uses `duckdb` as its source dir.

One alternative to this while still version controlling the build flags is the use of initial cache files, in combination with pixi tasks.

The initial cache files are under the `CMake` directory, for the different build types.

The available pixi tasks are:

```bash
$ pixi task list
Tasks that can run on this machine:
-----------------------------------
build-debug, build-release, build-relwithdebinfo, configure-debug, configure-release, configure-relwithdebinfo
Task                      Description
build-debug               Build in debug mode
build-release             Build in release mode
build-relwithdebinfo      Build in release mode with debug symbols
configure-debug           Configure a debug build with no optimizations
configure-release         Configure an optimized release build
configure-relwithdebinfo  Configure a release build with debug symbols
```
