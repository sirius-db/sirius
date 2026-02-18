# Contributing to SiriusDB

## Building in debug mode (using clang)

Since SiriusDB is an extension to DuckDB, the CMake project source is actually the `duckdb` directory (a submodule of
this project), which then pulls in SiriusDB as an extension, and because a `CMakePrests.json` must live alongside the
CMake source of the project (to gets its idiomatic benefits like IDE integrations), it is a bit of a hack (involving
symlinks) to version control the equivalent of a presets file which uses `duckdb` as its source dir.

One alternative to this while still version controlling the build flags is the use of initial cache files, in
combination with pixi tasks.

The initial cache files are under the `CMake` directory, for the different build types.

The available pixi tasks, which make use of these initial cache files, can be listed using:

```bash
$ pixi task list
Tasks that can run on this machine:
-----------------------------------
build-debug, build-release, build-relwithdebinfo, configure-debug, configure-release, configure-relwithdebinfo, sql-logic-tests-release
Task                      Description
build-debug               Build in debug mode
build-release             Build in release mode
build-relwithdebinfo      Build in release mode with debug symbols
configure-debug           Configure a debug build with no optimizations
configure-release         Configure an optimized release build
configure-relwithdebinfo  Configure a release build with debug symbols
sql-logic-tests-release   Run SQL logic tests for a release build
```

## Using CLion for development

CLion does not natively support pixi environments. The most reliable way to
use CLion with the pixi-provided toolchain (compilers, CMake, ninja, etc.) is
to launch CLion from a shell where the pixi environment is already active, so
that the CLion process inherits the correct `PATH` and environment variables.

### Launching CLion with the pixi environment

Quit all existing CLion instances (more on that below), then inside the
sirius root dir

```bash
pixi shell
/path/to/clion.sh .
```

Where `/path/to/clion.sh` is the **real** CLion launcher, not the JetBrains
Toolbox wrapper (again, more info below). For a typical Toolbox install
on Linux this is along the lines of:

```
~/.local/share/JetBrains/Toolbox/apps/clion/ch-0/<version>/bin/clion.sh
```

You can add a shell function to your `~/.bashrc` (or equivalent) for
convenience:

```bash
real_clion() {
    /path/to/clion.sh "$@"
}
```

Then the workflow simply becomes:

```bash
pixi shell
real_clion .
```

### Gotchas

1. **Toolbox wrapper does not pass your environment.**
   If CLion was installed via JetBrains Toolbox, the `clion` command on your
   `PATH` is a small wrapper script that talks to the Toolbox App over IPC.
   The Toolbox App then spawns CLion as its own child, so the environment from
   your shell is **not** inherited. Always use the real `clion.sh` launcher
   instead.

2. **An already-running CLion instance will absorb new projects.**
   When `clion.sh` detects a running CLion instance, it hands the "open
   project" request to that existing process (via IPC) and exits. The project
   opens in a new window, but it runs under the environment of the **original**
   CLion process. If you need the pixi environment, **quit CLion entirely**
   before relaunching from `pixi shell`.

3. **The direnv plugin is unreliable.**
   The third-party direnv plugins for JetBrains IDEs
   (`intellij-direnv`, `Better Direnv`) have open compatibility issues with
   recent IDE versions (2024.2+) and appear to be infrequently maintained.
   Launching CLion from an activated pixi shell is more dependable than relying
   on a direnv plugin.

### First time setup

- Choose correct `CMakeLists.txt`:
    - CLion will ask how load the project, as a `Makefile` project or a `CMake`project, choose `CMake`.
    - Choosing `CMake` project will fail with configure errors, because as detailed earlier, the actual CMake source of
      the project is the `duckdb` submodule dir, but CLion assumes the root `CMakeLists.txt` to be the source.
    - To fix, go to `Tools > CMake > Unload CMake Project`, then open any cpp file in the sirius src dir, a banner will
      showup: `This file does not belong to any project target` with a `Fix` button to the right.
    - Click `Fix` and select `Choose CMakeLists.txt`, and navigate to the `duckdb` subdir and choose its
      `CMakeLists.txt`.
- Add CMake configure presets:
    - Verify the toolchain has properly inherited the pixi env. Open settings, under
      `Build, Execution, Deployment > Toolchains > Default`, the `C Compiler` and `C++ Compiler` should have detected
      the pixi env compilers which can be verified by hovering over the `Detected ...` box.
    - Goto `Build, Execution, Deployment > CMake` and add a debug profile that uses the `Default` toolchain that has the
      pixi compilers. Under CMake Options field add `-C /home/dvats/repos/sirius/CMake/debug.cmake`. This initial cache
      file will load all the necessary cache variables to properly configure a debug build.
    - Similarly, a Release profile may be created, that uses the release initial cache file with
      `-C /home/dvats/repos/sirius/CMake/release.cmake`
