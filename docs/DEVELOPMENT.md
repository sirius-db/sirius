# Developing Sirius

Since SiriusDB is an extension to DuckDB, the CMake project source is actually the `duckdb` directory (a submodule of
this project), which then pulls in SiriusDB as an extension. We use

- symlinked sirius-specific `CMakePresets.json` (at `cmake/CMakePresets.json`) to version control the build config.
- [pixi](https://pixi.prefix.dev/) to manage build dependencies.

## Building Sirius

Clone the repository with all submodules:

```bash
git clone --recurse-submodules https://github.com/sirius-db/sirius.git
cd sirius
```

Build with Pixi (uses all available cores). The default target is `release` (GCC Release):

```bash
pixi run make                        # GCC Release (default)
pixi run make clang-relwithdebinfo   # Clang RelWithDebInfo
pixi run make clang-debug            # Clang Debug
```

If the build exhausts memory, reduce parallelism:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 pixi run make
```

Run the Sirius-linked DuckDB binary — the extension is statically built in and loads automatically:

```bash
./build/release/duckdb
```

Alternatively, load the extension into an existing DuckDB shell:

```sql
LOAD 'build/release/extension/sirius/sirius.duckdb_extension';
```

## Pre-commit

Sirius uses [pre-commit](https://pre-commit.com/) hooks to enforce formatting and linting. Install the hooks after cloning so every commit is checked automatically:

```bash
pixi run pre-commit install
```

To run all hooks manually across the whole tree:

```bash
pixi run pre-commit run -a
```

## Testing

Run the full C++ unit test suite (what CI runs):

```bash
pixi run make test
```

Run tests by Catch2 tag or name:

```bash
pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"
pixi run build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"
```

## Using CLion for development

CLion does not natively support pixi environments. One way to circumvent that is to create a custom
toolchain for sirius and load the pixi environment via an environment file. This allows a consistent use of CLion, both
when using natively on a system, or via the remote development workflow. To support this, we create a
`sirius_pixi_env_for_clion.sh` file inside the `build` directory (ignored by default) when the pixi environment is
activated.

### First time setup

- Run `pixi shell`. This will generate/update the `sirius_pixi_env_for_clion.sh`.
- Choose correct `CMakeLists.txt`:
    - Open the sirius directory in CLion. CLion will ask how load the project, as a `Makefile` project or a `CMake`
      project, choose `CMake`.
    - Choosing `CMake` project will fail with configure errors, because as detailed earlier, the actual CMake source of
      the project is the `duckdb` submodule dir, but CLion assumes the root `CMakeLists.txt` to be the source.
    - To fix, go to `Tools > CMake > Unload CMake Project`, then open any cpp file in the sirius src dir, a banner will
      showup: `This file does not belong to any project target` with a `Fix` button to the right.
    - Click `Fix` and select `Choose CMakeLists.txt`, and navigate to the `duckdb` subdir and choose its
      `CMakeLists.txt`.
- Add sirius-specific toolchain:
    - Open settings, under `Build, Execution, Deployment > Toolchains` click the `+` button to add a new toolchain. Name
      it something like `Sirius` for easy differentiation.
    - In the upper right corner there should be a link named `Add Environment`. Click it and choose `From File`.
    - Navigate to and select the `sirius_pixi_env_for_clion.sh` inside the build directory.
    - Verify the toolchain has properly inherited the pixi env. Open settings, under
      `Build, Execution, Deployment > Toolchains > Sirius`, the `C Compiler` and `C++ Compiler` should have detected
      the pixi env compilers which can be verified by hovering over the `Detected ...` box.
    - Goto `Build, Execution, Deployment > CMake`, this should have a set to presets already loaded. Duplicate and
      enable the ones you need and make them use the `Sirius` toolchain that has the pixi compilers. We need to
      duplicate the preset profiles to make the use our custom toolchain, as they use the `Default` toolchain by
      default.
- Click `Apply` to save the changes. CLion should now properly configure the project, allowing you to build and debug.

### Alternate setup using CLion

An alternate way to use CLion with the pixi-provided tools (compilers, CMake, ninja, etc.) is
to launch CLion from a shell where the pixi environment is already active, so
that the CLion process inherits the correct `PATH` and environment variables. This works best when one is only using
CLion natively on a machine directly, and not via remote connections.

#### Launching CLion within the pixi environment

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

In this case, the Default toolchain can be used as it -- verify it has properly inherited the pixi env: Open settings,
under `Build, Execution, Deployment > Toolchains > Default`, the `C Compiler` and `C++ Compiler` should have detected
the pixi env compilers which can be verified by hovering over the `Detected ...` box.

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
