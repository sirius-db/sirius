# Developing CLion

Since SiriusDB is an extension to DuckDB, the CMake project source is actually the `duckdb` directory (a submodule of
this project), which then pulls in SiriusDB as an extension. We use

- symlinked sirius-specific `CMakePresets.json` (at `cmake/CMakePresets.json`) to version control the build config.
- [pixi](https://pixi.prefix.dev/) to manage build dependencies.

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
