# Known Build Error Patterns

## GLIBCXX / CXXABI Undefined References

**Symptoms:** `undefined reference to '__cxx11::basic_string'`, `GLIBCXX_3.4.xx not found`, `CXXABI_1.3.xx not found`

**Cause:** Mismatch between the C++ standard library used by cuDF/conda and the system compiler.

**Fix:**
```bash
export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib $LDFLAGS"
rm -rf build
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
```

## CUDA Architecture Mismatch

**Symptoms:** `unsupported gpu architecture 'compute_XX'`, `nvcc fatal: Unsupported gpu architecture`

**Cause:** CUDA toolkit version doesn't support the target GPU architecture specified in CMakeLists.txt.

**Fix:** Check `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt and ensure it matches the installed CUDA toolkit version. Turing (75) requires CUDA 10+, Ampere (80, 86) requires CUDA 11+, Hopper (90a) requires CUDA 12+, Blackwell (100f, 120, 120a) requires CUDA 13+.

## sccache Frozen/Unresponsive

**Symptoms:** Build hangs indefinitely, no progress after initial cmake configuration.

**Cause:** sccache server is in a bad state.

**Fix:**
```bash
sccache --stop-server
sccache --start-server
```

## Submodule Out of Sync

**Symptoms:** Missing header files, undefined types from DuckDB, build errors in `duckdb/` or `cucascade/`.

**Cause:** Git submodules not at the expected commits.

**Fix:**
```bash
git submodule update --init --recursive
```

## cuDF API Changes

**Symptoms:** `no member named 'X' in namespace 'cudf'`, `no matching function for call to 'cudf::...'`

**Cause:** cuDF API changed between versions. The pixi environment pins a specific cuDF version.

**Fix:** Check `pixi.toml` for the expected cuDF version. Consult cuDF changelog for API changes. Update the call site to match the current API.

## Template Instantiation Errors

**Symptoms:** Long template error messages, often involving `std::variant`, `cudf::type_dispatcher`, or DuckDB types.

**Cause:** Usually a missing type case in a type dispatcher or visitor pattern.

**Fix:** Read the full error to find which type is missing from the dispatcher. Add the missing case. Check if the type needs to be added to the supported types list.

## Linker Errors: Multiple Definition

**Symptoms:** `multiple definition of 'symbol'`

**Cause:** Usually an inline function defined in a header without the `inline` keyword, or a variable defined in a header without `inline`.

**Fix:** Add `inline` keyword, or move the definition to a .cpp file.

## CUDA Separable Compilation Issues

**Symptoms:** `undefined reference` to device functions across translation units, `__device__` function not found.

**Cause:** CUDA separable compilation not enabled for a target that needs it.

**Fix:** Ensure `CMAKE_CUDA_SEPARABLE_COMPILATION ON` is set. Check that the relevant CMake target has `CUDA_SEPARABLE_COMPILATION ON` property.

## Missing Pixi Environment

**Symptoms:** Various missing library or header errors, especially for cuDF, RMM, or spdlog.

**Cause:** Build not running inside pixi shell.

**Fix:**
```bash
pixi shell
source setup_sirius.sh
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
```

## OOM During Build

**Symptoms:** Build killed by OS (signal 9), compiler internal error, or system becomes unresponsive.

**Cause:** Too many parallel compilation jobs consuming all RAM. CUDA compilation is especially memory-hungry.

**Fix:**
```bash
CMAKE_BUILD_PARALLEL_LEVEL=4 make
```
