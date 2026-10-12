# DuckDB v2 migration

The engine and wrapper use unmodified DuckDB revision
`561522aea03e400bd20adc64fcc63e78b8721f3f`. They use DuckDB's C++ APIs and must
agree on that revision and a compatible toolchain. This is not a stable C ABI
boundary.

## Branch order

1. Standalone Sirius CMake.
2. Installable shared and static libraries with exported CMake targets.
3. Separate `sirius-duckdb` build: shared Sirius for development, the local
   Sirius vcpkg port for static distribution.
4. DuckDB v2 engine and extension API migration.
5. Additional extension validation, followed by the lightweight `sirius`
   extension selecting a `sirius_cuda12` or `sirius_cuda13` backend.

Conda packaging is independent of this sequence. The extension stays in this
repository until its repository move is approved.

## Build and test

Follow [the wrapper build instructions](../sirius-duckdb/README.md).
`pixi run make` builds shared Sirius and the matching DuckDB host and extension.
`pixi run make test` runs the C++ tests, including dynamic extension checks in a
separate DuckDB-only test host. Avro and Iceberg test providers are built from DuckDB's pinned configurations:

```sh
pixi run make EXTRA_EXTENSION_CONFIGS="$PWD/sirius-duckdb/test/extension_config.cmake"
pixi run sirius-duckdb/build/release/duckdb -unsigned -c "
  INSTALL './sirius-duckdb/build/release/extension/avro/avro.duckdb_extension';
  INSTALL './sirius-duckdb/build/release/extension/iceberg/iceberg.duckdb_extension';"
```

CI transfers those binaries from the CPU build job to the GPU test job; it does
not download providers for this development revision. Test databases allow these
locally built, unsigned providers.

Sirius disables partial-aggregate pushdown alongside its existing unsupported
optimizers because the new `combine_aggr` expressions cannot execute on the GPU.
This setting also applies to CPU queries after loading Sirius. Correlated joins
retain v2's default CTE representation.

The packaged DuckDB 1.5 Python client remains
available for fixture generation; it cannot load the v2 extension.
The Python performance runners (`performance_test.py` and
`tpch_power_throughput.py`) also require a Python client built from the pinned
revision. Use the CLI-based TPC-H scripts with the built v2 host in the default
environment.

Distribution continues to use the Sirius fork of extension-ci-tools with
subdirectory support. Its vcpkg build bundles dependencies into the extension.
The post-link check permits NVIDIA driver libraries and ordinary Linux runtime
libraries, and rejects shared Sirius, RAPIDS and CUDA runtime dependencies.
Compiler runtimes are linked statically.

Use separate build directories when changing toolchains. For a smaller local
static build, set both `CUDAARCHS` and `VCPKG_CUDA_ARCHITECTURES` to the target
GPU's architecture. The latter participates in the vcpkg binary cache key.

A stable C boundary and cross-version support require host hooks for optimizer
interception, execution replacement and native-storage transaction access.
