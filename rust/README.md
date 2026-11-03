# Sirius Rust bindings

[`sirius-ffi`](crates/sirius-ffi) provides the existing C++ FFI bridge, safe query
execution wrappers, and object-store bindings for Sirius. StarRocks and Doris
use this crate. It remains supported without deprecation warnings while the
public C API gains execution support.

The `telemetry/*` crates are unrelated: CMake links those into the engine.

## Migrating imports

The safe wrappers previously in `sirius` now live in `sirius-ffi`:

```rust
use sirius_ffi::SiriusContext;
```

The raw C++ bridge previously in `sirius-sys` is available through
`sirius_ffi::bridge`. Its context factories and object-store types are also
re-exported at the crate root.

## Building and testing

Build Sirius first. The bridge requires a C++20 compiler matching the engine's
C++ toolchain:

```bash
pixi run make
pixi run cargo test --locked --no-run --manifest-path rust/Cargo.toml -p sirius-ffi
```

Context construction and execution tests require a GPU:

```bash
LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  pixi run cargo test --locked --manifest-path rust/Cargo.toml -p sirius-ffi
```

## Documentation

```bash
pixi run -e docs docs-rust
```

Open `build/docs/rust-target/doc/sirius_ffi/index.html`. The task sets
`DOCS_RS=1` to skip native compilation and linking. Use that variable only for
documentation builds.

## Linkage

The build script compiles the bridge against `include/` and links a Sirius
artifact under `SIRIUS_BUILD_DIR` (default `build/release`):

- By default, `libsirius.so` supplies its dependencies through `DT_NEEDED`.
- The `static` feature requires an externally bundled `libsirius.a`. The current
  CMake build does not produce that bundle, so its raw static archive is not
  supported by this feature. With a suitable bundle, the C++ runtime is
  linked separately. Applications must supply native dependencies not included
  in their archive, including NVIDIA driver libraries. Bundles with duplicate
  DuckDB symbols also require `-C link-arg=-Wl,--allow-multiple-definition` on
  the final application. The crate sets this flag for its own targets, but
  Cargo does not propagate it to dependent applications.

The build script searches the build root, then `extension/sirius`.
`CONDA_PREFIX`, set by Pixi, supplies search paths for CUDA/RAPIDS dependencies.
