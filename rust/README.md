# Sirius Rust bindings

Crates for driving [Sirius](https://github.com/sirius-db/sirius) from Rust.

| Crate | Role |
|-------|------|
| [`sirius-sys`](crates/sirius-sys) | Raw bindings and native library linking for the public C ABI. |
| [`sirius`](crates/sirius) | Safe configuration wrappers over `sirius-sys`. |
| [`sirius-ffi`](crates/sirius-ffi) | Existing C++ FFI bridge, execution wrappers, and object-store bindings. |

Before [#2436](https://github.com/sirius-db/sirius/pull/2436), `sirius` and
`sirius-sys` referred to the C++ bridge. These names now refer to the public C ABI
crates; the existing C++ bridge lives in `sirius-ffi`.

The `telemetry/*` crates are unrelated: CMake links those into the engine.

## Configuration

Build a validated, immutable configuration without initializing GPUs:

```rust
use sirius::{ConfigError, ContextConfigBuilder};

fn load_config() -> Result<sirius::ContextConfig, ConfigError> {
    ContextConfigBuilder::from_yaml("sirius.yaml")?.build()
}
```

Use `ContextConfigBuilder::new()?.build()?` for built-in defaults. Both the
builder and configuration implement `Clone` to create independent handles
sharing immutable settings. The file is read only by `from_yaml()`; later file
changes do not affect a loaded builder. Configuration errors retain their
I/O, YAML syntax, or invalid-setting category in `ConfigError`.

Configuration construction calls the C ABI directly. Building `sirius`
and `sirius-sys` requires no C++ compilation. The configuration types
implement `Send` and `Sync`: they may be moved or shared between threads,
including concurrent building and cloning.

## Existing execution and object-store API

Use `sirius-ffi` for the existing execution API:

```rust
use sirius_ffi::SiriusContext;
```

The raw C++ bridge is available through
`sirius_ffi::bridge` and re-exported from the `sirius-ffi` crate root, including
`ObjectStoreConfigArgs` and its factories. StarRocks and Doris use this crate.
This API remains supported without deprecation warnings while the public C API
adds execution support.

## Building and testing

All three crates link a Sirius library artifact. Only `sirius-ffi` compiles a C++20
bridge and requires a matching C++ toolchain. Build Sirius first, then use cargo:

```bash
pixi run make                       # builds the Sirius extension (+ artifact + headers)
# build + link the tests (no GPU needed):
pixi run cargo test --locked --no-run --manifest-path rust/Cargo.toml -p sirius -p sirius-sys -p sirius-ffi
```

The `sirius-ffi` execution tests need a GPU; the public configuration tests do not
initialize the engine. Point the runtime loader at the build-tree library:

```bash
LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  pixi run cargo test --locked --manifest-path rust/Cargo.toml -p sirius -p sirius-sys -p sirius-ffi
```

## Documentation

Build with the isolated Pixi docs environment:

```bash
pixi run -e docs docs-rust
```

Open `build/docs/rust-target/doc/sirius/index.html`. The task builds the `sirius`
and `sirius-sys` references without native compilation or linking. The
`sirius-ffi` reference is excluded from publication.

## Linkage

`build.rs` discovers the Sirius artifact under `$SIRIUS_BUILD_DIR` (default
`build/release`) and links one Sirius library:

- By default, `libsirius.so` supplies its dependencies through `DT_NEEDED`.
- The `static` feature requires an externally bundled `libsirius.a`. The current
  CMake build does not produce this bundle; its raw archive is unsupported.
  With a suitable bundle, the C++ standard library is linked separately. This
  feature alone does not produce a fully static executable. Applications must also supply any native
  dependencies not bundled in their archive (including the NVIDIA driver libraries).
  Bundles containing duplicate DuckDB symbols require
  `-C link-arg=-Wl,--allow-multiple-definition` on the final binary; this flag is
  applied to these crates' own targets but not propagated to dependent applications.

`sirius-sys/build.rs` supplies library linking for both API paths.
`sirius-ffi/build.rs` compiles the execution bridge against the public `include`
directory. Configuration uses the C declarations in
`sirius-sys/src/c_api.rs`. Neither path includes internal engine headers.

## Environment

- `DOCS_RS` — when set, skip Sirius's native build and link steps for documentation or binding maintenance.
- `SIRIUS_BUILD_DIR` — Sirius build tree (default `build/release`).
- `CONDA_PREFIX` — set by `pixi`; used to find the shared library's dependencies.
- `CARGO_NET_GIT_FETCH_WITH_CLI=true` — only on machines whose git config rewrites
  `https://github.com/` to SSH (the telemetry crate's `quent` git dep otherwise
  fails libgit2's ssh-agent path). CI is unaffected.

## Maintaining the C bindings

Raw declarations are checked in under `sirius-sys/src/generated.rs`. Enable the
maintainer-only `generate-bindings` Cargo feature to regenerate them in `build.rs`
using the pinned bindgen library. Normal builds do not compile bindgen or need
libclang.

After changing the public C headers, run
`pixi run --locked -e bindings generate-bindings` and commit the result. The task
uses the pinned Clang environment and sets `DOCS_RS=1` to skip native linking.
Generation checks that Linux amd64 and arm64 produce identical declarations.
CI regenerates the file and uses `git diff --exit-code` to detect drift.

Opaque handle definitions remain handwritten to preserve their thread-safety and
pinning restrictions. Generation rejects new opaque handles until those definitions
and the generator blocklist are updated. The ABI version accessor retains its safe Rust wrapper.
All other function declarations and status values come from the C headers.
Include new public headers in `bindgen/wrapper.h`, and add new functions to the
explicit exports in `c_api.rs`.

## Checking unsafe Rust with Miri

Install or update the nightly interpreter:

```bash
rustup toolchain install nightly --profile minimal --component miri --component rust-src
cargo +nightly miri test --locked --manifest-path rust/Cargo.toml -p sirius -p sirius-sys --test miri
```

CI runs these tests for `sirius` and `sirius-sys` without a GPU or native library.
The test executables supply Rust-owned C ABI stand-ins so Miri can inspect pointer
access, ownership, diagnostic lifetimes, and failure cleanup. They do not model
YAML parsing or validate C++/CUDA behavior. The existing native configuration and
GPU suites continue to check the real library. Legacy `sirius-ffi` is outside
this Miri suite.
