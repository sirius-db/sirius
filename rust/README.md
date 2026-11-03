# Sirius Rust bindings

Crates for driving [Sirius](https://github.com/sirius-db/sirius) from Rust
(sirius-db/sirius #835).

| Crate | Role |
|-------|------|
| [`sirius-sys`](crates/sirius-sys) | Raw bindings and native library linking for the public C ABI. |
| [`sirius`](crates/sirius) | Safe configuration wrappers over `sirius-sys`. |
| [`sirius-ffi`](crates/sirius-ffi) | Existing C++ FFI bridge, execution wrappers, and object-store bindings. |

(The `telemetry/*` crates are unrelated — Rust linked *into* the C++ extension via
CMake/Corrosion, the opposite direction.)

## Configuration

Build a validated, immutable configuration without initializing GPUs:

```rust
use sirius::{ConfigError, ContextConfigBuilder};

fn load_config() -> Result<sirius::ContextConfig, ConfigError> {
    ContextConfigBuilder::from_yaml("sirius.yaml")?.build()
}
```

Use `ContextConfigBuilder::new()?.build()?` for built-in defaults. Both the
builder and configuration offer `try_clone()` to create independent handles
sharing immutable settings. The file is read only by `from_yaml()`; later file
changes do not affect a loaded builder. Configuration errors retain their
I/O, YAML syntax, or invalid-setting category in `ConfigError`.

Configuration construction calls the C ABI directly. Building `sirius`
and `sirius-sys` requires no C++ compilation. The configuration types
are neither `Send` nor `Sync`.

## Existing execution and object-store API

Execution wrappers moved from `sirius` to `sirius-ffi`:

```rust
use sirius_ffi::SiriusContext;
```

The raw C++ bridge previously exposed by `sirius-sys` is now available through
`sirius_ffi::bridge` and re-exported from the `sirius-ffi` crate root, including
`ObjectStoreConfigArgs` and its factories. StarRocks and Doris use this crate.
This API remains supported without deprecation warnings while the public C API
adds execution support.

## Building & testing

All three crates link a Sirius library artifact. Only `sirius-ffi` compiles a C++20
bridge and requires a matching C++ toolchain. Build Sirius first, then use cargo:

```bash
pixi run make                       # builds the Sirius extension (+ artifact + headers)
# build + link the tests (no GPU needed):
pixi run cargo test --no-run --manifest-path rust/Cargo.toml -p sirius -p sirius-sys -p sirius-ffi
```

The `sirius-ffi` execution tests need a GPU; the public configuration tests do not
initialize the engine. Point the runtime loader at the build-tree library:

```bash
LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  pixi run cargo test --manifest-path rust/Cargo.toml -p sirius -p sirius-sys -p sirius-ffi
```

## Documentation

Build with the isolated Pixi docs environment:

```bash
pixi run -e docs docs-rust
```

Open `build/docs/rust-target/doc/sirius/index.html`. The task also publishes the
`sirius-sys` reference and sets `DOCS_RS` for the documentation build.
The `sirius-ffi` reference is excluded from publication.

Generate the `sirius` API reference without building or linking `libsirius`:

```bash
DOCS_RS=1 cargo doc --locked --manifest-path rust/Cargo.toml -p sirius --lib --no-deps
```

Open `rust/target/doc/sirius/index.html`. This requires only a Rust toolchain,
with no Sirius build, CUDA toolkit, GPU, or C++ compiler.
`DOCS_RS` skips Sirius's native bridge compilation and library lookup; use it
for documentation and the binding-maintenance tasks, leaving it unset for normal
builds and tests.

## Linkage

`build.rs` discovers the Sirius artifact under `$SIRIUS_BUILD_DIR` (default
`build/release`) and links one Sirius library:

- **default** → `libsirius.so` (shared; pulls its deps via `DT_NEEDED`).
- **`--features static`** → an externally bundled `libsirius.a`. The current
  CMake build does not produce this bundle; its raw archive is unsupported.
  With a suitable bundle, the C++ standard library is linked separately. This feature alone does
  not produce a fully static executable. Applications must also supply any native
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

Raw declarations are checked in under `sirius-sys/src/generated.rs`. The
`sirius-sys` build script runs the pinned bindgen tool only when `SIRIUS_BINDGEN`
is set to `write` (regenerate) or `check` (reject drift). Ordinary
builds require neither bindgen nor libclang. After changing the public C headers,
run `pixi run --locked -e bindings generate-bindings` and commit the result.
`pixi run --locked -e bindings check-bindings` checks for drift in CI using pinned
bindgen and Clang versions. These maintenance tasks set `DOCS_RS=1` to skip
native linking; they do not require a built Sirius library. The build script also
compares declarations for Linux x86-64,
Linux AArch64, and macOS AArch64; this checks declaration portability, not native
library support on those platforms.

Opaque handle definitions remain handwritten to preserve their thread-safety and
pinning restrictions. Generation rejects new opaque handles until those definitions
and the generator blocklist are updated. The ABI version accessor retains its safe Rust wrapper.
All other function declarations and status values come from the C headers.
When adding functions, also add them to the explicit exports in `c_api.rs`.

## Checking unsafe Rust with Miri

Install the pinned interpreter once:

```bash
rustup toolchain install nightly-2026-11-01 --profile minimal --component miri --component rust-src
pixi run --locked -e bindings miri
```

CI runs these tests for `sirius` and `sirius-sys` without a GPU or native library.
The test executables supply Rust-owned C ABI stand-ins so Miri can inspect pointer
access, ownership, diagnostic lifetimes, and failure cleanup. They do not model
YAML parsing or validate C++/CUDA behavior. The existing native configuration and
GPU suites continue to check the real library. Legacy `sirius-ffi` is outside
this Miri suite.
