//! Build script for `sirius`.
//!
//! The Sirius artifact is linked by `sirius-sys`'s build script (its
//! `cargo:rustc-link-{lib,search}` propagate here). Only one extra linker flag is
//! needed, and only for the static bundle: it embeds DuckDB's unity-build objects
//! whose duplicate strong symbols would otherwise conflict (the shared lib
//! resolves them internally, so the default path needs nothing). `rustc-link-arg`
//! doesn't propagate across crates, so it is emitted here, where the test binary
//! that triggers the link is built.

fn main() {
    if std::env::var_os("CARGO_FEATURE_STATIC").is_some() {
        println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
    }
}
