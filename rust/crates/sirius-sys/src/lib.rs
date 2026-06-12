//! Low-level `cxx` bindings to the Sirius C++ API.
//!
//! This crate is intentionally thin: it exposes the C++ types and free functions
//! declared in the `#[cxx::bridge]` module below and nothing else. Safe, idiomatic
//! wrappers live in the [`sirius`](https://docs.rs/sirius) crate.
//!
//! The bridge binds Sirius's **public C++ surface** (`src/include/sirius_ffi.hpp`):
//! an RAII [`Context`] held via [`cxx::UniquePtr`]. Constructing it brings up an
//! initialized engine; dropping the `UniquePtr` tears it down. The header is
//! lightweight, so the bridge compiles without any of Sirius's internal headers
//! (cudf/rmm/duckdb). It is the seed of the public API `libsirius` will expose;
//! the bindings link whichever Sirius artifact provides these symbols (the DuckDB
//! extension today, a dedicated `libsirius` later — see `build.rs`).

#[cxx::bridge(namespace = "sirius::ffi")]
mod ffi {
    unsafe extern "C++" {
        include!("sirius_ffi.hpp");

        /// RAII handle to an initialized Sirius engine context.
        type Context;

        /// Construct an initialized [`Context`], owned by the returned `UniquePtr`.
        fn make_context() -> UniquePtr<Context>;
    }
}

pub use ffi::{Context, make_context};
