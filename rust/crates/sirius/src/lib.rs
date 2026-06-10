//! Safe, idiomatic Rust bindings for [Sirius](https://github.com/sirius-db/sirius),
//! the GPU-native SQL engine.
//!
//! This crate wraps the low-level [`sirius-sys`] cxx bindings in safe Rust types
//! — the entry point for driving Sirius from Rust.
//!
//! Today it binds just enough to prove the toolchain links against the real
//! Sirius library: constructing a [`SiriusContext`]. More of the API surface is
//! added in later PRs.

use cxx::UniquePtr;

/// An initialized Sirius engine context.
///
/// Constructing one brings up the engine (GPU resources included); dropping it
/// tears the engine down. The `cxx::UniquePtr` owns the C++ object, so lifetime
/// is pure RAII — there is no uninitialized or manually-freed state.
pub struct SiriusContext {
    // RAII handle; read by methods added in later PRs (execute, ...). `expect`
    // flags this for removal once that happens.
    #[expect(dead_code, reason = "owned for its RAII lifetime until the API grows")]
    inner: UniquePtr<sirius_sys::Context>,
}

impl SiriusContext {
    /// Bring up a new, initialized Sirius engine context.
    pub fn new() -> Self {
        Self {
            inner: sirius_sys::make_context(),
        }
    }
}

impl Default for SiriusContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::SiriusContext;

    /// Proof-of-life: bring up a real Sirius engine context and drop it. This
    /// links the real Sirius library and exercises the full cxx round-trip +
    /// `initialize()`/teardown. Requires a GPU (construction does GPU bring-up).
    #[test]
    fn constructs_and_drops() {
        let _ctx = SiriusContext::new();
    }
}
