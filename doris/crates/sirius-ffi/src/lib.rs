//! FFI bridge between Rust and the Sirius C++ engine via cxx.
//!
//! Provides a safe Rust API around the C++ bridge:
//! - [`SiriusEngine::new`]: Create a headless DuckDB instance with Sirius loaded
//! - [`SiriusEngine::execute_plan`]: Execute Substrait plan bytes → Arrow IPC bytes
//!
//! # Feature gating
//!
//! With the `sirius-engine` feature enabled, this crate compiles and links the
//! C++ bridge (requires DuckDB + Sirius + Arrow C++ libraries).
//!
//! Without the feature (default), stub implementations are provided so that
//! `cargo check` and tests work without any C++ dependencies.

#[cfg(feature = "sirius-engine")]
#[cxx::bridge(namespace = "doris_bridge")]
mod ffi {
    unsafe extern "C++" {
        include!("doris_bridge.hpp");

        /// Opaque handle to the headless DuckDB + Sirius engine context.
        type BridgeContext;

        /// Create a headless DuckDB instance with the Sirius GPU extension loaded.
        fn create_context(
            config_path: &str,
            gpu_ids: &[i32],
        ) -> UniquePtr<BridgeContext>;

        /// Execute a Substrait plan and return Arrow IPC stream bytes.
        fn execute_substrait_plan(
            ctx: &BridgeContext,
            plan_bytes: &[u8],
        ) -> Vec<u8>;

        /// Get the number of rows in the last execution result.
        fn last_result_rows(ctx: &BridgeContext) -> i64;
    }
}

/// High-level Rust wrapper around the Sirius C++ engine.
///
/// Manages the lifecycle of a headless DuckDB + Sirius context and provides
/// a safe API for executing Substrait plans.
pub struct SiriusEngine {
    #[cfg(feature = "sirius-engine")]
    ctx: cxx::UniquePtr<ffi::BridgeContext>,

    #[cfg(not(feature = "sirius-engine"))]
    _phantom: (),
}

impl SiriusEngine {
    /// Create a new engine context.
    ///
    /// - `config_path`: Path to sirius.cfg (empty string for defaults)
    /// - `gpu_ids`: GPU device IDs to use (empty slice for all available)
    pub fn new(config_path: &str, gpu_ids: &[i32]) -> Result<Self, EngineError> {
        #[cfg(feature = "sirius-engine")]
        {
            let ctx = ffi::create_context(config_path, gpu_ids);
            if ctx.is_null() {
                return Err(EngineError::InitFailed(
                    "create_context returned null".into(),
                ));
            }
            Ok(Self { ctx })
        }

        #[cfg(not(feature = "sirius-engine"))]
        {
            let _ = (config_path, gpu_ids);
            Err(EngineError::NotCompiled)
        }
    }

    /// Execute a Substrait plan and return Arrow IPC stream bytes.
    ///
    /// The returned bytes contain a complete Arrow IPC stream (schema message
    /// followed by zero or more record-batch messages and an EOS marker).
    pub fn execute_plan(&self, substrait_bytes: &[u8]) -> Result<Vec<u8>, EngineError> {
        #[cfg(feature = "sirius-engine")]
        {
            let result = ffi::execute_substrait_plan(&self.ctx, substrait_bytes);
            Ok(result)
        }

        #[cfg(not(feature = "sirius-engine"))]
        {
            let _ = substrait_bytes;
            Err(EngineError::NotCompiled)
        }
    }

    /// Get the row count from the last execution (diagnostics).
    pub fn last_result_rows(&self) -> i64 {
        #[cfg(feature = "sirius-engine")]
        {
            ffi::last_result_rows(&self.ctx)
        }

        #[cfg(not(feature = "sirius-engine"))]
        {
            0
        }
    }
}

/// Errors from the Sirius engine bridge.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Sirius engine not compiled (enable the `sirius-engine` feature)")]
    NotCompiled,

    #[error("Engine initialization failed: {0}")]
    InitFailed(String),

    #[error("Plan execution failed: {0}")]
    ExecFailed(String),
}
