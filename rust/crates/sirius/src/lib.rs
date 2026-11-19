//! Safe Rust bindings to the public Sirius C ABI.
//!
//! Build immutable configurations with [`ContextConfigBuilder`] and physical plans with [`PlanNode`].
//! These bindings require no C++ compilation.
//! Existing execution and object-store bindings live in the `sirius-ffi` crate.
//!
//! Build and inspect a local Parquet scan with a typed filter:
//!
//! ```no_run
#![doc = include_str!("../examples/plan.rs")]
//! ```

mod config;
mod diagnostic;
pub use config::{ConfigError, ContextConfig, ContextConfigBuilder};
mod plan;
pub use plan::{
    Comparison, DataType, Expression, Field, Plan, PlanError, PlanNode, PlanNodeKind, Schema,
};
