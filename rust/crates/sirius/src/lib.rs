#![warn(missing_docs)]
#![doc(test(attr(forbid(unsafe_code))))]
//! Configure Sirius and construct immutable query plans in Rust.
//!
//! Use [`ContextConfigBuilder`] to load a YAML configuration or select the built-in
//! defaults. Use [`Schema`], [`PlanNode`], and [`Expression`] to describe a query,
//! then call [`Plan::from_root`] to finish it. Plan construction requires no GPU or
//! engine context. For a Parquet scan, supply the schema you expect to read.
//!
//! This crate manages native handle lifetimes for you. It uses the owning handles in
//! `sirius_sys::handles` to call the public C ABI. The [C reference](https://sirius-db.github.io/sirius/c/)
//! describes the underlying handles, and the [C++ reference](https://sirius-db.github.io/sirius/cpp/)
//! documents the corresponding C++ API. Rust applications can use these bindings
//! without compiling C++. Execution and object-store bindings live in `sirius-ffi`.
//!
//! # Example
//!
//! Construct a scan of `data.parquet` and keep rows where `id > 10`. The example
//! prints the plan description without opening the file or executing the query.
//!
//! ```rust
#![doc = include_str!("../examples/plan.rs")]
//! ```

mod config;
pub use config::{ConfigError, ContextConfig, ContextConfigBuilder};
mod plan;
pub use plan::{
    Comparison, DataType, Expression, Field, Plan, PlanError, PlanNode, PlanNodeKind, Schema,
};
