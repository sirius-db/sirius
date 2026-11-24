//! A single static library for the engine's Rust components.

// Include both crates in the native archive, including telemetry's NVTX initializer.
use sirius_substrait as _;
use telemetry_bridge as _;
