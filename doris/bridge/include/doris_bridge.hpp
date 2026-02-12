#pragma once

#include "rust/cxx.h"
#include <cstdint>
#include <memory>

namespace doris_bridge {

/// Opaque handle to a headless DuckDB + Sirius engine context.
struct BridgeContext;

/// Create a headless DuckDB instance with the Sirius GPU extension loaded.
///
/// @param config_path  Path to sirius.cfg (empty = defaults)
/// @param gpu_ids      GPU device IDs to use
/// @return             Opaque context handle
std::unique_ptr<BridgeContext> create_context(
    rust::Str config_path,
    rust::Slice<const int32_t> gpu_ids);

/// Execute a Substrait plan and return Arrow IPC stream bytes.
///
/// @param ctx         Engine context
/// @param plan_bytes  Serialized Substrait Plan protobuf
/// @return            Arrow IPC stream bytes (schema + record batches)
rust::Vec<uint8_t> execute_substrait_plan(
    const BridgeContext& ctx,
    rust::Slice<const uint8_t> plan_bytes);

/// Get the number of rows in the last execution result (for diagnostics).
int64_t last_result_rows(const BridgeContext& ctx);

}  // namespace doris_bridge
