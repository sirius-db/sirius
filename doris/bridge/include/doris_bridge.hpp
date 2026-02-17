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

/// Descriptor for a column to be registered as an exchange table.
struct ExchangeColumnDesc {
    rust::String name;
    int32_t type_id;       // PGenericType::TypeId value
    bool is_nullable;
    uint32_t precision;    // for decimals
    uint32_t scale;        // for decimals
};

/// Register exchange data as a GPU table in GPUBufferManager.
///
/// Copies decoded PBlock column data to GPU memory and creates a
/// GPUIntermediateRelation that the Substrait plan can reference.
///
/// @param ctx            Engine context (unused, but kept for API consistency)
/// @param table_name     Table name for GPUBufferManager (will be uppercased)
/// @param num_rows       Number of rows across all columns
/// @param columns        Column descriptors (type, name, nullable)
/// @param column_data    Raw column data per column (host memory)
/// @param null_masks     Null mask per column (byte-per-row, 0=valid, 1=null)
/// @param string_offsets String offset arrays per column (N+1 entries)
void register_exchange_table(
    const BridgeContext& ctx,
    rust::Str table_name,
    uint32_t num_rows,
    rust::Slice<const ExchangeColumnDesc> columns,
    rust::Slice<const rust::Vec<uint8_t>> column_data,
    rust::Slice<const rust::Vec<uint8_t>> null_masks,
    rust::Slice<const rust::Vec<uint64_t>> string_offsets);

}  // namespace doris_bridge
