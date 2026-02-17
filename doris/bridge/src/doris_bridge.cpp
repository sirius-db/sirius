//! C++ bridge implementation: headless DuckDB + Sirius execution.
//!
//! This file creates a DuckDB instance with the Sirius GPU extension loaded,
//! executes Substrait plans, and returns results as Arrow IPC stream bytes.

#include "doris_bridge.hpp"

#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/arrow/result_arrow_wrapper.hpp"

#include "from_substrait.hpp"

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/memory.h>

#include "gpu_buffer_manager.hpp"
#include "gpu_columns.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace doris_bridge {

/// Opaque context holding a headless DuckDB instance + connection.
struct BridgeContext {
    std::unique_ptr<duckdb::DuckDB> db;
    std::unique_ptr<duckdb::Connection> conn;
    int64_t last_rows = 0;
};

std::unique_ptr<BridgeContext> create_context(
    rust::Str config_path,
    rust::Slice<const int32_t> gpu_ids)
{
    // Point Sirius at a config file if provided.
    std::string cfg(config_path.data(), config_path.size());
    if (!cfg.empty()) {
        setenv("SIRIUS_CONFIG_FILE", cfg.c_str(), 1);
    }

    // GPU device selection via CUDA_VISIBLE_DEVICES.
    if (!gpu_ids.empty()) {
        std::string ids;
        for (size_t i = 0; i < gpu_ids.size(); ++i) {
            if (i > 0) ids += ',';
            ids += std::to_string(gpu_ids[i]);
        }
        setenv("CUDA_VISIBLE_DEVICES", ids.c_str(), 1);
    }

    // Create in-memory DuckDB instance.
    // The Sirius extension auto-registers via SiriusContextExtensionCallback
    // which is set up in SiriusExtension::LoadInternal().
    auto ctx = std::make_unique<BridgeContext>();
    ctx->db = std::make_unique<duckdb::DuckDB>(nullptr);
    ctx->conn = std::make_unique<duckdb::Connection>(*ctx->db);

    return ctx;
}

rust::Vec<uint8_t> execute_substrait_plan(
    const BridgeContext& ctx,
    rust::Slice<const uint8_t> plan_bytes)
{
    // 1. Parse Substrait plan bytes into a DuckDB Relation tree.
    std::string serialized(
        reinterpret_cast<const char*>(plan_bytes.data()),
        plan_bytes.size());

    auto client_ctx = ctx.conn->context;
    duckdb::SubstraitToDuckDB converter(client_ctx, serialized, /*json=*/false);
    auto relation = converter.TransformPlan();

    // 2. Execute the relation (goes through DuckDB's execution engine).
    //    With the Sirius extension loaded, this may use GPU operators.
    auto result = relation->Execute();
    if (result->HasError()) {
        throw std::runtime_error(
            "Query execution failed: " + result->GetError());
    }

    // 3. Wrap the DuckDB result as an ArrowArrayStream (C Data Interface).
    auto stream_wrapper =
        duckdb::make_uniq<duckdb::ResultArrowArrayStreamWrapper>(
            std::move(result), /*batch_size=*/1024);

    // 4. Import into Arrow C++ RecordBatchReader.
    auto reader_result =
        arrow::ImportRecordBatchReader(&stream_wrapper->stream);
    if (!reader_result.ok()) {
        throw std::runtime_error(
            "Arrow import failed: " + reader_result.status().ToString());
    }
    auto reader = reader_result.ValueUnsafe();

    // 5. Serialize to Arrow IPC stream format.
    auto buffer_result = arrow::io::BufferOutputStream::Create(1024 * 1024);
    if (!buffer_result.ok()) {
        throw std::runtime_error(
            "Buffer creation failed: " + buffer_result.status().ToString());
    }
    auto output = buffer_result.ValueUnsafe();

    auto writer_result =
        arrow::ipc::MakeStreamWriter(output, reader->schema());
    if (!writer_result.ok()) {
        throw std::runtime_error(
            "IPC writer creation failed: " + writer_result.status().ToString());
    }
    auto writer = writer_result.ValueUnsafe();

    int64_t total_rows = 0;
    while (true) {
        auto batch_result = reader->Next();
        if (!batch_result.ok()) {
            throw std::runtime_error(
                "Batch read failed: " + batch_result.status().ToString());
        }
        auto batch = batch_result.ValueUnsafe();
        if (!batch) break;

        total_rows += batch->num_rows();
        auto status = writer->WriteRecordBatch(*batch);
        if (!status.ok()) {
            throw std::runtime_error(
                "Batch write failed: " + status.ToString());
        }
    }

    auto close_status = writer->Close();
    if (!close_status.ok()) {
        throw std::runtime_error(
            "Writer close failed: " + close_status.ToString());
    }

    auto finish_result = output->Finish();
    if (!finish_result.ok()) {
        throw std::runtime_error(
            "Buffer finish failed: " + finish_result.status().ToString());
    }
    auto buffer = finish_result.ValueUnsafe();

    // 6. Copy Arrow IPC bytes into rust::Vec<uint8_t>.
    const auto* data = buffer->data();
    auto size = buffer->size();

    rust::Vec<uint8_t> out;
    out.reserve(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
        out.push_back(data[i]);
    }

    // Store row count for diagnostics.
    const_cast<BridgeContext&>(ctx).last_rows = total_rows;

    return out;
}

int64_t last_result_rows(const BridgeContext& ctx) {
    return ctx.last_rows;
}

/// Map PGenericType::TypeId → GPUColumnType.
static GPUColumnType map_ptype_to_gpu(int32_t type_id, uint32_t precision, uint32_t scale) {
    // PGenericType::TypeId values (from types.proto):
    //   INT8=6, INT16=7, INT32=8, INT64=9, FLOAT=12, DOUBLE=13, BOOLEAN=14,
    //   DATE=15, DATETIME=16, STRING=22, DATEV2=28, DATETIMEV2=29,
    //   DECIMAL32=23, DECIMAL64=24, DECIMAL128=25, DECIMAL128I=32, DECIMAL256=38
    switch (type_id) {
        case 7:  // INT16
            return GPUColumnType(GPUColumnTypeId::INT16);
        case 8:  // INT32
        case 6:  // INT8 (widen to INT32)
            return GPUColumnType(GPUColumnTypeId::INT32);
        case 9:  // INT64
        case 10: // INT128 (truncated to INT64 for now)
            return GPUColumnType(GPUColumnTypeId::INT64);
        case 12: // FLOAT
            return GPUColumnType(GPUColumnTypeId::FLOAT32);
        case 13: // DOUBLE
            return GPUColumnType(GPUColumnTypeId::FLOAT64);
        case 14: // BOOLEAN
            return GPUColumnType(GPUColumnTypeId::BOOLEAN);
        case 15: // DATE
        case 28: // DATEV2
            return GPUColumnType(GPUColumnTypeId::DATE);
        case 16: // DATETIME
        case 29: // DATETIMEV2
            return GPUColumnType(GPUColumnTypeId::TIMESTAMP_US);
        case 22: // STRING
        case 26: // BYTES
        case 31: // JSONB
            return GPUColumnType(GPUColumnTypeId::VARCHAR);
        case 23: // DECIMAL32
        case 24: // DECIMAL64
        case 25: // DECIMAL128
        case 32: // DECIMAL128I
        case 38: { // DECIMAL256
            GPUColumnType ct(GPUColumnTypeId::DECIMAL);
            ct.SetDecimalTypeInfo(
                static_cast<uint8_t>(precision),
                static_cast<uint8_t>(scale));
            return ct;
        }
        default:
            // Fallback: treat as INT64
            return GPUColumnType(GPUColumnTypeId::INT64);
    }
}

/// Convert a byte-per-row null mask (0=valid, 1=null) to cudf bitmask format.
/// cudf bitmask: bit=1 means valid, bit=0 means null. Packed as uint32_t words.
static cudf::bitmask_type* convert_null_mask_to_gpu(
    const uint8_t* host_mask,
    size_t num_rows,
    int gpu_id,
    GPUBufferManager& bm)
{
    // cudf bitmask: ceil(num_rows / 32) uint32_t words
    size_t num_words = (num_rows + 31) / 32;
    std::vector<cudf::bitmask_type> host_bitmask(num_words, 0);

    for (size_t i = 0; i < num_rows; ++i) {
        bool is_valid = (host_mask[i] == 0);
        if (is_valid) {
            host_bitmask[i / 32] |= (1u << (i % 32));
        }
    }

    auto* d_mask = bm.customCudaMalloc<cudf::bitmask_type>(num_words, gpu_id, false);
    callCudaMemcpyHostToDevice<cudf::bitmask_type>(
        d_mask, host_bitmask.data(), num_words, gpu_id);
    return d_mask;
}

void register_exchange_table(
    const BridgeContext& /*ctx*/,
    rust::Str table_name,
    uint32_t num_rows,
    rust::Slice<const ExchangeColumnDesc> columns,
    rust::Slice<const rust::Vec<uint8_t>> column_data,
    rust::Slice<const rust::Vec<uint8_t>> null_masks,
    rust::Slice<const rust::Vec<uint64_t>> string_offsets)
{
    if (columns.size() != column_data.size()) {
        throw std::runtime_error("register_exchange_table: columns/data size mismatch");
    }

    int gpu_id = 0;
    auto& bm = GPUBufferManager::GetInstance();

    // Table name must be uppercase for GPUBufferManager.
    std::string tname(table_name.data(), table_name.size());
    std::transform(tname.begin(), tname.end(), tname.begin(), ::toupper);

    size_t num_cols = columns.size();
    bm.createTable(tname, num_cols);

    auto& table = bm.tables[tname];
    table->names = tname;

    for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
        const auto& desc = columns[col_idx];
        const auto& data = column_data[col_idx];

        std::string col_name(desc.name.data(), desc.name.size());
        std::transform(col_name.begin(), col_name.end(), col_name.begin(), ::toupper);

        GPUColumnType gpu_type = map_ptype_to_gpu(desc.type_id, desc.precision, desc.scale);
        bool is_string = (gpu_type.id() == GPUColumnTypeId::VARCHAR);

        // Allocate and copy column data to GPU.
        size_t data_bytes = data.size();
        uint8_t* d_data = bm.customCudaMalloc<uint8_t>(data_bytes, gpu_id, false);
        callCudaMemcpyHostToDevice<uint8_t>(
            d_data, const_cast<uint8_t*>(data.data()), data_bytes, gpu_id);

        // Handle nullable columns.
        cudf::bitmask_type* d_mask = nullptr;
        if (desc.is_nullable && col_idx < null_masks.size() && !null_masks[col_idx].empty()) {
            d_mask = convert_null_mask_to_gpu(
                null_masks[col_idx].data(), num_rows, gpu_id, bm);
        }

        // Create GPUColumn.
        std::shared_ptr<GPUColumn> gpu_col;
        if (is_string && col_idx < string_offsets.size() && !string_offsets[col_idx].empty()) {
            // String column: copy offsets to GPU.
            size_t num_offsets = string_offsets[col_idx].size(); // N+1
            uint64_t* d_offsets = bm.customCudaMalloc<uint64_t>(num_offsets, gpu_id, false);
            callCudaMemcpyHostToDevice<uint64_t>(
                d_offsets,
                const_cast<uint64_t*>(string_offsets[col_idx].data()),
                num_offsets, gpu_id);

            gpu_col = std::make_shared<GPUColumn>(
                num_rows, gpu_type, d_data, d_offsets, data_bytes, true, d_mask);
        } else {
            gpu_col = std::make_shared<GPUColumn>(
                num_rows, gpu_type, d_data, d_mask);
        }

        table->column_names[col_idx] = col_name;
        table->columns[col_idx] = gpu_col;
    }
}

}  // namespace doris_bridge
