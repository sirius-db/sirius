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

}  // namespace doris_bridge
