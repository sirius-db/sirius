/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Implementation of the public FFI surface (sirius_ffi.hpp). This is the one
// translation unit that sees the heavy internal types, so consumers (e.g. the
// Rust bindings) never include sirius_context.hpp.

#include "sirius_ffi.hpp"

#include "core_functions_extension.hpp"                    // duckdb::CoreFunctionsExtension
#include "data/sirius_converter_registry.hpp"              // sirius::converter_registry
#include "duckdb/common/arrow/result_arrow_wrapper.hpp"    // duckdb::ResultArrowArrayStreamWrapper
#include "duckdb/common/enums/optimizer_type.hpp"          // duckdb::OptimizerType
#include "duckdb/execution/column_binding_resolver.hpp"    // duckdb::ColumnBindingResolver
#include "duckdb/main/client_context.hpp"                  // duckdb::ClientContext
#include "duckdb/main/config.hpp"                          // duckdb::DBConfig
#include "duckdb/main/connection.hpp"                      // duckdb::Connection
#include "duckdb/main/database.hpp"                        // duckdb::DuckDB
#include "duckdb/main/prepared_statement_data.hpp"         // duckdb::PreparedStatementData
#include "duckdb/main/query_result.hpp"                    // duckdb::QueryResult
#include "duckdb/main/relation.hpp"                        // duckdb::Relation
#include "duckdb/optimizer/optimizer.hpp"                  // duckdb::Optimizer
#include "duckdb/parser/statement/relation_statement.hpp"  // duckdb::RelationStatement
#include "duckdb/planner/planner.hpp"                      // duckdb::Planner
#include "from_substrait.hpp"     // duckdb::SubstraitToDuckDB (compiled into libsirius)
#include "parquet_extension.hpp"  // duckdb::ParquetExtension
#include "planner/sirius_physical_plan_generator.hpp"  // sirius::planner::sirius_physical_plan_generator
#include "planner/substrait_scan_ranges.hpp"           // sirius::planner::scan_byte_ranges_state
#include "sirius_config.hpp"                           // sirius::sirius_config
#include "sirius_context.hpp"                          // duckdb::SiriusContext
#include "sirius_interface.hpp"  // sirius::sirius_interface, sirius::sirius_prepared_statement_data

namespace sirius::ffi {

namespace {
// ClientContextState key the GPU engine resolves its SiriusContext under.
constexpr const char* kSiriusStateKey = "sirius_state";
constexpr const char* kQueryLabel     = "sirius_ffi";
// Arrow record-batch size for the exported stream; the consumer re-batches as needed.
constexpr duckdb::idx_t kArrowBatchSize = 1u << 20;
}  // namespace

// PIMPL: holds the engine + embedded DuckDB using DuckDB's own smart pointers
// (duckdb::shared_ptr, so the SiriusContext can register as a ClientContextState).
struct Context::Impl {
  duckdb::shared_ptr<duckdb::SiriusContext> context;
  duckdb::unique_ptr<duckdb::DuckDB> db;
  duckdb::unique_ptr<duckdb::Connection> conn;

  void bring_up(sirius::sirius_config& config)
  {
    context = duckdb::make_shared_ptr<duckdb::SiriusContext>();
    context->initialize(config);
    // Register the builtin + parquet representation converters the GPU scan/result
    // path needs. Idempotent; the transparent path does this at extension load.
    sirius::converter_registry::initialize();

    // Substrait lowering uses core functions and resolves local_files reads to parquet_scan.
    db = duckdb::make_uniq<duckdb::DuckDB>(nullptr);
    db->LoadStaticExtension<duckdb::CoreFunctionsExtension>();
    db->LoadStaticExtension<duckdb::ParquetExtension>();
    conn = duckdb::make_uniq<duckdb::Connection>(*db);
    // Register the engine on the connection and disable DuckDB optimizer rewrites this
    // no-fallback FFI path cannot safely execute. The transparent path only disables
    // IN_CLAUSE and COMPRESSED_MATERIALIZATION here; this path remains more conservative.
    auto& client = *conn->context;
    client.registered_state->Insert(kSiriusStateKey, context);
    // Per-connection Sirius state (guard depths, capture bookkeeping) for the
    // embedded connection, mirroring OnConnectionOpened on the transparent path.
    client.registered_state->Insert("sirius_connection_state",
                                    duckdb::make_shared_ptr<duckdb::SiriusConnectionState>());
    client.config.enable_optimizer = true;
    auto& disabled = duckdb::DBConfig::GetConfig(client).options.disabled_optimizers;
    disabled.insert(duckdb::OptimizerType::IN_CLAUSE);
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    // Keep this FFI-only restriction until its no-fallback execution path has
    // dedicated GPU_VALUES coverage.
    disabled.insert(duckdb::OptimizerType::STATISTICS_PROPAGATION);
    disabled.insert(duckdb::OptimizerType::COLUMN_LIFETIME);
    // Rewrites an ORDER BY ... LIMIT parquet scan into a semi-join on virtual
    // file_index/file_row_number columns the GPU scan drops.
    disabled.insert(duckdb::OptimizerType::LATE_MATERIALIZATION);
  }
};

Context::Context() : impl_(std::make_unique<Impl>())
{
  sirius::sirius_config config;
  config.apply_defaults();  // populate default GPU/host/disk memory spaces
  impl_->bring_up(config);
}

Context::Context(const std::string& config_path) : impl_(std::make_unique<Impl>())
{
  sirius::sirius_config config;
  config.load_from_file(config_path);  // throws on a missing/invalid config file
  impl_->bring_up(config);
}

// Defined here, where the heavy types are complete: destroying `impl_` tears down
// the embedded DuckDB and the initialized engine.
Context::~Context() = default;

void Context::execute_substrait(const std::string& plan, std::uintptr_t out_stream_addr)
{
  auto& client = *impl_->conn->context;

  // Binding (catalog lookups), optimization, and GPU execution all require an
  // active transaction. The transparent table-function path inherits one from the
  // enclosing query; this standalone entry has none, so open one explicitly. GPU
  // execution is eager (the result is materialized), so the transaction can close
  // before the Arrow stream is consumed.
  impl_->conn->BeginTransaction();
  duckdb::unique_ptr<duckdb::QueryResult> result;
  try {
    // 0. Byte-ranged parquet splits ride the plan's LocalFiles items, but DuckDB's Substrait
    //    consumer drops FileOrFiles.start/.length and parquet_scan has no byte-range
    //    parameter — extract them into a per-plan state the physical plan generator consumes.
    //    Always replaced (and removed when this plan carries none), so a stale registry can
    //    never leak a previous plan's ranges into this one.
    client.registered_state->Remove(sirius::planner::scan_byte_ranges_state::kStateKey);
    if (auto ranges = sirius::planner::extract_scan_byte_ranges(plan); !ranges.empty()) {
      client.registered_state->Insert(
        sirius::planner::scan_byte_ranges_state::kStateKey,
        duckdb::make_shared_ptr<sirius::planner::scan_byte_ranges_state>(std::move(ranges)));
    }

    // 1. Substrait bytes -> DuckDB Relation. DuckDB is used only for this lowering.
    duckdb::SubstraitToDuckDB transformer(impl_->conn->context, plan, /*json=*/false);
    auto relation = transformer.TransformPlan();

    // 2. Relation -> bound + optimized DuckDB LogicalOperator (mirrors the GPU bind path:
    //    SiriusTableFunctionData::ExtractPlan + GPUExecutionBind).
    duckdb::Planner planner(client);
    planner.CreatePlan(duckdb::make_uniq<duckdb::RelationStatement>(relation));

    auto prepared = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(
      duckdb::StatementType::SELECT_STATEMENT);
    prepared->names     = planner.names;
    prepared->types     = planner.types;
    prepared->value_map = std::move(planner.value_map);

    auto logical_plan = std::move(planner.plan);
    if (client.config.enable_optimizer) {
      duckdb::Optimizer optimizer(*planner.binder, client);
      logical_plan = optimizer.Optimize(std::move(logical_plan));
    }
    logical_plan->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    duckdb::ColumnBindingResolver::Verify(*logical_plan);
    resolver.VisitOperator(*logical_plan);

    // 3. DuckDB LogicalOperator -> Sirius GPU physical plan -> execute directly
    // on the engine, inside an execution window: begin mutations and slot
    // acquire in the constructor, mandatory cleanup and release in finish().
    // This standalone path bypasses DuckDB's normal query entry point, so
    // nothing else would clean up for it. (The old manual QueryBegin/QueryEnd
    // pairing could call QueryEnd twice when the first cleanup threw.)
    {
      duckdb::SiriusContext::StandaloneQueryScope window(*impl_->context, client, kQueryLabel);
      auto physical_plan = sirius::planner::sirius_physical_plan_generator(client).create_plan(
        std::move(logical_plan));
      auto gpu_prepared = duckdb::make_shared_ptr<sirius::sirius_prepared_statement_data>(
        std::move(prepared), std::move(physical_plan));

      sirius::sirius_interface iface(client, std::optional<std::string>(kQueryLabel));
      result = iface.sirius_execute_query(
        client, kQueryLabel, gpu_prepared, duckdb::PendingQueryParameters{}, window.query_id());
      window.finish();
    }
  } catch (...) {
    impl_->conn->Rollback();
    throw;
  }
  impl_->conn->Commit();
  if (result->HasError()) { result->ThrowError(); }

  // 4. Hand the result to the caller as a self-owning Arrow C Data Interface stream,
  //    written into the caller's ArrowArrayStream (addressed by `out_stream_addr`);
  //    its `release` callback deletes the heap wrapper (ResultArrowArrayStreamWrapper).
  auto* wrapper = new duckdb::ResultArrowArrayStreamWrapper(std::move(result), kArrowBatchSize);
  *reinterpret_cast<ArrowArrayStream*>(out_stream_addr) = wrapper->stream;
}

std::unique_ptr<Context> make_context() { return std::make_unique<Context>(); }

std::unique_ptr<Context> make_context_from_config(const std::string& config_path)
{
  return std::make_unique<Context>(config_path);
}

}  // namespace sirius::ffi
