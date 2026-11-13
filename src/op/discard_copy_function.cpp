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

#include "op/discard_copy_function.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/enums/copy_overwrite_mode.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"
#include "duckdb/parser/parsed_data/create_copy_function_info.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_copy_to_file.hpp"

#include <utility>

namespace sirius {
namespace op {

namespace {

constexpr const char* kCheckOrderOption = "check_order";

bool parse_boolean_option(const duckdb::Identifier& name,
                          const duckdb::vector<duckdb::Value>& values)
{
  // A bare option (`check_order`) means true, as for DuckDB's own boolean COPY options.
  if (values.empty()) { return true; }
  if (values.size() == 1 && !values[0].IsNull()) {
    if (auto value = values[0].DefaultTryCastAs(duckdb::LogicalType::BOOLEAN)) {
      return value->GetValue<bool>();
    }
  }
  throw duckdb::BinderException("COPY option %s of FORMAT %s expects a single BOOLEAN argument",
                                name.GetIdentifierName(),
                                kDiscardCopyFormat);
}

//! Bind data from the COPY options. Every option other than `check_order` is rejected, DuckDB's
//! generic file options (PARTITION_BY, PER_THREAD_OUTPUT, ...) included: they make DuckDB create
//! files or directories, and this format has no output.
duckdb::unique_ptr<discard_copy_bind_data> bind_discard_options(
  const duckdb::CopyInfo& info, const duckdb::vector<duckdb::LogicalType>& types)
{
  auto result   = duckdb::make_uniq<discard_copy_bind_data>();
  result->types = types;
  for (auto& [name, values] : info.options) {
    if (name == kCheckOrderOption) {
      result->check_order = parse_boolean_option(name, values);
      continue;
    }
    throw duckdb::BinderException(
      "Unrecognized option %s for FORMAT %s (the only option is check_order)",
      name.GetIdentifierName(),
      kDiscardCopyFormat);
  }
  if (info.parsed_options.find("partition_path") != info.parsed_options.end()) {
    throw duckdb::BinderException("PARTITION_PATH is not supported for FORMAT %s",
                                  kDiscardCopyFormat);
  }
  return result;
}

//! Binds the whole COPY instead of DuckDB's generic COPY TO binder. That binder writes to a
//! temporary file and renames it when the target already exists as a file, which fails for a
//! format that writes nothing; here the output settings are fixed so DuckDB touches no path.
duckdb::BoundStatement discard_copy_plan(duckdb::Binder& binder, duckdb::CopyStatement& stmt)
{
  auto node_copy   = stmt.info->select_statement->Copy();
  auto select_node = binder.Bind(*node_copy);
  auto bind_data   = bind_discard_options(*stmt.info, select_node.types);

  auto names = select_node.names;
  duckdb::QueryResult::DeduplicateColumns(names);
  auto file_path = stmt.info->file_path;

  auto copy            = duckdb::make_uniq<duckdb::LogicalCopyToFile>(get_discard_copy_function(),
                                                           std::move(bind_data),
                                                           std::move(stmt.info),
                                                           binder.GenerateTableIndex());
  copy->file_path      = std::move(file_path);
  copy->use_tmp_file   = false;
  copy->overwrite_mode = duckdb::CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;
  copy->per_thread_output       = false;
  copy->rotate                  = false;
  copy->return_type             = duckdb::CopyFunctionReturnType::CHANGED_ROWS;
  copy->partition_output        = false;
  copy->write_partition_columns = false;
  copy->names                   = std::move(names);
  copy->expected_types          = select_node.types;
  copy->AddChild(std::move(select_node.plan));

  duckdb::BoundStatement result;
  result.names = duckdb::GetCopyFunctionReturnNames(copy->return_type);
  result.types = duckdb::GetCopyFunctionReturnLogicalTypes(copy->return_type);
  result.plan  = std::move(copy);
  return result;
}

duckdb::unique_ptr<duckdb::FunctionData> discard_copy_bind(
  duckdb::ClientContext& /*context*/,
  duckdb::CopyFunctionBindInput& input,
  const duckdb::vector<duckdb::Identifier>& /*names*/,
  const duckdb::vector<duckdb::LogicalType>& sql_types)
{
  return bind_discard_options(input.info, sql_types);
}

// The CPU body. DuckDB's COPY operator counts the rows it hands to the sink and returns that
// count, so the callbacks below only have to drop the data. No state is kept and the file path
// is ignored.
duckdb::unique_ptr<duckdb::GlobalFunctionData> discard_copy_initialize_global(
  duckdb::ClientContext& /*context*/,
  duckdb::FunctionData& /*bind_data*/,
  const duckdb::string& /*file_path*/)
{
  return duckdb::make_uniq<duckdb::GlobalFunctionData>();
}

duckdb::unique_ptr<duckdb::LocalFunctionData> discard_copy_initialize_local(
  duckdb::ExecutionContext& /*context*/, duckdb::FunctionData& /*bind_data*/)
{
  return duckdb::make_uniq<duckdb::LocalFunctionData>();
}

void discard_copy_sink(duckdb::ExecutionContext& /*context*/,
                       duckdb::FunctionData& /*bind_data*/,
                       duckdb::GlobalFunctionData& /*gstate*/,
                       duckdb::LocalFunctionData& /*lstate*/,
                       duckdb::DataChunk& /*input*/)
{
}

void discard_copy_combine(duckdb::ExecutionContext& /*context*/,
                          duckdb::FunctionData& /*bind_data*/,
                          duckdb::GlobalFunctionData& /*gstate*/,
                          duckdb::LocalFunctionData& /*lstate*/)
{
}

void discard_copy_finalize(duckdb::ClientContext& /*context*/,
                           duckdb::FunctionData& /*bind_data*/,
                           duckdb::GlobalFunctionData& /*gstate*/)
{
}

// Nothing is written, so there is no row order to keep: every thread may sink at once.
duckdb::CopyFunctionExecutionMode discard_copy_execution_mode(bool /*preserve_insertion_order*/,
                                                              bool /*supports_batch_index*/)
{
  return duckdb::CopyFunctionExecutionMode::PARALLEL_COPY_TO_FILE;
}

}  // namespace

duckdb::CopyFunction get_discard_copy_function()
{
  duckdb::CopyFunction function{duckdb::Identifier(kDiscardCopyFormat)};
  function.plan                      = discard_copy_plan;
  function.copy_to_bind              = discard_copy_bind;
  function.copy_to_initialize_global = discard_copy_initialize_global;
  function.copy_to_initialize_local  = discard_copy_initialize_local;
  function.copy_to_sink              = discard_copy_sink;
  function.copy_to_combine           = discard_copy_combine;
  function.copy_to_finalize          = discard_copy_finalize;
  function.execution_mode            = discard_copy_execution_mode;
  return function;
}

void register_discard_copy_function(duckdb::DatabaseInstance& instance)
{
  auto transaction = duckdb::CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = duckdb::Catalog::GetSystemCatalog(instance);

  duckdb::CreateCopyFunctionInfo info(get_discard_copy_function());
  // Idempotent: extension callback and explicit callers may both register.
  info.on_conflict = duckdb::OnCreateConflict::IGNORE_ON_CONFLICT;
  catalog.CreateCopyFunction(transaction, info);
}

}  // namespace op
}  // namespace sirius
