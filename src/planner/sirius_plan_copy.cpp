// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_parquet_sink.hpp"
#include "output/parquet_writer.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_context.hpp"

#include <duckdb/common/file_system.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/parser/parsed_data/copy_info.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/parser/statement/copy_statement.hpp>
#include <duckdb/parser/statement/prepare_statement.hpp>
#include <duckdb/planner/operator/logical_copy_to_file.hpp>

namespace sirius::planner {
namespace {
output::parquet_field export_field(std::string name, duckdb::LogicalType const& type)
{
  using T = duckdb::LogicalTypeId;
  output::parquet_field field;
  field.name = std::move(name);
  // Do not turn HUGEINT into a Parquet DECIMAL silently: the SQL type would change.
  if (type.id() == T::HUGEINT || type.id() == T::UHUGEINT || type.id() == T::SQLNULL)
    throw duckdb::NotImplementedException("Sirius Parquet export cannot preserve type " +
                                          type.ToString());
  field.type = sirius::from_duckdb(type);
  if (type.id() == T::DECIMAL) field.decimal_precision = duckdb::DecimalType::GetWidth(type);
  if (duckdb::StructType::IsStruct(type)) {
    // row(...) produces TUPLE; DuckDB's Parquet writer assigns positional
    // element names through the same helper used here.
    for (auto const& [child_name, child_type] : duckdb::TupleType::NamedChildren(type))
      field.children.push_back(export_field(child_name.GetIdentifierName(), child_type));
  } else if (type.id() == T::LIST || type.id() == T::ARRAY) {
    field.children.push_back(export_field("element",
                                          type.id() == T::LIST
                                            ? duckdb::ListType::GetChildType(type)
                                            : duckdb::ArrayType::GetChildType(type)));
  } else if (type.id() == T::MAP) {
    // Native MAP metadata needs a nonnullable key hierarchy; decline until that
    // schema contract is represented and validated at the frontend boundary.
    throw duckdb::NotImplementedException("Sirius Parquet export does not yet support MAP");
  }
  return field;
}
}  // namespace
duckdb::unique_ptr<op::sirius_physical_operator> sirius_physical_plan_generator::create_plan(
  duckdb::LogicalCopyToFile& copy)
{
  if (!copy.copy_info || copy.copy_info->is_from ||
      duckdb::StringUtil::Lower(copy.copy_info->format) != "parquet")
    throw duckdb::NotImplementedException("Sirius COPY supports only Parquet output");
  if (!copy.per_thread_output)
    throw duckdb::NotImplementedException(
      "Sirius Parquet COPY requires PER_THREAD_OUTPUT true (directory output)");
  // Binding consumes generic COPY options (including ROW_GROUP_SIZE and
  // FILENAME_PATTERN). Validate their original names before discarding the
  // frontend representation, rather than mistaking normalized defaults for SQL.
  duckdb::Parser parser(context);
  parser.ParseQuery(copy_statement_sql.empty() ? context.GetCurrentQuery() : copy_statement_sql);
  if (parser.statements.size() != 1)
    throw duckdb::NotImplementedException(
      "Sirius COPY requires an identifiable original statement");
  auto* original = parser.statements.front().get();
  if (original->type == duckdb::StatementType::PREPARE_STATEMENT)
    original = original->Cast<duckdb::PrepareStatement>().statement.get();
  if (!original || original->type != duckdb::StatementType::COPY_STATEMENT)
    throw duckdb::NotImplementedException("Sirius COPY cannot validate original output options");
  auto const& original_info     = *original->Cast<duckdb::CopyStatement>().info;
  auto require_supported_option = [](duckdb::Identifier const& name) {
    auto key = duckdb::StringUtil::Lower(name.GetIdentifierName());
    if (key != "format" && key != "compression" && key != "per_thread_output")
      throw duckdb::NotImplementedException("Sirius Parquet COPY does not support option: " + key);
  };
  for (auto const& [name, value] : original_info.parsed_options)
    require_supported_option(name);
  for (auto const& [name, value] : original_info.options)
    require_supported_option(name);
  for (auto const& [name, values] : copy.copy_info->options) {
    if (duckdb::StringUtil::Lower(name.GetIdentifierName()) != "compression")
      throw duckdb::NotImplementedException("Sirius Parquet COPY does not support option: " +
                                            name.GetIdentifierName());
  }
  // DuckDB fills batch_size from the CPU writer even when SQL omits it.
  // Explicit sizing options were rejected above; the Sirius sink uses arriving batches.
  if (copy.partition_output || copy.rotate || copy.batch_size_bytes.IsValid() ||
      copy.file_size_bytes.IsValid() || copy.batches_per_file.IsValid() ||
      copy.return_type != duckdb::CopyFunctionReturnType::CHANGED_ROWS ||
      copy.overwrite_mode != duckdb::CopyOverwriteMode::COPY_ERROR_ON_CONFLICT)
    throw duckdb::NotImplementedException(
      "Sirius Parquet COPY supports batch-sized parts in a directory without existing files only");
  if (copy.file_path.empty() || copy.file_path.find("://") != std::string::npos ||
      copy.file_path == "/dev/stdout")
    throw duckdb::NotImplementedException("Sirius Parquet COPY supports local directories only");
  output::parquet_export_spec spec;
  spec.directory =
    std::filesystem::absolute(duckdb::FileSystem::GetFileSystem(context).ExpandPath(copy.file_path))
      .lexically_normal();
  auto runtime     = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  spec.compression = runtime->get_config().parquet_output_compression();
  if (auto option = copy.copy_info->options.find("compression");
      option != copy.copy_info->options.end()) {
    if (option->second.size() != 1)
      throw duckdb::NotImplementedException("Invalid Parquet compression option");
    try {
      spec.compression = output::parse_compression(option->second[0].ToString());
    } catch (std::invalid_argument const& error) {
      throw duckdb::NotImplementedException(error.what());
    }
  }
  if (!output::supported_parquet_compression(spec.compression))
    throw duckdb::NotImplementedException("cuDF Parquet writer does not support codec " +
                                          output::compression_name(spec.compression));
  for (std::size_t i = 0; i < copy.expected_types.size(); ++i)
    spec.fields.push_back(export_field(copy.names[i].GetIdentifierName(), copy.expected_types[i]));
  auto child = create_plan(*copy.children.at(0));
  auto sink  = duckdb::make_uniq<op::sirius_physical_parquet_sink>(
    sirius::from_duckdb_vec(copy.expected_types), std::move(spec));
  sink->children.push_back(std::move(child));
  return sink;
}
}  // namespace sirius::planner
