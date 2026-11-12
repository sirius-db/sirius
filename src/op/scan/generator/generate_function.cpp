/*
 * Copyright 2026, Sirius Contributors.
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

#include "op/scan/generator/generate_function.hpp"

#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>  // IWYU pragma: keep
#include <duckdb/common/exception.hpp>
#include <duckdb/common/exception/binder_exception.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <duckdb/storage/statistics/node_statistics.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace sirius::op::scan::generator {

namespace {

constexpr const char* kValidDistributions =
  "uniform, normal, zipf, sequential, sequential_desc, few_distinct";

duckdb::LogicalType columns_parameter_type()
{
  duckdb::child_list_t<duckdb::LogicalType> fields;
  fields.emplace_back("name", duckdb::LogicalType::VARCHAR);
  fields.emplace_back("type", duckdb::LogicalType::VARCHAR);
  fields.emplace_back("dist", duckdb::LogicalType::VARCHAR);
  fields.emplace_back("p1", duckdb::LogicalType::DOUBLE);
  fields.emplace_back("p2", duckdb::LogicalType::DOUBLE);
  fields.emplace_back("cardinality", duckdb::LogicalType::DOUBLE);
  fields.emplace_back("null_fraction", duckdb::LogicalType::DOUBLE);
  return duckdb::LogicalType::LIST(duckdb::LogicalType::STRUCT(std::move(fields)));
}

/// The named parameter, or nullptr when it is absent or NULL.
const duckdb::Value* find_parameter(const duckdb::TableFunctionBindInput& input, const char* name)
{
  auto it = input.named_parameters.find(name);
  if (it == input.named_parameters.end() || it->second.IsNull()) { return nullptr; }
  return &it->second;
}

std::uint64_t required_count(const duckdb::TableFunctionBindInput& input, const char* name)
{
  const auto* value = find_parameter(input, name);
  if (value == nullptr) {
    throw duckdb::BinderException(std::string("sirius_generate requires a '") + name +
                                  "' named parameter");
  }
  auto const count = value->GetValue<std::uint64_t>();
  if (count == 0) {
    throw duckdb::BinderException(std::string("sirius_generate: '") + name +
                                  "' must be greater than 0");
  }
  return count;
}

value_type parse_value_type(const std::string& column, const std::string& type_name)
{
  auto const upper = duckdb::StringUtil::Upper(type_name);
  if (upper == "INTEGER" || upper == "INT" || upper == "INT4") { return value_type::int32; }
  if (upper == "BIGINT" || upper == "INT8") { return value_type::int64; }
  if (upper == "DOUBLE" || upper == "FLOAT8") { return value_type::float64; }
  if (upper == "VARCHAR" || upper == "TEXT" || upper == "STRING") { return value_type::varchar; }
  throw duckdb::BinderException("sirius_generate: column '" + column + "' has unsupported type '" +
                                type_name + "' (expected INTEGER, BIGINT, DOUBLE or VARCHAR)");
}

duckdb::LogicalType to_logical_type(value_type type)
{
  switch (type) {
    case value_type::int32: return duckdb::LogicalType::INTEGER;
    case value_type::int64: return duckdb::LogicalType::BIGINT;
    case value_type::float64: return duckdb::LogicalType::DOUBLE;
    case value_type::varchar: return duckdb::LogicalType::VARCHAR;
  }
  throw duckdb::InternalException("sirius_generate: unknown value_type");
}

[[noreturn]] void invalid_column(const std::string& column, const std::string& problem)
{
  throw duckdb::BinderException("sirius_generate: column '" + column + "': " + problem);
}

/// One entry of `columns`. Fields are matched by name, so this works for the declared struct
/// type and for any struct a caller's literal was cast from.
column_spec parse_column(const duckdb::Value& entry, std::size_t index)
{
  if (entry.IsNull() || entry.type().id() != duckdb::LogicalTypeId::STRUCT) {
    throw duckdb::BinderException("sirius_generate: 'columns' entry " + std::to_string(index) +
                                  " must be a non-NULL struct");
  }
  auto const& fields = duckdb::StructType::GetChildTypes(entry.type());
  auto const& values = duckdb::StructValue::GetChildren(entry);
  auto field         = [&](const char* name) -> const duckdb::Value* {
    for (std::size_t i = 0; i < fields.size(); ++i) {
      if (fields[i].first == name) { return values[i].IsNull() ? nullptr : &values[i]; }
    }
    return nullptr;
  };

  column_spec column;
  if (const auto* name = field("name")) { column.name = name->ToString(); }
  if (column.name.empty()) {
    throw duckdb::BinderException("sirius_generate: 'columns' entry " + std::to_string(index) +
                                  " requires a non-empty 'name'");
  }

  const auto* type = field("type");
  if (type == nullptr) { invalid_column(column.name, "'type' is required"); }
  column.type = parse_value_type(column.name, type->ToString());

  const auto* dist = field("dist");
  if (dist == nullptr) { invalid_column(column.name, "'dist' is required"); }
  auto const dist_name = dist->ToString();
  auto const parsed    = parse_distribution(duckdb::StringUtil::Lower(dist_name));
  if (!parsed) {
    invalid_column(column.name,
                   "unknown dist '" + dist_name + "' (valid: " + kValidDistributions + ")");
  }
  column.dist = *parsed;

  auto number = [&](const char* name) {
    const auto* value = field(name);
    auto const result = value == nullptr ? 0.0 : value->GetValue<double>();
    if (!std::isfinite(result)) {
      invalid_column(column.name, std::string("'") + name + "' must be finite");
    }
    return result;
  };
  column.p1            = number("p1");
  column.p2            = number("p2");
  column.cardinality   = number("cardinality");
  column.null_fraction = number("null_fraction");
  return column;
}

void validate_column(const column_spec& column)
{
  if (column.null_fraction < 0 || column.null_fraction > 1) {
    invalid_column(column.name, "null_fraction must be in [0, 1]");
  }
  if (column.cardinality < 0) { invalid_column(column.name, "cardinality must not be negative"); }
  switch (column.dist) {
    case distribution::uniform:
      if (column.p1 > column.p2) {
        invalid_column(column.name, "uniform requires p1 (low) <= p2 (high)");
      }
      break;
    case distribution::normal:
      if (column.p2 < 0) { invalid_column(column.name, "normal requires p2 (stddev) >= 0"); }
      break;
    case distribution::zipf:
      if (column.p1 <= 0) { invalid_column(column.name, "zipf requires p1 (exponent) > 0"); }
      if (column.cardinality < 1) { invalid_column(column.name, "zipf requires cardinality >= 1"); }
      break;
    case distribution::few_distinct:
      if (column.p1 < 1) {
        invalid_column(column.name, "few_distinct requires p1 (distinct values) >= 1");
      }
      break;
    case distribution::sequential:
    case distribution::sequential_desc: break;
  }
  if (column.type == value_type::varchar && (column.p1 < 1 || column.p1 > column.p2)) {
    invalid_column(column.name, "VARCHAR requires 1 <= p1 (min length) <= p2 (max length)");
  }
}

generator_spec parse_spec(const duckdb::TableFunctionBindInput& input)
{
  generator_spec spec;
  spec.num_batches    = required_count(input, "num_batches");
  spec.rows_per_batch = required_count(input, "rows_per_batch");
  // One batch is one cudf table, whose row count is a cudf::size_type (int32).
  constexpr auto max_rows_per_batch =
    static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  if (spec.rows_per_batch > max_rows_per_batch) {
    throw duckdb::BinderException("sirius_generate: 'rows_per_batch' must not exceed " +
                                  std::to_string(max_rows_per_batch));
  }
  if (spec.num_batches > std::numeric_limits<std::uint64_t>::max() / spec.rows_per_batch) {
    throw duckdb::BinderException(
      "sirius_generate: num_batches * rows_per_batch overflows a 64-bit row count");
  }
  if (const auto* seed = find_parameter(input, "seed")) {
    spec.seed = seed->GetValue<std::uint64_t>();
  }

  const auto* columns = find_parameter(input, "columns");
  if (columns == nullptr) {
    throw duckdb::BinderException("sirius_generate requires a 'columns' named parameter");
  }
  if (columns->type().id() != duckdb::LogicalTypeId::LIST) {
    throw duckdb::BinderException("sirius_generate: 'columns' must be a list of structs");
  }
  auto const& entries = duckdb::ListValue::GetChildren(*columns);
  if (entries.empty()) {
    throw duckdb::BinderException("sirius_generate: 'columns' must not be empty");
  }
  spec.columns.reserve(entries.size());
  for (std::size_t i = 0; i < entries.size(); ++i) {
    auto column = parse_column(entries[i], i);
    validate_column(column);
    for (auto const& earlier : spec.columns) {
      if (duckdb::StringUtil::CIEquals(earlier.name, column.name)) {
        invalid_column(column.name, "duplicate column name");
      }
    }
    spec.columns.push_back(std::move(column));
  }
  return spec;
}

/// The bind callback, with the signature DuckDB's table_function_bind_t requires.
duckdb::unique_ptr<duckdb::FunctionData> generate_bind(
  duckdb::ClientContext&,
  duckdb::TableFunctionBindInput& input,
  duckdb::vector<duckdb::LogicalType>& return_types,
  duckdb::vector<duckdb::Identifier>& names)
{
  auto result  = duckdb::make_uniq<GenerateFunctionData>();
  result->spec = parse_spec(input);

  return_types.clear();
  names.clear();
  for (auto const& column : result->spec.columns) {
    return_types.push_back(to_logical_type(column.type));
    names.emplace_back(column.name);
  }
  return result;
}

duckdb::unique_ptr<duckdb::NodeStatistics> generate_cardinality(
  duckdb::ClientContext&, const duckdb::FunctionData* bind_data)
{
  if (bind_data == nullptr) { return nullptr; }
  auto const& spec = bind_data->Cast<GenerateFunctionData>().spec;
  // Exact: bind rejected a product that overflows.
  auto const rows = spec.num_batches * spec.rows_per_batch;
  return duckdb::make_uniq<duckdb::NodeStatistics>(rows, rows);
}

/// Never runs under Sirius: the plan generator replaces this scan with a GENERATOR source.
void generate_function(duckdb::ClientContext&, duckdb::TableFunctionInput&, duckdb::DataChunk&)
{
  throw duckdb::InvalidInputException(
    "sirius_generate requires Sirius GPU execution (SET gpu_execution = true)");
}

}  // namespace

duckdb::unique_ptr<duckdb::FunctionData> GenerateBind(
  duckdb::ClientContext& context,
  duckdb::TableFunctionBindInput& input,
  duckdb::vector<duckdb::LogicalType>& return_types,
  duckdb::vector<duckdb::string>& names)
{
  duckdb::vector<duckdb::Identifier> identifiers;
  auto result = generate_bind(context, input, return_types, identifiers);
  names.clear();
  for (auto const& identifier : identifiers) {
    names.push_back(identifier.GetIdentifierName());
  }
  return result;
}

duckdb::TableFunction get_generate_function()
{
  duckdb::TableFunction generate(kGenerateFunctionName, {}, generate_function, generate_bind);
  generate.GetSignature().WithTypedKwargs("options", [](duckdb::TypedKwargs& options) {
    options.Add("num_batches", duckdb::LogicalType::UBIGINT);
    options.Add("rows_per_batch", duckdb::LogicalType::UBIGINT);
    options.Add("seed", duckdb::LogicalType::UBIGINT);
    options.Add("columns", columns_parameter_type());
  });
  // Set here, not at registration: the scan contract compares a planned scan's callbacks,
  // cardinality included, against this definition.
  generate.cardinality         = generate_cardinality;
  generate.projection_pushdown = false;
  generate.filter_pushdown     = false;
  return generate;
}

void register_generate_function(duckdb::DatabaseInstance& instance)
{
  auto transaction = duckdb::CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = duckdb::Catalog::GetSystemCatalog(instance);

  auto generate = get_generate_function();

  duckdb::CreateTableFunctionInfo info(generate);
  // Idempotent: extension callback and explicit callers may both register.
  info.on_conflict = duckdb::OnCreateConflict::IGNORE_ON_CONFLICT;
  catalog.CreateTableFunction(transaction, info);
}

}  // namespace sirius::op::scan::generator
