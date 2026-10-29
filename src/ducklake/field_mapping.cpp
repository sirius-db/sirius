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

#include "ducklake/field_mapping.hpp"

#include "io/parquet_helpers.hpp"

#include <duckdb/common/multi_file/multi_file_reader.hpp>

#include <numeric>
#include <unordered_set>
namespace sirius::ducklake {
namespace scan = op::scan;
namespace {
bool internal_id(int32_t id)
{
  return id == duckdb::MultiFileReader::ROW_ID_FIELD_ID ||
         id == duckdb::MultiFileReader::LAST_UPDATED_SEQUENCE_NUMBER_ID;
}
}  // namespace
std::vector<size_t> required_field_indices(size_t bound_count, std::vector<size_t> selected)
{
  if (selected.empty()) {
    selected.resize(bound_count);
    std::iota(selected.begin(), selected.end(), 0);
  }
  std::unordered_set<size_t> seen;
  for (auto index : selected)
    if (index >= bound_count || !seen.insert(index).second)
      throw std::invalid_argument("DuckLake required column index is invalid or repeated");
  return selected;
}
std::optional<std::string> fixed_scalar_arrow_format(duckdb::LogicalType const& type)
{
  using D = duckdb::LogicalTypeId;
  switch (type.id()) {
    case D::BOOLEAN: return "b";
    case D::TINYINT: return "c";
    case D::SMALLINT: return "s";
    case D::INTEGER: return "i";
    case D::BIGINT: return "l";
    case D::UTINYINT: return "C";
    case D::USMALLINT: return "S";
    case D::UINTEGER: return "I";
    case D::UBIGINT: return "L";
    case D::FLOAT: return "f";
    case D::DOUBLE: return "g";
    case D::DATE: return "tdD";
    case D::TIMESTAMP: return "tsu:";
    case D::DECIMAL: {
      if (duckdb::DecimalType::GetWidth(type) < 5) return std::nullopt;
      auto mapped = scan::field_mapping_type(type);
      if (!mapped) return std::nullopt;
      return "d:" + std::to_string(duckdb::DecimalType::GetWidth(type)) + "," +
             std::to_string(duckdb::DecimalType::GetScale(type)) + "," +
             std::to_string(8 * cudf::size_of(*mapped));
    }
    default: return std::nullopt;
  }
}
scan::field_requirement bind_field(int32_t id,
                                   duckdb::LogicalType const& type,
                                   duckdb::Value const& initial_default)
{
  if (id <= 0 || internal_id(id) ||
      (!fixed_scalar_arrow_format(type) && type != duckdb::LogicalType::VARCHAR))
    throw std::invalid_argument("DuckLake field requires a qualified scalar type and user ID");
  scan::field_requirement result{id, type, std::nullopt};
  // Fixed scalars and exact VARCHAR values receive default approval. Values are
  // copied exactly; the two explicit widening pairs are shared mapper operations.
  result.allow_int32_to_int64  = type == duckdb::LogicalType::BIGINT;
  result.allow_float_to_double = type == duckdb::LogicalType::DOUBLE;
  if (initial_default.IsNull())
    result.missing_value = duckdb::Value(type);
  else if (initial_default.type() == type)
    result.missing_value = initial_default;
  else if (result.allow_int32_to_int64 && initial_default.type() == duckdb::LogicalType::INTEGER)
    result.missing_value = duckdb::Value::BIGINT(initial_default.GetValue<int32_t>());
  else if (result.allow_float_to_double && initial_default.type() == duckdb::LogicalType::FLOAT)
    result.missing_value = duckdb::Value::DOUBLE(initial_default.GetValue<float>());
  else
    throw std::invalid_argument(
      "DuckLake initial default requires an exact or qualified widening value");
  return result;
}
scan::field_id_mapping compile_field_mapping(std::shared_ptr<scan::parquet_metadata const> metadata,
                                             std::string const& identity,
                                             scan::materialization_program program,
                                             std::vector<scan::field_requirement> fields,
                                             bool partial_data)
{
  if (!metadata || !metadata->file_metadata())
    throw std::invalid_argument("DuckLake field mapping requires a footer");
  std::unordered_set<int32_t> bound_ids;
  for (auto const& field : fields) {
    bool snapshot = partial_data &&
                    field.id == duckdb::MultiFileReader::LAST_UPDATED_SEQUENCE_NUMBER_ID &&
                    field.type == duckdb::LogicalType::BIGINT && !field.missing_value &&
                    !field.allow_int32_to_int64 && !field.allow_float_to_double;
    if (field.id <= 0 || (internal_id(field.id) && !snapshot) || !bound_ids.insert(field.id).second)
      throw std::invalid_argument("DuckLake bound field IDs are invalid or duplicated");
    if ((field.allow_float_to_double && field.type != duckdb::LogicalType::DOUBLE) ||
        (field.allow_int32_to_int64 && field.type != duckdb::LogicalType::BIGINT) ||
        (field.missing_value &&
         ((!fixed_scalar_arrow_format(field.type) && field.type != duckdb::LogicalType::VARCHAR) ||
          field.missing_value->type() != field.type)))
      throw std::invalid_argument("DuckLake field mapping policy is not qualified");
  }
  if (partial_data && !bound_ids.contains(duckdb::MultiFileReader::LAST_UPDATED_SEQUENCE_NUMBER_ID))
    throw std::invalid_argument("DuckLake partial data requires the snapshot working column");
  std::vector<scan::field_requirement> internal_fields;
  auto const& schema = metadata->file_metadata()->schema;
  auto roots         = io::parquet_helpers::top_level_schema_indices(*metadata->file_metadata());
  std::unordered_set<size_t> root_set(roots.begin(), roots.end());
  for (auto root : roots)
    if (!schema[root].field_id)
      throw std::invalid_argument("DuckLake mapping requires identified top-level fields");
  for (size_t i = 1; i < schema.size(); ++i) {
    auto const& column = schema[i];
    if (!column.field_id) continue;  // LIST/MAP structural wrappers need not have IDs.
    auto id          = *column.field_id;
    char const* name = nullptr;
    if (id == duckdb::MultiFileReader::ROW_ID_FIELD_ID)
      name = "_ducklake_internal_row_id";
    else if (id == duckdb::MultiFileReader::LAST_UPDATED_SEQUENCE_NUMBER_ID)
      name = "_ducklake_internal_snapshot_id";
    if (name) {
      if (column.name != name || !root_set.contains(i) || column.num_children != 0)
        throw std::invalid_argument("DuckLake internal field identity differs from binding");
      internal_fields.push_back({id, duckdb::LogicalType::BIGINT, std::nullopt});
    }
  }
  // User names/order may change and dropped fields may remain physically present.
  // Only IDs determine their identity. Shared compilation checks duplicate IDs,
  // unambiguous physical names, types, annotations and typed missing defaults.
  if (!internal_fields.empty()) {
    auto internal_program = program;
    internal_program.input_types =
      std::vector<cudf::data_type>(internal_fields.size(), cudf::data_type{cudf::type_id::INT64});
    internal_program.output_types = *internal_program.input_types;
    (void)scan::field_id_mapping::compile(metadata,
                                          identity,
                                          internal_program,
                                          internal_fields,
                                          scan::field_mapping_scope::top_level_scalars);
  }
  return scan::field_id_mapping::compile(
    std::move(metadata),
    identity,
    std::move(program),
    std::move(fields),
    scan::field_mapping_scope::top_level_scalars,
    scan::row_count_carrier_policy::include_temporal_and_decimal);
}
bool add_partial_data_filter(unit_descriptor const& unit,
                             std::vector<scan::field_requirement>& fields,
                             unit_scan_options& options)
{
  validate_scan_options(options, fields.size());
  if (!unit.snapshot_min && !unit.snapshot_max) return false;
  if (!unit.snapshot_visibility_probe)
    throw std::invalid_argument("DuckLake partial data requires the snapshot visibility probe");
  if ((unit.snapshot_min && *unit.snapshot_min > INT64_MAX) ||
      (unit.snapshot_max && *unit.snapshot_max > INT64_MAX) ||
      (unit.snapshot_min && unit.snapshot_max && *unit.snapshot_min > *unit.snapshot_max))
    throw std::invalid_argument("DuckLake partial data snapshot bounds are invalid");
  auto column = fields.size();
  if (!options.output_columns) {
    options.output_columns.emplace();
    for (size_t i = 0; i < column; ++i)
      options.output_columns->push_back(i);
  }
  fields.push_back({duckdb::MultiFileReader::LAST_UPDATED_SEQUENCE_NUMBER_ID,
                    duckdb::LogicalType::BIGINT,
                    std::nullopt});
  if (!options.filters) options.filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  auto add = [&](uint64_t bound, duckdb::ExpressionType comparison) {
    options.filters->PushFilter(duckdb::ProjectionIndex(column),
                                duckdb::ExpressionFilter::CreateComparisonFilter(
                                  comparison, duckdb::Value::BIGINT(static_cast<int64_t>(bound))));
  };
  if (unit.snapshot_min)
    add(*unit.snapshot_min, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO);
  if (unit.snapshot_max) add(*unit.snapshot_max, duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO);
  return true;
}
}  // namespace sirius::ducklake
