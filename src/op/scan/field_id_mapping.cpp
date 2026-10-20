/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/field_id_mapping.hpp"

#include "io/parquet_helpers.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace sirius::op::scan {
std::unordered_map<int32_t, std::size_t> index_parquet_field_ids(
  cudf::io::parquet::FileMetaData const& metadata)
{
  std::unordered_map<int32_t, std::size_t> result;
  for (std::size_t i = 1; i < metadata.schema.size(); ++i) {
    auto const& element = metadata.schema[i];
    if (!element.field_id) continue;
    if (*element.field_id <= 0 || !result.emplace(*element.field_id, i).second)
      throw std::invalid_argument("Parquet has an invalid or duplicate field ID");
  }
  return result;
}

namespace {
cudf::data_type flat_type(duckdb::LogicalType const& type)
{
  using D = duckdb::LogicalTypeId;
  using C = cudf::type_id;
  switch (type.id()) {
    case D::BOOLEAN: return cudf::data_type{C::BOOL8};
    case D::TINYINT: return cudf::data_type{C::INT8};
    case D::SMALLINT: return cudf::data_type{C::INT16};
    case D::INTEGER: return cudf::data_type{C::INT32};
    case D::BIGINT: return cudf::data_type{C::INT64};
    case D::UTINYINT: return cudf::data_type{C::UINT8};
    case D::USMALLINT: return cudf::data_type{C::UINT16};
    case D::UINTEGER: return cudf::data_type{C::UINT32};
    case D::UBIGINT: return cudf::data_type{C::UINT64};
    case D::FLOAT: return cudf::data_type{C::FLOAT32};
    case D::DOUBLE: return cudf::data_type{C::FLOAT64};
    case D::VARCHAR: return cudf::data_type{C::STRING};
    default: throw std::invalid_argument("field_id_core_v1: unsupported logical type");
  }
}

bool same_default(duckdb::Value const& a, duckdb::Value const& b)
{
  if (a.IsNull() || b.IsNull()) return a.IsNull() && b.IsNull();
  // SQL equality merges +0 and -0, whose sign remains observable (e.g. 1/x).
  // A coalesced run must preserve the actual default, including NaN payloads.
  if (a.type().id() == duckdb::LogicalTypeId::FLOAT)
    return std::bit_cast<uint32_t>(a.GetValue<float>()) ==
           std::bit_cast<uint32_t>(b.GetValue<float>());
  if (a.type().id() == duckdb::LogicalTypeId::DOUBLE)
    return std::bit_cast<uint64_t>(a.GetValue<double>()) ==
           std::bit_cast<uint64_t>(b.GetValue<double>());
  return duckdb::Value::NotDistinctFrom(a, b);
}

std::unique_ptr<cudf::scalar> default_scalar(duckdb::Value const& value,
                                             cudf::data_type type,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  if (value.IsNull()) {
    auto result = cudf::make_default_constructed_scalar(type, stream, mr);
    result->set_valid_async(false, stream);
    return result;
  }
  switch (type.id()) {
    case cudf::type_id::BOOL8:
      return std::make_unique<cudf::numeric_scalar<bool>>(value.GetValue<bool>(), true, stream, mr);
    case cudf::type_id::INT8:
      return std::make_unique<cudf::numeric_scalar<int8_t>>(
        value.GetValue<int8_t>(), true, stream, mr);
    case cudf::type_id::INT16:
      return std::make_unique<cudf::numeric_scalar<int16_t>>(
        value.GetValue<int16_t>(), true, stream, mr);
    case cudf::type_id::INT32:
      return std::make_unique<cudf::numeric_scalar<int32_t>>(
        value.GetValue<int32_t>(), true, stream, mr);
    case cudf::type_id::INT64:
      return std::make_unique<cudf::numeric_scalar<int64_t>>(
        value.GetValue<int64_t>(), true, stream, mr);
    case cudf::type_id::UINT8:
      return std::make_unique<cudf::numeric_scalar<uint8_t>>(
        value.GetValue<uint8_t>(), true, stream, mr);
    case cudf::type_id::UINT16:
      return std::make_unique<cudf::numeric_scalar<uint16_t>>(
        value.GetValue<uint16_t>(), true, stream, mr);
    case cudf::type_id::UINT32:
      return std::make_unique<cudf::numeric_scalar<uint32_t>>(
        value.GetValue<uint32_t>(), true, stream, mr);
    case cudf::type_id::UINT64:
      return std::make_unique<cudf::numeric_scalar<uint64_t>>(
        value.GetValue<uint64_t>(), true, stream, mr);
    case cudf::type_id::FLOAT32:
      return std::make_unique<cudf::numeric_scalar<float>>(
        value.GetValue<float>(), true, stream, mr);
    case cudf::type_id::FLOAT64:
      return std::make_unique<cudf::numeric_scalar<double>>(
        value.GetValue<double>(), true, stream, mr);
    case cudf::type_id::STRING:
      return std::make_unique<cudf::string_scalar>(value.GetValue<std::string>(), true, stream, mr);
    default: throw std::invalid_argument("unsupported field default");
  }
}
}  // namespace

std::optional<cudf::data_type> field_mapping_type(duckdb::LogicalType const& type)
{
  try {
    return flat_type(type);
  } catch (std::invalid_argument const&) {
    return std::nullopt;
  }
}

field_id_mapping field_id_mapping::compile(std::shared_ptr<parquet_metadata const> metadata,
                                           std::string const& opened_identity,
                                           materialization_program program,
                                           std::vector<field_requirement> fields)
{
  if (!metadata || opened_identity.empty() || metadata->built_for_identity != opened_identity ||
      !program.contract || !program.generation)
    throw std::invalid_argument("field mapping has no matching source or program identity");
  auto footer = metadata->file_metadata();
  if (!footer || footer->schema.empty())
    throw std::invalid_argument("missing field mapping footer");
  auto index = index_parquet_field_ids(*footer);
  auto schema =
    io::parquet_helpers::extract_schema(*footer, true, metadata->original_logical_annotations);
  // v1 qualifies flat schemas only, without narrowing any legacy profile.
  if (footer->schema.size() != schema.names.size() + 1)
    throw std::invalid_argument("field_id_core_v1 requires a flat schema");
  std::unordered_set<std::string> physical_names;
  for (auto const& name : schema.names)
    if (!physical_names.insert(name).second)
      throw std::invalid_argument("field ID has an ambiguous physical reader address");
  field_id_mapping result;
  result.metadata_ = std::move(metadata);
  result.program_  = std::move(program);
  std::unordered_map<std::size_t, std::size_t> read_slots;
  for (auto& field : fields) {
    if (field.id <= 0) throw std::invalid_argument("invalid bound field ID");
    auto target = flat_type(field.type);
    operation op{std::move(field), {}, target, target};
    auto found = index.find(op.field.id);
    if (found == index.end()) {
      if (!op.field.missing_value || op.field.missing_value->type() != op.field.type)
        throw std::invalid_argument("missing field requires an explicitly typed default");
    } else {
      auto ordinal     = found->second - 1;
      op.physical_type = flat_type(schema.types.at(ordinal));
      if (op.physical_type != target &&
          !(op.field.allow_int32_to_int64 && op.physical_type.id() == cudf::type_id::INT32 &&
            target.id() == cudf::type_id::INT64) &&
          !(op.field.allow_float_to_double && op.physical_type.id() == cudf::type_id::FLOAT32 &&
            target.id() == cudf::type_id::FLOAT64))
        throw std::invalid_argument("unqualified field conversion");
      auto [it, inserted] = read_slots.emplace(ordinal, result.read_names_.size());
      if (inserted) {
        result.read_names_.push_back(schema.names.at(ordinal));
        result.read_types_.push_back(schema.types.at(ordinal));
        result.read_indices_.push_back(static_cast<cudf::size_type>(ordinal));
      }
      op.physical_slot = it->second;
    }
    result.operations_.push_back(std::move(op));
  }
  if (result.read_names_.empty()) {
    // All requested fields may be absent. Read one approved physical column as
    // a row-count carrier rather than decoding the entire file for its cardinality.
    for (std::size_t i = 0; i < schema.names.size(); ++i) {
      if (!field_mapping_type(schema.types[i])) continue;
      result.read_names_.push_back(schema.names[i]);
      result.read_types_.push_back(schema.types[i]);
      result.read_indices_.push_back(static_cast<cudf::size_type>(i));
      break;
    }
    if (result.read_names_.empty())
      throw std::invalid_argument("field mapping has no qualified row-count carrier");
  }
  std::vector<cudf::data_type> targets;
  for (auto const& op : result.operations_)
    targets.push_back(op.target_type);
  if (!result.program_.input_types || *result.program_.input_types != targets)
    throw std::invalid_argument("mapped schema does not match its bound program");
  return result;
}

bool field_id_mapping::compatible_with(field_id_mapping const& other) const
{
  if (program_.decimal_precisions != other.program_.decimal_precisions ||
      program_.contract != other.program_.contract ||
      program_.generation != other.program_.generation ||
      program_.input_types != other.program_.input_types ||
      program_.output_types != other.program_.output_types ||
      program_.validate_output_schema != other.program_.validate_output_schema ||
      program_.predicates != other.program_.predicates ||
      program_.needs_visibility != other.program_.needs_visibility ||
      metadata_->original_schema.empty() ||
      metadata_->original_schema != other.metadata_->original_schema ||
      metadata_->arrow_schema != other.metadata_->arrow_schema ||
      read_names_ != other.read_names_ || operations_.size() != other.operations_.size())
    return false;
  for (std::size_t i = 0; i < operations_.size(); ++i) {
    auto const& a = operations_[i];
    auto const& b = other.operations_[i];
    if (a.field.id != b.field.id || a.field.type != b.field.type ||
        a.physical_slot != b.physical_slot || a.physical_type != b.physical_type ||
        a.target_type != b.target_type ||
        a.field.missing_value.has_value() != b.field.missing_value.has_value())
      return false;
    if (a.field.missing_value && !same_default(*a.field.missing_value, *b.field.missing_value))
      return false;
  }
  // The coalesced reader uses the first mapping's physical projection for every
  // source in the run. Keep a run together only when reader positions and source
  // types are identical as well.
  return read_indices_ == other.read_indices_ && read_types_ == other.read_types_;
}

std::size_t field_id_mapping::additional_bytes(std::size_t rows) const
{
  if (rows > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()))
    throw std::invalid_argument("field mapping batch exceeds row limit");
  std::size_t total = 0;
  auto add          = [&](std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - total)
      throw std::overflow_error("field mapping reservation overflow");
    total += bytes;
  };
  std::unordered_set<std::size_t> seen;
  for (auto const& op : operations_) {
    bool duplicate = op.physical_slot && !seen.insert(*op.physical_slot).second;
    if (op.physical_slot && op.physical_type == op.target_type && !duplicate) continue;
    if (op.target_type.id() == cudf::type_id::STRING) {
      // A duplicate variable-width source needs its actual decoded size; this
      // static estimate cannot certify it. Do not silently under-reserve it.
      if (op.physical_slot)
        throw std::invalid_argument("duplicate strings require decoded-size reservation");
      auto size = op.field.missing_value->IsNull()
                    ? 0
                    : op.field.missing_value->GetValue<std::string>().size();
      if (size && rows > std::numeric_limits<std::size_t>::max() / size)
        throw std::overflow_error("field default reservation overflow");
      add(rows * size);
      add((rows + 1) * sizeof(int64_t));  // also covers cuDF large-string offsets
      add(size);                          // the default scalar coexists with its repeated output
    } else {
      add(rows * cudf::size_of(op.target_type));
    }
    if (!op.physical_slot) add(256);  // scalar payload, validity and small control buffers
    // cuDF mask allocations are padded to 64 bytes. Include a nullable result
    // even when the current physical column happens to have no nulls.
    add(((rows + 511) / 512) * 64);
  }
  if (operations_.empty()) add(rows);
  return total;
}

materialized_batch field_id_mapping::apply(owning_table_view decoded,
                                           int64_t rows,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr) const
{
  if (rows < 0 || rows > std::numeric_limits<cudf::size_type>::max() || decoded.num_rows() != rows)
    throw std::invalid_argument("mapped decode cardinality mismatch");
  auto input = decoded.release(stream, mr);
  if (!input) throw std::invalid_argument("missing decoded table");
  auto columns = input->release();
  if (!read_names_.empty() && columns.size() != read_names_.size())
    throw std::invalid_argument("decoded field projection width mismatch");
  std::vector<std::size_t> uses(columns.size());
  for (auto const& op : operations_)
    if (op.physical_slot) {
      if (*op.physical_slot >= columns.size() ||
          columns[*op.physical_slot]->type() != op.physical_type)
        throw std::invalid_argument("decoded field does not match conversion approval");
      ++uses[*op.physical_slot];
    }
  std::vector<std::unique_ptr<cudf::column>> output;
  for (auto const& op : operations_) {
    if (!op.physical_slot) {
      auto value = default_scalar(*op.field.missing_value, op.target_type, stream, mr);
      output.push_back(cudf::make_column_from_scalar(*value, rows, stream, mr));
    } else {
      auto slot = *op.physical_slot;
      if (op.physical_type != op.target_type)
        output.push_back(cudf::cast(columns[slot]->view(), op.target_type, stream, mr));
      else if (uses[slot] == 1)
        output.push_back(std::move(columns[slot]));
      else
        output.push_back(std::make_unique<cudf::column>(columns[slot]->view(), stream, mr));
      --uses[slot];
    }
  }
  materialized_batch batch;
  batch.table      = owning_table_view(std::make_unique<cudf::table>(std::move(output)));
  batch.contract   = program_.contract;
  batch.generation = program_.generation;
  batch.rows       = rows;
  batch.logical_schema_complete = true;
  ensure_row_count_carrier(batch, stream, mr);
  return batch;
}

namespace {
cudf::io::parquet_reader_options validate_mapped_source(parquet_source const& source,
                                                        field_id_mapping const& mapping,
                                                        bound_table_scan const& contract,
                                                        std::span<std::size_t const> groups)
{
  if (contract.contract_id != mapping.program().contract || !source.datasource ||
      source.datasource->get_io_object().identity_cache_key() != mapping.identity() ||
      source.metadata.get() != mapping.metadata()->file_metadata().get())
    throw std::invalid_argument("mapped Parquet source does not match its approval");
  if (source.row_group_indices.size() != groups.size() ||
      !std::equal(groups.begin(), groups.end(), source.row_group_indices.begin()))
    throw std::invalid_argument("mapped row groups do not match prepared input");
  auto const& metadata = *mapping.metadata();
  effective_reader_projection projection;
  projection.names       = mapping.read_names();
  projection.bound_types = mapping.read_types();
  // Binding compiled the physical projection once for this immutable footer.
  // Semantic consumers only see the separately approved logical conversions.
  for (std::size_t i = 0; i < projection.names.size(); ++i)
    projection.projected.push_back(i);
  projection.natural_read = projection.names.empty();
  auto approval           = check_parquet_split_profile(*source.metadata,
                                              metadata.encryption_evidence,
                                              contract,
                                              projection,
                                              groups,
                                                        {},
                                              metadata.original_logical_annotations);
  if (!approval.approved) throw std::invalid_argument(approval.text);
  auto options = cudf::io::parquet_reader_options::builder().build();
  if (!projection.names.empty()) options.set_column_indices(mapping.read_indices());
  if (groups.empty()) options.set_num_rows(0);
  return options;
}
cudf::io::parquet_reader_options validate_mapped_unit(mapped_parquet_unit const& unit,
                                                      bound_table_scan const& contract)
{
  std::vector<std::size_t> groups;
  for (auto group : unit.source.row_group_indices) {
    if (group < 0) throw std::invalid_argument("negative mapped row group");
    groups.push_back(static_cast<std::size_t>(group));
  }
  return validate_mapped_source(unit.source, *unit.mapping, contract, groups);
}
void append_lineage(materialized_batch& batch,
                    parquet_source const& source,
                    field_id_mapping const& mapping,
                    int64_t& batch_offset)
{
  auto file_index = batch.lineage.files.size();
  batch.lineage.files.push_back({mapping.identity(), source.metadata->num_rows});
  std::vector<int64_t> row_group_prefix(source.metadata->row_groups.size() + 1, 0);
  for (std::size_t i = 0; i < source.metadata->row_groups.size(); ++i) {
    row_group_prefix[i + 1] = row_group_prefix[i] + source.metadata->row_groups[i].num_rows;
  }
  for (auto group : source.row_group_indices) {
    int64_t file_offset = row_group_prefix.at(static_cast<std::size_t>(group));
    auto count          = source.metadata->row_groups.at(group).num_rows;
    batch.lineage.runs.push_back(
      {mapping.identity(), file_offset, batch_offset, count, file_index});
    batch_offset += count;
  }
}
}  // namespace

materialized_batch materialize_mapped_parquet(parquet_source const& source,
                                              field_id_mapping const& mapping,
                                              bound_table_scan const& contract,
                                              std::span<std::size_t const> groups,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  auto options   = validate_mapped_source(source, mapping, contract, groups);
  auto decoded   = materialize_parquet(std::span(&source, 1), options, {}, stream, mr);
  auto rows      = decoded->num_rows();
  auto batch     = mapping.apply(owning_table_view(std::move(decoded)), rows, stream, mr);
  int64_t offset = 0;
  append_lineage(batch, source, mapping, offset);
  validate_materialized_batch(batch, mapping.program());
  return batch;
}

std::vector<materialized_batch> materialize_mapped_parquet_runs(
  std::span<mapped_parquet_unit const> units,
  bound_table_scan const& contract,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<materialized_batch> batches;
  std::size_t begin = 0;
  while (begin < units.size()) {
    if (!units[begin].mapping) throw std::invalid_argument("missing prepared field mapping");
    auto const& first = *units[begin].mapping;
    std::size_t end   = begin + 1;
    while (
      end < units.size() && units[end].mapping && !units[begin].source.row_group_indices.empty() &&
      !units[end].source.row_group_indices.empty() && first.compatible_with(*units[end].mapping))
      ++end;
    std::vector<parquet_source> sources;
    auto options = validate_mapped_unit(units[begin], contract);
    sources.push_back(units[begin].source);
    for (std::size_t i = begin + 1; i < end; ++i) {
      (void)validate_mapped_unit(units[i], contract);
      sources.push_back(units[i].source);
    }
    // Capacity is admitted by the caller for the combined run. Never let cuDF's
    // 32-bit row count overflow merely because individually legal units coalesce.
    int64_t rows = 0;
    for (auto const& source : sources)
      for (auto rg : source.row_group_indices) {
        auto count = source.metadata->row_groups.at(rg).num_rows;
        if (count < 0 || count > std::numeric_limits<cudf::size_type>::max() - rows)
          throw std::invalid_argument("mapped run exceeds batch row limit");
        rows += count;
      }
    auto decoded   = materialize_parquet(sources, options, {}, stream, mr);
    auto batch     = first.apply(owning_table_view(std::move(decoded)), rows, stream, mr);
    int64_t offset = 0;
    for (std::size_t i = begin; i < end; ++i)
      append_lineage(batch, units[i].source, *units[i].mapping, offset);
    validate_materialized_batch(batch, first.program());
    batches.push_back(std::move(batch));
    begin = end;
  }
  return batches;
}

}  // namespace sirius::op::scan
