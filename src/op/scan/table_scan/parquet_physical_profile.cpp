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

#include "op/scan/table_scan/parquet_physical_profile.hpp"

#include "io/parquet_helpers.hpp"
#include <cucascade/cudf/datasource.hpp>
#include "op/scan/parquet_metadata.hpp"
#include "op/scan/parquet_schema_mapping.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet_io_utils.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace sirius::op::scan {
namespace {
// A bounded Compact Protocol walk. Unknown fields are skipped recursively;
// completeness is asserted only after the entire metadata struct is consumed.
class crypto_reader {
 public:
  explicit crypto_reader(std::span<uint8_t const> bytes) : bytes_(bytes) {}
  parquet_encryption_evidence read()
  {
    structure(1, 0);
    // A signed plaintext footer appends a 12-byte nonce and 16-byte GCM tag.
    result_.complete =
      offset_ == bytes_.size() ||
      ((result_.footer_encrypted || result_.columns_encrypted) && bytes_.size() - offset_ == 28);
    return result_;
  }
  std::vector<uint8_t> const& original_schema() const { return original_schema_; }
  std::vector<uint8_t> const& original_logical_annotations() const
  {
    return original_logical_annotations_;
  }

 private:
  uint8_t byte()
  {
    if (offset_ == bytes_.size()) throw std::runtime_error("truncated compact footer");
    return bytes_[offset_++];
  }
  uint64_t varint()
  {
    uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 7) {
      auto b = byte();
      if (shift == 63 && (b & 0xfe)) throw std::runtime_error("invalid compact varint");
      value |= uint64_t(b & 127) << shift;
      if (!(b & 128)) return value;
    }
    throw std::runtime_error("invalid compact varint");
  }
  void advance(uint64_t count)
  {
    if (count > bytes_.size() - offset_) throw std::runtime_error("truncated compact value");
    offset_ += count;
  }
  void structure(unsigned context, unsigned depth)
  {
    if (depth > 128) throw std::runtime_error("compact footer nesting limit");
    int64_t field = 0;
    while (auto header = byte()) {
      auto delta = header >> 4;
      if (delta)
        field += delta;
      else {
        auto z = varint();
        field  = static_cast<int64_t>(z >> 1) ^ -static_cast<int64_t>(z & 1);
      }
      auto type = header & 15;
      // FileMetaData.encryption_algorithm and ColumnChunk crypto/metadata fields.
      if ((context == 1 && field == 8) || (context == 3 && (field == 8 || field == 9)))
        result_.columns_encrypted = true;
      unsigned child = context == 1 && field == 4   ? 2
                       : context == 2 && field == 1 ? 3
                       : context == 1 && field == 2 ? 4
                                                    : 0;
      if (context == 4 && field == 10 && !original_logical_annotations_.empty())
        original_logical_annotations_.back() = 1;
      auto const value_start = offset_;
      skip(type, child, depth + 1, true);
      if (context == 1 && field == 2) {
        original_schema_.assign(bytes_.begin() + value_start, bytes_.begin() + offset_);
      }
    }
  }
  void skip(unsigned type, unsigned context, unsigned depth, bool field)
  {
    if (depth > 128) throw std::runtime_error("compact footer nesting limit");
    switch (type) {
      case 1:
      case 2:
        if (!field) byte();
        return;
      case 3: advance(1); return;
      case 4:
      case 5:
      case 6: varint(); return;
      case 7: advance(8); return;
      case 8: advance(varint()); return;
      case 9:
      case 10: {
        auto header    = byte();
        uint64_t count = header >> 4;
        if (count == 15) count = varint();
        if (count > bytes_.size() - offset_) throw std::runtime_error("invalid compact list");
        for (uint64_t i = 0; i < count; ++i) {
          if (context == 4) original_logical_annotations_.push_back(0);
          skip(header & 15, context, depth + 1, false);
        }
        return;
      }
      case 11: {
        auto count = varint();
        if (!count) return;
        auto types = byte();
        if (count > bytes_.size() - offset_) throw std::runtime_error("invalid compact map");
        for (uint64_t i = 0; i < count; ++i) {
          skip(types >> 4, 0, depth + 1, false);
          skip(types & 15, 0, depth + 1, false);
        }
        return;
      }
      case 12: structure(context, depth + 1); return;
      default: throw std::runtime_error("invalid compact field type");
    }
  }
  std::span<uint8_t const> bytes_;
  std::size_t offset_ = 0;
  parquet_encryption_evidence result_;
  std::vector<uint8_t> original_schema_;
  std::vector<uint8_t> original_logical_annotations_;
};
}  // namespace
parquet_encryption_evidence inspect_parquet_encryption(std::span<uint8_t const> footer)
{
  try {
    return crypto_reader(footer).read();
  } catch (std::runtime_error const&) {
    return {};
  }
}
std::shared_ptr<parquet_metadata> resolve_parquet_metadata(
  cucascade::io::datasource& source,
  scan_contract_id contract,
  std::string const& identity,
  cudf::io::parquet_reader_options const& options,
  bool* cache_hit)
{
  if (auto cached = std::dynamic_pointer_cast<parquet_metadata>(source.metadata())) {
    if (cache_hit) *cache_hit = true;
    return cached;
  }
  if (cache_hit) *cache_hit = false;
  auto footer = fetch_plaintext_parquet_footer(source, contract, identity);
  crypto_reader probe({footer->data(), footer->size()});
  auto encryption = probe.read();
  if (!encryption.complete || probe.original_schema().empty()) {
    throw std::runtime_error("incomplete Parquet footer evidence: " + identity);
  }
  cudf::io::parquet::experimental::hybrid_scan_reader reader(
    cudf::host_span<uint8_t const>(footer->data(), footer->size()), options);
  auto parsed = std::make_shared<cudf::io::parquet::FileMetaData const>(reader.parquet_metadata());
  std::string arrow_schema;
  for (auto const& entry : parsed->key_value_metadata) {
    if (entry.key == "ARROW:schema") {
      arrow_schema = entry.value;
      break;
    }
  }
  auto result = std::make_shared<parquet_metadata>(parsed,
                                                   footer->size(),
                                                   encryption,
                                                   probe.original_schema(),
                                                   std::move(arrow_schema),
                                                   probe.original_logical_annotations());
  std::ignore = source.store_metadata(result);
  return result;
}
std::unique_ptr<cudf::io::datasource::buffer> fetch_plaintext_parquet_footer(
  cudf::io::datasource& source, scan_contract_id contract, std::string const& identity)
{
  try {
    return cudf::io::parquet::fetch_footer_to_host(source);
  } catch (...) {
    auto original  = std::current_exception();
    bool encrypted = false;
    try {
      if (source.size() >= 4) {
        auto trailer = source.host_read(source.size() - 4, 4);
        encrypted    = trailer && trailer->size() == 4 &&
                    std::string_view(reinterpret_cast<char const*>(trailer->data()), 4) == "PARE";
      }
    } catch (...) { /* preserve the original IO failure */
    }
    if (encrypted)
      throw unsupported_physical_input(
        contract,
        identity,
        verdict_reason::parquet_encrypted,
        "Encrypted Parquet footer is not GPU-decodable: " + identity);
    std::rethrow_exception(original);
  }
}

namespace {
bool codec_supported(cudf::io::parquet::Compression codec)
{
  using C = cudf::io::parquet::Compression;
  switch (codec) {
    case C::UNCOMPRESSED:
    case C::SNAPPY:
    case C::GZIP:
    case C::ZSTD:
    case C::LZ4_RAW:
    case C::BROTLI: return true;
    default: return false;
  }
}
bool encoding_supported(cudf::io::parquet::Encoding encoding)
{
  using E = cudf::io::parquet::Encoding;
  switch (encoding) {
    case E::PLAIN:
    case E::PLAIN_DICTIONARY:
    case E::RLE_DICTIONARY:
    case E::RLE:
    case E::BIT_PACKED:
    case E::DELTA_BINARY_PACKED:
    case E::DELTA_LENGTH_BYTE_ARRAY:
    case E::DELTA_BYTE_ARRAY:
    case E::BYTE_STREAM_SPLIT: return true;
    default: return false;
  }
}
bool export_casts(duckdb::LogicalType const& file, duckdb::LogicalType const& bound, bool nested)
{
  using T = duckdb::LogicalTypeId;
  if (file == bound) return true;
  if (file.id() == T::STRUCT || bound.id() == T::STRUCT) {
    if (file.id() != bound.id()) return false;
    auto const& left  = duckdb::StructType::GetChildTypes(file);
    auto const& right = duckdb::StructType::GetChildTypes(bound);
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0; i < left.size(); ++i)
      if (left[i].first != right[i].first || !export_casts(left[i].second, right[i].second, true))
        return false;
    return true;
  }
  if (file.id() == T::LIST || bound.id() == T::LIST) {
    return file.id() == bound.id() &&
           export_casts(
             duckdb::ListType::GetChildType(file), duckdb::ListType::GetChildType(bound), true);
  }
  if (file.id() == T::MAP || bound.id() == T::MAP) {
    return file.id() == bound.id() &&
           export_casts(duckdb::MapType::KeyType(file), duckdb::MapType::KeyType(bound), true) &&
           export_casts(duckdb::MapType::ValueType(file), duckdb::MapType::ValueType(bound), true);
  }
  if (file.id() == T::ARRAY || bound.id() == T::ARRAY) return false;
  if (nested && (file.id() == T::VARCHAR || file.id() == T::BLOB)) return bound.id() == T::VARCHAR;
  // Only types the actual cuDF->DuckDB exporter represents with a castable
  // vector. Unknown/SQLNULL mappings never gain an export exception.
  switch (file.id()) {
    case T::BOOLEAN:
    case T::TINYINT:
    case T::UTINYINT:
    case T::SMALLINT:
    case T::USMALLINT:
    case T::INTEGER:
    case T::UINTEGER:
    case T::BIGINT:
    case T::UBIGINT:
    case T::FLOAT:
    case T::DOUBLE:
    case T::DECIMAL:
    case T::DATE:
    case T::TIMESTAMP:
    case T::TIMESTAMP_SEC:
    case T::TIMESTAMP_MS:
    case T::TIMESTAMP_NS:
    case T::VARCHAR:
    case T::BLOB: return true;
    default: return false;
  }
}
}  // namespace
std::optional<type_conversion> qualified_type_conversion(parquet_physical_type,
                                                         parquet_logical_type,
                                                         duckdb::LogicalType const&)
{
  return std::nullopt;
}

physical_profile_result check_parquet_split_profile(cudf::io::parquet::FileMetaData const& metadata,
                                                    parquet_encryption_evidence const& encryption,
                                                    bound_table_scan const& contract,
                                                    effective_reader_projection const& projection,
                                                    std::span<std::size_t const> retained,
                                                    leaf_set const& semantic,
                                                    std::span<uint8_t const> original_annotations)
{
  physical_profile_result result;
  auto refuse = [&](verdict_reason why, std::string text) {
    result.reason = why;
    result.text   = std::move(text);
    return result;
  };
  // Build the top-level name index once. Repeatedly walking every path for
  // each projected name makes a wide all-column scan quadratic in leaf count.
  std::unordered_map<std::string, std::vector<std::size_t>> leaves_by_name;
  if (!metadata.row_groups.empty()) {
    auto const& columns = metadata.row_groups.front().columns;
    for (std::size_t leaf = 0; leaf < columns.size(); ++leaf) {
      auto const& path = columns[leaf].meta_data.path_in_schema;
      if (!path.empty()) leaves_by_name[path.front()].push_back(leaf);
    }
  }
  std::vector<std::vector<std::size_t>> leaves_by_column(projection.names.size());
  for (std::size_t d = 0; d < projection.names.size(); ++d) {
    if (auto found = leaves_by_name.find(projection.names[d]); found != leaves_by_name.end())
      leaves_by_column[d] = found->second;
  }
  std::set<std::size_t> decoded;
  auto add = [&](auto const& inputs) {
    for (auto d : inputs) {
      if (d >= projection.names.size()) throw std::logic_error("invalid effective projection");
      auto const& leaves = leaves_by_column[d];
      decoded.insert(leaves.begin(), leaves.end());
    }
  };
  add(projection.projected);
  add(projection.filter);
  add(projection.carrier);
  if (projection.natural_read && !metadata.row_groups.empty())
    for (std::size_t i = 0; i < metadata.row_groups.front().columns.size(); ++i)
      decoded.insert(i);
  physical_profile profile;
  std::vector<std::size_t> profile_leaves;
  // Most row groups have the same evidence for a given leaf. Keep a short
  // per-leaf chain so each checked chunk avoids a tree lookup and tuple copy.
  auto const absent = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> first_by_leaf(
    metadata.row_groups.empty() ? 0 : metadata.row_groups.front().columns.size(), absent);
  std::vector<std::size_t> next_profile;
  for (auto rg : retained) {
    if (rg >= metadata.row_groups.size()) throw std::logic_error("invalid retained row group");
    auto const& group = metadata.row_groups[rg];
    for (auto leaf : decoded) {
      if (leaf >= group.columns.size())
        return refuse(verdict_reason::parquet_type_unqualified,
                      "Parquet row-group schema is incomplete");
      auto const& column = group.columns[leaf].meta_data;
      if (!codec_supported(column.codec))
        return refuse(verdict_reason::parquet_codec_unsupported,
                      "Parquet codec is not supported by pinned libcudf");
      uint64_t encodings = 0;
      for (auto encoding : column.encodings) {
        if (!encoding_supported(encoding))
          return refuse(verdict_reason::parquet_codec_unsupported,
                        "Parquet encoding is not supported by pinned libcudf");
        encodings |= uint64_t{1} << static_cast<unsigned>(encoding);
      }
      physical_column_profile entry{static_cast<uint32_t>(column.type),
                                    uint64_t{1} << static_cast<unsigned>(column.codec),
                                    encodings,
                                    false};
      auto const schema_index = group.columns[leaf].schema_idx;
      if (schema_index >= 0 && static_cast<std::size_t>(schema_index) < metadata.schema.size()) {
        auto const& schema = metadata.schema[schema_index];
        auto const original_logical =
          original_annotations.empty() ||
              static_cast<std::size_t>(schema_index) >= original_annotations.size()
            ? schema.logical_type.has_value()
            : original_annotations[schema_index] != 0;
        entry.logical_annotation = schema.logical_type && original_logical
                                     ? static_cast<uint32_t>(schema.logical_type->type)
                                     : 0;
        entry.converted_annotation =
          schema.converted_type ? static_cast<uint32_t>(*schema.converted_type) + 1 : 0;
        entry.scale     = schema.decimal_scale;
        entry.precision = schema.decimal_precision;
      }
      auto same_evidence = [&](physical_column_profile const& candidate) {
        return candidate.type == entry.type && candidate.data_codecs == entry.data_codecs &&
               candidate.validity_or_encodings == entry.validity_or_encodings &&
               candidate.logical_annotation == entry.logical_annotation &&
               candidate.converted_annotation == entry.converted_annotation &&
               candidate.scale == entry.scale && candidate.precision == entry.precision;
      };
      auto found = first_by_leaf[leaf];
      while (found != absent && !same_evidence(profile.columns[found]))
        found = next_profile[found];
      if (found == absent) {
        next_profile.push_back(first_by_leaf[leaf]);
        first_by_leaf[leaf] = profile.columns.size();
        profile_leaves.push_back(leaf);
        profile.columns.push_back(entry);
      } else {
        auto& count = profile.columns[found].checked_chunks;
        if (count != std::numeric_limits<uint32_t>::max()) ++count;
      }
    }
  }
  if (!retained.empty() && !decoded.empty()) {
    auto schema = io::parquet_helpers::extract_schema(metadata, true, original_annotations);
    std::unordered_map<std::string, std::size_t> type_by_name;
    for (std::size_t index = 0; index < schema.names.size(); ++index)
      type_by_name.emplace(schema.names[index], index);
    for (std::size_t d = 0; d < projection.names.size(); ++d) {
      auto const& leaves = leaves_by_column[d];
      if (std::none_of(
            leaves.begin(), leaves.end(), [&](auto leaf) { return decoded.contains(leaf); }))
        continue;
      auto it = type_by_name.find(projection.names[d]);
      if (it == type_by_name.end() || d >= projection.bound_types.size())
        return refuse(verdict_reason::parquet_type_unqualified,
                      "Parquet input has no bound type evidence");
      auto const& actual   = schema.types[it->second];
      auto const& expected = projection.bound_types[d];
      if (actual == expected) continue;
      ++result.type_mismatches;
      bool semantic_input = d >= semantic.size() || semantic[d];
      if (semantic_input || !contract.host_export_available ||
          !export_casts(actual, expected, false))
        return refuse(verdict_reason::parquet_type_unqualified,
                      "Parquet column '" + projection.names[d] +
                        "' has unqualified type drift from " + actual.ToString() + " to " +
                        expected.ToString());
      std::unordered_set<std::size_t> column_leaves(leaves.begin(), leaves.end());
      for (std::size_t i = 0; i < profile.columns.size(); ++i)
        if (column_leaves.contains(profile_leaves[i])) profile.columns[i].type_mismatch = true;
    }
  }
  if (encryption.footer_encrypted || encryption.columns_encrypted)
    return refuse(verdict_reason::parquet_encrypted,
                  "Encrypted Parquet input is not GPU-decodable");
  if (!encryption.complete)
    return refuse(verdict_reason::parquet_encryption_evidence_missing,
                  "Parquet encryption evidence is missing");
  result.approved = true;
  result.validation =
    check_bit(later_check::footer_per_file) | check_bit(later_check::profile_per_file);
  if (contract.profiles) result.profile = contract.profiles->add(std::move(profile));
  return result;
}

physical_profile_result check_iceberg_file_schema(cudf::io::parquet::FileMetaData const& metadata,
                                                  iceberg_table_schema const& table,
                                                  std::string_view probe_path,
                                                  std::span<uint8_t const> original_annotations)
{
  physical_profile_result result;
  result.approved = true;
  if (table.fields.empty()) return result;
  std::string path(probe_path);
  auto refuse = [&](verdict_reason reason, std::string text) {
    result.approved = false;
    result.reason   = reason;
    result.text     = "iceberg_scan declines the GPU scan path: " + std::move(text);
    return result;
  };
  if (metadata.schema.empty())
    return refuse(verdict_reason::iceberg_schema_no_rows,
                  "iceberg_scan data file '" + path +
                    "' returned no Parquet schema rows, so it could not be proven to carry the "
                    "table's current schema");
  std::map<std::pair<std::string, int32_t>, std::string> table_schema, file_schema;
  std::vector<int32_t> table_field_order, file_field_order;
  for (auto const& field : table.fields) {
    table_schema.emplace(std::make_pair(field.name, field.id), field.type);
    table_field_order.push_back(field.id);
  }
  for (std::size_t index = 0; index < metadata.schema.size(); ++index) {
    auto const& element = metadata.schema[index];
    if (!element.field_id) continue;
    file_schema.emplace(
      std::make_pair(element.name, *element.field_id),
      element.num_children > 0 || element.type == cudf::io::parquet::Type::UNDEFINED
        ? std::string{}
        : io::parquet_helpers::leaf_schema_type(
            element,
            original_annotations.empty() || index >= original_annotations.size()
              ? element.logical_type.has_value()
              : original_annotations[index] != 0)
            .ToString());
    file_field_order.push_back(*element.field_id);
  }
  if (file_schema.empty()) {
    return refuse(
      verdict_reason::iceberg_schema_no_field_ids,
      "iceberg_scan data file '" + path +
        "' carries no Parquet field ids while the table's schema declares them, so it is "
        "name-mapped; this scan path resolves columns by name and would read the wrong "
        "column or fail at scan time");
  }

  for (auto const& [key, table_type] : table_schema) {
    auto const found = file_schema.find(key);
    if (found == file_schema.end()) {
      return refuse(
        verdict_reason::iceberg_schema_missing_field,
        "iceberg_scan data file '" + path +
          "' does not carry the table's current schema (no match for " + key.first + "#" +
          std::to_string(key.second) +
          "), so the table's schema has evolved; this scan path resolves columns by name and "
          "would read the wrong column or fail at scan time");
    }
    // Empty on either side is a nested container, whose type is implied by its children.
    if (!table_type.empty() && !found->second.empty() && found->second != table_type) {
      return refuse(
        verdict_reason::iceberg_schema_promoted_type,
        "iceberg_scan data file '" + path + "' stores " + key.first + "#" +
          std::to_string(key.second) + " as " + found->second + " while the table declares " +
          table_type +
          ", so the column's type was promoted; this scan path reads the file's own physical "
          "type and would hand back a column of the wrong type");
    }
  }
  // Extra fields are dropped columns, which a name-based lookup would happily resolve to.
  if (file_schema.size() != table_schema.size()) {
    return refuse(
      verdict_reason::iceberg_schema_field_count,
      "iceberg_scan data file '" + path + "' carries " + std::to_string(file_schema.size()) +
        " field ids where the table declares " + std::to_string(table_schema.size()) +
        ", so the table's schema has evolved; this scan path resolves columns by name and "
        "would read the wrong column or fail at scan time");
  }

  // Everything above is MEMBERSHIP, which a permuted file satisfies. Order matters because the
  // GPU path does not map columns by field id: for a full `SELECT *`, build_scan_plan leaves
  // needs_reader_projection false, so cuDF emits columns in the first footer's order while the
  // rest of the plan expects the bound snapshot's. DuckDB's own reader uses BY_FIELD_ID and is
  // unaffected, so the two disagree silently -- and a castable permutation converts the values
  // rather than erroring, since the runtime schema check only logs. Nested children are included
  // because a reordered struct child fails the same way.
  if (file_field_order != table_field_order) {
    return refuse(
      verdict_reason::iceberg_schema_physical_order,
      "iceberg_scan data file '" + path +
        "' stores the table's fields in a different physical order than the bound snapshot's "
        "schema declares; this scan path emits columns in the file's own order and would hand "
        "back the right columns under the wrong names");
  }
  result.validation = check_bit(later_check::schema_per_file);
  return result;
}
}  // namespace sirius::op::scan
