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

#include "op/scan/table_scan/bound_read_view.hpp"

#include "exec/stream_plan_bindings.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/connector_registry.hpp"
#include "sirius_registration.hpp"

#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/common/multi_file/multi_file_function.hpp>
#include <duckdb/common/multi_file/multi_file_list.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/common/serializer/binary_deserializer.hpp>
#include <duckdb/common/serializer/binary_serializer.hpp>
#include <duckdb/common/serializer/memory_stream.hpp>
#include <duckdb/common/types/hash.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <duckdb/transaction/meta_transaction.hpp>
#include <parquet_crypto.hpp>
#include <parquet_multi_file_info.hpp>
#include <parquet_reader.hpp>

#include <algorithm>
#include <charconv>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace sirius::op::scan {
namespace {
void decimal(std::string& out, uint64_t value)
{
  char buffer[std::numeric_limits<uint64_t>::digits10 + 1];
  auto const [end, error] = std::to_chars(std::begin(buffer), std::end(buffer), value);
  if (error != std::errc{}) throw std::runtime_error("read-view integer encoding failed");
  out.append(buffer, end);
}

std::size_t decimal_digits(std::size_t value)
{
  std::size_t result = 1;
  while (value >= 10) {
    value /= 10;
    ++result;
  }
  return result;
}

void append(std::string& out, char tag, std::string_view value)
{
  out += tag;
  decimal(out, value.size());
  out += ':';
  out.append(value);
}
void number(std::string& out, uint64_t value)
{
  char buffer[std::numeric_limits<uint64_t>::digits10 + 1];
  auto const [end, error] = std::to_chars(std::begin(buffer), std::end(buffer), value);
  if (error != std::errc{}) throw std::runtime_error("read-view integer encoding failed");
  out += 'u';
  decimal(out, static_cast<uint64_t>(end - buffer));
  out += ':';
  out.append(buffer, end);
}

void boolean(std::string& out, bool value) { append(out, 'b', value ? "1" : "0"); }

template <typename MAP, typename VALUE>
void append_sorted_map(std::string& out, MAP const& map, VALUE&& encode_value)
{
  std::vector<std::string> keys;
  keys.reserve(map.size());
  for (auto const& [key, unused] : map) {
    (void)unused;
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());
  number(out, keys.size());
  for (auto const& key : keys) {
    append(out, 'k', key);
    encode_value(out, map.at(key));
  }
}

void append_multi_file_options(std::string& out, duckdb::MultiFileOptions const& options)
{
  boolean(out, options.filename);
  boolean(out, options.hive_partitioning);
  boolean(out, options.auto_detect_hive_partitioning);
  boolean(out, options.union_by_name);
  boolean(out, options.hive_types_autocast);
  number(out, static_cast<uint8_t>(options.mapping));
  append(out, 's', options.filename_column);
  append_sorted_map(out, options.hive_types_schema, [](std::string& target, auto const& value) {
    append(target, 't', value.ToString());
  });
  append_sorted_map(out, options.custom_options, [](std::string& target, auto const& value) {
    append(target, 'v', canonical_value_text(value));
  });
}

// ParquetEncryptionConfig deliberately exposes its contents only through DuckDB's public
// serializer. Capture that small schema structurally so the unordered column-key map can be
// sorted before it becomes identity text.
class canonical_encryption_serializer final : public duckdb::Serializer {
 public:
  explicit canonical_encryption_serializer(std::string footer_key)
    : footer_key_(std::move(footer_key))
  {
  }

  std::string finish() const
  {
    std::string out;
    append(out, 's', footer_key_);
    append_sorted_map(out, column_keys_, [](std::string& target, auto const& value) {
      append(target, 's', value);
    });
    return out;
  }

 private:
  void OnPropertyBegin(duckdb::field_id_t, char const* tag) override { property_ = tag ? tag : ""; }
  void OnPropertyEnd() override { property_.clear(); }
  void OnOptionalPropertyBegin(duckdb::field_id_t, char const* tag, bool) override
  {
    property_ = tag ? tag : "";
  }
  void OnOptionalPropertyEnd(bool) override { property_.clear(); }
  void OnObjectBegin() override {}
  void OnObjectEnd() override {}
  void OnListBegin(duckdb::idx_t) override {}
  void OnListEnd() override {}
  void OnNullableBegin(bool) override {}
  void OnNullableEnd() override {}

  void capture(std::string value)
  {
    if (property_ == "footer_key") {
      footer_key_ = std::move(value);
    } else if (property_ == "key") {
      pending_key_ = std::move(value);
    } else if (property_ == "value") {
      column_keys_.insert_or_assign(pending_key_, std::move(value));
    }
  }
  [[noreturn]] static void unexpected()
  {
    throw std::runtime_error("Parquet encryption config serialized an unexpected value type");
  }
  void WriteNull() override { unexpected(); }
  void WriteValue(char) override { unexpected(); }
  void WriteValue(bool) override { unexpected(); }
  void WriteValue(uint8_t) override { unexpected(); }
  void WriteValue(int8_t) override { unexpected(); }
  void WriteValue(uint16_t) override { unexpected(); }
  void WriteValue(int16_t) override { unexpected(); }
  void WriteValue(uint32_t) override { unexpected(); }
  void WriteValue(int32_t) override { unexpected(); }
  void WriteValue(uint64_t) override { unexpected(); }
  void WriteValue(int64_t) override { unexpected(); }
  void WriteValue(duckdb::hugeint_t) override { unexpected(); }
  void WriteValue(duckdb::uhugeint_t) override { unexpected(); }
  void WriteValue(float) override { unexpected(); }
  void WriteValue(double) override { unexpected(); }
  void WriteValue(duckdb::string_t value) override { capture(value.GetString()); }
  void WriteValue(std::string const& value) override { capture(value); }
  void WriteValue(char const* value) override { capture(value ? value : ""); }
  void WriteDataPtr(duckdb::const_data_ptr_t, duckdb::idx_t) override { unexpected(); }

  std::string property_;
  std::string pending_key_;
  std::string footer_key_;
  std::unordered_map<std::string, std::string> column_keys_;
};

std::string canonical_encryption_config(duckdb::ParquetEncryptionConfig const& config)
{
  canonical_encryption_serializer serializer(config.GetFooterKey());
  config.Serialize(serializer);
  return serializer.finish();
}

duckdb::ParquetOptionsSerialization export_parquet_options(duckdb::TableFunction const& function,
                                                           duckdb::MultiFileBindData const& bind)
{
  if (!function.serialize || !bind.bind_data) {
    throw std::runtime_error("verified parquet source has no serializable format bind data");
  }
  duckdb::MultiFileBindData temporary;
  auto format_copy = bind.bind_data->Copy();
  auto* typed_copy = dynamic_cast<duckdb::TableFunctionData*>(format_copy.get());
  if (!typed_copy) {
    throw std::runtime_error("verified parquet source copied an invalid format bind payload");
  }
  (void)format_copy.release();
  temporary.bind_data = duckdb::unique_ptr<duckdb::TableFunctionData>(typed_copy);
  temporary.file_list =
    duckdb::make_shared_ptr<duckdb::SimpleMultiFileList>(duckdb::vector<duckdb::OpenFileInfo>{});
  temporary.file_options = bind.file_options;
  temporary.initial_reader.reset();
  temporary.union_readers.clear();

  duckdb::MemoryStream buffer;
  duckdb::BinarySerializer serializer(buffer);
  serializer.Begin();
  function.serialize(serializer, &temporary, function);
  serializer.End();
  buffer.Rewind();

  duckdb::BinaryDeserializer deserializer(buffer);
  deserializer.Begin();
  (void)deserializer.ReadProperty<duckdb::vector<std::string>>(100, "files");
  (void)deserializer.ReadProperty<duckdb::vector<duckdb::LogicalType>>(101, "types");
  (void)deserializer.ReadProperty<duckdb::vector<std::string>>(102, "names");
  auto options =
    deserializer.ReadProperty<duckdb::ParquetOptionsSerialization>(103, "parquet_options");
  deserializer.End();
  return options;
}

duckdb::TableFunction const& trusted_parquet_serializer()
{
  static duckdb::TableFunction const function = [] {
    auto functions = duckdb::ParquetScanFunction::GetFunctionSet().functions;
    if (functions.empty() || !functions.front().serialize) {
      throw std::runtime_error("trusted Parquet function has no serializer");
    }
    return functions.front();
  }();
  return function;
}

std::string parquet_selector(duckdb::TableFunction const& function,
                             duckdb::MultiFileBindData const& bind)
{
  auto const exported = export_parquet_options(function, bind);
  std::string out;
  append(out, 'v', "sirius.parquet-options.1");
  append_multi_file_options(out, exported.file_options);
  auto const& options = exported.parquet_options;
  boolean(out, options.binary_as_string);
  boolean(out, options.file_row_number);
  number(out, options.explicit_cardinality);
  boolean(out, options.can_have_nan);
  number(out, options.schema.size());
  for (auto const& column : options.schema) {
    number(out, static_cast<uint32_t>(column.field_id));
    append(out, 's', column.name);
    append(out, 't', column.type.ToString());
    append(out, 'v', canonical_value_text(column.default_value));
    append(out, 'v', canonical_value_text(column.identifier));
  }
  boolean(out, options.encryption_config != nullptr);
  if (options.encryption_config) {
    append(out, 'e', canonical_encryption_config(*options.encryption_config));
  }
  return out;
}

std::string selector_evidence(duckdb::vector<duckdb::Value> const& parameters,
                              duckdb::named_parameter_map_t const& named_parameters)
{
  std::string out;
  append(out, 'v', "sirius.selector-evidence.1");
  number(out, parameters.size());
  for (auto const& value : parameters)
    append(out, 'v', canonical_value_text(value));
  append_sorted_map(out, named_parameters, [](std::string& target, auto const& value) {
    append(target, 'v', canonical_value_text(value));
  });
  return out;
}

uint64_t transaction_id(duckdb::ClientContext& context)
{
  return context.transaction.HasActiveTransaction()
           ? context.ActiveTransaction().global_transaction_id
           : 0;
}

struct capture_input {
  duckdb::TableFunction const& function;
  duckdb::FunctionData const* bind_data;
  duckdb::vector<duckdb::Value> const& parameters;
  duckdb::named_parameter_map_t const* named_parameters;
  std::span<std::string const> resolved_paths;
  bool collect_evidence;
};

struct path_encoding_info {
  bool sorted               = true;
  std::size_t encoded_bytes = 0;
};

// A capture owns the bulk OpenFileInfo snapshot, while a candidate can borrow its
// already resolved strings. Encode either representation without a second path vector.
struct path_sequence {
  std::span<std::string const> strings;
  std::span<duckdb::OpenFileInfo const> files;

  std::size_t size() const { return files.empty() ? strings.size() : files.size(); }
  bool empty() const { return size() == 0; }
  std::string const& operator[](std::size_t index) const
  {
    return files.empty() ? strings[index] : files[index].path;
  }
};

void analyze_path(path_encoding_info& result, std::string const& path, std::string const* previous)
{
  result.encoded_bytes += path.size() + decimal_digits(path.size()) + 2;
  if (previous && *previous > path) result.sorted = false;
}

path_encoding_info analyze_paths(path_sequence paths)
{
  path_encoding_info result;
  for (std::size_t i = 0; i < paths.size(); ++i) {
    analyze_path(result, paths[i], i ? &paths[i - 1] : nullptr);
  }
  return result;
}

std::shared_ptr<bound_read_identity const> make_bound_read_identity_preanalyzed(
  bound_read_identity, path_sequence, path_encoding_info, read_view_capture_metrics*);

bound_read_view capture(capture_input input, duckdb::ClientContext& context)
{
  auto const* source = planner::lookup_connector(input.function, input.bind_data, context);
  if (!source) { throw std::runtime_error("cannot capture an unverified scan source"); }

  bound_read_identity identity;
  identity.source = {source->function_name, source->kind, source->registry_profile};
  bound_read_view view;
  view.transaction_id = transaction_id(context);
  std::vector<std::string> owned_paths;
  duckdb::vector<duckdb::OpenFileInfo> owned_files;
  path_sequence paths{input.resolved_paths, {}};
  auto path_info                             = analyze_paths(paths);
  std::size_t transient_path_string_capacity = 0;
  std::size_t evidence_tag_string_capacity   = 0;

  if (source->kind == source_kind::duckdb_native) {
    auto const* typed = dynamic_cast<duckdb::TableScanBindData const*>(input.bind_data);
    if (!typed) throw std::runtime_error("seq_scan capture has no TableScanBindData");
    auto& table          = typed->table.Cast<duckdb::DuckTableEntry>();
    auto const& columns  = table.GetColumns();
    identity.bound_types = columns.GetColumnTypes();
    identity.bound_names = columns.GetColumnNames();
    auto& attached       = table.GetStorage().GetAttached();
    identity.data_view   = native_table_identity{table.ParentCatalog().GetName(),
                                               table.ParentCatalog().GetOid(),
                                               table.ParentSchema().name,
                                               table.name,
                                               table.oid,
                                               attached.GetStorageManager().GetDBPath()};
    view.provider        = provider_borrow{
      context.transaction.GetActiveQuery(), view.transaction_id, &context, &table.GetStorage()};
  } else if (source->kind == source_kind::stream_source) {
    auto const* typed = dynamic_cast<exec::stream_source_bind_data const*>(input.bind_data);
    if (!typed) throw std::runtime_error("stream capture has no stream_source_bind_data");
    auto const& binding  = exec::catalog_for(context)->get(typed->stream_id);
    identity.bound_types = sirius::to_duckdb_vec(binding.types);
    identity.bound_names = duckdb::vector<std::string>(binding.names.begin(), binding.names.end());
    identity.data_view   = stream_identity{typed->stream_id};
  } else if (source->kind == source_kind::parquet_s3) {
    auto const* typed = dynamic_cast<duckdb::SiriusReadParquetBindData const*>(input.bind_data);
    if (!typed) throw std::runtime_error("Sirius S3 capture has no typed bind data");
    if (paths.empty()) {
      owned_paths.push_back(typed->uri);
      transient_path_string_capacity += owned_paths.back().capacity() + 1;
      paths     = {owned_paths, {}};
      path_info = analyze_paths(paths);
    }
    identity.bound_types = typed->bound_types;
    identity.bound_names = typed->bound_names;
    identity.selector    = canonical_value_text(duckdb::Value(typed->uri));
    identity.data_view   = file_inventory{static_cast<uint32_t>(paths.size())};
  } else {
    auto const* typed = dynamic_cast<duckdb::MultiFileBindData const*>(input.bind_data);
    if (!typed || !typed->file_list) {
      throw std::runtime_error("parquet capture has no MultiFileBindData file list");
    }
    std::shared_ptr<file_evidence_arrays> unsorted_evidence;
    bool any_size = false;
    bool any_etag = false;
    if (input.collect_evidence) unsorted_evidence = std::make_shared<file_evidence_arrays>();
    if (paths.empty()) {
      // GetAllFiles copies the bound inventory under one lock. Files()/Scan()
      // would lock and copy OpenFileInfo for every entry before copying its path again.
      owned_files           = typed->file_list->GetAllFiles();
      paths                 = {{}, {owned_files.data(), owned_files.size()}};
      auto const file_count = paths.size();
      if (unsorted_evidence) {
        unsorted_evidence->size.resize(file_count);
        unsorted_evidence->last_modified.resize(file_count);
        unsorted_evidence->size_present.resize(file_count);
        unsorted_evidence->last_modified_present.resize(file_count);
        unsorted_evidence->etag.resize(file_count);
      }
      for (std::size_t index = 0; index < file_count; ++index) {
        auto const& file = owned_files[index];
        analyze_path(path_info, file.path, index ? &owned_files[index - 1].path : nullptr);
        transient_path_string_capacity += file.path.capacity() + 1;
        if (!unsorted_evidence) continue;
        evidence_tag_string_capacity += unsorted_evidence->etag[index].capacity() + 1;
        if (!file.extended_info) continue;
        auto const& options = file.extended_info->options;
        if (auto found = options.find("file_size");
            found != options.end() && !found->second.IsNull()) {
          unsorted_evidence->size[index]         = found->second.GetValue<int64_t>();
          unsorted_evidence->size_present[index] = 1;
          any_size                               = true;
        }
        if (auto found = options.find("last_modified");
            found != options.end() && !found->second.IsNull()) {
          unsorted_evidence->last_modified[index] =
            found->second.GetValue<duckdb::timestamp_t>().value;
          unsorted_evidence->last_modified_present[index] = 1;
        }
        if (auto found = options.find("etag"); found != options.end() && !found->second.IsNull()) {
          evidence_tag_string_capacity -= unsorted_evidence->etag[index].capacity() + 1;
          unsorted_evidence->etag[index] = duckdb::StringValue::Get(found->second);
          evidence_tag_string_capacity += unsorted_evidence->etag[index].capacity() + 1;
          any_etag = true;
        }
      }
    }
    identity.bound_types = typed->types;
    identity.bound_names = typed->names;
    // iceberg_scan owns a different outer serializer. Its nested format payload is still the
    // Parquet bind payload, so export it through the verified built-in Parquet protocol.
    identity.selector  = parquet_selector(trusted_parquet_serializer(), *typed);
    identity.data_view = file_inventory{static_cast<uint32_t>(paths.size())};

    if (input.collect_evidence) {
      std::vector<std::size_t> order;
      if (!path_info.sorted) {
        order.resize(paths.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](auto left, auto right) {
          return paths[left] < paths[right];
        });
        constexpr auto visited = std::size_t{1} << (sizeof(std::size_t) * 8 - 1);
        if (order.size() >= visited) {
          throw std::length_error("Read-view file inventory is too large to sort");
        }
        // Invert sorted-position -> original-position in place. The high bit marks completed
        // cycles, so evidence sorting retains exactly one F-element index allocation.
        for (std::size_t start = 0; start < order.size(); ++start) {
          if ((order[start] & visited) != 0) continue;
          auto current = start;
          auto next    = order[current];
          while (next != start) {
            auto const following = order[next];
            order[next]          = current | visited;
            current              = next;
            next                 = following;
          }
          order[start] = current | visited;
        }
        for (auto& index : order)
          index &= ~visited;
        for (std::size_t current = 0; current < order.size(); ++current) {
          while (order[current] != current) {
            auto const target = order[current];
            std::swap(owned_files[current], owned_files[target]);
            std::swap(unsorted_evidence->size[current], unsorted_evidence->size[target]);
            std::swap(unsorted_evidence->last_modified[current],
                      unsorted_evidence->last_modified[target]);
            std::swap(unsorted_evidence->size_present[current],
                      unsorted_evidence->size_present[target]);
            std::swap(unsorted_evidence->last_modified_present[current],
                      unsorted_evidence->last_modified_present[target]);
            std::swap(unsorted_evidence->etag[current], unsorted_evidence->etag[target]);
            std::swap(order[current], order[target]);
          }
        }
      }
      path_info.sorted                 = true;
      view.metrics.sort_index_capacity = order.capacity() * sizeof(std::size_t);
      view.evidence                    = std::move(unsorted_evidence);
      view.depth                       = any_etag   ? evidence_depth::path_size_and_tag
                                         : any_size ? evidence_depth::path_and_size
                                                    : evidence_depth::path;
    }
  }

  view.replay_policy = {source->function_name, source->byte_source, source->permits_cpu_replay, {}};
  for (std::size_t i = 0; i < paths.size(); ++i) {
    auto const& path = paths[i];
    if (path.size() > 5 && (path[0] == 's' || path[0] == 'S') && path[1] == '3' && path[2] == ':' &&
        path[3] == '/' && path[4] == '/') {
      view.replay_policy.source             = transparent::byte_source_class::sirius_owned_s3;
      view.replay_policy.permits_cpu_replay = false;
      break;
    }
  }
  if (!view.replay_policy.permits_cpu_replay) {
    view.replay_policy.reason =
      view.replay_policy.source == transparent::byte_source_class::stream ? "stream" : "s3";
  }
  view.metrics.file_count              = paths.size();
  view.metrics.transient_path_capacity = owned_paths.capacity() * sizeof(std::string) +
                                         owned_files.capacity() * sizeof(duckdb::OpenFileInfo) +
                                         transient_path_string_capacity;
  if (view.evidence) {
    view.metrics.evidence_capacity =
      view.evidence->size.capacity() * sizeof(int64_t) +
      view.evidence->last_modified.capacity() * sizeof(int64_t) +
      view.evidence->size_present.capacity() * sizeof(uint8_t) +
      view.evidence->last_modified_present.capacity() * sizeof(uint8_t) +
      view.evidence->etag.capacity() * sizeof(std::string) + evidence_tag_string_capacity;
  }
  view.selector_evidence_required = source->selector_outside_bind_data;
  if (view.selector_evidence_required && input.named_parameters) {
    view.logical_selector_evidence = selector_evidence(input.parameters, *input.named_parameters);
  }
  view.identity =
    make_bound_read_identity_preanalyzed(std::move(identity), paths, path_info, &view.metrics);
  view.metrics.canonical_capacity = view.identity->fingerprint.canonical.capacity() + 1;
  return view;
}
}  // namespace

// Keep this allocation boundary visible to T6's stack attribution. It contains only
// R1 evidence correspondence storage, not baseline Parquet construction allocations.
__attribute__((noinline)) std::vector<std::size_t> make_read_view_evidence_index(
  std::span<std::string const> paths)
{
  std::vector<std::size_t> result(paths.size());
  std::vector<std::size_t> order(paths.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(
    order.begin(), order.end(), [&](auto left, auto right) { return paths[left] < paths[right]; });
  for (std::size_t sorted = 0; sorted < order.size(); ++sorted)
    result[order[sorted]] = sorted;
  return result;
}

namespace {
std::shared_ptr<bound_read_identity const> make_bound_read_identity_preanalyzed(
  bound_read_identity identity,
  path_sequence paths,
  path_encoding_info path_info,
  read_view_capture_metrics* metrics)
{
  if (identity.bound_types.size() != identity.bound_names.size()) {
    throw std::invalid_argument("Bound read schema has different type and name counts");
  }
  std::string tail;
  number(tail, identity.bound_types.size());
  for (size_t i = 0; i < identity.bound_types.size(); ++i) {
    duckdb::MemoryStream buffer;
    duckdb::BinarySerializer::Serialize(identity.bound_types[i], buffer);
    append(tail,
           't',
           std::string_view(reinterpret_cast<char const*>(buffer.GetData()), buffer.GetPosition()));
    append(tail, 's', identity.bound_names[i]);
  }
  append(tail, 'o', identity.selector);
  auto& text = identity.fingerprint.canonical;
  text.clear();
  append(text, 'v', "sirius.read-view.1");
  append(text, 's', identity.source.function_name);
  number(text, static_cast<uint8_t>(identity.source.kind));
  append(text, 's', identity.source.registry_profile);
  number(text, identity.data_view.index());
  if (auto const* native = std::get_if<native_table_identity>(&identity.data_view)) {
    if (!paths.empty()) throw std::invalid_argument("Native read view has file inventory");
    append(text, 's', native->catalog_name);
    number(text, native->catalog_oid);
    append(text, 's', native->schema_name);
    append(text, 's', native->table_name);
    number(text, native->table_oid);
    append(text, 's', native->db_path);
  } else if (auto const* files = std::get_if<file_inventory>(&identity.data_view)) {
    if (files->count != paths.size()) throw std::invalid_argument("Read-view file count mismatch");
    number(text, files->count);
    text.reserve(text.size() + path_info.encoded_bytes + tail.size());
    auto const path_begin = text.size();
    text.resize(path_begin + path_info.encoded_bytes);
    auto* destination = text.data() + path_begin;
    char prefix[std::numeric_limits<std::size_t>::digits10 + 3];
    prefix[0]                   = 'p';
    std::size_t previous_length = std::numeric_limits<std::size_t>::max();
    std::size_t prefix_length   = 0;
    auto encode_path            = [&](std::string const& path) {
      // Equal-length paths share the same length-delimited prefix. Write into the
      // pre-sized canonical buffer; the bytes are identical to append(text, 'p', path).
      if (path.size() != previous_length) {
        auto const [end, error] =
          std::to_chars(std::begin(prefix) + 1, std::end(prefix) - 1, path.size());
        if (error != std::errc{}) throw std::runtime_error("read-view path encoding failed");
        *end            = ':';
        prefix_length   = static_cast<std::size_t>(end - prefix) + 1;
        previous_length = path.size();
      }
      std::memcpy(destination, prefix, prefix_length);
      destination += prefix_length;
      std::memcpy(destination, path.data(), path.size());
      destination += path.size();
    };
    if (path_info.sorted) {
      for (std::size_t i = 0; i < paths.size(); ++i)
        encode_path(paths[i]);
    } else {
      std::vector<size_t> order(paths.size());
      if (metrics) {
        metrics->sort_index_capacity =
          std::max(metrics->sort_index_capacity, order.capacity() * sizeof(size_t));
      }
      std::iota(order.begin(), order.end(), 0);
      std::sort(
        order.begin(), order.end(), [&](size_t a, size_t b) { return paths[a] < paths[b]; });
      for (auto index : order)
        encode_path(paths[index]);
    }
  } else {
    if (!paths.empty()) throw std::invalid_argument("Stream read view has file inventory");
    number(text, std::get<stream_identity>(identity.data_view).stream_id);
  }
  text += tail;
  identity.fingerprint.hash = duckdb::Hash(text.data(), text.size());
  return std::make_shared<bound_read_identity const>(std::move(identity));
}
}  // namespace

std::shared_ptr<bound_read_identity const> make_bound_read_identity(
  bound_read_identity identity,
  std::span<std::string const> paths,
  read_view_capture_metrics* metrics)
{
  return make_bound_read_identity_preanalyzed(
    std::move(identity), {paths, {}}, analyze_paths({paths, {}}), metrics);
}

std::string canonical_value_text(duckdb::Value const& value)
{
  duckdb::MemoryStream buffer;
  duckdb::BinarySerializer::Serialize(value, buffer);
  std::string text;
  append(text,
         'v',
         std::string_view(reinterpret_cast<char const*>(buffer.GetData()), buffer.GetPosition()));
  return text;
}

std::string canonical_read_view_text(bound_read_view const& view)
{
  if (!view.identity) throw std::invalid_argument("Bound read view has no captured identity");
  return view.identity->fingerprint.canonical;
}

bound_read_view capture_bound_read_view(duckdb::LogicalGet const& get,
                                        duckdb::ClientContext& context)
{
  return capture(
    {get.function, get.bind_data.get(), get.parameters, &get.named_parameters, {}, true}, context);
}

std::vector<logical_bound_read_view> capture_bound_read_views(duckdb::LogicalOperator const& root,
                                                              duckdb::ClientContext& context)
{
  std::vector<logical_bound_read_view> captured;
  auto visit = [&](auto&& self, duckdb::LogicalOperator const& op) -> void {
    if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
      auto const& get = op.Cast<duckdb::LogicalGet>();
      if (planner::lookup_connector(get, context)) {
        captured.push_back({get.table_index, capture_bound_read_view(get, context)});
      }
    }
    for (auto const& child : op.children) {
      self(self, *child);
    }
  };
  visit(visit, root);
  return captured;
}

bound_read_view capture_bound_read_view(duckdb::PhysicalTableScan const& get,
                                        duckdb::ClientContext& context)
{
  return capture({get.function, get.bind_data.get(), get.parameters, nullptr, {}, true}, context);
}

std::vector<bound_read_view> capture_bound_read_views(duckdb::PhysicalOperator const& root,
                                                      duckdb::ClientContext& context)
{
  std::vector<bound_read_view> captured;
  auto visit = [&](auto&& self, duckdb::PhysicalOperator const& op) -> void {
    if (op.type == duckdb::PhysicalOperatorType::TABLE_SCAN) {
      auto const& get = op.Cast<duckdb::PhysicalTableScan>();
      if (planner::lookup_connector(get, context)) {
        captured.push_back(capture_bound_read_view(get, context));
      }
    }
    for (auto const& child : op.GetChildren()) {
      self(self, child.get());
    }
  };
  visit(visit, root);
  return captured;
}

bound_read_view capture_bound_read_view(sirius::op::sirius_physical_table_scan const& get,
                                        duckdb::ClientContext& context,
                                        std::span<std::string const> resolved_paths)
{
  return capture({get.function,
                  get.bind_data.get(),
                  get.parameters,
                  &get.named_parameters,
                  resolved_paths,
                  false},
                 context);
}
}  // namespace sirius::op::scan
