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

#include "ducklake/binding_bridge.hpp"

#include "duckdb/common/file_system.hpp"
#include "duckdb/common/multi_file/multi_file_function.hpp"
#include "duckdb/planner/filter/table_filter_functions.hpp"
#include "duckdb/storage/external_file_cache/caching_file_system.hpp"
#include "ducklake/dv_visibility.hpp"
#include "ducklake/field_mapping.hpp"
#include "ducklake/inline_batch_producer.hpp"
#include "ducklake/inline_reader.hpp"
#include "ducklake/parquet_unit_ingestible.hpp"
#include "ducklake/provider_api.hpp"
#include "ducklake/remote_io.hpp"
#include "ducklake/source_policy.hpp"
#include "io/s3/duckdb_secret_config.hpp"
#include "io/s3/sirius_httpfs.hpp"
#include "op/scan/parquet_delete_reader.hpp"
#include "parquet_multi_file_info.hpp"
#include "planner/connector_registry.hpp"
#include "storage/ducklake_stats.hpp"
#include "storage/ducklake_table_entry.hpp"

#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/object_identity.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/common/multi_file/table_function_multi_file.hpp>
#include <duckdb/main/extension_manager.hpp>
#include <duckdb/transaction/meta_transaction.hpp>

#include <stdexcept>
#include <thread>

// Include io_uring after Parquet has declared its BLOCK_SIZE constant.
#include "sirius_context.hpp"

namespace sirius::ducklake {
using namespace duckdb;
namespace {
DuckLakeFunctionInfo const* read_info(provider_api const& api,
                                      BaseTableFunction const& function,
                                      FunctionData const* data)
{
  if (function.get_multi_file_reader != api.create_reader) return nullptr;
  auto bind = dynamic_cast<MultiFileBindData const*>(data);
  auto info = api.function_info(function.function_info.get());
  if (!bind || !info || !bind->file_list ||
      !dynamic_cast<TableFunctionMultiFileWrapper const*>(bind->interface.get()))
    return nullptr;
  auto reader = api.reader(bind->multi_file_reader.get());
  if (!reader || !api.file_list(bind->file_list.get()) || &reader->read_info != info)
    return nullptr;
  return info;
}
void field(std::string& out, std::string const& value)
{
  out += std::to_string(value.size()) + ":" + value;
}
}  // namespace
duckdb::vector<TableFunction> trusted_scan_functions(DatabaseInstance& db,
                                                     TableFunction const& parquet,
                                                     provider_api const& api)
{
  if (!ExtensionManager::Get(db).ExtensionIsLoaded("parquet")) return {};
  auto provider = api.scan_function(db);
  if (provider.get_multi_file_reader != api.create_reader) return {};
  // The provider factory inherits mutable catalog Parquet callbacks. Start from
  // the host-verified definition instead and copy only provider-owned overrides.
  auto trusted = parquet;
  trusted.SetName("ducklake_scan");
  trusted.get_multi_file_reader = provider.get_multi_file_reader;
  trusted.statistics            = provider.statistics;
  trusted.statistics_extended   = provider.statistics_extended;
  trusted.get_table_entry       = provider.get_table_entry;
  trusted.get_virtual_columns   = provider.get_virtual_columns;
  trusted.get_row_id_columns    = provider.get_row_id_columns;
  trusted.get_partition_stats   = provider.get_partition_stats;
  trusted.serialize             = provider.serialize;
  trusted.deserialize           = provider.deserialize;
  return {std::move(trusted)};
}
namespace {
DuckLakeFunctionInfo const* verified_info(std::shared_ptr<const provider_api> const& api,
                                          BaseTableFunction const& function,
                                          FunctionData const* data,
                                          ClientContext& context)
{
  if (!api || !api->matches_loaded_file() || !context.transaction.HasActiveTransaction())
    return nullptr;
  auto info = read_info(*api, function, data);
  if (!info || !planner::verified_parquet_interface(data, context)) return nullptr;
  auto transaction = info->transaction.lock();
  if (!transaction) return nullptr;
  // Observe only an existing provider transaction in the same connection.
  auto existing = context.transaction.ActiveTransaction().TryGetTransaction(
    info->table.ParentCatalog().GetAttached());
  return existing && existing.get() == transaction.get() ? info : nullptr;
}
}  // namespace
bool binding_matches(BaseTableFunction const& function,
                     FunctionData const* data,
                     ClientContext& context)
{
  return verified_info(
           planner::registered_ducklake_provider(*context.db), function, data, context) != nullptr;
}
namespace {
// MultiFileBindData::Copy also copies the file inventory. Retain the original
// list and copy only configuration/factories; never copy initial/union readers.
unique_ptr<MultiFileBindData> copy_reader_binding(MultiFileBindData const& original)
{
  auto copy = make_uniq<MultiFileBindData>();
  if (original.bind_data)
    copy->bind_data = unique_ptr_cast<FunctionData, TableFunctionData>(original.bind_data->Copy());
  copy->file_list         = original.file_list;
  copy->multi_file_reader = original.multi_file_reader->Copy();
  copy->interface         = original.interface->Copy();
  copy->columns           = original.columns;
  copy->reader_bind       = original.reader_bind;
  copy->file_options      = original.file_options;
  copy->types             = original.types;
  copy->names             = original.names;
  copy->virtual_columns   = original.virtual_columns;
  copy->table_columns     = original.table_columns;
  return copy;
}
}  // namespace
struct binding_handle::impl {
  std::shared_ptr<const provider_api> api;
  duckdb::weak_ptr<ClientContext> context;
  duckdb::weak_ptr<DuckLakeTransaction> transaction;
  uint64_t statement, transaction_id;
  std::thread::id owner          = std::this_thread::get_id();
  bool execution_bound           = false;
  mutable bool discovery_started = false;
  // Keep only original binding metadata and a separate provider factory. The
  // transaction/context remain weak; no CPU global/local/initial reader is held.
  duckdb::shared_ptr<const DuckLakeFunctionInfo> function_info;
  // reader_binding owns the list, including any replacement from filter pushdown.
  DuckLakeMultiFileList const* list;
  duckdb::unique_ptr<MultiFileBindData> reader_binding;
  op::scan::bound_read_view captured;
  impl(BaseTableFunction const& function, FunctionData const* data, ClientContext& c)
    : api(planner::registered_ducklake_provider(*c.db)), context(c.shared_from_this())
  {
    auto verified = verified_info(api, function, data, c);
    if (!verified || c.transaction.GetActiveQuery() == MAXIMUM_QUERY_ID)
      throw std::invalid_argument("DuckLake binding requires a verified active statement");
    auto const& info = *verified;
    if (info.scan_type != DuckLakeScanType::SCAN_TABLE || info.start_snapshot)
      throw std::invalid_argument("DuckLake scan mode is not ordinary table scan");
    auto active = info.transaction.lock();
    if (!active || info.snapshot.snapshot_id == DConstants::INVALID_INDEX ||
        info.snapshot.schema_version == DConstants::INVALID_INDEX ||
        info.table.GetTableUUID().empty())
      throw std::invalid_argument("DuckLake original binding lacks snapshot or table evidence");
    auto const& bind = *dynamic_cast<MultiFileBindData const*>(data);
    function_info =
      duckdb::shared_ptr<const DuckLakeFunctionInfo>(function.function_info, verified);
    list = api->file_list(bind.file_list.get());
    // DuckLake's verified Copy preserves transaction_local_data and read_info,
    // and creates fresh factory state without opening a physical file reader.
    reader_binding = copy_reader_binding(bind);
    transaction    = active;
    statement      = c.transaction.GetActiveQuery();
    transaction_id = c.transaction.ActiveTransaction().global_transaction_id;
    op::scan::bound_read_identity identity;
    identity.source = {
      "ducklake_scan", op::scan::source_kind::parquet_local, "duckdb.ducklake.local.v1"};
    identity.bound_types = info.column_types;
    identity.bound_names = info.column_names;
    std::string key;
    field(key, "ducklake.binding.v1");
    field(key, std::to_string(info.table.ParentCatalog().GetOid()));
    field(key, info.table.GetTableUUID());
    field(key, std::to_string(info.table_id.index));
    field(key, std::to_string(info.snapshot.snapshot_id));
    field(key, std::to_string(info.snapshot.schema_version));
    field(key, info.include_local_changes ? "1" : "0");
    // Until overlay generation is qualified, conservatively scope correspondence
    // to this statement/transaction. This is not a physical cache identity.
    field(key, std::to_string(transaction_id));
    field(key, std::to_string(statement));
    identity.data_view = op::scan::provider_read_identity{std::move(key)};
    captured.identity  = op::scan::make_bound_read_identity(std::move(identity), {});
    captured.metrics.canonical_capacity = captured.identity->fingerprint.canonical.capacity() + 1;
    captured.transaction_id             = transaction_id;
    captured.provider = op::scan::provider_borrow{statement, transaction_id, &c, nullptr};
    // No byte inventory has been qualified yet; do not invent a local/S3 claim.
    captured.replay_policy = {
      "ducklake_scan", transparent::byte_source_class::unclassified, true, {}};
  }
  void validate() const
  {
    if (std::this_thread::get_id() != owner)
      throw std::runtime_error("DuckLake provider work must run on its statement owner thread");
    validate_context();
  }
  void validate_context() const
  {
    auto c = context.lock();
    if (!c || !c->transaction.HasActiveTransaction() ||
        c->transaction.GetActiveQuery() != statement ||
        c->transaction.ActiveTransaction().global_transaction_id != transaction_id ||
        transaction.expired())
      throw std::runtime_error("DuckLake binding is outside its captured statement/transaction");
  }
};
binding_handle::binding_handle(std::unique_ptr<impl> p) : impl_(std::move(p)) {}
binding_handle::~binding_handle() = default;
std::shared_ptr<binding_handle> binding_handle::capture(BoundTableFunction const& function,
                                                        FunctionData const* data,
                                                        ClientContext& context)
{
  auto const* source = planner::lookup_connector(function, data, context);
  if (!source || source->function_name != "ducklake_scan")
    throw std::invalid_argument("DuckLake capture requires verified source callbacks");
  return std::shared_ptr<binding_handle>(
    new binding_handle(std::make_unique<impl>(function, data, context)));
}
op::scan::bound_read_view const& binding_handle::view() const noexcept { return impl_->captured; }
void binding_handle::validate_statement() const { impl_->validate(); }
transparent::scan_source_policy classify_bound_sources(BoundTableFunction const& function,
                                                       FunctionData const* data,
                                                       ClientContext& context)
{
  auto api  = planner::registered_ducklake_provider(*context.db);
  auto info = verified_info(api, function, data, context);
  if (!info)
    throw std::invalid_argument("DuckLake source policy requires a verified original binding");
  auto const& bind = *dynamic_cast<MultiFileBindData const*>(data);
  auto const* list = api->file_list(bind.file_list.get());
  dependency_source_policy policy;
  // The bound provider list includes transaction-local changes and snapshot-selected
  // delete descriptors. Inline table names are not filesystem paths. No Parquet or
  // Puffin reader is opened by this inspection.
  auto const& files = (list->*api->get_files)();
  for (size_t i = 0; i < files.size(); ++i) {
    auto const& entry = files[i];
    if (entry.data_type == DuckLakeDataType::DATA_FILE) {
      policy.observe(entry.file.path);
      if (info->scan_type == DuckLakeScanType::SCAN_DELETIONS) {
        // GetFiles initialized the parallel deletion-scan descriptors. Inline entries
        // appended to the file list have no corresponding deletion-scan descriptor.
        auto& mutable_list   = const_cast<DuckLakeMultiFileList&>(*list);
        auto const& deletion = (mutable_list.*api->get_delete_scan_entry)(i);
        policy.observe(deletion.delete_file.path);
        policy.observe(deletion.previous_delete_file.path);
      }
    }
    policy.observe(entry.delete_file.path);
  }
  return policy.finish();
}

void binding_handle::bind_execution_owner()
{
  if (impl_->execution_bound || impl_->discovery_started) {
    impl_->validate();
    return;
  }
  impl_->validate_context();
  impl_->owner           = std::this_thread::get_id();
  impl_->execution_bound = true;
}
void binding_handle::configure_scan(std::vector<size_t> const& selected, TableFilterSet* filters)
{
  validate_statement();
  if (impl_->discovery_started)
    throw std::logic_error("DuckLake filters must be configured before unit discovery");
  duckdb::vector<duckdb::ColumnIndex> columns;
  for (auto index : selected)
    columns.emplace_back(index);
  auto context = impl_->context.lock();
  auto& bind   = *impl_->reader_binding;
  if (filters) {
    MultiFileDynamicPushdownInfo info(
      *context, bind.file_options, bind.names, bind.types, columns, *filters);
    auto filtered = bind.multi_file_reader->DynamicFilterPushdown(*bind.file_list, info);
    if (filtered) {
      auto* list = impl_->api->file_list(filtered.get());
      if (!list)
        throw std::invalid_argument("DuckLake filter pushdown returned a foreign file list");
      bind.file_list = shared_ptr<MultiFileList>(std::move(filtered));
      impl_->list    = list;
    }
  }
  // Committed inline rows may be read while expanding the file list.
  // Publish the projection before discovery, not only when opening its reader.
  auto& schema = bind.reader_bind.schema.empty() ? bind.columns : bind.reader_bind.schema;
  (impl_->list->*impl_->api->set_scan_columns)(schema, columns);
}

bool binding_handle::can_prune_unit(uint64_t index) const
{
  validate_statement();
  auto& list       = *impl_->list;
  auto const& file = (list.*impl_->api->get_files)()[index];
  auto filters     = list.GetFilterInfo();
  if (!filters || file.data_type != DuckLakeDataType::DATA_FILE) return false;
  for (auto const& entry : filters->column_filters) {
    auto const& column = entry.second;
    auto dynamic       = ExpressionFilter::GetOptionalDynamicFilterData(*column.table_filter);
    auto stats         = file.column_min_max.find(column.column_field_index);
    if (!dynamic || stats == file.column_min_max.end()) continue;
    // Offset-free bounds cannot establish a TIMESTAMPTZ comparison.
    if (StatsBoundsRequireOffset(column.column_type)) continue;
    ExpressionType comparison;
    Value threshold;
    {
      annotated_lock_guard<annotated_shared_mutex> guard(dynamic->lock);
      if (!dynamic->initialized) continue;
      comparison = dynamic->comparison_type;
      threshold  = dynamic->constant;
    }
    auto const& bounds = stats->second;
    string bound;
    switch (comparison) {
      case ExpressionType::COMPARE_GREATERTHAN:
      case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
        if (!bounds.has_max) continue;
        bound = bounds.max;
        break;
      case ExpressionType::COMPARE_LESSTHAN:
      case ExpressionType::COMPARE_LESSTHANOREQUALTO:
        if (!bounds.has_min) continue;
        bound = bounds.min;
        break;
      default: continue;
    }
    auto value    = Value(bound).DefaultTryCastAs(column.column_type);
    auto constant = threshold.DefaultTryCastAs(column.column_type);
    if (value && constant && !value->IsNull() && !constant->IsNull() &&
        !DynamicFilterData::CompareValue(comparison, *constant, *value))
      return true;
  }
  return false;
}

std::optional<unit_descriptor> binding_handle::unit_at(uint64_t index) const
{
  validate_statement();
  impl_->discovery_started = true;
  auto& list               = *impl_->list;
  auto const& entries      = (list.*impl_->api->get_files)();
  if (index >= entries.size()) return std::nullopt;
  auto const& entry = entries[index];
  auto const& info  = *impl_->function_info;
  unit_descriptor result;
  result.original_file_index = index;
  result.read_view           = impl_->captured.identity;
  MultiFileListScanData cursor;
  cursor.current_file_idx = index;
  if (!list.Scan(cursor, result.input) || result.input.path != entry.file.path)
    throw std::runtime_error("DuckLake typed entry and original open-file descriptor disagree");
  switch (entry.data_type) {
    case DuckLakeDataType::DATA_FILE: result.kind = unit_kind::parquet; break;
    case DuckLakeDataType::INLINED_DATA: result.kind = unit_kind::inline_table; break;
    case DuckLakeDataType::TRANSACTION_LOCAL_INLINED_DATA:
      result.kind = unit_kind::transaction_local_inline;
      break;
    default: throw std::runtime_error("Unknown DuckLake unit type");
  }
  auto optional = [](optional_idx value) -> std::optional<uint64_t> {
    return value.IsValid() ? std::optional<uint64_t>(value.GetIndex()) : std::nullopt;
  };
  result.bound_snapshot       = info.snapshot.snapshot_id;
  result.bound_schema_version = info.snapshot.schema_version;
  if (entry.file_id.IsValid()) result.file_id = entry.file_id.index;
  if (entry.mapping_id.IsValid()) result.mapping_id = entry.mapping_id.index;
  result.data_file_id       = optional(entry.data_file_id);
  result.row_id_start       = optional(entry.row_id_start);
  result.insertion_snapshot = optional(entry.snapshot_id);
  result.snapshot_min       = optional(entry.snapshot_filter_min);
  result.snapshot_max       = optional(entry.snapshot_filter_max);
  result.file_size          = entry.file.file_size_bytes;
  result.footer_size        = optional(entry.file.footer_size);
  result.encrypted          = !entry.file.encryption_key.empty();
  result.delete_path        = entry.delete_file.path;
  result.delete_file_size   = entry.delete_file.file_size_bytes;
  result.delete_footer_size = optional(entry.delete_file.footer_size);
  result.delete_encrypted   = !entry.delete_file.encryption_key.empty();
  if (!result.delete_path.empty()) {
    switch (entry.delete_file.format) {
      case DeleteFileFormat::PARQUET: result.deletes = delete_format::parquet; break;
      case DeleteFileFormat::PUFFIN: result.deletes = delete_format::puffin; break;
      default: result.deletes = delete_format::unknown; break;
    }
  }
  result.inline_delete_count = entry.inlined_file_deletions.size();
  auto transaction           = impl_->transaction.lock();
  if (!transaction) throw std::runtime_error("DuckLake transaction expired during discovery");
  result.transaction_local_delete =
    info.include_local_changes &&
    ((*transaction).*impl_->api->has_local_delete)(info.table_id, entry.file.path);
  bool inlined = false, local = false;
  uint64_t schema_version;
  if (result.input.extended_info) {
    result.input.extended_info->TryGetOption("inlined_data", inlined);
    result.input.extended_info->TryGetOption("transaction_local_data", local);
    if (result.input.extended_info->TryGetOption("schema_version", schema_version))
      result.inline_schema_version = schema_version;
  }
  if (inlined != (result.kind != unit_kind::parquet) ||
      local != (result.kind == unit_kind::transaction_local_inline) ||
      (result.kind == unit_kind::inline_table && !result.inline_schema_version))
    throw std::runtime_error("DuckLake typed entry lacks matching inline/schema/local evidence");
  return result;
}
position_page binding_handle::inline_delete_positions(uint64_t index,
                                                      std::span<uint64_t> destination,
                                                      std::optional<uint64_t> after) const
{
  validate_statement();
  if (destination.empty()) throw std::invalid_argument("DuckLake delete page needs storage");
  auto& list          = *impl_->list;
  auto const& entries = (list.*impl_->api->get_files)();
  if (index >= entries.size()) throw std::out_of_range("DuckLake original file index");
  auto const& positions = entries[index].inlined_file_deletions;
  auto next             = after ? positions.upper_bound(*after) : positions.begin();
  position_page result;
  while (next != positions.end() && result.size < destination.size())
    destination[result.size++] = *next++;
  result.more = next != positions.end();
  return result;
}
std::shared_ptr<dv_visibility const> binding_handle::read_parquet_deletes(
  uint64_t index,
  scan_manager::charging_allocator& allocator,
  scan_manager::preparation_stop_token stop,
  op::scan::physical_check_counters* counters,
  std::shared_ptr<cucascade::io::ioctx> delete_context)
{
  validate_statement();
  auto unit = unit_at(index);
  if (!unit || unit->deletes != delete_format::parquet)
    throw std::invalid_argument(
      "DuckLake ordinary delete reader requires an original Parquet delete unit");
  (void)parquet_unit_envelope(*unit, dependency_scope::s3_deletes);
  auto context    = impl_->context.lock();
  auto check_stop = [&] {
    if (stop.stop_requested())
      throw std::system_error(std::make_error_code(std::errc::operation_canceled));
  };
  check_stop();
  bool remote = io::s3::is_s3_path(unit->delete_path);
  std::optional<io::s3::scoped_native_read_context> access;
  if (remote) access.emplace(*context, unit->delete_path, std::move(delete_context));
  OpenFileInfo file(unit->delete_path);
  if (remote) {
    file.extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
    file.extended_info->options["sirius_require_strong_identity"] = Value::BOOLEAN(true);
    file.extended_info->options["sirius_expected_size"] = Value::UBIGINT(unit->delete_file_size);
  }
  auto result = std::make_shared<dv_visibility>();
  std::optional<io::s3::validated_file_identity> remote_identity;
  std::optional<parquet_delete_accumulator> accumulator;
  bool embedded = false;
  op::scan::parquet_delete_read_options options;
  bool ordered = false;
  std::span<int64_t> positions;
  options.expected_size          = unit->delete_file_size;
  options.expected_footer_size   = unit->delete_footer_size;
  options.verify_unbound_trailer = !remote;
  options.check_stop             = check_stop;
  options.verify_remote          = [&](CachingFileHandle& handle) {
    if (!remote)
      throw cucascade::io::descriptor_mismatch_error(
        "DuckLake local Parquet delete requires a regular file");
    auto current = io::s3::validated_identity(*handle.GetFileHandle());
    // The initial footer check runs on the owner before any workers are scheduled.
    if (!remote_identity)
      remote_identity = current;
    else if (current != *remote_identity)
      throw cucascade::io::object_changed_error(
        unit->delete_path, remote_identity->object_key, current.object_key);
  };
  op::scan::read_parquet_delete_batches(
    *context,
    std::move(file),
    options,
    [&](auto const& columns, uint64_t rows) {
      if ((columns.size() != 2 && columns.size() != 3) ||
          columns[0].type != duckdb::LogicalType::VARCHAR ||
          columns[1].type != duckdb::LogicalType::BIGINT)
        throw std::runtime_error(
          "DuckLake Parquet delete schema requires file name and BIGINT position");
      embedded = columns.size() == 3 &&
                 StringUtil::CIEquals(columns[2].name, "_ducklake_internal_snapshot_id");
      if (embedded && columns[2].type != duckdb::LogicalType::BIGINT)
        throw std::runtime_error("DuckLake Parquet delete snapshot requires BIGINT");
      if (rows > dv_max_positions)
        throw scan_manager::preparation_resource_error(
          "DuckLake Parquet delete positions exceed limit", true);
      result->merged = allocator.allocate_retained(rows * sizeof(int64_t));
      positions = {reinterpret_cast<int64_t*>(result->merged.data()), static_cast<size_t>(rows)};
      accumulator.emplace(
        positions, embedded, unit->bound_snapshot, unit->transaction_local_delete);
      return embedded ? std::vector<uint64_t>{1, 2} : std::vector<uint64_t>{1};
    },
    [&](bool in_file_order) {
      ordered = in_file_order;
      return [&, in_file_order](uint64_t offset, DataChunk& chunk) {
        UnifiedVectorFormat pos;
        chunk.data[0].ToUnifiedFormat(pos);
        UnifiedVectorFormat snapshots_column;
        if (embedded) chunk.data[1].ToUnifiedFormat(snapshots_column);
        auto read = [](UnifiedVectorFormat const& column, idx_t row) -> std::optional<int64_t> {
          auto index = column.sel->get_index(row);
          if (!column.validity.RowIsValid(index)) return std::nullopt;
          return column.GetData<int64_t>()[index];
        };
        for (idx_t row = 0; row < chunk.size(); ++row) {
          auto position = read(pos, row);
          auto snapshot = embedded ? read(snapshots_column, row) : std::nullopt;
          if (in_file_order)
            accumulator->append(position, snapshot);
          else
            accumulator->stage(offset + row, position, snapshot);
        }
      };
    });
  validate_statement();
  if (!ordered) {
    if (embedded && counters && counters->track_units)
      counters->ducklake_parallel_snapshot_delete_rows.fetch_add(positions.size(),
                                                                 std::memory_order_relaxed);
    for (size_t row = 0; row < positions.size(); ++row) {
      if (row % STANDARD_VECTOR_SIZE == 0) check_stop();
      accumulator->consume_staged(row);
    }
  }
  check_stop();
  // The shared reader verifies local physical identity across every handle.
  // This local token identifies this completed read only; no cross-read cache
  // may treat it as a reusable file version. Remote reads retain backend identity.
  result->delete_identity =
    remote_identity
      ? remote_identity->object_key
      : "duckdb.parquet.read:" + std::to_string(cucascade::io::next_open_generation());
  result->positions               = accumulator->positions();
  result->last_validated_position = accumulator->last_position();
  result->parquet_validated       = true;
  return result;
}
std::shared_ptr<parquet_unit_ingestible> binding_handle::make_parquet_unit(
  uint64_t index,
  uint64_t contract,
  uint64_t generation,
  std::shared_ptr<scan_manager::preparation_admission> admission,
  std::vector<size_t> required_columns,
  bool count_only,
  unit_scan_options options)
{
  if (count_only && (!required_columns.empty() || options.filters || options.output_columns))
    throw std::invalid_argument(
      "DuckLake count-only cannot request user columns or a filter projection");
  auto unit = unit_at(index);
  if (!unit) throw std::out_of_range("DuckLake original file index");
  auto context = impl_->context.lock();
  auto runtime = context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!runtime) throw std::logic_error("DuckLake preparation requires SiriusContext");
  auto& manager = runtime->get_scan_manager();
  validate_unit_io(*unit, admission && admission->decision.deferred, [&](auto const& path) {
    return manager.planning_io_backend(path);
  });
  auto const& defaults = runtime->get_config().get_scan_manager_config().object_store;
  auto data_context =
    io::s3::capture_duckdb_s3_context(*context, manager, unit->input.path, defaults);
  auto delete_context =
    io::s3::capture_duckdb_s3_context(*context, manager, unit->delete_path, defaults);
  auto const& info = *impl_->function_info;
  std::vector<op::scan::field_requirement> fields;
  auto const& schema = info.table.GetFieldData().GetFieldIds();
  if (schema.size() != info.column_types.size())
    throw std::invalid_argument("DuckLake original field schema is inconsistent");
  if (!count_only)
    required_columns = required_field_indices(schema.size(), std::move(required_columns));
  for (auto i : required_columns) {
    auto const& field = *schema[i];
    if (field.HasChildren() || field.Name() != info.column_names[i] ||
        field.Type() != info.column_types[i] || !field.GetFieldIndex().IsValid() ||
        field.GetFieldIndex().index > INT32_MAX)
      throw std::invalid_argument("DuckLake plain Parquet requires original flat field identities");
    fields.push_back(bind_field(static_cast<int32_t>(field.GetFieldIndex().index),
                                field.Type(),
                                field.GetColumnData().initial_default));
  }
  return std::shared_ptr<parquet_unit_ingestible>(
    new parquet_unit_ingestible(shared_from_this(),
                                std::move(*unit),
                                std::move(fields),
                                std::move(required_columns),
                                contract,
                                generation,
                                std::move(admission),
                                count_only,
                                std::move(options),
                                std::move(data_context),
                                std::move(delete_context)));
}
struct inline_reader::impl {
  std::shared_ptr<binding_handle> handle;
  ClientContext& context;
  unique_ptr<MultiFileBindData> bind;
  MultiFileGlobalState global;
  MultiFileLocalState local;
  unique_ptr<MultiFileReaderData> reader;
  DataChunk output;
  bool finished = false;
  impl(std::shared_ptr<binding_handle> h,
       ClientContext& c,
       unique_ptr<MultiFileBindData> b,
       unit_descriptor const& unit,
       std::vector<size_t> const& selected)
    : handle(std::move(h)), context(c), bind(std::move(b)), global(*bind->file_list), local(c)
  {
    duckdb::vector<duckdb::LogicalType> output_types;
    for (auto index : selected) {
      if (index >= bind->types.size())
        throw std::invalid_argument("DuckLake inline projection index");
      global.column_indexes.emplace_back(index);
      output_types.push_back(bind->types[index]);
    }
    if (selected.empty()) {
      global.column_indexes.emplace_back(COLUMN_IDENTIFIER_EMPTY);
      output_types.push_back(duckdb::LogicalType::BOOLEAN);
    }
    auto& columns = bind->reader_bind.schema.empty() ? bind->columns : bind->reader_bind.schema;
    global.multi_file_reader_state =
      bind->multi_file_reader->InitializeGlobalState(context,
                                                     bind->file_options,
                                                     bind->reader_bind,
                                                     *bind->file_list,
                                                     columns,
                                                     global.column_indexes);
    global.global_state = bind->interface->InitializeGlobalState(context, *bind, global);
    auto opened         = bind->multi_file_reader->CreateReader(
      context, *global.global_state, unit.input, unit.original_file_index, *bind);
    reader     = make_uniq<MultiFileReaderData>(std::move(opened));
    auto state = MultiFileFunction<ParquetMultiFileInfo>::InitializeReader(
      *reader, *bind, global.column_indexes, nullptr, context, unit.original_file_index, global);
    output.Initialize(context, output_types);
    if (state == ReaderInitializeType::SKIP_READING_FILE) {
      finished = true;
      return;
    }
    reader->reader->PrepareReader(context, *global.global_state);
    local.job              = make_uniq<ScanReadAheadJobWrapper<MultiFileScanJobState>>();
    local.job->reader      = reader->reader;
    local.job->reader_data = reader.get();
    local.job->file_index  = unit.original_file_index;
    local.job->scan_state  = bind->interface->InitializeLocalState(context, *global.global_state);
    if (!reader->reader->TryInitializeScan(context, *global.global_state, *local.job->scan_state)) {
      finished = true;
      return;
    }
    MultiFileFunction<ParquetMultiFileInfo>::InitializeDecodeChunk(context, local, global);
  }
  ~impl()
  {
    local.scan_chunk.Destroy();
    output.Destroy();
    local.executor.ClearExpressions();
    local.job.reset();
    reader.reset();
  }
  DataChunk const* next()
  {
    handle->validate_statement();
    if (finished) return nullptr;
    local.scan_chunk.Reset();
    output.Reset();
    auto result =
      reader->reader->Scan(context, *global.global_state, *local.job->scan_state, local.scan_chunk)
        .GetResultType();
    if (result != AsyncResultType::HAVE_MORE_OUTPUT && result != AsyncResultType::FINISHED)
      throw std::runtime_error(
        "DuckLake inline reader returned an unsupported asynchronous result");
    output.SetChildCardinality(local.scan_chunk.size());
    if (local.scan_chunk.size())
      bind->multi_file_reader->FinalizeChunk(context,
                                             *bind,
                                             *reader->reader,
                                             *reader,
                                             local.scan_chunk,
                                             output,
                                             local.executor,
                                             global.multi_file_reader_state);
    finished = result == AsyncResultType::FINISHED;
    return &output;
  }
};
inline_reader::inline_reader(std::unique_ptr<impl> value) : impl_(std::move(value)) {}
inline_reader::~inline_reader() = default;
duckdb::DataChunk const* inline_reader::next() { return impl_->next(); }
std::unique_ptr<inline_reader> binding_handle::open_inline_unit(uint64_t index,
                                                                std::vector<size_t> const& selected)
{
  auto unit = unit_at(index);
  if (!unit || unit->kind == unit_kind::parquet || !assess_local_scope(*unit).reasons.empty() ||
      unit->inline_delete_count || unit->transaction_local_delete)
    throw std::invalid_argument("DuckLake inline unit has unqualified dependencies");
  auto context = impl_->context.lock();
  if (!context) throw std::runtime_error("DuckLake inline statement expired");
  return std::unique_ptr<inline_reader>(new inline_reader(std::make_unique<inline_reader::impl>(
    shared_from_this(), *context, copy_reader_binding(*impl_->reader_binding), *unit, selected)));
}
std::shared_ptr<op::scan::arrow_import_ingestible> binding_handle::make_inline_unit(
  uint64_t index,
  uint64_t contract,
  uint64_t generation,
  std::shared_ptr<scan_manager::preparation_admission> admission,
  bool count_only,
  std::vector<size_t> required_columns,
  unit_scan_options options)
{
  auto unit = unit_at(index);
  if (!unit) throw std::out_of_range("DuckLake original inline index");
  auto producer =
    std::shared_ptr<inline_batch_producer>(new inline_batch_producer(shared_from_this(),
                                                                     std::move(*unit),
                                                                     contract,
                                                                     generation,
                                                                     std::move(admission),
                                                                     count_only,
                                                                     std::move(required_columns),
                                                                     std::move(options)));
  auto bound = producer->take_binding();
  return std::make_shared<op::scan::arrow_import_ingestible>(std::move(bound), std::move(producer));
}
op::scan::bound_read_view capture_read_view(BoundTableFunction const& function,
                                            FunctionData const* data,
                                            ClientContext& context)
{
  return binding_handle::capture(function, data, context)->view();
}
}  // namespace sirius::ducklake
