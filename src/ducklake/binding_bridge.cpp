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

#include "ducklake/dv_visibility.hpp"
#include "ducklake/field_mapping.hpp"
#include "ducklake/inline_reader.hpp"
#include "ducklake/inline_unit_ingestible.hpp"
#include "ducklake/parquet_unit_ingestible.hpp"
#include "io/io_errors.hpp"
#include "planner/connector_registry.hpp"

#include <stdexcept>
#include <thread>

#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/multi_file/multi_file_function.hpp"
#include "duckdb/storage/external_file_cache/caching_file_system.hpp"
#include "parquet_multi_file_info.hpp"
#include "storage/ducklake_multi_file_list.hpp"
#include "storage/ducklake_multi_file_reader.hpp"
#include "storage/ducklake_table_entry.hpp"
#include "storage/ducklake_transaction.hpp"

#include <dlfcn.h>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/common/multi_file/table_function_multi_file.hpp>
#include <duckdb/main/extension_manager.hpp>
#include <duckdb/transaction/meta_transaction.hpp>

#include <filesystem>
#endif

namespace sirius::ducklake {
using namespace duckdb;
bool bridge_available() noexcept
{
#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
  return true;
#else
  return false;
#endif
}
#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
namespace {
// Verify the *loaded* provider, not just its source directory or function name.
bool qualified_provider(DatabaseInstance& db)
{
  auto& manager = ExtensionManager::Get(db);
  if (!manager.ExtensionIsLoaded("ducklake")) return false;
  auto info = manager.GetExtensionInfo("ducklake");
  if (!info || !info->install_info || info->install_info->version != "80e6155e41" ||
      info->install_info->full_path.empty())
    return false;
  Dl_info code{};
  if (!::dladdr(reinterpret_cast<void*>(&DuckLakeFunctions::GetDuckLakeScanFunction), &code) ||
      !code.dli_fname)
    return false;
  std::error_code error;
  return std::filesystem::equivalent(info->install_info->full_path, code.dli_fname, error) &&
         !error;
}
DuckLakeFunctionInfo const* read_info(BaseTableFunction const& function, FunctionData const* data)
{
  auto bind = dynamic_cast<MultiFileBindData const*>(data);
  auto info = dynamic_cast<DuckLakeFunctionInfo const*>(function.function_info.get());
  if (!bind || !info || !bind->file_list ||
      !dynamic_cast<TableFunctionMultiFileWrapper const*>(bind->interface.get()))
    return nullptr;
  auto reader = dynamic_cast<DuckLakeMultiFileReader const*>(bind->multi_file_reader.get());
  if (!reader || !dynamic_cast<DuckLakeMultiFileList const*>(bind->file_list.get()) ||
      &reader->read_info != info)
    return nullptr;
  return info;
}
void field(std::string& out, std::string const& value)
{
  out += std::to_string(value.size()) + ":" + value;
}
}  // namespace
duckdb::vector<TableFunction> trusted_scan_functions(DatabaseInstance& db,
                                                     TableFunction const& parquet)
{
  if (!qualified_provider(db) || !ExtensionManager::Get(db).ExtensionIsLoaded("parquet")) return {};
  auto provider = DuckLakeFunctions::GetDuckLakeScanFunction(db);
  if (provider.get_multi_file_reader != &DuckLakeMultiFileReader::CreateInstance) return {};
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
bool binding_matches(BaseTableFunction const& function,
                     FunctionData const* data,
                     ClientContext& context)
{
  if (!qualified_provider(*context.db) || !context.transaction.HasActiveTransaction()) return false;
  auto info = read_info(function, data);
  if (!info || !planner::verified_parquet_interface(data, context)) return false;
  auto transaction = info->transaction.lock();
  if (!transaction) return false;
  // TryGetTransaction observes an existing provider transaction; it must not
  // start one to make a foreign binding appear valid in this connection.
  auto existing = context.transaction.ActiveTransaction().TryGetTransaction(
    info->table.ParentCatalog().GetAttached());
  return existing && existing.get() == transaction.get();
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
  duckdb::weak_ptr<ClientContext> context;
  duckdb::weak_ptr<DuckLakeTransaction> transaction;
  uint64_t statement, transaction_id;
  std::thread::id owner          = std::this_thread::get_id();
  bool execution_bound           = false;
  mutable bool discovery_started = false;
  // Keep only original binding metadata and a separate provider factory. The
  // transaction/context remain weak; no CPU global/local/initial reader is held.
  duckdb::shared_ptr<TableFunctionInfo> function_info;
  duckdb::shared_ptr<MultiFileList> files;
  duckdb::unique_ptr<MultiFileBindData> reader_binding;
  op::scan::bound_read_view captured;
  impl(BaseTableFunction const& function, FunctionData const* data, ClientContext& c)
    : context(c.shared_from_this())
  {
    if (!binding_matches(function, data, c) || !c.transaction.HasActiveTransaction() ||
        c.transaction.GetActiveQuery() == MAXIMUM_QUERY_ID)
      throw std::invalid_argument("DuckLake binding requires a verified active statement");
    auto const& info = *read_info(function, data);
    if (info.scan_type != DuckLakeScanType::SCAN_TABLE || info.start_snapshot)
      throw std::invalid_argument("DuckLake scan mode is not ordinary table scan");
    auto active = info.transaction.lock();
    if (!active || info.snapshot.snapshot_id == DConstants::INVALID_INDEX ||
        info.snapshot.schema_version == DConstants::INVALID_INDEX ||
        info.table.GetTableUUID().empty())
      throw std::invalid_argument("DuckLake original binding lacks snapshot or table evidence");
    auto const& bind = *dynamic_cast<MultiFileBindData const*>(data);
    function_info    = function.function_info;
    files            = bind.file_list;
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
#else
struct binding_handle::impl {
  op::scan::bound_read_view captured;
};
duckdb::vector<TableFunction> trusted_scan_functions(DatabaseInstance&, TableFunction const&)
{
  return {};
}
bool binding_matches(BaseTableFunction const&, FunctionData const*, ClientContext&)
{
  return false;
}
#endif
binding_handle::binding_handle(std::unique_ptr<impl> p) : impl_(std::move(p)) {}
binding_handle::~binding_handle() = default;
std::shared_ptr<binding_handle> binding_handle::capture(BoundTableFunction const& function,
                                                        FunctionData const* data,
                                                        ClientContext& context)
{
#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
  auto const* source = planner::lookup_connector(function, data, context);
  if (!source || source->function_name != "ducklake_scan")
    throw std::invalid_argument("DuckLake capture requires verified source callbacks");
  return std::shared_ptr<binding_handle>(
    new binding_handle(std::make_unique<impl>(function, data, context)));
#else
  throw std::runtime_error("DuckLake private binding bridge is not built");
#endif
}
op::scan::bound_read_view const& binding_handle::view() const noexcept { return impl_->captured; }
void binding_handle::validate_statement() const
{
#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
  impl_->validate();
#else
  throw std::runtime_error("DuckLake private binding bridge is not built");
#endif
}
void binding_handle::bind_execution_owner()
{
#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
  if (impl_->execution_bound || impl_->discovery_started) {
    impl_->validate();
    return;
  }
  impl_->validate_context();
  impl_->owner           = std::this_thread::get_id();
  impl_->execution_bound = true;
#else
  throw std::runtime_error("DuckLake private binding bridge is not built");
#endif
}
std::optional<unit_descriptor> binding_handle::unit_at(uint64_t index) const
{
  validate_statement();
#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
  impl_->discovery_started = true;
  auto& list               = dynamic_cast<DuckLakeMultiFileList&>(*impl_->files);
  auto const& entries      = list.GetFiles();
  if (index >= entries.size()) return std::nullopt;
  auto const& entry = list.GetFileEntry(index);
  auto const& info  = dynamic_cast<DuckLakeFunctionInfo const&>(*impl_->function_info);
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
  result.max_row_count      = optional(entry.max_row_count);
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
    transaction->HasLocalDeleteForFile(info.table_id, entry.file.path);
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
#else
  return std::nullopt;
#endif
}
position_page binding_handle::inline_delete_positions(uint64_t index,
                                                      std::span<uint64_t> destination,
                                                      std::optional<uint64_t> after) const
{
  validate_statement();
#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
  if (destination.empty()) throw std::invalid_argument("DuckLake delete page needs storage");
  auto& list = dynamic_cast<DuckLakeMultiFileList&>(*impl_->files);
  if (index >= list.GetFiles().size()) throw std::out_of_range("DuckLake original file index");
  auto const& positions = list.GetFileEntry(index).inlined_file_deletions;
  auto next             = after ? positions.upper_bound(*after) : positions.begin();
  position_page result;
  while (next != positions.end() && result.size < destination.size())
    destination[result.size++] = *next++;
  result.more = next != positions.end();
  return result;
#else
  return {};
#endif
}
std::shared_ptr<dv_visibility const> binding_handle::read_parquet_deletes(
  uint64_t index,
  scan_manager::charging_allocator& allocator,
  scan_manager::preparation_stop_token stop)
{
  validate_statement();
#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
  auto unit = unit_at(index);
  if (!unit || unit->deletes != delete_format::parquet ||
      !parquet_unit_dependencies_supported(*unit))
    throw std::invalid_argument("DuckLake ordinary delete reader requires a local original unit");
  (void)parquet_unit_envelope(*unit);
  auto context    = impl_->context.lock();
  auto check_stop = [&] {
    if (stop.stop_requested())
      throw std::system_error(std::make_error_code(std::errc::operation_canceled));
  };
  check_stop();
  // Construct the host implementation directly: no catalog callback lookup,
  // SQL rebind, new connection, or access to the saved CPU reader.
  ParquetReader reader(*context, OpenFileInfo(unit->delete_path), ParquetOptions(*context));
  auto& opened  = reader.GetHandle();
  auto identity = opened.GetFileHandle()->Stats();
  if (identity.file_type != FileType::FILE_TYPE_REGULAR || !identity.device_id.IsValid() ||
      !identity.file_id.IsValid() || identity.file_size < 0 ||
      static_cast<uint64_t>(identity.file_size) != unit->delete_file_size ||
      opened.GetFileSize() != unit->delete_file_size ||
      opened.GetLastModifiedTime() != identity.last_modification_time)
    throw io::descriptor_mismatch_error("DuckLake Parquet delete object differs from binding");
  auto verify = [&](CachingFileHandle& handle) {
    auto current = handle.GetFileHandle()->Stats();
    if (!current.device_id.IsValid() || !current.file_id.IsValid() ||
        current.device_id.GetIndex() != identity.device_id.GetIndex() ||
        current.file_id.GetIndex() != identity.file_id.GetIndex() ||
        current.file_size != identity.file_size ||
        current.last_modification_time != identity.last_modification_time ||
        current.version_tag != identity.version_tag)
      throw io::object_changed_error(unit->delete_path,
                                     std::to_string(identity.file_size) + ":" +
                                       std::to_string(identity.last_modification_time.value),
                                     std::to_string(current.file_size) + ":" +
                                       std::to_string(current.last_modification_time.value));
  };
  uint8_t tail[8];
  if (unit->delete_file_size < 12)
    throw io::descriptor_mismatch_error("DuckLake Parquet delete file is truncated");
  opened.GetFileHandle()->Read(tail, 8, unit->delete_file_size - 8);
  verify(opened);
  uint64_t footer_bytes = uint64_t(tail[0]) | (uint64_t(tail[1]) << 8) | (uint64_t(tail[2]) << 16) |
                          (uint64_t(tail[3]) << 24);
  if (tail[4] != 'P' || tail[5] != 'A' || tail[6] != 'R' || tail[7] != '1' ||
      footer_bytes > unit->delete_file_size - 12 ||
      (unit->delete_footer_size && *unit->delete_footer_size != footer_bytes))
    throw io::descriptor_mismatch_error("DuckLake Parquet delete footer differs from binding");
  auto const& columns = reader.GetColumns();
  if ((columns.size() != 2 && columns.size() != 3) ||
      columns[0].type != duckdb::LogicalType::VARCHAR ||
      columns[1].type != duckdb::LogicalType::BIGINT)
    throw std::runtime_error(
      "DuckLake Parquet delete schema requires file name and BIGINT position");
  if (columns.size() == 3 && columns[2].name == "_ducklake_internal_snapshot_id")
    throw std::runtime_error(
      "DuckLake snapshot-bearing Parquet deletes are outside the local scan profile");
  if (reader.NumRows() > dv_max_positions)
    throw scan_manager::preparation_resource_error("DuckLake Parquet delete positions exceed limit",
                                                   true);
  auto result     = std::make_shared<dv_visibility>();
  result->merged  = allocator.allocate_retained(reader.NumRows() * sizeof(int64_t));
  auto* positions = reinterpret_cast<int64_t*>(result->merged.data());
  reader.column_indexes.emplace_back(1);
  reader.column_ids.emplace_back(1);
  ParquetReadGlobalState global(nullptr);
  ParquetReadLocalState local;
  DataChunk chunk;
  chunk.Initialize(*context, {duckdb::LogicalType::BIGINT});
  size_t count     = 0;
  int64_t previous = -1;
  while (reader.TryInitializeScan(*context, global, local)) {
    check_stop();
    reader.PrepareScan(*context, global, local);
    verify(*local.scan_state.file_handle);
    auto io = reader.ScheduleIO(*context, global, local);
    if (io.HasTasks()) io.ExecuteTasksSynchronously();
    if (io.GetResultType() != AsyncResultType::HAVE_MORE_OUTPUT)
      throw std::runtime_error("DuckLake Parquet delete I/O did not complete");
    while (true) {
      check_stop();
      verify(*local.scan_state.file_handle);
      chunk.Reset();
      auto scan = reader.Scan(*context, global, local, chunk);
      if (scan.HasTasks()) {
        scan.ExecuteTasksSynchronously();
        continue;
      }
      auto status = scan.GetResultType();
      verify(*local.scan_state.file_handle);
      if (status != AsyncResultType::HAVE_MORE_OUTPUT && status != AsyncResultType::FINISHED)
        throw std::runtime_error("DuckLake Parquet deletes require synchronous native decoding");
      for (idx_t row = 0; row < chunk.size(); ++row) {
        auto value = chunk.GetValue(0, row);
        if (value.IsNull())
          throw std::runtime_error("DuckLake Parquet delete position cannot be NULL");
        auto position = value.GetValue<int64_t>();
        validate_delete_position(position, previous);
        if (count >= reader.NumRows())
          throw std::runtime_error("DuckLake Parquet deletes exceed footer coverage");
        positions[count++] = position;
      }
      if (status == AsyncResultType::FINISHED) break;
    }
  }
  verify(opened);
  check_stop();
  if (count != reader.NumRows())
    throw std::runtime_error("DuckLake Parquet deletes have incomplete footer coverage");
  result->delete_identity =
    "duckdb.local.parquet:" + std::to_string(identity.device_id.GetIndex()) + ":" +
    std::to_string(identity.file_id.GetIndex()) + ":" + std::to_string(identity.file_size) + ":" +
    std::to_string(identity.last_modification_time.value) + ":" + identity.version_tag;
  result->positions         = {positions, count};
  result->parquet_validated = true;
  return result;
#else
  throw std::runtime_error("DuckLake private binding bridge is not built");
#endif
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
#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
  auto const& info = dynamic_cast<DuckLakeFunctionInfo const&>(*impl_->function_info);
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
                                std::move(options)));
#else
  throw std::runtime_error("DuckLake private binding bridge is not built");
#endif
}
#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
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
       unit_descriptor const& unit)
    : handle(std::move(h)), context(c), bind(std::move(b)), global(*bind->file_list), local(c)
  {
    for (idx_t i = 0; i < bind->types.size(); ++i)
      global.column_indexes.emplace_back(i);
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
    output.Initialize(context, bind->types);
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
#else
struct inline_reader::impl {};
#endif
inline_reader::inline_reader(std::unique_ptr<impl> value) : impl_(std::move(value)) {}
inline_reader::~inline_reader() = default;
duckdb::DataChunk const* inline_reader::next()
{
#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
  return impl_->next();
#else
  throw std::runtime_error("DuckLake private binding bridge is not built");
#endif
}
std::unique_ptr<inline_reader> binding_handle::open_inline_unit(uint64_t index)
{
  auto unit = unit_at(index);
  if (!unit || unit->kind == unit_kind::parquet || !assess_local_scope(*unit).reasons.empty() ||
      unit->max_row_count || unit->inline_delete_count || unit->transaction_local_delete)
    throw std::invalid_argument("DuckLake inline unit has unqualified dependencies");
#ifdef SIRIUS_DUCKLAKE_PRIVATE_BRIDGE
  auto context = impl_->context.lock();
  if (!context) throw std::runtime_error("DuckLake inline statement expired");
  return std::unique_ptr<inline_reader>(new inline_reader(std::make_unique<inline_reader::impl>(
    shared_from_this(), *context, copy_reader_binding(*impl_->reader_binding), *unit)));
#else
  throw std::runtime_error("DuckLake private binding bridge is not built");
#endif
}
std::shared_ptr<inline_unit_ingestible> binding_handle::make_inline_unit(
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
  return std::shared_ptr<inline_unit_ingestible>(
    new inline_unit_ingestible(shared_from_this(),
                               std::move(*unit),
                               contract,
                               generation,
                               std::move(admission),
                               count_only,
                               std::move(required_columns),
                               std::move(options)));
}
op::scan::bound_read_view capture_read_view(BoundTableFunction const& function,
                                            FunctionData const* data,
                                            ClientContext& context)
{
  return binding_handle::capture(function, data, context)->view();
}
}  // namespace sirius::ducklake
