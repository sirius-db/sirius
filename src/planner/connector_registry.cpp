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

#include "planner/connector_registry.hpp"

#include "exec/stream_plan_bindings.hpp"
#include "log/logging.hpp"
#include "op/scan/dynamic_filter_merge.hpp"
#include "planner/connector_reference_cache.hpp"
#include "planner/duckdb_host.hpp"
#include "sirius_registration.hpp"

#include <dlfcn.h>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/common/file_system.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/main/extension_helper.hpp>
#include <duckdb/main/extension_manager.hpp>
#include <duckdb/planner/extension_callback.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <parquet_extension.hpp>
#include <parquet_multi_file_info.hpp>

#include <array>
#include <mutex>
#include <tuple>
#include <unordered_set>

namespace sirius::planner {
namespace {
using kind  = op::scan::source_kind;
using mode  = op::scan::dynamic_filter_apply_mode;
using bytes = transparent::byte_source_class;

template <class T>
bool matches_bind(duckdb::FunctionData const* bind)
{
  return dynamic_cast<T const*>(bind) != nullptr;
}

connector make_seq_scan_connector()
{
  return {.function_name              = "seq_scan",
          .kind                       = kind::duckdb_native,
          .registry_profile           = "duckdb.seq_scan.v1",
          .bind_data_matches          = matches_bind<duckdb::TableScanBindData>,
          .lower                      = lower_native_scan,
          .filter_mode                = mode::INCLUDE_AST_ROW_MASKS,
          .byte_source                = bytes::duckdb_native,
          .permits_cpu_replay         = true,
          .selector_outside_bind_data = false,
          .decline_reason             = nullptr,
          .provider                   = std::nullopt};
}

connector make_parquet_scan_connector()
{
  return {.function_name              = "parquet_scan",
          .kind                       = kind::parquet_local,
          .registry_profile           = "duckdb.parquet_scan.v1",
          .bind_data_matches          = matches_bind<duckdb::MultiFileBindData>,
          .lower                      = lower_parquet_scan,
          .filter_mode                = mode::MEMBERSHIP_MASKS_ONLY,
          .byte_source                = bytes::local_file,
          .permits_cpu_replay         = true,
          .selector_outside_bind_data = false,
          .decline_reason             = nullptr,
          .provider                   = std::nullopt};
}

connector make_read_parquet_connector()
{
  return {.function_name              = "read_parquet",
          .kind                       = kind::parquet_local,
          .registry_profile           = "duckdb.read_parquet.v1",
          .bind_data_matches          = matches_bind<duckdb::MultiFileBindData>,
          .lower                      = lower_parquet_scan,
          .filter_mode                = mode::MEMBERSHIP_MASKS_ONLY,
          .byte_source                = bytes::local_file,
          .permits_cpu_replay         = true,
          .selector_outside_bind_data = false,
          .decline_reason             = nullptr,
          .provider                   = std::nullopt};
}

connector make_sirius_read_parquet_connector()
{
  return {.function_name              = "sirius_read_parquet",
          .kind                       = kind::parquet_s3,
          .registry_profile           = "sirius.read_parquet.v1",
          .bind_data_matches          = matches_bind<duckdb::SiriusReadParquetBindData>,
          .lower                      = lower_parquet_scan,
          .filter_mode                = mode::MEMBERSHIP_MASKS_ONLY,
          .byte_source                = bytes::sirius_owned_s3,
          .permits_cpu_replay         = false,
          .selector_outside_bind_data = false,
          .decline_reason             = nullptr,
          .provider                   = std::nullopt};
}

connector make_iceberg_scan_connector()
{
  return {.function_name              = "iceberg_scan",
          .kind                       = kind::parquet_local,
          .registry_profile           = "duckdb.iceberg_scan.v1",
          .bind_data_matches          = matches_bind<duckdb::MultiFileBindData>,
          .lower                      = lower_iceberg_scan,
          .filter_mode                = mode::MEMBERSHIP_MASKS_ONLY,
          .byte_source                = bytes::local_file,
          .permits_cpu_replay         = true,
          .selector_outside_bind_data = true,
          .decline_reason             = registered_iceberg_decline_reason,
          .provider                   = std::nullopt};
}

connector make_sirius_stream_source_connector()
{
  return {.function_name              = "sirius_stream_source",
          .kind                       = kind::stream_source,
          .registry_profile           = "sirius.stream_source.v1",
          .bind_data_matches          = matches_bind<exec::stream_source_bind_data>,
          .lower                      = nullptr,
          .filter_mode                = mode::MEMBERSHIP_MASKS_ONLY,
          .byte_source                = bytes::stream,
          .permits_cpu_replay         = false,
          .selector_outside_bind_data = false,
          .decline_reason             = nullptr,
          .provider                   = std::nullopt};
}

// A .hpln is read by Sirius alone: the CPU body of read_simpatico only raises, so there is
// nothing to replay a failed GPU run on.
connector make_read_simpatico_connector()
{
  return {.function_name              = "read_simpatico",
          .kind                       = kind::simpatico_file,
          .registry_profile           = "sirius.read_simpatico.v1",
          .bind_data_matches          = matches_bind<duckdb::SiriusReadSimpaticoBindData>,
          .lower                      = lower_simpatico_scan,
          .filter_mode                = mode::INCLUDE_AST_ROW_MASKS,
          .byte_source                = bytes::local_file,
          .permits_cpu_replay         = false,
          .selector_outside_bind_data = false,
          .decline_reason             = nullptr,
          .provider                   = std::nullopt};
}

std::array<connector, 7> const entries{make_seq_scan_connector(),
                                       make_parquet_scan_connector(),
                                       make_read_parquet_connector(),
                                       make_sirius_read_parquet_connector(),
                                       make_iceberg_scan_connector(),
                                       make_sirius_stream_source_connector(),
                                       make_read_simpatico_connector()};

duckdb::vector<duckdb::TableFunction> iceberg_reference_functions(duckdb::DatabaseInstance& db)
{
  auto info = duckdb::ExtensionManager::Get(db).GetExtensionInfo("iceberg");
  if (!info || !info->is_loaded || !info->install_info) return {};

  auto& fs = db.GetFileSystem();
  duckdb::vector<std::string> paths;
  if (info->install_info->mode == duckdb::ExtensionInstallMode::NOT_INSTALLED) {
    paths.push_back(info->install_info->full_path);
  } else {
    for (auto const& directory : duckdb::ExtensionHelper::GetExtensionDirectoryPath(db, fs))
      paths.push_back(fs.JoinPath(directory, "iceberg.duckdb_extension"));
  }
  for (auto const& path : paths) {
    // Only an extension DuckDB has already loaded may supply the reference definition.
    auto* handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
    if (!handle) continue;
    auto close = [](void* value) { ::dlclose(value); };
    std::unique_ptr<void, decltype(close)> guard(handle, close);
    using initialize = void (*)(duckdb::ExtensionLoader&);
    auto init        = reinterpret_cast<initialize>(::dlsym(handle, "iceberg_duckdb_cpp_init"));
    if (!init) continue;

    // Iceberg hides its function factory. Run its existing registration entry point in a
    // private, CPU-only catalog, never the caller's mutable catalog. No scan is bound here.
    duckdb::DBConfig config;
    config.options.load_extensions = false;
    config.options.maximum_threads = 1;
    duckdb::DuckDB reference(nullptr, &config);
    reference.LoadStaticExtension<duckdb::ParquetExtension>();
    // Iceberg derives scan callbacks from Parquet; use the caller's DuckDB factory.
    auto get_parquet = detail::host_factory(db,
                                            &duckdb::ParquetScanFunction::GetFunctionSet,
                                            "_ZN6duckdb19ParquetScanFunction14GetFunctionSetEv");
    if (!get_parquet) return {};
    duckdb::ExtensionLoader parquet_loader(*reference.instance, "parquet");
    auto functions = get_parquet();
    for (auto const* name : {"read_parquet", "parquet_scan"}) {
      functions.name = name;
      duckdb::CreateTableFunctionInfo info(functions);
      info.on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
      parquet_loader.RegisterFunction(std::move(info));
    }
    duckdb::ExtensionLoader loader(*reference.instance, "iceberg");
    init(loader);
    auto entry = loader.TryGetTableFunction("iceberg_scan");
    if (entry) return entry->Cast<duckdb::TableFunctionCatalogEntry>().functions.functions;
  }
  return {};
}

duckdb::vector<duckdb::TableFunction> reference_functions(std::string const& name,
                                                          duckdb::ClientContext& context)
{
  auto& db = duckdb::DatabaseInstance::GetDatabase(context);
  if (name == "seq_scan") {
    auto get = detail::host_factory(
      db, &duckdb::TableScanFunction::GetFunction, "_ZN6duckdb17TableScanFunction11GetFunctionEv");
    return get ? duckdb::vector<duckdb::TableFunction>{get()}
               : duckdb::vector<duckdb::TableFunction>{};
  }
  if (name == "parquet_scan" || name == "read_parquet") {
    auto get = detail::host_factory(db,
                                    &duckdb::ParquetScanFunction::GetFunctionSet,
                                    "_ZN6duckdb19ParquetScanFunction14GetFunctionSetEv");
    return get ? get().functions : duckdb::vector<duckdb::TableFunction>{};
  }
  if (name == "sirius_read_parquet") return {duckdb::GetSiriusReadParquetFunction()};
  if (name == "read_simpatico") return {duckdb::GetSiriusReadSimpaticoFunction()};
  if (name == "sirius_stream_source") return {exec::get_stream_source_function()};
  // Iceberg is initialized at extension load, never during a planning lookup.
  return {};
}

/// Compares a candidate scan against an independently obtained trusted function definition.
struct verified_callbacks {
  explicit verified_callbacks(duckdb::TableFunction const& value) : reference(value) {}
  /// Borrowed trusted definition; it must outlive this comparison object.
  duckdb::TableFunction const& reference;

  bool matches(duckdb::TableFunction const& candidate) const
  {
    // Initialization, binding/copy, and optimizer callbacks can change the reader's semantics
    // even when its scan body is unchanged. Only presentation/profiling callbacks are excluded.
    return std::tie(reference.function,
                    reference.bind,
                    reference.bind_replace,
                    reference.bind_operator,
                    reference.init_global,
                    reference.init_local,
                    reference.in_out_function,
                    reference.in_out_function_final,
                    reference.statistics,
                    reference.statistics_extended,
                    reference.dependency,
                    reference.cardinality,
                    reference.pushdown_complex_filter,
                    reference.pushdown_expression,
                    reference.get_partition_data,
                    reference.get_bind_info,
                    reference.type_pushdown,
                    reference.get_multi_file_reader,
                    reference.supports_pushdown_type,
                    reference.supports_pushdown_extract,
                    reference.get_partition_info,
                    reference.get_partition_stats,
                    reference.get_virtual_columns,
                    reference.get_row_id_columns,
                    reference.set_scan_order,
                    reference.serialize,
                    reference.deserialize,
                    reference.projection_pushdown,
                    reference.filter_pushdown,
                    reference.filter_prune,
                    reference.sampling_pushdown,
                    reference.late_materialization,
                    reference.order_preservation_type,
                    reference.global_initialization,
                    reference.arguments,
                    reference.varargs,
                    reference.named_parameters) == std::tie(candidate.function,
                                                            candidate.bind,
                                                            candidate.bind_replace,
                                                            candidate.bind_operator,
                                                            candidate.init_global,
                                                            candidate.init_local,
                                                            candidate.in_out_function,
                                                            candidate.in_out_function_final,
                                                            candidate.statistics,
                                                            candidate.statistics_extended,
                                                            candidate.dependency,
                                                            candidate.cardinality,
                                                            candidate.pushdown_complex_filter,
                                                            candidate.pushdown_expression,
                                                            candidate.get_partition_data,
                                                            candidate.get_bind_info,
                                                            candidate.type_pushdown,
                                                            candidate.get_multi_file_reader,
                                                            candidate.supports_pushdown_type,
                                                            candidate.supports_pushdown_extract,
                                                            candidate.get_partition_info,
                                                            candidate.get_partition_stats,
                                                            candidate.get_virtual_columns,
                                                            candidate.get_row_id_columns,
                                                            candidate.set_scan_order,
                                                            candidate.serialize,
                                                            candidate.deserialize,
                                                            candidate.projection_pushdown,
                                                            candidate.filter_pushdown,
                                                            candidate.filter_prune,
                                                            candidate.sampling_pushdown,
                                                            candidate.late_materialization,
                                                            candidate.order_preservation_type,
                                                            candidate.global_initialization,
                                                            candidate.arguments,
                                                            candidate.varargs,
                                                            candidate.named_parameters);
  }
};
/// Process-wide verification state for one connector, indexed alongside entries.
/// Catalog entries are checked against this independent reference and never grant trust.
struct accepted_callbacks {
  /// Serializes reference resolution/publication and the one-time warning state.
  std::mutex mutex;
  /// Caches trusted definitions, including an unavailable resolution result.
  detail::connector_reference_cache reference;
  /// Prevents repeated warnings when a connector has no trusted definition.
  std::unordered_set<void const*> missing_reference_reported;
};
std::array<accepted_callbacks, entries.size()> accepted;

void initialize_iceberg_callbacks(duckdb::DatabaseInstance& db)
{
  if (!duckdb::ExtensionManager::Get(db).ExtensionIsLoaded("iceberg")) return;
  auto const* host = detail::host_code_address(duckdb::Catalog::GetSystemCatalog(db));
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (entries[i].function_name != "iceberg_scan") continue;
    auto& cache = accepted[i];
    std::lock_guard lock(cache.mutex);
    if (cache.reference.has_verified_functions(host)) return;
    try {
      // Publish the complete independent reference only after registration succeeds.
      cache.reference.publish(host, iceberg_reference_functions(db));
    } catch (std::exception const& error) {
      // Optional GPU admission must not break LOAD or fall back to trusting the caller's
      // catalog. An empty cache makes lookup decline without retrying initialization there.
      SIRIUS_LOG_WARN("Iceberg scan source verification failed during extension load: {}",
                      error.what());
    }
    return;
  }
}

/// Publishes trusted Iceberg callbacks when Iceberg loads after Sirius.
class scan_source_extension_callback final : public duckdb::ExtensionCallback {
 public:
  void OnExtensionLoaded(duckdb::DatabaseInstance& db, std::string const& name) override
  {
    if (name == "iceberg") initialize_iceberg_callbacks(db);
  }
};
}  // namespace

std::span<connector const> registered_connectors() { return entries; }

void register_scan_source_callbacks(duckdb::DatabaseInstance& db)
{
  // Register first so a subsequent Iceberg load is observed. The explicit initialization
  // also covers Iceberg loaded before Sirius, including a catalog already modified by users.
  duckdb::DBConfig::GetConfig(db).GetCallbackManager().Register(
    duckdb::make_shared_ptr<scan_source_extension_callback>());
  initialize_iceberg_callbacks(db);
}

connector const* lookup_connector(duckdb::TableFunction const& function,
                                  duckdb::FunctionData const* bind,
                                  duckdb::ClientContext& context)
{
  for (size_t i = 0; i < entries.size(); ++i) {
    auto const& entry = entries[i];
    if (entry.function_name != function.name || !entry.bind_data_matches(bind)) continue;
    // The catalog's destructor identifies its host without a loader lookup per scan.
    auto const* host = detail::host_code_address(duckdb::Catalog::GetSystemCatalog(context));
    auto& cache      = accepted[i];
    std::lock_guard lock(cache.mutex);
    auto catalog_entry =
      duckdb::Catalog::GetSystemCatalog(context).GetEntry<duckdb::TableFunctionCatalogEntry>(
        context, DEFAULT_SCHEMA, entry.function_name, duckdb::OnEntryNotFound::RETURN_NULL);
    if (!catalog_entry) return nullptr;
    // A mutable catalog can confirm registration, but must never grant trust.
    auto const& references = cache.reference.get_or_resolve(
      host, [&] { return reference_functions(entry.function_name, context); });
    if (references.empty()) {
      if (cache.missing_reference_reported.insert(host).second) {
        auto const* requirement =
          "The source extension must provide a verifiable function definition.";
        if (entry.function_name == "seq_scan" || entry.function_name == "parquet_scan" ||
            entry.function_name == "read_parquet") {
          requirement =
            "An external DuckDB host must export "
            "TableScanFunction::GetFunction and ParquetScanFunction::GetFunctionSet.";
        }
        SIRIUS_LOG_WARN(
          "GPU scan source '{}' has no trusted reference definition; GPU lowering "
          "is disabled for this source. {}",
          entry.function_name,
          requirement);
      }
      return nullptr;
    }
    for (auto const& reference : references) {
      verified_callbacks const callbacks(reference);
      if (!callbacks.matches(function)) continue;
      for (auto const& registered : catalog_entry->functions.functions) {
        if (callbacks.matches(registered)) return &entry;
      }
    }
    return nullptr;
  }
  return nullptr;
}

connector const* lookup_connector(duckdb::LogicalGet const& get, duckdb::ClientContext& context)
{
  return lookup_connector(get.function, get.bind_data.get(), context);
}

connector const* lookup_connector(duckdb::PhysicalTableScan const& get,
                                  duckdb::ClientContext& context)
{
  return lookup_connector(get.function, get.bind_data.get(), context);
}
}  // namespace sirius::planner
