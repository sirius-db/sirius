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
#include <link.h>
#include <parquet_extension.hpp>
#include <parquet_multi_file_info.hpp>

#include <array>
#include <mutex>
#include <typeinfo>

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

std::array<connector, 6> const entries{{{"seq_scan",
                                         kind::duckdb_native,
                                         "duckdb.seq_scan.v1",
                                         matches_bind<duckdb::TableScanBindData>,
                                         lower_native_scan,
                                         mode::include_ast_row_masks,
                                         bytes::duckdb_native,
                                         true,
                                         false,
                                         nullptr,
                                         std::nullopt},
                                        {"parquet_scan",
                                         kind::parquet_local,
                                         "duckdb.parquet_scan.v1",
                                         matches_bind<duckdb::MultiFileBindData>,
                                         lower_parquet_scan,
                                         mode::membership_masks_only,
                                         bytes::local_file,
                                         true,
                                         false,
                                         nullptr,
                                         std::nullopt},
                                        {"read_parquet",
                                         kind::parquet_local,
                                         "duckdb.read_parquet.v1",
                                         matches_bind<duckdb::MultiFileBindData>,
                                         lower_parquet_scan,
                                         mode::membership_masks_only,
                                         bytes::local_file,
                                         true,
                                         false,
                                         nullptr,
                                         std::nullopt},
                                        {"sirius_read_parquet",
                                         kind::parquet_s3,
                                         "sirius.read_parquet.v1",
                                         matches_bind<duckdb::SiriusReadParquetBindData>,
                                         lower_parquet_scan,
                                         mode::membership_masks_only,
                                         bytes::sirius_owned_s3,
                                         false,
                                         false,
                                         nullptr,
                                         std::nullopt},
                                        {"iceberg_scan",
                                         kind::parquet_local,
                                         "duckdb.iceberg_scan.v1",
                                         matches_bind<duckdb::MultiFileBindData>,
                                         lower_iceberg_scan,
                                         mode::membership_masks_only,
                                         bytes::local_file,
                                         true,
                                         true,
                                         registered_iceberg_decline_reason,
                                         std::nullopt},
                                        {"sirius_stream_source",
                                         kind::stream_source,
                                         "sirius.stream_source.v1",
                                         matches_bind<exec::stream_source_bind_data>,
                                         nullptr,
                                         mode::membership_masks_only,
                                         bytes::stream,
                                         false,
                                         false,
                                         nullptr,
                                         std::nullopt}}};

#ifdef DUCKDB_BUILD_LOADABLE_EXTENSION
void* host_factory(duckdb::DatabaseInstance& db, char const* symbol);
#endif

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
#ifdef DUCKDB_BUILD_LOADABLE_EXTENSION
    // Iceberg derives part of its scan callbacks from Parquet. The reference must
    // use the host factory, not the loadable Sirius module's hidden DuckDB copy.
    using parquet_factory = duckdb::TableFunctionSet (*)();
    auto get_parquet      = reinterpret_cast<parquet_factory>(
      host_factory(db, "_ZN6duckdb19ParquetScanFunction14GetFunctionSetEv"));
    if (!get_parquet) return {};
    duckdb::ExtensionLoader parquet_loader(*reference.instance, "parquet");
    auto functions = get_parquet();
    for (auto const* name : {"read_parquet", "parquet_scan"}) {
      functions.name = name;
      duckdb::CreateTableFunctionInfo info(functions);
      info.on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
      parquet_loader.RegisterFunction(std::move(info));
    }
#endif
    duckdb::ExtensionLoader loader(*reference.instance, "iceberg");
    init(loader);
    auto entry = loader.TryGetTableFunction("iceberg_scan");
    if (entry) return entry->Cast<duckdb::TableFunctionCatalogEntry>().functions.functions;
  }
  return {};
}

#ifdef DUCKDB_BUILD_LOADABLE_EXTENSION
void* host_factory(duckdb::DatabaseInstance& db, char const* symbol)
{
  // The system catalog object is created by the host, independently of mutable function
  // registrations. Its dynamic type locates that DuckDB module even for Python's RTLD_LOCAL
  // import. Do not locate the host through a candidate callback or promote it to RTLD_GLOBAL.
  auto& catalog = duckdb::Catalog::GetSystemCatalog(db);
  Dl_info owner{};
  void* owner_map{};
  if (!::dladdr1(&typeid(catalog), &owner, &owner_map, RTLD_DL_LINKMAP) || !owner_map)
    return nullptr;
  auto const* module = static_cast<link_map const*>(owner_map);
  // Use the loader's own name so NOLOAD finds its existing record without probing a file.
  // The main executable has an empty name: use its main handle, never reopen its disk path.
  auto const* name = module->l_name && module->l_name[0] ? module->l_name : nullptr;
  auto* handle     = ::dlopen(name, RTLD_NOW | RTLD_NOLOAD);
  if (!handle) return nullptr;
  auto close = [](void* value) { ::dlclose(value); };
  std::unique_ptr<void, decltype(close)> guard(handle, close);
  auto* factory = ::dlsym(handle, symbol);
  Dl_info implementation{};
  // A handle can also resolve symbols from dependencies. Only this host may grant trust.
  if (!factory || !::dladdr(factory, &implementation) ||
      implementation.dli_fbase != owner.dli_fbase) {
    return nullptr;
  }
  return factory;
}
#endif

duckdb::vector<duckdb::TableFunction> reference_functions(std::string const& name,
                                                          duckdb::ClientContext& context)
{
#ifdef DUCKDB_BUILD_LOADABLE_EXTENSION
  // The loadable extension links a hidden DuckDB copy. Its factory addresses are not the
  // host's. Resolve the host's exported factories (pinned DuckDB C++ ABI), never an entry in
  // the caller's mutable catalog. Missing host symbols leave the source unverified.
  if (name == "seq_scan") {
    using factory = duckdb::TableFunction (*)();
    auto get =
      reinterpret_cast<factory>(host_factory(duckdb::DatabaseInstance::GetDatabase(context),
                                             "_ZN6duckdb17TableScanFunction11GetFunctionEv"));
    return get ? duckdb::vector<duckdb::TableFunction>{get()}
               : duckdb::vector<duckdb::TableFunction>{};
  }
  if (name == "parquet_scan" || name == "read_parquet") {
    using factory = duckdb::TableFunctionSet (*)();
    auto get =
      reinterpret_cast<factory>(host_factory(duckdb::DatabaseInstance::GetDatabase(context),
                                             "_ZN6duckdb19ParquetScanFunction14GetFunctionSetEv"));
    return get ? get().functions : duckdb::vector<duckdb::TableFunction>{};
  }
#else
  if (name == "seq_scan") return {duckdb::TableScanFunction::GetFunction()};
  if (name == "parquet_scan" || name == "read_parquet")
    return duckdb::ParquetScanFunction::GetFunctionSet().functions;
#endif
  if (name == "sirius_read_parquet") return {duckdb::GetSiriusReadParquetFunction()};
  if (name == "sirius_stream_source") return {exec::get_stream_source_function()};
  // Iceberg is initialized at extension load, never during a planning lookup.
  return {};
}

struct verified_callbacks {
  explicit verified_callbacks(duckdb::TableFunction const& value) : reference(value) {}
  duckdb::TableFunction reference;

  bool matches(duckdb::TableFunction const& candidate) const
  {
    // Initialization, binding/copy, and optimizer callbacks can change the reader's semantics
    // even when its scan body is unchanged. Only presentation/profiling callbacks are excluded.
    return reference.function == candidate.function && reference.bind == candidate.bind &&
           reference.bind_replace == candidate.bind_replace &&
           reference.bind_operator == candidate.bind_operator &&
           reference.init_global == candidate.init_global &&
           reference.init_local == candidate.init_local &&
           reference.in_out_function == candidate.in_out_function &&
           reference.in_out_function_final == candidate.in_out_function_final &&
           reference.statistics == candidate.statistics &&
           reference.statistics_extended == candidate.statistics_extended &&
           reference.dependency == candidate.dependency &&
           reference.cardinality == candidate.cardinality &&
           reference.pushdown_complex_filter == candidate.pushdown_complex_filter &&
           reference.pushdown_expression == candidate.pushdown_expression &&
           reference.get_partition_data == candidate.get_partition_data &&
           reference.get_bind_info == candidate.get_bind_info &&
           reference.type_pushdown == candidate.type_pushdown &&
           reference.get_multi_file_reader == candidate.get_multi_file_reader &&
           reference.supports_pushdown_type == candidate.supports_pushdown_type &&
           reference.supports_pushdown_extract == candidate.supports_pushdown_extract &&
           reference.get_partition_info == candidate.get_partition_info &&
           reference.get_partition_stats == candidate.get_partition_stats &&
           reference.get_virtual_columns == candidate.get_virtual_columns &&
           reference.get_row_id_columns == candidate.get_row_id_columns &&
           reference.set_scan_order == candidate.set_scan_order &&
           reference.serialize == candidate.serialize &&
           reference.deserialize == candidate.deserialize &&
           reference.projection_pushdown == candidate.projection_pushdown &&
           reference.filter_pushdown == candidate.filter_pushdown &&
           reference.filter_prune == candidate.filter_prune &&
           reference.sampling_pushdown == candidate.sampling_pushdown &&
           reference.late_materialization == candidate.late_materialization &&
           reference.order_preservation_type == candidate.order_preservation_type &&
           reference.global_initialization == candidate.global_initialization &&
           reference.arguments == candidate.arguments && reference.varargs == candidate.varargs &&
           reference.named_parameters == candidate.named_parameters;
  }
};
struct accepted_callbacks {
  std::mutex mutex;
  std::vector<verified_callbacks> values;
  bool missing_reference_reported = false;
};
std::array<accepted_callbacks, entries.size()> accepted;

void initialize_iceberg_callbacks(duckdb::DatabaseInstance& db)
{
  if (!duckdb::ExtensionManager::Get(db).ExtensionIsLoaded("iceberg")) return;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (entries[i].function_name != "iceberg_scan") continue;
    auto& cache = accepted[i];
    std::lock_guard lock(cache.mutex);
    if (!cache.values.empty()) return;
    try {
      std::vector<verified_callbacks> verified;
      for (auto const& value : iceberg_reference_functions(db)) {
        verified.emplace_back(value);
      }
      // Publish the complete independent reference only after registration succeeds.
      cache.values = std::move(verified);
    } catch (std::exception const& error) {
      // Optional GPU admission must not break LOAD or fall back to trusting the caller's
      // catalog. An empty cache makes lookup decline without retrying initialization there.
      SIRIUS_LOG_WARN("Iceberg scan source verification failed during extension load: {}",
                      error.what());
    }
    return;
  }
}

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
    auto& cache = accepted[i];
    std::lock_guard lock(cache.mutex);
    auto catalog_entry =
      duckdb::Catalog::GetSystemCatalog(context).GetEntry<duckdb::TableFunctionCatalogEntry>(
        context, DEFAULT_SCHEMA, entry.function_name, duckdb::OnEntryNotFound::RETURN_NULL);
    if (!catalog_entry) return nullptr;
    if (cache.values.empty()) {
      // A mutable catalog can confirm registration, but must never grant trust.
      for (auto const& value : reference_functions(entry.function_name, context)) {
        cache.values.emplace_back(value);
      }
    }
    if (cache.values.empty()) {
      if (!cache.missing_reference_reported) {
        auto const* requirement =
          "The source extension must provide a verifiable function definition.";
#ifdef DUCKDB_BUILD_LOADABLE_EXTENSION
        if (entry.function_name == "seq_scan" || entry.function_name == "parquet_scan" ||
            entry.function_name == "read_parquet") {
          requirement =
            "Loadable Sirius requires the matching host DuckDB build to export "
            "TableScanFunction::GetFunction and ParquetScanFunction::GetFunctionSet.";
        }
#endif
        SIRIUS_LOG_WARN(
          "GPU scan source '{}' has no trusted reference definition; GPU lowering "
          "is disabled for this source. {}",
          entry.function_name,
          requirement);
        cache.missing_reference_reported = true;
      }
      return nullptr;
    }
    for (auto const& callbacks : cache.values) {
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
