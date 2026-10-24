// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "planner/paimon_bound_read.hpp"

#include "planner/paimon_read_bridge.hpp"
#include "planner/verified_scan_callbacks.hpp"

#include <dlfcn.h>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/main/extension_helper.hpp>
#include <duckdb/main/extension_manager.hpp>
#include <duckdb/main/settings.hpp>
#include <link.h>

#include <stdexcept>
#include <utility>

namespace sirius::planner {
namespace {
[[noreturn]] void refuse(char const* reason)
{
  // Deliberately do not include provider diagnostics, paths, options, or credentials.
  throw std::runtime_error(std::string("Paimon descriptor refused: ") + reason);
}

bool belongs_to(void const* address, link_map const* module)
{
  Dl_info info{};
  void* owner{};
  return address && ::dladdr1(address, &info, &owner, RTLD_DL_LINKMAP) && owner == module;
}

void validate_string(PaimonReadString const& value)
{
  if (value.size && !value.data) refuse("invalid descriptor string");
}

void validate_view(PaimonBoundReadViewV1 const& view)
{
  if (view.has_snapshot > 1 || (view.has_snapshot && view.snapshot_id <= 0) ||
      (!view.has_snapshot && view.snapshot_id != 0) || view.schema_id < 0 ||
      (view.option_count && !view.options))
    refuse("invalid descriptor identity");
  for (auto const* value : {&view.warehouse,
                            &view.database,
                            &view.table,
                            &view.location,
                            &view.branch,
                            &view.schema_json})
    validate_string(*value);
  if (!view.schema_json.size || !view.location.size || !view.branch.size)
    refuse("incomplete descriptor identity");
  for (uint64_t i = 0; i < view.option_count; ++i) {
    validate_string(view.options[i].key);
    validate_string(view.options[i].value);
  }
}
}  // namespace

struct paimon_read_provider::state {
  // Declaration order is intentional: function-info destructors run before dlclose.
  std::shared_ptr<void> module;
  link_map const* origin = nullptr;
  duckdb::vector<duckdb::TableFunction> references;

  PaimonReadBridgeV1 const& authenticate(duckdb::BoundTableFunction const& candidate) const
  {
    for (auto const& reference : references) {
      if (!detail::same_scan_callbacks(reference, candidate) || !reference.function_info ||
          candidate.function_info != reference.function_info)
        continue;

      // The wrapper is only interpreted after independent callback AND singleton checks.
      auto const& api =
        static_cast<duckdb::PaimonReadBridgeInfo const&>(*reference.function_info).api;
      if (api.magic != PAIMON_READ_BRIDGE_MAGIC ||
          !((api.version == PAIMON_READ_BRIDGE_VERSION &&
             api.struct_size == sizeof(PaimonReadBridgeV1)) ||
            (api.version == PAIMON_READ_BRIDGE_NATIVE_VERSION &&
             api.struct_size == sizeof(PaimonReadBridgeV2))))
        refuse("unsupported bridge ABI");
      for (auto const* callback : {reinterpret_cast<void const*>(api.accepts_bind),
                                   reinterpret_cast<void const*>(api.retain),
                                   reinterpret_cast<void const*>(api.describe),
                                   reinterpret_cast<void const*>(api.release)}) {
        if (!belongs_to(callback, origin)) refuse("foreign or missing bridge callback");
      }
      return api;
    }
    refuse("unverified callbacks or function-info");
  }
};

struct paimon_bound_read::state {
  std::shared_ptr<paimon_read_provider::state> provider;
  PaimonReadBridgeV1 const* api = nullptr;
  void* descriptor              = nullptr;
  PaimonBoundReadViewV1 view{};

  ~state()
  {
    if (descriptor) api->release(descriptor);
  }
};

paimon_read_provider::paimon_read_provider(std::shared_ptr<state> value) : state_(std::move(value))
{
}

paimon_read_provider paimon_read_provider::from_loaded(duckdb::DatabaseInstance& db)
{
  if (duckdb::Settings::Get<duckdb::AllowExtensionsMetadataMismatchSetting>(db))
    refuse("extension ABI checking is disabled");
  auto const info = duckdb::ExtensionManager::Get(db).GetExtensionInfo("paimon");
  if (!info || !info->is_loaded || !info->install_info)
    refuse("provider is not dynamically loaded");
  // At this host pin, direct LOAD records the actual loaded filename. Installed/static
  // providers do not expose that same provenance; do not guess or open a new module.
  if (info->install_info->mode != duckdb::ExtensionInstallMode::NOT_INSTALLED ||
      info->install_info->full_path.empty())
    refuse("provider has no qualified loaded path");

  auto owner   = std::make_shared<state>();
  auto* handle = ::dlopen(info->install_info->full_path.c_str(), RTLD_NOW | RTLD_NOLOAD);
  if (!handle) refuse("loaded provider module is unavailable");
  owner->module = std::shared_ptr<void>(handle, [](void* value) { ::dlclose(value); });
  link_map* origin{};
  if (::dlinfo(handle, RTLD_DI_LINKMAP, &origin) != 0 || !origin)
    refuse("provider module provenance is unavailable");
  owner->origin    = origin;
  auto* initialize = ::dlsym(handle, "paimon_duckdb_cpp_init");
  // dlsym(handle) can resolve a dependency's symbol; it must belong to this module.
  if (!belongs_to(initialize, origin)) refuse("provider initializer is unavailable or foreign");

  duckdb::DBConfig config;
  config.options.load_extensions = false;
  config.options.maximum_threads = 1;
  duckdb::DuckDB reference(nullptr, &config);
  duckdb::ExtensionLoader loader(*reference.instance, "paimon");
  reinterpret_cast<void (*)(duckdb::ExtensionLoader&)>(initialize)(loader);
  auto entry = loader.TryGetTableFunction("paimon_scan");
  if (!entry) refuse("provider has no scan definition");
  for (auto const& function :
       entry->Cast<duckdb::TableFunctionCatalogEntry>().functions.functions) {
    if (!belongs_to(reinterpret_cast<void const*>(function->function), origin) ||
        !belongs_to(reinterpret_cast<void const*>(function->bind), origin) ||
        !belongs_to(reinterpret_cast<void const*>(function->init_global), origin) ||
        !belongs_to(reinterpret_cast<void const*>(function->init_local), origin) ||
        !function->function_info)
      refuse("provider has an incomplete scan definition");
    owner->references.push_back(*function);
    // Validate the independently obtained wrapper now, before retaining any user bind.
    owner->authenticate(duckdb::BoundTableFunction(*function));
  }
  if (owner->references.empty()) refuse("provider has no scan overloads");
  return paimon_read_provider(std::move(owner));
}

paimon_bound_read paimon_read_provider::retain(duckdb::BoundTableFunction const& candidate,
                                               duckdb::FunctionData const* bind) const
{
  if (!state_) refuse("provider owner was moved");
  auto const& api = state_->authenticate(candidate);
  if (api.accepts_bind(bind) != 1) refuse("bind data is not from this provider");
  auto retained      = std::make_unique<paimon_bound_read::state>();
  retained->provider = state_;
  retained->api      = &api;
  // The state owns partial output even when a provider reports failure.
  auto status = api.retain(bind, &retained->descriptor, nullptr, 0);
  if (status != PAIMON_READ_OK || !retained->descriptor) refuse("retain failed");
  if (api.describe(retained->descriptor, &retained->view, sizeof(retained->view)) != PAIMON_READ_OK)
    refuse("describe failed");
  validate_view(retained->view);
  return paimon_bound_read(std::move(retained));
}

paimon_bound_read::paimon_bound_read(std::unique_ptr<state> value) : state_(std::move(value)) {}
paimon_bound_read::paimon_bound_read(paimon_bound_read&&) noexcept            = default;
paimon_bound_read& paimon_bound_read::operator=(paimon_bound_read&&) noexcept = default;
paimon_bound_read::~paimon_bound_read()                                       = default;

PaimonBoundReadViewV1 const& paimon_bound_read::view() const
{
  if (!state_) refuse("descriptor owner was moved");
  return state_->view;
}
}  // namespace sirius::planner
