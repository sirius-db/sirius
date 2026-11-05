#define DUCKDB_EXTENSION_MAIN

#include "sirius_extension.hpp"

#include "duckdb/storage/object_cache.hpp"
#include "sirius/duckdb.hpp"

#define SIRIUS_STRINGIFY_IMPL(name) #name
#define SIRIUS_STRINGIFY(name)      SIRIUS_STRINGIFY_IMPL(name)
#define SIRIUS_ENTRY(name)          DUCKDB_CPP_EXTENSION_ENTRY(name, loader)

namespace duckdb {

// Tells the sirius loader that backend initialization started, so it propagates a failure instead
// of falling back to the CPU. The loader looks the key up by name.
struct SiriusBackendInitStarted : ObjectCacheEntry {
  static constexpr char const* KEY = "sirius_backend_init_started";
  std::string GetObjectType() override { return KEY; }
  optional_idx GetEstimatedCacheMemory() const override { return optional_idx(); }
};

static void LoadInternal(ExtensionLoader& loader)
{
  std::string const name = SIRIUS_STRINGIFY(SIRIUS_EXTENSION_NAME);
  if (name == "sirius_cuda12" || name == "sirius_cuda13") {
    // Conflict checks rely on DuckDB registering each backend under its own name.
    if (loader.GetRegisteredExtensionName() != name) {
      throw InvalidInputException("Cannot load %s under an alias; use LOAD sirius instead", name);
    }
    auto const* other = name == "sirius_cuda12" ? "sirius_cuda13" : "sirius_cuda12";
    if (loader.GetDatabaseInstance().ExtensionIsLoaded(other)) {
      throw InvalidInputException("Cannot load %s while %s is already loaded", name, other);
    }
    auto& cache = loader.GetDatabaseInstance().GetObjectCache();
    cache.Put(SiriusBackendInitStarted::KEY, make_shared_ptr<SiriusBackendInitStarted>());
  }
  sirius::register_duckdb_extension(loader);
}

void SiriusExtension::Load(ExtensionLoader& loader) { LoadInternal(loader); }
std::string SiriusExtension::Name() { return SIRIUS_STRINGIFY(SIRIUS_EXTENSION_NAME); }

std::string SiriusExtension::Version() const { return SIRIUS_EXTENSION_VERSION; }

}  // namespace duckdb

extern "C" {

SIRIUS_ENTRY(SIRIUS_EXTENSION_NAME) { duckdb::LoadInternal(loader); }
}
