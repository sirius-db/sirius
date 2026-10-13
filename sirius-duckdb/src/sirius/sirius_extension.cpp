#define DUCKDB_EXTENSION_MAIN

#include "sirius_extension.hpp"

#include "sirius/duckdb.hpp"

#define SIRIUS_STRINGIFY_IMPL(name) #name
#define SIRIUS_STRINGIFY(name)      SIRIUS_STRINGIFY_IMPL(name)
#define SIRIUS_ENTRY(name)          DUCKDB_CPP_EXTENSION_ENTRY(name, loader)

namespace duckdb {

static void LoadInternal(ExtensionLoader& loader) { sirius::register_duckdb_extension(loader); }

void SiriusExtension::Load(ExtensionLoader& loader) { LoadInternal(loader); }
std::string SiriusExtension::Name() { return SIRIUS_STRINGIFY(SIRIUS_EXTENSION_NAME); }

std::string SiriusExtension::Version() const { return SIRIUS_EXTENSION_VERSION; }

}  // namespace duckdb

extern "C" {

SIRIUS_ENTRY(SIRIUS_EXTENSION_NAME) { duckdb::LoadInternal(loader); }
}
