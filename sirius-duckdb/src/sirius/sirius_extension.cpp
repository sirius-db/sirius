#define DUCKDB_EXTENSION_MAIN

#include "sirius_extension.hpp"

#include "sirius/duckdb.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader& loader) { sirius::register_duckdb_extension(loader); }

void SiriusExtension::Load(ExtensionLoader& loader) { LoadInternal(loader); }
std::string SiriusExtension::Name() { return "sirius"; }

std::string SiriusExtension::Version() const
{
#ifdef EXT_VERSION_SIRIUS
  return EXT_VERSION_SIRIUS;
#else
  return "";
#endif
}

}  // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(sirius, loader) { duckdb::LoadInternal(loader); }
}
