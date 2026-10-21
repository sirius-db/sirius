#define DUCKDB_EXTENSION_MAIN

#include "sirius_extension.hpp"

#include "sirius/duckdb.hpp"
#include "sirius_logging.hpp"

#define SIRIUS_STRINGIFY_IMPL(name) #name
#define SIRIUS_STRINGIFY(name)      SIRIUS_STRINGIFY_IMPL(name)
#define SIRIUS_ENTRY(name)          DUCKDB_CPP_EXTENSION_ENTRY(name, loader)

namespace duckdb {

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
  }

  // Grab the logging config from possibly set env variables and install it
  // before the rest of Sirius starts so all startup logs are included.
  auto logging = sirius::duckdb_extension::logging_options_from_environment();
  sirius::duckdb_extension::install_log_sink(logging, &loader.GetDatabaseInstance());

  sirius::register_duckdb_extension(loader);

  // After Sirius has booted, ensure that the logging options are exposed as SQL
  // options in DuckDB, so users can change logging again through DuckDB queries
  // if they want.
  sirius::duckdb_extension::register_logging_options(loader.GetDatabaseInstance(), logging);
}

void SiriusExtension::Load(ExtensionLoader& loader) { LoadInternal(loader); }
std::string SiriusExtension::Name() { return SIRIUS_STRINGIFY(SIRIUS_EXTENSION_NAME); }

std::string SiriusExtension::Version() const { return SIRIUS_EXTENSION_VERSION; }

}  // namespace duckdb

extern "C" {

SIRIUS_ENTRY(SIRIUS_EXTENSION_NAME) { duckdb::LoadInternal(loader); }
}
