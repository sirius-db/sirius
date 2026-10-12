// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "sirius_extension.hpp"

#include <core_functions_extension.hpp>
#include <duckdb/main/extension_helper.hpp>
#include <parquet_extension.hpp>

// Preserve automatic registration in engine tests without building the wrapper.
// These definitions precede DuckDB in the link; duplicate definitions are allowed.
namespace duckdb {

ExtensionLoadResult ExtensionHelper::LoadExtension(DuckDB& db, const std::string& extension)
{
  if (extension == "sirius") {
    db.LoadStaticExtension<SiriusExtension>();
  } else if (extension == "core_functions") {
    db.LoadStaticExtension<CoreFunctionsExtension>();
  } else if (extension == "parquet") {
    db.LoadStaticExtension<ParquetExtension>();
  } else {
    return ExtensionLoadResult::NOT_LOADED;
  }
  return ExtensionLoadResult::LOADED_EXTENSION;
}

vector<string> LinkedExtensions() { return {"sirius", "core_functions", "parquet"}; }

void ExtensionHelper::LoadAllExtensions(DuckDB& db)
{
  for (auto const& name : LinkedExtensions()) {
    LoadExtension(db, name);
  }
}

vector<string> ExtensionHelper::LoadedExtensionTestPaths() { return {}; }

}  // namespace duckdb
