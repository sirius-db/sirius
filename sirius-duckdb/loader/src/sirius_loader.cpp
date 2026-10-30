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

#define DUCKDB_EXTENSION_MAIN

#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/sql_identifier.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/main/extension_install_info.hpp"
#include "duckdb/main/settings.hpp"

#include <dlfcn.h>

#include <exception>
#include <memory>

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(sirius, loader)
{
  // The nested backend load must not acquire this loader's own registration lock.
  auto const& name = loader.GetRegisteredExtensionName();
  if (name == "sirius_cuda12" || name == "sirius_cuda13") {
    throw duckdb::InvalidInputException(
      "Cannot load sirius under a CUDA backend name; use LOAD sirius instead");
  }
  auto close_driver = [](void* handle) { dlclose(handle); };
  std::unique_ptr<void, decltype(close_driver)> driver(
    dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL), close_driver);
  if (!driver) {
    throw duckdb::InvalidInputException(
      "Sirius requires an NVIDIA driver supporting CUDA 12 or newer: cannot open libcuda.so.1 (%s)",
      dlerror());
  }
  using get_version_t = int (*)(int*);
  auto get_version    = reinterpret_cast<get_version_t>(dlsym(driver.get(), "cuDriverGetVersion"));
  if (!get_version) {
    throw duckdb::InvalidInputException(
      "Sirius cannot determine CUDA driver support: libcuda.so.1 has no cuDriverGetVersion");
  }
  int version = 0;
  auto status = get_version(&version);
  if (status != 0) {
    throw duckdb::InvalidInputException(
      "Sirius cannot determine CUDA driver support: cuDriverGetVersion failed with error %d",
      status);
  }
  if (version < 12000) {
    throw duckdb::InvalidInputException(
      "Sirius requires an NVIDIA driver supporting CUDA 12 or newer (reported CUDA version: %d.%d)",
      version / 1000,
      (version % 1000) / 10);
  }
  auto const* backend = version < 13000 ? "sirius_cuda12" : "sirius_cuda13";
  auto const* other   = version < 13000 ? "sirius_cuda13" : "sirius_cuda12";
  auto& db            = loader.GetDatabaseInstance();
  // This guards sequential loads in this database, not process-wide use.
  if (db.ExtensionIsLoaded(other)) {
    throw duckdb::InvalidInputException(
      "Cannot load %s while %s is already loaded", backend, other);
  }
  if (db.ExtensionIsLoaded(backend)) { return; }
  try {
    if (duckdb::Settings::Get<duckdb::AutoinstallKnownExtensionsSetting>(db)) {
      // The backends are community extensions. Only redirect the default (core) repository so
      // explicit mirrors and local test repositories keep working.
      auto repository = duckdb::ExtensionHelper::GetAutoinstallRepository(db);
      if (repository.path == duckdb::ExtensionRepository::CORE_REPOSITORY_URL &&
          !duckdb::ExtensionRepository::TryGetKnownRepository("community", repository)) {
        throw duckdb::InternalException("DuckDB has no community extension repository");
      }
      duckdb::ExtensionInstallOptions options;
      options.repository = repository;
      duckdb::ExtensionHelper::InstallExtension(
        db, duckdb::FileSystem::GetLocal(db), backend, options);
    }
    auto connection = duckdb::Connection::CreateInternal(db);
    connection.Query(duckdb::StringUtil::Format("LOAD %s", duckdb::SQLString(backend)))
      ->ThrowIfError();
  } catch (std::exception const& error) {
    auto message = duckdb::StringUtil::Format(
      "Sirius selected %s for CUDA driver version %d.%d, but loading it failed. ",
      backend,
      version / 1000,
      (version % 1000) / 10);
    duckdb::ErrorData(error).Throw(message);
  }
}
}
