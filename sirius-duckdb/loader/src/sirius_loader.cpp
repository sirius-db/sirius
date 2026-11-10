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
#include "duckdb/common/printer.hpp"
#include "duckdb/common/sql_identifier.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/logging/logger.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/main/extension_install_info.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/storage/object_cache.hpp"

#include <dlfcn.h>

#include <chrono>
#include <exception>
#include <memory>
#include <string>

namespace {

constexpr char const* gpu_execution_setting = "gpu_execution";
// Set by the backend entry point (sirius-duckdb/src/sirius/sirius_extension.cpp).
constexpr char const* backend_init_started_key = "sirius_backend_init_started";
constexpr char const* runtime_ready_key        = "sirius_runtime_ready";

/// Outcome of looking for a CUDA 12.8+ driver. Probed once per process.
struct cuda_driver {
  bool found         = false;  ///< libcuda.so.1 could be opened
  bool version_known = false;  ///< cuDriverGetVersion succeeded
  int version        = 0;      ///< CUDA version reported by cuDriverGetVersion
  std::string error;           ///< Why the driver is unusable, empty if usable

  [[nodiscard]] bool usable() const { return error.empty(); }

  [[nodiscard]] std::string version_string() const
  {
    return duckdb::StringUtil::Format("%d.%d", version / 1000, (version % 1000) / 10);
  }

  static cuda_driver const& get()
  {
    static cuda_driver const driver = probe();
    return driver;
  }

 private:
  static cuda_driver probe()
  {
    cuda_driver result;
    auto fail = [&](std::string const& detail) {
      result.error = "Sirius requires an NVIDIA driver supporting CUDA 12.8 or newer: " + detail;
      return result;
    };
    auto close = [](void* handle) { dlclose(handle); };
    std::unique_ptr<void, decltype(close)> handle(dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL),
                                                  close);
    if (!handle) {
      auto const* error = dlerror();
      return fail(duckdb::StringUtil::Format("cannot open libcuda.so.1 (%s)", error ? error : ""));
    }
    result.found        = true;
    using get_version_t = int (*)(int*);
    auto get_version = reinterpret_cast<get_version_t>(dlsym(handle.get(), "cuDriverGetVersion"));
    if (!get_version) { return fail("libcuda.so.1 has no cuDriverGetVersion"); }
    int version = 0;
    if (auto status = get_version(&version); status != 0) {
      return fail(duckdb::StringUtil::Format("cuDriverGetVersion failed with error %d", status));
    }
    result.version_known = true;
    result.version       = version;
    // The CUDA 12 backend calls cudaMemcpyBatchAsync, which needs a CUDA 12.8 driver.
    if (version < 12080) { return fail("reported CUDA version " + result.version_string()); }
    return result;
  }
};

/// What LOAD sirius resolved for one database. Immutable once the load finishes.
struct loader_status {
  cuda_driver driver;
  std::string backend;  ///< Selected CUDA backend, empty without a usable driver
  bool backend_loaded = false;
  bool runtime_ready  = false;  ///< The backend initialized its runtime (not under SIRIUS_DISABLE)
  std::string fallback_reason;  ///< Why GPU execution is unavailable, empty if the backend loaded

  /// Rejects an explicit request for GPU execution while the backend is unavailable.
  void require_gpu() const
  {
    if (!backend_loaded) { throw duckdb::InvalidInputException(fallback_reason); }
  }
};

using loader_status_ptr = duckdb::shared_ptr<loader_status const>;

/// Refuses SET gpu_execution = true while the CPU fallback is active.
struct fallback_gpu_execution_option : duckdb::ExtensionOptionInfo {
  explicit fallback_gpu_execution_option(loader_status_ptr status) : status(std::move(status)) {}

  void OnSet(duckdb::ClientContext&,
             duckdb::SetScope,
             duckdb::Value& parameter,
             bool) const override
  {
    reject_enable(*status, parameter);
  }

  static void reject_enable(loader_status const& status, duckdb::Value const& value)
  {
    if (!value.IsNull() && value.DefaultCastAs(duckdb::LogicalType::BOOLEAN).GetValue<bool>()) {
      status.require_gpu();
    }
  }

  loader_status_ptr status;
};

struct status_function_info : duckdb::TableFunctionInfo {
  explicit status_function_info(loader_status_ptr status) : status(std::move(status)) {}

  loader_status_ptr status;
};

/// sirius_status(): one row describing the driver probe and the backend.
struct status_function {
  struct state : duckdb::GlobalTableFunctionState {
    duckdb::vector<duckdb::Value> row;
    bool done = false;
  };

  static duckdb::TableFunction create(loader_status_ptr status)
  {
    duckdb::TableFunction function("sirius_status", {}, scan, bind, init);
    function.function_info = duckdb::make_shared_ptr<status_function_info>(std::move(status));
    return function;
  }

 private:
  struct status_data : duckdb::FunctionData {
    explicit status_data(loader_status_ptr status) : status(std::move(status)) {}

    duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
    {
      return duckdb::make_uniq<status_data>(status);
    }

    bool Equals(duckdb::FunctionData const& other) const override
    {
      return status == other.Cast<status_data>().status;
    }

    loader_status_ptr status;
  };

  static duckdb::unique_ptr<duckdb::FunctionData> bind(duckdb::ClientContext&,
                                                       duckdb::TableFunctionBindInput& input,
                                                       duckdb::vector<duckdb::LogicalType>& types,
                                                       duckdb::vector<duckdb::Identifier>& names)
  {
    auto add = [&](char const* name, duckdb::LogicalType type) {
      names.emplace_back(name);
      types.push_back(std::move(type));
    };
    add("driver_found", duckdb::LogicalType::BOOLEAN);
    add("cuda_driver_version", duckdb::LogicalType::VARCHAR);
    add("backend", duckdb::LogicalType::VARCHAR);
    add("backend_loaded", duckdb::LogicalType::BOOLEAN);
    add("gpu_execution", duckdb::LogicalType::BOOLEAN);
    add("fallback_reason", duckdb::LogicalType::VARCHAR);
    return duckdb::make_uniq<status_data>(input.info->Cast<status_function_info>().status);
  }

  // Runs per execution, so a prepared statement sees the current gpu_execution setting.
  static duckdb::unique_ptr<duckdb::GlobalTableFunctionState> init(
    duckdb::ClientContext& context, duckdb::TableFunctionInitInput& input)
  {
    auto const& status = *input.bind_data->Cast<status_data>().status;
    auto optional      = [](bool present, std::string const& value) {
      return present ? duckdb::Value(value) : duckdb::Value(duckdb::LogicalType::VARCHAR);
    };
    // Transparent execution runs only when the runtime initialized and this session enables it.
    duckdb::Value enabled;
    auto const active = status.runtime_ready &&
                        context.TryGetCurrentSetting(gpu_execution_setting, enabled) &&
                        !enabled.IsNull() && enabled.GetValue<bool>();
    auto result = duckdb::make_uniq<state>();
    result->row.push_back(duckdb::Value::BOOLEAN(status.driver.found));
    result->row.push_back(optional(status.driver.version_known, status.driver.version_string()));
    result->row.push_back(optional(!status.backend.empty(), status.backend));
    result->row.push_back(duckdb::Value::BOOLEAN(status.backend_loaded));
    result->row.push_back(duckdb::Value::BOOLEAN(active));
    result->row.push_back(optional(!status.fallback_reason.empty(), status.fallback_reason));
    return std::move(result);
  }

  static void scan(duckdb::ClientContext&,
                   duckdb::TableFunctionInput& input,
                   duckdb::DataChunk& output)
  {
    auto& data = input.global_state->Cast<state>();
    if (data.done) { return; }
    for (duckdb::idx_t i = 0; i < data.row.size(); i++) {
      output.data[i].Append(data.row[i]);
    }
    data.done = true;
  }
};

/// gpu_execution(sql): the explicit GPU entry point. Without a backend it explains why.
struct fallback_gpu_execution_function {
  static duckdb::TableFunction create(loader_status_ptr status)
  {
    duckdb::TableFunction function("gpu_execution", {duckdb::LogicalType::VARCHAR}, nullptr, bind);
    function.GetSignature().AddKwargs("options", duckdb::LogicalType::ANY);
    function.function_info = duckdb::make_shared_ptr<status_function_info>(std::move(status));
    return function;
  }

 private:
  static duckdb::unique_ptr<duckdb::FunctionData> bind(duckdb::ClientContext&,
                                                       duckdb::TableFunctionBindInput& input,
                                                       duckdb::vector<duckdb::LogicalType>&,
                                                       duckdb::vector<duckdb::Identifier>&)
  {
    input.info->Cast<status_function_info>().status->require_gpu();
    throw duckdb::InternalException("gpu_execution fallback bound with a loaded backend");
  }
};

/// Loads the CUDA backend for the probed driver and records why it could not be used.
class sirius_loader {
 public:
  explicit sirius_loader(duckdb::ExtensionLoader& loader) : loader_(loader) {}

  void load()
  {
    // The nested backend load must not acquire this loader's own registration lock.
    auto const& name = loader_.GetRegisteredExtensionName();
    if (name == "sirius_cuda12" || name == "sirius_cuda13") {
      throw duckdb::InvalidInputException(
        "Cannot load sirius under a CUDA backend name; use LOAD sirius instead");
    }
    auto status    = duckdb::make_shared_ptr<loader_status>();
    status->driver = cuda_driver::get();
    if (status->driver.usable()) {
      load_backend(*status);
    } else {
      status->fallback_reason = status->driver.error;
    }
    loader_status_ptr result = std::move(status);
    if (!result->backend_loaded) { register_fallback(result); }
    loader_.RegisterFunction(status_function::create(result));
  }

 private:
  void load_backend(loader_status& status)
  {
    auto const old_driver = status.driver.version < 13000;
    status.backend        = old_driver ? "sirius_cuda12" : "sirius_cuda13";
    auto const* other     = old_driver ? "sirius_cuda13" : "sirius_cuda12";
    auto& db              = loader_.GetDatabaseInstance();
    // This guards sequential loads in this database, not process-wide use.
    if (db.ExtensionIsLoaded(other)) {
      throw duckdb::InvalidInputException(
        "Cannot load %s while %s is already loaded", status.backend, other);
    }
    if (!db.ExtensionIsLoaded(status.backend)) {
      try {
        install_and_load(status.backend);
      } catch (std::exception const& error) {
        auto const prefix = duckdb::StringUtil::Format(
          "Sirius selected %s for CUDA driver version %s, but loading it failed. ",
          status.backend,
          status.driver.version_string());
        duckdb::ErrorData data(error);
        // Fall back only when the backend never started initializing (not installed, install
        // failed, unloadable file). A failure inside its initialization is a real GPU problem and
        // may leave state behind, so report it.
        if (data.Type() == duckdb::ExceptionType::FATAL ||
            db.GetObjectCache().GetObject(backend_init_started_key)) {
          data.Throw(prefix);
        }
        status.fallback_reason = prefix + data.RawMessage();
        return;
      }
    }
    status.backend_loaded = true;
    status.runtime_ready  = db.GetObjectCache().GetObject(runtime_ready_key) != nullptr;
  }

  void install_and_load(std::string const& backend)
  {
    auto& db = loader_.GetDatabaseInstance();
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
      auto& fs           = duckdb::FileSystem::GetLocal(db);
      // Mirrors core's install layout, where install is a no-op when the file exists. If this
      // drifts, the worst case is a missing or extra status line.
      auto directory = duckdb::ExtensionHelper::ExtensionDirectory(db, fs);
      if (repository.type == duckdb::ExtensionRepositoryType::USER_PROVIDED) {
        directory = fs.JoinPath(fs.JoinPath(directory, "repositories"), repository.name);
      }
      auto const installed = fs.FileExists(fs.JoinPath(directory, backend + ".duckdb_extension"));
      if (installed) {
        duckdb::ExtensionHelper::InstallExtension(db, fs, backend, options);
      } else {
        install_with_status(backend, options);
      }
    }
    auto connection = duckdb::Connection::CreateInternal(db);
    connection.Query(duckdb::StringUtil::Format("LOAD %s", duckdb::SQLString(backend)))
      ->ThrowIfError();
  }

  /// Installs a backend that is not installed yet, telling the user on stderr since the
  /// download is large and core DuckDB installs silently.
  void install_with_status(std::string const& backend, duckdb::ExtensionInstallOptions& options)
  {
    auto& db = loader_.GetDatabaseInstance();
    report(duckdb::StringUtil::Format(
      "Sirius: installing %s. This one-time download is large and may take a few minutes.",
      backend));
    auto const start   = std::chrono::steady_clock::now();
    auto const elapsed = [&] {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    try {
      duckdb::ExtensionHelper::InstallExtension(
        db, duckdb::FileSystem::GetLocal(db), backend, options);
    } catch (...) {
      // The caller may fall back to the CPU, which is only logged; close the status line here.
      // The reason stays out of it: it can span lines and contain the URL with credentials.
      report(duckdb::StringUtil::Format(
        "Sirius: installing %s failed after %.1f s.", backend, elapsed()));
      throw;
    }
    report(duckdb::StringUtil::Format("Sirius: installed %s in %.1f s.", backend, elapsed()));
  }

  void report(std::string const& message)
  {
    duckdb::Printer::Print(duckdb::OutputStream::STREAM_STDERR, message);
    duckdb::Printer::Flush(duckdb::OutputStream::STREAM_STDERR);
    log_info(message);
  }

  void log_info(std::string const& message)
  {
    auto& db = loader_.GetDatabaseInstance();
    using namespace duckdb;  // The logging macros use unqualified DuckDB names.
    DUCKDB_LOG_INFO(db, message);
  }

  /// Leaves DuckDB's CPU execution untouched and makes explicit GPU requests fail.
  void register_fallback(loader_status_ptr const& status)
  {
    auto& db     = loader_.GetDatabaseInstance();
    auto& config = duckdb::DBConfig::GetConfig(db);
    // A value configured before the load is still an explicit request; check it before
    // registering anything so a failed LOAD can be retried.
    auto configured = config.options.unrecognized_options.find(gpu_execution_setting);
    if (configured != config.options.unrecognized_options.end()) {
      fallback_gpu_execution_option::reject_enable(*status, configured->second);
    }
    duckdb::ExtensionOption option(
      "Whether to transparently intercept SQL queries and execute them on GPU",
      duckdb::LogicalType::BOOLEAN,
      nullptr,
      duckdb::Value::BOOLEAN(false),
      duckdb::SetScope::SESSION,
      false,
      false);
    option.info = duckdb::make_shared_ptr<fallback_gpu_execution_option>(status);
    config.AddExtensionOption(gpu_execution_setting, std::move(option));
    loader_.RegisterFunction(fallback_gpu_execution_function::create(status));
    log_info("Sirius GPU execution is unavailable; queries run on DuckDB's CPU engine. " +
             status->fallback_reason);
  }

  duckdb::ExtensionLoader& loader_;
};

}  // namespace

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(sirius, loader) { sirius_loader(loader).load(); }
}
