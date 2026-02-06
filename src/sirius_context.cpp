/*
 * Copyright 2025, Sirius Contributors.
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

#include "sirius_context.hpp"

#include "duckdb/common/helper.hpp"
#include "duckdb/main/client_context.hpp"
#include "exec/config.hpp"
#include "extension_lock.hpp"
#include "log/logging.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/duckdb_scan_executor.hpp"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <cstdlib>  // for std::getenv
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

namespace duckdb {

namespace {

static constexpr std::string_view CONFIG_FILE_NAME     = "sirius.cfg";
static constexpr std::string_view CONFIG_FILE_DIR      = ".sirius";
static constexpr std::string_view CONFIG_FILE_ENV_NAME = "SIRIUS_CONFIG_FILE";

std::string get_config_file_path()
{
  std::string config_path;

  const char* config_path_cstr = std::getenv(std::string(CONFIG_FILE_ENV_NAME).c_str());
  if (config_path_cstr != nullptr) {
    config_path = std::string(config_path_cstr);
  } else {
    // construct default path
    const char* home_dir = std::getenv("HOME");
    if (home_dir == nullptr) {
      throw std::runtime_error(
        "HOME environment variable is not set. Skipping Sirius config file loading.");
    }
    config_path = std::string(home_dir) + "/" + std::string(CONFIG_FILE_DIR) + "/" +
                  std::string(CONFIG_FILE_NAME);
  }

  // check if file exists
  return config_path;
}

}  // namespace

// ================= sirius_context ================= //

SiriusContext::SiriusContext() = default;

SiriusContext::~SiriusContext() noexcept
{
  if (is_initialized_) { terminate(); }
}

void SiriusContext::QueryBegin(ClientContext& context)
{
  auto query = context.GetCurrentQuery();
  if (config_.is_scan_caching_enabled()) {
    pipeline_executor_->get_scan_executor().cache_scan_results_for_query(query);
  }

  // Reset task creator state (including scan operator global state map) for the new query
  task_creator_->reset();
  task_creator_->set_client_context(context);
}

void SiriusContext::QueryEnd() { query_.reset(); }

void SiriusContext::QueryEnd(ClientContext& context) { QueryEnd(); }

void SiriusContext::QueryEnd(ClientContext& context, optional_ptr<ErrorData> error)
{
  QueryEnd(context);
}

void SiriusContext::initialize(const sirius::sirius_config& config)
{
  if (is_initialized_) { throw std::runtime_error("Sirius context is already initialized."); }

  config_ = config;

  memory_manager_ = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(
    config_.get_memory_space_configs());

  data_repository_manager_ = std::make_unique<cucascade::shared_data_repository_manager>();

  pipeline_executor_ = std::make_unique<sirius::pipeline::pipeline_executor>(
    config_.get_gpu_pipeline_executor_config(),
    config_.get_duckdb_scan_executor_config(),
    *memory_manager_,
    &config_.get_hw_topology());

  downgrade_executor_ = std::make_unique<sirius::parallel::downgrade_executor>(
    sirius::parallel::task_executor_config{}, *data_repository_manager_);

  task_creator_ = std::make_unique<sirius::creator::task_creator>(config_.get_task_creator_config(),
                                                                  *memory_manager_);
  task_creator_->set_pipeline_executor(*pipeline_executor_);
  pipeline_executor_->set_task_creator(*task_creator_);
  task_creator_->start_thread_pool();
  pipeline_executor_->start();

  // Configure scan caching based on config
  pipeline_executor_->set_scan_caching_enabled(config_.is_scan_caching_enabled());

  is_initialized_ = true;
}

void SiriusContext::terminate()
{
  throw_if_not_initialized();

  memory_manager_->shutdown();
  memory_manager_.reset();

  is_initialized_ = false;
}

sirius::memory::sirius_memory_reservation_manager& SiriusContext::get_memory_manager()
{
  throw_if_not_initialized();
  return *memory_manager_;
}

const sirius::memory::sirius_memory_reservation_manager& SiriusContext::get_memory_manager() const
{
  throw_if_not_initialized();
  return *memory_manager_;
}

cucascade::shared_data_repository_manager& SiriusContext::get_data_repository_manager()
{
  throw_if_not_initialized();
  return *data_repository_manager_;
}

const cucascade::shared_data_repository_manager& SiriusContext::get_data_repository_manager() const
{
  throw_if_not_initialized();
  return *data_repository_manager_;
}

sirius::pipeline::pipeline_executor& SiriusContext::get_pipeline_executor()
{
  throw_if_not_initialized();
  return *pipeline_executor_;
}

const sirius::pipeline::pipeline_executor& SiriusContext::get_pipeline_executor() const
{
  throw_if_not_initialized();
  return *pipeline_executor_;
}

sirius::parallel::downgrade_executor& SiriusContext::get_downgrade_executor()
{
  throw_if_not_initialized();
  return *downgrade_executor_;
}

const sirius::parallel::downgrade_executor& SiriusContext::get_downgrade_executor() const
{
  throw_if_not_initialized();
  return *downgrade_executor_;
}

sirius::creator::task_creator& SiriusContext::get_task_creator()
{
  throw_if_not_initialized();
  return *task_creator_;
}

const sirius::creator::task_creator& SiriusContext::get_task_creator() const
{
  throw_if_not_initialized();
  return *task_creator_;
}

void SiriusContext::create_query(sirius::sirius_pipeline_hashmap pipeline_hashmap)
{
  throw_if_not_initialized();
  query_ = duckdb::make_shared_ptr<sirius::planner::query>(std::move(pipeline_hashmap));
  pipeline_executor_->prepare_for_query(query_);
}

duckdb::shared_ptr<sirius::planner::query> SiriusContext::get_query()
{
  throw_if_not_initialized();
  return query_;
}

duckdb::shared_ptr<const sirius::planner::query> SiriusContext::get_query() const
{
  throw_if_not_initialized();
  return query_;
}

void SiriusContext::throw_if_not_initialized() const
{
  if (!is_initialized_) { throw std::runtime_error("Sirius context is not initialized."); }
}

// ================= Free Functions ================= //

SiriusContextExtensionCallback::SiriusContextExtensionCallback()
{
  InitGlobalLogger();
  read_config_file_if_exists();
}

void SiriusContextExtensionCallback::OnConnectionOpened(ClientContext& context)
{
  spdlog::info("Connection opened.");
  if (context_) { context.registered_state->Insert("sirius_state", context_); }
}

void SiriusContextExtensionCallback::OnConnectionClosed(ClientContext& context)
{
  spdlog::info("Connection closed.");
}

void SiriusContextExtensionCallback::OnExtensionLoaded(DatabaseInstance& db, const string& name)
{
  spdlog::info("Extension loaded: {}", name);
}

void SiriusContextExtensionCallback::OnBeginExtensionLoad(DatabaseInstance& db, const string& name)
{
  spdlog::info("Beginning to load extension: {}", name);
}

void SiriusContextExtensionCallback::OnExtensionLoadFail(DatabaseInstance& db,
                                                         const string& name,
                                                         const ErrorData& error)
{
  spdlog::error("Failed to load extension: {}. Error: {}", name, error.RawMessage());
}

void SiriusContextExtensionCallback::read_config_file_if_exists()
{
  auto config_path = get_config_file_path();
  if (!std::filesystem::exists(config_path)) {
    spdlog::info("Sirius configuration file does not exist at path: {}. Skipping loading.",
                 config_path);
    return;
  }
  config_.load_from_file(config_path);
  spdlog::info("Loaded Sirius configuration from file: {}", config_path);

  // Determine lock prefix: check if $HOME/.sirius directory exists
  std::string lock_prefix = "/var/tmp";
  const char* home_dir    = std::getenv("HOME");
  if (home_dir != nullptr) {
    std::string sirius_dir = std::string(home_dir) + "/" + std::string(CONFIG_FILE_DIR);
    if (!std::filesystem::exists(sirius_dir)) {
      // Create the directory if it doesn't exist
      std::filesystem::create_directories(sirius_dir);
      spdlog::info("Created Sirius directory: {}", sirius_dir);
    }
    lock_prefix = sirius_dir;
  }

  extension_lock_ = std::make_unique<sirius::extension_lock>("sirius", lock_prefix);
  context_        = duckdb::make_shared_ptr<SiriusContext>();
  context_->initialize(config_);
}

}  // namespace duckdb
