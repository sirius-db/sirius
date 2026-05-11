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

#include "catch.hpp"
#include "io/s3/mock_credential_provider.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

using sirius::io::s3::mock_credential_provider;
using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace {

s3_ioctx_config make_mock_s3_config()
{
  s3_ioctx_config cfg{};
  cfg.creds             = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  cfg.max_connections   = 2;
  cfg.request_timeout_s = 1;
  cfg.max_retry_attempts = 1;
  cfg.retry_backoff_base = std::chrono::milliseconds{0};
  cfg.retry_jitter       = std::chrono::milliseconds{0};
  cfg.honor_retry_after  = false;
  return cfg;
}

std::string make_file_uri(std::string const& tag)
{
  auto path = std::filesystem::temp_directory_path() / ("sirius_scan_manager_s3_" + tag);
  std::ofstream out(path);
  out << "local";
  out.close();
  return "file://" + path.string();
}

sirius::sirius_config make_context_config()
{
  auto const config_path = std::filesystem::path(__FILE__).parent_path().parent_path() /
                           "integration" / "integration.yaml";
  sirius::sirius_config cfg{};
  cfg.load_from_file(config_path);
  REQUIRE_FALSE(cfg.get_memory_space_configs().empty());
  return cfg;
}

}  // namespace

TEST_CASE("sirius_scan_manager constructs S3 backend and dispatches by path", "[scan_manager][s3]")
{
  auto const local_uri = make_file_uri("dispatch.dat");

  scan_manager_config cfg{};
  cfg.s3_config                         = make_mock_s3_config();
  cfg.s3_thread_pool.num_threads        = 2;
  cfg.s3_thread_pool.thread_name_prefix = "s3_io_test";

  sirius_scan_manager manager(std::move(cfg));

  auto* default_ctx = manager.io_ctx();
  REQUIRE(default_ctx != nullptr);

  CHECK(manager.io_ctx_for(local_uri) == default_ctx);

  auto* s3_ctx = manager.io_ctx_for("s3://bucket/key.parquet");
  REQUIRE(s3_ctx != nullptr);
  CHECK(s3_ctx != default_ctx);
  CHECK(dynamic_cast<s3_ioctx*>(s3_ctx) != nullptr);

  CHECK(manager.io_ctx_for("unsupported://bucket/key.parquet") == nullptr);
}

TEST_CASE("sirius_scan_manager leaves S3 disabled when s3_config is empty", "[scan_manager][s3]")
{
  scan_manager_config cfg{};
  sirius_scan_manager manager(std::move(cfg));

  REQUIRE(manager.io_ctx() != nullptr);
  CHECK(manager.io_ctx_for("s3://bucket/key.parquet") == nullptr);
  CHECK(manager.io_ctx_for("unsupported://bucket/key.parquet") == nullptr);
}

TEST_CASE("sirius_scan_manager stop is idempotent with both uring and S3 backends",
          "[scan_manager][s3]")
{
  scan_manager_config cfg{};
  cfg.s3_config = make_mock_s3_config();

  sirius_scan_manager manager(std::move(cfg));
  REQUIRE(manager.io_ctx_for("s3://bucket/key.parquet") != nullptr);

  manager.stop();
  manager.stop();
}

TEST_CASE("sirius_config carries object_store_config and defaults keep S3 disabled",
          "[sirius][config][s3]")
{
  sirius::sirius_config cfg{};

  CHECK(cfg.object_store_config.endpoint.empty());
  CHECK(cfg.object_store_config.access_key.empty());
  CHECK_FALSE(cfg.get_scan_manager_config().s3_config.has_value());
}

TEST_CASE("SiriusContext initialize keeps empty object_store_config inert",
          "[sirius][context][s3][isolated_context]")
{
  auto cfg = make_context_config();

  duckdb::SiriusContext context;
  context.initialize(cfg);
  CHECK(context.get_scan_manager().io_ctx_for("s3://bucket/key.parquet") == nullptr);
  context.terminate();
}

TEST_CASE("SiriusContext initialize wires populated object_store_config into scan_manager",
          "[sirius][context][s3][isolated_context]")
{
  auto cfg                           = make_context_config();
  cfg.object_store_config.endpoint   = "http://127.0.0.1:9000";
  cfg.object_store_config.region     = "us-east-1";
  cfg.object_store_config.access_key = "minioadmin";
  cfg.object_store_config.secret_key = "minioadmin";

  duckdb::SiriusContext context;
  context.initialize(cfg);

  auto* s3_ctx = context.get_scan_manager().io_ctx_for("s3://bucket/nation.parquet");
  REQUIRE(s3_ctx != nullptr);
  CHECK(dynamic_cast<s3_ioctx*>(s3_ctx) != nullptr);

  context.terminate();
}
