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
#include "io/rest/rest_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "io/types.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/parquet_metadata.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "utils/s3_container.hpp"

#include <cucascade/memory/topology_discovery.hpp>
#include <duckdb.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using sirius::io::io_context_type;
using sirius::io::rest::rest_ioctx;
using sirius::scan_manager::parquet_bind_result;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

namespace fs = std::filesystem;

std::string env_or(std::string const& name, std::string fallback = {})
{
  if (auto* value = std::getenv(name.c_str()); value != nullptr) { return value; }
  return fallback;
}

std::string require_env(std::string const& name)
{
  auto value = env_or(name);
  REQUIRE_FALSE(value.empty());
  return value;
}

cucascade::memory::system_topology_info single_gpu_topology()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return topology;
}

std::shared_ptr<const sirius::memory::topology_index> single_gpu_index()
{
  return std::make_shared<sirius::memory::topology_index>(single_gpu_topology(),
                                                          std::vector<int>{0});
}

struct scan_manager_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  std::shared_ptr<const sirius::memory::topology_index> topology = single_gpu_index();
};

scan_manager_config make_minio_rest_config()
{
  scan_manager_config cfg{};
  cfg.backend                 = sirius::scan_manager::io_backend::sirius;
  cfg.object_store.endpoint   = require_env("SIRIUS_TEST_S3_ENDPOINT");
  cfg.object_store.region     = env_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  cfg.object_store.access_key = require_env("SIRIUS_TEST_S3_ACCESS_KEY");
  cfg.object_store.secret_key = require_env("SIRIUS_TEST_S3_SECRET_KEY");
  cfg.object_store.tls_verify = false;
  cfg.rest.request_timeout_s  = 30;
  cfg.rest.max_connections    = 8;
  cfg.rest_n_reactors         = 1;
  cfg.cache.mode              = sirius::io::cache::cache_mode::none;
  return cfg;
}

std::string parquet_uri(std::string const& bucket, std::string const& file_name)
{
  return "s3://" + bucket + "/parquet/" + file_name;
}

std::string sql_quote(std::string_view value)
{
  std::string out{"'"};
  for (char c : value) {
    if (c == '\'') { out.push_back('\''); }
    out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

fs::path parquet_fixture(std::string_view file_name)
{
  return fs::path{SIRIUS_PROJECT_ROOT} / "test" / "cpp" / "integration" / "data" / "parquet" /
         file_name;
}

struct duckdb_parquet_bind_shape {
  duckdb::vector<duckdb::LogicalType> types;
  duckdb::vector<std::string> names;
};

duckdb_parquet_bind_shape duckdb_read_parquet_shape(fs::path const& path)
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto result = con.Query("SELECT * FROM read_parquet(" + sql_quote(path.string()) + ") LIMIT 0");
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
  return duckdb_parquet_bind_shape{result->types, result->names};
}

std::vector<std::string> bind_names(parquet_bind_result const& result)
{
  return {result.names.begin(), result.names.end()};
}

std::vector<std::string> bind_type_strings(parquet_bind_result const& result)
{
  std::vector<std::string> out;
  out.reserve(result.return_types.size());
  for (auto const& type : result.return_types) {
    out.push_back(type.ToString());
  }
  return out;
}

void require_same_bind_result(parquet_bind_result const& lhs, parquet_bind_result const& rhs)
{
  REQUIRE(lhs.object_size == rhs.object_size);
  REQUIRE(lhs.total_num_rows == rhs.total_num_rows);
  REQUIRE(bind_names(lhs) == bind_names(rhs));
  REQUIRE(bind_type_strings(lhs) == bind_type_strings(rhs));
}

void check_bind_shape_matches_duckdb(parquet_bind_result const& actual,
                                     duckdb_parquet_bind_shape const& expected)
{
  REQUIRE(actual.names == expected.names);
  REQUIRE(actual.return_types.size() == expected.types.size());
  for (std::size_t i = 0; i < expected.types.size(); ++i) {
    INFO("column=" << expected.names[i] << " actual=" << actual.return_types[i].ToString()
                   << " expected=" << expected.types[i].ToString());
    CHECK(actual.return_types[i] == expected.types[i]);
  }
}

rest_ioctx* require_rest_ioctx(std::shared_ptr<sirius::io::sirius_datasource> const& ds)
{
  REQUIRE(ds != nullptr);
  REQUIRE(ds->io_ctx() != nullptr);
  CHECK(ds->io_ctx()->type() == io_context_type::restful);
  auto* rest_ctx = dynamic_cast<rest_ioctx*>(ds->io_ctx().get());
  REQUIRE(rest_ctx != nullptr);
  return rest_ctx;
}

rest_ioctx* require_rest_ioctx_for(sirius_scan_manager& manager, std::string const& uri)
{
  return require_rest_ioctx(manager.create_datasource(uri));
}

}  // namespace

TEST_CASE("describe_parquet routes S3 parquet through rest_ioctx and returns nation schema",
          "[s3][integration][describe_parquet]")
{
  if (!sirius::test::ensure_s3_container_env()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  auto const uri    = parquet_uri(bucket, "nation.parquet");

  scan_manager_fixture fixture;
  sirius_scan_manager manager{make_minio_rest_config(), *fixture.memory, fixture.topology};

  auto result = manager.describe_parquet(uri);
  require_rest_ioctx_for(manager, uri);

  CHECK(bind_names(result) ==
        std::vector<std::string>{"n_nationkey", "n_name", "n_regionkey", "n_comment"});
  REQUIRE(result.return_types.size() == result.names.size());
  REQUIRE(result.return_types.size() == 4);
  auto const type_strings = bind_type_strings(result);
  INFO("nation type strings: " << type_strings[0] << ", " << type_strings[1] << ", "
                               << type_strings[2] << ", " << type_strings[3]);
  for (auto const& type : type_strings) {
    CHECK_FALSE(type.empty());
  }
  CHECK(result.total_num_rows == 25);

  auto datasource = manager.create_datasource(uri);
  REQUIRE(datasource != nullptr);
  CHECK(result.object_size == datasource->size());
  CHECK(result.object_size > 0);
}

TEST_CASE("describe_parquet reports stable row counts for multiple S3 parquet objects",
          "[s3][integration][describe_parquet]")
{
  if (!sirius::test::ensure_s3_container_env()) { return; }

  auto const bucket     = require_env("SIRIUS_TEST_S3_BUCKET");
  auto const nation_uri = parquet_uri(bucket, "nation.parquet");
  auto const region_uri = parquet_uri(bucket, "region.parquet");

  scan_manager_fixture fixture;
  sirius_scan_manager manager{make_minio_rest_config(), *fixture.memory, fixture.topology};

  auto nation = manager.describe_parquet(nation_uri);
  auto region = manager.describe_parquet(region_uri);

  CHECK(nation.total_num_rows == 25);
  CHECK(region.total_num_rows == 5);
  CHECK(bind_names(region) == std::vector<std::string>{"r_regionkey", "r_name", "r_comment"});
  CHECK(region.object_size > 0);
}

TEST_CASE("describe_parquet maps nested local parquet bind shape like DuckDB CPU read_parquet",
          "[scan_manager][describe_parquet][s3][nested]")
{
  scan_manager_fixture fixture;
  scan_manager_config cfg{};
  cfg.backend = sirius::scan_manager::io_backend::sirius;
  sirius_scan_manager manager{std::move(cfg), *fixture.memory, fixture.topology};

  for (auto const fixture_name : {"nested_struct.parquet",
                                  "nested_list.parquet",
                                  "nested_map.parquet",
                                  "nested_deep.parquet"}) {
    auto const path     = parquet_fixture(fixture_name);
    auto const expected = duckdb_read_parquet_shape(path);
    auto const uri      = "file://" + path.string();

    auto bind_info = manager.describe_parquet(uri);

    INFO("fixture=" << fixture_name);
    check_bind_shape_matches_duckdb(bind_info, expected);
    CHECK(bind_info.total_num_rows > 0);
    CHECK(bind_info.object_size > 0);
  }
}

TEST_CASE("describe_parquet maps nested S3 parquet bind shape like DuckDB CPU read_parquet",
          "[s3][integration][describe_parquet][nested]")
{
  if (!sirius::test::ensure_s3_container_env()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");

  scan_manager_fixture fixture;
  sirius_scan_manager manager{make_minio_rest_config(), *fixture.memory, fixture.topology};

  for (auto const fixture_name : {"nested_struct.parquet",
                                  "nested_list.parquet",
                                  "nested_map.parquet",
                                  "nested_deep.parquet"}) {
    auto const local_path = parquet_fixture(fixture_name);
    auto const expected   = duckdb_read_parquet_shape(local_path);
    auto const uri        = parquet_uri(bucket, fixture_name);

    auto bind_info = manager.describe_parquet(uri);

    INFO("fixture=" << fixture_name);
    check_bind_shape_matches_duckdb(bind_info, expected);
    CHECK(bind_info.total_num_rows > 0);
    CHECK(bind_info.object_size > 0);
  }
}

TEST_CASE("describe_parquet surfaces missing S3 parquet objects from HEAD",
          "[s3][integration][describe_parquet]")
{
  if (!sirius::test::ensure_s3_container_env()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  auto const uri    = parquet_uri(bucket, "does-not-exist.parquet");

  scan_manager_fixture fixture;
  sirius_scan_manager manager{make_minio_rest_config(), *fixture.memory, fixture.topology};

  try {
    (void)manager.describe_parquet(uri);
    FAIL("describe_parquet unexpectedly succeeded for a missing S3 object");
  } catch (std::runtime_error const& e) {
    auto const message = std::string{e.what()};
    CHECK(message.find("404") != std::string::npos);
  }
}

TEST_CASE("describe_parquet parks parsed parquet metadata in the rest metadata store",
          "[s3][integration][describe_parquet]")
{
  if (!sirius::test::ensure_s3_container_env()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  auto const uri    = parquet_uri(bucket, "nation.parquet");

  scan_manager_fixture fixture;
  sirius_scan_manager manager{make_minio_rest_config(), *fixture.memory, fixture.topology};

  auto result = manager.describe_parquet(uri);

  auto datasource = manager.create_datasource(uri);
  REQUIRE(datasource != nullptr);
  auto metadata = datasource->metadata();
  REQUIRE(metadata != nullptr);
  auto parquet_metadata =
    std::dynamic_pointer_cast<sirius::op::scan::parquet_metadata>(std::move(metadata));
  REQUIRE(parquet_metadata != nullptr);
  REQUIRE(parquet_metadata->file_metadata() != nullptr);
  CHECK(static_cast<std::size_t>(parquet_metadata->file_metadata()->num_rows) ==
        result.total_num_rows);
  CHECK(result.total_num_rows == 25);
  CHECK(parquet_metadata->footer_byte_len() > 0);
}

TEST_CASE("describe_parquet reuses the metadata store on repeated S3 binds",
          "[s3][integration][describe_parquet]")
{
  if (!sirius::test::ensure_s3_container_env()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  auto const uri    = parquet_uri(bucket, "nation.parquet");

  scan_manager_fixture fixture;
  sirius_scan_manager manager{make_minio_rest_config(), *fixture.memory, fixture.topology};

  auto cold = manager.describe_parquet(uri);
  auto warm = manager.describe_parquet(uri);
  require_same_bind_result(cold, warm);
}

TEST_CASE("describe_parquet handles small and larger S3 parquet objects",
          "[s3][integration][describe_parquet]")
{
  if (!sirius::test::ensure_s3_container_env()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");

  auto require_valid_description = [&](std::string const& file_name) {
    scan_manager_fixture fixture;
    sirius_scan_manager manager{make_minio_rest_config(), *fixture.memory, fixture.topology};
    auto result = manager.describe_parquet(parquet_uri(bucket, file_name));

    INFO(file_name << " object_size: " << result.object_size);
    CHECK(result.object_size > 0);
    CHECK(result.total_num_rows > 0);
  };

  require_valid_description("nation.parquet");
  require_valid_description("lineitem.parquet");
}
