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

// An overwritten S3 parquet object must bind with its new schema: describe_parquet
// may skip the footer probe on a path hint (metadata_store::has_path), but metadata
// is reused only for the exact object generation (raw_file_cache_id). The cache
// internals behind this are covered by cuCascade's test_cache_object_identity and
// test_rest_cache_identity; these cases cover Sirius's bind path on top of them.

#include "catch.hpp"
#include "op/scan/parquet_metadata.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "utils/s3_backend.hpp"
#include "utils/s3_test_env.hpp"

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <duckdb.hpp>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <system_error>
#include <vector>

namespace {

using cucascade::io::rest::rest_io_object;
using sirius::test::s3::env_or;
using sirius::test::s3::require_env;
using sirius::test::s3::sql_quote;

/// Two same-size parquet objects with different schemas, published in turn to one key.
struct parquet_objects {
  parquet_objects()
  {
    std::string pattern = (std::filesystem::temp_directory_path() / "sirius-c2-XXXXXX").string();
    auto* dir           = ::mkdtemp(pattern.data());
    REQUIRE(dir != nullptr);
    directory = dir;
    key       = "cache-identity/" + directory.filename().string() + "/object.parquet";
    uri       = "s3://" + require_env("SIRIUS_TEST_S3_BUCKET") + "/" + key;
    first     = generate(1, "column_a");
    second    = generate(2, "column_b");
    REQUIRE(first.size() == second.size());
    REQUIRE(first != second);
  }

  ~parquet_objects()
  {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }

  std::vector<std::uint8_t> generate(int value, std::string const& name)
  {
    auto const path = directory / ("generation-" + std::to_string(value) + ".parquet");
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    auto result = con.Query("COPY (SELECT " + std::to_string(value) + "::INTEGER AS " + name +
                            " FROM range(4096)) TO " + sql_quote(path.string()) +
                            " (FORMAT PARQUET, COMPRESSION UNCOMPRESSED)");
    REQUIRE(result != nullptr);
    INFO((result->HasError() ? result->GetError() : ""));
    REQUIRE_FALSE(result->HasError());
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }

  void publish(std::span<std::uint8_t const> bytes)
  {
    REQUIRE(sirius::test::put_s3_test_object(key, bytes));
  }

  std::filesystem::path directory;
  std::string key;
  std::string uri;
  std::vector<std::uint8_t> first;
  std::vector<std::uint8_t> second;
};

void metadata_overwrite(cucascade::io::cache::cache_mode mode)
{
  parquet_objects objects;
  objects.publish(objects.first);
  auto memory = initialize_memory_manager(1);
  sirius::scan_manager::scan_manager_config cfg;
  cfg.backend                                = sirius::scan_manager::io_backend::native;
  cfg.object_store.endpoint                  = require_env("SIRIUS_TEST_S3_ENDPOINT");
  cfg.object_store.region                    = env_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  cfg.object_store.access_key                = require_env("SIRIUS_TEST_S3_ACCESS_KEY");
  cfg.object_store.secret_key                = require_env("SIRIUS_TEST_S3_SECRET_KEY");
  cfg.object_store.session_token             = env_or("SIRIUS_TEST_S3_SESSION_TOKEN");
  cfg.object_store.tls_verify                = false;
  cfg.cache.mode                             = mode;
  cfg.cache.eviction                         = cucascade::io::cache::eviction_policy::lru;
  cfg.cache.min_prefetching_budget_fraction = 0.5;
  cfg.cache.eviction_threshold_fraction     = 1.0;
  cfg.uring_n_reactors                       = 1;
  cfg.rest_n_reactors                        = 1;
  cfg.apply_cache_mode();
  sirius::scan_manager::sirius_scan_manager manager{
    cfg, *memory, sirius::test::s3::single_gpu_index(0)};

  auto first_shape = manager.describe_parquet(objects.uri);
  REQUIRE(first_shape.names.size() == 1);
  CHECK(first_shape.names.front() == "column_a");
  auto first        = manager.create_datasource(objects.uri);
  auto old_metadata = first->metadata();
  REQUIRE(old_metadata != nullptr);
  auto const old_key = first->get_io_object().raw_file_cache_id();
  auto const old_tag = std::string(first->get_io_object().validation_tag());
  REQUIRE_FALSE(old_tag.empty());
  CHECK(old_key == rest_io_object::generation_key(objects.uri, old_tag));
  auto& store = first->io_ctx()->metadata_store();
  CHECK(store.has_path(objects.uri));

  objects.publish(objects.second);
  auto second_shape = manager.describe_parquet(objects.uri);
  REQUIRE(second_shape.names.size() == 1);
  CHECK(second_shape.names.front() == "column_b");
  auto second = manager.create_datasource(objects.uri);
  REQUIRE(second->get_io_object().validation_tag() != old_tag);
  CHECK(second->get_io_object().raw_file_cache_id() != old_key);
  CHECK(store.has_path(objects.uri));
  CHECK(store.get_metadata(old_key) == nullptr);
  CHECK(second->metadata() != nullptr);

  // The first datasource still holds the old generation's parsed footer.
  auto parsed = std::dynamic_pointer_cast<sirius::op::scan::parquet_metadata>(old_metadata);
  REQUIRE(parsed != nullptr);
  REQUIRE(parsed->file_metadata() != nullptr);
  REQUIRE(parsed->file_metadata()->schema.size() == 2);
  CHECK(parsed->file_metadata()->schema[1].name == "column_a");
}

}  // namespace

TEST_CASE("describe_parquet replaces metadata of an overwritten object with the byte cache enabled",
          "[s3][integration][cache_identity][describe_parquet]")
{
  if (sirius::test::s3::skip_or_fail_unless(sirius::test::ensure_s3_test_env(),
                                            "SeaweedFS test environment is not available")) {
    return;
  }
  metadata_overwrite(cucascade::io::cache::cache_mode::cucs);
}

TEST_CASE("describe_parquet replaces metadata of an overwritten object with the byte cache disabled",
          "[s3][integration][cache_identity][describe_parquet]")
{
  if (sirius::test::s3::skip_or_fail_unless(sirius::test::ensure_s3_test_env(),
                                            "SeaweedFS test environment is not available")) {
    return;
  }
  metadata_overwrite(cucascade::io::cache::cache_mode::none);
}
