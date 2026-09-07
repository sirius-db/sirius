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

#include <catch.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <io/object_store_config.hpp>
#include <io/sirius_datasource.hpp>
#include <unistd.h>  // getpid

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace io = sirius::io;

/// RAII temp file with known contents, so the local branch of
/// create_io_object is exercised without touching the repo tree.
class scoped_temp_file {
 public:
  explicit scoped_temp_file(std::string const& contents)
    : _path(std::filesystem::temp_directory_path() /
            ("sirius_kvikio_local_" + std::to_string(::getpid()) + ".bin"))
  {
    std::ofstream out{_path, std::ios::binary | std::ios::trunc};
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  }

  ~scoped_temp_file()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
  }

  scoped_temp_file(scoped_temp_file const&)            = delete;
  scoped_temp_file& operator=(scoped_temp_file const&) = delete;

  [[nodiscard]] std::string string() const { return _path.string(); }

 private:
  std::filesystem::path _path;
};

io::object_store_config configured_store()
{
  io::object_store_config os;
  os.endpoint   = "https://s3.invalid.test";
  os.region     = "us-east-1";
  os.access_key = "test-access-key";
  os.secret_key = "test-secret-key";
  return os;
}

}  // namespace

TEST_CASE("kvikio_context rejects s3 URIs when the object store is unconfigured", "[kvikio]")
{
  auto ctx = std::make_shared<io::kvikio_context>(io::kvikio_config{}, io::object_store_config{});

  REQUIRE_THROWS_WITH(ctx->open_datasource("s3://bucket/key.parquet"),
                      Catch::Contains("object store not configured"));
}

TEST_CASE("kvikio_context rejects malformed s3 URIs before any network call", "[kvikio]")
{
  auto ctx = std::make_shared<io::kvikio_context>(io::kvikio_config{}, configured_store());

  // No object part: kvikIO's URI split rejects it, so the failure happens
  // locally rather than as a connection error against the endpoint.
  REQUIRE_THROWS(ctx->open_datasource("s3://bucket-only"));
  REQUIRE_THROWS(ctx->open_datasource("s3://bucket-only/"));
}

TEST_CASE("kvikio_context still serves local paths when an object store is configured", "[kvikio]")
{
  std::string const contents = "sirius-kvikio-local-bytes";
  scoped_temp_file file{contents};
  auto ctx = std::make_shared<io::kvikio_context>(io::kvikio_config{}, configured_store());

  auto ds = ctx->open_datasource(file.string());
  REQUIRE(ds != nullptr);
  REQUIRE(ds->size() == contents.size());

  std::vector<uint8_t> buffer(contents.size(), 0);
  auto const read = ds->host_read(0, buffer.size(), buffer.data());
  REQUIRE(read == contents.size());
  REQUIRE(std::string(buffer.begin(), buffer.end()) == contents);
}

TEST_CASE("kvikio_context clamps local reads past the end of the object", "[kvikio]")
{
  std::string const contents = "0123456789";
  scoped_temp_file file{contents};
  auto ctx = std::make_shared<io::kvikio_context>(io::kvikio_config{});

  auto ds = ctx->open_datasource(file.string());
  REQUIRE(ds != nullptr);

  std::vector<uint8_t> buffer(64, 0);
  REQUIRE(ds->host_read(4, buffer.size(), buffer.data()) == contents.size() - 4);
  REQUIRE(ds->host_read(contents.size(), buffer.size(), buffer.data()) == 0);
  REQUIRE(ds->host_read(contents.size() + 100, buffer.size(), buffer.data()) == 0);
}
