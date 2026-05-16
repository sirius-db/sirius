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

#include "catch.hpp"
#include "io/datasource_factory.hpp"
#include "sirius_config.hpp"

#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

using namespace sirius;
using namespace sirius::io;

namespace {

class mock_ioctx : public sirius_ioctx {
 public:
  void shutdown() override {}

  std::shared_ptr<sirius_io_object> create_io_object(std::string) override
  {
    throw std::logic_error("unused");
  }

  std::unique_ptr<cudf::io::datasource> make_datasource(std::shared_ptr<sirius_io_object>) override
  {
    ++make_datasource_calls;
    throw std::logic_error("mock_ioctx::make_datasource should not run in PR1");
  }

  [[nodiscard]] bool supports(std::string_view) const override { return true; }

  size_t host_read_io(sirius_io_object&, size_t, size_t, uint8_t*) override
  {
    throw std::logic_error("unused");
  }

  void host_read_async_io(
    sirius_io_object&, size_t, size_t, uint8_t*, io_completion_handler) override
  {
    throw std::logic_error("unused");
  }

  size_t device_read_io(sirius_io_object&, size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    throw std::logic_error("unused");
  }

  void device_read_async_io(sirius_io_object&,
                            size_t,
                            size_t,
                            uint8_t*,
                            rmm::cuda_stream_view,
                            io_completion_handler) override
  {
    throw std::logic_error("unused");
  }

  void host_read_ranges_async_io(sirius_io_object&,
                                 std::vector<cudf::io::text::byte_range_info> const&,
                                 std::span<cudf::host_span<std::byte>>,
                                 io_completion_handler) override
  {
    throw std::logic_error("unused");
  }

  cudf::io::text::byte_range_info compute_physical_range(cudf::io::text::byte_range_info logical,
                                                         size_t) const override
  {
    return logical;
  }

  int make_datasource_calls{0};
};

class scoped_test_file {
 public:
  scoped_test_file(std::string_view contents, bool relative)
  {
    static int next_id = 0;
    auto name = "sirius_factory_" + std::to_string(static_cast<long long>(::getpid())) + "_" +
                std::to_string(next_id++) + ".tmp";
    if (relative) {
      display_path_ = name;
      cleanup_path_ = std::filesystem::current_path() / name;
    } else {
      cleanup_path_ = std::filesystem::temp_directory_path() / name;
      display_path_ = cleanup_path_.string();
    }

    std::ofstream out(cleanup_path_, std::ios::binary);
    if (!out.good()) throw std::runtime_error("failed to create temp file");
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!out.good()) throw std::runtime_error("failed to write temp file");
  }

  ~scoped_test_file()
  {
    std::error_code ec;
    std::filesystem::remove(cleanup_path_, ec);
  }

  scoped_test_file(scoped_test_file const&)            = delete;
  scoped_test_file& operator=(scoped_test_file const&) = delete;

  [[nodiscard]] std::string const& path() const noexcept { return display_path_; }

 private:
  std::filesystem::path cleanup_path_;
  std::string display_path_;
};

void require_datasource_contents(cudf::io::datasource& ds, std::string_view expected)
{
  REQUIRE(ds.size() == expected.size());
  auto buf = ds.host_read(0, expected.size());
  REQUIRE(buf != nullptr);
  REQUIRE(buf->size() == expected.size());
  CHECK(std::memcmp(buf->data(), expected.data(), expected.size()) == 0);
}

}  // namespace

TEST_CASE("datasource_factory extract helpers delegate to uri_parser", "[datasource_factory]")
{
  CHECK(datasource_factory::extract_scheme("/abs/path.parquet") == "file");
  CHECK(datasource_factory::extract_scheme("file:///abs/path.parquet") == "file");
  CHECK(datasource_factory::extract_scheme("S3://bucket/key") == "s3");
  CHECK(datasource_factory::extract_path("/abs/path.parquet") == "/abs/path.parquet");
  CHECK(datasource_factory::extract_path("file:///abs/path.parquet") == "/abs/path.parquet");
  CHECK(datasource_factory::extract_path("s3://bucket/key") == "key");
  CHECK_THROWS_AS(datasource_factory::extract_scheme("relative/f.parquet"), std::invalid_argument);
  CHECK_THROWS_AS(datasource_factory::extract_path(""), std::invalid_argument);
}

TEST_CASE("datasource_registry supports register lookup replace and clear", "[datasource_factory]")
{
  datasource_registry reg;
  auto ctx         = std::make_shared<mock_ioctx>();
  auto replacement = std::make_shared<mock_ioctx>();

  CHECK(reg.lookup("s3") == nullptr);

  reg.register_ioctx("s3", ctx);
  REQUIRE(reg.lookup("s3") != nullptr);
  CHECK(reg.lookup("s3").get() == ctx.get());

  reg.register_ioctx("s3", replacement);
  REQUIRE(reg.lookup("s3") != nullptr);
  CHECK(reg.lookup("s3").get() == replacement.get());

  auto schemes = reg.schemes();
  REQUIRE(schemes.size() == 1);
  CHECK(schemes.front() == "s3");

  reg.clear();
  CHECK(reg.lookup("s3") == nullptr);
}

TEST_CASE("datasource_registry scheme matching is case-insensitive", "[datasource_factory]")
{
  datasource_registry reg;
  auto ctx         = std::make_shared<mock_ioctx>();
  auto replacement = std::make_shared<mock_ioctx>();

  reg.register_ioctx("S3", ctx);
  REQUIRE(reg.lookup("s3") != nullptr);
  CHECK(reg.lookup("s3").get() == ctx.get());
  REQUIRE(reg.lookup("S3") != nullptr);
  CHECK(reg.lookup("S3").get() == ctx.get());

  reg.register_ioctx("s3", replacement);
  REQUIRE(reg.lookup("S3") != nullptr);
  CHECK(reg.lookup("S3").get() == replacement.get());

  auto schemes = reg.schemes();
  REQUIRE(schemes.size() == 1);
  CHECK(schemes.front() == "s3");
}

TEST_CASE("datasource_registry rejects invalid registrations", "[datasource_factory]")
{
  datasource_registry reg;

  CHECK_THROWS_AS(reg.register_ioctx("", std::make_shared<mock_ioctx>()), std::invalid_argument);
  CHECK_THROWS_AS(reg.register_ioctx("s3", nullptr), std::invalid_argument);
}

TEST_CASE("datasource_registry handles concurrent lookup", "[datasource_factory]")
{
  datasource_registry reg;
  reg.register_ioctx("s3", std::make_shared<mock_ioctx>());

  constexpr int n_threads = 8;
  constexpr int n_iters   = 1000;
  std::atomic<int> misses{0};
  std::vector<std::thread> threads;
  threads.reserve(n_threads);

  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < n_iters; ++i) {
        if (reg.lookup("s3") == nullptr) { ++misses; }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  CHECK(misses.load() == 0);
}

TEST_CASE("datasource_factory strict create rejects malformed input", "[datasource_factory]")
{
  datasource_registry reg;
  sirius_config cfg;

  CHECK_THROWS_AS(datasource_factory::create("", reg, cfg), std::invalid_argument);
  CHECK_THROWS_AS(datasource_factory::create("relative.parquet", reg, cfg), std::invalid_argument);
}

TEST_CASE("datasource_factory create uses cudf default for file paths", "[datasource_factory]")
{
  constexpr std::string_view expected = "hello sirius factory";
  scoped_test_file tmp(expected, false);
  datasource_registry reg;
  sirius_config cfg;

  auto ds = datasource_factory::create(tmp.path(), reg, cfg);
  REQUIRE(ds != nullptr);
  require_datasource_contents(*ds, expected);

  auto file_uri         = "file://" + tmp.path();
  auto ds_from_file_uri = datasource_factory::create(file_uri, reg, cfg);
  REQUIRE(ds_from_file_uri != nullptr);
  require_datasource_contents(*ds_from_file_uri, expected);
}

TEST_CASE("datasource_factory create_for_parquet_scan accepts relative file paths",
          "[datasource_factory]")
{
  constexpr std::string_view expected = "relative parquet scan path";
  scoped_test_file tmp(expected, true);
  datasource_registry reg;
  sirius_config cfg;

  auto ds = datasource_factory::create_for_parquet_scan(tmp.path(), reg, cfg);
  REQUIRE(ds != nullptr);
  require_datasource_contents(*ds, expected);
}

TEST_CASE("datasource_factory object-store schemes require registered backend",
          "[datasource_factory]")
{
  datasource_registry reg;
  sirius_config cfg;

  CHECK_THROWS_AS(datasource_factory::create("s3://bucket/key.parquet", reg, cfg),
                  std::runtime_error);
  CHECK_THROWS_AS(datasource_factory::create_for_parquet_scan("s3://bucket/key.parquet", reg, cfg),
                  std::runtime_error);
}

TEST_CASE("datasource_factory dispatches mixed-case scheme through registry",
          "[datasource_factory]")
{
  datasource_registry reg;
  sirius_config cfg;
  auto ctx = std::make_shared<mock_ioctx>();
  reg.register_ioctx("s3", ctx);

  try {
    (void)datasource_factory::create("S3://bucket/key.parquet", reg, cfg);
    FAIL("expected object-store construction to be deferred");
  } catch (std::runtime_error const& e) {
    std::string msg = e.what();
    CHECK(msg.find("object construction is not yet implemented") != std::string::npos);
    CHECK(msg.find("no backend registered") == std::string::npos);
  }

  CHECK(ctx->make_datasource_calls == 0);
}

TEST_CASE("datasource_factory PR1 does not construct object-store datasources",
          "[datasource_factory]")
{
  datasource_registry reg;
  sirius_config cfg;
  auto ctx = std::make_shared<mock_ioctx>();
  reg.register_ioctx("s3", ctx);

  try {
    (void)datasource_factory::create("s3://bucket/key.parquet", reg, cfg);
    FAIL("expected object-store construction to be deferred");
  } catch (std::runtime_error const& e) {
    std::string msg = e.what();
    CHECK(msg.find("object construction is not yet implemented") != std::string::npos);
  }

  CHECK(ctx->make_datasource_calls == 0);
}
