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
#include <barrier>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace sirius;
using namespace sirius::io;

namespace {

// ---------------------------------------------------------------------------
// mock ioctx
// ---------------------------------------------------------------------------
// A test-only sirius_ioctx that satisfies the current IO-framework contract.
// These tests never drive actual IO through the mock; every read entry point
// throws so misuse is surfaced immediately.
// ---------------------------------------------------------------------------

class mock_ioctx : public sirius_ioctx {
 public:
  std::atomic<int> make_datasource_calls{0};

  void shutdown() override {}

  std::unique_ptr<cudf::io::datasource> make_datasource(
    std::shared_ptr<sirius_io_object> /*io_object*/) override
  {
    make_datasource_calls.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error("mock_ioctx::make_datasource: not exercised in PR1");
  }

  // Unused read APIs — datasource_factory coverage never drives IO through the
  // mock backend.
  size_t host_read(sirius_io_object&, size_t, size_t, uint8_t*) override
  {
    throw std::logic_error("unused");
  }
  std::unique_ptr<cudf::io::datasource::buffer> host_read(sirius_io_object&,
                                                          size_t,
                                                          size_t) override
  {
    throw std::logic_error("unused");
  }
  void host_read_async(sirius_io_object&, size_t, size_t, uint8_t*, io_completion_handler) override
  {
    throw std::logic_error("unused");
  }
  std::unique_ptr<cudf::io::datasource::buffer> device_read_io(sirius_io_object&,
                                                               size_t,
                                                               size_t,
                                                               rmm::cuda_stream_view) override
  {
    throw std::logic_error("unused");
  }
  size_t device_read_io(sirius_io_object&, size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    throw std::logic_error("unused");
  }
  void device_read_io_async(sirius_io_object&,
                            size_t,
                            size_t,
                            uint8_t*,
                            rmm::cuda_stream_view,
                            io_completion_handler) override
  {
    throw std::logic_error("unused");
  }
  void host_read_ranges_async(sirius_io_object&,
                              std::vector<cudf::io::text::byte_range_info> const&,
                              std::span<cudf::host_span<std::byte>>,
                              io_completion_handler) override
  {
    throw std::logic_error("unused");
  }
  size_t host_read_ranges(sirius_io_object&,
                          std::vector<cudf::io::text::byte_range_info> const&,
                          std::span<cudf::host_span<std::byte>>) override
  {
    throw std::logic_error("unused");
  }
  cudf::io::text::byte_range_info compute_physical_range(cudf::io::text::byte_range_info logical,
                                                         size_t) const override
  {
    return logical;
  }
};

// ---------------------------------------------------------------------------
// temp-file RAII for the happy-path test
// ---------------------------------------------------------------------------

class scoped_temp_file {
 public:
  explicit scoped_temp_file(std::string_view contents)
  {
    auto tmpl = std::filesystem::temp_directory_path() / "sirius_factory_XXXXXX";
    _path     = tmpl.string();
    int fd    = ::mkstemp(_path.data());
    if (fd < 0)
      throw std::runtime_error("scoped_temp_file: mkstemp failed: " +
                               std::string(std::strerror(errno)));
    ::close(fd);
    std::ofstream os(_path, std::ios::binary);
    os.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  }

  ~scoped_temp_file()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
  }

  scoped_temp_file(scoped_temp_file const&)            = delete;
  scoped_temp_file& operator=(scoped_temp_file const&) = delete;

  [[nodiscard]] std::string const& path() const noexcept { return _path; }

 private:
  std::string _path;
};

}  // namespace

// ===========================================================================
// extract_scheme / extract_path
// ===========================================================================

TEST_CASE("datasource_factory::extract_scheme — basic forms", "[datasource_factory]")
{
  CHECK(datasource_factory::extract_scheme("/abs/path.parquet") == "file");
  CHECK(datasource_factory::extract_scheme("file:///abs/path.parquet") == "file");
  CHECK(datasource_factory::extract_scheme("s3://bucket/key") == "s3");
  CHECK(datasource_factory::extract_scheme("gs://bucket/key") == "gs");
  CHECK(datasource_factory::extract_scheme("rdma_s3://bucket/key") == "rdma_s3");
  // PR8 rejects relative bare paths; callers must pass absolute or scheme://.
  CHECK_THROWS_AS(datasource_factory::extract_scheme("relative/f.parquet"), std::invalid_argument);
  CHECK_THROWS_AS(datasource_factory::extract_scheme("file.parquet"), std::invalid_argument);
}

TEST_CASE("datasource_factory::extract_scheme — malformed URIs throw", "[datasource_factory]")
{
  CHECK_THROWS_AS(datasource_factory::extract_scheme(""), std::invalid_argument);
  CHECK_THROWS_AS(datasource_factory::extract_scheme("://nopath"), std::invalid_argument);
}

TEST_CASE("datasource_factory::extract_path — strips scheme delimiter", "[datasource_factory]")
{
  CHECK(datasource_factory::extract_path("/abs/path.parquet") == "/abs/path.parquet");
  CHECK(datasource_factory::extract_path("file:///abs/path.parquet") == "/abs/path.parquet");
  // PR8: host is split out of path; s3://bucket/key -> host="bucket", path="key".
  CHECK(datasource_factory::extract_path("s3://bucket/key") == "key");
  CHECK_THROWS_AS(datasource_factory::extract_path("relative/f.parquet"), std::invalid_argument);
  CHECK_THROWS_AS(datasource_factory::extract_path(""), std::invalid_argument);
}

// ===========================================================================
// datasource_registry
// ===========================================================================

TEST_CASE("datasource_registry — register, lookup, clear", "[datasource_factory]")
{
  datasource_registry reg;
  auto ctx = std::make_shared<mock_ioctx>();

  CHECK(reg.lookup("file") == nullptr);

  reg.register_ioctx("file", ctx);
  REQUIRE(reg.lookup("file") != nullptr);
  CHECK(reg.lookup("file").get() == ctx.get());

  // Re-registering replaces the old entry.
  auto ctx2 = std::make_shared<mock_ioctx>();
  reg.register_ioctx("file", ctx2);
  CHECK(reg.lookup("file").get() == ctx2.get());

  auto schemes = reg.schemes();
  REQUIRE(schemes.size() == 1);
  CHECK(schemes.front() == "file");

  reg.clear();
  CHECK(reg.lookup("file") == nullptr);
}

TEST_CASE("datasource_registry — rejects null ioctx / empty scheme", "[datasource_factory]")
{
  datasource_registry reg;
  CHECK_THROWS_AS(reg.register_ioctx("", std::make_shared<mock_ioctx>()), std::invalid_argument);
  CHECK_THROWS_AS(reg.register_ioctx("file", nullptr), std::invalid_argument);
}

TEST_CASE("datasource_registry — thread-safe for concurrent lookup", "[datasource_factory]")
{
  datasource_registry reg;
  auto ctx = std::make_shared<mock_ioctx>();
  reg.register_ioctx("file", ctx);

  constexpr int n_threads = 8;
  constexpr int n_iters   = 1000;
  std::barrier sync(n_threads);
  std::atomic<int> hits{0};

  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&] {
      sync.arrive_and_wait();
      for (int i = 0; i < n_iters; ++i) {
        if (reg.lookup("file")) hits.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& th : threads)
    th.join();
  CHECK(hits.load() == n_threads * n_iters);
}

// ===========================================================================
// datasource_factory::create — negative paths
// ===========================================================================

TEST_CASE("datasource_factory::create — empty URI rejected", "[datasource_factory]")
{
  datasource_registry reg;
  sirius_config cfg;
  CHECK_THROWS_AS(datasource_factory::create("", reg, cfg), std::invalid_argument);
}

TEST_CASE("datasource_factory::create — throws when object-store scheme unregistered",
          "[datasource_factory]")
{
  datasource_registry reg;
  sirius_config cfg;
  CHECK_THROWS_AS(datasource_factory::create("s3://bucket/key", reg, cfg), std::runtime_error);
}

TEST_CASE("datasource_factory::create — s3 scheme requires an s3_ioctx", "[datasource_factory]")
{
  // PR9: s3:// is wired, but construction still requires the registered ioctx
  // to actually be an s3::s3_ioctx (it must HEAD the object to cache size).
  // Registering a mock_ioctx lets us assert the factory refuses to proceed
  // with the wrong backend rather than silently HEADing through a mock.
  datasource_registry reg;
  sirius_config cfg;
  reg.register_ioctx("s3", std::make_shared<mock_ioctx>());
  try {
    (void)datasource_factory::create("s3://bucket/key", reg, cfg);
    FAIL("expected datasource_factory::create to throw for s3 + mock ioctx");
  } catch (std::runtime_error const& e) {
    std::string msg{e.what()};
    CHECK(msg.find("non-s3 ioctx") != std::string::npos);
  }
}

// ===========================================================================
// datasource_factory::create — happy path with cudf default datasource
// ===========================================================================

TEST_CASE("datasource_factory::create — local file uses cudf default datasource",
          "[datasource_factory]")
{
  constexpr std::string_view expected{"hello sirius"};
  scoped_temp_file tmp(expected);
  datasource_registry reg;
  sirius_config cfg;

  auto ds = datasource_factory::create(tmp.path(), reg, cfg);
  REQUIRE(ds != nullptr);
  CHECK(ds->size() == expected.size());

  auto buf = ds->host_read(0, expected.size());
  REQUIRE(buf != nullptr);
  REQUIRE(buf->size() == expected.size());
  CHECK(std::memcmp(buf->data(), expected.data(), expected.size()) == 0);

  // file:// form must produce an equivalent datasource.
  auto ds2 = datasource_factory::create("file://" + tmp.path(), reg, cfg);
  REQUIRE(ds2 != nullptr);
  CHECK(ds2->size() == ds->size());
}
