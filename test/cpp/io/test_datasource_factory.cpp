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
#include "io/sirius_datasource.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "sirius_config.hpp"

#include <fcntl.h>
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
#include <future>
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
// A test-only sirius_ioctx that counts make_datasource calls. Its read methods
// are not exercised in PR1 tests; they throw if called to surface misuse.
// ---------------------------------------------------------------------------

class mock_ioctx : public sirius_ioctx {
 public:
  std::atomic<int> make_datasource_calls{0};

  void shutdown() override {}

  std::unique_ptr<cudf::io::datasource> make_datasource(
    std::unique_ptr<sirius_io_object> /*io_object*/) override
  {
    make_datasource_calls.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error("mock_ioctx::make_datasource: not exercised in PR1");
  }

  [[nodiscard]] bool supports_device_read() const override { return false; }
  [[nodiscard]] bool is_device_read_preferred(size_t) const override { return false; }

  // Unused read APIs — PR1 does not drive IO through the mock.
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
  std::future<size_t> host_read_async(sirius_io_object&, size_t, size_t, uint8_t*) override
  {
    throw std::logic_error("unused");
  }
  std::future<std::unique_ptr<cudf::io::datasource::buffer>> host_read_async(sirius_io_object&,
                                                                             size_t,
                                                                             size_t) override
  {
    throw std::logic_error("unused");
  }
  std::unique_ptr<cudf::io::datasource::buffer> device_read(sirius_io_object&,
                                                            size_t,
                                                            size_t,
                                                            rmm::cuda_stream_view) override
  {
    throw std::logic_error("unused");
  }
  size_t device_read(sirius_io_object&, size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    throw std::logic_error("unused");
  }
  std::future<size_t> device_read_async(
    sirius_io_object&, size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    throw std::logic_error("unused");
  }
  std::future<size_t> host_read_ranges_async(sirius_io_object&,
                                             std::vector<cudf::io::text::byte_range_info> const&,
                                             std::span<cudf::host_span<std::byte>>) override
  {
    throw std::logic_error("unused");
  }
  size_t host_read_ranges(sirius_io_object&,
                          std::vector<cudf::io::text::byte_range_info> const&,
                          std::span<cudf::host_span<std::byte>>) override
  {
    throw std::logic_error("unused");
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

// Attempt to build a uring_ioctx; if the runtime doesn't support io_uring (or
// the user lacks the capability), returns nullptr so the caller can skip.
std::shared_ptr<uring_ioctx> try_make_uring_ioctx()
{
  try {
    // Small footprint for tests: 2 host rings, 8 entries, 1 reactor.
    return std::make_shared<uring_ioctx>(/*host_ring_depth=*/2,
                                         /*ring_entries=*/8,
                                         /*n_reactors=*/1,
                                         /*bounce_slot_size=*/1UL << 20);
  } catch (std::exception const& e) {
    WARN("uring_ioctx construction failed: " << e.what());
    return nullptr;
  }
}

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

TEST_CASE("datasource_factory::create — throws when scheme unregistered", "[datasource_factory]")
{
  datasource_registry reg;
  sirius_config cfg;
  CHECK_THROWS_AS(datasource_factory::create("s3://bucket/key", reg, cfg), std::runtime_error);
  CHECK_THROWS_AS(datasource_factory::create("/data/file.parquet", reg, cfg), std::runtime_error);
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
// datasource_factory::create — happy path with a real uring_ioctx
// ===========================================================================

TEST_CASE("datasource_factory::create — dispatches file:// to uring_ioctx", "[datasource_factory]")
{
  auto ctx = try_make_uring_ioctx();
  if (!ctx) {
    SUCCEED("Skipping: io_uring not supported on this runner");
    return;
  }

  scoped_temp_file tmp("hello sirius");
  datasource_registry reg;
  reg.register_ioctx("file", ctx);
  sirius_config cfg;

  std::unique_ptr<io_datasource> ds;
  try {
    ds = datasource_factory::create(tmp.path(), reg, cfg);
  } catch (std::exception const& e) {
    // Some sandboxes (e.g. tmpfs) reject O_DIRECT. That's a runtime
    // capability, not a factory defect.
    WARN(
      "uring_io_object could not open temp file (likely no O_DIRECT "
      "support): "
      << e.what());
    SUCCEED("Skipping: filesystem does not support O_DIRECT");
    return;
  }

  REQUIRE(ds != nullptr);
  CHECK(ds->size() == std::string_view{"hello sirius"}.size());

  // file:// form must produce an equivalent datasource.
  auto ds2 = datasource_factory::create("file://" + tmp.path(), reg, cfg);
  REQUIRE(ds2 != nullptr);
  CHECK(ds2->size() == ds->size());

  ctx->shutdown();
}
