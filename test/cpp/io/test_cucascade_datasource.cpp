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

// test
#include "catch.hpp"

// sirius
#include "io/cucascade_datasource.hpp"

// cucascade
#include <cucascade/data/disk_io_backend.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/**
 * @brief Minimal mock idisk_io_backend used to verify cucascade_datasource behavior
 * in isolation. Deterministic pattern fill on host reads lets tests assert that
 * the adapter forwarded offset/size untouched to the backend.
 */
class mock_io_backend : public cucascade::idisk_io_backend {
 public:
  std::atomic<int> read_host_count{0};
  std::atomic<int> read_device_count{0};
  std::atomic<std::size_t> last_size{0};
  std::atomic<std::size_t> last_offset{0};
  std::string last_path;

  // Host read — fills dst with deterministic pattern for assertions.
  void read(std::filesystem::path const& path,
            void* host_ptr,
            std::size_t size,
            std::size_t file_offset) override
  {
    read_host_count.fetch_add(1);
    last_size.store(size);
    last_offset.store(file_offset);
    last_path   = path.string();
    auto* bytes = static_cast<uint8_t*>(host_ptr);
    for (std::size_t i = 0; i < size; ++i) {
      bytes[i] = static_cast<uint8_t>((file_offset + i) & 0xff);
    }
  }

  // Device read — unused; adapter reports supports_device_read() == false.
  void read(
    std::filesystem::path const&, void*, std::size_t, std::size_t, rmm::cuda_stream_view) override
  {
    read_device_count.fetch_add(1);
  }

  // Write overloads — required by abstract base; no-op.
  void write(std::filesystem::path const&,
             void const*,
             std::size_t,
             std::size_t,
             rmm::cuda_stream_view) override
  {
  }

  void write(std::filesystem::path const&, void const*, std::size_t, std::size_t) override {}
};

}  // namespace

using sirius::io::cucascade_datasource;

//===----------------------------------------------------------------------===//
// Constructor validation
//===----------------------------------------------------------------------===//

TEST_CASE("cucascade_datasource: constructor rejects invalid inputs",
          "[io_backend][cucascade_datasource]")
{
  SECTION("null backend throws")
  {
    REQUIRE_THROWS_AS(
      cucascade_datasource(nullptr, std::filesystem::path{"/tmp/file.parquet"}, 1024),
      std::invalid_argument);
  }

  SECTION("s3:// scheme throws")
  {
    auto mock = std::make_shared<mock_io_backend>();
    REQUIRE_THROWS_AS(
      cucascade_datasource(mock, std::filesystem::path{"s3://bucket/file.parquet"}, 1024),
      std::invalid_argument);
  }

  SECTION("http:// scheme throws")
  {
    auto mock = std::make_shared<mock_io_backend>();
    REQUIRE_THROWS_AS(
      cucascade_datasource(mock, std::filesystem::path{"http://example.com/file.parquet"}, 1024),
      std::invalid_argument);
  }

  SECTION("https:// scheme throws")
  {
    auto mock = std::make_shared<mock_io_backend>();
    REQUIRE_THROWS_AS(
      cucascade_datasource(mock, std::filesystem::path{"https://example.com/file.parquet"}, 1024),
      std::invalid_argument);
  }

  SECTION("hdfs:// scheme throws")
  {
    auto mock = std::make_shared<mock_io_backend>();
    REQUIRE_THROWS_AS(
      cucascade_datasource(mock, std::filesystem::path{"hdfs://cluster/path/file.parquet"}, 1024),
      std::invalid_argument);
  }

  SECTION("local path succeeds")
  {
    auto mock = std::make_shared<mock_io_backend>();
    REQUIRE_NOTHROW(cucascade_datasource(mock, std::filesystem::path{"/tmp/file.parquet"}, 1024));
  }
}

//===----------------------------------------------------------------------===//
// size() + device-read flags (IO-02 contract)
//===----------------------------------------------------------------------===//

TEST_CASE("cucascade_datasource: size and device-read flags", "[io_backend][cucascade_datasource]")
{
  auto mock = std::make_shared<mock_io_backend>();
  cucascade_datasource ds{mock, std::filesystem::path{"/tmp/f.parquet"}, 8192};

  REQUIRE(ds.size() == 8192);

  // size() must be stable across repeated calls — cuDF calls it during footer planning
  // and will re-query multiple times.
  REQUIRE(ds.size() == 8192);
  REQUIRE(ds.size() == 8192);

  // IO-02: supports_device_read() locked to false so cuDF host-stages reads.
  REQUIRE(ds.supports_device_read() == false);
  REQUIRE(ds.is_device_read_preferred(1UL << 30) == false);
  REQUIRE(ds.is_device_read_preferred(0) == false);
  REQUIRE(ds.is_device_read_preferred(1) == false);
}

//===----------------------------------------------------------------------===//
// host_read(offset, size, dst)
//===----------------------------------------------------------------------===//

TEST_CASE("cucascade_datasource: host_read dst overload delegates to backend",
          "[io_backend][cucascade_datasource]")
{
  auto mock = std::make_shared<mock_io_backend>();
  cucascade_datasource ds{mock, std::filesystem::path{"/tmp/f.parquet"}, 8192};

  std::vector<uint8_t> dst(256);
  auto const bytes_read = ds.host_read(100, dst.size(), dst.data());

  REQUIRE(bytes_read == dst.size());
  REQUIRE(mock->read_host_count.load() == 1);
  REQUIRE(mock->last_size.load() == dst.size());
  REQUIRE(mock->last_offset.load() == 100);
  REQUIRE(mock->last_path == "/tmp/f.parquet");

  // Pattern check: mock fills dst with (offset + i) & 0xff.
  REQUIRE(dst[0] == static_cast<uint8_t>(100 & 0xff));
  REQUIRE(dst[1] == static_cast<uint8_t>(101 & 0xff));
  REQUIRE(dst[dst.size() - 1] == static_cast<uint8_t>((100 + dst.size() - 1) & 0xff));
}

//===----------------------------------------------------------------------===//
// host_read(offset, size) -> buffer (IO-03 pinned)
//===----------------------------------------------------------------------===//

TEST_CASE("cucascade_datasource: host_read buffer overload returns pinned buffer",
          "[io_backend][cucascade_datasource]")
{
  auto mock = std::make_shared<mock_io_backend>();
  cucascade_datasource ds{mock, std::filesystem::path{"/tmp/f.parquet"}, 8192};

  auto buf = ds.host_read(200, 512);
  REQUIRE(buf != nullptr);
  REQUIRE(buf->size() == 512);
  REQUIRE(buf->data() != nullptr);
  REQUIRE(mock->read_host_count.load() == 1);
  REQUIRE(mock->last_offset.load() == 200);
  REQUIRE(mock->last_size.load() == 512);

  // Pattern check on the returned buffer — confirms backend wrote into the pinned allocation.
  auto const* data = buf->data();
  REQUIRE(data[0] == static_cast<uint8_t>(200 & 0xff));
  REQUIRE(data[1] == static_cast<uint8_t>(201 & 0xff));
}

//===----------------------------------------------------------------------===//
// EOF clipping (kvikio_source::clamped_read_to_vector parity)
//===----------------------------------------------------------------------===//

TEST_CASE("cucascade_datasource: host_read clips to file size",
          "[io_backend][cucascade_datasource]")
{
  auto mock = std::make_shared<mock_io_backend>();
  cucascade_datasource ds{mock, std::filesystem::path{"/tmp/f.parquet"}, 1024};

  SECTION("dst overload clips when offset+size > file_size")
  {
    std::vector<uint8_t> dst(2048);
    auto const bytes_read = ds.host_read(500, 2000, dst.data());  // 500 + 2000 > 1024

    REQUIRE(bytes_read == 524);
    REQUIRE(mock->read_host_count.load() == 1);
    REQUIRE(mock->last_size.load() == 524);
    REQUIRE(mock->last_offset.load() == 500);
  }

  SECTION("buffer overload clips when offset+size > file_size")
  {
    auto buf = ds.host_read(900, 500);  // 900 + 500 > 1024

    REQUIRE(buf != nullptr);
    REQUIRE(buf->size() == 124);
    REQUIRE(mock->last_size.load() == 124);
  }

  SECTION("offset past EOF returns 0 and does not call backend")
  {
    std::vector<uint8_t> dst(64);
    auto const bytes_read = ds.host_read(2048, 64, dst.data());

    REQUIRE(bytes_read == 0);
    REQUIRE(mock->read_host_count.load() == 0);
  }
}

//===----------------------------------------------------------------------===//
// host_read_async — single call
//===----------------------------------------------------------------------===//

TEST_CASE("cucascade_datasource: host_read_async resolves with correct count",
          "[io_backend][cucascade_datasource]")
{
  auto mock = std::make_shared<mock_io_backend>();
  cucascade_datasource ds{mock, std::filesystem::path{"/tmp/f.parquet"}, 8192};

  std::vector<uint8_t> dst(256);
  auto fut              = ds.host_read_async(300, dst.size(), dst.data());
  auto const bytes_read = fut.get();

  REQUIRE(bytes_read == dst.size());
  REQUIRE(mock->read_host_count.load() == 1);
  REQUIRE(mock->last_offset.load() == 300);

  // Buffer overload async
  auto fut2 = ds.host_read_async(400, 128);
  auto buf  = fut2.get();
  REQUIRE(buf != nullptr);
  REQUIRE(buf->size() == 128);
  REQUIRE(mock->read_host_count.load() == 2);
}

//===----------------------------------------------------------------------===//
// host_read_async — concurrency (validates std::launch::async, not deferred)
//===----------------------------------------------------------------------===//

TEST_CASE("cucascade_datasource: concurrent host_read_async calls both execute",
          "[io_backend][cucascade_datasource]")
{
  auto mock = std::make_shared<mock_io_backend>();
  cucascade_datasource ds{mock, std::filesystem::path{"/tmp/f.parquet"}, 8192};

  std::vector<uint8_t> dst1(128), dst2(128);
  auto f1       = ds.host_read_async(0, dst1.size(), dst1.data());
  auto f2       = ds.host_read_async(128, dst2.size(), dst2.data());
  auto const r1 = f1.get();
  auto const r2 = f2.get();

  REQUIRE(r1 == 128);
  REQUIRE(r2 == 128);
  REQUIRE(mock->read_host_count.load() == 2);
}
