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

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <io/object_store_config.hpp>
#include <io/sirius_datasource.hpp>
#include <unistd.h>  // getpid
#include <utils/s3_container.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <thread>
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

std::string env_or(char const* name, std::string fallback = {})
{
  auto const* value = std::getenv(name);
  return value != nullptr ? std::string{value} : std::move(fallback);
}

std::string require_env(char const* name)
{
  auto value = env_or(name);
  REQUIRE_FALSE(value.empty());
  return value;
}

/// Credentials for the managed MinIO instance brought up by
/// ensure_s3_container_env(); TLS verification is off for its self-signed cert.
io::object_store_config minio_store()
{
  io::object_store_config os;
  os.endpoint   = require_env("SIRIUS_TEST_S3_ENDPOINT");
  os.region     = env_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  os.access_key = require_env("SIRIUS_TEST_S3_ACCESS_KEY");
  os.secret_key = require_env("SIRIUS_TEST_S3_SECRET_KEY");
  os.tls_verify = false;
  return os;
}

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

TEST_CASE("kvikio_context orders remote device reads behind the destination stream",
          "[s3][integration][kvikio]")
{
  using namespace std::chrono_literals;

  if (!sirius::test::ensure_s3_container_env()) { return; }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    WARN("Skipping kvikio remote stream-ordering test: no CUDA device");
    return;
  }

  std::vector<uint8_t> payload(256 * 1024);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<uint8_t>((i * 31 + 7) & 0xff);
  }
  std::string const key = "kvikio-stream-gate.bin";
  if (!sirius::test::put_s3_container_object(key, payload)) {
    WARN("Skipping kvikio remote stream-ordering test: S3 environment is externally managed");
    return;
  }

  auto ctx = std::make_shared<io::kvikio_context>(io::kvikio_config{}, minio_store());
  auto ds  = ctx->open_datasource("s3://" + require_env("SIRIUS_TEST_S3_BUCKET") + "/" + key);
  REQUIRE(ds != nullptr);
  REQUIRE(ds->size() == payload.size());

  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);

  // A host function that spins until released stands in for unfinished work the
  // caller already queued on `stream` — exactly the case where a write issued on
  // kvikIO's private stream would land in a buffer that is still in use.
  struct stream_gate {
    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
  } gate;
  REQUIRE(cudaLaunchHostFunc(
            stream.value(),
            [](void* opaque) {
              auto& state = *static_cast<stream_gate*>(opaque);
              state.entered.store(true, std::memory_order_release);
              while (!state.release.load(std::memory_order_acquire)) {
                std::this_thread::yield();
              }
            },
            &gate) == cudaSuccess);

  auto const gate_deadline = std::chrono::steady_clock::now() + 5s;
  while (!gate.entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < gate_deadline) {
    std::this_thread::yield();
  }
  REQUIRE(gate.entered.load(std::memory_order_acquire));

  // mixed_readv_async_io runs eagerly on the calling thread, so the read has to
  // be issued off-thread for "not yet complete" to be observable at all.
  auto read = std::async(std::launch::async, [&] {
    return std::move(ctx->device_read_async_io(ds->get_io_object(),
                                               0,
                                               payload.size(),
                                               static_cast<uint8_t*>(destination.data()),
                                               stream.value()))
      .get(120s);
  });

  REQUIRE(read.wait_for(200ms) == std::future_status::timeout);

  gate.release.store(true, std::memory_order_release);
  REQUIRE(read.wait_for(120s) == std::future_status::ready);
  REQUIRE(read.get() == payload.size());

  std::vector<uint8_t> got(payload.size(), 0);
  REQUIRE(
    cudaMemcpyAsync(
      got.data(), destination.data(), payload.size(), cudaMemcpyDeviceToHost, stream.value()) ==
    cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
  REQUIRE(got == payload);
}
