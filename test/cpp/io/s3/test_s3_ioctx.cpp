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
#include "exec/thread_pool.hpp"
#include "io/io_errors.hpp"
#include "io/prefetching_cache.hpp"
#include "io/s3/credential_provider.hpp"
#include "io/s3/mock_credential_provider.hpp"
#include "io/s3/s3_io_object.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/s3/sirius_sigv4_credential_provider.hpp"

#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <arpa/inet.h>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using sirius::io::buffer_pool;
using sirius::io::credential_error;
using sirius::io::s3::credential_provider;
using sirius::io::s3::mock_credential_provider;
using sirius::io::s3::presign_method;
using sirius::io::s3::s3_io_object;
using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;
using sirius::io::s3::s3_object_ref;
using sirius::io::s3::sirius_sigv4_credential_provider;
using sirius::io::s3::static_credentials;

namespace {

namespace fs = std::filesystem;
using namespace std::chrono_literals;

std::string env_or(std::string_view name, std::string fallback = {})
{
  auto const* value = std::getenv(std::string{name}.c_str());
  return value ? std::string{value} : std::move(fallback);
}

bool truthy_env(std::string_view name)
{
  auto value = env_or(name);
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
}

struct s3_test_env {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  fs::path local_dir;
  bool strict{false};
};

std::optional<s3_test_env> read_s3_test_env()
{
  auto endpoint   = env_or("SIRIUS_TEST_S3_ENDPOINT");
  auto access_key = env_or("SIRIUS_TEST_S3_ACCESS_KEY");
  auto secret_key = env_or("SIRIUS_TEST_S3_SECRET_KEY");
  auto bucket     = env_or("SIRIUS_TEST_S3_BUCKET");
  auto local_dir  = env_or("SIRIUS_TEST_S3_LOCAL_DIR");

  if (endpoint.empty() || access_key.empty() || secret_key.empty() || bucket.empty() ||
      local_dir.empty()) {
    return std::nullopt;
  }

  return s3_test_env{std::move(endpoint),
                     env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
                     std::move(access_key),
                     std::move(secret_key),
                     std::move(bucket),
                     fs::path{local_dir},
                     truthy_env("SIRIUS_TEST_S3_STRICT")};
}

std::vector<std::uint8_t> read_binary_file(fs::path const& path)
{
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string s3_uri(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string{bucket} + "/" + std::string{key};
}

std::shared_ptr<s3_io_object> make_s3_object(std::string bucket, std::string key, std::size_t size)
{
  auto path = s3_uri(bucket, key);
  return std::make_shared<s3_io_object>(std::move(bucket), std::move(key), size, std::move(path));
}

std::shared_ptr<credential_provider> make_live_provider(s3_test_env const& env)
{
  static_credentials creds;
  creds.access_key_id     = env.access_key;
  creds.secret_access_key = env.secret_key;
  return std::make_shared<sirius_sigv4_credential_provider>(
    std::move(creds), env.region, env.endpoint, 30min);
}

std::shared_ptr<s3_ioctx> make_live_ioctx(s3_test_env const& env)
{
  auto provider = make_live_provider(env);
  return std::make_shared<s3_ioctx>(s3_ioctx_config{std::move(provider), 4, 20});
}

std::shared_ptr<s3_ioctx> make_live_ioctx_with_fsmr(
  std::shared_ptr<credential_provider> provider,
  cucascade::memory::fixed_size_host_memory_resource& host_mr)
{
  s3_ioctx_config cfg{};
  cfg.creds                = std::move(provider);
  cfg.max_connections      = 16;
  cfg.request_timeout_s    = 20;
  cfg.host_memory_resource = &host_mr;
  return std::make_shared<s3_ioctx>(std::move(cfg));
}

std::optional<std::size_t> try_head_or_skip(s3_ioctx& ctx,
                                            s3_test_env const& env,
                                            std::string_view key)
{
  try {
    return ctx.head_object_size(env.bucket, key);
  } catch (std::exception const& e) {
    if (env.strict) { FAIL("S3 test environment is configured but HEAD failed: " << e.what()); }
    WARN("Skipping live S3 test because HEAD failed: " << e.what());
    return std::nullopt;
  }
}

void require_bytes_equal(std::vector<std::byte> const& got,
                         std::vector<std::uint8_t> const& expected,
                         std::size_t offset)
{
  REQUIRE(offset + got.size() <= expected.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    CHECK(static_cast<std::uint8_t>(got[i]) == expected[offset + i]);
  }
}

void require_bytes_equal(std::vector<std::uint8_t> const& got,
                         std::vector<std::uint8_t> const& expected,
                         std::size_t offset)
{
  REQUIRE(offset + got.size() <= expected.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == expected[offset + i]);
  }
}

void require_cuda_success(cudaError_t status)
{
  INFO(cudaGetErrorString(status));
  REQUIRE(status == cudaSuccess);
}

class device_byte_buffer {
 public:
  explicit device_byte_buffer(std::size_t size) : _size(size)
  {
    if (_size > 0) {
      void* p = nullptr;
      require_cuda_success(cudaMalloc(&p, _size));
      _data = static_cast<std::uint8_t*>(p);
    }
  }

  ~device_byte_buffer()
  {
    if (_data != nullptr) { cudaFree(_data); }
  }

  device_byte_buffer(device_byte_buffer const&)            = delete;
  device_byte_buffer& operator=(device_byte_buffer const&) = delete;

  [[nodiscard]] std::uint8_t* data() noexcept { return _data; }
  [[nodiscard]] std::uint8_t const* data() const noexcept { return _data; }
  [[nodiscard]] std::size_t size() const noexcept { return _size; }

 private:
  std::uint8_t* _data{nullptr};
  std::size_t _size{0};
};

std::vector<std::uint8_t> copy_device_to_host(device_byte_buffer const& device, std::size_t bytes)
{
  REQUIRE(bytes <= device.size());
  std::vector<std::uint8_t> host(bytes);
  if (bytes > 0) {
    require_cuda_success(cudaMemcpy(host.data(), device.data(), bytes, cudaMemcpyDeviceToHost));
  }
  return host;
}

struct fsmr_test_resources {
  fsmr_test_resources(std::size_t block_size,
                      std::size_t capacity,
                      std::size_t memory_limit,
                      std::size_t pool_size     = 1,
                      std::size_t initial_pools = 0)
    : upstream(0, true),
      host_mr(0, upstream, memory_limit, capacity, block_size, pool_size, initial_pools)
  {
  }

  cucascade::memory::numa_region_pinned_host_memory_resource upstream;
  cucascade::memory::fixed_size_host_memory_resource host_mr;
};

using async_read_result = std::pair<std::size_t, std::exception_ptr>;

async_read_result read_ranges_async(s3_ioctx& ctx,
                                    s3_io_object& obj,
                                    std::vector<cudf::io::text::byte_range_info> const& ranges,
                                    std::span<cudf::host_span<std::byte>> dst)
{
  std::promise<async_read_result> done;
  auto fut = done.get_future();

  ctx.host_read_ranges_async_io(
    obj, ranges, dst, [&done](auto bytes, auto ep) { done.set_value({bytes, ep}); });

  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  return fut.get();
}

class blocking_throwing_provider final : public credential_provider {
 public:
  blocking_throwing_provider()
    : _entered(_entered_promise.get_future().share()),
      _release(_release_promise.get_future().share())
  {
  }

  std::string get_presigned_url(s3_object_ref const&, presign_method method) override
  {
    _method.store(method == presign_method::GET ? 1 : 2, std::memory_order_relaxed);
    std::call_once(_entered_once, [&] { _entered_promise.set_value(); });
    _release.wait();
    throw credential_error("blocking_throwing_provider: forced failure");
  }

  bool wait_until_entered(std::chrono::milliseconds timeout) const
  {
    return _entered.wait_for(timeout) == std::future_status::ready;
  }

  void release()
  {
    std::call_once(_release_once, [&] { _release_promise.set_value(); });
  }

  presign_method method() const
  {
    return _method.load(std::memory_order_relaxed) == 1 ? presign_method::GET
                                                        : presign_method::HEAD;
  }

 private:
  std::promise<void> _entered_promise;
  std::shared_future<void> _entered;
  std::promise<void> _release_promise;
  std::shared_future<void> _release;
  mutable std::once_flag _entered_once;
  std::once_flag _release_once;
  std::atomic<int> _method{1};
};

class blocking_first_get_provider final : public credential_provider {
 public:
  explicit blocking_first_get_provider(std::shared_ptr<credential_provider> delegate)
    : _delegate(std::move(delegate)),
      _entered(_entered_promise.get_future().share()),
      _release(_release_promise.get_future().share())
  {
  }

  std::string get_presigned_url(s3_object_ref const& obj, presign_method method) override
  {
    if (method == presign_method::GET && !_blocked_first_get.exchange(true)) {
      std::call_once(_entered_once, [&] { _entered_promise.set_value(); });
      _release.wait();
    }
    return _delegate->get_presigned_url(obj, method);
  }

  bool wait_until_first_get_entered(std::chrono::milliseconds timeout) const
  {
    return _entered.wait_for(timeout) == std::future_status::ready;
  }

  void release()
  {
    std::call_once(_release_once, [&] { _release_promise.set_value(); });
  }

 private:
  std::shared_ptr<credential_provider> _delegate;
  std::promise<void> _entered_promise;
  std::shared_future<void> _entered;
  std::promise<void> _release_promise;
  std::shared_future<void> _release;
  mutable std::once_flag _entered_once;
  std::once_flag _release_once;
  std::atomic<bool> _blocked_first_get{false};
};

std::string current_thread_name()
{
  std::array<char, 16> name{};
  if (pthread_getname_np(pthread_self(), name.data(), name.size()) != 0) { return {}; }
  return std::string{name.data()};
}

struct async_observation {
  std::size_t bytes{0};
  std::exception_ptr error;
  std::string thread_name;
};

template <typename Launch>
async_observation observe_async_completion(Launch&& launch)
{
  auto done = std::make_shared<std::promise<async_observation>>();
  auto fut  = done->get_future();
  launch([done](std::size_t bytes, std::exception_ptr ep) {
    done->set_value(async_observation{bytes, ep, current_thread_name()});
  });
  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  return fut.get();
}

struct scripted_http_response {
  long status{200};
  std::string reason{"OK"};
  std::vector<std::string> headers;
  std::string body;
  bool close_without_response{false};
};

std::string serialize(scripted_http_response const& response)
{
  std::ostringstream out;
  out << "HTTP/1.1 " << response.status << " " << response.reason << "\r\n";

  bool has_content_length = false;
  bool has_connection     = false;
  for (auto const& header : response.headers) {
    has_content_length = has_content_length || header.rfind("Content-Length:", 0) == 0;
    has_connection     = has_connection || header.rfind("Connection:", 0) == 0;
    out << header << "\r\n";
  }
  if (!has_content_length) { out << "Content-Length: " << response.body.size() << "\r\n"; }
  if (!has_connection) { out << "Connection: close\r\n"; }
  out << "\r\n" << response.body;
  return out.str();
}

void send_all(int fd, std::string const& payload)
{
  auto const* data      = payload.data();
  std::size_t remaining = payload.size();
  while (remaining > 0) {
    auto sent = ::send(fd, data, remaining, 0);
    if (sent <= 0) { return; }
    data += sent;
    remaining -= static_cast<std::size_t>(sent);
  }
}

class scripted_http_server {
 public:
  explicit scripted_http_server(std::vector<scripted_http_response> responses)
    : _responses(std::move(responses))
  {
    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(_listen_fd >= 0);

    int opt = 1;
    REQUIRE(::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == 0);

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;

    REQUIRE(::bind(_listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    REQUIRE(::listen(_listen_fd, 16) == 0);

    sockaddr_in bound{};
    socklen_t len = sizeof(bound);
    REQUIRE(::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&bound), &len) == 0);
    _port = ntohs(bound.sin_port);

    _thread = std::thread([this] { serve(); });
  }

  ~scripted_http_server()
  {
    _stop.store(true, std::memory_order_relaxed);
    if (_listen_fd >= 0) {
      ::shutdown(_listen_fd, SHUT_RDWR);
      ::close(_listen_fd);
      _listen_fd = -1;
    }
    if (_thread.joinable()) { _thread.join(); }
  }

  std::string url(std::string_view path = "/object") const
  {
    return "http://127.0.0.1:" + std::to_string(_port) + std::string{path};
  }

  std::size_t request_count() const noexcept { return _request_count.load(); }

 private:
  void serve()
  {
    while (!_stop.load(std::memory_order_relaxed)) {
      sockaddr_in client_addr{};
      socklen_t client_len = sizeof(client_addr);
      int client = ::accept(_listen_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
      if (client < 0) { return; }

      ++_request_count;
      read_request(client);

      auto const index = _next_response.fetch_add(1);
      scripted_http_response response;
      if (index < _responses.size()) {
        response = _responses[index];
      } else if (!_responses.empty()) {
        response = _responses.back();
      }

      if (!response.close_without_response) { send_all(client, serialize(response)); }
      ::shutdown(client, SHUT_RDWR);
      ::close(client);
    }
  }

  void read_request(int client)
  {
    std::string request;
    std::array<char, 1024> buffer{};
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 8192) {
      auto got = ::recv(client, buffer.data(), buffer.size(), 0);
      if (got <= 0) { break; }
      request.append(buffer.data(), static_cast<std::size_t>(got));
    }
  }

  std::vector<scripted_http_response> _responses;
  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::atomic<std::size_t> _next_response{0};
  std::atomic<std::size_t> _request_count{0};
  std::atomic<bool> _stop{false};
  std::thread _thread;
};

scripted_http_response http_error(long status, std::string reason, std::string body = {})
{
  return scripted_http_response{status, std::move(reason), {}, std::move(body)};
}

scripted_http_response range_ok(std::string body, std::size_t object_size)
{
  auto const end = body.empty() ? 0 : body.size() - 1;
  return scripted_http_response{
    206,
    "Partial Content",
    {"Content-Range: bytes 0-" + std::to_string(end) + "/" + std::to_string(object_size)},
    std::move(body)};
}

scripted_http_response head_ok(std::size_t object_size)
{
  return scripted_http_response{200, "OK", {"Content-Length: " + std::to_string(object_size)}, ""};
}

std::shared_ptr<s3_ioctx> make_retry_ioctx(std::string url,
                                           std::size_t attempts,
                                           std::chrono::milliseconds base   = 0ms,
                                           std::chrono::milliseconds jitter = 0ms,
                                           bool honor_retry_after           = false)
{
  auto provider = std::make_shared<mock_credential_provider>(std::move(url));
  s3_ioctx_config cfg{provider, 1, 1};
  cfg.max_retry_attempts = attempts;
  cfg.retry_backoff_base = base;
  cfg.retry_jitter       = jitter;
  cfg.honor_retry_after  = honor_retry_after;
  return std::make_shared<s3_ioctx>(std::move(cfg));
}

class delayed_get_provider final : public credential_provider {
 public:
  delayed_get_provider(std::shared_ptr<credential_provider> delegate,
                       std::chrono::milliseconds delay)
    : _delegate(std::move(delegate)), _delay(delay)
  {
  }

  std::string get_presigned_url(s3_object_ref const& obj, presign_method method) override
  {
    if (method == presign_method::GET) {
      _get_count.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::sleep_for(_delay);
    }
    return _delegate->get_presigned_url(obj, method);
  }

  [[nodiscard]] int get_count() const noexcept
  {
    return _get_count.load(std::memory_order_relaxed);
  }

 private:
  std::shared_ptr<credential_provider> _delegate;
  std::chrono::milliseconds _delay;
  std::atomic<int> _get_count{0};
};

std::size_t cache_capacity_bytes(std::size_t block_size, std::uint32_t max_slabs)
{
  return block_size * static_cast<std::size_t>(buffer_pool::CHUNKS_PER_SLAB) *
         static_cast<std::size_t>(max_slabs);
}

struct cache_test_resources {
  static constexpr std::uint32_t max_slabs = 1;
  static constexpr std::size_t block_size  = 4096;

  cache_test_resources()
    : upstream(0, true),
      host_mr(0,
              upstream,
              cache_capacity_bytes(block_size, max_slabs),
              cache_capacity_bytes(block_size, max_slabs),
              block_size,
              static_cast<std::size_t>(buffer_pool::CHUNKS_PER_SLAB),
              1),
      pool(host_mr, max_slabs)
  {
  }

  cucascade::memory::numa_region_pinned_host_memory_resource upstream;
  cucascade::memory::fixed_size_host_memory_resource host_mr;
  buffer_pool pool;
};

bool wait_until_cached(s3_ioctx& ctx,
                       s3_io_object& obj,
                       cudf::io::text::byte_range_info range,
                       std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (auto view = ctx.cache()->read(obj,
                                      static_cast<std::size_t>(range.offset()),
                                      static_cast<std::size_t>(range.size()),
                                      nullptr);
        view) {
      return true;
    }
    std::this_thread::sleep_for(10ms);
  }
  return false;
}

}  // namespace

TEST_CASE("s3_io_object preserves S3 identity and cache id", "[s3][ioctx]")
{
  s3_io_object obj("bucket", "/path/to/object.parquet", 123, "s3://bucket//path/to/object.parquet");

  CHECK(obj.bucket() == "bucket");
  CHECK(obj.key() == "/path/to/object.parquet");
  CHECK(obj.size() == 123);
  CHECK(obj.raw_file_cache_id() == "s3://bucket//path/to/object.parquet");
}

TEST_CASE("s3_io_object object_path returns the constructor path", "[s3][io_object]")
{
  s3_io_object obj("bucket", "key", 1024, "s3://bucket/key");

  CHECK(obj.object_path() == "s3://bucket/key");
  CHECK(obj.raw_file_cache_id() == "s3://bucket/key");
}

TEST_CASE("s3_ioctx supports accepts S3 URIs without presigning", "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  s3_ioctx ctx{s3_ioctx_config{provider, 1, 1}};

  CHECK(ctx.supports("s3://bucket/key"));
  CHECK(provider->call_count() == 0);
}

TEST_CASE("s3_ioctx supports rejects non-S3 paths without throwing", "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  s3_ioctx ctx{s3_ioctx_config{provider, 1, 1}};

  CHECK_FALSE(ctx.supports("file:///tmp/foo"));
  CHECK_FALSE(ctx.supports("/abs/path"));
  CHECK_FALSE(ctx.supports("https://example.com"));
  CHECK(provider->call_count() == 0);
}

TEST_CASE("s3_ioctx supports treats the S3 scheme case-insensitively", "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  s3_ioctx ctx{s3_ioctx_config{provider, 1, 1}};

  CHECK(ctx.supports("S3://bucket/key"));
  CHECK(ctx.supports("s3://bucket/key"));
  CHECK(provider->call_count() == 0);
}

TEST_CASE("s3_ioctx create_io_object rejects non-S3 paths before presigning", "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  s3_ioctx ctx{s3_ioctx_config{provider, 1, 1}};

  try {
    (void)ctx.create_io_object("file:///foo");
    FAIL("create_io_object accepted a non-S3 path");
  } catch (std::invalid_argument const& e) {
    CHECK(std::string_view{e.what()}.find("unsupported") != std::string_view::npos);
  }
  CHECK(provider->call_count() == 0);
}

TEST_CASE("s3_ioctx validates config and clips EOF host reads before presigning", "[s3][ioctx]")
{
  CHECK_THROWS_AS(s3_ioctx(s3_ioctx_config{}), std::invalid_argument);

  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  auto ctx      = std::make_shared<s3_ioctx>(s3_ioctx_config{provider, 1, 1});
  auto obj      = make_s3_object("bucket", "key", 8);
  std::vector<std::uint8_t> dst(4, 0xAB);

  CHECK(ctx->host_read(*obj, 8, dst.size(), dst.data()) == 0);
  CHECK(ctx->host_read(*obj, 20, dst.size(), dst.data()) == 0);
  CHECK(provider->call_count() == 0);
  CHECK(std::all_of(dst.begin(), dst.end(), [](auto b) { return b == 0xAB; }));
}

TEST_CASE("s3_ioctx config exposes async pool and retry defaults", "[s3][ioctx]")
{
  s3_ioctx_config cfg{};

  CHECK(cfg.async_thread_pool == nullptr);
  CHECK(cfg.max_retry_attempts == 3);
  CHECK(cfg.retry_backoff_base == 100ms);
  CHECK(cfg.retry_jitter == 50ms);
  CHECK(cfg.honor_retry_after);
}

TEST_CASE("s3_ioctx asks credential_provider for method-specific presigned URLs", "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  provider->set_throw("stop before libcurl performs a request");
  auto ctx = std::make_shared<s3_ioctx>(s3_ioctx_config{provider, 1, 1});
  auto obj = make_s3_object("bucket", "key", 8);
  std::vector<std::uint8_t> dst(4);

  CHECK_THROWS_AS(ctx->head_object_size("bucket", "key"), credential_error);
  CHECK(provider->head_count() == 1);
  CHECK(provider->last_bucket() == "bucket");
  CHECK(provider->last_key() == "key");

  CHECK_THROWS_AS(ctx->host_read(*obj, 0, dst.size(), dst.data()), credential_error);
  CHECK(provider->get_count() == 1);
  CHECK(provider->last_bucket() == "bucket");
  CHECK(provider->last_key() == "key");
}

TEST_CASE("s3_ioctx clips physical byte ranges to file size", "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  auto ctx      = std::make_shared<s3_ioctx>(s3_ioctx_config{provider, 1, 1});

  auto clipped = ctx->compute_physical_range({4, 16}, 10);
  CHECK(clipped.offset() == 4);
  CHECK(clipped.size() == 6);

  auto eof = ctx->compute_physical_range({10, 16}, 10);
  CHECK(eof.offset() == 10);
  CHECK(eof.size() == 0);
}

TEST_CASE("s3_ioctx async host reads keep context and object alive until completion", "[s3][ioctx]")
{
  auto provider = std::make_shared<blocking_throwing_provider>();
  auto ctx      = std::make_shared<s3_ioctx>(s3_ioctx_config{provider, 1, 1});
  auto obj      = make_s3_object("bucket", "key", 16);
  std::vector<std::uint8_t> dst(4);

  auto done = std::make_shared<std::promise<std::pair<std::size_t, std::exception_ptr>>>();
  auto fut  = done->get_future();
  ctx->host_read_async_io(
    *obj, 0, dst.size(), dst.data(), [done](auto bytes, auto ep) { done->set_value({bytes, ep}); });

  REQUIRE(provider->wait_until_entered(5s));
  ctx.reset();
  obj.reset();
  provider->release();

  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  auto [bytes, ep] = fut.get();
  CHECK(bytes == 0);
  REQUIRE(ep != nullptr);
  CHECK_THROWS_AS(std::rethrow_exception(ep), credential_error);
  CHECK(provider->method() == presign_method::GET);
}

TEST_CASE("s3_ioctx schedules async read entry points on the injected pool", "[s3][ioctx]")
{
  sirius::exec::static_thread_pool pool(2, "s3tp");
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  s3_ioctx_config cfg{provider, 1, 1};
  cfg.async_thread_pool = &pool;

  auto ctx = std::make_shared<s3_ioctx>(std::move(cfg));
  auto obj = make_s3_object("bucket", "empty", 0);

  std::vector<std::uint8_t> host_dst(1);
  auto host = observe_async_completion([&](auto handler) {
    ctx->host_read_async_io(*obj, 0, 0, host_dst.data(), std::move(handler));
  });

  std::vector<cudf::io::text::byte_range_info> ranges;
  std::vector<cudf::host_span<std::byte>> spans;
  auto range = observe_async_completion([&](auto handler) {
    ctx->host_read_ranges_async_io(
      *obj, ranges, std::span<cudf::host_span<std::byte>>{spans}, std::move(handler));
  });

  auto device = observe_async_completion([&](auto handler) {
    ctx->device_read_async_io(*obj, 0, 0, nullptr, rmm::cuda_stream_default, std::move(handler));
  });

  for (auto const& obs : {host, range, device}) {
    CHECK(obs.bytes == 0);
    CHECK(obs.error == nullptr);
    CHECK(obs.thread_name.rfind("s3tp_", 0) == 0);
  }
}

TEST_CASE("s3_ioctx keeps standalone detached async fallback when no pool is injected",
          "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  s3_ioctx_config cfg{provider, 1, 1};
  cfg.async_thread_pool = nullptr;

  auto ctx = std::make_shared<s3_ioctx>(std::move(cfg));
  auto obj = make_s3_object("bucket", "empty", 0);
  std::vector<std::uint8_t> dst(1);

  auto obs = observe_async_completion(
    [&](auto handler) { ctx->host_read_async_io(*obj, 0, 0, dst.data(), std::move(handler)); });

  CHECK(obs.bytes == 0);
  CHECK(obs.error == nullptr);
}

TEST_CASE("s3_ioctx shutdown does not stop the caller-owned async pool", "[s3][ioctx]")
{
  sirius::exec::static_thread_pool pool(1, "s3alive");
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  s3_ioctx_config cfg{provider, 1, 1};
  cfg.async_thread_pool = &pool;

  auto ctx = std::make_shared<s3_ioctx>(std::move(cfg));
  ctx->shutdown();

  std::promise<void> ran;
  auto fut = ran.get_future();
  pool.schedule([&ran] { ran.set_value(); });
  CHECK(fut.wait_for(1s) == std::future_status::ready);
}

TEST_CASE("s3_ioctx retries HTTP 5xx range GETs", "[s3][ioctx][retry]")
{
  scripted_http_server server({http_error(500, "Internal Server Error"),
                               http_error(500, "Internal Server Error"),
                               range_ok("hello", 5)});
  auto ctx = make_retry_ioctx(server.url(), 3);
  auto obj = make_s3_object("bucket", "key", 5);
  std::vector<std::uint8_t> dst(5);

  REQUIRE(ctx->host_read(*obj, 0, dst.size(), dst.data()) == dst.size());
  CHECK(std::string(dst.begin(), dst.end()) == "hello");
  CHECK(server.request_count() == 3);
}

TEST_CASE("s3_ioctx honors Retry-After for HTTP 429 range GET retries", "[s3][ioctx][retry]")
{
  scripted_http_response throttled{429, "Too Many Requests", {"Retry-After: 1"}, "slow down"};
  scripted_http_server server({throttled, range_ok("ok", 2)});
  auto ctx = make_retry_ioctx(server.url(), 2, 0ms, 0ms, true);
  auto obj = make_s3_object("bucket", "key", 2);
  std::vector<std::uint8_t> dst(2);

  auto const start = std::chrono::steady_clock::now();
  REQUIRE(ctx->host_read(*obj, 0, dst.size(), dst.data()) == dst.size());
  auto const elapsed = std::chrono::steady_clock::now() - start;

  CHECK(std::string(dst.begin(), dst.end()) == "ok");
  CHECK(server.request_count() == 2);
  CHECK(elapsed >= 900ms);
}

TEST_CASE("s3_ioctx retries transient libcurl receive failures", "[s3][ioctx][retry]")
{
  scripted_http_response transient{};
  transient.close_without_response = true;
  scripted_http_server server({transient, range_ok("abc", 3)});
  auto ctx = make_retry_ioctx(server.url(), 2);
  auto obj = make_s3_object("bucket", "key", 3);
  std::vector<std::uint8_t> dst(3);

  REQUIRE(ctx->host_read(*obj, 0, dst.size(), dst.data()) == dst.size());
  CHECK(std::string(dst.begin(), dst.end()) == "abc");
  CHECK(server.request_count() == 2);
}

TEST_CASE("s3_ioctx does not retry authorization or missing-key HTTP failures",
          "[s3][ioctx][retry]")
{
  scripted_http_server forbidden(
    {http_error(403, "Forbidden", "<Error><Code>SignatureDoesNotMatch</Code></Error>"),
     range_ok("bad", 3)});
  auto forbidden_ctx = make_retry_ioctx(forbidden.url(), 3);
  auto obj           = make_s3_object("bucket", "key", 3);
  std::vector<std::uint8_t> dst(3);

  CHECK_THROWS_AS(forbidden_ctx->host_read(*obj, 0, dst.size(), dst.data()), std::runtime_error);
  CHECK(forbidden.request_count() == 1);

  scripted_http_server missing(
    {http_error(404, "Not Found", "<Error><Code>NoSuchKey</Code></Error>"), range_ok("bad", 3)});
  auto missing_ctx = make_retry_ioctx(missing.url(), 3);

  CHECK_THROWS_AS(missing_ctx->host_read(*obj, 0, dst.size(), dst.data()), std::runtime_error);
  CHECK(missing.request_count() == 1);
}

TEST_CASE("s3_ioctx retries HEAD object-size requests with the same policy", "[s3][ioctx][retry]")
{
  scripted_http_server server({http_error(503, "Service Unavailable"), head_ok(123)});
  auto ctx = make_retry_ioctx(server.url(), 2);

  CHECK(ctx->head_object_size("bucket", "key") == 123);
  CHECK(server.request_count() == 2);
}

TEST_CASE("s3_ioctx reports retry exhaustion with attempt diagnostics", "[s3][ioctx][retry]")
{
  scripted_http_server server({http_error(503, "Service Unavailable"),
                               http_error(503, "Service Unavailable"),
                               http_error(503, "Service Unavailable")});
  auto ctx = make_retry_ioctx(server.url(), 3);
  auto obj = make_s3_object("bucket", "key", 1);
  std::vector<std::uint8_t> dst(1);

  try {
    (void)ctx->host_read(*obj, 0, dst.size(), dst.data());
    FAIL("range GET succeeded after every scripted attempt returned 503");
  } catch (std::runtime_error const& e) {
    auto const msg = std::string{e.what()};
    CHECK(msg.find("503") != std::string::npos);
    CHECK(msg.find("attempt") != std::string::npos);
    CHECK(msg.find("3") != std::string::npos);
  }
  CHECK(server.request_count() == 3);
}

TEST_CASE("s3_ioctx retry backoff timing stays within tolerance", "[s3][ioctx][retry]")
{
  scripted_http_server server({http_error(500, "Internal Server Error"), range_ok("x", 1)});
  auto ctx = make_retry_ioctx(server.url(), 2, 100ms, 0ms);
  auto obj = make_s3_object("bucket", "key", 1);
  std::vector<std::uint8_t> dst(1);

  auto const start = std::chrono::steady_clock::now();
  REQUIRE(ctx->host_read(*obj, 0, dst.size(), dst.data()) == dst.size());
  auto const elapsed = std::chrono::steady_clock::now() - start;

  CHECK(elapsed >= 100ms);
  CHECK(elapsed < 700ms);
  CHECK(server.request_count() == 2);
}

TEST_CASE("s3_ioctx reads single objects from MinIO with presigned range GETs",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  auto key   = env_or("SIRIUS_TEST_S3_KEY", "hello.txt");
  auto local = read_binary_file(env->local_dir / key);
  auto ctx   = make_live_ioctx(*env);
  auto size  = try_head_or_skip(*ctx, *env, key);
  if (!size) return;

  REQUIRE(*size == local.size());
  auto obj = make_s3_object(env->bucket, key, *size);

  std::vector<std::uint8_t> full(local.size());
  CHECK(ctx->host_read(*obj, 0, full.size(), full.data()) == full.size());
  CHECK(full == local);

  std::vector<std::uint8_t> tail(5);
  CHECK(ctx->host_read(*obj, local.size() - tail.size(), tail.size(), tail.data()) == tail.size());
  CHECK(
    std::equal(tail.begin(), tail.end(), local.end() - static_cast<std::ptrdiff_t>(tail.size())));

  std::vector<std::uint8_t> eof(8, 0xCC);
  CHECK(ctx->host_read(*obj, local.size(), eof.size(), eof.data()) == 0);
  CHECK(std::all_of(eof.begin(), eof.end(), [](auto b) { return b == 0xCC; }));
}

TEST_CASE("s3_ioctx create_io_object populates S3 object metadata from MinIO",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  auto key   = env_or("SIRIUS_TEST_S3_KEY", "hello.txt");
  auto local = read_binary_file(env->local_dir / key);
  auto ctx   = make_live_ioctx(*env);
  auto path  = s3_uri(env->bucket, key);

  auto object = ctx->create_io_object(path);
  REQUIRE(object != nullptr);
  CHECK(object->size() == local.size());
  CHECK(object->object_path() == path);
  CHECK(object->raw_file_cache_id() == path);

  auto s3_object = std::dynamic_pointer_cast<s3_io_object>(object);
  REQUIRE(s3_object != nullptr);
  CHECK(s3_object->bucket() == env->bucket);
  CHECK(s3_object->key() == key);
}

TEST_CASE("s3_ioctx create_io_object propagates missing S3 key HEAD failures",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  auto ctx = make_live_ioctx(*env);
  CHECK_THROWS(ctx->create_io_object(s3_uri(env->bucket, "nonexistent-key-xyz")));
}

TEST_CASE("s3_ioctx reads multiple MinIO byte ranges", "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  auto ctx              = make_live_ioctx(*env);
  auto size             = try_head_or_skip(*ctx, *env, key);
  if (!size) return;
  REQUIRE(*size == local.size());

  auto obj = make_s3_object(env->bucket, key, *size);
  std::vector<cudf::io::text::byte_range_info> ranges{
    {0, 16},
    {17, 31},
    {4093, 64},
    {static_cast<int64_t>(local.size() - 33), 33},
  };

  std::vector<std::vector<std::byte>> buffers;
  buffers.reserve(ranges.size());
  for (auto const& range : ranges) {
    buffers.emplace_back(static_cast<std::size_t>(range.size()));
  }

  std::vector<cudf::host_span<std::byte>> spans;
  spans.reserve(buffers.size());
  for (auto& buffer : buffers) {
    spans.emplace_back(buffer.data(), buffer.size());
  }

  auto const [total, ep] =
    read_ranges_async(*ctx, *obj, ranges, std::span<cudf::host_span<std::byte>>{spans});
  REQUIRE(ep == nullptr);
  CHECK(total == 16 + 31 + 64 + 33);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    require_bytes_equal(buffers[i], local, static_cast<std::size_t>(ranges[i].offset()));
  }
}

TEST_CASE("s3_ioctx async range reads copy caller span descriptors before dispatch",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= 32);

  static_credentials creds;
  creds.access_key_id     = env->access_key;
  creds.secret_access_key = env->secret_key;
  auto delegate           = std::make_shared<sirius_sigv4_credential_provider>(
    std::move(creds), env->region, env->endpoint, 30min);
  auto provider = std::make_shared<blocking_first_get_provider>(std::move(delegate));
  auto ctx      = std::make_shared<s3_ioctx>(s3_ioctx_config{provider, 2, 20});
  auto obj      = make_s3_object(env->bucket, key, local.size());

  std::vector<cudf::io::text::byte_range_info> ranges{{0, 8}, {8, 8}};
  std::vector<std::byte> first(8);
  std::vector<std::byte> second(8);
  std::vector<std::byte> poison_second(8, std::byte{0xA5});

  auto done = std::make_shared<std::promise<std::pair<std::size_t, std::exception_ptr>>>();
  auto fut  = done->get_future();

  {
    std::vector<cudf::host_span<std::byte>> descriptors;
    descriptors.emplace_back(first.data(), first.size());
    descriptors.emplace_back(second.data(), second.size());

    ctx->host_read_ranges_async_io(*obj,
                                   ranges,
                                   std::span<cudf::host_span<std::byte>>{descriptors},
                                   [done](auto bytes, auto ep) { done->set_value({bytes, ep}); });

    auto const entered = provider->wait_until_first_get_entered(5s);
    if (!entered) { provider->release(); }
    REQUIRE(entered);
    descriptors[1] = cudf::host_span<std::byte>{poison_second.data(), poison_second.size()};
  }

  provider->release();

  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  auto [bytes, ep] = fut.get();
  REQUIRE(ep == nullptr);
  CHECK(bytes == 16);
  require_bytes_equal(first, local, 0);
  require_bytes_equal(second, local, 8);
  CHECK(std::all_of(
    poison_second.begin(), poison_second.end(), [](auto b) { return b == std::byte{0xA5}; }));
}

TEST_CASE("s3_ioctx host_read_ranges clips EOF-crossing range before dst validation",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= 200);
  auto ctx = make_live_ioctx(*env);
  auto obj = make_s3_object(env->bucket, key, local.size());

  auto const offset = local.size() - 100;
  std::vector<cudf::io::text::byte_range_info> ranges{
    {static_cast<int64_t>(offset), 200},
  };
  std::vector<std::byte> buffer(100);
  std::vector<cudf::host_span<std::byte>> spans;
  spans.emplace_back(buffer.data(), buffer.size());

  auto const [total, ep] =
    read_ranges_async(*ctx, *obj, ranges, std::span<cudf::host_span<std::byte>>{spans});

  REQUIRE(ep == nullptr);
  CHECK(total == 100);
  require_bytes_equal(buffer, local, offset);
}

TEST_CASE("s3_ioctx host_read_ranges returns zero for ranges starting at EOF",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  auto ctx              = make_live_ioctx(*env);
  auto obj              = make_s3_object(env->bucket, key, local.size());

  std::vector<cudf::io::text::byte_range_info> ranges{
    {static_cast<int64_t>(local.size()), 50},
  };
  std::vector<std::byte> empty;
  std::vector<cudf::host_span<std::byte>> spans;
  spans.emplace_back(empty.data(), empty.size());

  auto const [total, ep] =
    read_ranges_async(*ctx, *obj, ranges, std::span<cudf::host_span<std::byte>>{spans});

  REQUIRE(ep == nullptr);
  CHECK(total == 0);
}

TEST_CASE("s3_ioctx host_read_ranges rejects dst smaller than clipped size",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= 200);
  auto ctx = make_live_ioctx(*env);
  auto obj = make_s3_object(env->bucket, key, local.size());

  auto const offset = local.size() - 100;
  std::vector<cudf::io::text::byte_range_info> ranges{
    {static_cast<int64_t>(offset), 200},
  };
  std::vector<std::byte> buffer(50);
  std::vector<cudf::host_span<std::byte>> spans;
  spans.emplace_back(buffer.data(), buffer.size());

  auto const [bytes, ep] =
    read_ranges_async(*ctx, *obj, ranges, std::span<cudf::host_span<std::byte>>{spans});

  CHECK(bytes == 0);
  REQUIRE(ep != nullptr);
  CHECK_THROWS_WITH(std::rethrow_exception(ep), "s3_ioctx::host_read_ranges: dst span too small");
}

TEST_CASE("s3_ioctx host_read_ranges clips EOF-crossing ranges independently",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= 150);
  auto ctx = make_live_ioctx(*env);
  auto obj = make_s3_object(env->bucket, key, local.size());

  auto const eof_crossing_offset = local.size() - 50;
  std::vector<cudf::io::text::byte_range_info> ranges{
    {0, 100},
    {static_cast<int64_t>(eof_crossing_offset), 100},
  };
  std::vector<std::byte> head(100);
  std::vector<std::byte> tail(50);
  std::vector<cudf::host_span<std::byte>> spans;
  spans.emplace_back(head.data(), head.size());
  spans.emplace_back(tail.data(), tail.size());

  auto const [total, ep] =
    read_ranges_async(*ctx, *obj, ranges, std::span<cudf::host_span<std::byte>>{spans});

  REQUIRE(ep == nullptr);
  CHECK(total == 150);
  require_bytes_equal(head, local, 0);
  require_bytes_equal(tail, local, eof_crossing_offset);
}

TEST_CASE("s3_ioctx device_read uses bounded FSMR staging for multi-chunk S3 objects",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  constexpr std::size_t block_size = 1 << 20;
  std::string const key            = "medium.bin";
  auto local                       = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() > block_size);

  fsmr_test_resources memory(block_size, block_size, block_size);
  auto before_peak = memory.host_mr.get_peak_total_allocated_bytes();

  auto provider = std::make_shared<delayed_get_provider>(make_live_provider(*env), 0ms);
  auto ctx      = make_live_ioctx_with_fsmr(provider, memory.host_mr);
  auto obj      = make_s3_object(env->bucket, key, local.size());

  rmm::cuda_stream stream;
  device_byte_buffer dst(local.size());
  auto const got = ctx->device_read(*obj, 0, local.size(), dst.data(), stream.view());

  CHECK(got == local.size());
  require_bytes_equal(copy_device_to_host(dst, got), local, 0);

  auto const peak_delta = memory.host_mr.get_peak_total_allocated_bytes() - before_peak;
  CHECK(peak_delta > 0);
  CHECK(peak_delta <= block_size);
  CHECK(memory.host_mr.get_free_blocks() == 1);
  CHECK(provider->get_count() == static_cast<int>((local.size() + block_size - 1) / block_size));
}

TEST_CASE("s3_ioctx device_read with FSMR staging handles nonzero-offset remainder chunks",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  constexpr std::size_t block_size = 4096;
  std::string const key            = "medium.bin";
  auto local                       = read_binary_file(env->local_dir / key);
  auto const offset                = std::size_t{17};
  auto const request_size          = block_size + 123;
  REQUIRE(local.size() >= offset + request_size);

  fsmr_test_resources memory(block_size, block_size, block_size);
  auto before_peak = memory.host_mr.get_peak_total_allocated_bytes();

  auto provider = std::make_shared<delayed_get_provider>(make_live_provider(*env), 0ms);
  auto ctx      = make_live_ioctx_with_fsmr(provider, memory.host_mr);
  auto obj      = make_s3_object(env->bucket, key, local.size());

  rmm::cuda_stream stream;
  device_byte_buffer dst(request_size);
  auto const got = ctx->device_read(*obj, offset, request_size, dst.data(), stream.view());

  CHECK(got == request_size);
  require_bytes_equal(copy_device_to_host(dst, got), local, offset);
  CHECK(memory.host_mr.get_peak_total_allocated_bytes() - before_peak <= block_size);
  CHECK(provider->get_count() == 2);
}

TEST_CASE("s3_ioctx device_read with FSMR staging clips EOF-crossing reads",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  constexpr std::size_t block_size = 4096;
  std::string const key            = "medium.bin";
  auto local                       = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() > 512);

  fsmr_test_resources memory(block_size, block_size, block_size);
  auto ctx = make_live_ioctx_with_fsmr(make_live_provider(*env), memory.host_mr);
  auto obj = make_s3_object(env->bucket, key, local.size());

  auto const offset       = local.size() - 123;
  auto const request_size = block_size + 99;

  rmm::cuda_stream stream;
  device_byte_buffer dst(request_size);
  auto const got = ctx->device_read(*obj, offset, request_size, dst.data(), stream.view());

  CHECK(got == 123);
  require_bytes_equal(copy_device_to_host(dst, got), local, offset);
}

TEST_CASE("s3_ioctx device_read with FSMR staging returns zero without borrowing a block",
          "[s3][ioctx][integration]")
{
  constexpr std::size_t block_size = 4096;
  fsmr_test_resources memory(block_size, block_size, block_size);
  auto before_peak = memory.host_mr.get_peak_total_allocated_bytes();

  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  auto ctx      = make_live_ioctx_with_fsmr(provider, memory.host_mr);
  auto obj      = make_s3_object("bucket", "key", 128);

  rmm::cuda_stream stream;
  CHECK(ctx->device_read(*obj, 0, 0, nullptr, stream.view()) == 0);
  CHECK(memory.host_mr.get_peak_total_allocated_bytes() == before_peak);
  CHECK(memory.host_mr.get_free_blocks() == 0);
}

TEST_CASE("s3_ioctx device_read keeps vector fallback when no FSMR is injected",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "small.bin";
  auto local            = read_binary_file(env->local_dir / key);
  auto ctx              = make_live_ioctx(*env);
  auto obj              = make_s3_object(env->bucket, key, local.size());

  rmm::cuda_stream stream;
  device_byte_buffer dst(local.size());
  auto const got = ctx->device_read(*obj, 0, local.size(), dst.data(), stream.view());

  CHECK(got == local.size());
  require_bytes_equal(copy_device_to_host(dst, got), local, 0);
}

TEST_CASE("s3_ioctx device_read rejects an injected FSMR with zero block size",
          "[s3][ioctx][integration]")
{
  fsmr_test_resources memory(/*block_size=*/0, /*capacity=*/0, /*memory_limit=*/0);
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  auto ctx      = make_live_ioctx_with_fsmr(provider, memory.host_mr);
  auto obj      = make_s3_object("bucket", "key", 1);

  rmm::cuda_stream stream;
  device_byte_buffer dst(1);

  try {
    (void)ctx->device_read(*obj, 0, 1, dst.data(), stream.view());
    FAIL("expected zero-block-size FSMR to be rejected");
  } catch (std::exception const& e) {
    auto const msg = std::string{e.what()};
    CHECK(msg.find("s3_ioctx::device_read_io") != std::string::npos);
    CHECK(msg.find("block size is zero") != std::string::npos);
  }
}

TEST_CASE("s3_ioctx device_read reports context when FSMR staging allocation is exhausted",
          "[s3][ioctx][integration]")
{
  constexpr std::size_t block_size = 4096;
  fsmr_test_resources memory(block_size, /*capacity=*/0, /*memory_limit=*/0);
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  auto ctx      = make_live_ioctx_with_fsmr(provider, memory.host_mr);
  auto obj      = make_s3_object("bucket", "key", 1);

  rmm::cuda_stream stream;
  device_byte_buffer dst(1);

  try {
    (void)ctx->device_read(*obj, 0, 1, dst.data(), stream.view());
    FAIL("expected FSMR exhaustion to surface as rmm::out_of_memory");
  } catch (rmm::out_of_memory const& e) {
    auto const msg = std::string{e.what()};
    CHECK(msg.find("s3_ioctx::device_read_io") != std::string::npos);
    CHECK(msg.find("fixed_size_host_memory_resource") != std::string::npos);
  } catch (std::exception const& e) {
    FAIL("expected rmm::out_of_memory, got: " << e.what());
  }
}

TEST_CASE("s3_ioctx host_read_ranges_async_io fans ranges across the injected pool",
          "[s3][ioctx][parallel][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  constexpr std::size_t range_count  = 8;
  constexpr std::size_t worker_count = 4;
  constexpr auto per_get_delay       = 100ms;
  constexpr auto fanout_waves        = (range_count + worker_count - 1) / worker_count;
  constexpr auto expected_budget     = per_get_delay * 3 * fanout_waves;

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= range_count * 128);

  auto provider = std::make_shared<delayed_get_provider>(make_live_provider(*env), per_get_delay);
  sirius::exec::static_thread_pool pool(static_cast<int>(worker_count), "s3_range_fanout_test");

  s3_ioctx_config cfg{};
  cfg.creds             = provider;
  cfg.max_connections   = range_count;
  cfg.request_timeout_s = 20;
  cfg.async_thread_pool = &pool;
  auto ctx              = std::make_shared<s3_ioctx>(std::move(cfg));
  auto obj              = make_s3_object(env->bucket, key, local.size());

  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.reserve(range_count);
  std::vector<std::vector<std::byte>> buffers;
  buffers.reserve(range_count);
  for (std::size_t i = 0; i < range_count; ++i) {
    ranges.push_back({static_cast<int64_t>(i * 128), 64});
    buffers.emplace_back(64);
  }

  std::vector<cudf::host_span<std::byte>> spans;
  spans.reserve(buffers.size());
  for (auto& buffer : buffers) {
    spans.emplace_back(buffer.data(), buffer.size());
  }

  auto const t0 = std::chrono::steady_clock::now();
  auto const [total, ep] =
    read_ranges_async(*ctx, *obj, ranges, std::span<cudf::host_span<std::byte>>{spans});
  auto const elapsed =
    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);

  REQUIRE(ep == nullptr);
  CHECK(total == range_count * 64);
  CHECK(provider->get_count() == static_cast<int>(range_count));
  CHECK(elapsed <= expected_budget);
}

TEST_CASE("s3_ioctx destroys its prefetch cache before shutting down S3 async workers",
          "[s3][ioctx][teardown][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= 128);

  cache_test_resources cache_resources;
  auto ctx = make_live_ioctx(*env);
  ctx->initialize_cache(cache_resources.pool, 8);
  REQUIRE(ctx->cache() != nullptr);

  auto obj = make_s3_object(env->bucket, key, local.size());
  std::vector<cudf::io::text::byte_range_info> ranges{{0, 128}};
  for (int i = 0; i < 64; ++i) {
    ctx->cache()->insert(*obj, nullptr, ranges);
  }
  REQUIRE(wait_until_cached(*ctx, *obj, ranges.front(), 10s));

  auto const destroy_start = std::chrono::steady_clock::now();
  ctx.reset();
  auto const destroy_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now() - destroy_start);

  CHECK(destroy_elapsed < 2s);
}
