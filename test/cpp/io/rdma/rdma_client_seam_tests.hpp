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

#pragma once

#include "catch.hpp"
#include "io/object_store_config.hpp"
#include "io/rdma/cuobj_rdma_client.hpp"
#include "io/rdma/cuobj_rdma_reactor.hpp"
#include "io/rdma/mock_rdma_client.hpp"
#include "io/rdma/rdma_client.hpp"
#include "io/s3/s3_rdma_ioctx.hpp"
#include "io/s3/sirius_sigv4_authorizer.hpp"
#include "io/s3/static_credentials.hpp"
#include "io/sirius_datasource.hpp"
#include "rdma_test_transport.hpp"
#include "utils/log_test_utils.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#ifdef SIRIUS_HAVE_TESTCONTAINERS
#include "utils/s3_container.hpp"

#include <curl/curl.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace s3_rdma_client_seam_tests {

using sirius::io::object_store_config;
using sirius::io::rdma::cuda_delivery_ops;
using sirius::io::rdma::cuobj_rdma_reactor;
using sirius::io::rdma::curl_s3_control_client;
using sirius::io::rdma::data_commit_state;
using sirius::io::rdma::data_get_result;
using sirius::io::rdma::rx_route;
using sirius::io::s3::s3_rdma_ioctx;
using sirius::test::rdma::mock_transport_fixture;
using sirius::test::rdma::seeded_mock_transport;
using namespace std::chrono_literals;

constexpr std::size_t k_slot_size   = 64UL << 10;
constexpr std::string_view k_bucket = "bucket";

object_store_config mock_config(std::size_t max_inflight = 1)
{
  object_store_config cfg;
  cfg.endpoint                     = "http://control.example.invalid";
  cfg.region                       = "us-east-1";
  cfg.access_key                   = "mock-access-key";
  cfg.secret_key                   = "mock-secret-key";
  cfg.s3_signing_mode              = object_store_config::signing_mode::header;
  cfg.s3_transport                 = object_store_config::transport::RDMA;
  cfg.s3_rdma_max_inflight         = max_inflight;
  cfg.s3_rdma_arena_slot_size      = k_slot_size;
  cfg.s3_rdma_data.endpoint        = "http://data.example.invalid";
  cfg.s3_rdma_data.region          = cfg.region;
  cfg.s3_rdma_data.access_key      = cfg.access_key;
  cfg.s3_rdma_data.secret_key      = cfg.secret_key;
  cfg.s3_rdma_data.s3_signing_mode = object_store_config::signing_mode::header;
  cfg.s3_rdma_data.tls_verify      = false;
  return cfg;
}

std::vector<std::uint8_t> pattern_bytes(std::size_t size, std::uint8_t salt = 61)
{
  std::vector<std::uint8_t> bytes(size);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131U + salt) & 0xffU);
  }
  return bytes;
}

bool cuda_device_available()
{
  int count       = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    WARN("Skipping S3 RDMA client-seam device test: no CUDA device is available");
    return false;
  }
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  return true;
}

std::shared_ptr<s3_rdma_ioctx> make_started_ioctx(
  std::shared_ptr<mock_transport_fixture> const& transport,
  object_store_config cfg                         = mock_config(),
  sirius::io::rdma::reply_tag_predicate predicate = &sirius::io::rdma::non_empty_reply_tag)
{
  auto ctx = std::make_shared<s3_rdma_ioctx>(
    std::move(cfg), transport->clients(predicate), sirius::io::rdma::cuda_delivery_ops{});
  ctx->start();
  return ctx;
}

std::unique_ptr<sirius::io::sirius_datasource> open_ds(std::shared_ptr<s3_rdma_ioctx> const& ctx,
                                                       std::string_view key)
{
  return ctx->open_datasource("s3://" + std::string{k_bucket} + "/" + std::string{key});
}

std::string ready_error(std::future<std::size_t>& future, std::chrono::milliseconds timeout = 5s)
{
  REQUIRE(future.wait_for(timeout) == std::future_status::ready);
  try {
    (void)future.get();
    FAIL("expected S3 RDMA read to fail");
  } catch (std::exception const& error) {
    return error.what();
  }
  return {};
}

template <typename Fn>
std::string thrown_error(Fn&& fn)
{
  try {
    std::forward<Fn>(fn)();
  } catch (std::exception const& error) {
    return error.what();
  }
  return {};
}

std::future<std::size_t> issue_device_read(sirius::io::sirius_datasource& datasource,
                                           rmm::device_buffer& destination,
                                           rmm::cuda_stream_view stream)
{
  return datasource.device_read_async(
    0, destination.size(), static_cast<std::uint8_t*>(destination.data()), stream);
}

bool wait_until(auto&& predicate, std::chrono::milliseconds timeout = 5s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) { return true; }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
}

bool accepts_test_tag(std::string_view tag) noexcept { return tag == "accepted-test-tag"; }

class blocking_error_log_sink final : public sirius::log::sink {
 public:
  void set_level(sirius::log::level level) override
  {
    _level.store(level, std::memory_order_relaxed);
  }

  bool should_log(sirius::log::level level) const override
  {
    return static_cast<int>(level) >= static_cast<int>(_level.load(std::memory_order_relaxed));
  }

  void log(sirius::log::level level,
           const std::source_location& /*loc*/,
           std::string_view /*message*/) override
  {
    if (!should_log(level)) { return; }
    std::unique_lock lock{_mutex};
    _entered = true;
    _cv.notify_all();
    _cv.wait(lock, [&] { return _released; });
  }

  bool flush() override { return true; }

  bool wait_until_entered(std::chrono::milliseconds timeout = 5s)
  {
    std::unique_lock lock{_mutex};
    return _cv.wait_for(lock, timeout, [&] { return _entered; });
  }

  void release()
  {
    {
      std::lock_guard lock{_mutex};
      _released = true;
    }
    _cv.notify_all();
  }

 private:
  std::atomic<sirius::log::level> _level{sirius::log::level::off};
  std::mutex _mutex;
  std::condition_variable _cv;
  bool _entered{false};
  bool _released{false};
};

class release_log_sink_on_exit {
 public:
  explicit release_log_sink_on_exit(std::shared_ptr<blocking_error_log_sink> sink)
    : _sink(std::move(sink))
  {
  }

  ~release_log_sink_on_exit() { _sink->release(); }

  release_log_sink_on_exit(release_log_sink_on_exit const&)            = delete;
  release_log_sink_on_exit& operator=(release_log_sink_on_exit const&) = delete;

 private:
  std::shared_ptr<blocking_error_log_sink> _sink;
};

void require_no_long_fragment(std::string_view text, std::string_view secret)
{
  REQUIRE(secret.size() > 8);
  for (std::size_t begin = 0; begin + 9 <= secret.size(); ++begin) {
    CHECK(text.find(secret.substr(begin, 9)) == std::string_view::npos);
  }
}

class head_without_length_server {
 public:
  head_without_length_server()
  {
    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_listen_fd < 0) { throw_errno("socket"); }

    try {
      int one = 1;
      if (::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
        throw_errno("setsockopt");
      }

      sockaddr_in address{};
      address.sin_family      = AF_INET;
      address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      address.sin_port        = 0;
      if (::bind(_listen_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        throw_errno("bind");
      }
      if (::listen(_listen_fd, 1) != 0) { throw_errno("listen"); }

      socklen_t length = sizeof(address);
      if (::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
        throw_errno("getsockname");
      }
      _port            = ntohs(address.sin_port);
      int const socket = _listen_fd;
      _thread          = std::thread([this, socket] { serve_one(socket); });
    } catch (...) {
      close_listener();
      throw;
    }
  }

  ~head_without_length_server()
  {
    _stopping.store(true, std::memory_order_relaxed);
    close_listener();
    if (_thread.joinable()) { _thread.join(); }
  }

  head_without_length_server(head_without_length_server const&)            = delete;
  head_without_length_server& operator=(head_without_length_server const&) = delete;

  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }

  void wait()
  {
    if (_thread.joinable()) { _thread.join(); }
    close_listener();
    if (_error) { std::rethrow_exception(_error); }
  }

 private:
  [[noreturn]] static void throw_errno(char const* operation)
  {
    throw std::system_error(errno, std::generic_category(), operation);
  }

  void close_listener() noexcept
  {
    int const socket = std::exchange(_listen_fd, -1);
    if (socket >= 0) {
      (void)::shutdown(socket, SHUT_RDWR);
      (void)::close(socket);
    }
  }

  static void send_all(int socket, std::string_view response)
  {
    std::size_t sent = 0;
    while (sent < response.size()) {
      auto const bytes =
        ::send(socket, response.data() + sent, response.size() - sent, MSG_NOSIGNAL);
      if (bytes > 0) {
        sent += static_cast<std::size_t>(bytes);
      } else if (bytes < 0 && errno == EINTR) {
        continue;
      } else {
        throw_errno("send");
      }
    }
  }

  void serve_one(int listen_socket) noexcept
  {
    int client = -1;
    try {
      pollfd ready{listen_socket, POLLIN, 0};
      int polled = 0;
      do {
        polled = ::poll(&ready, 1, 5000);
      } while (polled < 0 && errno == EINTR);
      if (polled == 0) {
        throw std::runtime_error("HEAD test server timed out waiting for a client");
      }
      if (polled < 0) { throw_errno("poll"); }

      client = ::accept(listen_socket, nullptr, nullptr);
      if (client < 0) { throw_errno("accept"); }

      timeval timeout{};
      timeout.tv_sec = 5;
      if (::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0) {
        throw_errno("setsockopt receive timeout");
      }

      std::string request;
      std::array<char, 4096> buffer{};
      while (request.find("\r\n\r\n") == std::string::npos) {
        auto const bytes = ::recv(client, buffer.data(), buffer.size(), 0);
        if (bytes > 0) {
          request.append(buffer.data(), static_cast<std::size_t>(bytes));
        } else if (bytes < 0 && errno == EINTR) {
          continue;
        } else if (bytes == 0) {
          throw std::runtime_error("HEAD test client closed before sending all headers");
        } else {
          throw_errno("recv");
        }
        if (request.size() > 64UL << 10) {
          throw std::runtime_error("HEAD test request exceeded 64 KiB");
        }
      }
      if (!request.starts_with("HEAD ")) {
        throw std::runtime_error("HEAD test server received a non-HEAD request");
      }

      send_all(client, "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n");
    } catch (...) {
      if (!_stopping.load(std::memory_order_relaxed)) { _error = std::current_exception(); }
    }
    if (client >= 0) {
      (void)::shutdown(client, SHUT_RDWR);
      (void)::close(client);
    }
  }

  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::thread _thread;
  std::atomic<bool> _stopping{false};
  std::exception_ptr _error;
};

std::shared_ptr<sirius::io::s3::sirius_sigv4_header_authorizer> make_loopback_authorizer(
  std::string endpoint)
{
  sirius::io::s3::static_credentials credentials;
  credentials.access_key_id     = "test-access-key";
  credentials.secret_access_key = "test-secret-key";
  return std::make_shared<sirius::io::s3::sirius_sigv4_header_authorizer>(
    std::move(credentials), "us-east-1", std::move(endpoint));
}

void require_rejected_content_range(std::string_view key, std::string content_range)
{
  auto payload   = pattern_bytes(4096);
  auto transport = seeded_mock_transport(std::string{k_bucket}, std::string{key}, payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = open_ds(ctx, key);
  std::array<std::uint8_t, 32> destination{};
  auto const calls_before = transport->control->range_gets_issued();
  transport->control->override_content_range(std::move(content_range));

  auto const error =
    thrown_error([&] { (void)ds->host_read(0, destination.size(), destination.data()); });
  REQUIRE_FALSE(error.empty());
  CHECK(error.find("Content-Range") != std::string::npos);
  CHECK(transport->control->range_gets_issued() == calls_before + 1);
  auto snapshot = ctx->perf_snapshot();
  CHECK(snapshot.fail_stop_total == 0);
  CHECK(snapshot.retries_total == 0);

  CHECK(ds->host_read(0, destination.size(), destination.data()) == destination.size());
  CHECK(std::equal(destination.begin(), destination.end(), payload.begin()));
  CHECK(transport->control->range_gets_issued() == calls_before + 2);
  snapshot = ctx->perf_snapshot();
  CHECK(snapshot.fail_stop_total == 0);
  CHECK(snapshot.retries_total == 0);
}

#ifdef SIRIUS_HAVE_TESTCONTAINERS

constexpr std::string_view k_fixture_key = "small.bin";

std::string env_or(std::string_view name, std::string fallback = {})
{
  if (auto* value = std::getenv(std::string{name}.c_str()); value != nullptr) { return value; }
  return fallback;
}

std::string require_env(std::string_view name)
{
  auto value = env_or(name);
  REQUIRE_FALSE(value.empty());
  return value;
}

bool ensure_minio_env()
{
  if (sirius::test::ensure_s3_container_env()) { return true; }
  SUCCEED("SIRIUS_TEST_S3_* not set; skipping S3 RDMA client-seam MinIO test");
  return false;
}

struct minio_env {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  std::filesystem::path local_dir;
};

minio_env read_minio_env()
{
  return minio_env{require_env("SIRIUS_TEST_S3_ENDPOINT"),
                   env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
                   require_env("SIRIUS_TEST_S3_ACCESS_KEY"),
                   require_env("SIRIUS_TEST_S3_SECRET_KEY"),
                   require_env("SIRIUS_TEST_S3_BUCKET"),
                   std::filesystem::path{require_env("SIRIUS_TEST_S3_LOCAL_DIR")}};
}

std::vector<std::uint8_t> read_binary_file(std::filesystem::path const& path)
{
  std::ifstream input(path, std::ios::binary);
  REQUIRE(input.good());
  std::vector<char> chars((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  return std::vector<std::uint8_t>(chars.begin(), chars.end());
}

std::shared_ptr<sirius::io::s3::sirius_sigv4_header_authorizer> make_header_authorizer(
  minio_env const& env, std::string endpoint = {})
{
  sirius::io::s3::static_credentials credentials;
  credentials.access_key_id     = env.access_key;
  credentials.secret_access_key = env.secret_key;
  return std::make_shared<sirius::io::s3::sirius_sigv4_header_authorizer>(
    std::move(credentials), env.region, endpoint.empty() ? env.endpoint : std::move(endpoint));
}

std::unique_ptr<curl_s3_control_client> make_control_client(minio_env const& env,
                                                            std::string endpoint = {})
{
  return std::make_unique<curl_s3_control_client>(
    make_header_authorizer(env, std::move(endpoint)), std::string{}, false);
}

std::string header_value(std::vector<std::pair<std::string, std::string>> const& headers,
                         std::string_view name)
{
  for (auto const& [key, value] : headers) {
    if (key.size() == name.size() &&
        std::equal(key.begin(), key.end(), name.begin(), [](unsigned char lhs, unsigned char rhs) {
          return std::tolower(lhs) == std::tolower(rhs);
        })) {
      return value;
    }
  }
  return {};
}

std::size_t curl_write(char* data, std::size_t size, std::size_t count, void* opaque)
{
  auto* bytes  = static_cast<std::vector<std::uint8_t>*>(opaque);
  auto const n = size * count;
  bytes->insert(
    bytes->end(), reinterpret_cast<std::uint8_t*>(data), reinterpret_cast<std::uint8_t*>(data) + n);
  return n;
}

struct wire_response {
  long status{0};
  std::vector<std::uint8_t> body;
};

wire_response perform_get(sirius::io::s3::s3_authorized_request const& request)
{
  CURL* curl = curl_easy_init();
  REQUIRE(curl != nullptr);
  curl_slist* headers = nullptr;
  for (auto const& [name, value] : request.headers) {
    auto line = name + ": " + value;
    headers   = curl_slist_append(headers, line.c_str());
    REQUIRE(headers != nullptr);
  }

  wire_response response;
  curl_easy_setopt(curl, CURLOPT_URL, request.url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &curl_write);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response.body);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
  auto const rc = curl_easy_perform(curl);
  if (rc == CURLE_OK) { (void)curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.status); }
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  REQUIRE(rc == CURLE_OK);
  return response;
}

#endif

}  // namespace s3_rdma_client_seam_tests

#ifdef SIRIUS_HAVE_TESTCONTAINERS

TEST_CASE("s3_rdma AC1 control HEAD reports success and missing keys without throwing",
          "[s3][rdma][client-seam][integration]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!ensure_minio_env()) { return; }

  auto const env     = read_minio_env();
  auto const payload = read_binary_file(env.local_dir / std::string{k_fixture_key});
  auto client        = make_control_client(env);

  auto const found = client->head(rx_route{env.bucket, std::string{k_fixture_key}});
  CHECK(found.outcome.http_status == 200);
  CHECK(found.outcome.transport_error.empty());
  CHECK(found.object_size == payload.size());

  auto const missing = client->head(rx_route{env.bucket, "step5-ac1-missing.bin"});
  CHECK(missing.outcome.http_status == 404);
  CHECK(missing.outcome.transport_error.empty());
}

TEST_CASE("s3_rdma AC1 control range GET reports partial and past-EOF results",
          "[s3][rdma][client-seam][integration]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!ensure_minio_env()) { return; }

  auto const env     = read_minio_env();
  auto const payload = read_binary_file(env.local_dir / std::string{k_fixture_key});
  REQUIRE(payload.size() > 64);
  auto client = make_control_client(env);
  rx_route route{env.bucket, std::string{k_fixture_key}};

  std::array<std::uint8_t, 32> bytes{};
  auto const partial = client->range_get(route, 17, bytes.size(), bytes.data());
  CHECK(partial.outcome.http_status == 206);
  CHECK(partial.outcome.transport_error.empty());
  CHECK(partial.delivered_bytes == bytes.size());
  CHECK(partial.content_range == "bytes 17-48/" + std::to_string(payload.size()));
  CHECK(std::equal(bytes.begin(), bytes.end(), payload.begin() + 17));

  auto const past_eof = client->range_get(route, payload.size(), bytes.size(), bytes.data());
  CHECK(past_eof.outcome.http_status == 416);
  CHECK(past_eof.outcome.transport_error.empty());
  CHECK(past_eof.delivered_bytes == 0);
}

TEST_CASE("s3_rdma AC1 control transport failure is a result rather than an exception",
          "[s3][rdma][client-seam][integration]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!ensure_minio_env()) { return; }

  auto client       = make_control_client(read_minio_env(), "http://127.0.0.1:1");
  auto const result = client->head(rx_route{"bucket", "unreachable"});
  CHECK(result.outcome.http_status == 0);
  CHECK_FALSE(result.outcome.transport_error.empty());
}

TEST_CASE("s3_rdma AC2 control calls make exactly one HTTP attempt",
          "[s3][rdma][client-seam][integration]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!ensure_minio_env()) { return; }
  auto const env = read_minio_env();

  SECTION("successful HEAD")
  {
    auto client = make_control_client(env);
    (void)client->head(rx_route{env.bucket, std::string{k_fixture_key}});
    CHECK(client->attempts_total() == 1);
  }
  SECTION("successful range GET")
  {
    auto client = make_control_client(env);
    std::array<std::uint8_t, 16> bytes{};
    (void)client->range_get(
      rx_route{env.bucket, std::string{k_fixture_key}}, 0, bytes.size(), bytes.data());
    CHECK(client->attempts_total() == 1);
  }
  SECTION("HTTP failure")
  {
    auto client = make_control_client(env);
    (void)client->head(rx_route{env.bucket, "step5-ac2-missing.bin"});
    CHECK(client->attempts_total() == 1);
  }
  SECTION("transport failure")
  {
    auto client = make_control_client(env, "http://127.0.0.1:1");
    std::array<std::uint8_t, 16> bytes{};
    (void)client->range_get(rx_route{"bucket", "unreachable"}, 0, bytes.size(), bytes.data());
    CHECK(client->attempts_total() == 1);
  }
}

TEST_CASE("s3_rdma AC3 one control client reuses its persistent connection",
          "[s3][rdma][client-seam][integration]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!ensure_minio_env()) { return; }

  auto const env = read_minio_env();
  auto client    = make_control_client(env);
  rx_route route{env.bucket, std::string{k_fixture_key}};
  std::array<std::uint8_t, 16> first{};
  std::array<std::uint8_t, 16> second{};
  auto const attempts_before    = client->attempts_total();
  auto const connections_before = client->connections_total();

  CHECK(client->range_get(route, 0, first.size(), first.data()).outcome.http_status == 206);
  CHECK(client->range_get(route, first.size(), second.size(), second.data()).outcome.http_status ==
        206);
  CHECK(client->attempts_total() - attempts_before == 2);
  CHECK(client->connections_total() - connections_before <= 1);
}

TEST_CASE("s3_rdma AC7 data headers are signed and accepted on the wire",
          "[s3][rdma][client-seam][integration]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!ensure_minio_env()) { return; }

  auto const env  = read_minio_env();
  auto authorizer = make_header_authorizer(env);
  std::vector<std::pair<std::string, std::string>> extra_headers{
    {"x-amz-rdma-token", "step5-wire-token"}, {"Range", "bytes=0-15"}};
  auto request = authorizer->authorize_with_headers({env.bucket, std::string{k_fixture_key}},
                                                    sirius::io::s3::s3_request_method::GET,
                                                    30s,
                                                    extra_headers);

  auto const authorization =
    s3_rdma_client_seam_tests::header_value(request.headers, "Authorization");
  CHECK(authorization.find("range") != std::string::npos);
  CHECK(authorization.find("x-amz-rdma-token") != std::string::npos);
  CHECK(s3_rdma_client_seam_tests::header_value(request.headers, "Range") == "bytes=0-15");
  CHECK(s3_rdma_client_seam_tests::header_value(request.headers, "x-amz-rdma-token") ==
        "step5-wire-token");

  auto const response = perform_get(request);
  CHECK(response.status == 206);
  CHECK(response.body.size() == 16);
}

#endif

TEST_CASE("s3_rdma AC4 not-sent data errors do not poison the transport",
          "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "not-sent", payload);
  transport->data->script_result(
    data_get_result{data_commit_state::not_sent, 0, 0, {}, "request not sent"});
  auto ctx = make_started_ioctx(transport);
  auto ds  = open_ds(ctx, "not-sent");
  rmm::cuda_stream stream;
  rmm::device_buffer first(payload.size(), stream);

  auto failed = issue_device_read(*ds, first, stream);
  CHECK(ready_error(failed).find("not sent") != std::string::npos);
  CHECK(ctx->perf_snapshot().fail_stop_total == 0);
  CHECK(ctx->perf_snapshot().arena_leak_total == 0);

  rmm::device_buffer follow_up(payload.size(), stream);
  auto succeeded = issue_device_read(*ds, follow_up, stream);
  REQUIRE(succeeded.wait_for(5s) == std::future_status::ready);
  CHECK(succeeded.get() == payload.size());
}

TEST_CASE("s3_rdma AC4 sent-unknown data errors fail-stop", "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "sent-unknown", payload);
  transport->data->script_result(
    data_get_result{data_commit_state::sent_unknown, 0, 0, {}, "completion unknown"});
  auto ctx = make_started_ioctx(transport);
  auto ds  = open_ds(ctx, "sent-unknown");
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);

  auto failed = issue_device_read(*ds, destination, stream);
  CHECK(ready_error(failed).find("completion unknown") != std::string::npos);
  CHECK(ctx->perf_snapshot().fail_stop_total == 1);
  CHECK(ctx->perf_snapshot().arena_leak_total == 1);
}

TEST_CASE("s3_rdma AC4 completed data succeeds only with all authority legs",
          "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "completed", payload);
  transport->data->script_result(
    data_get_result{data_commit_state::completed, payload.size(), 200, "accepted-test-tag", {}});
  auto ctx = make_started_ioctx(transport, mock_config(), &accepts_test_tag);
  auto ds  = open_ds(ctx, "completed");
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);

  auto succeeded = issue_device_read(*ds, destination, stream);
  REQUIRE(succeeded.wait_for(5s) == std::future_status::ready);
  CHECK(succeeded.get() == payload.size());
  CHECK(ctx->perf_snapshot().fail_stop_total == 0);
}

TEST_CASE("s3_rdma AC5 completion authority validates tag bytes and status",
          "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  struct row {
    std::string_view name;
    data_get_result result;
    bool succeeds;
  };
  auto const expected = k_slot_size;
  std::vector<row> rows{
    {"missing tag", {data_commit_state::completed, expected, 200, {}, {}}, false},
    {"predicate rejected",
     {data_commit_state::completed, expected, 200, "rejected-test-tag", {}},
     false},
    {"short bytes",
     {data_commit_state::completed, expected - 1, 200, "accepted-test-tag", {}},
     false},
    {"over bytes",
     {data_commit_state::completed, expected + 1, 200, "accepted-test-tag", {}},
     false},
    {"negative status",
     {data_commit_state::completed, expected, 503, "accepted-test-tag", {}},
     false},
    {"all positive", {data_commit_state::completed, expected, 200, "accepted-test-tag", {}}, true},
  };

  for (auto const& test : rows) {
    DYNAMIC_SECTION(test.name)
    {
      auto payload   = pattern_bytes(expected);
      auto transport = seeded_mock_transport(std::string{k_bucket}, "authority", payload);
      transport->data->script_result(test.result);
      auto ctx = make_started_ioctx(transport, mock_config(), &accepts_test_tag);
      auto ds  = open_ds(ctx, "authority");
      rmm::cuda_stream stream;
      rmm::device_buffer destination(expected, stream);
      auto future = issue_device_read(*ds, destination, stream);

      if (test.succeeds) {
        REQUIRE(future.wait_for(5s) == std::future_status::ready);
        CHECK(future.get() == expected);
        CHECK(ctx->perf_snapshot().fail_stop_total == 0);
      } else {
        CHECK_FALSE(ready_error(future).empty());
        auto const gets_before_follow_up = transport->data->gets_issued();
        auto follow_up                   = issue_device_read(*ds, destination, stream);
        CHECK_FALSE(ready_error(follow_up).empty());
        auto const snapshot = ctx->perf_snapshot();
        CHECK(snapshot.fail_stop_total == 1);
        CHECK(snapshot.retries_total == 0);
        CHECK(snapshot.arena_leak_total == 1);
        CHECK(transport->data->gets_issued() == gets_before_follow_up);
      }
    }
  }
}

TEST_CASE("s3_rdma AC6 host reads use only the control plane", "[s3][rdma][client-seam]")
{
  using namespace s3_rdma_client_seam_tests;

  auto payload   = pattern_bytes(256);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "host-plane", payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = open_ds(ctx, "host-plane");
  std::array<std::uint8_t, 32> destination{};

  CHECK(ds->host_read(11, destination.size(), destination.data()) == destination.size());
  CHECK(transport->control->range_gets_issued() == 1);
  CHECK(transport->data->gets_issued() == 0);
  CHECK(std::equal(destination.begin(), destination.end(), payload.begin() + 11));
}

TEST_CASE("s3_rdma AC6 host retries are bounded and never poison the transport",
          "[s3][rdma][client-seam]")
{
  using namespace s3_rdma_client_seam_tests;

  SECTION("one transient 503 is retried and succeeds")
  {
    auto payload   = pattern_bytes(256);
    auto transport = seeded_mock_transport(std::string{k_bucket}, "host-transient", payload);
    auto ctx       = make_started_ioctx(transport);
    auto ds        = open_ds(ctx, "host-transient");
    std::array<std::uint8_t, 32> destination{};
    auto const calls_before = transport->control->range_gets_issued();
    transport->control->fail_next_n_range_gets(1, 503);

    CHECK(ds->host_read(0, destination.size(), destination.data()) == destination.size());
    CHECK(transport->control->range_gets_issued() - calls_before == 2);
    CHECK(std::equal(destination.begin(), destination.end(), payload.begin()));
    auto const snapshot = ctx->perf_snapshot();
    CHECK(snapshot.fail_stop_total == 0);
    CHECK(snapshot.retries_total == 0);
    CHECK(transport->data->gets_issued() == 0);
  }

  SECTION("three persistent 503 responses exhaust the retry budget without poisoning")
  {
    auto payload   = pattern_bytes(256);
    auto transport = seeded_mock_transport(std::string{k_bucket}, "host-exhaustion", payload);
    auto ctx       = make_started_ioctx(transport);
    auto ds        = open_ds(ctx, "host-exhaustion");
    std::array<std::uint8_t, 32> destination{};
    auto const calls_before = transport->control->range_gets_issued();
    transport->control->fail_next_n_range_gets(3, 503);

    CHECK_THROWS(ds->host_read(0, destination.size(), destination.data()));
    CHECK(transport->control->range_gets_issued() - calls_before == 3);
    auto snapshot = ctx->perf_snapshot();
    CHECK(snapshot.fail_stop_total == 0);
    CHECK(snapshot.retries_total == 0);

    auto const recovery_calls = transport->control->range_gets_issued();
    CHECK(ds->host_read(0, destination.size(), destination.data()) == destination.size());
    CHECK(transport->control->range_gets_issued() - recovery_calls == 1);
    CHECK(std::equal(destination.begin(), destination.end(), payload.begin()));
    snapshot = ctx->perf_snapshot();
    CHECK(snapshot.fail_stop_total == 0);
    CHECK(snapshot.retries_total == 0);
    CHECK(transport->data->gets_issued() == 0);
  }

  SECTION("a 404 is permanent and does not poison later reads")
  {
    auto payload   = pattern_bytes(256);
    auto transport = seeded_mock_transport(std::string{k_bucket}, "host-permanent", payload);
    auto ctx       = make_started_ioctx(transport);
    auto ds        = open_ds(ctx, "host-permanent");
    std::array<std::uint8_t, 32> destination{};
    auto const calls_before = transport->control->range_gets_issued();
    transport->control->fail_next_n_range_gets(1, 404);

    CHECK_THROWS(ds->host_read(0, destination.size(), destination.data()));
    CHECK(transport->control->range_gets_issued() - calls_before == 1);
    auto const snapshot = ctx->perf_snapshot();
    CHECK(snapshot.fail_stop_total == 0);
    CHECK(snapshot.retries_total == 0);
    CHECK(ds->host_read(0, destination.size(), destination.data()) == destination.size());
  }

  SECTION("a 416 is a consistency error (append-only violation)")
  {
    auto payload   = pattern_bytes(256);
    auto transport = seeded_mock_transport(std::string{k_bucket}, "host-empty", payload);
    auto ctx       = make_started_ioctx(transport);
    auto ds        = open_ds(ctx, "host-empty");
    std::array<std::uint8_t, 32> destination{};
    auto const calls_before = transport->control->range_gets_issued();
    transport->control->fail_next_n_range_gets(1, 416);

    CHECK_THROWS(ds->host_read(0, destination.size(), destination.data()));
    CHECK(transport->control->range_gets_issued() - calls_before == 1);
    auto const snapshot = ctx->perf_snapshot();
    CHECK(snapshot.fail_stop_total == 0);
    CHECK(snapshot.retries_total == 0);
  }
}

TEST_CASE("s3_rdma fail-stop closes both planes before diagnostics", "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "fatal-before-log", payload);
  transport->data->script_result(
    data_get_result{data_commit_state::sent_unknown, 0, 0, {}, "completion unknown"});
  auto ctx = make_started_ioctx(transport);
  auto ds  = open_ds(ctx, "fatal-before-log");
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);
  rmm::device_buffer rejected_destination(payload.size(), stream);

  auto blocking_sink = std::make_shared<blocking_error_log_sink>();
  blocking_sink->set_level(sirius::log::level::error);
  sirius::test::scoped_recording_log_sink restore_sink;
  sirius::log::set_sink(blocking_sink);
  release_log_sink_on_exit unblock{blocking_sink};

  auto failed = issue_device_read(*ds, destination, stream);
  REQUIRE(blocking_sink->wait_until_entered());
  CHECK(ctx->perf_snapshot().fail_stop_total == 1);

  auto const control_calls = transport->control->range_gets_issued();
  std::array<std::uint8_t, 16> host_destination{};
  CHECK_THROWS(ds->host_read(0, host_destination.size(), host_destination.data()));
  CHECK(transport->control->range_gets_issued() == control_calls);

  auto const data_calls = transport->data->gets_issued();
  auto rejected         = issue_device_read(*ds, rejected_destination, stream);
  CHECK_FALSE(ready_error(rejected).empty());
  CHECK(transport->data->gets_issued() == data_calls);

  blocking_sink->release();
  CHECK(ready_error(failed).find("completion unknown") != std::string::npos);
}

TEST_CASE("s3_rdma arena registrar lifetime follows the teardown outcome",
          "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  SECTION("normal teardown deregisters once and destroys every session")
  {
    auto payload   = pattern_bytes(k_slot_size);
    auto transport = seeded_mock_transport(std::string{k_bucket}, "registrar-normal", payload);
    {
      auto ctx = make_started_ioctx(transport);
      auto ds  = open_ds(ctx, "registrar-normal");
      rmm::cuda_stream stream;
      rmm::device_buffer destination(payload.size(), stream);
      auto completed = issue_device_read(*ds, destination, stream);

      REQUIRE(completed.wait_for(5s) == std::future_status::ready);
      CHECK(completed.get() == payload.size());
      CHECK(ctx->perf_snapshot().arena_leak_total == 0);
    }
    CHECK(transport->data->register_count() == 1);
    CHECK(transport->data->deregister_count() == 1);
    CHECK(transport->data->live_sessions() == 0);
  }

  SECTION("fail-stop teardown keeps the registrar and registration alive")
  {
    auto payload   = pattern_bytes(k_slot_size);
    auto transport = seeded_mock_transport(std::string{k_bucket}, "registrar-leaked", payload);
    transport->data->script_result(
      data_get_result{data_commit_state::sent_unknown, 0, 0, {}, "registration may be in use"});
    {
      auto ctx = make_started_ioctx(transport);
      auto ds  = open_ds(ctx, "registrar-leaked");
      rmm::cuda_stream stream;
      rmm::device_buffer destination(payload.size(), stream);
      auto failed = issue_device_read(*ds, destination, stream);

      CHECK(ready_error(failed).find("registration may be in use") != std::string::npos);
      auto const snapshot = ctx->perf_snapshot();
      CHECK(snapshot.fail_stop_total == 1);
      CHECK(snapshot.arena_leak_total >= 1);
    }
    CHECK(transport->data->register_count() == 1);
    CHECK(transport->data->deregister_count() == 0);
    CHECK(transport->data->live_sessions() > 0);
  }

  SECTION("registration failure never deregisters an unregistered arena")
  {
    auto payload   = pattern_bytes(k_slot_size);
    auto transport = seeded_mock_transport(std::string{k_bucket}, "registrar-failure", payload);
    transport->data->fail_register(1, "injected registration failure");
    {
      auto ctx = make_started_ioctx(transport);
      auto ds  = open_ds(ctx, "registrar-failure");
      rmm::cuda_stream stream;
      rmm::device_buffer destination(payload.size(), stream);
      auto failed = issue_device_read(*ds, destination, stream);

      CHECK(ready_error(failed).find("injected registration failure") != std::string::npos);
      auto const snapshot = ctx->perf_snapshot();
      CHECK(snapshot.fail_stop_total == 0);
      CHECK(snapshot.arena_leak_total == 0);
    }
    CHECK(transport->data->register_count() == 1);
    CHECK(transport->data->deregister_count() == 0);
    CHECK(transport->data->live_sessions() == 0);
  }
}

TEST_CASE("s3_rdma null data sessions surface capability errors", "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  SECTION("a null first worker session fails start without crashing")
  {
    auto transport = std::make_shared<mock_transport_fixture>();
    transport->data->null_acquire_after(0);
    auto ctx = std::make_shared<s3_rdma_ioctx>(
      mock_config(), transport->clients(), sirius::io::rdma::cuda_delivery_ops{});

    auto const error = thrown_error([&] { ctx->start(); });
    REQUIRE_FALSE(error.empty());
    CHECK(error.find("data-session acquisition failed") != std::string::npos);
    CHECK(error.find("no session") != std::string::npos);
    CHECK(transport->data->register_count() == 0);
    CHECK(transport->data->deregister_count() == 0);
    CHECK(transport->data->live_sessions() == 0);
  }

  SECTION("a null arena registrar fails the read without poisoning or crashing")
  {
    auto payload   = pattern_bytes(k_slot_size);
    auto transport = seeded_mock_transport(std::string{k_bucket}, "null-registrar", payload);
    transport->data->null_acquire_after(1);
    {
      auto ctx = make_started_ioctx(transport, mock_config(/*max_inflight=*/1));
      auto ds  = open_ds(ctx, "null-registrar");
      rmm::cuda_stream stream;
      rmm::device_buffer destination(payload.size(), stream);
      auto failed = issue_device_read(*ds, destination, stream);

      auto const error = ready_error(failed);
      CHECK(error.find("no session for arena registration") != std::string::npos);
      auto const snapshot = ctx->perf_snapshot();
      CHECK(snapshot.fail_stop_total == 0);
      CHECK(snapshot.arena_leak_total == 0);
      CHECK(transport->data->register_count() == 0);
      CHECK(transport->data->deregister_count() == 0);
    }
    CHECK(transport->data->live_sessions() == 0);
  }
}

TEST_CASE("s3_rdma AC10 token-bearing diagnostics are redacted at publication",
          "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  constexpr std::string_view sentinel = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef";
  auto payload                        = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "redaction", payload);
  transport->data->script_result(data_get_result{
    data_commit_state::sent_unknown,
    0,
    0,
    {},
    "gateway error; x-amz-rdma-token: " + std::string{sentinel} + "; completion unknown"});
  sirius::test::scoped_recording_log_sink logs{"trace"};
  auto ctx = make_started_ioctx(transport);
  auto ds  = open_ds(ctx, "redaction");
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);

  auto future      = issue_device_read(*ds, destination, stream);
  auto const error = ready_error(future);
  CHECK(error.find("x-amz-rdma-token") != std::string::npos);
  require_no_long_fragment(error, sentinel);

  auto const records = logs.records();
  REQUIRE(std::any_of(records.begin(), records.end(), [](auto const& record) {
    return record.message.find("x-amz-rdma-token") != std::string::npos;
  }));
  for (auto const& record : records) {
    require_no_long_fragment(record.message, sentinel);
  }
}

TEST_CASE("s3_rdma AC13 start acquires exactly one data session per worker",
          "[s3][rdma][client-seam]")
{
  using namespace s3_rdma_client_seam_tests;
  constexpr std::size_t workers = 4;
  auto transport                = std::make_shared<mock_transport_fixture>();
  auto ctx                      = make_started_ioctx(transport, mock_config(workers));

  REQUIRE(wait_until([&] { return transport->data->acquired_total() == workers; }));
  auto const threads = transport->data->acquisition_thread_ids();
  CHECK(transport->data->acquired_total() == workers);
  REQUIRE(threads.size() == workers);
  std::set<std::thread::id> unique_threads(threads.begin(), threads.end());
  CHECK(unique_threads.size() == workers);
  ctx->shutdown();
}

TEST_CASE("s3_rdma thrown data GET conservatively fail-stops both planes",
          "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "throwing-get", payload);
  transport->data->throw_gets("injected throwing GET");
  auto ctx = make_started_ioctx(transport);
  auto ds  = open_ds(ctx, "throwing-get");
  rmm::cuda_stream stream;
  rmm::device_buffer first(payload.size(), stream);
  rmm::device_buffer second(payload.size(), stream);

  auto failed               = issue_device_read(*ds, first, stream);
  auto const terminal_error = ready_error(failed);
  REQUIRE_FALSE(terminal_error.empty());
  auto snapshot = ctx->perf_snapshot();
  CHECK(snapshot.fail_stop_total == 1);
  CHECK(snapshot.arena_leak_total >= 1);
  CHECK(snapshot.retries_total == 0);

  auto const control_calls = transport->control->range_gets_issued();
  std::array<std::uint8_t, 16> host_destination{};
  auto const host_error =
    thrown_error([&] { (void)ds->host_read(0, host_destination.size(), host_destination.data()); });
  CHECK(host_error == terminal_error);
  CHECK(transport->control->range_gets_issued() == control_calls);

  auto const data_calls = transport->data->gets_issued();
  auto rejected         = issue_device_read(*ds, second, stream);
  CHECK(ready_error(rejected) == terminal_error);
  CHECK(transport->data->gets_issued() == data_calls);
  CHECK(ctx->perf_snapshot().retries_total == 0);
}

TEST_CASE("s3_rdma thrown data GET redacts its token before fail-stop publication",
          "[s3][rdma][client-seam][gpu][security]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  constexpr std::string_view sentinel = "aaa.bbb.ccc-sentinel";
  constexpr std::string_view label    = "x-amz-rdma-token";
  auto payload                        = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "throwing-get-token", payload);
  transport->data->throw_gets("boom x-amz-rdma-token: " + std::string{sentinel});
  sirius::test::scoped_recording_log_sink logs{"trace"};
  auto ctx = make_started_ioctx(transport);
  auto ds  = open_ds(ctx, "throwing-get-token");
  rmm::cuda_stream stream;
  rmm::device_buffer first(payload.size(), stream);
  rmm::device_buffer second(payload.size(), stream);

  auto failed               = issue_device_read(*ds, first, stream);
  auto const terminal_error = ready_error(failed);
  CHECK(terminal_error.find(label) != std::string::npos);
  require_no_long_fragment(terminal_error, sentinel);
  auto const snapshot = ctx->perf_snapshot();
  CHECK(snapshot.fail_stop_total == 1);
  CHECK(snapshot.retries_total == 0);

  auto const control_calls = transport->control->range_gets_issued();
  std::array<std::uint8_t, 16> host_destination{};
  auto const host_error =
    thrown_error([&] { (void)ds->host_read(0, host_destination.size(), host_destination.data()); });
  CHECK(host_error == terminal_error);
  CHECK(host_error.find(label) != std::string::npos);
  require_no_long_fragment(host_error, sentinel);
  CHECK(transport->control->range_gets_issued() == control_calls);

  auto const data_calls   = transport->data->gets_issued();
  auto rejected           = issue_device_read(*ds, second, stream);
  auto const device_error = ready_error(rejected);
  CHECK(device_error == terminal_error);
  CHECK(device_error.find(label) != std::string::npos);
  require_no_long_fragment(device_error, sentinel);
  CHECK(transport->data->gets_issued() == data_calls);

  auto const records = logs.records();
  REQUIRE(std::any_of(records.begin(), records.end(), [label](auto const& record) {
    return record.message.find(label) != std::string::npos;
  }));
  for (auto const& record : records) {
    require_no_long_fragment(record.message, sentinel);
  }
}

TEST_CASE("s3_rdma rejects arena byte-size overflow at construction",
          "[s3][rdma][client-seam][config]")
{
  using namespace s3_rdma_client_seam_tests;

  auto cfg                    = mock_config(/*max_inflight=*/8);
  cfg.s3_rdma_arena_slot_size = std::numeric_limits<std::size_t>::max() / 4;
  auto transport              = std::make_shared<mock_transport_fixture>();
  bool caught_expected_type   = false;
  std::string overflow_error_message;
  try {
    auto ctx =
      std::make_shared<s3_rdma_ioctx>(std::move(cfg), transport->clients(), cuda_delivery_ops{});
    (void)ctx;
  } catch (std::overflow_error const& error) {
    caught_expected_type   = true;
    overflow_error_message = error.what();
  }

  REQUIRE(caught_expected_type);
  CHECK(overflow_error_message.find("arena") != std::string::npos);
}

TEST_CASE("s3_rdma injected completion status set is authoritative", "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "status-418-accepted", payload);
  transport->data->script_result(
    data_get_result{data_commit_state::completed, payload.size(), 418, "accepted-test-tag", {}});
  auto clients              = transport->clients(&accepts_test_tag);
  clients.accepted_statuses = {418};
  auto ctx =
    std::make_shared<s3_rdma_ioctx>(mock_config(), std::move(clients), cuda_delivery_ops{});
  ctx->start();
  auto ds = open_ds(ctx, "status-418-accepted");
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);

  auto completed = issue_device_read(*ds, destination, stream);
  REQUIRE(completed.wait_for(5s) == std::future_status::ready);
  CHECK(completed.get() == payload.size());
  CHECK(ctx->perf_snapshot().fail_stop_total == 0);
}

TEST_CASE("s3_rdma default completion status set accepts 206", "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "status-206", payload);
  transport->data->script_result(
    data_get_result{data_commit_state::completed, payload.size(), 206, "reply-tag", {}});
  auto ctx = make_started_ioctx(transport);
  auto ds  = open_ds(ctx, "status-206");
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);

  auto completed = issue_device_read(*ds, destination, stream);
  REQUIRE(completed.wait_for(5s) == std::future_status::ready);
  CHECK(completed.get() == payload.size());
  CHECK(ctx->perf_snapshot().fail_stop_total == 0);
}

TEST_CASE("s3_rdma default completion status set rejects 418", "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "status-418-rejected", payload);
  transport->data->script_result(
    data_get_result{data_commit_state::completed, payload.size(), 418, "reply-tag", {}});
  auto ctx = make_started_ioctx(transport);
  auto ds  = open_ds(ctx, "status-418-rejected");
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);

  auto failed = issue_device_read(*ds, destination, stream);
  CHECK(ready_error(failed).find("418") != std::string::npos);
  auto const snapshot = ctx->perf_snapshot();
  CHECK(snapshot.fail_stop_total == 1);
  CHECK(snapshot.arena_leak_total >= 1);
  CHECK(snapshot.retries_total == 0);
}

TEST_CASE("s3_rdma non-sticky arena allocation error is per-chunk and recoverable",
          "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "malloc-recovery", payload);
  auto calls     = std::make_shared<std::atomic<int>>(0);
  cuda_delivery_ops ops;
  ops.malloc_device = [calls](void** ptr, std::size_t bytes) {
    if (calls->fetch_add(1, std::memory_order_relaxed) == 0) { return cudaErrorMemoryAllocation; }
    return cudaMalloc(ptr, bytes);
  };
  auto ctx = std::make_shared<s3_rdma_ioctx>(mock_config(), transport->clients(), std::move(ops));
  ctx->start();
  auto ds = open_ds(ctx, "malloc-recovery");
  rmm::cuda_stream stream;
  rmm::device_buffer first(payload.size(), stream);
  rmm::device_buffer second(payload.size(), stream);

  auto failed = issue_device_read(*ds, first, stream);
  REQUIRE(failed.wait_for(5s) == std::future_status::ready);
  auto const error = thrown_error([&] { (void)failed.get(); });
  CHECK_FALSE(error.empty());
  CHECK(ctx->perf_snapshot().fail_stop_total == 0);
  CHECK(ctx->perf_snapshot().arena_leak_total == 0);
  CHECK(transport->data->gets_issued() == 0);

  auto completed = issue_device_read(*ds, second, stream);
  REQUIRE(completed.wait_for(5s) == std::future_status::ready);
  CHECK(completed.get() == payload.size());
  CHECK(calls->load(std::memory_order_relaxed) == 2);
  CHECK(transport->data->gets_issued() == 1);
  CHECK(ctx->perf_snapshot().fail_stop_total == 0);
}

TEST_CASE("s3_rdma token redaction removes the complete header value",
          "[s3][rdma][client-seam][security]")
{
  using sirius::io::rdma::redact_rdma_tokens;

  SECTION("JWT-style value")
  {
    auto const redacted = redact_rdma_tokens("x-amz-rdma-token: aaa.bbb.ccc\r\nnext: v");
    CHECK(redacted.find("x-amz-rdma-token") != std::string::npos);
    CHECK(redacted.find("[REDACTED]") != std::string::npos);
    CHECK(redacted.find("aaa") == std::string::npos);
    CHECK(redacted.find("bbb") == std::string::npos);
    CHECK(redacted.find("ccc") == std::string::npos);
    CHECK(redacted.find("next: v") != std::string::npos);
  }

  SECTION("value at end of string")
  {
    auto const redacted = redact_rdma_tokens("x-amz-rdma-token=YWJjZA==");
    CHECK(redacted.find("x-amz-rdma-token") != std::string::npos);
    CHECK(redacted.find("YWJjZA==") == std::string::npos);
    CHECK(redacted.find("[REDACTED]") != std::string::npos);
  }

  SECTION("base64-style value")
  {
    auto const redacted = redact_rdma_tokens("x-amz-rdma-token: YWJjZC8rXw==\nnext");
    CHECK(redacted.find("YWJjZC8rXw==") == std::string::npos);
    CHECK(redacted.find("[REDACTED]") != std::string::npos);
    CHECK(redacted.find("\nnext") != std::string::npos);
  }
}

TEST_CASE("s3_rdma host read rejects a mismatched Content-Range without poisoning",
          "[s3][rdma][client-seam]")
{
  using namespace s3_rdma_client_seam_tests;

  auto payload   = pattern_bytes(4096);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "content-range-mismatch", payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = open_ds(ctx, "content-range-mismatch");
  std::array<std::uint8_t, 32> destination{};
  transport->control->override_content_range("bytes 999-1998/4096");

  auto const error =
    thrown_error([&] { (void)ds->host_read(0, destination.size(), destination.data()); });
  CHECK_FALSE(error.empty());
  CHECK(error.find("Content-Range") != std::string::npos);
  auto snapshot = ctx->perf_snapshot();
  CHECK(snapshot.fail_stop_total == 0);
  CHECK(snapshot.retries_total == 0);

  CHECK(ds->host_read(0, destination.size(), destination.data()) == destination.size());
  CHECK(std::equal(destination.begin(), destination.end(), payload.begin()));
  snapshot = ctx->perf_snapshot();
  CHECK(snapshot.fail_stop_total == 0);
  CHECK(snapshot.retries_total == 0);
}

TEST_CASE("s3_rdma host read rejects an unstructured Content-Range", "[s3][rdma][client-seam]")
{
  s3_rdma_client_seam_tests::require_rejected_content_range("content-range-garbage", "garbage");
}

TEST_CASE("s3_rdma host read rejects a case-mismatched Content-Range unit",
          "[s3][rdma][client-seam]")
{
  s3_rdma_client_seam_tests::require_rejected_content_range("content-range-case",
                                                            "Bytes 0-99/4096");
}

TEST_CASE("s3_rdma host read rejects a malformed Content-Range interval", "[s3][rdma][client-seam]")
{
  s3_rdma_client_seam_tests::require_rejected_content_range("content-range-invalid",
                                                            "bytes invalid");
}

TEST_CASE("s3_rdma host read accepts a response without Content-Range", "[s3][rdma][client-seam]")
{
  using namespace s3_rdma_client_seam_tests;

  auto payload   = pattern_bytes(256);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "content-range-absent", payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = open_ds(ctx, "content-range-absent");
  std::array<std::uint8_t, 32> destination{};
  transport->control->override_content_range("");

  CHECK(ds->host_read(0, destination.size(), destination.data()) == destination.size());
  CHECK(std::equal(destination.begin(), destination.end(), payload.begin()));
  auto const snapshot = ctx->perf_snapshot();
  CHECK(snapshot.fail_stop_total == 0);
  CHECK(snapshot.retries_total == 0);
}

TEST_CASE("s3_rdma HEAD 200 without Content-Length cannot create a size-zero object",
          "[s3][rdma][client-seam]")
{
  using namespace s3_rdma_client_seam_tests;

  SECTION("the control client reports the missing length as a transport error")
  {
    head_without_length_server server;
    curl_s3_control_client client{make_loopback_authorizer(server.endpoint()), "", false};

    auto const result = client.head(rx_route{std::string{k_bucket}, "missing-content-length"});
    server.wait();

    CHECK(result.outcome.http_status == 200);
    CHECK_FALSE(result.outcome.transport_ok());
    CHECK(result.outcome.transport_error.find("Content-Length") != std::string::npos);
    CHECK(result.object_size == 0);
  }

  SECTION("the RDMA ioctx refuses to open the object")
  {
    head_without_length_server server;
    auto control = std::make_shared<curl_s3_control_client>(
      make_loopback_authorizer(server.endpoint()), "", false);
    auto transport = std::make_shared<mock_transport_fixture>();
    sirius::io::rdma::rdma_transport_clients clients{control, transport->data};
    auto ctx = std::make_shared<s3_rdma_ioctx>(
      mock_config(), std::move(clients), sirius::io::rdma::cuda_delivery_ops{});
    ctx->start();

    auto const error =
      thrown_error([&] { (void)ctx->open_datasource("s3://bucket/missing-content-length"); });
    server.wait();

    REQUIRE_FALSE(error.empty());
    CHECK(error.find("Content-Length") != std::string::npos);
    CHECK(control->attempts_total() == 1);
    ctx->shutdown();
  }
}

TEST_CASE("s3_rdma flush policy is immutable after reactor start", "[s3][rdma][client-seam]")
{
  using namespace s3_rdma_client_seam_tests;

  cuobj_rdma_reactor::config cfg;
  cfg.max_inflight    = 1;
  cfg.arena_slot_size = k_slot_size;
  auto transport      = std::make_shared<mock_transport_fixture>();
  auto context        = std::make_shared<cuobj_rdma_reactor::reactor_context>(
    cfg, transport->clients(), cuda_delivery_ops{});
  context->set_flush_before_copy(true);
  CHECK(context->flush_before_copy());

  cuobj_rdma_reactor reactor{context};
  reactor.start();
  CHECK_THROWS_AS(context->set_flush_before_copy(false), std::logic_error);
  CHECK(context->flush_before_copy());
  reactor.shutdown();
}
