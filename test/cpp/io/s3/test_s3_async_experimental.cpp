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
#include "io/s3/mock_request_authorizer.hpp"
#include "io/s3/s3_async_experimental_ioctx.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/s3/s3_request_authorizer.hpp"
#include "io/s3/sirius_sigv4_authorizer.hpp"

#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

using sirius::io::s3::mock_request_authorizer;
using sirius::io::s3::s3_async_experimental_ioctx;
using sirius::io::s3::s3_authorized_request;
using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;
using sirius::io::s3::s3_object_ref;
using sirius::io::s3::s3_request_authorizer;
using sirius::io::s3::s3_request_method;
using sirius::io::s3::sirius_sigv4_presigned_authorizer;
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

bool skip_if_no_s3_env(std::optional<s3_test_env> const& env)
{
  if (env) { return false; }
  WARN("Skipping async-curl S3 test because SIRIUS_TEST_S3_* is not configured");
  return true;
}

std::string s3_uri(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string{bucket} + "/" + std::string{key};
}

std::vector<std::uint8_t> read_binary_file(fs::path const& path)
{
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::shared_ptr<s3_request_authorizer> make_live_provider(s3_test_env const& env)
{
  static_credentials creds;
  creds.access_key_id     = env.access_key;
  creds.secret_access_key = env.secret_key;
  return std::make_shared<sirius_sigv4_presigned_authorizer>(
    std::move(creds), env.region, env.endpoint, 30min);
}

std::shared_ptr<s3_async_experimental_ioctx> make_async_ioctx(
  std::shared_ptr<s3_request_authorizer> provider,
  std::size_t max_connections = 4,
  long request_timeout_s      = 20)
{
  return std::make_shared<s3_async_experimental_ioctx>(
    std::move(provider), request_timeout_s, std::string{}, true, max_connections, nullptr);
}

std::shared_ptr<s3_async_experimental_ioctx> make_live_async_ioctx(s3_test_env const& env,
                                                                   std::size_t max_connections = 4)
{
  return make_async_ioctx(make_live_provider(env), max_connections);
}

struct async_read_result {
  std::size_t bytes{0};
  std::exception_ptr ep;
};

std::future<async_read_result> read_ranges_async_future(
  sirius::io::sirius_ioctx& ctx,
  sirius::io::sirius_io_object& obj,
  std::vector<cudf::io::text::byte_range_info> const& ranges,
  std::span<cudf::host_span<std::byte>> dst)
{
  auto done = std::make_shared<std::promise<async_read_result>>();
  auto fut  = done->get_future();
  ctx.host_read_ranges_async_io(
    obj, ranges, dst, [done](auto bytes, auto ep) { done->set_value({bytes, ep}); });
  return fut;
}

async_read_result read_ranges_async(sirius::io::sirius_ioctx& ctx,
                                    sirius::io::sirius_io_object& obj,
                                    std::vector<cudf::io::text::byte_range_info> const& ranges,
                                    std::span<cudf::host_span<std::byte>> dst)
{
  return read_ranges_async_future(ctx, obj, ranges, dst).get();
}

void require_bytes_equal(std::vector<std::byte> const& got,
                         std::vector<std::uint8_t> const& expected,
                         std::size_t offset)
{
  REQUIRE(offset + got.size() <= expected.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == static_cast<std::byte>(expected[offset + i]));
  }
}

std::string request_header(std::string const& request, std::string const& header)
{
  std::istringstream in(request);
  std::string line;
  auto target = header;
  std::transform(target.begin(), target.end(), target.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') { line.pop_back(); }
    auto lowered = line;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (lowered.rfind(target + ":", 0) == 0) { return line.substr(header.size() + 1); }
  }
  return {};
}

bool parse_range(std::string const& request, std::size_t& begin, std::size_t& end)
{
  auto value = request_header(request, "Range");
  auto pos   = value.find("bytes=");
  if (pos == std::string::npos) { return false; }
  pos += 6;
  auto dash = value.find('-', pos);
  if (dash == std::string::npos) { return false; }
  begin = static_cast<std::size_t>(std::stoull(value.substr(pos, dash - pos)));
  end   = static_cast<std::size_t>(std::stoull(value.substr(dash + 1)));
  return true;
}

void send_all(int fd, std::string const& body)
{
  auto const* data      = body.data();
  std::size_t remaining = body.size();
  while (remaining > 0) {
    auto sent = ::send(fd, data, remaining, 0);
    if (sent <= 0) { return; }
    data += sent;
    remaining -= static_cast<std::size_t>(sent);
  }
}

class range_http_server {
 public:
  explicit range_http_server(std::vector<std::uint8_t> object,
                             std::chrono::milliseconds delay = 0ms)
    : _object(std::move(object)), _delay(delay)
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
    REQUIRE(::listen(_listen_fd, 64) == 0);

    sockaddr_in bound{};
    socklen_t len = sizeof(bound);
    REQUIRE(::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&bound), &len) == 0);
    _port = ntohs(bound.sin_port);

    _thread = std::thread([this] { serve(); });
  }

  ~range_http_server()
  {
    _stop.store(true, std::memory_order_relaxed);
    if (_listen_fd >= 0) {
      ::shutdown(_listen_fd, SHUT_RDWR);
      ::close(_listen_fd);
      _listen_fd = -1;
    }
    if (_thread.joinable()) { _thread.join(); }
    for (auto& t : _clients) {
      if (t.joinable()) { t.join(); }
    }
  }

  std::string url(std::string_view path = "/object") const
  {
    return "http://127.0.0.1:" + std::to_string(_port) + std::string{path};
  }

  std::size_t peak_active_gets() const noexcept { return _peak_active_gets.load(); }
  std::size_t get_count() const noexcept { return _get_count.load(); }
  std::size_t accept_count() const noexcept { return _accept_count.load(); }

 private:
  void serve()
  {
    while (!_stop.load(std::memory_order_relaxed)) {
      sockaddr_in client_addr{};
      socklen_t client_len = sizeof(client_addr);
      int client = ::accept(_listen_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
      if (client < 0) { return; }
      _accept_count.fetch_add(1, std::memory_order_relaxed);
      _clients.emplace_back([this, client] { handle_client(client); });
    }
  }

  void handle_client(int client)
  {
    while (!_stop.load(std::memory_order_relaxed)) {
      auto request = read_request(client);
      if (request.empty()) { break; }

      if (request.rfind("HEAD ", 0) == 0) {
        send_all(client,
                 "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size()) +
                   "\r\nConnection: keep-alive\r\n\r\n");
        continue;
      }

      if (request.rfind("GET ", 0) == 0) {
        _get_count.fetch_add(1, std::memory_order_relaxed);
        auto active = _active_gets.fetch_add(1, std::memory_order_acq_rel) + 1;
        auto peak   = _peak_active_gets.load(std::memory_order_relaxed);
        while (active > peak &&
               !_peak_active_gets.compare_exchange_weak(peak, active, std::memory_order_relaxed)) {}

        std::size_t begin = 0;
        std::size_t end   = _object.empty() ? 0 : _object.size() - 1;
        bool const ranged = parse_range(request, begin, end);
        if (!ranged || begin >= _object.size() || end < begin) {
          _active_gets.fetch_sub(1, std::memory_order_acq_rel);
          send_all(client,
                   "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: "
                   "close\r\n\r\n");
          break;
        }

        end = std::min(end, _object.size() - 1);
        if (_delay > 0ms) { std::this_thread::sleep_for(_delay); }
        std::string body(reinterpret_cast<char const*>(_object.data() + begin), end - begin + 1);
        std::ostringstream out;
        out << "HTTP/1.1 206 Partial Content\r\n"
            << "Content-Length: " << body.size() << "\r\n"
            << "Content-Range: bytes " << begin << "-" << end << "/" << _object.size() << "\r\n"
            << "Connection: keep-alive\r\n\r\n"
            << body;
        send_all(client, out.str());
        _active_gets.fetch_sub(1, std::memory_order_acq_rel);
        continue;
      }

      send_all(client,
               "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
      break;
    }
    ::shutdown(client, SHUT_RDWR);
    ::close(client);
  }

  std::string read_request(int client)
  {
    std::string request;
    std::array<char, 1024> buffer{};
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 8192) {
      auto got = ::recv(client, buffer.data(), buffer.size(), 0);
      if (got <= 0) { break; }
      request.append(buffer.data(), static_cast<std::size_t>(got));
    }
    return request;
  }

  std::vector<std::uint8_t> _object;
  std::chrono::milliseconds _delay;
  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::atomic<bool> _stop{false};
  std::thread _thread;
  std::vector<std::thread> _clients;
  std::atomic<std::size_t> _active_gets{0};
  std::atomic<std::size_t> _peak_active_gets{0};
  std::atomic<std::size_t> _get_count{0};
  std::atomic<std::size_t> _accept_count{0};
};

class fixed_url_authorizer final : public s3_request_authorizer {
 public:
  explicit fixed_url_authorizer(std::string url) : _url(std::move(url)) {}

  s3_authorized_request authorize(s3_object_ref const&,
                                  s3_request_method,
                                  std::chrono::seconds) override
  {
    return s3_authorized_request{_url, {}};
  }

 private:
  std::string _url;
};

std::shared_ptr<s3_async_experimental_ioctx> make_scripted_async_ioctx(
  range_http_server& server, std::size_t max_connections = 4, long request_timeout_s = 20)
{
  return make_async_ioctx(
    std::make_shared<fixed_url_authorizer>(server.url()), max_connections, request_timeout_s);
}

std::vector<std::uint8_t> test_payload(std::size_t size = 4096)
{
  std::vector<std::uint8_t> out(size);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<std::uint8_t>((i * 17 + 3) % 251);
  }
  return out;
}

}  // namespace

TEST_CASE("async-curl S3 empty range read may complete inline", "[.][s3][integration][asynccurl]")
{
  range_http_server server(test_payload(32));
  auto ctx = make_scripted_async_ioctx(server);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  std::vector<cudf::io::text::byte_range_info> ranges;
  std::vector<cudf::host_span<std::byte>> spans;
  std::atomic<int> calls{0};
  std::promise<async_read_result> done;
  auto fut = done.get_future();

  REQUIRE_NOTHROW(ctx->host_read_ranges_async_io(
    *obj, ranges, std::span<cudf::host_span<std::byte>>{spans}, [&](auto bytes, auto ep) {
      calls.fetch_add(1, std::memory_order_relaxed);
      done.set_value({bytes, ep});
    }));

  auto result = fut.get();
  CHECK(calls.load(std::memory_order_relaxed) == 1);
  CHECK(result.bytes == 0);
  CHECK(result.ep == nullptr);
}

TEST_CASE("async-curl S3 sync host_read returns exact bytes and clips EOF",
          "[.][s3][integration][asynccurl]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  std::string const key = "parquet/nation.parquet";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() > 128);

  auto ctx = make_live_async_ioctx(*env);
  auto obj = ctx->create_io_object(s3_uri(env->bucket, key));
  REQUIRE(obj->size() == local.size());

  std::vector<std::uint8_t> got(32);
  CHECK(ctx->host_read(*obj, 4, got.size(), got.data()) == got.size());
  CHECK(std::equal(got.begin(), got.end(), local.begin() + 4));

  std::vector<std::uint8_t> tail(64, 0);
  auto const offset = local.size() - 7;
  CHECK(ctx->host_read(*obj, offset, tail.size(), tail.data()) == 7);
  CHECK(std::equal(
    tail.begin(), tail.begin() + 7, local.begin() + static_cast<std::ptrdiff_t>(offset)));
}

TEST_CASE("async-curl S3 host_read_ranges_async fans out and fills every range",
          "[.][s3][integration][asynccurl]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  std::string const key = "parquet/nation.parquet";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() > 512);

  auto ctx = make_live_async_ioctx(*env, 4);
  auto obj = ctx->create_io_object(s3_uri(env->bucket, key));

  std::vector<cudf::io::text::byte_range_info> ranges{{0, 16}, {37, 24}, {101, 31}, {211, 64}};
  std::vector<std::vector<std::byte>> buffers;
  std::vector<cudf::host_span<std::byte>> spans;
  for (auto const& range : ranges) {
    buffers.emplace_back(static_cast<std::size_t>(range.size()));
    spans.emplace_back(buffers.back().data(), buffers.back().size());
  }

  auto result = read_ranges_async(*ctx, *obj, ranges, std::span{spans});
  REQUIRE(result.ep == nullptr);
  CHECK(result.bytes == 16 + 24 + 31 + 64);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    require_bytes_equal(buffers[i], local, static_cast<std::size_t>(ranges[i].offset()));
  }
}

TEST_CASE("async-curl S3 host_read_ranges_async reports validation errors through handler",
          "[.][s3][integration][asynccurl]")
{
  range_http_server server(test_payload(128));
  auto ctx = make_scripted_async_ioctx(server);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  SECTION("dst span too small")
  {
    std::vector<cudf::io::text::byte_range_info> ranges{{96, 16}};
    std::vector<std::byte> tiny(2);
    std::vector<cudf::host_span<std::byte>> spans{{tiny.data(), tiny.size()}};

    std::promise<async_read_result> done;
    auto fut = done.get_future();
    REQUIRE_NOTHROW(
      ctx->host_read_ranges_async_io(*obj, ranges, std::span{spans}, [&done](auto bytes, auto ep) {
        done.set_value({bytes, ep});
      }));
    auto result = fut.get();
    CHECK(result.ep != nullptr);
  }

  SECTION("ranges and dst size mismatch")
  {
    std::vector<cudf::io::text::byte_range_info> ranges{{0, 4}, {8, 4}};
    std::vector<std::byte> out(4);
    std::vector<cudf::host_span<std::byte>> spans{{out.data(), out.size()}};

    std::promise<async_read_result> done;
    auto fut = done.get_future();
    REQUIRE_NOTHROW(
      ctx->host_read_ranges_async_io(*obj, ranges, std::span{spans}, [&done](auto bytes, auto ep) {
        done.set_value({bytes, ep});
      }));
    auto result = fut.get();
    CHECK(result.ep != nullptr);
  }
}

TEST_CASE("async-curl S3 bounded submit respects max_connections",
          "[.][s3][integration][asynccurl]")
{
  constexpr std::size_t max_connections = 2;
  constexpr std::size_t range_count     = 8;
  auto payload                          = test_payload(2048);
  range_http_server server(payload, 75ms);
  auto ctx = make_scripted_async_ioctx(server, max_connections);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  std::vector<cudf::io::text::byte_range_info> ranges;
  std::vector<std::vector<std::byte>> buffers;
  std::vector<cudf::host_span<std::byte>> spans;
  for (std::size_t i = 0; i < range_count; ++i) {
    ranges.push_back({static_cast<int64_t>(i * 128), 64});
    buffers.emplace_back(64);
    spans.emplace_back(buffers.back().data(), buffers.back().size());
  }

  auto result = read_ranges_async(*ctx, *obj, ranges, std::span{spans});
  REQUIRE(result.ep == nullptr);
  CHECK(result.bytes == range_count * 64);
  CHECK(server.get_count() == range_count);
  CHECK(server.peak_active_gets() <= max_connections);
}

TEST_CASE("async-curl S3 single reactor completes many concurrent ranges without deadlock",
          "[.][s3][integration][asynccurl]")
{
  constexpr std::size_t range_count = 32;
  auto payload                      = test_payload(8192);
  range_http_server server(payload, 20ms);
  auto ctx = make_scripted_async_ioctx(server, 8);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  std::vector<cudf::io::text::byte_range_info> ranges;
  std::vector<std::vector<std::byte>> buffers;
  std::vector<cudf::host_span<std::byte>> spans;
  for (std::size_t i = 0; i < range_count; ++i) {
    ranges.push_back({static_cast<int64_t>(i * 128), 32});
    buffers.emplace_back(32);
    spans.emplace_back(buffers.back().data(), buffers.back().size());
  }

  auto fut = read_ranges_async_future(*ctx, *obj, ranges, std::span{spans});
  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  auto result = fut.get();
  REQUIRE(result.ep == nullptr);
  CHECK(result.bytes == range_count * 32);
}

TEST_CASE("async-curl S3 shutdown resolves queued and in-flight reads exactly once",
          "[.][s3][integration][asynccurl]")
{
  auto payload = test_payload(4096);
  range_http_server server(payload, 5s);
  auto ctx = make_scripted_async_ioctx(server, 1, 30);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  std::vector<cudf::io::text::byte_range_info> ranges{{0, 64}, {128, 64}, {256, 64}, {384, 64}};
  std::vector<std::vector<std::byte>> buffers;
  std::vector<cudf::host_span<std::byte>> spans;
  for (auto const& range : ranges) {
    buffers.emplace_back(static_cast<std::size_t>(range.size()));
    spans.emplace_back(buffers.back().data(), buffers.back().size());
  }

  std::atomic<int> calls{0};
  std::promise<async_read_result> done;
  auto fut = done.get_future();
  ctx->host_read_ranges_async_io(*obj, ranges, std::span{spans}, [&](auto bytes, auto ep) {
    calls.fetch_add(1, std::memory_order_relaxed);
    done.set_value({bytes, ep});
  });

  auto const before = std::chrono::steady_clock::now();
  ctx.reset();
  auto const elapsed = std::chrono::steady_clock::now() - before;

  REQUIRE(fut.wait_for(2s) == std::future_status::ready);
  auto result = fut.get();
  CHECK(calls.load(std::memory_order_relaxed) == 1);
  CHECK(result.ep != nullptr);
  CHECK(elapsed < 2s);

  try {
    std::rethrow_exception(result.ep);
  } catch (std::exception const& e) {
    std::string const message = e.what();
    CHECK(message.find("handler resolved by safety net") == std::string::npos);
  }
}

TEST_CASE("async-curl S3 keep-alive is observable but report-only",
          "[.][s3][integration][asynccurl]")
{
  auto payload = test_payload(512);
  range_http_server server(payload);
  auto ctx = make_scripted_async_ioctx(server, 4);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  for (std::size_t i = 0; i < 4; ++i) {
    std::vector<std::uint8_t> out(16);
    auto fut = ctx->host_read_async(*obj, i * 16, out.size(), out.data());
    REQUIRE(fut.get() == out.size());
  }

  INFO("async-curl accept_count=" << server.accept_count());
  CHECK(server.get_count() == 4);
}

TEST_CASE("async-curl S3 device reads are guarded until Phase 2", "[.][s3][integration][asynccurl]")
{
  range_http_server server(test_payload(256));
  auto ctx = make_scripted_async_ioctx(server);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  auto stream = rmm::cuda_stream_default;
  rmm::device_buffer dst(16, stream);
  CHECK_THROWS_AS(ctx->device_read(*obj, 0, 16, static_cast<std::uint8_t*>(dst.data()), stream),
                  std::logic_error);
  CHECK_THROWS_AS(
    ctx->device_read_async(*obj, 0, 16, static_cast<std::uint8_t*>(dst.data()), stream).get(),
    std::logic_error);
}

TEST_CASE("async-curl S3 experimental ioctx is isolated from production s3_ioctx",
          "[.][s3][integration][asynccurl]")
{
  auto provider = std::make_shared<mock_request_authorizer>(
    s3_authorized_request{"http://127.0.0.1:1/object", {}});
  s3_ioctx_config cfg{};
  cfg.creds           = provider;
  auto production_ctx = std::make_shared<s3_ioctx>(std::move(cfg));

  CHECK(dynamic_cast<s3_async_experimental_ioctx*>(production_ctx.get()) == nullptr);
}
