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
#include "io/s3/s3_blocking_ioctx.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/s3/s3_request_authorizer.hpp"
#include "io/s3/sirius_sigv4_authorizer.hpp"

#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <arpa/inet.h>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
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
using sirius::io::s3::s3_authorized_request;
using sirius::io::s3::s3_blocking_ioctx;
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
  std::string session_token;
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
                     env_or("SIRIUS_TEST_S3_SESSION_TOKEN"),
                     fs::path{local_dir},
                     truthy_env("SIRIUS_TEST_S3_STRICT")};
}

bool skip_if_no_s3_env(std::optional<s3_test_env> const& env)
{
  if (env) { return false; }
  WARN("Skipping async-curl S3 test because SIRIUS_TEST_S3_* is not configured");
  return true;
}

std::optional<s3_test_env> read_aws_live_env()
{
  auto endpoint   = env_or("SIRIUS_TEST_S3_ENDPOINT");
  auto access_key = env_or("SIRIUS_TEST_S3_ACCESS_KEY");
  auto secret_key = env_or("SIRIUS_TEST_S3_SECRET_KEY");
  auto bucket     = env_or("SIRIUS_TEST_S3_BUCKET");
  auto token      = env_or("SIRIUS_TEST_S3_SESSION_TOKEN");

  if (endpoint.empty() || access_key.empty() || secret_key.empty() || bucket.empty()) {
    SUCCEED("SIRIUS_TEST_S3_* not set; skipping live AWS async-curl benchmark");
    return std::nullopt;
  }
  if (token.empty()) {
    FAIL("SIRIUS_TEST_S3_SESSION_TOKEN is required for live AWS async-curl benchmark");
  }

  return s3_test_env{std::move(endpoint),
                     env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
                     std::move(access_key),
                     std::move(secret_key),
                     std::move(bucket),
                     std::move(token),
                     fs::path{},
                     false};
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
  creds.session_token     = env.session_token;
  return std::make_shared<sirius_sigv4_presigned_authorizer>(
    std::move(creds), env.region, env.endpoint, 30min);
}

std::shared_ptr<s3_ioctx> make_async_ioctx(std::shared_ptr<s3_request_authorizer> provider,
                                           std::size_t max_connections = 4,
                                           long request_timeout_s      = 20)
{
  return std::make_shared<s3_ioctx>(
    std::move(provider), request_timeout_s, std::string{}, true, max_connections, nullptr);
}

std::shared_ptr<s3_ioctx> make_device_async_ioctx(
  std::shared_ptr<s3_request_authorizer> provider,
  cucascade::memory::fixed_size_host_memory_resource& host_mr,
  std::size_t max_connections = 4,
  long request_timeout_s      = 20)
{
  return std::make_shared<s3_ioctx>(
    std::move(provider), request_timeout_s, std::string{}, true, max_connections, &host_mr);
}

std::shared_ptr<s3_ioctx> make_live_async_ioctx(s3_test_env const& env,
                                                std::size_t max_connections = 4)
{
  return make_async_ioctx(make_live_provider(env), max_connections);
}

void require_cuda_success(cudaError_t status)
{
  INFO(cudaGetErrorString(status));
  REQUIRE(status == cudaSuccess);
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

std::vector<std::uint8_t> copy_device_to_host(rmm::device_buffer const& device, std::size_t bytes)
{
  REQUIRE(bytes <= device.size());
  std::vector<std::uint8_t> host(bytes);
  if (bytes > 0) {
    require_cuda_success(cudaMemcpy(host.data(), device.data(), bytes, cudaMemcpyDeviceToHost));
  }
  return host;
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

std::uint64_t millis(std::chrono::steady_clock::duration d)
{
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::milliseconds>(d).count());
}

std::chrono::steady_clock::duration median(std::vector<std::chrono::steady_clock::duration> values)
{
  REQUIRE_FALSE(values.empty());
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

struct device_read_bench_result {
  std::chrono::steady_clock::duration median_time{};
  std::uint64_t device_stream_sync_total{0};
  std::uint64_t device_peak_inflight{0};
};

void fill_device_buffer(rmm::device_buffer& device, int value, rmm::cuda_stream_view stream)
{
  if (device.size() == 0) return;
  require_cuda_success(cudaMemsetAsync(device.data(), value, device.size(), stream.value()));
  require_cuda_success(cudaStreamSynchronize(stream.value()));
}

std::size_t expected_chunks(std::size_t offset, std::size_t size, std::size_t file_size)
{
  if (size == 0 || offset >= file_size) return 0;
  size                             = std::min(size, file_size - offset);
  constexpr std::size_t chunk_size = sirius::io::CHUNK_SIZE;
  return (size + chunk_size - 1) / chunk_size;
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

std::string exception_message(std::exception_ptr ep)
{
  if (!ep) { return {}; }
  try {
    std::rethrow_exception(ep);
  } catch (std::exception const& e) {
    return e.what();
  }
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

enum class scripted_response_mode {
  normal,
  interior_range_200_full_body,
  wrong_content_range_start,
  body_longer_than_requested
};

struct range_fault_policy {
  std::size_t fail_first_gets{0};
  long fail_status{503};
  std::string fail_reason{"Service Unavailable"};
  std::optional<int> retry_after_seconds;
  bool always_fail{false};
  scripted_response_mode response_mode{scripted_response_mode::normal};
};

class range_http_server {
 public:
  explicit range_http_server(std::vector<std::uint8_t> object,
                             std::chrono::milliseconds delay = 0ms,
                             range_fault_policy fault        = {})
    : _object(std::move(object)), _delay(delay), _fault(std::move(fault))
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
  std::vector<std::chrono::steady_clock::time_point> get_times() const
  {
    std::lock_guard<std::mutex> lk(_get_times_mtx);
    return _get_times;
  }

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
        auto const get_index = _get_count.fetch_add(1, std::memory_order_relaxed) + 1;
        {
          std::lock_guard<std::mutex> lk(_get_times_mtx);
          _get_times.push_back(std::chrono::steady_clock::now());
        }
        auto active = _active_gets.fetch_add(1, std::memory_order_acq_rel) + 1;
        auto peak   = _peak_active_gets.load(std::memory_order_relaxed);
        while (active > peak &&
               !_peak_active_gets.compare_exchange_weak(peak, active, std::memory_order_relaxed)) {}

        if (_fault.always_fail || get_index <= _fault.fail_first_gets) {
          if (_delay > 0ms) { std::this_thread::sleep_for(_delay); }
          std::ostringstream out;
          out << "HTTP/1.1 " << _fault.fail_status << " " << _fault.fail_reason << "\r\n"
              << "Content-Length: 0\r\n";
          if (_fault.retry_after_seconds.has_value()) {
            out << "Retry-After: " << *_fault.retry_after_seconds << "\r\n";
          }
          out << "Connection: keep-alive\r\n\r\n";
          send_all(client, out.str());
          _active_gets.fetch_sub(1, std::memory_order_acq_rel);
          continue;
        }

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
        if (_fault.response_mode == scripted_response_mode::interior_range_200_full_body &&
            begin > 0) {
          std::string body(reinterpret_cast<char const*>(_object.data()), _object.size());
          std::ostringstream out;
          out << "HTTP/1.1 200 OK\r\n"
              << "Content-Length: " << body.size() << "\r\n"
              << "Connection: keep-alive\r\n\r\n"
              << body;
          send_all(client, out.str());
          _active_gets.fetch_sub(1, std::memory_order_acq_rel);
          continue;
        }

        std::string body(reinterpret_cast<char const*>(_object.data() + begin), end - begin + 1);
        auto content_range_begin = begin;
        auto content_range_end   = end;
        if (_fault.response_mode == scripted_response_mode::wrong_content_range_start) {
          content_range_begin = std::min(begin + 1, _object.size() - 1);
          content_range_end   = std::min(content_range_begin + body.size() - 1, _object.size() - 1);
        } else if (_fault.response_mode == scripted_response_mode::body_longer_than_requested &&
                   end + 1 < _object.size()) {
          body.push_back(static_cast<char>(_object[end + 1]));
          content_range_end = end + 1;
        }
        std::ostringstream out;
        out << "HTTP/1.1 206 Partial Content\r\n"
            << "Content-Length: " << body.size() << "\r\n"
            << "Content-Range: bytes " << content_range_begin << "-" << content_range_end << "/"
            << _object.size() << "\r\n"
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
  range_fault_policy _fault;
  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::atomic<bool> _stop{false};
  std::thread _thread;
  std::vector<std::thread> _clients;
  std::atomic<std::size_t> _active_gets{0};
  std::atomic<std::size_t> _peak_active_gets{0};
  std::atomic<std::size_t> _get_count{0};
  std::atomic<std::size_t> _accept_count{0};
  mutable std::mutex _get_times_mtx;
  std::vector<std::chrono::steady_clock::time_point> _get_times;
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

class get_throwing_then_fixed_authorizer final : public s3_request_authorizer {
 public:
  get_throwing_then_fixed_authorizer(std::string url, int throws_remaining)
    : _url(std::move(url)), _throws_remaining(throws_remaining)
  {
  }

  s3_authorized_request authorize(s3_object_ref const&,
                                  s3_request_method method,
                                  std::chrono::seconds) override
  {
    if (method == s3_request_method::GET && _throws_remaining.fetch_sub(1) > 0) {
      throw std::runtime_error("scripted authorizer failure");
    }
    return s3_authorized_request{_url, {}};
  }

 private:
  std::string _url;
  std::atomic<int> _throws_remaining;
};

std::shared_ptr<s3_ioctx> make_scripted_async_ioctx(range_http_server& server,
                                                    std::size_t max_connections = 4,
                                                    long request_timeout_s      = 20)
{
  return make_async_ioctx(
    std::make_shared<fixed_url_authorizer>(server.url()), max_connections, request_timeout_s);
}

std::shared_ptr<s3_blocking_ioctx> make_device_blocking_ioctx(
  std::shared_ptr<s3_request_authorizer> provider,
  fsmr_test_resources& memory,
  std::size_t max_connections = 16,
  long request_timeout_s      = 20)
{
  s3_ioctx_config cfg{};
  cfg.creds                = std::move(provider);
  cfg.max_connections      = max_connections;
  cfg.request_timeout_s    = request_timeout_s;
  cfg.host_memory_resource = &memory.host_mr;
  return std::make_shared<s3_blocking_ioctx>(std::move(cfg));
}

std::shared_ptr<s3_ioctx> make_perf_device_async_ioctx(
  std::shared_ptr<s3_request_authorizer> provider,
  fsmr_test_resources& memory,
  bool enable_perf_instrumentation,
  std::size_t max_connections = 4,
  long request_timeout_s      = 20)
{
  s3_ioctx_config cfg{};
  cfg.creds                   = std::move(provider);
  cfg.max_connections         = max_connections;
  cfg.request_timeout_s       = request_timeout_s;
  cfg.host_memory_resource    = &memory.host_mr;
  cfg.s3_perf_instrumentation = enable_perf_instrumentation;
  return std::make_shared<s3_ioctx>(std::move(cfg));
}

std::vector<std::uint8_t> test_payload(std::size_t size = 4096)
{
  std::vector<std::uint8_t> out(size);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<std::uint8_t>((i * 17 + 3) % 251);
  }
  return out;
}

template <typename Ioctx>
std::chrono::steady_clock::duration time_device_read_once(Ioctx& ctx,
                                                          sirius::io::sirius_io_object& obj,
                                                          std::vector<std::uint8_t> const& payload,
                                                          std::size_t bytes)
{
  REQUIRE(bytes <= payload.size());
  auto stream = rmm::cuda_stream_default;
  rmm::device_buffer dst(bytes, stream);

  auto const start = std::chrono::steady_clock::now();
  auto const got   = ctx.device_read(obj, 0, bytes, static_cast<std::uint8_t*>(dst.data()), stream);
  auto const stop  = std::chrono::steady_clock::now();

  REQUIRE(got == bytes);
  auto const host = copy_device_to_host(dst, got);
  CHECK(std::equal(host.begin(), host.end(), payload.begin()));
  return stop - start;
}

template <typename Ioctx>
std::pair<std::chrono::steady_clock::duration, std::vector<std::uint8_t>> time_device_read_to_host(
  Ioctx& ctx, sirius::io::sirius_io_object& obj, std::size_t bytes)
{
  auto stream = rmm::cuda_stream_default;
  rmm::device_buffer dst(bytes, stream);

  auto const start = std::chrono::steady_clock::now();
  auto const got   = ctx.device_read(obj, 0, bytes, static_cast<std::uint8_t*>(dst.data()), stream);
  auto const stop  = std::chrono::steady_clock::now();

  CHECK(got == bytes);
  return {stop - start, copy_device_to_host(dst, got)};
}

device_read_bench_result run_async_device_benchmark(range_http_server& server,
                                                    std::vector<std::uint8_t> const& payload,
                                                    fsmr_test_resources& memory,
                                                    std::size_t max_connections,
                                                    std::size_t iterations = 3)
{
  auto ctx = make_device_async_ioctx(
    std::make_shared<fixed_url_authorizer>(server.url()), memory.host_mr, max_connections);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  (void)time_device_read_once(*ctx, *obj, payload, payload.size());  // warmup
  std::vector<std::chrono::steady_clock::duration> samples;
  samples.reserve(iterations);
  for (std::size_t i = 0; i < iterations; ++i) {
    samples.push_back(time_device_read_once(*ctx, *obj, payload, payload.size()));
  }

  return device_read_bench_result{
    median(samples), ctx->device_stream_sync_total(), ctx->device_peak_inflight()};
}

std::chrono::steady_clock::duration run_blocking_device_benchmark(
  range_http_server& server,
  std::vector<std::uint8_t> const& payload,
  fsmr_test_resources& memory,
  std::size_t iterations = 3)
{
  auto ctx = make_device_blocking_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                        memory,
                                        /*max_connections=*/16);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  (void)time_device_read_once(*ctx, *obj, payload, payload.size());  // warmup
  std::vector<std::chrono::steady_clock::duration> samples;
  samples.reserve(iterations);
  for (std::size_t i = 0; i < iterations; ++i) {
    samples.push_back(time_device_read_once(*ctx, *obj, payload, payload.size()));
  }
  return median(samples);
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

TEST_CASE("async-curl S3 retries transient async reads", "[.][s3][integration][asynccurl]")
{
  auto payload = test_payload(1024);

  SECTION("ranges async retries two 503s and succeeds")
  {
    range_fault_policy fault;
    fault.fail_first_gets = 2;
    fault.fail_status     = 503;
    fault.fail_reason     = "Service Unavailable";
    range_http_server server(payload, 0ms, std::move(fault));
    auto ctx = make_scripted_async_ioctx(server, 4);
    auto obj = ctx->create_io_object("s3://bucket/object.bin");

    std::vector<cudf::io::text::byte_range_info> ranges{{16, 32}, {128, 48}, {400, 64}};
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

    REQUIRE(fut.wait_for(5s) == std::future_status::ready);
    auto result = fut.get();
    CHECK(calls.load(std::memory_order_relaxed) == 1);
    REQUIRE(result.ep == nullptr);
    CHECK(result.bytes == 32 + 48 + 64);
    for (std::size_t i = 0; i < ranges.size(); ++i) {
      require_bytes_equal(buffers[i], payload, static_cast<std::size_t>(ranges[i].offset()));
    }
    CHECK(server.get_count() >= ranges.size() + 2);
  }

  SECTION("host_read_async retries two 503s and succeeds")
  {
    range_fault_policy fault;
    fault.fail_first_gets = 2;
    fault.fail_status     = 503;
    fault.fail_reason     = "Service Unavailable";
    range_http_server server(payload, 0ms, std::move(fault));
    auto ctx = make_scripted_async_ioctx(server, 4);
    auto obj = ctx->create_io_object("s3://bucket/object.bin");

    std::vector<std::uint8_t> out(40);
    auto fut = ctx->host_read_async(*obj, 200, out.size(), out.data());
    REQUIRE(fut.wait_for(5s) == std::future_status::ready);
    std::size_t bytes = 0;
    REQUIRE_NOTHROW(bytes = fut.get());
    CHECK(bytes == out.size());
    CHECK(std::equal(out.begin(), out.end(), payload.begin() + 200));
    CHECK(server.get_count() >= 3);
  }
}

TEST_CASE("async-curl S3 retry exhaustion reports the final transient failure",
          "[.][s3][integration][asynccurl]")
{
  range_fault_policy fault;
  fault.always_fail = true;
  fault.fail_status = 503;
  fault.fail_reason = "Service Unavailable";
  range_http_server server(test_payload(512), 0ms, std::move(fault));
  auto ctx = make_scripted_async_ioctx(server, 1);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  std::vector<cudf::io::text::byte_range_info> ranges{{32, 32}};
  std::vector<std::byte> out(32);
  std::vector<cudf::host_span<std::byte>> spans{{out.data(), out.size()}};

  auto before = std::chrono::steady_clock::now();
  auto fut    = read_ranges_async_future(*ctx, *obj, ranges, std::span{spans});
  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  auto elapsed = std::chrono::steady_clock::now() - before;
  auto result  = fut.get();

  REQUIRE(result.ep != nullptr);
  CHECK(exception_message(result.ep).find("503") != std::string::npos);
  CHECK(server.get_count() > ranges.size());
  CHECK(elapsed < 5s);
}

TEST_CASE("async-curl S3 honors Retry-After before retrying", "[.][s3][integration][asynccurl]")
{
  range_fault_policy fault;
  fault.fail_first_gets     = 1;
  fault.fail_status         = 503;
  fault.fail_reason         = "Service Unavailable";
  fault.retry_after_seconds = 1;
  range_http_server server(test_payload(512), 0ms, std::move(fault));
  auto ctx = make_scripted_async_ioctx(server, 1);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  std::vector<cudf::io::text::byte_range_info> ranges{{64, 32}};
  std::vector<std::byte> out(32);
  std::vector<cudf::host_span<std::byte>> spans{{out.data(), out.size()}};

  auto before = std::chrono::steady_clock::now();
  auto fut    = read_ranges_async_future(*ctx, *obj, ranges, std::span{spans});
  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  auto elapsed = std::chrono::steady_clock::now() - before;
  auto result  = fut.get();

  REQUIRE(result.ep == nullptr);
  CHECK(result.bytes == 32);
  CHECK(elapsed >= 900ms);
  CHECK(server.get_count() >= 2);
}

TEST_CASE("async-curl S3 does not retry non-retriable HTTP failures",
          "[.][s3][integration][asynccurl]")
{
  range_fault_policy fault;
  fault.always_fail = true;
  fault.fail_status = 404;
  fault.fail_reason = "Not Found";
  range_http_server server(test_payload(1024), 0ms, std::move(fault));
  auto ctx = make_scripted_async_ioctx(server, 4);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  std::vector<cudf::io::text::byte_range_info> ranges{{0, 16}, {32, 16}, {64, 16}};
  std::vector<std::vector<std::byte>> buffers;
  std::vector<cudf::host_span<std::byte>> spans;
  for (auto const& range : ranges) {
    buffers.emplace_back(static_cast<std::size_t>(range.size()));
    spans.emplace_back(buffers.back().data(), buffers.back().size());
  }

  auto fut = read_ranges_async_future(*ctx, *obj, ranges, std::span{spans});
  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  auto result = fut.get();
  REQUIRE(result.ep != nullptr);
  CHECK(exception_message(result.ep).find("404") != std::string::npos);
  CHECK(server.get_count() == ranges.size());
}

TEST_CASE("async-curl S3 sync host_read retries transient failures",
          "[.][s3][integration][asynccurl]")
{
  auto payload = test_payload(1024);
  range_fault_policy fault;
  fault.fail_first_gets = 2;
  fault.fail_status     = 503;
  fault.fail_reason     = "Service Unavailable";
  range_http_server server(payload, 0ms, std::move(fault));
  auto ctx = make_scripted_async_ioctx(server, 4);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  std::vector<std::uint8_t> got(48);
  std::size_t bytes = 0;
  REQUIRE_NOTHROW(bytes = ctx->host_read(*obj, 144, got.size(), got.data()));
  CHECK(bytes == got.size());
  CHECK(std::equal(got.begin(), got.end(), payload.begin() + 144));
  CHECK(server.get_count() >= 3);
}

TEST_CASE("async-curl S3 retry backoff spacing is observable", "[.][s3][integration][asynccurl]")
{
  range_fault_policy fault;
  fault.fail_first_gets = 2;
  fault.fail_status     = 503;
  fault.fail_reason     = "Service Unavailable";
  range_http_server server(test_payload(512), 0ms, std::move(fault));
  auto ctx = make_scripted_async_ioctx(server, 1);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  std::vector<std::uint8_t> out(16);
  auto fut = ctx->host_read_async(*obj, 80, out.size(), out.data());
  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  std::size_t bytes = 0;
  REQUIRE_NOTHROW(bytes = fut.get());
  CHECK(bytes == out.size());

  auto times = server.get_times();
  INFO("async-curl retry GET count=" << times.size());
  if (times.size() >= 3) {
    auto const first_gap =
      std::chrono::duration_cast<std::chrono::milliseconds>(times[1] - times[0]).count();
    auto const second_gap =
      std::chrono::duration_cast<std::chrono::milliseconds>(times[2] - times[1]).count();
    INFO("async-curl retry gaps ms: first=" << first_gap << " second=" << second_gap);
  }
}

TEST_CASE("async-curl S3 rejects malformed range responses", "[.][s3][integration][asynccurl]")
{
  auto payload = test_payload(512);

  SECTION("sync read rejects 200 full body for an interior range")
  {
    range_fault_policy fault;
    fault.response_mode = scripted_response_mode::interior_range_200_full_body;
    range_http_server server(payload, 0ms, std::move(fault));
    auto ctx = make_scripted_async_ioctx(server);
    auto obj = ctx->create_io_object("s3://bucket/object.bin");

    std::vector<std::uint8_t> got(16);
    CHECK_THROWS(ctx->host_read(*obj, 40, got.size(), got.data()));
  }

  SECTION("async read rejects 200 full body for an interior range")
  {
    range_fault_policy fault;
    fault.response_mode = scripted_response_mode::interior_range_200_full_body;
    range_http_server server(payload, 0ms, std::move(fault));
    auto ctx = make_scripted_async_ioctx(server);
    auto obj = ctx->create_io_object("s3://bucket/object.bin");

    std::vector<cudf::io::text::byte_range_info> ranges{{40, 16}};
    std::vector<std::byte> got(16);
    std::vector<cudf::host_span<std::byte>> spans{{got.data(), got.size()}};
    auto fut = read_ranges_async_future(*ctx, *obj, ranges, std::span{spans});
    REQUIRE(fut.wait_for(5s) == std::future_status::ready);
    auto result = fut.get();
    CHECK(result.ep != nullptr);
  }

  SECTION("sync read rejects a mismatched Content-Range start")
  {
    range_fault_policy fault;
    fault.response_mode = scripted_response_mode::wrong_content_range_start;
    range_http_server server(payload, 0ms, std::move(fault));
    auto ctx = make_scripted_async_ioctx(server);
    auto obj = ctx->create_io_object("s3://bucket/object.bin");

    std::vector<std::uint8_t> got(16);
    CHECK_THROWS(ctx->host_read(*obj, 40, got.size(), got.data()));
  }

  SECTION("async read rejects a mismatched Content-Range start")
  {
    range_fault_policy fault;
    fault.response_mode = scripted_response_mode::wrong_content_range_start;
    range_http_server server(payload, 0ms, std::move(fault));
    auto ctx = make_scripted_async_ioctx(server);
    auto obj = ctx->create_io_object("s3://bucket/object.bin");

    std::vector<cudf::io::text::byte_range_info> ranges{{40, 16}};
    std::vector<std::byte> got(16);
    std::vector<cudf::host_span<std::byte>> spans{{got.data(), got.size()}};
    auto fut = read_ranges_async_future(*ctx, *obj, ranges, std::span{spans});
    REQUIRE(fut.wait_for(5s) == std::future_status::ready);
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

TEST_CASE("async-curl S3 authorizer failures are reported without killing the reactor",
          "[.][s3][integration][asynccurl]")
{
  range_http_server server(test_payload(512));
  auto provider = std::make_shared<get_throwing_then_fixed_authorizer>(server.url(), 1);
  auto ctx      = make_async_ioctx(std::move(provider), 4);
  auto obj      = ctx->create_io_object("s3://bucket/object.bin");

  std::vector<cudf::io::text::byte_range_info> ranges{{32, 16}};
  std::vector<std::byte> failed(16);
  std::vector<cudf::host_span<std::byte>> failed_spans{{failed.data(), failed.size()}};
  auto failed_fut = read_ranges_async_future(*ctx, *obj, ranges, std::span{failed_spans});
  REQUIRE(failed_fut.wait_for(5s) == std::future_status::ready);
  auto failed_result = failed_fut.get();
  REQUIRE(failed_result.ep != nullptr);
  CHECK(exception_message(failed_result.ep).find("scripted authorizer failure") !=
        std::string::npos);

  std::vector<std::uint8_t> got(16);
  auto ok_fut = ctx->host_read_async(*obj, 64, got.size(), got.data());
  REQUIRE(ok_fut.wait_for(5s) == std::future_status::ready);
  std::size_t bytes = 0;
  REQUIRE_NOTHROW(bytes = ok_fut.get());
  CHECK(bytes == got.size());
}

TEST_CASE("async-curl S3 sync device read copies a single chunk without stream sync",
          "[.][s3][integration][asynccurl]")
{
  auto payload = test_payload(4096);
  range_http_server server(payload);
  fsmr_test_resources memory(/*block_size=*/1 << 20,
                             /*capacity=*/4 << 20,
                             /*memory_limit=*/4 << 20,
                             /*pool_size=*/4,
                             /*initial_pools=*/1);
  auto ctx = make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                     memory.host_mr,
                                     /*max_connections=*/2);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  auto const offset = std::size_t{123};
  auto const size   = std::size_t{2048};
  auto before_fsmr  = ctx->fsmr_borrows_total();
  auto stream       = rmm::cuda_stream_default;
  rmm::device_buffer dst(size, stream);

  auto const got =
    ctx->device_read(*obj, offset, size, static_cast<std::uint8_t*>(dst.data()), stream);

  CHECK(got == size);
  require_bytes_equal(copy_device_to_host(dst, got), payload, offset);
  CHECK(ctx->device_copies_total() == 1);
  CHECK(ctx->device_stream_sync_total() == 0);
  CHECK(ctx->fsmr_borrows_total() == before_fsmr + 1);
}

TEST_CASE("async-curl S3 multi-chunk device read uses async copies without per-chunk sync",
          "[.][s3][integration][asynccurl]")
{
  constexpr std::size_t payload_size = 3 * sirius::io::CHUNK_SIZE + 123;
  auto payload                       = test_payload(payload_size);
  range_http_server server(payload);
  fsmr_test_resources memory(/*block_size=*/sirius::io::CHUNK_SIZE,
                             /*capacity=*/8 * sirius::io::CHUNK_SIZE,
                             /*memory_limit=*/8 * sirius::io::CHUNK_SIZE,
                             /*pool_size=*/8,
                             /*initial_pools=*/1);
  auto ctx = make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                     memory.host_mr,
                                     /*max_connections=*/4);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  auto const chunks      = expected_chunks(0, payload.size(), payload.size());
  auto const before_fsmr = ctx->fsmr_borrows_total();
  REQUIRE(chunks >= 3);
  auto stream = rmm::cuda_stream_default;
  rmm::device_buffer dst(payload.size(), stream);

  auto const got =
    ctx->device_read(*obj, 0, payload.size(), static_cast<std::uint8_t*>(dst.data()), stream);

  CHECK(got == payload.size());
  require_bytes_equal(copy_device_to_host(dst, got), payload, 0);
  CHECK(ctx->device_copies_total() == chunks);
  CHECK(ctx->device_stream_sync_total() == 0);
  CHECK(ctx->fsmr_borrows_total() == before_fsmr + chunks);
}

TEST_CASE("async-curl S3 async device read resolves once and copies bytes",
          "[.][s3][integration][asynccurl]")
{
  auto payload = test_payload(8192);
  range_http_server server(payload);
  fsmr_test_resources memory(/*block_size=*/1 << 20,
                             /*capacity=*/2 << 20,
                             /*memory_limit=*/2 << 20,
                             /*pool_size=*/2,
                             /*initial_pools=*/1);
  auto ctx = make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                     memory.host_mr,
                                     /*max_connections=*/2);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  auto const offset = std::size_t{321};
  auto const size   = std::size_t{1024};
  auto stream       = rmm::cuda_stream_default;
  rmm::device_buffer dst(size, stream);
  std::atomic<int> calls{0};
  std::promise<async_read_result> done;
  auto fut = done.get_future();

  ctx->device_read_async_io(
    *obj, offset, size, static_cast<std::uint8_t*>(dst.data()), stream, [&](auto bytes, auto ep) {
      calls.fetch_add(1, std::memory_order_relaxed);
      done.set_value({bytes, ep});
    });

  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  auto result = fut.get();
  CHECK(calls.load(std::memory_order_relaxed) == 1);
  REQUIRE(result.ep == nullptr);
  CHECK(result.bytes == size);
  require_bytes_equal(copy_device_to_host(dst, result.bytes), payload, offset);
}

TEST_CASE("async-curl S3 device GET retries transient failures", "[.][s3][integration][asynccurl]")
{
  auto payload = test_payload(8192);
  range_fault_policy fault;
  fault.fail_first_gets = 2;
  fault.fail_status     = 503;
  fault.fail_reason     = "Service Unavailable";
  range_http_server server(payload, 0ms, std::move(fault));
  fsmr_test_resources memory(/*block_size=*/1 << 20,
                             /*capacity=*/2 << 20,
                             /*memory_limit=*/2 << 20,
                             /*pool_size=*/2,
                             /*initial_pools=*/1);
  auto ctx = make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                     memory.host_mr,
                                     /*max_connections=*/1);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  auto stream = rmm::cuda_stream_default;
  rmm::device_buffer dst(1024, stream);
  auto const got = ctx->device_read(*obj, 64, 1024, static_cast<std::uint8_t*>(dst.data()), stream);

  CHECK(got == 1024);
  require_bytes_equal(copy_device_to_host(dst, got), payload, 64);
  CHECK(server.get_count() >= 3);
  CHECK(ctx->device_stream_sync_total() == 0);
}

TEST_CASE("async-curl S3 device retry releases staging during backoff",
          "[.][s3][integration][asynccurl]")
{
  constexpr std::size_t payload_size = 3 * sirius::io::CHUNK_SIZE;
  auto payload                       = test_payload(payload_size);
  range_fault_policy fault;
  fault.fail_first_gets     = 1;
  fault.fail_status         = 503;
  fault.fail_reason         = "Service Unavailable";
  fault.retry_after_seconds = 1;
  range_http_server server(payload, 0ms, std::move(fault));
  fsmr_test_resources memory(/*block_size=*/sirius::io::CHUNK_SIZE,
                             /*capacity=*/sirius::io::CHUNK_SIZE,
                             /*memory_limit=*/sirius::io::CHUNK_SIZE,
                             /*pool_size=*/1,
                             /*initial_pools=*/1);
  auto before_free = memory.host_mr.get_free_blocks();
  auto ctx         = make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                     memory.host_mr,
                                     /*max_connections=*/1);
  auto obj         = ctx->create_io_object("s3://bucket/object.bin");

  auto stream = rmm::cuda_stream_default;
  rmm::device_buffer dst(payload.size(), stream);

  auto const got =
    ctx->device_read(*obj, 0, payload.size(), static_cast<std::uint8_t*>(dst.data()), stream);

  CHECK(got == payload.size());
  require_bytes_equal(copy_device_to_host(dst, got), payload, 0);
  CHECK(server.get_count() >= 4);
  CHECK(memory.host_mr.get_free_blocks() == before_free);
  CHECK(ctx->device_stream_sync_total() == 0);
}

TEST_CASE("async-curl S3 device staging peak stays within the active window",
          "[.][s3][integration][asynccurl]")
{
  constexpr std::size_t payload_size = 8 * sirius::io::CHUNK_SIZE;
  auto payload                       = test_payload(payload_size);
  range_http_server server(payload);
  fsmr_test_resources memory(/*block_size=*/sirius::io::CHUNK_SIZE,
                             /*capacity=*/16 * sirius::io::CHUNK_SIZE,
                             /*memory_limit=*/16 * sirius::io::CHUNK_SIZE,
                             /*pool_size=*/16,
                             /*initial_pools=*/1);
  auto ctx = make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                     memory.host_mr,
                                     /*max_connections=*/2);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  auto stream = rmm::cuda_stream_default;
  rmm::device_buffer dst(payload.size(), stream);

  auto const got =
    ctx->device_read(*obj, 0, payload.size(), static_cast<std::uint8_t*>(dst.data()), stream);

  CHECK(got == payload.size());
  require_bytes_equal(copy_device_to_host(dst, got), payload, 0);
  CHECK(ctx->device_peak_inflight() >= 1);
  CHECK(ctx->device_peak_inflight() <= 2);
  CHECK(ctx->device_stream_sync_total() == 0);
}

TEST_CASE("async-curl S3 benchmark hides injected range latency versus blocking S3",
          "[.][s3][bench]")
{
  constexpr std::size_t payload_size = 16 * sirius::io::CHUNK_SIZE;
  auto payload                       = test_payload(payload_size);
  range_http_server server(payload, 20ms);
  fsmr_test_resources memory(/*block_size=*/sirius::io::CHUNK_SIZE,
                             /*capacity=*/16 * sirius::io::CHUNK_SIZE,
                             /*memory_limit=*/16 * sirius::io::CHUNK_SIZE,
                             /*pool_size=*/16,
                             /*initial_pools=*/1);

  auto const blocking = run_blocking_device_benchmark(server, payload, memory);
  WARN("blocking median ms=" << millis(blocking));

  std::vector<std::pair<std::size_t, device_read_bench_result>> async_results;
  for (auto const max_connections :
       {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{16}}) {
    auto result = run_async_device_benchmark(server, payload, memory, max_connections);
    WARN("async mc=" << max_connections << " median ms=" << millis(result.median_time)
                     << " peak_inflight=" << result.device_peak_inflight
                     << " stream_syncs=" << result.device_stream_sync_total);
    CHECK(result.device_stream_sync_total == 0);
    CHECK(result.device_peak_inflight >= 1);
    CHECK(result.device_peak_inflight <= max_connections);
    CHECK(result.median_time * 10 <= blocking * 11);
    async_results.emplace_back(max_connections, result);
  }

  auto const& async_mc1  = async_results.front().second;
  auto const& async_best = async_results.back().second;
  CHECK(async_mc1.median_time * 2 < blocking * 3);
  CHECK(async_best.median_time < blocking / 2);
}

TEST_CASE("async-curl S3 perf instrumentation reports transient retry counters", "[.][s3][bench]")
{
  auto payload = test_payload(4096);
  range_fault_policy fault;
  fault.fail_first_gets = 1;
  fault.fail_status     = 503;
  fault.fail_reason     = "Service Unavailable";
  range_http_server server(payload, 0ms, std::move(fault));
  fsmr_test_resources memory(/*block_size=*/sirius::io::CHUNK_SIZE,
                             /*capacity=*/2 * sirius::io::CHUNK_SIZE,
                             /*memory_limit=*/2 * sirius::io::CHUNK_SIZE,
                             /*pool_size=*/2,
                             /*initial_pools=*/1);
  auto ctx = make_perf_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                          memory,
                                          /*enable_perf_instrumentation=*/true,
                                          /*max_connections=*/1);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  auto stream = rmm::cuda_stream_default;
  rmm::device_buffer dst(payload.size(), stream);
  auto const got =
    ctx->device_read(*obj, 0, payload.size(), static_cast<std::uint8_t*>(dst.data()), stream);

  CHECK(got == payload.size());
  require_bytes_equal(copy_device_to_host(dst, got), payload, 0);
  CHECK(ctx->retries_total() >= 1);
  CHECK(ctx->terminal_failures_total() == 0);
  WARN("s3 perf retry bucket=bucket key=object.bin attempts=" << server.get_count() << " retries="
                                                              << ctx->retries_total());
}

TEST_CASE("async-curl S3 benchmark reports real AWS S3 headline numbers",
          "[.][s3][bench][aws][live]")
{
  auto env = read_aws_live_env();
  if (!env) { return; }
  auto key = env_or("SIRIUS_ASYNCCURL_BENCH_S3_KEY");
  if (key.empty()) {
    SUCCEED("SIRIUS_ASYNCCURL_BENCH_S3_KEY not set; skipping live AWS async-curl benchmark");
    return;
  }

  fsmr_test_resources memory(/*block_size=*/sirius::io::CHUNK_SIZE,
                             /*capacity=*/16 * sirius::io::CHUNK_SIZE,
                             /*memory_limit=*/16 * sirius::io::CHUNK_SIZE,
                             /*pool_size=*/16,
                             /*initial_pools=*/1);
  auto blocking = make_device_blocking_ioctx(make_live_provider(*env), memory);
  auto async =
    make_device_async_ioctx(make_live_provider(*env), memory.host_mr, /*max_connections=*/8);

  auto blocking_obj = blocking->create_io_object(s3_uri(env->bucket, key));
  auto async_obj    = async->create_io_object(s3_uri(env->bucket, key));
  auto const bytes  = std::min<std::size_t>(blocking_obj->size(), 16 * sirius::io::CHUNK_SIZE);
  REQUIRE(bytes > 0);
  if (bytes < 2 * sirius::io::CHUNK_SIZE) {
    WARN("live AWS async-curl benchmark object is smaller than 2 chunks; key=" << key << " bytes="
                                                                               << bytes);
  }

  auto [blocking_time, blocking_bytes] = time_device_read_to_host(*blocking, *blocking_obj, bytes);
  auto [async_time, async_bytes]       = time_device_read_to_host(*async, *async_obj, bytes);

  CHECK(async_bytes == blocking_bytes);
  CHECK(async->device_stream_sync_total() == 0);
  WARN("aws key=" << key << " bytes=" << bytes << " blocking_ms=" << millis(blocking_time)
                  << " async_mc8_ms=" << millis(async_time)
                  << " async_peak_inflight=" << async->device_peak_inflight());
}

TEST_CASE("async-curl S3 device reads fail cleanly before copy on non-retriable GET errors",
          "[.][s3][integration][asynccurl]")
{
  auto payload = test_payload(2048);
  range_fault_policy fault;
  fault.always_fail = true;
  fault.fail_status = 404;
  fault.fail_reason = "Not Found";
  range_http_server server(payload, 0ms, std::move(fault));
  fsmr_test_resources memory(/*block_size=*/1 << 20,
                             /*capacity=*/2 << 20,
                             /*memory_limit=*/2 << 20,
                             /*pool_size=*/2,
                             /*initial_pools=*/1);
  auto ctx = make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                     memory.host_mr,
                                     /*max_connections=*/1);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  auto stream = rmm::cuda_stream_default;
  rmm::device_buffer dst(512, stream);
  fill_device_buffer(dst, 0x5A, stream);
  CHECK_THROWS(ctx->device_read(*obj, 0, 512, static_cast<std::uint8_t*>(dst.data()), stream));
  CHECK(ctx->device_stream_sync_total() == 0);
  auto got = copy_device_to_host(dst, dst.size());
  CHECK(std::all_of(got.begin(), got.end(), [](std::uint8_t b) { return b == 0x5A; }));

  range_fault_policy bad_range;
  bad_range.response_mode = scripted_response_mode::interior_range_200_full_body;
  range_http_server bad_range_server(payload, 0ms, std::move(bad_range));
  fsmr_test_resources async_memory(/*block_size=*/1 << 20,
                                   /*capacity=*/2 << 20,
                                   /*memory_limit=*/2 << 20,
                                   /*pool_size=*/2,
                                   /*initial_pools=*/1);
  auto async_ctx =
    make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(bad_range_server.url()),
                            async_memory.host_mr,
                            /*max_connections=*/1);
  auto async_obj = async_ctx->create_io_object("s3://bucket/object.bin");

  rmm::device_buffer async_dst(256, stream);
  std::atomic<int> calls{0};
  std::promise<async_read_result> done;
  auto fut = done.get_future();
  async_ctx->device_read_async_io(*async_obj,
                                  64,
                                  256,
                                  static_cast<std::uint8_t*>(async_dst.data()),
                                  stream,
                                  [&](auto bytes, auto ep) {
                                    calls.fetch_add(1, std::memory_order_relaxed);
                                    done.set_value({bytes, ep});
                                  });
  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  auto result = fut.get();
  CHECK(calls.load(std::memory_order_relaxed) == 1);
  CHECK(result.ep != nullptr);
  CHECK(async_ctx->device_stream_sync_total() == 0);
}

TEST_CASE("async-curl S3 device staging blocks are returned and reusable",
          "[.][s3][integration][asynccurl]")
{
  constexpr std::size_t payload_size = 3 * sirius::io::CHUNK_SIZE + 17;
  auto payload                       = test_payload(payload_size);
  range_http_server server(payload);
  fsmr_test_resources memory(/*block_size=*/sirius::io::CHUNK_SIZE,
                             /*capacity=*/4 * sirius::io::CHUNK_SIZE,
                             /*memory_limit=*/4 * sirius::io::CHUNK_SIZE,
                             /*pool_size=*/4,
                             /*initial_pools=*/1);
  auto before_free = memory.host_mr.get_free_blocks();
  auto ctx         = make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                     memory.host_mr,
                                     /*max_connections=*/2);
  auto obj         = ctx->create_io_object("s3://bucket/object.bin");

  auto stream = rmm::cuda_stream_default;
  for (int pass = 0; pass < 2; ++pass) {
    rmm::device_buffer dst(payload.size(), stream);
    auto const got =
      ctx->device_read(*obj, 0, payload.size(), static_cast<std::uint8_t*>(dst.data()), stream);
    CHECK(got == payload.size());
    require_bytes_equal(copy_device_to_host(dst, got), payload, 0);
    CHECK(memory.host_mr.get_free_blocks() == before_free);
  }
}

TEST_CASE("async-curl S3 device read guards missing FSMR, exhausted FSMR, and empty reads",
          "[.][s3][integration][asynccurl]")
{
  SECTION("no FSMR fails device read but leaves host reads usable")
  {
    auto payload = test_payload(1024);
    range_http_server server(payload);
    auto ctx = make_scripted_async_ioctx(server);
    auto obj = ctx->create_io_object("s3://bucket/object.bin");

    auto stream = rmm::cuda_stream_default;
    rmm::device_buffer dst(128, stream);
    try {
      (void)ctx->device_read(*obj, 0, 128, static_cast<std::uint8_t*>(dst.data()), stream);
      FAIL("expected device read without FSMR to fail");
    } catch (std::exception const& e) {
      CHECK(std::string{e.what()}.find("host memory resource") != std::string::npos);
    }

    std::vector<std::uint8_t> host(32);
    REQUIRE(ctx->host_read(*obj, 16, host.size(), host.data()) == host.size());
    CHECK(std::equal(host.begin(), host.end(), payload.begin() + 16));
  }

  SECTION("single-block FSMR streams multi-chunk device reads")
  {
    constexpr std::size_t payload_size = 3 * sirius::io::CHUNK_SIZE;
    auto payload                       = test_payload(payload_size);
    range_http_server server(payload, 50ms);
    fsmr_test_resources memory(/*block_size=*/sirius::io::CHUNK_SIZE,
                               /*capacity=*/sirius::io::CHUNK_SIZE,
                               /*memory_limit=*/sirius::io::CHUNK_SIZE,
                               /*pool_size=*/1,
                               /*initial_pools=*/1);
    auto before_free = memory.host_mr.get_free_blocks();
    auto ctx         = make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                       memory.host_mr,
                                       /*max_connections=*/1);
    auto obj         = ctx->create_io_object("s3://bucket/object.bin");

    auto const chunks      = expected_chunks(0, payload.size(), payload.size());
    auto const before_fsmr = ctx->fsmr_borrows_total();
    REQUIRE(chunks == 3);
    auto stream = rmm::cuda_stream_default;
    rmm::device_buffer dst(payload.size(), stream);

    auto const got =
      ctx->device_read(*obj, 0, payload.size(), static_cast<std::uint8_t*>(dst.data()), stream);

    CHECK(got == payload.size());
    require_bytes_equal(copy_device_to_host(dst, got), payload, 0);
    CHECK(memory.host_mr.get_free_blocks() == before_free);
    CHECK(ctx->fsmr_borrows_total() == before_fsmr + chunks);
    CHECK(ctx->device_stream_sync_total() == 0);
  }

  SECTION("misconfigured block smaller than chunk fails")
  {
    auto payload = test_payload(4096);
    range_http_server server(payload);
    fsmr_test_resources memory(/*block_size=*/2048,
                               /*capacity=*/4 * 2048,
                               /*memory_limit=*/4 * 2048,
                               /*pool_size=*/4,
                               /*initial_pools=*/1);
    auto ctx = make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                       memory.host_mr,
                                       /*max_connections=*/1);
    auto obj = ctx->create_io_object("s3://bucket/object.bin");

    auto stream = rmm::cuda_stream_default;
    rmm::device_buffer dst(payload.size(), stream);
    try {
      (void)ctx->device_read(
        *obj, 0, payload.size(), static_cast<std::uint8_t*>(dst.data()), stream);
      FAIL("expected block-size mismatch to fail");
    } catch (std::exception const& e) {
      CHECK(std::string{e.what()}.find("block size") != std::string::npos);
    }
  }

  SECTION("empty and past-EOF reads return zero without borrowing")
  {
    auto payload = test_payload(128);
    range_http_server server(payload);
    fsmr_test_resources memory(/*block_size=*/1 << 20,
                               /*capacity=*/1 << 20,
                               /*memory_limit=*/1 << 20,
                               /*pool_size=*/1,
                               /*initial_pools=*/1);
    auto ctx         = make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                       memory.host_mr,
                                       /*max_connections=*/1);
    auto obj         = ctx->create_io_object("s3://bucket/object.bin");
    auto before_fsmr = ctx->fsmr_borrows_total();
    auto stream      = rmm::cuda_stream_default;
    rmm::device_buffer dst(16, stream);

    CHECK(ctx->device_read(*obj, 0, 0, static_cast<std::uint8_t*>(dst.data()), stream) == 0);
    CHECK(ctx->device_read(
            *obj, payload.size(), 16, static_cast<std::uint8_t*>(dst.data()), stream) == 0);
    CHECK(ctx->fsmr_borrows_total() == before_fsmr);
  }
}

TEST_CASE("async-curl S3 shutdown resolves in-flight device reads exactly once",
          "[.][s3][integration][asynccurl]")
{
  auto payload = test_payload(2 * sirius::io::CHUNK_SIZE);
  range_http_server server(payload, 5s);
  fsmr_test_resources memory(/*block_size=*/sirius::io::CHUNK_SIZE,
                             /*capacity=*/2 * sirius::io::CHUNK_SIZE,
                             /*memory_limit=*/2 * sirius::io::CHUNK_SIZE,
                             /*pool_size=*/2,
                             /*initial_pools=*/1);
  auto ctx = make_device_async_ioctx(std::make_shared<fixed_url_authorizer>(server.url()),
                                     memory.host_mr,
                                     /*max_connections=*/1,
                                     /*request_timeout_s=*/30);
  auto obj = ctx->create_io_object("s3://bucket/object.bin");

  auto stream = rmm::cuda_stream_default;
  rmm::device_buffer dst(payload.size(), stream);
  std::atomic<int> calls{0};
  std::promise<async_read_result> done;
  auto fut = done.get_future();
  ctx->device_read_async_io(*obj,
                            0,
                            payload.size(),
                            static_cast<std::uint8_t*>(dst.data()),
                            stream,
                            [&](auto bytes, auto ep) {
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
  auto message = exception_message(result.ep);
  CHECK(message.find("handler resolved by safety net") == std::string::npos);
}

TEST_CASE("async-curl S3 sync host_read authorizer failures do not poison the ioctx",
          "[.][s3][integration][asynccurl]")
{
  auto payload = test_payload(512);
  range_http_server server(payload);
  auto provider = std::make_shared<get_throwing_then_fixed_authorizer>(server.url(), 1);
  auto ctx      = make_async_ioctx(std::move(provider), 4);
  auto obj      = ctx->create_io_object("s3://bucket/object.bin");

  std::vector<std::uint8_t> failed(16);
  try {
    (void)ctx->host_read(*obj, 32, failed.size(), failed.data());
    FAIL("expected sync host_read authorizer failure");
  } catch (std::exception const& e) {
    CHECK(std::string{e.what()}.find("scripted authorizer failure") != std::string::npos);
  }

  std::vector<std::uint8_t> got(16);
  REQUIRE(ctx->host_read(*obj, 64, got.size(), got.data()) == got.size());
  CHECK(std::equal(got.begin(), got.end(), payload.begin() + 64));
}

TEST_CASE("async-curl S3 ioctx is distinct from the blocking fallback",
          "[.][s3][integration][asynccurl]")
{
  auto provider = std::make_shared<mock_request_authorizer>(
    s3_authorized_request{"http://127.0.0.1:1/object", {}});
  s3_ioctx_config cfg{};
  cfg.creds           = provider;
  auto production_ctx = std::make_shared<s3_blocking_ioctx>(std::move(cfg));

  CHECK(dynamic_cast<s3_ioctx*>(production_ctx.get()) == nullptr);
}
