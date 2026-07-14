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
#include "io/rest/rest_ioctx.hpp"
#include "io/s3/s3_request_authorizer.hpp"
#include "io/sirius_datasource.hpp"
#include "io/types.hpp"
#include "memory/topology_index.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "utils/s3_container.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <config.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <log/logging.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <utils/log_test_utils.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using sirius::io::io_context_type;
using sirius::io::rest::rest_ioctx;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;
using namespace std::chrono_literals;

std::string env_or(std::string const& name, std::string fallback = {})
{
  if (auto* value = std::getenv(name.c_str()); value != nullptr) { return value; }
  return fallback;
}

std::string require_env(std::string const& name)
{
  auto value = env_or(name);
  REQUIRE_FALSE(value.empty());
  return value;
}

cucascade::memory::system_topology_info single_gpu_topology()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return topology;
}

std::shared_ptr<const sirius::memory::topology_index> single_gpu_index()
{
  return std::make_shared<sirius::memory::topology_index>(single_gpu_topology(),
                                                          std::vector<int>{0});
}

struct scan_manager_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  std::shared_ptr<const sirius::memory::topology_index> topology = single_gpu_index();
};

scan_manager_config make_minio_rest_config()
{
  scan_manager_config cfg{};
  cfg.use_sirius_datasource   = true;
  cfg.object_store.endpoint   = require_env("SIRIUS_TEST_S3_ENDPOINT");
  cfg.object_store.region     = env_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  cfg.object_store.access_key = require_env("SIRIUS_TEST_S3_ACCESS_KEY");
  cfg.object_store.secret_key = require_env("SIRIUS_TEST_S3_SECRET_KEY");
  cfg.object_store.tls_verify = false;
  cfg.rest.request_timeout_s  = 30;
  cfg.rest.max_connections    = 8;
  cfg.rest_n_reactors         = 1;
  return cfg;
}

scan_manager_config make_tls_minio_rest_config()
{
  auto cfg                        = make_minio_rest_config();
  cfg.object_store.endpoint       = require_env("SIRIUS_TEST_S3_HTTPS_ENDPOINT");
  cfg.object_store.tls_verify     = true;
  cfg.object_store.ca_bundle_path = require_env("SIRIUS_TEST_S3_CA_BUNDLE");
  return cfg;
}

scan_manager_config make_fake_rest_config(std::string endpoint)
{
  scan_manager_config cfg{};
  cfg.use_sirius_datasource        = true;
  cfg.object_store.endpoint        = std::move(endpoint);
  cfg.object_store.region          = "us-east-1";
  cfg.object_store.access_key      = "rest-integration-access-key";
  cfg.object_store.secret_key      = "rest-integration-secret-key";
  cfg.object_store.tls_verify      = false;
  cfg.rest.request_timeout_s       = 5;
  cfg.rest.max_connections         = 4;
  cfg.rest.max_retry_attempts      = 3;
  cfg.rest.max_auth_retry_attempts = 1;
  cfg.rest.retry_backoff_base      = std::chrono::milliseconds{5};
  cfg.rest.retry_jitter            = std::chrono::milliseconds{0};
  cfg.rest.honor_retry_after       = false;
  cfg.rest_n_reactors              = 1;
  cfg.enable_prefetch_cache        = false;
  return cfg;
}

std::filesystem::path project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path{SIRIUS_PROJECT_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path local_fixture_path(std::string const& key)
{
  return std::filesystem::path{require_env("SIRIUS_TEST_S3_LOCAL_DIR")} / key;
}

std::filesystem::path committed_parquet_fixture(std::string const& name)
{
  return project_root() / "test" / "cpp" / "integration" / "data" / "parquet" / name;
}

std::vector<std::uint8_t> read_binary_file(std::filesystem::path const& path)
{
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  std::vector<char> chars((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return std::vector<std::uint8_t>(chars.begin(), chars.end());
}

std::vector<std::uint8_t> deterministic_payload(std::size_t size)
{
  std::vector<std::uint8_t> out(size);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<std::uint8_t>((i * 131U + 17U) & 0xffU);
  }
  return out;
}

void require_bytes_equal(std::span<std::uint8_t const> got, std::span<std::uint8_t const> expected)
{
  REQUIRE(got.size() == expected.size());
  CHECK(std::equal(got.begin(), got.end(), expected.begin(), expected.end()));
}

std::vector<std::uint8_t> copy_device_to_host(rmm::device_buffer const& device,
                                              std::size_t size,
                                              rmm::cuda_stream_view stream)
{
  std::vector<std::uint8_t> out(size);
  REQUIRE(
    cudaMemcpyAsync(out.data(), device.data(), size, cudaMemcpyDeviceToHost, stream.value()) ==
    cudaSuccess);
  stream.synchronize();
  return out;
}

rest_ioctx* require_rest_ioctx(std::shared_ptr<sirius::io::sirius_datasource> const& ds)
{
  REQUIRE(ds != nullptr);
  REQUIRE(ds->io_ctx() != nullptr);
  CHECK(ds->io_ctx()->type() == io_context_type::restful);
  auto* rest_ctx = dynamic_cast<rest_ioctx*>(ds->io_ctx().get());
  REQUIRE(rest_ctx != nullptr);
  return rest_ctx;
}

std::uint32_t read_le32(std::span<std::uint8_t const> bytes)
{
  REQUIRE(bytes.size() >= 4);
  return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint32_t parquet_footer_len(std::span<std::uint8_t const> parquet)
{
  REQUIRE(parquet.size() >= 8);
  CHECK(parquet[parquet.size() - 4] == static_cast<std::uint8_t>('P'));
  CHECK(parquet[parquet.size() - 3] == static_cast<std::uint8_t>('A'));
  CHECK(parquet[parquet.size() - 2] == static_cast<std::uint8_t>('R'));
  CHECK(parquet[parquet.size() - 1] == static_cast<std::uint8_t>('1'));
  return read_le32(parquet.subspan(parquet.size() - 8, 4));
}

struct range_fault_policy {
  std::size_t fail_first_gets{0};
  bool fail_all_gets{false};
  std::size_t fail_first_heads{0};
  bool fail_all_heads{false};
  int fail_status{503};
  int fail_head_status{503};
  std::chrono::milliseconds response_delay{0};
  bool omit_content_range{false};
  bool unknown_content_range_total{false};
  bool ignore_range_with_200{false};
  bool fail_suffix_with_416{false};
};

class fixed_url_authorizer final : public sirius::io::s3::s3_request_authorizer {
 public:
  explicit fixed_url_authorizer(std::string endpoint) : _endpoint(std::move(endpoint)) {}

  sirius::io::s3::s3_authorized_request authorize(sirius::io::s3::s3_object_ref const& obj,
                                                  sirius::io::s3::s3_request_method /*method*/,
                                                  std::chrono::seconds /*timeout*/) override
  {
    return {_endpoint + "/" + obj.bucket + "/" + obj.key, {}};
  }

 private:
  std::string _endpoint;
};

class range_http_server {
 public:
  explicit range_http_server(std::vector<std::uint8_t> object, range_fault_policy fault = {})
    : _object(std::move(object)), _fault(fault)
  {
    REQUIRE_FALSE(_object.empty());

    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_listen_fd < 0) { throw std::runtime_error("socket failed: " + errno_message()); }
    int one = 1;
    if (::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
      throw std::runtime_error("setsockopt failed: " + errno_message());
    }

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    if (::bind(_listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      throw std::runtime_error("bind failed: " + errno_message());
    }
    if (::listen(_listen_fd, 64) != 0) {
      throw std::runtime_error("listen failed: " + errno_message());
    }

    socklen_t len = sizeof(addr);
    if (::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
      throw std::runtime_error("getsockname failed: " + errno_message());
    }
    _port   = ntohs(addr.sin_port);
    _thread = std::thread([this] { accept_loop(); });
  }

  ~range_http_server()
  {
    _stop.store(true);
    if (_listen_fd >= 0) {
      ::shutdown(_listen_fd, SHUT_RDWR);
      ::close(_listen_fd);
      _listen_fd = -1;
    }
    if (_thread.joinable()) { _thread.join(); }
    for (auto& w : _workers) {
      if (w.joinable()) { w.join(); }
    }
  }

  range_http_server(range_http_server const&)            = delete;
  range_http_server& operator=(range_http_server const&) = delete;

  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }
  [[nodiscard]] std::size_t head_count() const noexcept { return _head_count.load(); }
  [[nodiscard]] std::size_t get_count() const noexcept { return _get_count.load(); }
  [[nodiscard]] std::size_t body_bytes_sent() const noexcept { return _body_bytes_sent.load(); }
  [[nodiscard]] int peak_active_gets() const noexcept { return _peak_active_gets.load(); }

 private:
  struct active_get_guard {
    explicit active_get_guard(range_http_server& server) : _server(server)
    {
      int const active = _server._active_gets.fetch_add(1, std::memory_order_relaxed) + 1;
      int peak         = _server._peak_active_gets.load(std::memory_order_relaxed);
      while (active > peak && !_server._peak_active_gets.compare_exchange_weak(
                                peak, active, std::memory_order_relaxed)) {}
    }
    ~active_get_guard() { _server._active_gets.fetch_sub(1, std::memory_order_relaxed); }
    range_http_server& _server;
  };

  static std::string errno_message() { return std::strerror(errno); }

  void accept_loop()
  {
    while (!_stop.load()) {
      sockaddr_in client{};
      socklen_t len = sizeof(client);
      int fd        = ::accept(_listen_fd, reinterpret_cast<sockaddr*>(&client), &len);
      if (fd < 0) {
        if (_stop.load()) { return; }
        continue;
      }
      _workers.emplace_back([this, fd] {
        handle_client(fd);
        ::close(fd);
      });
    }
  }

  void handle_client(int fd)
  {
    timeval timeout{};
    timeout.tv_sec = 3;
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    std::string request(4096, '\0');
    ssize_t n = ::recv(fd, request.data(), request.size(), 0);
    if (n <= 0) { return; }
    request.resize(static_cast<std::size_t>(n));

    bool const is_head = request.rfind("HEAD ", 0) == 0;
    bool const is_get  = request.rfind("GET ", 0) == 0;
    if (is_head) {
      auto const head_idx = _head_count.fetch_add(1, std::memory_order_relaxed);
      if (_fault.fail_all_heads || head_idx < _fault.fail_first_heads) {
        std::string response =
          "HTTP/1.1 " + std::to_string(_fault.fail_head_status) +
          " Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        send_all(fd, response);
        return;
      }
      std::string response =
        "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size()) +
        "\r\nConnection: close\r\n\r\n";
      send_all(fd, response);
      return;
    }
    if (!is_get) {
      send_all(fd,
               "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
      return;
    }

    active_get_guard active{*this};
    if (_fault.response_delay.count() > 0) { std::this_thread::sleep_for(_fault.response_delay); }

    auto const get_idx = _get_count.fetch_add(1, std::memory_order_relaxed);
    if (_fault.fail_all_gets || get_idx < _fault.fail_first_gets) {
      std::string response =
        "HTTP/1.1 " + std::to_string(_fault.fail_status) +
        " Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
      send_all(fd, response);
      return;
    }

    if (auto range = parse_range(request)) {
      auto const [start, end] = *range;
      if (_fault.fail_suffix_with_416 && is_suffix_range(request)) {
        send_all(fd,
                 "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: "
                 "close\r\n\r\n");
        return;
      }
      if (_fault.ignore_range_with_200 && is_suffix_range(request)) {
        std::string response =
          "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size()) +
          "\r\nConnection: close\r\n\r\n";
        send_all(fd, response);
        send_body(fd, _object.data(), _object.size());
        return;
      }
      auto const len = end - start + 1;
      std::string response =
        "HTTP/1.1 206 Partial Content\r\nContent-Length: " + std::to_string(len);
      if (!_fault.omit_content_range) {
        response +=
          "\r\nContent-Range: bytes " + std::to_string(start) + "-" + std::to_string(end) + "/" +
          (_fault.unknown_content_range_total ? std::string{"*"} : std::to_string(_object.size()));
      }
      response += "\r\nConnection: close\r\n\r\n";
      send_all(fd, response);
      send_body(fd, _object.data() + start, len);
      return;
    }

    std::string response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size()) +
                           "\r\nConnection: close\r\n\r\n";
    send_all(fd, response);
    send_body(fd, _object.data(), _object.size());
  }

  [[nodiscard]] static bool is_suffix_range(std::string const& request)
  {
    std::string lower = request;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return lower.find("range: bytes=-") != std::string::npos;
  }

  void send_body(int fd, std::uint8_t const* bytes, std::size_t size)
  {
    _body_bytes_sent.fetch_add(send_all(fd, bytes, size), std::memory_order_relaxed);
  }

  static void send_all(int fd, std::string_view bytes)
  {
    (void)send_all(fd, reinterpret_cast<std::uint8_t const*>(bytes.data()), bytes.size());
  }

  static std::size_t send_all(int fd, std::uint8_t const* bytes, std::size_t size)
  {
    std::size_t sent = 0;
    while (sent < size) {
      ssize_t n = ::send(fd, bytes + sent, size - sent, MSG_NOSIGNAL);
      if (n <= 0) { return sent; }
      sent += static_cast<std::size_t>(n);
    }
    return sent;
  }

  [[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>> parse_range(
    std::string const& request) const
  {
    std::string lower = request;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    std::string const prefix = "range: bytes=";
    auto pos                 = lower.find(prefix);
    if (pos == std::string::npos) { return std::nullopt; }
    pos += prefix.size();
    auto const eol     = lower.find("\r\n", pos);
    auto const end_pos = eol == std::string::npos ? lower.size() : eol;
    auto const spec    = lower.substr(pos, end_pos - pos);
    auto const dash    = spec.find('-');
    if (dash == std::string::npos) { return std::nullopt; }
    try {
      std::size_t start = 0;
      std::size_t end   = _object.size() - 1;
      if (dash == 0) {
        auto const suffix = static_cast<std::size_t>(std::stoull(spec.substr(1)));
        if (suffix == 0) { return std::nullopt; }
        start = suffix >= _object.size() ? 0 : _object.size() - suffix;
      } else {
        start = static_cast<std::size_t>(std::stoull(spec.substr(0, dash)));
      }
      if (dash != 0 && dash + 1 < spec.size()) {
        end = static_cast<std::size_t>(std::stoull(spec.substr(dash + 1)));
      }
      if (start >= _object.size()) { return std::nullopt; }
      end = std::min(end, _object.size() - 1);
      if (end < start) { return std::nullopt; }
      return std::make_pair(start, end);
    } catch (...) {
      return std::nullopt;
    }
  }

  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::vector<std::uint8_t> _object;
  range_fault_policy _fault;
  std::atomic<bool> _stop{false};
  std::atomic<std::size_t> _head_count{0};
  std::atomic<std::size_t> _get_count{0};
  std::atomic<std::size_t> _body_bytes_sent{0};
  std::atomic<int> _active_gets{0};
  std::atomic<int> _peak_active_gets{0};
  std::thread _thread;
  std::vector<std::thread> _workers;
};

sirius::io::rest::config direct_rest_test_config()
{
  sirius::io::rest::config cfg{};
  cfg.request_timeout_s       = 5;
  cfg.max_connections         = 4;
  cfg.max_retry_attempts      = 2;
  cfg.max_auth_retry_attempts = 1;
  cfg.retry_backoff_base      = std::chrono::milliseconds{1};
  cfg.retry_jitter            = std::chrono::milliseconds{0};
  cfg.honor_retry_after       = false;
  return cfg;
}

std::shared_ptr<rest_ioctx> make_direct_rest_ioctx(std::string endpoint,
                                                   sirius::io::rest::config cfg)
{
  auto authorizer = std::make_shared<fixed_url_authorizer>(std::move(endpoint));
  auto ctx        = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
    cfg, std::move(authorizer), nullptr);
  auto ioctx = std::make_shared<rest_ioctx>(1, std::move(ctx));
  ioctx->start();
  return ioctx;
}

std::shared_ptr<rest_ioctx> make_direct_rest_ioctx(std::string endpoint)
{
  return make_direct_rest_ioctx(std::move(endpoint), direct_rest_test_config());
}

using capture_backend    = sirius::test::recording_log_backend;
using scoped_log_capture = sirius::test::scoped_recording_log_backend;

}  // namespace

TEST_CASE("rest footer suffix parses Content-Range totals", "[s3][integration][rest][footerbind]")
{
  using sirius::io::rest::content_range_total;

  CHECK(content_range_total("bytes 42-99/12345") == std::optional<std::size_t>{12345});
  CHECK(content_range_total("Bytes 0-7/8") == std::optional<std::size_t>{8});
  CHECK_FALSE(content_range_total("bytes 42-99/*").has_value());
  CHECK_FALSE(content_range_total("bytes */12345").has_value());
  CHECK_FALSE(content_range_total("not-a-content-range").has_value());
}

TEST_CASE("rest footer suffix probe discovers size without HEAD",
          "[s3][integration][rest][footerbind]")
{
  auto const parquet = read_binary_file(committed_parquet_fixture("nation.parquet"));
  range_http_server server(parquet);
  auto authorizer = std::make_shared<fixed_url_authorizer>(server.endpoint());
  sirius::io::rest::config cfg{};
  cfg.request_timeout_s       = 5;
  cfg.max_retry_attempts      = 1;
  cfg.max_auth_retry_attempts = 1;
  auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
    cfg, std::move(authorizer), nullptr);
  sirius::io::rest::rest_reactor reactor(ctx, "footer-suffix-test");

  auto probe = reactor.fetch_footer_suffix("footer-bucket", "nation.parquet", 1UL << 20);

  CHECK(probe.object_size == parquet.size());
  CHECK(probe.window_lo == 0);
  REQUIRE(probe.bytes != nullptr);
  CHECK(probe.bytes->size() == parquet.size());
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 1);
}

TEST_CASE("parquet footer bind is served by one suffix GET and then the stash",
          "[s3][integration][rest][footerbind]")
{
  auto const parquet    = read_binary_file(committed_parquet_fixture("nation.parquet"));
  auto const footer_len = static_cast<std::size_t>(parquet_footer_len(parquet));
  auto const footer_off = parquet.size() - 8 - footer_len;
  range_http_server server(parquet);
  auto ioctx = make_direct_rest_ioctx(server.endpoint());

  auto datasource = ioctx->open_datasource("s3://footer-bucket/nation.parquet",
                                           sirius::io::open_hint::parquet_footer_probe);

  REQUIRE(datasource != nullptr);
  CHECK(datasource->size() == parquet.size());
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 1);

  std::array<std::uint8_t, 8> trailer{};
  REQUIRE(datasource->host_read(parquet.size() - trailer.size(), trailer.size(), trailer.data()) ==
          trailer.size());
  require_bytes_equal(trailer,
                      std::span<std::uint8_t const>(parquet.data() + parquet.size() - 8, 8));

  std::vector<std::uint8_t> footer(footer_len);
  REQUIRE(datasource->host_read(footer_off, footer.size(), footer.data()) == footer.size());
  require_bytes_equal(footer,
                      std::span<std::uint8_t const>(parquet.data() + footer_off, footer_len));

  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 1);
}

TEST_CASE("footer probe open uses the configured suffix window",
          "[s3][integration][rest][footerbind]")
{
  auto const parquet    = read_binary_file(committed_parquet_fixture("nation.parquet"));
  auto const footer_len = static_cast<std::size_t>(parquet_footer_len(parquet));
  auto const footer_off = parquet.size() - 8 - footer_len;

  SECTION("tiny configured window misses the footer body and re-GETs it")
  {
    std::size_t constexpr suffix_bytes = 8;
    REQUIRE(footer_len + 8 > suffix_bytes);
    range_http_server server(parquet);
    auto cfg               = direct_rest_test_config();
    cfg.footer_probe_bytes = suffix_bytes;
    auto ioctx             = make_direct_rest_ioctx(server.endpoint(), cfg);
    auto datasource        = ioctx->open_datasource("s3://footer-bucket/nation.parquet",
                                             sirius::io::open_hint::parquet_footer_probe);
    auto const* rest_object =
      dynamic_cast<sirius::io::rest::rest_io_object const*>(&datasource->io_object());

    REQUIRE(rest_object != nullptr);
    CHECK(rest_object->stash_window_lo() == parquet.size() - suffix_bytes);
    REQUIRE(rest_object->stash() != nullptr);
    CHECK(rest_object->stash()->size() == suffix_bytes);
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 1);

    std::vector<std::uint8_t> footer(footer_len);
    REQUIRE(datasource->host_read(footer_off, footer.size(), footer.data()) == footer.size());
    require_bytes_equal(footer,
                        std::span<std::uint8_t const>(parquet.data() + footer_off, footer_len));
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 2);
  }

  SECTION("configured window covering the footer serves both cudf footer reads from stash")
  {
    auto const suffix_bytes = footer_len + 8;
    REQUIRE(suffix_bytes <= parquet.size());
    range_http_server server(parquet);
    auto cfg               = direct_rest_test_config();
    cfg.footer_probe_bytes = suffix_bytes;
    auto ioctx             = make_direct_rest_ioctx(server.endpoint(), cfg);
    auto datasource        = ioctx->open_datasource("s3://footer-bucket/nation.parquet",
                                             sirius::io::open_hint::parquet_footer_probe);
    auto const* rest_object =
      dynamic_cast<sirius::io::rest::rest_io_object const*>(&datasource->io_object());

    REQUIRE(rest_object != nullptr);
    CHECK(rest_object->stash_window_lo() == parquet.size() - suffix_bytes);
    REQUIRE(rest_object->stash() != nullptr);
    CHECK(rest_object->stash()->size() == suffix_bytes);
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 1);

    std::array<std::uint8_t, 8> trailer{};
    REQUIRE(datasource->host_read(
              parquet.size() - trailer.size(), trailer.size(), trailer.data()) == trailer.size());
    require_bytes_equal(trailer,
                        std::span<std::uint8_t const>(parquet.data() + parquet.size() - 8, 8));

    std::vector<std::uint8_t> footer(footer_len);
    REQUIRE(datasource->host_read(footer_off, footer.size(), footer.data()) == footer.size());
    require_bytes_equal(footer,
                        std::span<std::uint8_t const>(parquet.data() + footer_off, footer_len));
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 1);
  }
}

TEST_CASE("footer outside the suffix window falls back to one body GET",
          "[s3][integration][rest][footerbind]")
{
  auto const parquet             = read_binary_file(committed_parquet_fixture("nation.parquet"));
  auto const footer_len          = static_cast<std::size_t>(parquet_footer_len(parquet));
  auto const footer_off          = parquet.size() - 8 - footer_len;
  std::size_t const suffix_bytes = 8;
  REQUIRE(footer_len + 8 > suffix_bytes);
  range_http_server server(parquet);
  auto authorizer = std::make_shared<fixed_url_authorizer>(server.endpoint());
  sirius::io::rest::config cfg{};
  cfg.request_timeout_s       = 5;
  cfg.max_retry_attempts      = 1;
  cfg.max_auth_retry_attempts = 1;
  auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
    cfg, std::move(authorizer), nullptr);
  sirius::io::rest::rest_reactor reactor(ctx, "footer-window-test");
  reactor.start();

  auto probe = reactor.fetch_footer_suffix("footer-bucket", "nation.parquet", suffix_bytes);
  CHECK(probe.object_size == parquet.size());
  CHECK(probe.window_lo == parquet.size() - suffix_bytes);
  REQUIRE(probe.bytes != nullptr);

  sirius::io::rest::rest_io_object object("s3://footer-bucket/nation.parquet",
                                          "footer-bucket",
                                          "nation.parquet",
                                          probe.object_size,
                                          probe.window_lo,
                                          probe.bytes);
  CHECK(server.get_count() == 1);

  std::vector<std::uint8_t> footer(footer_len);
  REQUIRE(reactor.host_read(object, footer_off, footer.size(), footer.data()) == footer.size());
  require_bytes_equal(footer,
                      std::span<std::uint8_t const>(parquet.data() + footer_off, footer_len));
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 2);
}

TEST_CASE("describe_parquet over S3 uses footer probe and preserves schema",
          "[s3][integration][rest][footerbind]")
{
  auto const parquet = read_binary_file(committed_parquet_fixture("nation.parquet"));
  range_http_server server(parquet);
  scan_manager_fixture fixture;
  sirius_scan_manager manager{
    make_fake_rest_config(server.endpoint()), *fixture.memory, fixture.topology};

  auto result = manager.describe_parquet("s3://footer-bucket/nation.parquet");

  CHECK(result.object_size == parquet.size());
  CHECK(result.total_num_rows == 25);
  REQUIRE(result.names.size() == 4);
  CHECK(result.names[0] == "n_nationkey");
  CHECK(result.names[1] == "n_name");
  CHECK(result.names[2] == "n_regionkey");
  CHECK(result.names[3] == "n_comment");
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 1);

  auto second = manager.describe_parquet("s3://footer-bucket/nation.parquet");
  CHECK(second.object_size == result.object_size);
  CHECK(second.total_num_rows == result.total_num_rows);
  CHECK(second.names == result.names);
  CHECK(server.head_count() == 1);
  CHECK(server.get_count() == 1);
}

TEST_CASE("footer suffix probe falls back safely on unusable suffix responses",
          "[s3][integration][rest][footerbind]")
{
  auto const parquet = read_binary_file(committed_parquet_fixture("nation.parquet"));

  SECTION("missing Content-Range")
  {
    range_fault_policy fault{};
    fault.omit_content_range = true;
    range_http_server server(parquet, fault);
    auto ioctx      = make_direct_rest_ioctx(server.endpoint());
    auto datasource = ioctx->open_datasource("s3://footer-bucket/nation.parquet",
                                             sirius::io::open_hint::parquet_footer_probe);
    REQUIRE(datasource != nullptr);
    CHECK(datasource->size() == parquet.size());
    CHECK(server.head_count() == 1);
    CHECK(server.get_count() == 1);
  }

  SECTION("unknown Content-Range total")
  {
    range_fault_policy fault{};
    fault.unknown_content_range_total = true;
    range_http_server server(parquet, fault);
    auto ioctx      = make_direct_rest_ioctx(server.endpoint());
    auto datasource = ioctx->open_datasource("s3://footer-bucket/nation.parquet",
                                             sirius::io::open_hint::parquet_footer_probe);
    REQUIRE(datasource != nullptr);
    CHECK(datasource->size() == parquet.size());
    CHECK(server.head_count() == 1);
    CHECK(server.get_count() == 1);
  }

  SECTION("server ignores Range with 200 full-body")
  {
    range_fault_policy fault{};
    fault.ignore_range_with_200 = true;
    range_http_server server(parquet, fault);
    auto ioctx      = make_direct_rest_ioctx(server.endpoint());
    auto datasource = ioctx->open_datasource("s3://footer-bucket/nation.parquet",
                                             sirius::io::open_hint::parquet_footer_probe);
    REQUIRE(datasource != nullptr);
    CHECK(datasource->size() == parquet.size());
    CHECK(server.head_count() == 1);
    CHECK(server.get_count() == 1);
  }

  SECTION("server ignores Range with 200 full-body on a large object")
  {
    auto payload = deterministic_payload(8 * 1024 * 1024);
    range_fault_policy fault{};
    fault.ignore_range_with_200 = true;
    range_http_server server(payload, fault);
    auto ioctx = make_direct_rest_ioctx(server.endpoint());

    auto datasource = ioctx->open_datasource("s3://footer-bucket/large.bin",
                                             sirius::io::open_hint::parquet_footer_probe);

    REQUIRE(datasource != nullptr);
    CHECK(datasource->size() == payload.size());
    CHECK(server.head_count() == 1);
    CHECK(server.get_count() == 1);
    CHECK(server.body_bytes_sent() < payload.size());
  }

  SECTION("suffix 416 falls back to HEAD")
  {
    range_fault_policy fault{};
    fault.fail_suffix_with_416 = true;
    range_http_server server(parquet, fault);
    auto ioctx = make_direct_rest_ioctx(server.endpoint());

    auto datasource = ioctx->open_datasource("s3://footer-bucket/nation.parquet",
                                             sirius::io::open_hint::parquet_footer_probe);

    REQUIRE(datasource != nullptr);
    CHECK(datasource->size() == parquet.size());
    CHECK(server.head_count() == 1);
    CHECK(server.get_count() == 1);
  }
}

TEST_CASE("footer suffix probe retries transient GET failures",
          "[s3][integration][rest][footerbind]")
{
  auto const parquet = read_binary_file(committed_parquet_fixture("nation.parquet"));

  SECTION("transient 503s are retried and the final 206 produces a probe")
  {
    range_fault_policy fault{};
    fault.fail_first_gets = 2;
    fault.fail_status     = 503;
    range_http_server server(parquet, fault);
    auto authorizer             = std::make_shared<fixed_url_authorizer>(server.endpoint());
    auto cfg                    = direct_rest_test_config();
    cfg.max_retry_attempts      = 3;
    cfg.max_auth_retry_attempts = 1;
    auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, std::move(authorizer), nullptr);
    sirius::io::rest::rest_reactor reactor(ctx, "footer-suffix-retry-success");

    auto probe = reactor.fetch_footer_suffix("footer-bucket", "nation.parquet", 1UL << 20);

    CHECK(probe.object_size == parquet.size());
    CHECK(probe.window_lo == 0);
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.bytes->size() == parquet.size());
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 3);
    auto const perf = reactor.perf_snapshot();
    CHECK(perf.retries_total == 2);
    CHECK(perf.terminal_failures_total == 0);
  }

  SECTION("exhausted transient 503s throw after the retry budget")
  {
    range_fault_policy fault{};
    fault.fail_all_gets = true;
    fault.fail_status   = 503;
    range_http_server server(parquet, fault);
    auto authorizer             = std::make_shared<fixed_url_authorizer>(server.endpoint());
    auto cfg                    = direct_rest_test_config();
    cfg.max_retry_attempts      = 2;
    cfg.max_auth_retry_attempts = 1;
    auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, std::move(authorizer), nullptr);
    sirius::io::rest::rest_reactor reactor(ctx, "footer-suffix-retry-exhausted");

    try {
      (void)reactor.fetch_footer_suffix("footer-bucket", "nation.parquet", 1UL << 20);
      FAIL("fetch_footer_suffix should throw after exhausting transient retries");
    } catch (std::runtime_error const& e) {
      auto const message = std::string{e.what()};
      CHECK(message.find("exhausted retries") != std::string::npos);
      CHECK(message.find("HTTP 503") != std::string::npos);
    }
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 2);
    auto const perf = reactor.perf_snapshot();
    CHECK(perf.retries_total == cfg.max_retry_attempts - 1);
    CHECK(perf.terminal_failures_total == 1);
  }

  SECTION("hard non-retriable errors fail without retrying")
  {
    range_fault_policy fault{};
    fault.fail_all_gets = true;
    fault.fail_status   = 403;
    range_http_server server(parquet, fault);
    auto authorizer             = std::make_shared<fixed_url_authorizer>(server.endpoint());
    auto cfg                    = direct_rest_test_config();
    cfg.max_retry_attempts      = 3;
    cfg.max_auth_retry_attempts = 1;
    auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, std::move(authorizer), nullptr);
    sirius::io::rest::rest_reactor reactor(ctx, "footer-suffix-hard-failure");

    try {
      (void)reactor.fetch_footer_suffix("footer-bucket", "nation.parquet", 1UL << 20);
      FAIL("fetch_footer_suffix should throw on a hard non-retriable HTTP error");
    } catch (std::runtime_error const& e) {
      auto const message = std::string{e.what()};
      CHECK(message.find("HTTP 403") != std::string::npos);
    }
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 1);
    auto const perf = reactor.perf_snapshot();
    CHECK(perf.retries_total == 0);
    CHECK(perf.terminal_failures_total == 1);
  }

  SECTION("clean 206 has no retry or terminal-failure telemetry")
  {
    range_http_server server(parquet);
    auto authorizer             = std::make_shared<fixed_url_authorizer>(server.endpoint());
    auto cfg                    = direct_rest_test_config();
    cfg.max_retry_attempts      = 3;
    cfg.max_auth_retry_attempts = 1;
    auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, std::move(authorizer), nullptr);
    sirius::io::rest::rest_reactor reactor(ctx, "footer-suffix-clean");

    auto probe = reactor.fetch_footer_suffix("footer-bucket", "nation.parquet", 1UL << 20);

    CHECK(probe.object_size == parquet.size());
    CHECK(probe.window_lo == 0);
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.bytes->size() == parquet.size());
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 1);
    auto const perf = reactor.perf_snapshot();
    CHECK(perf.retries_total == 0);
    CHECK(perf.terminal_failures_total == 0);
  }
}

TEST_CASE("HEAD object-size retries update REST perf counters",
          "[s3][integration][rest][footerbind]")
{
  auto const payload = deterministic_payload(4096);

  SECTION("transient 503s are retried and the final HEAD returns the size")
  {
    range_fault_policy fault{};
    fault.fail_first_heads = 2;
    fault.fail_head_status = 503;
    range_http_server server(payload, fault);
    auto authorizer             = std::make_shared<fixed_url_authorizer>(server.endpoint());
    auto cfg                    = direct_rest_test_config();
    cfg.max_retry_attempts      = 3;
    cfg.max_auth_retry_attempts = 1;
    auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, std::move(authorizer), nullptr);
    sirius::io::rest::rest_reactor reactor(ctx, "head-retry-success");

    CHECK(reactor.head_object_size("head-bucket", "head-success.bin") == payload.size());
    CHECK(server.head_count() == 3);
    CHECK(server.get_count() == 0);
    auto const perf = reactor.perf_snapshot();
    CHECK(perf.retries_total == 2);
    CHECK(perf.terminal_failures_total == 0);
  }

  SECTION("exhausted transient 503s are reported as one terminal failure")
  {
    range_fault_policy fault{};
    fault.fail_all_heads   = true;
    fault.fail_head_status = 503;
    range_http_server server(payload, fault);
    auto authorizer             = std::make_shared<fixed_url_authorizer>(server.endpoint());
    auto cfg                    = direct_rest_test_config();
    cfg.max_retry_attempts      = 2;
    cfg.max_auth_retry_attempts = 1;
    auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, std::move(authorizer), nullptr);
    sirius::io::rest::rest_reactor reactor(ctx, "head-retry-exhausted");

    try {
      (void)reactor.head_object_size("head-bucket", "head-exhausted.bin");
      FAIL("head_object_size should throw after exhausting transient retries");
    } catch (std::runtime_error const& e) {
      auto const message = std::string{e.what()};
      CHECK(message.find("exhausted retries") != std::string::npos);
      CHECK(message.find("HTTP 503") != std::string::npos);
    }
    CHECK(server.head_count() == 2);
    CHECK(server.get_count() == 0);
    auto const perf = reactor.perf_snapshot();
    CHECK(perf.retries_total == cfg.max_retry_attempts - 1);
    CHECK(perf.terminal_failures_total == 1);
  }

  SECTION("hard non-retriable HEAD errors fail without retrying")
  {
    range_fault_policy fault{};
    fault.fail_all_heads   = true;
    fault.fail_head_status = 403;
    range_http_server server(payload, fault);
    auto authorizer             = std::make_shared<fixed_url_authorizer>(server.endpoint());
    auto cfg                    = direct_rest_test_config();
    cfg.max_retry_attempts      = 3;
    cfg.max_auth_retry_attempts = 1;
    auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, std::move(authorizer), nullptr);
    sirius::io::rest::rest_reactor reactor(ctx, "head-hard-failure");

    try {
      (void)reactor.head_object_size("head-bucket", "head-forbidden.bin");
      FAIL("head_object_size should throw on a hard non-retriable HTTP error");
    } catch (std::runtime_error const& e) {
      auto const message = std::string{e.what()};
      CHECK(message.find("HTTP 403") != std::string::npos);
    }
    CHECK(server.head_count() == 1);
    CHECK(server.get_count() == 0);
    auto const perf = reactor.perf_snapshot();
    CHECK(perf.retries_total == 0);
    CHECK(perf.terminal_failures_total == 1);
  }

  SECTION("clean HEAD has no retry or terminal-failure telemetry")
  {
    range_http_server server(payload);
    auto authorizer             = std::make_shared<fixed_url_authorizer>(server.endpoint());
    auto cfg                    = direct_rest_test_config();
    cfg.max_retry_attempts      = 3;
    cfg.max_auth_retry_attempts = 1;
    auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, std::move(authorizer), nullptr);
    sirius::io::rest::rest_reactor reactor(ctx, "head-clean");

    CHECK(reactor.head_object_size("head-bucket", "head-clean.bin") == payload.size());
    CHECK(server.head_count() == 1);
    CHECK(server.get_count() == 0);
    auto const perf = reactor.perf_snapshot();
    CHECK(perf.retries_total == 0);
    CHECK(perf.terminal_failures_total == 0);
  }
}

TEST_CASE("REST retry logging includes object keys and stays quiet on clean requests",
          "[s3][integration][rest][footerbind]")
{
  auto const payload = deterministic_payload(4096);

  SECTION("a retried request emits a warning with the object key")
  {
    range_fault_policy fault{};
    fault.fail_first_gets = 1;
    fault.fail_status     = 503;
    range_http_server server(payload, fault);
    auto authorizer             = std::make_shared<fixed_url_authorizer>(server.endpoint());
    auto cfg                    = direct_rest_test_config();
    cfg.max_retry_attempts      = 2;
    cfg.max_auth_retry_attempts = 1;
    auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, std::move(authorizer), nullptr);
    sirius::io::rest::rest_reactor reactor(ctx, "retry-log-warning");

    scoped_log_capture logs;
    auto probe = reactor.fetch_footer_suffix("log-bucket", "warn-retry.parquet", 1024);
    REQUIRE(probe.bytes != nullptr);

    auto const records = logs.records();
    auto const found   = std::any_of(records.begin(), records.end(), [](auto const& r) {
      return r.level == sirius::log::level::warn &&
             r.message.find("warn-retry.parquet") != std::string::npos;
    });
    CHECK(found);
  }

  SECTION("clean first-try suffix GET and HEAD do not emit warnings or errors")
  {
    range_http_server server(payload);
    auto authorizer             = std::make_shared<fixed_url_authorizer>(server.endpoint());
    auto cfg                    = direct_rest_test_config();
    cfg.max_retry_attempts      = 2;
    cfg.max_auth_retry_attempts = 1;
    auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, std::move(authorizer), nullptr);
    sirius::io::rest::rest_reactor reactor(ctx, "retry-log-clean");

    scoped_log_capture logs;
    auto probe = reactor.fetch_footer_suffix("log-bucket", "clean.parquet", 1024);
    REQUIRE(probe.bytes != nullptr);
    CHECK(reactor.head_object_size("log-bucket", "clean.parquet") == payload.size());

    auto const records        = logs.records();
    auto const warning_or_bad = std::any_of(records.begin(), records.end(), [](auto const& r) {
      return r.level >= sirius::log::level::warn;
    });
    CHECK_FALSE(warning_or_bad);
  }
}

TEST_CASE("small objects can be resolved by one suffix probe",
          "[s3][integration][rest][footerbind]")
{
  auto payload = deterministic_payload(512);
  range_http_server server(payload);
  auto authorizer = std::make_shared<fixed_url_authorizer>(server.endpoint());
  sirius::io::rest::config cfg{};
  cfg.request_timeout_s       = 5;
  cfg.max_retry_attempts      = 1;
  cfg.max_auth_retry_attempts = 1;
  auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
    cfg, std::move(authorizer), nullptr);
  sirius::io::rest::rest_reactor reactor(ctx, "small-footer-suffix-test");

  auto probe = reactor.fetch_footer_suffix("footer-bucket", "small.bin", 1UL << 20);

  CHECK(probe.object_size == payload.size());
  CHECK(probe.window_lo == 0);
  REQUIRE(probe.bytes != nullptr);
  CHECK(probe.bytes->size() == payload.size());
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 1);
}

TEST_CASE("concurrent footer probes each get an object-local suffix stash",
          "[s3][integration][rest][footerbind]")
{
  auto const parquet = read_binary_file(committed_parquet_fixture("nation.parquet"));
  range_http_server server(parquet);
  auto ioctx            = make_direct_rest_ioctx(server.endpoint());
  std::string const uri = "s3://footer-bucket/nation.parquet";

  std::vector<std::future<std::size_t>> futures;
  for (int i = 0; i < 8; ++i) {
    futures.push_back(std::async(std::launch::async, [ioctx, uri] {
      auto datasource = ioctx->open_datasource(uri, sirius::io::open_hint::parquet_footer_probe);
      return datasource->size();
    }));
  }

  for (auto& f : futures) {
    CHECK(f.get() == parquet.size());
  }
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == futures.size());
}

TEST_CASE("rest_ioctx reads the MinIO hello fixture through scan_manager create_datasource",
          "[s3][integration][rest]")
{
  if (!sirius::test::ensure_s3_container_env()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  scan_manager_fixture fixture;
  sirius_scan_manager manager{make_minio_rest_config(), *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://" + bucket + "/hello.txt");

  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  CHECK(datasource->io_ctx()->type() == io_context_type::restful);
  auto* rest_ctx = dynamic_cast<rest_ioctx*>(datasource->io_ctx().get());
  REQUIRE(rest_ctx != nullptr);

  REQUIRE(datasource->size() == 16);

  std::array<std::uint8_t, 16> got{};
  REQUIRE(datasource->host_read(0, got.size(), got.data()) == got.size());

  std::array<std::uint8_t, 16> const expected{
    's', 'i', 'r', 'i', 'u', 's', '-', 's', '3', '-', 'h', 'e', 'l', 'l', 'o', '\n'};
  CHECK(got == expected);
}

TEST_CASE("rest_ioctx reads exact host ranges and clips EOF on MinIO fixtures",
          "[s3][integration][rest]")
{
  if (!sirius::test::ensure_s3_container_env()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  auto const small  = read_binary_file(local_fixture_path("small.bin"));
  scan_manager_fixture fixture;
  sirius_scan_manager manager{make_minio_rest_config(), *fixture.memory, fixture.topology};
  auto datasource = manager.create_datasource("s3://" + bucket + "/small.bin");
  require_rest_ioctx(datasource);

  REQUIRE(datasource->size() == small.size());

  std::vector<std::uint8_t> full(small.size());
  REQUIRE(datasource->host_read(0, full.size(), full.data()) == full.size());
  require_bytes_equal(full, small);

  std::vector<std::uint8_t> mid(4096);
  std::size_t const mid_offset = 1234;
  REQUIRE(datasource->host_read(mid_offset, mid.size(), mid.data()) == mid.size());
  require_bytes_equal(mid, std::span<std::uint8_t const>(small.data() + mid_offset, mid.size()));

  std::vector<std::uint8_t> crossing(64, std::uint8_t{0xaa});
  std::size_t const tail_offset = small.size() - 17;
  REQUIRE(datasource->host_read(tail_offset, crossing.size(), crossing.data()) == 17);
  require_bytes_equal(std::span<std::uint8_t const>(crossing.data(), 17),
                      std::span<std::uint8_t const>(small.data() + tail_offset, 17));

  std::array<std::uint8_t, 8> eof{};
  eof.fill(std::uint8_t{0xcc});
  CHECK(datasource->host_read(small.size(), eof.size(), eof.data()) == 0);
  CHECK(datasource->host_read(small.size() + 99, eof.size(), eof.data()) == 0);
  CHECK(std::all_of(eof.begin(), eof.end(), [](std::uint8_t b) { return b == 0xcc; }));
}

TEST_CASE("rest_ioctx fans out host_read_ranges against the MinIO medium fixture",
          "[s3][integration][rest]")
{
  if (!sirius::test::ensure_s3_container_env()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  auto const medium = read_binary_file(local_fixture_path("medium.bin"));
  scan_manager_fixture fixture;
  auto cfg                 = make_minio_rest_config();
  cfg.rest.max_connections = 4;
  cfg.rest.chunk_size      = 1UL << 20;
  cfg.rest.max_n_chunks    = 1;
  sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};
  auto datasource = manager.create_datasource("s3://" + bucket + "/medium.bin");
  require_rest_ioctx(datasource);

  std::vector<std::vector<std::uint8_t>> buffers;
  std::vector<sirius::io::io_object_segment> segments;
  std::vector<std::pair<std::size_t, std::size_t>> ranges{
    {17, 257},
    {64 * 1024 + 9, 1024},
    {2 * 1024 * 1024 + 11, 4096},
    {medium.size() - 333, 333},
  };
  buffers.reserve(ranges.size());
  segments.reserve(ranges.size());
  std::size_t total = 0;
  for (auto const& [offset, size] : ranges) {
    buffers.emplace_back(size);
    segments.emplace_back(offset, size, buffers.back().data());
    total += size;
  }

  auto got =
    std::move(datasource->io_ctx()->host_read_ranges_async_io(
                datasource->io_object(), std::span<sirius::io::io_object_segment>(segments)))
      .get(5s);
  REQUIRE(got == total);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    auto const [offset, size] = ranges[i];
    require_bytes_equal(buffers[i], std::span<std::uint8_t const>(medium.data() + offset, size));
  }
}

TEST_CASE("rest_ioctx stages device reads through FSMR for single and multi chunk MinIO reads",
          "[s3][integration][rest]")
{
  if (!sirius::test::ensure_s3_container_env()) { return; }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    WARN("Skipping rest_ioctx device-read integration test: no CUDA device");
    return;
  }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  auto const small  = read_binary_file(local_fixture_path("small.bin"));
  auto const medium = read_binary_file(local_fixture_path("medium.bin"));

  scan_manager_fixture fixture;
  auto cfg                 = make_minio_rest_config();
  cfg.rest.max_connections = 4;
  cfg.rest.chunk_size      = 1UL << 20;
  sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

  SECTION("single chunk")
  {
    auto datasource = manager.create_datasource("s3://" + bucket + "/small.bin");
    require_rest_ioctx(datasource);

    rmm::cuda_stream stream;
    rmm::device_buffer dst(small.size(), stream);
    REQUIRE(datasource->device_read(
              0, small.size(), static_cast<std::uint8_t*>(dst.data()), stream) == small.size());
    auto got = copy_device_to_host(dst, small.size(), stream);
    require_bytes_equal(got, small);
  }

  SECTION("multi chunk")
  {
    auto datasource = manager.create_datasource("s3://" + bucket + "/medium.bin");
    require_rest_ioctx(datasource);

    std::size_t const offset = 123;
    std::size_t const size   = (3UL << 20) + 17;
    rmm::cuda_stream stream;
    rmm::device_buffer dst(size, stream);
    REQUIRE(datasource->device_read(offset, size, static_cast<std::uint8_t*>(dst.data()), stream) ==
            size);
    auto got = copy_device_to_host(dst, size, stream);
    require_bytes_equal(got, std::span<std::uint8_t const>(medium.data() + offset, size));
  }
}

TEST_CASE("rest_ioctx retries transient fake-server failures and reports terminal failures",
          "[s3][integration][rest]")
{
  auto payload = deterministic_payload(64 * 1024);
  scan_manager_fixture fixture;

  SECTION("transient 503s are retried")
  {
    range_fault_policy fault{};
    fault.fail_first_gets = 2;
    fault.fail_status     = 503;
    range_http_server server(payload, fault);
    auto cfg                    = make_fake_rest_config(server.endpoint());
    cfg.rest.max_retry_attempts = 4;
    sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

    auto datasource = manager.create_datasource("s3://retry-bucket/object.bin");
    require_rest_ioctx(datasource);
    std::vector<std::uint8_t> got(4096);
    REQUIRE(datasource->host_read(128, got.size(), got.data()) == got.size());
    require_bytes_equal(got, std::span<std::uint8_t const>(payload.data() + 128, got.size()));
    CHECK(server.get_count() >= 3);
  }

  SECTION("exhausted retries surface as an error")
  {
    range_fault_policy fault{};
    fault.fail_all_gets = true;
    fault.fail_status   = 503;
    range_http_server server(payload, fault);
    auto cfg                    = make_fake_rest_config(server.endpoint());
    cfg.rest.max_retry_attempts = 2;
    sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

    auto datasource = manager.create_datasource("s3://retry-bucket/object.bin");
    require_rest_ioctx(datasource);
    std::array<std::uint8_t, 128> got{};
    CHECK_THROWS(datasource->host_read(0, got.size(), got.data()));
    CHECK(server.get_count() >= 2);
  }
}

TEST_CASE("rest_ioctx honors max_connections under concurrent fake range reads",
          "[s3][integration][rest]")
{
  auto payload = deterministic_payload(512 * 1024);
  range_fault_policy fault{};
  fault.response_delay = 50ms;
  range_http_server server(payload, fault);
  scan_manager_fixture fixture;
  auto cfg                 = make_fake_rest_config(server.endpoint());
  cfg.rest.max_connections = 2;
  cfg.rest.chunk_size      = 1024;
  cfg.rest.max_n_chunks    = 1;
  sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://concurrency-bucket/object.bin");
  require_rest_ioctx(datasource);

  std::vector<std::vector<std::uint8_t>> buffers;
  std::vector<sirius::io::io_object_segment> segments;
  for (std::size_t i = 0; i < 8; ++i) {
    std::size_t const offset = i * 4096 + 13;
    std::size_t const size   = 512;
    buffers.emplace_back(size);
    segments.emplace_back(offset, size, buffers.back().data());
  }

  auto got =
    std::move(datasource->io_ctx()->host_read_ranges_async_io(
                datasource->io_object(), std::span<sirius::io::io_object_segment>(segments)))
      .get(10s);
  REQUIRE(got == 8 * 512);
  CHECK(server.peak_active_gets() <= 2);
  CHECK(server.peak_active_gets() >= 1);
  for (std::size_t i = 0; i < segments.size(); ++i) {
    require_bytes_equal(
      buffers[i],
      std::span<std::uint8_t const>(payload.data() + segments[i].offset, segments[i].size));
  }
}

TEST_CASE("rest_ioctx reads through the TLS MinIO endpoint with the harness CA bundle",
          "[s3][integration][rest]")
{
  if (!sirius::test::ensure_s3_container_env()) { return; }

  auto const bucket = require_env("SIRIUS_TEST_S3_BUCKET");
  scan_manager_fixture fixture;
  sirius_scan_manager manager{make_tls_minio_rest_config(), *fixture.memory, fixture.topology};
  auto datasource = manager.create_datasource("s3://" + bucket + "/hello.txt");
  require_rest_ioctx(datasource);

  std::array<std::uint8_t, 16> got{};
  REQUIRE(datasource->host_read(0, got.size(), got.data()) == got.size());
  std::array<std::uint8_t, 16> const expected{
    's', 'i', 'r', 'i', 'u', 's', '-', 's', '3', '-', 'h', 'e', 'l', 'l', 'o', '\n'};
  CHECK(got == expected);
}

TEST_CASE("rest_ioctx teardown resolves an in-flight async read without hanging",
          "[s3][integration][rest]")
{
  auto payload = deterministic_payload(128 * 1024);
  range_fault_policy fault{};
  fault.response_delay = 100ms;
  range_http_server server(payload, fault);

  std::vector<std::uint8_t> got(4096);
  std::future<std::size_t> future;
  {
    scan_manager_fixture fixture;
    auto cfg                 = make_fake_rest_config(server.endpoint());
    cfg.rest.max_connections = 1;
    sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};
    auto datasource = manager.create_datasource("s3://lifecycle-bucket/object.bin");
    require_rest_ioctx(datasource);
    future = datasource->host_read_async(0, got.size(), got.data());
  }

  REQUIRE(future.valid());
  REQUIRE(future.wait_for(5s) == std::future_status::ready);
  try {
    auto const bytes = future.get();
    if (bytes != 0) {
      REQUIRE(bytes == got.size());
      require_bytes_equal(got, std::span<std::uint8_t const>(payload.data(), got.size()));
    }
  } catch (std::exception const& e) {
    INFO("teardown completed in-flight request with exception: " << e.what());
    SUCCEED("future resolved with an exception during teardown");
  }
}
