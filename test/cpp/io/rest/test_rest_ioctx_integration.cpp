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
#include "io/rest/authorizer.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/rest/s3/list_parser.hpp"
#include "io/sirius_datasource.hpp"
#include "io/types.hpp"
#include "memory/topology_index.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "utils/s3_container.hpp"
#include "utils/sirius_test_env.hpp"

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
#include <sys/wait.h>
#include <unistd.h>
#include <utils/log_test_utils.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <csignal>
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
#include <sstream>
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
  cfg.backend                 = sirius::scan_manager::io_backend::sirius;
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
  cfg.backend                      = sirius::scan_manager::io_backend::sirius;
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
  cfg.cache.mode                   = sirius::io::cache::cache_mode::none;
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

struct cuda_host_deleter {
  void operator()(std::uint8_t* ptr) const noexcept
  {
    if (ptr != nullptr) { std::ignore = cudaFreeHost(ptr); }
  }
};

using unique_cuda_host_buffer = std::unique_ptr<std::uint8_t, cuda_host_deleter>;

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
  std::size_t fail_first_lists{0};
  bool fail_all_lists{false};
  int fail_status{503};
  int fail_head_status{503};
  int fail_list_status{503};
  std::chrono::milliseconds response_delay{0};
  bool omit_content_range{false};
  bool unknown_content_range_total{false};
  bool ignore_range_with_200{false};
  bool fail_suffix_with_416{false};
  std::string failed_get_etag;
  std::string successful_get_etag;
  std::string successful_head_etag;
};

struct listed_object {
  std::string key;
  std::uint64_t size{0};
};

struct generated_listing {
  std::string prefix;
  std::size_t total{0};
};

enum class scripted_list_mode {
  normal,
  repeated_empty_token,
  alternating_empty_tokens,
  truncated_empty_without_token
};

class fixed_url_authorizer final : public sirius::io::rest::request_authorizer {
 public:
  explicit fixed_url_authorizer(std::string endpoint) : _endpoint(std::move(endpoint)) {}

  sirius::io::rest::authorized_request authorize(sirius::io::rest::object_ref const& obj,
                                                 sirius::io::rest::request_method /*method*/,
                                                 std::chrono::seconds /*timeout*/) override
  {
    return {_endpoint + "/" + obj.bucket + "/" + obj.key, {}};
  }

  sirius::io::rest::authorized_request authorize_list(std::string_view bucket,
                                                      std::string_view canonical_query,
                                                      std::chrono::seconds /*timeout*/) override
  {
    return {_endpoint + "/" + std::string{bucket} + "?" + std::string{canonical_query}, {}};
  }

 private:
  std::string _endpoint;
};

class range_http_server {
 public:
  explicit range_http_server(std::vector<std::uint8_t> object,
                             range_fault_policy fault          = {},
                             std::vector<listed_object> listed = {},
                             scripted_list_mode list_mode      = scripted_list_mode::normal)
    : _object(std::move(object)), _fault(fault), _listed(std::move(listed)), _list_mode(list_mode)
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
  [[nodiscard]] std::size_t list_count() const noexcept { return _list_count.load(); }
  [[nodiscard]] std::size_t body_bytes_sent() const noexcept { return _body_bytes_sent.load(); }
  [[nodiscard]] std::size_t max_requested_range() const noexcept
  {
    return _max_requested_range.load();
  }
  [[nodiscard]] int peak_active_gets() const noexcept { return _peak_active_gets.load(); }

  void set_generated_listing(std::string prefix, std::size_t total)
  {
    _generated_listing = generated_listing{std::move(prefix), total};
  }

 private:
  static void append_etag_header(std::string& response, std::string const& etag)
  {
    if (!etag.empty()) { response += "\r\nETag: " + etag; }
  }

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

  static std::string request_target(std::string const& request)
  {
    auto first_space = request.find(' ');
    if (first_space == std::string::npos) { return {}; }
    auto second_space = request.find(' ', first_space + 1);
    if (second_space == std::string::npos) { return {}; }
    return request.substr(first_space + 1, second_space - first_space - 1);
  }

  static std::string xml_escape(std::string_view value)
  {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
      switch (c) {
        case '&': out += "&amp;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '"': out += "&quot;"; break;
        case '\'': out += "&apos;"; break;
        default: out.push_back(c); break;
      }
    }
    return out;
  }

  static int from_hex(char c)
  {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
  }

  static std::string percent_decode(std::string_view value)
  {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
      if (value[i] == '%' && i + 2 < value.size()) {
        auto hi = from_hex(value[i + 1]);
        auto lo = from_hex(value[i + 2]);
        if (hi >= 0 && lo >= 0) {
          out.push_back(static_cast<char>((hi << 4) | lo));
          i += 2;
          continue;
        }
      }
      out.push_back(value[i]);
    }
    return out;
  }

  static std::string query_value(std::string const& target, std::string_view key)
  {
    auto qpos = target.find('?');
    if (qpos == std::string::npos) { return {}; }
    auto query      = std::string_view{target}.substr(qpos + 1);
    auto needle     = std::string{key} + "=";
    std::size_t pos = 0;
    while (pos < query.size()) {
      auto amp = query.find('&', pos);
      if (amp == std::string_view::npos) { amp = query.size(); }
      auto part        = query.substr(pos, amp - pos);
      auto needle_view = std::string_view{needle};
      if (part.size() >= needle_view.size() && part.substr(0, needle_view.size()) == needle_view) {
        return percent_decode(part.substr(needle.size()));
      }
      pos = amp + 1;
    }
    return {};
  }

  static std::size_t page_start_from_token(std::string const& token)
  {
    if (token.empty()) { return 0; }
    std::string const prefix = "page/";
    std::string const suffix = "+=";
    if (token.rfind(prefix, 0) != 0 || token.size() <= prefix.size() + suffix.size() ||
        token.substr(token.size() - suffix.size()) != suffix) {
      throw std::runtime_error("unexpected continuation-token: " + token);
    }
    return static_cast<std::size_t>(
      std::stoull(token.substr(prefix.size(), token.size() - prefix.size() - suffix.size())));
  }

  static std::string continuation_token_for(std::size_t index)
  {
    return "page/" + std::to_string(index) + "+=";
  }

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
    auto const target  = request_target(request);
    if (is_head) {
      auto const head_idx = _head_count.fetch_add(1, std::memory_order_relaxed);
      if (_fault.fail_all_heads || head_idx < _fault.fail_first_heads) {
        std::string response =
          "HTTP/1.1 " + std::to_string(_fault.fail_head_status) +
          " Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        send_all(fd, response);
        return;
      }
      std::string response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size());
      append_etag_header(response, _fault.successful_head_etag);
      response += "\r\nConnection: close\r\n\r\n";
      send_all(fd, response);
      return;
    }
    if (is_get && target.find("list-type=2") != std::string::npos) {
      handle_list_request(fd, target);
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
      std::string response = "HTTP/1.1 " + std::to_string(_fault.fail_status) +
                             " Service Unavailable\r\nContent-Length: 0";
      append_etag_header(response, _fault.failed_get_etag);
      response += "\r\nConnection: close\r\n\r\n";
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
          "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size());
        append_etag_header(response, _fault.successful_get_etag);
        response += "\r\nConnection: close\r\n\r\n";
        send_all(fd, response);
        send_body(fd, _object.data(), _object.size());
        return;
      }
      auto const len    = end - start + 1;
      auto previous_max = _max_requested_range.load(std::memory_order_relaxed);
      while (len > previous_max && !_max_requested_range.compare_exchange_weak(
                                     previous_max, len, std::memory_order_relaxed)) {}
      std::string response =
        "HTTP/1.1 206 Partial Content\r\nContent-Length: " + std::to_string(len);
      if (!_fault.omit_content_range) {
        response +=
          "\r\nContent-Range: bytes " + std::to_string(start) + "-" + std::to_string(end) + "/" +
          (_fault.unknown_content_range_total ? std::string{"*"} : std::to_string(_object.size()));
      }
      append_etag_header(response, _fault.successful_get_etag);
      response += "\r\nConnection: close\r\n\r\n";
      send_all(fd, response);
      send_body(fd, _object.data() + start, len);
      return;
    }

    std::string response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size());
    append_etag_header(response, _fault.successful_get_etag);
    response += "\r\nConnection: close\r\n\r\n";
    send_all(fd, response);
    send_body(fd, _object.data(), _object.size());
  }

  void handle_list_request(int fd, std::string const& target)
  {
    auto const list_idx = _list_count.fetch_add(1, std::memory_order_relaxed);
    if (_fault.fail_all_lists || list_idx < _fault.fail_first_lists) {
      std::string response =
        "HTTP/1.1 " + std::to_string(_fault.fail_list_status) +
        " Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
      send_all(fd, response);
      return;
    }
    if (_fault.response_delay.count() > 0) { std::this_thread::sleep_for(_fault.response_delay); }

    if (_list_mode != scripted_list_mode::normal) {
      std::string token_out;
      if (_list_mode == scripted_list_mode::repeated_empty_token) {
        token_out = "repeat-token";
      } else if (_list_mode == scripted_list_mode::alternating_empty_tokens) {
        token_out = list_idx % 2 == 0 ? "token-A" : "token-B";
      }

      std::ostringstream xml;
      xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
             "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
             "<IsTruncated>true</IsTruncated>";
      if (!token_out.empty()) {
        xml << "<NextContinuationToken>" << xml_escape(token_out) << "</NextContinuationToken>";
      }
      xml << "</ListBucketResult>";

      auto body = xml.str();
      std::string response =
        "HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
      send_all(fd, response);
      send_all(fd, body);
      return;
    }

    auto prefix          = query_value(target, "prefix");
    auto token           = query_value(target, "continuation-token");
    std::size_t max_keys = 1000;
    auto max_keys_text   = query_value(target, "max-keys");
    if (!max_keys_text.empty()) {
      max_keys = std::max<std::size_t>(
        1, std::min<std::size_t>(1000, static_cast<std::size_t>(std::stoull(max_keys_text))));
    }
    auto start = page_start_from_token(token);

    std::ostringstream xml;
    xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
           "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
           "<IsTruncated>";

    auto const emit_contents = [&](std::string_view key, std::uint64_t size) {
      xml << "<Contents><Key>" << xml_escape(key) << "</Key><Size>" << size << "</Size></Contents>";
    };

    bool truncated        = false;
    std::string token_out = {};
    if (_generated_listing.has_value() && prefix == _generated_listing->prefix) {
      auto const total = _generated_listing->total;
      if (start > total) { start = total; }
      auto const end = std::min(start + max_keys, total);
      truncated      = end < total;
      token_out      = truncated ? continuation_token_for(end) : std::string{};
      xml << (truncated ? "true" : "false") << "</IsTruncated>";
      for (std::size_t i = start; i < end; ++i) {
        emit_contents(_generated_listing->prefix + std::to_string(i),
                      static_cast<std::uint64_t>(i % 4096U));
      }
    } else {
      std::vector<listed_object> matching;
      for (auto const& object : _listed) {
        if (object.key.rfind(prefix, 0) == 0) { matching.push_back(object); }
      }

      if (start > matching.size()) { start = matching.size(); }
      auto const end = std::min(start + max_keys, matching.size());
      truncated      = end < matching.size();
      token_out      = truncated ? continuation_token_for(end) : std::string{};
      xml << (truncated ? "true" : "false") << "</IsTruncated>";
      for (auto i = start; i < end; ++i) {
        emit_contents(matching[i].key, matching[i].size);
      }
    }
    if (truncated) {
      xml << "<NextContinuationToken>" << xml_escape(token_out) << "</NextContinuationToken>";
    }
    xml << "</ListBucketResult>";

    auto body            = xml.str();
    std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\nContent-Length: " +
                           std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
    send_all(fd, response);
    send_all(fd, body);
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
  std::vector<listed_object> _listed;
  scripted_list_mode _list_mode;
  std::optional<generated_listing> _generated_listing;
  std::atomic<bool> _stop{false};
  std::atomic<std::size_t> _head_count{0};
  std::atomic<std::size_t> _get_count{0};
  std::atomic<std::size_t> _list_count{0};
  std::atomic<std::size_t> _body_bytes_sent{0};
  std::atomic<std::size_t> _max_requested_range{0};
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
                                                   sirius::io::rest::config cfg,
                                                   std::size_t n_reactors = 1)
{
  auto authorizer = std::make_shared<fixed_url_authorizer>(std::move(endpoint));
  auto ctx        = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
    cfg, std::move(authorizer), nullptr);
  auto ioctx = std::make_shared<rest_ioctx>(n_reactors, std::move(ctx));
  ioctx->start();
  return ioctx;
}

std::shared_ptr<rest_ioctx> make_direct_rest_ioctx(std::string endpoint)
{
  return make_direct_rest_ioctx(std::move(endpoint), direct_rest_test_config());
}

using capture_sink       = sirius::test::recording_log_sink;
using scoped_log_capture = sirius::test::scoped_recording_log_sink;

struct list_watchdog_result {
  bool timed_out{false};
  bool exited_normally{false};
  int exit_code{-1};
};

list_watchdog_result run_list_watchdog(range_http_server const& server,
                                       std::chrono::milliseconds startup_timeout,
                                       std::chrono::milliseconds operation_timeout)
{
  if (sirius::test::g_integration_env != nullptr && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }

  constexpr char k_endpoint_env[] = "SIRIUS_TEST_REST_LIST_WATCHDOG_ENDPOINT";
  std::optional<std::string> old_endpoint;
  if (auto const* current = std::getenv(k_endpoint_env); current != nullptr) {
    old_endpoint = current;
  }
  auto const endpoint = server.endpoint();
  REQUIRE(::setenv(k_endpoint_env, endpoint.c_str(), 1) == 0);

  auto const pid = ::fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    ::execl("/proc/self/exe",
            "sirius_unittest",
            "rest LIST loop watchdog child runner",
            static_cast<char*>(nullptr));
    ::_exit(127);
  }

  if (old_endpoint.has_value()) {
    REQUIRE(::setenv(k_endpoint_env, old_endpoint->c_str(), 1) == 0);
  } else {
    REQUIRE(::unsetenv(k_endpoint_env) == 0);
  }

  int status        = 0;
  bool request_seen = false;
  auto stop         = std::chrono::steady_clock::now() + startup_timeout;
  while (std::chrono::steady_clock::now() < stop) {
    if (!request_seen && server.list_count() > 0) {
      request_seen = true;
      stop         = std::chrono::steady_clock::now() + operation_timeout;
    }
    auto const waited = ::waitpid(pid, &status, WNOHANG);
    if (waited == pid) {
      return {.timed_out       = false,
              .exited_normally = WIFEXITED(status),
              .exit_code       = WIFEXITED(status) ? WEXITSTATUS(status) : -1};
    }
    if (waited < 0) { return {}; }
    std::this_thread::sleep_for(20ms);
  }

  (void)::kill(pid, SIGKILL);
  (void)::waitpid(pid, &status, 0);
  return {.timed_out = true, .exited_normally = false, .exit_code = -1};
}

}  // namespace

TEST_CASE("rest_ioctx lists S3 objects with sizes and follows encoded continuation tokens",
          "[s3][integration][rest][list]")
{
  std::vector<listed_object> objects = {{"data/part-000.parquet", 11},
                                        {"data/year=2024/part-001.parquet", 22},
                                        {"data/year=2025/part-002.parquet", 0},
                                        {"other/ignored.parquet", 99}};
  range_http_server server(deterministic_payload(16), {}, objects);
  auto ctx = make_direct_rest_ioctx(server.endpoint());

  auto listed = ctx->list_objects("bucket", "data/", /*page_size=*/1);

  REQUIRE(listed.size() == 3);
  CHECK(listed[0].key == "data/part-000.parquet");
  CHECK(listed[0].size == 11);
  CHECK(listed[1].key == "data/year=2024/part-001.parquet");
  CHECK(listed[1].size == 22);
  CHECK(listed[2].key == "data/year=2025/part-002.parquet");
  CHECK(listed[2].size == 0);
  CHECK(server.list_count() == 3);
  CHECK(server.get_count() == 0);
  CHECK(server.head_count() == 0);
}

TEST_CASE("rest LIST loop watchdog child runner", "[.][rest][list][watchdog_child]")
{
  auto const* endpoint = std::getenv("SIRIUS_TEST_REST_LIST_WATCHDOG_ENDPOINT");
  if (endpoint == nullptr) { return; }

  auto ctx = make_direct_rest_ioctx(endpoint);
  std::string error;
  try {
    (void)ctx->list_objects("bucket", "loop/", /*page_size=*/1);
  } catch (std::exception const& e) {
    error = e.what();
  }

  INFO(error);
  CHECK((error.find("continuation token did not advance") != std::string::npos ||
         (error.find("truncated") != std::string::npos &&
          error.find("no entries") != std::string::npos)));
}

TEST_CASE("rest_ioctx terminates a repeated-token empty LIST loop", "[s3][integration][rest][list]")
{
  range_fault_policy fault;
  fault.response_delay = 100ms;
  range_http_server server(
    deterministic_payload(16), fault, {}, scripted_list_mode::repeated_empty_token);

  auto const result = run_list_watchdog(server, 10s, 1s);
  INFO("LIST requests=" << server.list_count());
  CHECK_FALSE(result.timed_out);
  CHECK(result.exited_normally);
  CHECK(result.exit_code == 0);
  CHECK(server.list_count() <= 3);
}

TEST_CASE("rest_ioctx terminates an alternating-token empty LIST loop",
          "[s3][integration][rest][list]")
{
  range_fault_policy fault;
  fault.response_delay = 100ms;
  range_http_server server(
    deterministic_payload(16), fault, {}, scripted_list_mode::alternating_empty_tokens);

  auto const result = run_list_watchdog(server, 10s, 1s);
  INFO("LIST requests=" << server.list_count());
  CHECK_FALSE(result.timed_out);
  CHECK(result.exited_normally);
  CHECK(result.exit_code == 0);
  CHECK(server.list_count() <= 3);
}

TEST_CASE("rest_ioctx accepts an empty final LIST page and rejects an empty continuation token",
          "[s3][integration][rest][list]")
{
  SECTION("empty final page")
  {
    range_http_server server(deterministic_payload(16));
    auto ctx    = make_direct_rest_ioctx(server.endpoint());
    auto listed = ctx->list_objects("bucket", "empty/", /*page_size=*/1);

    CHECK(listed.empty());
    CHECK(server.list_count() == 1);
  }

  SECTION("truncated page without a continuation token")
  {
    range_http_server server(
      deterministic_payload(16), {}, {}, scripted_list_mode::truncated_empty_without_token);
    auto ctx = make_direct_rest_ioctx(server.endpoint());

    CHECK_THROWS_WITH(ctx->list_objects("bucket", "empty/", /*page_size=*/1),
                      Catch::Contains("without") && Catch::Contains("continuation token"));
    CHECK(server.list_count() == 1);
  }
}

TEST_CASE("rest_ioctx paged LIST supports early stop and explicit safety caps",
          "[s3][integration][rest][list]")
{
  std::vector<listed_object> objects = {
    {"data/a.parquet", 1}, {"data/b.parquet", 2}, {"data/c.parquet", 3}};

  SECTION("sink false stops after one page")
  {
    range_http_server server(deterministic_payload(16), {}, objects);
    auto ctx = make_direct_rest_ioctx(server.endpoint());

    std::size_t pages = 0;
    ctx->list_objects_paged("bucket",
                            "data/",
                            /*page_size=*/1,
                            [&](sirius::io::rest::s3::list_objects_v2_page const& page) {
                              ++pages;
                              REQUIRE(page.entries.size() == 1);
                              CHECK(page.entries[0].key == "data/a.parquet");
                              return false;
                            });

    CHECK(pages == 1);
    CHECK(server.list_count() == 1);
  }

  SECTION("wrapper throws instead of truncating past max_keys")
  {
    range_http_server server(deterministic_payload(16), {}, objects);
    auto ctx = make_direct_rest_ioctx(server.endpoint());

    CHECK_THROWS_WITH(ctx->list_objects("bucket", "data/", /*page_size=*/1000, /*max_keys=*/2),
                      Catch::Contains("narrow the glob prefix"));

    auto all = ctx->list_objects("bucket", "data/", /*page_size=*/1000, /*max_keys=*/3);
    CHECK(all.size() == 3);
  }

  SECTION("primitive throws instead of scanning unbounded pages")
  {
    range_http_server server(deterministic_payload(16), {}, objects);
    auto ctx = make_direct_rest_ioctx(server.endpoint());

    CHECK_THROWS_WITH(ctx->list_objects_paged(
                        "bucket",
                        "data/",
                        /*page_size=*/1,
                        [](sirius::io::rest::s3::list_objects_v2_page const&) { return true; },
                        /*max_scanned=*/2),
                      Catch::Contains("narrow the glob prefix"));
  }

  SECTION("primitive uses configured scanned cap unless an explicit override is passed")
  {
    auto cfg             = direct_rest_test_config();
    cfg.list_max_scanned = 2;

    range_http_server capped_server(deterministic_payload(16), {}, objects);
    auto capped_ctx = make_direct_rest_ioctx(capped_server.endpoint(), cfg);

    CHECK_THROWS_WITH(capped_ctx->list_objects_paged(
                        "bucket",
                        "data/",
                        /*page_size=*/1,
                        [](sirius::io::rest::s3::list_objects_v2_page const&) { return true; }),
                      Catch::Contains("narrow the glob prefix"));

    range_http_server override_server(deterministic_payload(16), {}, objects);
    auto override_ctx = make_direct_rest_ioctx(override_server.endpoint(), cfg);

    std::size_t pages = 0;
    override_ctx->list_objects_paged(
      "bucket",
      "data/",
      /*page_size=*/1,
      [&](sirius::io::rest::s3::list_objects_v2_page const& page) {
        ++pages;
        REQUIRE(page.entries.size() == 1);
        return true;
      },
      /*max_scanned=*/3);
    CHECK(pages == 3);
    CHECK(override_server.list_count() == 3);
  }

  SECTION("wrapper uses configured matched cap unless an explicit override is passed")
  {
    auto cfg             = direct_rest_test_config();
    cfg.list_max_matches = 2;

    range_http_server capped_server(deterministic_payload(16), {}, objects);
    auto capped_ctx = make_direct_rest_ioctx(capped_server.endpoint(), cfg);

    CHECK_THROWS_WITH(capped_ctx->list_objects("bucket", "data/", /*page_size=*/1000),
                      Catch::Contains("narrow the glob prefix"));

    range_http_server override_server(deterministic_payload(16), {}, objects);
    auto override_ctx = make_direct_rest_ioctx(override_server.endpoint(), cfg);

    auto all = override_ctx->list_objects("bucket", "data/", /*page_size=*/1000, /*max_keys=*/3);
    CHECK(all.size() == 3);
    CHECK(override_server.list_count() == 1);
  }

  SECTION("empty prefix returns an empty vector")
  {
    range_http_server server(deterministic_payload(16), {}, objects);
    auto ctx = make_direct_rest_ioctx(server.endpoint());

    auto none = ctx->list_objects("bucket", "missing/", /*page_size=*/1);
    CHECK(none.empty());
    CHECK(server.list_count() == 1);
  }
}

TEST_CASE("rest_ioctx generated LIST scale obeys configured caps without accumulating pages",
          "[s3][integration][rest][list]")
{
  SECTION("scanned cap triggers page-by-page with one page held at a time")
  {
    range_http_server server(deterministic_payload(16));
    server.set_generated_listing("big/", 6000);
    auto cfg               = direct_rest_test_config();
    cfg.list_max_scanned   = 5000;
    auto ctx               = make_direct_rest_ioctx(server.endpoint(), cfg);
    std::size_t max_seen   = 0;
    std::size_t sink_calls = 0;

    bool threw = false;
    try {
      ctx->list_objects_paged("bucket",
                              "big/",
                              /*page_size=*/1000,
                              [&](sirius::io::rest::s3::list_objects_v2_page const& page) {
                                max_seen = std::max(max_seen, page.entries.size());
                                ++sink_calls;
                                return true;
                              });
    } catch (std::exception const& e) {
      threw            = true;
      auto const error = std::string{e.what()};
      CHECK(error.find("scanned more than 5000") != std::string::npos);
      CHECK(error.find("narrow the glob prefix") != std::string::npos);
    }

    CHECK(threw);
    CHECK(max_seen <= 1000);
    CHECK(sink_calls == 5);
    CHECK(server.list_count() == 6);
  }

  SECTION("matched cap stops the accumulating wrapper")
  {
    range_http_server server(deterministic_payload(16));
    server.set_generated_listing("big/", 600);
    auto cfg             = direct_rest_test_config();
    cfg.list_max_matches = 500;
    auto ctx             = make_direct_rest_ioctx(server.endpoint(), cfg);

    bool threw = false;
    try {
      (void)ctx->list_objects("bucket", "big/", /*page_size=*/1000);
    } catch (std::exception const& e) {
      threw            = true;
      auto const error = std::string{e.what()};
      CHECK(error.find("more than 500") != std::string::npos);
      CHECK(error.find("narrow the glob prefix") != std::string::npos);
    }
    CHECK(threw);
    CHECK(server.list_count() == 1);
  }

  SECTION("under the default caps succeeds and preserves generated key order")
  {
    range_http_server server(deterministic_payload(16));
    server.set_generated_listing("big/", 500);
    auto ctx = make_direct_rest_ioctx(server.endpoint());

    auto listed = ctx->list_objects("bucket", "big/", /*page_size=*/1000);

    REQUIRE(listed.size() == 500);
    CHECK(listed.front().key == "big/0");
    CHECK(listed.front().size == 0);
    CHECK(listed.back().key == "big/499");
    CHECK(listed.back().size == 499);
    CHECK(server.list_count() == 1);
  }

  SECTION("early stop returns after the first generated page even for a large prefix")
  {
    range_http_server server(deterministic_payload(16));
    server.set_generated_listing("big/", 100000);
    auto ctx               = make_direct_rest_ioctx(server.endpoint());
    std::size_t max_seen   = 0;
    std::size_t sink_calls = 0;

    ctx->list_objects_paged("bucket",
                            "big/",
                            /*page_size=*/1000,
                            [&](sirius::io::rest::s3::list_objects_v2_page const& page) {
                              max_seen = std::max(max_seen, page.entries.size());
                              ++sink_calls;
                              return false;
                            });

    CHECK(max_seen <= 1000);
    CHECK(sink_calls == 1);
    CHECK(server.list_count() == 1);
  }
}

TEST_CASE("rest_ioctx generated LIST scale obeys the default safety caps",
          "[.][s3][integration][rest][list][stress]")
{
  SECTION("default scanned cap trips at one million entries without retaining pages")
  {
    range_http_server server(deterministic_payload(16));
    server.set_generated_listing("big/", 1000001);
    auto ctx               = make_direct_rest_ioctx(server.endpoint());
    std::size_t max_seen   = 0;
    std::size_t sink_calls = 0;

    bool threw = false;
    try {
      ctx->list_objects_paged("bucket",
                              "big/",
                              /*page_size=*/1000,
                              [&](sirius::io::rest::s3::list_objects_v2_page const& page) {
                                max_seen = std::max(max_seen, page.entries.size());
                                ++sink_calls;
                                return true;
                              });
    } catch (std::exception const& e) {
      threw            = true;
      auto const error = std::string{e.what()};
      CHECK(error.find("scanned more than 1000000") != std::string::npos);
      CHECK(error.find("narrow the glob prefix") != std::string::npos);
    }

    CHECK(threw);
    CHECK(max_seen <= 1000);
    CHECK(sink_calls == 1000);
    CHECK(server.list_count() == 1001);
  }

  SECTION("default matched cap trips at one hundred thousand accumulated entries")
  {
    range_http_server server(deterministic_payload(16));
    server.set_generated_listing("big/", 100001);
    auto ctx = make_direct_rest_ioctx(server.endpoint());

    bool threw = false;
    try {
      (void)ctx->list_objects("bucket", "big/", /*page_size=*/1000);
    } catch (std::exception const& e) {
      threw            = true;
      auto const error = std::string{e.what()};
      CHECK(error.find("more than 100000") != std::string::npos);
      CHECK(error.find("narrow the glob prefix") != std::string::npos);
    }
    CHECK(threw);
    CHECK(server.list_count() == 101);
  }
}

TEST_CASE("rest_ioctx opens LIST-sized objects without a HEAD round trip",
          "[s3][integration][rest][list][filesystem]")
{
  auto payload = deterministic_payload(4096);
  range_http_server server(payload);
  auto ctx = make_direct_rest_ioctx(server.endpoint());

  auto datasource =
    ctx->open_datasource("s3://bucket/list-sized.bin", static_cast<std::uint64_t>(payload.size()));
  REQUIRE(datasource != nullptr);
  CHECK(datasource->get_io_object().size() == payload.size());
  CHECK(server.head_count() == 0);

  std::vector<std::uint8_t> out(payload.size());
  auto const got = datasource->host_read(0, out.size(), out.data());
  CHECK(got == out.size());
  CHECK(out == payload);
  CHECK(server.get_count() == 1);
}

TEST_CASE("rest reactor dynamically splits one logical read below the physical GET maximum",
          "[s3][integration][rest]")
{
  constexpr std::size_t max_rest_segment = 16UL << 20;
  auto payload                           = deterministic_payload((33UL << 20) + 17);
  range_http_server server(payload);
  auto config            = direct_rest_test_config();
  config.max_connections = 4;
  auto ioctx             = make_direct_rest_ioctx(server.endpoint(), config);
  auto datasource        = ioctx->open_datasource("s3://split-bucket/object.bin", payload.size());
  std::vector<std::uint8_t> destination(payload.size());
  std::vector<sirius::io::slice> slices;
  slices.emplace_back(0, destination.size(), destination.data());

  auto future = ioctx->host_readv_async_io(datasource->get_io_object(), slices);
  REQUIRE(std::move(future).get(10s) == payload.size());

  CHECK(server.get_count() > 1);
  CHECK(server.max_requested_range() <= max_rest_segment);
  require_bytes_equal(destination, payload);
}

TEST_CASE("rest cache fill at the object tail is clipped to EOF", "[s3][integration][rest][cache]")
{
  constexpr std::size_t page_size = 4096;
  constexpr std::size_t tail_size = 17;
  auto payload                    = deterministic_payload(page_size + tail_size);
  range_http_server server(payload);
  scan_manager_fixture fixture;
  auto cfg       = make_fake_rest_config(server.endpoint());
  cfg.cache.mode = sirius::io::cache::cache_mode::sirius;
  sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};
  auto datasource = manager.create_datasource("s3://tail-cache-bucket/object.bin");
  REQUIRE(datasource != nullptr);

  std::array<cudf::io::text::byte_range_info, 1> ranges{
    cudf::io::text::byte_range_info{page_size, tail_size}};
  datasource->fadvise(ranges, std::nullopt);
  REQUIRE(datasource->prepare_prefetch(false) == sirius::io::prepare_result::prepared);

  std::array<std::uint8_t, tail_size> destination{};
  REQUIRE(datasource->host_read(page_size, destination.size(), destination.data()) == tail_size);
  require_bytes_equal(destination,
                      std::span<std::uint8_t const>(payload.data() + page_size, tail_size));
  CHECK(server.max_requested_range() == payload.size());
}

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
      dynamic_cast<sirius::io::rest::rest_io_object const*>(&datasource->get_io_object());

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
      dynamic_cast<sirius::io::rest::rest_io_object const*>(&datasource->get_io_object());

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
    fault.successful_get_etag   = "\"discarded-200-tag\"";
    range_http_server server(parquet, fault);
    auto ioctx      = make_direct_rest_ioctx(server.endpoint());
    auto datasource = ioctx->open_datasource("s3://footer-bucket/nation.parquet",
                                             sirius::io::open_hint::parquet_footer_probe);
    REQUIRE(datasource != nullptr);
    CHECK(datasource->size() == parquet.size());
    CHECK(datasource->get_io_object().validation_tag().empty());
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

  SECTION("missing object fails the footer-probe open after HEAD fallback")
  {
    range_fault_policy fault{};
    fault.fail_all_gets    = true;
    fault.fail_all_heads   = true;
    fault.fail_status      = 404;
    fault.fail_head_status = 404;
    range_http_server server(parquet, fault);
    auto ioctx = make_direct_rest_ioctx(server.endpoint());

    CHECK_THROWS_WITH(ioctx->open_datasource("s3://footer-bucket/missing.parquet",
                                             sirius::io::open_hint::parquet_footer_probe),
                      Catch::Matchers::Contains("HTTP 404"));
  }

  SECTION("forbidden object fails the footer-probe open after HEAD fallback")
  {
    range_fault_policy fault{};
    fault.fail_all_gets    = true;
    fault.fail_all_heads   = true;
    fault.fail_status      = 403;
    fault.fail_head_status = 403;
    range_http_server server(parquet, fault);
    auto ioctx = make_direct_rest_ioctx(server.endpoint());

    CHECK_THROWS_WITH(ioctx->open_datasource("s3://footer-bucket/forbidden.parquet",
                                             sirius::io::open_hint::parquet_footer_probe),
                      Catch::Matchers::Contains("HTTP 403"));
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
    fault.failed_get_etag = "\"stale-retry-tag\"";
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
    CHECK(probe.etag.empty());
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.bytes->size() == parquet.size());
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 3);
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
  }

  SECTION("clean 206 succeeds without retries")
  {
    range_fault_policy fault{};
    fault.successful_get_etag = "\"footer-v1\"";
    range_http_server server(parquet, fault);
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
    CHECK(probe.etag == "\"footer-v1\"");
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.bytes->size() == parquet.size());
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 1);
  }
}

TEST_CASE("HEAD object-size retries preserve retry behavior", "[s3][integration][rest][footerbind]")
{
  auto const payload = deterministic_payload(4096);

  SECTION("transient 503s are retried and the final HEAD returns the size")
  {
    range_fault_policy fault{};
    fault.fail_first_heads     = 2;
    fault.fail_head_status     = 503;
    fault.successful_head_etag = "\"head-v2\"";
    range_http_server server(payload, fault);
    auto authorizer             = std::make_shared<fixed_url_authorizer>(server.endpoint());
    auto cfg                    = direct_rest_test_config();
    cfg.max_retry_attempts      = 3;
    cfg.max_auth_retry_attempts = 1;
    auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, std::move(authorizer), nullptr);
    sirius::io::rest::rest_reactor reactor(ctx, "head-retry-success");

    auto const result = reactor.head_object("head-bucket", "head-success.bin");
    CHECK(result.object_size == payload.size());
    CHECK(result.etag == "\"head-v2\"");
    CHECK(server.head_count() == 3);
    CHECK(server.get_count() == 0);
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
      (void)reactor.head_object("head-bucket", "head-exhausted.bin");
      FAIL("head_object should throw after exhausting transient retries");
    } catch (std::runtime_error const& e) {
      auto const message = std::string{e.what()};
      CHECK(message.find("exhausted retries") != std::string::npos);
      CHECK(message.find("HTTP 503") != std::string::npos);
    }
    CHECK(server.head_count() == 2);
    CHECK(server.get_count() == 0);
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
      (void)reactor.head_object("head-bucket", "head-forbidden.bin");
      FAIL("head_object should throw on a hard non-retriable HTTP error");
    } catch (std::runtime_error const& e) {
      auto const message = std::string{e.what()};
      CHECK(message.find("HTTP 403") != std::string::npos);
    }
    CHECK(server.head_count() == 1);
    CHECK(server.get_count() == 0);
  }

  SECTION("clean HEAD succeeds without retries")
  {
    range_http_server server(payload);
    auto authorizer             = std::make_shared<fixed_url_authorizer>(server.endpoint());
    auto cfg                    = direct_rest_test_config();
    cfg.max_retry_attempts      = 3;
    cfg.max_auth_retry_attempts = 1;
    auto ctx                    = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, std::move(authorizer), nullptr);
    sirius::io::rest::rest_reactor reactor(ctx, "head-clean");

    auto const result = reactor.head_object("head-bucket", "head-clean.bin");
    CHECK(result.object_size == payload.size());
    CHECK(result.etag.empty());
    CHECK(server.head_count() == 1);
    CHECK(server.get_count() == 0);
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
  cfg.rest.merge_max_gap   = 0;  // one GET per range: no bridging across the gaps
  sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};
  auto datasource = manager.create_datasource("s3://" + bucket + "/medium.bin");
  require_rest_ioctx(datasource);

  std::vector<std::vector<std::uint8_t>> buffers;
  std::vector<sirius::io::slice> segments;
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

  auto got = std::move(datasource->io_ctx()->host_readv_async_io(
                         datasource->get_io_object(), std::span<const sirius::io::slice>(segments)))
               .get(5s);
  REQUIRE(got == total);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    auto const [offset, size] = ranges[i];
    require_bytes_equal(buffers[i], std::span<std::uint8_t const>(medium.data() + offset, size));
  }
}

TEST_CASE("rest_ioctx keeps caller host buffers until the device event completes",
          "[s3][integration][rest]")
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    WARN("Skipping rest_ioctx caller-host device-event test: no CUDA device");
    return;
  }

  auto payload = deterministic_payload(256 * 1024);
  range_http_server server(payload);
  auto ioctx      = make_direct_rest_ioctx(server.endpoint());
  auto datasource = ioctx->open_datasource("s3://device-event-bucket/object.bin", payload.size());

  void* raw_host = nullptr;
  REQUIRE(cudaMallocHost(&raw_host, payload.size()) == cudaSuccess);
  unique_cuda_host_buffer host{static_cast<std::uint8_t*>(raw_host)};

  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);
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

  std::jthread release_gate([&gate] {
    std::this_thread::sleep_for(500ms);
    gate.release.store(true, std::memory_order_release);
  });
  auto const gate_deadline = std::chrono::steady_clock::now() + 5s;
  while (!gate.entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < gate_deadline) {
    std::this_thread::yield();
  }
  REQUIRE(gate.entered.load(std::memory_order_acquire));
  std::vector<sirius::io::prepared_io_slice> slices;
  slices.emplace_back(
    sirius::io::range{0, payload.size()},
    sirius::io::host_buffer{host.get()},
    sirius::io::device_buffer{static_cast<std::uint8_t*>(destination.data()), stream});

  auto future = ioctx->host_device_readv_async_io(datasource->get_io_object(), std::move(slices));
  REQUIRE(std::move(future).get(5s) == payload.size());

  // The future owns the caller's source lifetime through the recorded event. If
  // it settled at enqueue time, this overwrite would race the gated H2D copy.
  std::memset(host.get(), 0, payload.size());
  auto got = copy_device_to_host(destination, payload.size(), stream);
  require_bytes_equal(got, payload);
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
  cfg.rest.merge_max_gap   = 0;  // keep the 8 segments as 8 separate GETs
  sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://concurrency-bucket/object.bin");
  require_rest_ioctx(datasource);

  std::vector<std::vector<std::uint8_t>> buffers;
  std::vector<sirius::io::slice> segments;
  for (std::size_t i = 0; i < 8; ++i) {
    std::size_t const offset = i * 4096 + 13;
    std::size_t const size   = 512;
    buffers.emplace_back(size);
    segments.emplace_back(offset, size, buffers.back().data());
  }

  auto got = std::move(datasource->io_ctx()->host_readv_async_io(
                         datasource->get_io_object(), std::span<const sirius::io::slice>(segments)))
               .get(10s);
  REQUIRE(got == 8 * 512);
  CHECK(server.peak_active_gets() <= 2);
  CHECK(server.peak_active_gets() >= 1);
  for (std::size_t i = 0; i < segments.size(); ++i) {
    require_bytes_equal(
      buffers[i],
      std::span<std::uint8_t const>(payload.data() + segments[i].offset(), segments[i].size()));
  }
}

// One batched request must be driven by the whole reactor pool, not by whichever
// reactor the round-robin happened to land on.  max_connections is per reactor,
// so with a one-connection reactor the concurrency the server observes IS the
// number of reactors the request reached: a single-reactor dispatch can never
// exceed one in-flight GET no matter how many segments it was handed.
TEST_CASE("rest_ioctx spreads one batched range read across the reactor pool",
          "[s3][integration][rest]")
{
  auto payload = deterministic_payload(512 * 1024);
  range_fault_policy fault{};
  fault.response_delay = 50ms;
  range_http_server server(payload, fault);
  scan_manager_fixture fixture;
  auto cfg                 = make_fake_rest_config(server.endpoint());
  cfg.rest_n_reactors      = 4;
  cfg.rest.max_connections = 1;
  cfg.rest.merge_max_gap   = 0;  // keep the 8 segments as 8 separate GETs
  sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://fanout-bucket/object.bin");
  require_rest_ioctx(datasource);

  // Two chunks per reactor, so the split is even and every reactor has work
  // queued behind its single connection.
  std::vector<std::vector<std::uint8_t>> buffers;
  std::vector<sirius::io::slice> segments;
  for (std::size_t i = 0; i < 8; ++i) {
    std::size_t const offset = i * 4096 + 13;
    std::size_t const size   = 512;
    buffers.emplace_back(size);
    segments.emplace_back(offset, size, buffers.back().data());
  }

  auto got = std::move(datasource->io_ctx()->host_readv_async_io(
                         datasource->get_io_object(), std::span<const sirius::io::slice>(segments)))
               .get(10s);
  REQUIRE(got == 8 * 512);
  // > 1 is the fan-out itself; <= n_reactors is max_connections still holding
  // per reactor rather than the pool becoming unbounded.
  CHECK(server.peak_active_gets() > 1);
  CHECK(server.peak_active_gets() <= 4);
  for (std::size_t i = 0; i < segments.size(); ++i) {
    require_bytes_equal(
      buffers[i],
      std::span<std::uint8_t const>(payload.data() + segments[i].offset(), segments[i].size()));
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
