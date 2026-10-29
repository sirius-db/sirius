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
#include "io/ioctx_resolver.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/parquet_metadata.hpp"
#include "op/scan/table_scan/parquet_physical_profile.hpp"
#include "planner/query.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "scan_manager/split_provider.hpp"
#include "telemetry-bridge/gen/quent.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/s3_test_env.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <arpa/inet.h>
#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using cucascade::io::io_context_registry;
using cucascade::io::io_context_type;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;
using sirius::test::s3::single_gpu_index;

std::filesystem::path make_regular_file()
{
  auto dir =
    std::filesystem::temp_directory_path() / ("sirius-s3-routing-" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  auto file =
    dir / ("local-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
           ".parquet");
  std::ofstream out(file, std::ios::binary);
  out << "not a parquet file; routing only\n";
  out.close();
  REQUIRE(std::filesystem::is_regular_file(file));
  return file;
}

std::filesystem::path project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path{SIRIUS_PROJECT_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

std::vector<std::uint8_t> read_binary_file(std::filesystem::path const& path)
{
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  std::vector<char> chars((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return std::vector<std::uint8_t>(chars.begin(), chars.end());
}

void prime_parquet_metadata(cucascade::io::datasource& source, std::string const& path)
{
  auto const options = cudf::io::parquet_reader_options::builder().build();
  auto metadata      = sirius::op::scan::resolve_parquet_metadata(source, 0, path, options);
  REQUIRE(metadata != nullptr);
  REQUIRE(metadata->built_for_identity == source.get_io_object().identity_cache_key());
  REQUIRE(source.metadata() == metadata);
}

std::string strip_file_scheme_for_registry(std::string const& path)
{
  static constexpr std::string_view kFile = "file://";
  if (path.size() <= kFile.size()) { return path; }
  for (std::size_t i = 0; i < kFile.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(path[i])) != static_cast<unsigned char>(kFile[i])) {
      return path;
    }
  }
  return path.substr(kFile.size());
}

bool is_local_backend(std::optional<io_context_type> type)
{
  return type.has_value() && *type == io_context_type::uring;
}

scan_manager_config make_s3_scan_config(std::string endpoint,
                                        sirius::scan_manager::io_backend backend)
{
  scan_manager_config cfg{};
  cfg.backend                      = backend;
  cfg.object_store.endpoint        = std::move(endpoint);
  cfg.object_store.region          = "us-east-1";
  cfg.object_store.access_key      = "routing-test-access-key";
  cfg.object_store.secret_key      = "routing-test-secret-key";
  cfg.object_store.tls_verify      = false;
  cfg.rest.request_timeout_s       = 2;
  cfg.rest.max_retry_attempts      = 1;
  cfg.rest.max_auth_retry_attempts = 1;
  cfg.rest.retry_backoff_base      = std::chrono::milliseconds{0};
  cfg.rest.retry_jitter            = std::chrono::milliseconds{0};
  cfg.rest.honor_retry_after       = false;
  cfg.rest.max_connections         = 1;
  cfg.thread_pool.num_threads      = 1;
  cfg.uring_n_reactors             = 1;
  cfg.rest_n_reactors              = 1;
  cfg.cache.mode                   = cucascade::io::cache::cache_mode::none;
  return cfg;
}

struct scan_manager_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  std::shared_ptr<const sirius::memory::topology_index> topology =
    single_gpu_index(/*numa_node=*/0);
};

std::unique_ptr<sirius::op::scan::parquet_ingestible_table_info> make_nation_table_info(
  std::vector<std::string> uris)
{
  auto info                 = std::make_unique<sirius::op::scan::parquet_ingestible_table_info>();
  info->resolved_file_paths = std::move(uris);
  info->names               = {"n_nationkey", "n_name", "n_regionkey", "n_comment"};
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::VARCHAR));
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::VARCHAR));
  for (duckdb::idx_t i = 0; i < info->returned_types.size(); ++i) {
    info->column_ids.push_back(duckdb::ColumnIndex(i));
  }
  info->projected_output_count = info->returned_types.size();
  return info;
}

[[maybe_unused]] std::unique_ptr<sirius::op::scan::parquet_ingestible_table_info>
make_nation_table_info(std::string uri)
{
  std::vector<std::string> uris;
  uris.push_back(std::move(uri));
  return make_nation_table_info(std::move(uris));
}

using routing_observations = std::unordered_map<std::string, io_context_type>;

routing_observations collect_routing_observations(
  sirius::op::scan::parquet_gpu_ingestible& ingestible, sirius::io::ioctx_resolver resolver)
{
  routing_observations out;
  while (ingestible.has_processed_all_metadata() == false) {
    auto task = ingestible.next_split_provider(resolver);
    if (!task) { continue; }
    auto scan = task({});
    REQUIRE(scan != nullptr);
    auto* file_scan = dynamic_cast<sirius::op::scan::parquet_file_scan_info*>(scan.get());
    REQUIRE(file_scan != nullptr);
    REQUIRE(file_scan->datasource != nullptr);
    REQUIRE(file_scan->datasource->io_ctx() != nullptr);
    out.emplace(file_scan->file_path, file_scan->datasource->io_ctx()->type());
  }
  return out;
}

sirius::io::ioctx_resolver make_datasource_resolver(sirius_scan_manager& manager)
{
  return [&manager](std::string_view path) -> std::shared_ptr<cucascade::io::ioctx> {
    auto ds = manager.create_datasource(path);
    if (!ds) {
      throw std::runtime_error("test datasource resolver: no backend supports path: " +
                               std::string(path));
    }
    return ds->io_ctx();
  };
}

sirius::planner::query make_empty_query(sirius::query_id_t query_id = sirius::make_query_id(1))
{
  auto tctx = sirius::test::make_test_telemetry_context();
  sirius::telemetry::query_telemetry_info tinfo{quent::now_v7(), tctx->worker_id().raw(), query_id};
  return sirius::planner::query(std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>>{},
                                tctx->context(),
                                query_id,
                                tinfo);
}

struct range_fault_policy {
  std::size_t fail_first_gets{0};
  bool fail_all_gets{false};
  int fail_status{503};
  /// Strong validator sent on HEAD 200, GET 206 and GET 200 when non-empty. Empty keeps the
  /// object tagless: every open then gets a per-open identity and shares nothing across opens.
  std::string etag;
  /// Held before every GET response is sent, so a read stays in flight that long.
  std::chrono::milliseconds response_delay{0};
};

class range_s3_server {
 public:
  explicit range_s3_server(std::vector<std::uint8_t> object, range_fault_policy fault = {})
    : _object(std::move(object)), _fault(fault)
  {
    if (_object.empty()) { throw std::runtime_error("range_s3_server object must be non-empty"); }
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
    if (::listen(_listen_fd, 16) != 0) {
      throw std::runtime_error("listen failed: " + errno_message());
    }

    socklen_t len = sizeof(addr);
    if (::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
      throw std::runtime_error("getsockname failed: " + errno_message());
    }
    _port   = ntohs(addr.sin_port);
    _thread = std::thread([this] { accept_loop(); });
  }

  ~range_s3_server()
  {
    _stop.store(true);
    if (_listen_fd >= 0) {
      ::shutdown(_listen_fd, SHUT_RDWR);
      ::close(_listen_fd);
      _listen_fd = -1;
    }
    if (_thread.joinable()) { _thread.join(); }
  }

  range_s3_server(range_s3_server const&)            = delete;
  range_s3_server& operator=(range_s3_server const&) = delete;

  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }

  /// Every request the server has served, of any method.
  [[nodiscard]] int request_count() const { return _request_count.load(std::memory_order_relaxed); }
  /// Accepted TCP connections, including native REST's request-free warmup.
  [[nodiscard]] int connection_count() const
  {
    return _connection_count.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::size_t head_count() const { return _head_count.load(); }
  [[nodiscard]] std::size_t get_count() const { return _get_count.load(); }
  /// GETs that named a suffix range (`Range: bytes=-N`), the footer probe's shape.
  [[nodiscard]] std::size_t suffix_get_count() const { return _suffix_get_count.load(); }
  /// The largest exclusive range end any GET asked for, before the server clipped it.
  [[nodiscard]] std::size_t max_requested_end() const { return _max_requested_end.load(); }

 private:
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
      _connection_count.fetch_add(1, std::memory_order_relaxed);
      handle_client(fd);
      ::close(fd);
    }
  }

  void handle_client(int fd)
  {
    timeval timeout{};
    timeout.tv_sec = 2;
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    std::string request;
    request.resize(4096);
    ssize_t n = ::recv(fd, request.data(), request.size(), 0);
    if (n <= 0) { return; }
    request.resize(static_cast<std::size_t>(n));
    ++_request_count;

    bool const is_head = request.rfind("HEAD ", 0) == 0;
    bool const is_get  = request.rfind("GET ", 0) == 0;
    std::string response;
    if (is_head) {
      _head_count.fetch_add(1, std::memory_order_relaxed);
      response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size()) + "\r\n" +
                 etag_header() + "Connection: close\r\n\r\n";
      send_all(fd, response);
    } else if (is_get) {
      auto const get_idx = _get_count.fetch_add(1, std::memory_order_relaxed);
      if (_fault.response_delay.count() > 0) { std::this_thread::sleep_for(_fault.response_delay); }
      if (_fault.fail_all_gets || get_idx < _fault.fail_first_gets) {
        response = "HTTP/1.1 " + std::to_string(_fault.fail_status) +
                   " Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        send_all(fd, response);
        return;
      }
      if (auto range = parse_range(request)) {
        auto const [start, end, requested_end, suffix] = *range;
        if (suffix) { _suffix_get_count.fetch_add(1, std::memory_order_relaxed); }
        // Single writer: requests are served one at a time on the accept thread.
        if (requested_end > _max_requested_end.load()) { _max_requested_end.store(requested_end); }
        auto const len = end - start + 1;
        response       = "HTTP/1.1 206 Partial Content\r\nContent-Length: " + std::to_string(len) +
                   "\r\nContent-Range: bytes " + std::to_string(start) + "-" + std::to_string(end) +
                   "/" + std::to_string(_object.size()) + "\r\n" + etag_header() +
                   "Connection: close\r\n\r\n";
        send_all(fd, response);
        send_all(fd, _object.data() + start, len);
      } else {
        response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size()) + "\r\n" +
                   etag_header() + "Connection: close\r\n\r\n";
        send_all(fd, response);
        send_all(fd, _object.data(), _object.size());
      }
    } else {
      response =
        "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n"
        "Connection: close\r\n\r\n";
      send_all(fd, response);
    }
  }

  [[nodiscard]] std::string etag_header() const
  {
    return _fault.etag.empty() ? std::string{} : "ETag: " + _fault.etag + "\r\n";
  }

  static void send_all(int fd, std::string_view bytes)
  {
    send_all(fd, reinterpret_cast<std::uint8_t const*>(bytes.data()), bytes.size());
  }

  static void send_all(int fd, std::uint8_t const* bytes, std::size_t size)
  {
    std::size_t sent = 0;
    while (sent < size) {
      ssize_t n = ::send(fd, bytes + sent, size - sent, MSG_NOSIGNAL);
      if (n <= 0) { return; }
      sent += static_cast<std::size_t>(n);
    }
  }

  struct byte_range {
    std::size_t start;          ///< First byte served.
    std::size_t end;            ///< Last byte served (inclusive), clipped to the object.
    std::size_t requested_end;  ///< Exclusive end as requested, before clipping.
    bool suffix;                ///< `bytes=-N`.
  };

  [[nodiscard]] std::optional<byte_range> parse_range(std::string const& request) const
  {
    std::string lower = request;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    auto const prefix = std::string{"range: bytes="};
    auto pos          = lower.find(prefix);
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
        if (dash + 1 < spec.size()) {
          end = static_cast<std::size_t>(std::stoull(spec.substr(dash + 1)));
        }
      }
      auto const requested_end = end + 1;
      if (start >= _object.size()) { return std::nullopt; }
      end = std::min(end, _object.size() - 1);
      if (end < start) { return std::nullopt; }
      return byte_range{start, end, requested_end, dash == 0};
    } catch (...) {
      return std::nullopt;
    }
  }

  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::vector<std::uint8_t> _object;
  range_fault_policy _fault;
  std::atomic<bool> _stop{false};
  std::atomic<int> _request_count{0};
  std::atomic<int> _connection_count{0};
  std::atomic<std::size_t> _head_count{0};
  std::atomic<std::size_t> _get_count{0};
  std::atomic<std::size_t> _suffix_get_count{0};
  std::atomic<std::size_t> _max_requested_end{0};
  std::thread _thread;
};

}  // namespace

TEST_CASE("io_context_registry routes full paths to their own backends", "[s3][routing]")
{
  scan_manager_fixture fixture;
  auto cfg = make_s3_scan_config("http://127.0.0.1:1", sirius::scan_manager::io_backend::native);
  auto const local_path = make_regular_file();

  io_context_registry registry{cfg.to_io_config(), *fixture.memory};

  CHECK(registry.lookup_path("s3://bucket/key.parquet") == io_context_type::restful);
  CHECK(is_local_backend(registry.lookup_path(local_path.string())));

  auto file_uri = std::string{"file://"} + local_path.string();
  auto stripped = strip_file_scheme_for_registry(file_uri);
  CHECK(is_local_backend(registry.lookup_path(stripped)));
  CHECK(registry.lookup_path(stripped) != io_context_type::restful);
  CHECK_FALSE(registry.lookup_path("http://localhost/key.parquet").has_value());
  CHECK_FALSE(registry.lookup_path("https://localhost/key.parquet").has_value());
  CHECK_FALSE(registry.lookup_path("unsupported://bucket/key.parquet").has_value());
}

TEST_CASE("io_context_registry routes local paths to uring under backend=native", "[s3][routing]")
{
  scan_manager_fixture fixture;
  auto const local_path = make_regular_file();

  io_context_registry native_registry{
    make_s3_scan_config("http://127.0.0.1:1", sirius::scan_manager::io_backend::native)
      .to_io_config(),
    *fixture.memory};
  CHECK(native_registry.lookup_path(local_path.string()) == io_context_type::uring);
}

TEST_CASE("scan_manager concurrent first-touch reuses one routed S3 ioctx", "[s3][routing]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{0}));
  scan_manager_fixture fixture;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle,
    make_s3_scan_config(server.endpoint(), sirius::scan_manager::io_backend::native),
    *fixture.memory,
    fixture.topology};

  auto constexpr kThreads = std::size_t{16};
  auto const uri          = std::string{"s3://routing-bucket/data.parquet"};
  std::atomic<std::size_t> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::shared_ptr<cucascade::io::ioctx>> ioctxs(kThreads);
  std::vector<std::exception_ptr> errors(kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (std::size_t i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      try {
        ready.fetch_add(1, std::memory_order_acq_rel);
        while (!go.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }

        auto datasource = manager.create_datasource(uri);
        if (datasource == nullptr) {
          throw std::runtime_error("create_datasource returned nullptr");
        }
        if (datasource->io_ctx() == nullptr) {
          throw std::runtime_error("datasource has no ioctx");
        }
        ioctxs[i] = datasource->io_ctx();
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != kThreads) {
    std::this_thread::yield();
  }
  go.store(true, std::memory_order_release);

  for (auto& thread : threads) {
    thread.join();
  }
  for (auto const& error : errors) {
    if (error) { std::rethrow_exception(error); }
  }

  REQUIRE(ioctxs[0] != nullptr);
  auto* const expected = ioctxs[0].get();
  REQUIRE(ioctxs[0]->type() == io_context_type::restful);
  for (auto const& ioctx : ioctxs) {
    REQUIRE(ioctx != nullptr);
    CHECK(ioctx->type() == io_context_type::restful);
    CHECK(ioctx.get() == expected);
  }

  auto datasource = manager.create_datasource(uri);
  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  CHECK(datasource->io_ctx().get() == expected);
}

TEST_CASE("scan_manager planning backend lookup does not open remote objects", "[s3][routing]")
{
  scan_manager_fixture fixture;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle,
    make_s3_scan_config("http://127.0.0.1:1", sirius::scan_manager::io_backend::native),
    *fixture.memory,
    fixture.topology};

  // Port 1 has no test server. A successful lookup proves the planner only
  // consulted URI/backend capability and did not construct an ioctx or issue
  // a HEAD/credential probe.
  CHECK(manager.planning_io_backend("s3://planning-bucket/data/a.puffin") ==
        io_context_type::restful);
}

TEST_CASE("scan_manager rejects raw HTTP and HTTPS URLs with S3 URI guidance", "[s3][routing]")
{
  auto const uri          = GENERATE("http://127.0.0.1:1/routing-bucket/data.parquet",
                            "https://127.0.0.1:1/routing-bucket/data.parquet");
  auto const footer_probe = GENERATE(false, true);
  CAPTURE(uri, footer_probe);
  scan_manager_fixture fixture;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle,
    make_s3_scan_config("http://127.0.0.1:1", sirius::scan_manager::io_backend::native),
    *fixture.memory,
    fixture.topology};

  // Both public entry points reach ioctx_for_path. Even with an object store
  // configured, raw web URLs must be rejected before any HEAD or footer read.
  try {
    if (footer_probe) {
      (void)manager.describe_parquet(uri);
    } else {
      (void)manager.create_datasource(uri);
    }
    FAIL("raw HTTP(S) URL unexpectedly accepted");
  } catch (std::invalid_argument const& error) {
    CHECK_THAT(std::string{error.what()},
               Catch::Matchers::ContainsSubstring("unsupported URL scheme") &&
                 Catch::Matchers::ContainsSubstring("s3://"));
  }
}

TEST_CASE("scan_manager create_datasource normalizes file URI paths before routing",
          "[s3][routing]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{0}));
  scan_manager_fixture fixture;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle,
    make_s3_scan_config(server.endpoint(), sirius::scan_manager::io_backend::native),
    *fixture.memory,
    fixture.topology};

  auto const local_path = make_regular_file();
  auto datasource       = manager.create_datasource("file://" + local_path.string());

  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  CHECK(datasource->io_ctx()->type() == io_context_type::uring);
}

TEST_CASE("scan_manager re-primes routed S3 cache on every query", "[s3][routing]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{0}));
  scan_manager_fixture fixture;
  auto cfg       = make_s3_scan_config(server.endpoint(), sirius::scan_manager::io_backend::native);
  cfg.cache.mode = cucascade::io::cache::cache_mode::cucs;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{scan_lifecycle, cfg, *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://routing-bucket/data.parquet");
  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  auto* routed_cache = datasource->io_ctx()->cache();
  REQUIRE(routed_cache != nullptr);
  REQUIRE(routed_cache->query_epoch() == 0);

  auto* default_cache = manager.io_ctx()->cache();
  REQUIRE(default_cache != nullptr);
  REQUIRE(default_cache->query_epoch() == 0);

  // The queries intentionally have no scan operators: routed caches must still
  // advance once per query, matching the default ioctx's query-wide refresh.
  // Distinct query ids because per-query state is keyed by id — the epoch bump is per
  // prepare_for_query call, and using one id twice would exercise the stale-state replacement
  // path rather than the two-queries case this gate is about.
  auto q1 = make_empty_query(sirius::make_query_id(1));
  manager.prepare_for_query(q1, true, {});
  REQUIRE(routed_cache->query_epoch() == 1);
  REQUIRE(default_cache->query_epoch() == 1);

  auto q2 = make_empty_query(sirius::make_query_id(2));
  manager.prepare_for_query(q2, true, {});
  REQUIRE(routed_cache->query_epoch() == 2);
  REQUIRE(default_cache->query_epoch() == 2);
}

TEST_CASE("scan_manager tolerates a routed S3 ioctx when cache.mode is none", "[s3][routing]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{0}));
  scan_manager_fixture fixture;
  auto cfg = make_s3_scan_config(server.endpoint(), sirius::scan_manager::io_backend::native);
  REQUIRE_FALSE(cfg.cache.use_fs_cache());
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{scan_lifecycle, cfg, *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://routing-bucket/data.parquet");
  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  CHECK(datasource->io_ctx()->type() == io_context_type::restful);
  REQUIRE(datasource->io_ctx()->cache() == nullptr);

  auto q = make_empty_query();
  REQUIRE_NOTHROW(manager.prepare_for_query(q, true, {}));
}

TEST_CASE("warmup opens every reactor's connection pool, and only once per bucket", "[s3][routing]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{7}));
  scan_manager_fixture fixture;
  auto cfg = make_s3_scan_config(server.endpoint(), sirius::scan_manager::io_backend::native);
  // One connection per reactor over two reactors: small enough for the serial
  // test server to serve the burst, but still per-reactor rather than global
  // (each reactor owns its own, thread-confined connection pool).
  cfg.rest.max_connections = 1;
  // Keep the rate-limit window above the serial server's accept/receive deadlines.
  cfg.rest.conn_max_age = std::chrono::seconds{30};
  cfg.rest_n_reactors   = 2;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{scan_lifecycle, cfg, *fixture.memory, fixture.topology};

  // Routes s3:// to the REST ioctx and starts its reactors. The size HEAD
  // establishes the baseline; native warmup opens sockets without sending HEADs.
  auto datasource = manager.create_datasource("s3://warm-bucket/data.parquet");
  REQUIRE(datasource != nullptr);
  auto io_ctx = datasource->io_ctx();
  REQUIRE(io_ctx != nullptr);

  auto const wait_for_connections = [&server](int at_least) {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (server.connection_count() < at_least && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return server.connection_count();
  };

  int const baseline          = server.connection_count();
  int const baseline_requests = server.request_count();
  constexpr int expected_warm = 2;  // rest_n_reactors * max_connections

  // A bucket URL, with no object in it: warmup opens TCP connections only.
  io_ctx->warmup("s3://warm-bucket");
  CHECK(wait_for_connections(baseline + expected_warm) >= baseline + expected_warm);
  CHECK(server.request_count() == baseline_requests);
  int const after_warm = server.connection_count();

  // Warming the same bucket again well inside conn_max_age is redundant — the
  // pools it would fill are already full — so it must not reach the server.
  io_ctx->warmup("s3://warm-bucket");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  CHECK(server.connection_count() == after_warm);
  CHECK(server.request_count() == baseline_requests);

  // An object URL is accepted and its key ignored: same bucket, still rate-limited.
  io_ctx->warmup("s3://warm-bucket/some/deep/key.parquet");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  CHECK(server.connection_count() == after_warm);
  CHECK(server.request_count() == baseline_requests);
}

TEST_CASE("warmup is a no-op for backends with nothing to connect", "[s3][routing]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{7}));
  scan_manager_fixture fixture;
  auto cfg = make_s3_scan_config(server.endpoint(), sirius::scan_manager::io_backend::native);
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{scan_lifecycle, cfg, *fixture.memory, fixture.topology};

  // The default ioctx is the local uring one: a file it can open is already
  // "connected", so the base class no-op is the whole implementation.
  REQUIRE(manager.io_ctx() != nullptr);
  manager.io_ctx()->warmup("s3://warm-bucket");
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  CHECK(server.request_count() == 0);
}

TEST_CASE("parquet_gpu_ingestible resolver routes each parquet file independently", "[s3][routing]")
{
  auto const fixture_path = project_root() / "test/cpp/integration/data/parquet/nation.parquet";
  auto parquet_bytes      = read_binary_file(fixture_path);
  range_s3_server server(std::move(parquet_bytes));
  scan_manager_fixture fixture;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle,
    make_s3_scan_config(server.endpoint(), sirius::scan_manager::io_backend::native),
    *fixture.memory,
    fixture.topology};

  std::string const s3_uri     = "s3://routing-bucket/nation.parquet";
  std::string const local_path = fixture_path.string();

  auto s3_ds = manager.create_datasource(s3_uri);
  REQUIRE(s3_ds != nullptr);
  REQUIRE(s3_ds->io_ctx() != nullptr);
  REQUIRE(s3_ds->io_ctx()->type() == io_context_type::restful);
  prime_parquet_metadata(*s3_ds, s3_uri);

  auto local_ds = manager.create_datasource(local_path);
  REQUIRE(local_ds != nullptr);
  REQUIRE(local_ds->io_ctx() != nullptr);
  REQUIRE(is_local_backend(local_ds->io_ctx()->type()));
  prime_parquet_metadata(*local_ds, local_path);

  auto ingestible = sirius::op::scan::make_ingestible(
    make_nation_table_info(std::vector<std::string>{s3_uri, local_path}));
  auto routed = collect_routing_observations(*ingestible, make_datasource_resolver(manager));

  REQUIRE(routed.size() == 2);
  REQUIRE(routed.contains(s3_uri));
  REQUIRE(routed.contains(local_path));
  CHECK(routed.at(s3_uri) == io_context_type::restful);
  CHECK(is_local_backend(routed.at(local_path)));
}

TEST_CASE("split_provider resolver routes mixed parquet files independently", "[s3][routing]")
{
  auto const fixture_path = project_root() / "test/cpp/integration/data/parquet/nation.parquet";
  auto parquet_bytes      = read_binary_file(fixture_path);
  range_s3_server server(std::move(parquet_bytes));
  scan_manager_fixture fixture;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle,
    make_s3_scan_config(server.endpoint(), sirius::scan_manager::io_backend::native),
    *fixture.memory,
    fixture.topology};

  std::string const s3_uri     = "s3://routing-bucket/nation.parquet";
  std::string const local_path = fixture_path.string();

  auto s3_ds = manager.create_datasource(s3_uri);
  REQUIRE(s3_ds != nullptr);
  prime_parquet_metadata(*s3_ds, s3_uri);

  auto local_ds = manager.create_datasource(local_path);
  REQUIRE(local_ds != nullptr);
  prime_parquet_metadata(*local_ds, local_path);

  auto provider_ingestible = sirius::op::scan::make_ingestible(
    make_nation_table_info(std::vector<std::string>{s3_uri, local_path}));
  sirius::scan_manager::split_provider provider(*provider_ingestible,
                                                make_datasource_resolver(manager));

  routing_observations routed;
  auto sources = provider.make_preparation_sources();
  REQUIRE(sources.size() == 1);
  for (;;) {
    auto claim = sources.front().try_claim();
    if (std::holds_alternative<sirius::scan_manager::preparation_exhausted>(claim)) break;
    REQUIRE(std::holds_alternative<sirius::op::scan::gpu_ingestible::metadata_job>(claim));
    auto scan = std::get<sirius::op::scan::gpu_ingestible::metadata_job>(std::move(claim)).run({});
    REQUIRE(scan != nullptr);
    auto* file_scan = dynamic_cast<sirius::op::scan::parquet_file_scan_info*>(scan.get());
    REQUIRE(file_scan != nullptr);
    REQUIRE(file_scan->datasource != nullptr);
    REQUIRE(file_scan->datasource->io_ctx() != nullptr);
    routed.emplace(file_scan->file_path, file_scan->datasource->io_ctx()->type());
  }

  REQUIRE(routed.size() == 2);
  REQUIRE(routed.contains(s3_uri));
  REQUIRE(routed.contains(local_path));
  CHECK(routed.at(s3_uri) == io_context_type::restful);
  CHECK(is_local_backend(routed.at(local_path)));
}

TEST_CASE("describe_parquet probes the S3 footer on a cold bind and reuses the metadata store warm",
          "[s3][routing]")
{
  auto const fixture_path = project_root() / "test/cpp/integration/data/parquet/nation.parquet";
  auto parquet_bytes      = read_binary_file(fixture_path);
  auto const object_size  = parquet_bytes.size();
  // The parsed footer is shared across opens only for an object with a usable strong validator:
  // a tagless open gets a per-open identity and caches nothing beyond itself. The same strong
  // ETag must therefore come back from the probe GET and from the warm bind's HEAD.
  range_fault_policy fault;
  fault.etag            = GENERATE(std::string{"\"nation-generation\""}, std::string{});
  bool const strong_tag = !fault.etag.empty();
  CAPTURE(fault.etag);
  range_s3_server server(std::move(parquet_bytes), fault);
  scan_manager_fixture fixture;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle,
    make_s3_scan_config(server.endpoint(), sirius::scan_manager::io_backend::native),
    *fixture.memory,
    fixture.topology};
  std::string const uri = "s3://footer-bucket/nation.parquet";

  // Cold bind opens with open_hint::parquet_footer_probe: one suffix-range GET resolves the size
  // and stashes the footer that cuDF then reads, so there is no HEAD and no second GET.
  auto const first = manager.describe_parquet(uri);
  CHECK(first.object_size == object_size);
  CHECK(first.total_num_rows == 25);
  REQUIRE(first.names.size() == 4);
  CHECK(first.names[0] == "n_nationkey");
  CHECK(first.names[1] == "n_name");
  CHECK(first.names[2] == "n_regionkey");
  CHECK(first.names[3] == "n_comment");
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 1);
  CHECK(server.suffix_get_count() == 1);

  // Warm bind. With a strong tag the path is a reusable candidate in the metadata store
  // (has_path), so the open takes open_hint::generic (a HEAD for the size) and finds the stored
  // footer under the same identity: no footer bytes are downloaded. A tagless open's footer is
  // scoped to that open (an opened-handle identity), which cannot justify skipping the probe for
  // the next open (#2132): the warm bind probes again, one more suffix GET and no HEAD.
  auto const second = manager.describe_parquet(uri);
  CHECK(second.object_size == first.object_size);
  CHECK(second.total_num_rows == first.total_num_rows);
  CHECK(second.names == first.names);
  CHECK(server.head_count() == (strong_tag ? 1 : 0));
  CHECK(server.get_count() == (strong_tag ? 1 : 2));
  CHECK(server.suffix_get_count() == (strong_tag ? 1 : 2));
}

TEST_CASE("routed S3 cache fill at the object tail is clipped to EOF", "[s3][routing]")
{
  constexpr std::size_t page_size = 4096;
  constexpr std::size_t tail_size = 17;
  std::vector<std::uint8_t> payload(page_size + tail_size);
  for (std::size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<std::uint8_t>(i * 31 + 7);
  }
  range_s3_server server(payload);
  scan_manager_fixture fixture;
  auto cfg       = make_s3_scan_config(server.endpoint(), sirius::scan_manager::io_backend::native);
  cfg.cache.mode = cucascade::io::cache::cache_mode::cucs;
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{scan_lifecycle, cfg, *fixture.memory, fixture.topology};
  auto datasource = manager.create_datasource("s3://tail-cache-bucket/object.bin");
  REQUIRE(datasource != nullptr);

  std::array<cudf::io::text::byte_range_info, 1> ranges{
    cudf::io::text::byte_range_info{page_size, tail_size}};
  datasource->fadvise(ranges, std::nullopt);
  REQUIRE(datasource->prepare_prefetch(false) == cucascade::io::prepare_result::prepared);

  std::array<std::uint8_t, tail_size> destination{};
  REQUIRE(datasource->host_read(page_size, destination.size(), destination.data()) == tail_size);
  CHECK(std::equal(destination.begin(), destination.end(), payload.begin() + page_size));
  // The cache fills whole pages; the one holding the tail must not ask past the object's end.
  CHECK(server.max_requested_end() == payload.size());
}

TEST_CASE("scan_manager teardown resolves an in-flight S3 read", "[s3][routing]")
{
  using namespace std::chrono_literals;
  std::vector<std::uint8_t> payload(128 * 1024);
  for (std::size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<std::uint8_t>((i * 131U + 17U) & 0xffU);
  }
  range_fault_policy fault;
  fault.response_delay = 500ms;
  auto server          = std::make_shared<range_s3_server>(payload, fault);

  auto fixture = std::make_unique<scan_manager_fixture>();
  auto cfg     = make_s3_scan_config(server->endpoint(), sirius::scan_manager::io_backend::native);
  auto lifecycle = std::make_shared<sirius::exec::query_lifecycle_registry>();
  auto manager =
    std::make_unique<sirius_scan_manager>(*lifecycle, cfg, *fixture->memory, fixture->topology);
  auto datasource = manager->create_datasource("s3://teardown-bucket/object.bin");
  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  REQUIRE(datasource->io_ctx()->type() == io_context_type::restful);
  std::weak_ptr<cucascade::io::ioctx> const weak_ctx = datasource->io_ctx();

  auto buffer = std::make_shared<std::vector<std::uint8_t>>(4096, std::uint8_t{0});
  std::array<cucascade::io::slice, 1> const slices{
    cucascade::io::slice{0, buffer->size(), buffer->data()}};
  auto future = datasource->host_read_ranges_async(slices);

  // Tear down on a worker so a hang fails the test instead of blocking it.
  std::promise<void> completed;
  auto teardown = completed.get_future();
  std::thread worker([datasource = std::move(datasource),
                      manager    = std::move(manager),
                      fixture    = std::move(fixture),
                      lifecycle  = std::move(lifecycle),
                      completed  = std::move(completed)]() mutable {
    datasource.reset();
    manager.reset();
    lifecycle.reset();
    fixture.reset();
    completed.set_value();
  });
  if (teardown.wait_for(10s) != std::future_status::ready) {
    worker.detach();
    FAIL("scan manager teardown did not return within 10 s");
  }
  worker.join();

  REQUIRE(future.valid());
  REQUIRE(future.wait_for(5s) == std::future_status::ready);
  try {
    auto const bytes = future.get();
    if (bytes != 0) {
      REQUIRE(bytes == buffer->size());
      CHECK(std::equal(buffer->begin(), buffer->end(), payload.begin()));
    }
  } catch (std::exception const& e) {
    INFO("in-flight read settled with an exception during teardown: " << e.what());
    SUCCEED("in-flight read settled with an exception during teardown");
  }

  // The settled read holds no reference to the torn-down REST ioctx.
  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (!weak_ctx.expired() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  CHECK(weak_ctx.expired());
}

TEST_CASE("backend=native without an object store declines s3 paths and says why", "[s3][routing]")
{
  scan_manager_fixture fixture;
  scan_manager_config cfg{};
  cfg.backend                 = sirius::scan_manager::io_backend::native;
  cfg.thread_pool.num_threads = 1;
  cfg.uring_n_reactors        = 1;
  cfg.cache.mode              = cucascade::io::cache::cache_mode::none;
  REQUIRE(cfg.object_store.endpoint.empty());
  REQUIRE(cfg.object_store.access_key.empty());
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{scan_lifecycle, cfg, *fixture.memory, fixture.topology};

  // No backend can serve the path, so the datasource is null (callers map that
  // to their own error); the cause is logged once, as a WARN.
  {
    sirius::test::scoped_recording_log_sink logs{"warn"};
    CHECK(manager.create_datasource("s3://unconfigured-bucket/key.parquet") == nullptr);
    auto const records = logs.records();
    CHECK(std::ranges::any_of(records, [](auto const& record) {
      return record.level == sirius::log::level::warn &&
             record.message.find("object store not configured") != std::string::npos;
    }));
  }

  // Local paths are still served by uring on the same manager.
  auto const local_path = make_regular_file();
  auto local            = manager.create_datasource(local_path.string());
  REQUIRE(local != nullptr);
  REQUIRE(local->io_ctx() != nullptr);
  CHECK(local->io_ctx()->type() == io_context_type::uring);
}
