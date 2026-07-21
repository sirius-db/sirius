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
#include "io/cache/prefetching_cache.hpp"
#include "io/datasource_factory.hpp"
#include "io/io_context.hpp"
#include "io/rest/config.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/parquet_metadata.hpp"
#include "planner/query.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "scan_manager/split_provider.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet_io_utils.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <arpa/inet.h>
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

using sirius::io::io_context_registry;
using sirius::io::io_context_type;
using sirius::io::rest::rest_ioctx;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

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

std::unique_ptr<cudf::io::datasource::buffer> read_local_parquet_footer(
  cudf::io::datasource& source)
{
  auto constexpr footer_tail_size = sizeof(cudf::io::parquet::file_ender_s);
  auto const file_size            = source.size();
  REQUIRE(file_size >= footer_tail_size);

  auto tail = source.host_read(file_size - footer_tail_size, footer_tail_size);

  std::uint32_t footer_size = 0;
  std::memcpy(&footer_size, tail->data(), sizeof(footer_size));
  REQUIRE(file_size >= footer_tail_size + footer_size);

  return source.host_read(file_size - footer_tail_size - footer_size, footer_size);
}

std::shared_ptr<cudf::io::parquet::FileMetaData const> read_local_parquet_metadata(
  std::filesystem::path const& path)
{
  auto source = cudf::io::datasource::create(path.string());
  auto footer = read_local_parquet_footer(*source);
  auto opts   = cudf::io::parquet_reader_options::builder().build();
  cudf::io::parquet::experimental::hybrid_scan_reader reader{
    cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), opts};
  return std::make_shared<cudf::io::parquet::FileMetaData const>(reader.parquet_metadata());
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
  return type.has_value() && (*type == io_context_type::uring || *type == io_context_type::kvikio);
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

scan_manager_config make_s3_scan_config(std::string endpoint, bool use_sirius_datasource)
{
  scan_manager_config cfg{};
  cfg.use_sirius_datasource        = use_sirius_datasource;
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
  cfg.enable_prefetch_cache        = false;
  return cfg;
}

struct scan_manager_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  std::shared_ptr<const sirius::memory::topology_index> topology = single_gpu_index();
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
  info->scan_output_arity = info->returned_types.size();
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
    auto scan = task();
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
  return [&manager](std::string_view path) -> std::shared_ptr<sirius::io::sirius_ioctx> {
    auto ds = manager.create_datasource(path);
    if (!ds) {
      throw std::runtime_error("test datasource resolver: no backend supports path: " +
                               std::string(path));
    }
    return ds->io_ctx();
  };
}

sirius::planner::query make_empty_query()
{
  auto tctx = sirius::test::make_test_telemetry_context();
  sirius::telemetry::query_telemetry_info tinfo{tctx->engine_id(), tctx->worker_id()};
  return sirius::planner::query(
    duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>>{},
    tctx->context(),
    tinfo);
}

struct range_fault_policy {
  std::size_t fail_first_gets{0};
  bool fail_all_gets{false};
  int fail_status{503};
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
      response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size()) +
                 "\r\nConnection: close\r\n\r\n";
      send_all(fd, response);
    } else if (is_get) {
      auto const get_idx = _get_count.fetch_add(1, std::memory_order_relaxed);
      if (_fault.fail_all_gets || get_idx < _fault.fail_first_gets) {
        response = "HTTP/1.1 " + std::to_string(_fault.fail_status) +
                   " Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        send_all(fd, response);
        return;
      }
      if (auto range = parse_range(request)) {
        auto const [start, end] = *range;
        auto const len          = end - start + 1;
        response = "HTTP/1.1 206 Partial Content\r\nContent-Length: " + std::to_string(len) +
                   "\r\nContent-Range: bytes " + std::to_string(start) + "-" + std::to_string(end) +
                   "/" + std::to_string(_object.size()) + "\r\nConnection: close\r\n\r\n";
        send_all(fd, response);
        send_all(fd, _object.data() + start, len);
      } else {
        response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size()) +
                   "\r\nConnection: close\r\n\r\n";
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

  [[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>> parse_range(
    std::string const& request) const
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
  std::atomic<int> _request_count{0};
  std::atomic<std::size_t> _get_count{0};
  std::thread _thread;
};

rest_ioctx* require_rest_ioctx(std::shared_ptr<sirius::io::sirius_datasource> const& ds)
{
  REQUIRE(ds != nullptr);
  REQUIRE(ds->io_ctx() != nullptr);
  auto* ctx = dynamic_cast<rest_ioctx*>(ds->io_ctx().get());
  REQUIRE(ctx != nullptr);
  return ctx;
}

void read_one_host_range(sirius::io::sirius_datasource& ds)
{
  std::array<std::uint8_t, 128> dst{};
  REQUIRE(ds.host_read(0, dst.size(), dst.data()) == dst.size());
}

void read_one_device_range(sirius::io::sirius_datasource& ds)
{
  rmm::cuda_stream stream;
  rmm::device_buffer dst(128, stream);
  REQUIRE(ds.device_read(0, 128, reinterpret_cast<std::uint8_t*>(dst.data()), stream) == 128);
}

}  // namespace

TEST_CASE("io_context_registry routes full paths before the kvikio catch-all", "[s3][routing]")
{
  scan_manager_fixture fixture;
  auto cfg              = make_s3_scan_config("http://127.0.0.1:1", true);
  auto const local_path = make_regular_file();

  io_context_registry registry{cfg, *fixture.memory};

  CHECK(registry.lookup_path("s3://bucket/key.parquet") == io_context_type::restful);
  CHECK(is_local_backend(registry.lookup_path(local_path.string())));

  auto file_uri = std::string{"file://"} + local_path.string();
  auto stripped = strip_file_scheme_for_registry(file_uri);
  CHECK(is_local_backend(registry.lookup_path(stripped)));
  CHECK(registry.lookup_path(stripped) != io_context_type::restful);
}

TEST_CASE(
  "io_context_registry keeps local paths on kvikio when Sirius local datasource is disabled",
  "[s3][routing]")
{
  scan_manager_fixture fixture;
  auto const local_path = make_regular_file();

  io_context_registry sirius_registry{
    make_s3_scan_config("http://127.0.0.1:1", /*use_sirius_datasource=*/true), *fixture.memory};
  CHECK(sirius_registry.lookup_path(local_path.string()) == io_context_type::uring);

  io_context_registry fallback_registry{
    make_s3_scan_config("http://127.0.0.1:1", /*use_sirius_datasource=*/false), *fixture.memory};
  CHECK(fallback_registry.lookup_path(local_path.string()) == io_context_type::kvikio);
  CHECK(fallback_registry.lookup_path(local_path.string()) != io_context_type::uring);
  CHECK(fallback_registry.lookup_path("s3://bucket/key.parquet") == io_context_type::restful);
}

TEST_CASE("scan_manager create_datasource resolves s3 paths to restful ioctx",
          "[s3][routing][scan_manager]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{0}));
  scan_manager_fixture fixture;
  sirius_scan_manager manager{
    make_s3_scan_config(server.endpoint(), true), *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://routing-bucket/data.parquet");

  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  CHECK(datasource->io_ctx()->type() == io_context_type::restful);
}

TEST_CASE("scan_manager concurrent first-touch reuses one routed S3 ioctx",
          "[s3][routing][scan_manager]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{0}));
  scan_manager_fixture fixture;
  sirius_scan_manager manager{
    make_s3_scan_config(server.endpoint(), true), *fixture.memory, fixture.topology};

  auto constexpr kThreads = std::size_t{16};
  auto const uri          = std::string{"s3://routing-bucket/data.parquet"};
  std::atomic<std::size_t> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::shared_ptr<sirius::io::sirius_ioctx>> ioctxs(kThreads);
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

TEST_CASE("scan_manager create_datasource normalizes file URI paths before routing",
          "[s3][routing][scan_manager]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{0}));
  scan_manager_fixture fixture;
  sirius_scan_manager manager{
    make_s3_scan_config(server.endpoint(), /*use_sirius_datasource=*/true),
    *fixture.memory,
    fixture.topology};

  auto const local_path = make_regular_file();
  auto datasource       = manager.create_datasource("file://" + local_path.string());

  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  CHECK(datasource->io_ctx()->type() == io_context_type::uring);
}

TEST_CASE("scan_manager preserves S3 routing when local Sirius datasource fallback is disabled",
          "[s3][routing][scan_manager]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{0}));
  scan_manager_fixture fixture;
  sirius_scan_manager manager{
    make_s3_scan_config(server.endpoint(), false), *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://routing-bucket/data.parquet");

  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  CHECK(datasource->io_ctx()->type() == io_context_type::restful);
  CHECK(datasource->io_ctx()->type() != io_context_type::kvikio);
}

TEST_CASE("scan_manager re-primes routed S3 cache on every query",
          "[s3][routing][scan_manager][cache]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{0}));
  scan_manager_fixture fixture;
  auto cfg = make_s3_scan_config(server.endpoint(), /*use_sirius_datasource=*/true);
  cfg.enable_prefetch_cache = true;
  sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://routing-bucket/data.parquet");
  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  auto* routed_cache = datasource->io_ctx()->cache();
  REQUIRE(routed_cache != nullptr);
  REQUIRE(routed_cache->query_epoch() == 0);

  auto* default_cache = manager.io_ctx()->cache();
  REQUIRE(default_cache != nullptr);
  REQUIRE(default_cache->query_epoch() == 0);

  auto q = make_empty_query();
  // The query intentionally has no scan operators: routed caches must still
  // advance once per query, matching the default ioctx's query-wide refresh.
  manager.prepare_for_query(q, true);
  REQUIRE(routed_cache->query_epoch() == 1);
  REQUIRE(default_cache->query_epoch() == 1);

  manager.prepare_for_query(q, true);
  REQUIRE(routed_cache->query_epoch() == 2);
  REQUIRE(default_cache->query_epoch() == 2);
}

TEST_CASE("scan_manager tolerates routed S3 ioctx without a prefetch cache",
          "[s3][routing][scan_manager][cache]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{0}));
  scan_manager_fixture fixture;
  auto cfg = make_s3_scan_config(server.endpoint(), /*use_sirius_datasource=*/true);
  REQUIRE_FALSE(cfg.enable_prefetch_cache);
  sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://routing-bucket/data.parquet");
  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  REQUIRE(datasource->io_ctx()->cache() == nullptr);

  auto q = make_empty_query();
  REQUIRE_NOTHROW(manager.prepare_for_query(q, true));
}

TEST_CASE("rest perf instrumentation flag gates micro counters", "[s3][rest][perf]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{7}));
  scan_manager_fixture fixture;

  SECTION("flag off leaves latency micro-counters at zero while safety counters are readable")
  {
    auto cfg                      = make_s3_scan_config(server.endpoint(), true);
    cfg.rest.perf_instrumentation = false;
    sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

    auto datasource = manager.create_datasource("s3://routing-bucket/data.parquet");
    auto* rest_ctx  = require_rest_ioctx(datasource);

    read_one_device_range(*datasource);

    auto const snapshot = rest_ctx->perf_snapshot();
    CHECK(snapshot.chunk_get_count == 0);
    CHECK(snapshot.chunk_get_ns_total == 0);
    CHECK(snapshot.chunk_get_ns_max == 0);
    CHECK(snapshot.queue_wait_count == 0);
    CHECK(snapshot.queue_wait_ns_total == 0);
    CHECK(snapshot.h2d_observed_count == 0);
    CHECK(snapshot.h2d_observed_ns_total == 0);
    CHECK(snapshot.h2d_observed_ns_max == 0);
    CHECK(snapshot.ttfb_ns == 0);
    CHECK(snapshot.device_stream_sync_total == 0);
    CHECK(snapshot.retries_total == 0);
    CHECK(snapshot.terminal_failures_total == 0);
  }

  SECTION("flag on records GET, queue, H2D, and TTFB timings")
  {
    auto cfg                      = make_s3_scan_config(server.endpoint(), true);
    cfg.rest.perf_instrumentation = true;
    sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

    auto datasource = manager.create_datasource("s3://routing-bucket/data.parquet");
    auto* rest_ctx  = require_rest_ioctx(datasource);

    read_one_device_range(*datasource);

    auto const snapshot = rest_ctx->perf_snapshot();
    CHECK(snapshot.chunk_get_count > 0);
    CHECK(snapshot.chunk_get_ns_total > 0);
    CHECK(snapshot.chunk_get_ns_max > 0);
    CHECK(snapshot.chunk_get_ns_max <= snapshot.chunk_get_ns_total);
    CHECK(snapshot.queue_wait_count > 0);
    CHECK(snapshot.queue_wait_ns_total > 0);
    CHECK(snapshot.h2d_observed_count > 0);
    CHECK(snapshot.h2d_observed_ns_total > 0);
    CHECK(snapshot.h2d_observed_ns_max > 0);
    CHECK(snapshot.h2d_observed_ns_max <= snapshot.h2d_observed_ns_total);
    CHECK(snapshot.ttfb_ns > 0);
    CHECK(snapshot.device_stream_sync_total == 0);
    CHECK(snapshot.retries_total == 0);
    CHECK(snapshot.terminal_failures_total == 0);
  }
}

TEST_CASE("rest perf safety counters track retries and terminal failures", "[s3][rest][perf]")
{
  scan_manager_fixture fixture;

  SECTION("503 retries are counted even when latency instrumentation is off")
  {
    range_fault_policy fault{};
    fault.fail_first_gets = 2;
    fault.fail_status     = 503;
    range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{8}), fault);

    auto cfg                      = make_s3_scan_config(server.endpoint(), true);
    cfg.rest.max_retry_attempts   = 4;
    cfg.rest.perf_instrumentation = false;
    sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

    auto datasource = manager.create_datasource("s3://routing-bucket/retry.parquet");
    auto* rest_ctx  = require_rest_ioctx(datasource);

    read_one_host_range(*datasource);

    auto const snapshot = rest_ctx->perf_snapshot();
    CHECK(snapshot.retries_total == 2);
    CHECK(snapshot.terminal_failures_total == 0);
    CHECK(snapshot.device_stream_sync_total == 0);
    CHECK(snapshot.chunk_get_count == 0);
    CHECK(snapshot.queue_wait_count == 0);
  }

  SECTION("exhausted retries are reported as terminal failures")
  {
    range_fault_policy fault{};
    fault.fail_all_gets = true;
    fault.fail_status   = 503;
    range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{9}), fault);

    auto cfg                      = make_s3_scan_config(server.endpoint(), true);
    cfg.rest.max_retry_attempts   = 2;
    cfg.rest.perf_instrumentation = false;
    sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

    auto datasource = manager.create_datasource("s3://routing-bucket/terminal.parquet");
    auto* rest_ctx  = require_rest_ioctx(datasource);

    std::array<std::uint8_t, 128> dst{};
    CHECK_THROWS(datasource->host_read(0, dst.size(), dst.data()));

    auto const snapshot = rest_ctx->perf_snapshot();
    CHECK(snapshot.retries_total >= 1);
    CHECK(snapshot.terminal_failures_total >= 1);
    CHECK(snapshot.device_stream_sync_total == 0);
  }
}

TEST_CASE("rest perf queue wait counts original requests rather than retry attempts",
          "[s3][rest][perf]")
{
  range_fault_policy fault{};
  fault.fail_first_gets = 2;
  fault.fail_status     = 503;
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{10}), fault);
  scan_manager_fixture fixture;
  auto cfg                      = make_s3_scan_config(server.endpoint(), true);
  cfg.rest.max_retry_attempts   = 4;
  cfg.rest.perf_instrumentation = true;
  sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://routing-bucket/queue.parquet");
  auto* rest_ctx  = require_rest_ioctx(datasource);

  read_one_host_range(*datasource);

  auto const snapshot = rest_ctx->perf_snapshot();
  CHECK(snapshot.retries_total == 2);
  CHECK(snapshot.terminal_failures_total == 0);
  CHECK(snapshot.queue_wait_count == 1);
  CHECK(snapshot.chunk_get_count == 1);
}

TEST_CASE("rest perf snapshot aggregates counters across the reactor pool", "[s3][rest][perf]")
{
  range_s3_server server(std::vector<std::uint8_t>(4096, std::uint8_t{11}));
  scan_manager_fixture fixture;
  auto cfg                      = make_s3_scan_config(server.endpoint(), true);
  cfg.rest.perf_instrumentation = true;
  cfg.rest_n_reactors           = 2;
  sirius_scan_manager manager{cfg, *fixture.memory, fixture.topology};

  auto datasource = manager.create_datasource("s3://routing-bucket/pool.parquet");
  auto* rest_ctx  = require_rest_ioctx(datasource);

  for (int i = 0; i < 4; ++i) {
    read_one_host_range(*datasource);
  }

  auto const snapshot = rest_ctx->perf_snapshot();
  CHECK(snapshot.chunk_get_count == 4);
  CHECK(snapshot.queue_wait_count == 4);
  CHECK(snapshot.chunk_get_ns_total > 0);
  CHECK(snapshot.chunk_get_ns_max > 0);
  CHECK(snapshot.chunk_get_ns_max <= snapshot.chunk_get_ns_total);
  CHECK(snapshot.queue_wait_ns_total > 0);
  CHECK(snapshot.ttfb_ns > 0);
  CHECK(snapshot.retries_total == 0);
  CHECK(snapshot.terminal_failures_total == 0);
  CHECK(snapshot.device_stream_sync_total == 0);
}

TEST_CASE("parquet_gpu_ingestible resolver routes each parquet file independently",
          "[s3][routing][scan_manager][parquet_gpu_ingestible]")
{
  auto const fixture_path = project_root() / "test/cpp/integration/data/parquet/nation.parquet";
  auto parquet_bytes      = read_binary_file(fixture_path);
  range_s3_server server(std::move(parquet_bytes));
  scan_manager_fixture fixture;
  sirius_scan_manager manager{
    make_s3_scan_config(server.endpoint(), true), *fixture.memory, fixture.topology};

  std::string const s3_uri     = "s3://routing-bucket/nation.parquet";
  std::string const local_path = fixture_path.string();
  auto const metadata          = read_local_parquet_metadata(fixture_path);

  auto s3_ds = manager.create_datasource(s3_uri);
  REQUIRE(s3_ds != nullptr);
  REQUIRE(s3_ds->io_ctx() != nullptr);
  REQUIRE(s3_ds->io_ctx()->type() == io_context_type::restful);
  REQUIRE(s3_ds->store_metadata(
    std::make_shared<sirius::op::scan::parquet_metadata>(metadata, /*footer_byte_len=*/0)));

  auto local_ds = manager.create_datasource(local_path);
  REQUIRE(local_ds != nullptr);
  REQUIRE(local_ds->io_ctx() != nullptr);
  REQUIRE(is_local_backend(local_ds->io_ctx()->type()));
  REQUIRE(local_ds->store_metadata(
    std::make_shared<sirius::op::scan::parquet_metadata>(metadata, /*footer_byte_len=*/0)));

  auto ingestible = sirius::op::scan::make_ingestible(
    make_nation_table_info(std::vector<std::string>{s3_uri, local_path}));
  auto routed = collect_routing_observations(*ingestible, make_datasource_resolver(manager));

  REQUIRE(routed.size() == 2);
  REQUIRE(routed.contains(s3_uri));
  REQUIRE(routed.contains(local_path));
  CHECK(routed.at(s3_uri) == io_context_type::restful);
  CHECK(is_local_backend(routed.at(local_path)));
}

TEST_CASE("split_provider resolver routes mixed parquet files independently",
          "[s3][routing][scan_manager][split_provider]")
{
  auto const fixture_path = project_root() / "test/cpp/integration/data/parquet/nation.parquet";
  auto parquet_bytes      = read_binary_file(fixture_path);
  range_s3_server server(std::move(parquet_bytes));
  scan_manager_fixture fixture;
  sirius_scan_manager manager{
    make_s3_scan_config(server.endpoint(), true), *fixture.memory, fixture.topology};

  std::string const s3_uri     = "s3://routing-bucket/nation.parquet";
  std::string const local_path = fixture_path.string();
  auto const metadata          = read_local_parquet_metadata(fixture_path);

  auto s3_ds = manager.create_datasource(s3_uri);
  REQUIRE(s3_ds != nullptr);
  REQUIRE(s3_ds->store_metadata(
    std::make_shared<sirius::op::scan::parquet_metadata>(metadata, /*footer_byte_len=*/0)));

  auto local_ds = manager.create_datasource(local_path);
  REQUIRE(local_ds != nullptr);
  REQUIRE(local_ds->store_metadata(
    std::make_shared<sirius::op::scan::parquet_metadata>(metadata, /*footer_byte_len=*/0)));

  auto provider_ingestible = sirius::op::scan::make_ingestible(
    make_nation_table_info(std::vector<std::string>{s3_uri, local_path}));
  sirius::scan_manager::split_provider provider(*provider_ingestible,
                                                make_datasource_resolver(manager));

  routing_observations routed;
  while (provider.has_more_splits()) {
    auto task = provider.next_split_provider();
    if (!task) { continue; }
    auto scan = task();
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
