#pragma once

#include "catch.hpp"
#include "data/sirius_converter_registry.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/io_errors.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "memory/topology_index.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <arpa/inet.h>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::test::cache_bypass {
using namespace std::chrono_literals;
inline constexpr std::size_t chunk_bytes   = 1U << 20;
inline constexpr std::size_t object_bytes  = 4U << 20;
inline std::string const object_uri        = "s3://c3-fixture/object.bin";
inline std::string const object_tag        = "\"c3-generation-one\"";
inline std::string const first_chunk_range = "bytes=0-1048575";

struct counters {
  std::uint64_t hits;
  std::uint64_t loads;
  std::uint64_t misses;
  std::uint64_t evictions;
  bool operator==(counters const&) const = default;
};

inline counters snapshot(io::ioctx& context)
{
  REQUIRE(context.cache() != nullptr);
  auto const text = context.cache()->summary();
  INFO(text);
  static std::regex const fields{
    R"(global\[reads=\d+ hits=(\d+) h2d=(\d+) miss=(\d+) evictions=(\d+)\])"};
  std::smatch match;
  REQUIRE(std::regex_search(text, match, fields));
  return {std::stoull(match[1].str()),
          std::stoull(match[2].str()),
          std::stoull(match[3].str()),
          std::stoull(match[4].str())};
}

inline std::vector<std::uint8_t> payload()
{
  std::vector<std::uint8_t> bytes(object_bytes);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131 + i / 257) % 251);
  }
  return bytes;
}

class held_response {
 public:
  held_response(std::string range, std::chrono::milliseconds duration)
    : requested_range(std::move(range)),
      deadline(std::chrono::steady_clock::now() + duration),
      entered(entered_promise.get_future().share())
  {
  }

  void park(std::string const& range)
  {
    std::unique_lock lock(mutex);
    if (range != requested_range || !armed || released) { return; }
    armed  = false;
    parked = true;
    entered_promise.set_value();
    if (!cv.wait_until(lock, deadline, [&] { return released; })) {
      timed_out = true;
      released  = true;
    }
    parked = false;
  }

  void release() noexcept
  {
    std::lock_guard lock(mutex);
    released = true;
    cv.notify_all();
  }

  bool is_parked() const
  {
    std::lock_guard lock(mutex);
    return parked && !released;
  }

  bool expired() const
  {
    std::lock_guard lock(mutex);
    return timed_out;
  }

 private:
  std::string requested_range;
  std::chrono::steady_clock::time_point deadline;
  std::promise<void> entered_promise;
  mutable std::mutex mutex;
  std::condition_variable cv;
  bool armed     = true;
  bool parked    = false;
  bool released  = false;
  bool timed_out = false;

 public:
  std::shared_future<void> entered;
};

struct release_held_response {
  std::shared_ptr<held_response> hold;
  ~release_held_response() { hold->release(); }
};

class recording_server {
 public:
  struct request_record {
    std::string path;
    std::vector<std::string> ranges;
    std::vector<std::string> if_match;
    bool suffix;
  };

  explicit recording_server(std::string tag = object_tag)
    : bytes(payload()),
      head_tag(std::move(tag)),
      data_tag(head_tag),
      listener(::socket(AF_INET, SOCK_STREAM, 0))
  {
    if (listener.fd < 0) { throw std::runtime_error("C3 server socket failed"); }
    int one = 1;
    if (::setsockopt(listener.fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
      throw std::runtime_error("C3 server setsockopt failed");
    }
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(listener.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(listener.fd, 16) != 0) {
      throw std::runtime_error("C3 server bind/listen failed");
    }
    socklen_t length = sizeof(address);
    if (::getsockname(listener.fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
      throw std::runtime_error("C3 server getsockname failed");
    }
    port     = ntohs(address.sin_port);
    acceptor = std::thread([this] { accept_loop(); });
  }

  ~recording_server()
  {
    stopping.store(true);
    std::shared_ptr<held_response> held;
    {
      std::lock_guard lock(mutex);
      held = hold;
    }
    if (held) { held->release(); }
    ::shutdown(listener.fd, SHUT_RDWR);
    if (acceptor.joinable()) { acceptor.join(); }
    for (auto& worker : workers) {
      if (worker.joinable()) { worker.join(); }
    }
  }

  recording_server(recording_server const&)            = delete;
  recording_server& operator=(recording_server const&) = delete;

  std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port); }
  std::size_t head_count() const { return heads.load(); }
  std::vector<std::uint8_t> expected(std::size_t offset = 0, std::size_t size = chunk_bytes) const
  {
    return {bytes.begin() + offset, bytes.begin() + offset + size};
  }
  std::vector<request_record> requests() const
  {
    std::lock_guard lock(mutex);
    return records;
  }
  void data_response(int status, std::string tag)
  {
    std::lock_guard lock(mutex);
    data_status = status;
    data_tag    = std::move(tag);
  }
  std::shared_ptr<held_response> hold_next(std::string range = first_chunk_range)
  {
    std::lock_guard lock(mutex);
    if (hold) { throw std::runtime_error("C3 fixture already has a hold"); }
    hold = std::make_shared<held_response>(std::move(range), 6s);
    return hold;
  }
  void check() const
  {
    std::exception_ptr error;
    {
      std::lock_guard lock(mutex);
      error = worker_error;
    }
    if (error) { std::rethrow_exception(error); }
  }

 private:
  static std::vector<std::string> headers(std::string_view request, std::string_view name)
  {
    std::vector<std::string> values;
    auto at = request.find("\r\n");
    while (at != std::string_view::npos) {
      at += 2;
      auto const end = request.find("\r\n", at);
      if (end == std::string_view::npos || end == at) { break; }
      auto const line  = request.substr(at, end - at);
      auto const colon = line.find(':');
      if (colon == name.size() &&
          std::equal(name.begin(), name.end(), line.begin(), [](unsigned char a, unsigned char b) {
            return std::tolower(a) == std::tolower(b);
          })) {
        auto value       = line.substr(colon + 1);
        auto const begin = value.find_first_not_of(" \t");
        auto const last  = value.find_last_not_of(" \t");
        values.emplace_back(begin == std::string_view::npos
                              ? std::string_view{}
                              : value.substr(begin, last - begin + 1));
      }
      at = end;
    }
    return values;
  }

  static bool send_all(int fd, char const* data, std::size_t size)
  {
    while (size > 0) {
      auto const sent = ::send(fd, data, size, MSG_NOSIGNAL);
      if (sent <= 0) { return false; }
      data += sent;
      size -= static_cast<std::size_t>(sent);
    }
    return true;
  }

  void accept_loop()
  {
    while (!stopping.load()) {
      int fd = ::accept(listener.fd, nullptr, nullptr);
      if (fd < 0) {
        if (stopping.load()) { return; }
        continue;
      }
      workers.emplace_back([this, fd] {
        io::file_descriptor client(fd);
        try {
          serve(client.fd);
        } catch (...) {
          std::lock_guard lock(mutex);
          if (!worker_error) { worker_error = std::current_exception(); }
        }
      });
    }
  }

  void serve(int fd)
  {
    timeval timeout{};
    timeout.tv_sec = 3;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    std::array<char, 4096> buffer{};
    std::string request;
    while (request.find("\r\n\r\n") == std::string::npos) {
      auto const received = ::recv(fd, buffer.data(), buffer.size(), 0);
      if (received <= 0) { return; }
      request.append(buffer.data(), static_cast<std::size_t>(received));
      if (request.size() > 65536) { throw std::runtime_error("C3 request headers too large"); }
    }
    std::istringstream first_line(request.substr(0, request.find("\r\n")));
    std::string method;
    std::string target;
    first_line >> method >> target;
    if (method == "GET" && target.find("list-type=2") != std::string::npos) {
      std::string const body =
        "<ListBucketResult><Name>c3-fixture</Name><KeyCount>0</KeyCount>"
        "<IsTruncated>false</IsTruncated></ListBucketResult>";
      auto response = "HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\nContent-Length: " +
                      std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
      send_all(fd, response.data(), response.size());
      return;
    }
    if (method == "HEAD") {
      heads.fetch_add(1);
      auto response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(bytes.size());
      if (!head_tag.empty()) { response += "\r\nETag: " + head_tag; }
      response += "\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n";
      send_all(fd, response.data(), response.size());
      return;
    }
    if (method != "GET") { throw std::runtime_error("C3 server received an unexpected method"); }
    auto ranges = headers(request, "Range");
    if (ranges.size() != 1 || !ranges.front().starts_with("bytes=")) {
      throw std::runtime_error("C3 server requires exactly one byte range");
    }
    auto const range = ranges.front().substr(6);
    auto const dash  = range.find('-');
    if (dash == std::string::npos) { throw std::runtime_error("C3 malformed range"); }
    bool const suffix = dash == 0;
    std::size_t low;
    std::size_t high;
    if (suffix) {
      auto const length =
        std::min(bytes.size(), static_cast<std::size_t>(std::stoull(range.substr(1))));
      low  = bytes.size() - length;
      high = bytes.size() - 1;
    } else {
      low  = std::stoull(range.substr(0, dash));
      high = range.size() == dash + 1 ? bytes.size() - 1 : std::stoull(range.substr(dash + 1));
    }
    if (low > high || high >= bytes.size()) {
      throw std::runtime_error("C3 range exceeds fixture");
    }
    std::string tag;
    int status;
    std::shared_ptr<held_response> held;
    {
      std::lock_guard lock(mutex);
      records.push_back(
        {target.substr(0, target.find('?')), ranges, headers(request, "If-Match"), suffix});
      status = suffix ? 206 : data_status;
      tag    = suffix ? head_tag : data_tag;
      held   = hold;
    }
    if (status != 206) {
      auto response = "HTTP/1.1 " + std::to_string(status) +
                      " Fixture response\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
      send_all(fd, response.data(), response.size());
      return;
    }
    auto response =
      "HTTP/1.1 206 Partial Content\r\nContent-Length: " + std::to_string(high - low + 1) +
      "\r\nContent-Range: bytes " + std::to_string(low) + "-" + std::to_string(high) + "/" +
      std::to_string(bytes.size());
    if (!tag.empty()) { response += "\r\nETag: " + tag; }
    response += "\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n";
    if (!send_all(fd, response.data(), response.size())) { return; }
    if (held && !suffix) { held->park(ranges.front()); }
    send_all(fd, reinterpret_cast<char const*>(bytes.data() + low), high - low + 1);
  }

  std::vector<std::uint8_t> bytes;
  std::string head_tag;
  std::string data_tag;
  int data_status = 206;
  io::file_descriptor listener;
  std::uint16_t port = 0;
  std::atomic<bool> stopping{false};
  std::atomic<std::size_t> heads{0};
  mutable std::mutex mutex;
  std::vector<request_record> records;
  std::shared_ptr<held_response> hold;
  std::exception_ptr worker_error;
  std::thread acceptor;
  std::vector<std::thread> workers;
};

inline std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_memory()
{
  sirius::converter_registry::reset_for_testing();
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(2ULL << 30)
    .set_reservation_fraction_per_gpu(0.75)
    .set_per_numa_region_capacity(256ULL << 20)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(1.0);
  auto memory =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
  sirius::converter_registry::initialize();
  return memory;
}

inline std::shared_ptr<const sirius::memory::topology_index> topology()
{
  cucascade::memory::system_topology_info info;
  info.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  info.gpus.push_back(std::move(gpu));
  return std::make_shared<sirius::memory::topology_index>(info, std::vector<int>{0});
}

inline scan_manager::scan_manager_config server_config(std::string endpoint)
{
  scan_manager::scan_manager_config config;
  config.backend                           = scan_manager::io_backend::sirius;
  config.thread_pool.num_threads           = 2;
  config.uring_n_reactors                  = 1;
  config.rest_n_reactors                   = 1;
  config.object_store.endpoint             = std::move(endpoint);
  config.object_store.region               = "us-east-1";
  config.object_store.access_key           = "c3-fixture-access";
  config.object_store.secret_key           = "c3-fixture-secret";
  config.object_store.tls_verify           = false;
  config.rest.max_connections              = 1;
  config.rest.max_retry_attempts           = 1;
  config.rest.max_auth_retry_attempts      = 1;
  config.rest.request_timeout_s            = 20;
  config.cache.mode                        = io::cache::cache_mode::sirius;
  config.cache.eviction                    = io::cache::eviction_policy::lru;
  config.cache.eviction_threshold_fraction = 1.0;
  config.apply_cache_mode();
  return config;
}

struct rest_fixture {
  explicit rest_fixture(std::string tag = object_tag)
    : server(std::move(tag)),
      memory(make_memory()),
      manager(server_config(server.endpoint()), *memory, topology()),
      context(manager.ioctx_for_path(object_uri))
  {
    REQUIRE(context != nullptr);
    REQUIRE(context->type() == io::io_context_type::restful);
    REQUIRE(context->cache() != nullptr);
    REQUIRE(context->cache()->chunk_size() == chunk_bytes);
    REQUIRE(manager.ioctx_for_path(object_uri).get() == context.get());
  }

  recording_server server;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory;
  scan_manager::sirius_scan_manager manager;
  std::shared_ptr<io::ioctx> context;
};

inline std::vector<std::uint8_t> read(io::sirius_datasource& source,
                                      std::size_t offset = 0,
                                      std::size_t size   = chunk_bytes)
{
  std::vector<std::uint8_t> bytes(size);
  auto const count = source.host_read(offset, size, bytes.data());
  if (count != size) { throw std::runtime_error("C3 fixture received a short read"); }
  return bytes;
}

inline std::shared_ptr<io::sirius_datasource> populate(rest_fixture& fixture)
{
  auto source =
    fixture.manager.open_datasource_on(fixture.context, object_uri, io::open_hint::generic);
  REQUIRE(source != nullptr);
  REQUIRE(source->get_io_object().validation_tag() == object_tag);
  std::array<cudf::io::text::byte_range_info, 1> ranges{
    cudf::io::text::byte_range_info(0, static_cast<std::int64_t>(chunk_bytes))};
  source->fadvise(ranges, 0);
  REQUIRE(source->prepare_prefetch(false) == io::prepare_result::prepared);
  auto const before   = snapshot(*fixture.context);
  auto const requests = fixture.server.requests().size();
  REQUIRE(read(*source) == fixture.server.expected());
  auto const filled = snapshot(*fixture.context);
  REQUIRE(filled.loads == before.loads + 1);
  REQUIRE(filled.misses == before.misses);
  REQUIRE(fixture.server.requests().size() == requests + 1);
  REQUIRE(read(*source) == fixture.server.expected());
  auto const resident = snapshot(*fixture.context);
  REQUIRE(resident.hits == filled.hits + 1);
  REQUIRE(resident.loads == filled.loads);
  REQUIRE(resident.misses == filled.misses);
  REQUIRE(fixture.server.requests().size() == requests + 1);
  fixture.server.check();
  return source;
}

template <class Reader>
std::vector<std::uint8_t> reset_during_read(rest_fixture& fixture, Reader&& read_operation)
{
  auto held = fixture.server.hold_next();
  std::future<std::vector<std::uint8_t>> reader;
  std::future<void> reset;
  release_held_response release{held};
  reader                 = std::async(std::launch::async, [&] { return read_operation(); });
  reset                  = std::async(std::launch::async, [&] {
    if (held->entered.wait_for(2s) != std::future_status::ready) {
      throw std::runtime_error("C3 request did not reach the held response");
    }
    held->entered.get();
    fixture.manager.reset_caches();
  });
  auto const held_ready  = held->entered.wait_for(2s) == std::future_status::ready;
  auto const reset_ready = reset.wait_for(2s) == std::future_status::ready;
  bool reset_ok          = false;
  std::string reset_error;
  if (reset_ready) {
    try {
      reset.get();
      reset_ok = true;
    } catch (std::exception const& error) {
      reset_error = error.what();
    } catch (...) {
      reset_error = "non-standard reset exception";
    }
  }
  auto const still_held   = held->is_parked();
  auto const read_pending = reader.wait_for(0s) == std::future_status::timeout;
  held->release();
  auto const read_ready = reader.wait_for(5s) == std::future_status::ready;
  bool reset_joined     = reset_ready;
  if (!reset_ready) {
    reset_joined = reset.wait_for(5s) == std::future_status::ready;
    if (reset_joined) {
      try {
        reset.get();
      } catch (...) {
      }
    }
  }
  INFO("held=" << held_ready << " reset_ready=" << reset_ready << " reset_ok=" << reset_ok
               << " reset_error=" << reset_error << " still_held=" << still_held
               << " read_pending=" << read_pending);
  REQUIRE(read_ready);
  auto bytes = reader.get();
  REQUIRE(reset_joined);
  REQUIRE(held_ready);
  REQUIRE(reset_ready);
  REQUIRE(reset_ok);
  REQUIRE(still_held);
  REQUIRE(read_pending);
  REQUIRE_FALSE(held->expired());
  return bytes;
}
}  // namespace sirius::test::cache_bypass
