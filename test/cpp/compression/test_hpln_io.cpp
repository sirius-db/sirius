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

// Reading a .hpln through an io_context, and the coalescing that decides how many requests that
// costs.
//
// Two things are asserted throughout, because neither fails loudly on its own:
//
//  * WHICH BYTES. A request issued at the wrong offset, or a plan that drops one of its ranges,
//    returns a neighbour's bytes rather than an error. So every read here is compared against the
//    exact bytes that were written, and the .hpln cases compare decoded VALUES.
//  * WHICH TRANSPORT. A remote read that silently fell back to `std::ifstream` returns identical
//    bytes for a local file, so the tests assert on hpln_io_stats::transport as well as on the
//    data. That is also why a scheme path with no io_context must THROW rather than be tried
//    locally.
//
// The REST case runs against a loopback HTTP server that speaks just enough of S3's range-GET
// protocol, so the object-store transport is exercised without Docker or credentials; the [s3]
// case additionally runs it against MinIO when SIRIUS_TEST_S3_AUTO brought one up, and skips
// cleanly when it did not.

#include "catch.hpp"
#include "compression/hpln_io.hpp"
#include "compression/simpatico_file_ingest.hpp"
#include "io/datasource_factory.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/s3_container.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <api/compressed_table_io.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct io_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;

  io_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0))
  {
  }
};

io_env& env()
{
  static io_env e;
  return e;
}

bool no_gpu()
{
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count >= 1) { return false; }
  WARN("hpln io test requires a GPU — skipping");
  return true;
}

fs::path scratch_dir(std::string const& tag)
{
  auto const dir =
    fs::temp_directory_path() / ("sirius_hpln_io_" + tag + "_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  return dir;
}

//===----------------------------------------------------------------------===//
// plan execution against an in-memory "file"
//===----------------------------------------------------------------------===//

/// Serve @p plan out of @p file the way a transport does: each request copies one contiguous file
/// range across its destinations in order. Any mistake the planner could make -- a wrong offset,
/// a dropped range, a destination listed twice -- shows up as wrong bytes in the caller's buffers,
/// which is exactly how it would show up in a query.
void serve(sirius::hpln_read_plan const& plan, std::vector<std::uint8_t> const& file)
{
  for (auto const& r : plan.requests) {
    REQUIRE(r.offset + r.bytes <= file.size());
    std::uint64_t at      = r.offset;
    std::uint64_t covered = 0;
    for (auto const& d : r.dst) {
      std::memcpy(d.data, file.data() + at, static_cast<std::size_t>(d.bytes));
      at += d.bytes;
      covered += d.bytes;
    }
    // The invariant every backend relies on: the destinations cover the request exactly.
    REQUIRE(covered == r.bytes);
  }
}

std::vector<std::uint8_t> random_bytes(std::size_t n, std::uint32_t seed)
{
  std::mt19937 rng(seed);
  std::vector<std::uint8_t> v(n);
  for (auto& b : v) {
    b = static_cast<std::uint8_t>(rng() & 0xff);
  }
  return v;
}

void write_file(fs::path const& path, std::vector<std::uint8_t> const& bytes)
{
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<char const*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  REQUIRE(f.good());
}

std::vector<std::uint8_t> read_file(fs::path const& path)
{
  std::ifstream f(path, std::ios::binary);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f),
                                   std::istreambuf_iterator<char>());
}

//===----------------------------------------------------------------------===//
// backends
//===----------------------------------------------------------------------===//

std::shared_ptr<sirius::io::sirius_ioctx> make_uring_ioctx()
{
  sirius::scan_manager::scan_manager_config cfg{};
  cfg.use_sirius_datasource = true;
  auto ctx                  = sirius::io::make_uring_ioctx_factory(*env().mgr)(cfg);
  if (ctx) { ctx->start(); }
  return ctx;
}

std::shared_ptr<sirius::io::sirius_ioctx> make_rest_ioctx(std::string const& endpoint,
                                                          std::string const& access_key,
                                                          std::string const& secret_key,
                                                          std::string const& region)
{
  sirius::scan_manager::scan_manager_config cfg{};
  cfg.use_sirius_datasource   = true;
  cfg.object_store.endpoint   = endpoint;
  cfg.object_store.region     = region;
  cfg.object_store.access_key = access_key;
  cfg.object_store.secret_key = secret_key;
  cfg.object_store.tls_verify = false;
  cfg.rest.request_timeout_s  = 30;
  cfg.rest.max_connections    = 4;
  cfg.rest_n_reactors         = 1;
  cfg.enable_prefetch_cache   = false;
  auto ctx                    = sirius::io::make_rest_ioctx_factory(*env().mgr)(cfg);
  if (ctx) { ctx->start(); }
  return ctx;
}

/// A loopback HTTP server that answers HEAD and ranged GET for one object.
///
/// Enough of S3 for the REST io_context to read through, and no more. It counts GETs, which is
/// what makes "the coalescer turned N ranges into one request" an assertable claim rather than an
/// intention.
class range_server {
 public:
  explicit range_server(std::vector<std::uint8_t> object) : _object(std::move(object))
  {
    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(_listen_fd >= 0);
    int one = 1;
    REQUIRE(::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) == 0);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    REQUIRE(::bind(_listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    REQUIRE(::listen(_listen_fd, 64) == 0);
    socklen_t len = sizeof(addr);
    REQUIRE(::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
    _port   = ntohs(addr.sin_port);
    _thread = std::thread([this] { accept_loop(); });
  }

  ~range_server()
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

  range_server(range_server const&)            = delete;
  range_server& operator=(range_server const&) = delete;

  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }
  [[nodiscard]] std::size_t get_count() const noexcept { return _gets.load(); }
  [[nodiscard]] std::size_t body_bytes() const noexcept { return _body_bytes.load(); }

 private:
  void accept_loop()
  {
    while (!_stop.load()) {
      int fd = ::accept(_listen_fd, nullptr, nullptr);
      if (fd < 0) {
        if (_stop.load()) { return; }
        continue;
      }
      _workers.emplace_back([this, fd] { serve_connection(fd); });
    }
  }

  static void send_all(int fd, std::string const& s)
  {
    std::size_t sent = 0;
    while (sent < s.size()) {
      auto const n = ::send(fd, s.data() + sent, s.size() - sent, MSG_NOSIGNAL);
      if (n <= 0) { return; }
      sent += static_cast<std::size_t>(n);
    }
  }

  static void send_all(int fd, std::uint8_t const* data, std::size_t bytes)
  {
    std::size_t sent = 0;
    while (sent < bytes) {
      auto const n = ::send(fd, data + sent, bytes - sent, MSG_NOSIGNAL);
      if (n <= 0) { return; }
      sent += static_cast<std::size_t>(n);
    }
  }

  /// Keep-alive so one connection can carry many ranged GETs, as a real client does.
  void serve_connection(int fd)
  {
    timeval timeout{};
    timeout.tv_sec = 5;
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    std::string pending;
    while (!_stop.load()) {
      std::array<char, 4096> buf{};
      auto const n = ::recv(fd, buf.data(), buf.size(), 0);
      if (n <= 0) { break; }
      pending.append(buf.data(), static_cast<std::size_t>(n));
      std::size_t end = 0;
      while ((end = pending.find("\r\n\r\n")) != std::string::npos) {
        auto const request = pending.substr(0, end + 4);
        pending.erase(0, end + 4);
        if (!handle_request(fd, request)) {
          ::close(fd);
          return;
        }
      }
    }
    ::close(fd);
  }

  bool handle_request(int fd, std::string const& request)
  {
    if (request.rfind("HEAD ", 0) == 0) {
      send_all(fd,
               "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size()) +
                 "\r\nAccept-Ranges: bytes\r\n\r\n");
      return true;
    }
    if (request.rfind("GET ", 0) != 0) {
      send_all(fd, "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n");
      return true;
    }
    _gets.fetch_add(1, std::memory_order_relaxed);

    std::size_t start = 0;
    std::size_t end   = _object.empty() ? 0 : _object.size() - 1;
    bool ranged       = false;
    auto const pos    = request.find("Range: bytes=");
    if (pos != std::string::npos) {
      auto const spec = request.substr(pos + std::strlen("Range: bytes="));
      auto const dash = spec.find('-');
      auto const stop = spec.find_first_of("\r\n");
      if (dash != std::string::npos && stop != std::string::npos) {
        auto const lo = spec.substr(0, dash);
        auto const hi = spec.substr(dash + 1, stop - dash - 1);
        if (lo.empty()) {
          // Suffix range: the last N bytes -- the read that locates a .hpln.
          auto const want = static_cast<std::size_t>(std::stoull(hi));
          start           = _object.size() > want ? _object.size() - want : 0;
        } else {
          start = static_cast<std::size_t>(std::stoull(lo));
          if (!hi.empty()) { end = static_cast<std::size_t>(std::stoull(hi)); }
        }
        end    = std::min(end, _object.empty() ? 0 : _object.size() - 1);
        ranged = true;
      }
    }
    if (start >= _object.size() || end < start) {
      send_all(fd, "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\n\r\n");
      return true;
    }
    auto const len = end - start + 1;
    std::string header;
    if (ranged) {
      header = "HTTP/1.1 206 Partial Content\r\nContent-Length: " + std::to_string(len) +
               "\r\nContent-Range: bytes " + std::to_string(start) + "-" + std::to_string(end) +
               "/" + std::to_string(_object.size()) + "\r\n\r\n";
    } else {
      header = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(len) + "\r\n\r\n";
    }
    send_all(fd, header);
    send_all(fd, _object.data() + start, len);
    _body_bytes.fetch_add(len, std::memory_order_relaxed);
    return true;
  }

  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::vector<std::uint8_t> _object;
  std::atomic<std::size_t> _gets{0};
  std::atomic<std::size_t> _body_bytes{0};
  std::atomic<bool> _stop{false};
  std::thread _thread;
  std::vector<std::thread> _workers;
};

//===----------------------------------------------------------------------===//
// .hpln fixture
//===----------------------------------------------------------------------===//

// Big enough that the 64 KiB tail probe is a rounding error against the payload -- otherwise
// "reading two of four chunks moved less than the whole object" is not a statement about
// skipping, it is a statement about the file being smaller than one probe.
constexpr int kRowsPerChunk = 262144;
constexpr int kChunks       = 4;

std::unique_ptr<cudf::column> int32_column(std::vector<std::int32_t> const& values)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED);
  REQUIRE(cudaMemcpy(col->mutable_view().head<std::int32_t>(),
                     values.data(),
                     values.size() * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  return col;
}

std::vector<std::int32_t> read_back(cudf::column_view const& v)
{
  std::vector<std::int32_t> host(static_cast<std::size_t>(v.size()));
  REQUIRE(cudaMemcpy(host.data(),
                     v.head<std::int32_t>(),
                     host.size() * sizeof(std::int32_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return host;
}

/// A multi-chunk file whose key column encodes the chunk it came from, so reading the wrong chunk
/// is visible in the values rather than only in the row count.
std::vector<std::vector<std::int32_t>> write_fixture(std::string const& path)
{
  auto stream = cudf::get_default_stream();
  std::vector<std::vector<std::int32_t>> expected(2);
  std::vector<std::unique_ptr<cudf::table>> tables;
  std::vector<cudf::table_view> views;
  for (int c = 0; c < kChunks; ++c) {
    std::vector<std::int32_t> keys(kRowsPerChunk), vals(kRowsPerChunk);
    for (int i = 0; i < kRowsPerChunk; ++i) {
      keys[static_cast<std::size_t>(i)] = c * 1000000 + i;
      vals[static_cast<std::size_t>(i)] = i * 7 + c;
    }
    expected[0].insert(expected[0].end(), keys.begin(), keys.end());
    expected[1].insert(expected[1].end(), vals.begin(), vals.end());
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(int32_column(keys));
    cols.push_back(int32_column(vals));
    tables.push_back(std::make_unique<cudf::table>(std::move(cols)));
    views.push_back(tables.back()->view());
  }
  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
                                            duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER)};
  auto const leaf = std::string("input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n");
  REQUIRE(sirius::write_tables_to_hpln(views,
                                       types,
                                       {"k", "v"},
                                       leaf + "---\n" + leaf,
                                       /*group_rows=*/1024,
                                       path,
                                       stream,
                                       rmm::mr::get_current_device_resource_ref())
            .empty());
  return expected;
}

/// Decode the staged chunks and check they are the file's chunks @p want, in order.
void require_chunks_decode_to(std::vector<sirius::ingested_hpln_chunk> const& ingested,
                              std::vector<std::size_t> const& want,
                              std::vector<std::vector<std::int32_t>> const& expected)
{
  auto stream = cudf::get_default_stream();
  REQUIRE(ingested.size() == want.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    auto const& blob = *ingested[i].blob;
    simpatico::payload_fetch_fn fetch =
      [&blob](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
        sirius::copy_pinned_blocks_to_device(*blob.payload, off, dst, sz, s);
      };
    std::string err;
    auto const ct = simpatico::read_compressed_table_from_memory(
      blob.header, fetch, stream, rmm::mr::get_current_device_resource_ref(), &err);
    REQUIRE(err.empty());
    auto table = ct.decompress(stream, rmm::mr::get_current_device_resource_ref());
    stream.synchronize();
    REQUIRE(table->num_rows() == kRowsPerChunk);
    auto const offset = static_cast<std::ptrdiff_t>(want[i] * kRowsPerChunk);
    REQUIRE(read_back(table->view().column(0)) ==
            std::vector<std::int32_t>(expected[0].begin() + offset,
                                      expected[0].begin() + offset + kRowsPerChunk));
    REQUIRE(read_back(table->view().column(1)) ==
            std::vector<std::int32_t>(expected[1].begin() + offset,
                                      expected[1].begin() + offset + kRowsPerChunk));
  }
}

sirius::hpln_extent extent_into(std::uint64_t offset, std::vector<std::uint8_t>& dst)
{
  sirius::hpln_extent e;
  e.offset = offset;
  e.dst.push_back({dst.data(), dst.size()});
  return e;
}

}  // namespace

//===----------------------------------------------------------------------===//
// the read planner
//===----------------------------------------------------------------------===//

TEST_CASE("hpln io - a gap smaller than the policy's is bridged into one request",
          "[compression][hpln_io]")
{
  auto const file = random_bytes(1 << 16, 11);
  std::vector<std::uint8_t> a(1000), b(2000);

  sirius::hpln_io_policy policy;
  policy.max_gap_bytes        = 4096;
  policy.target_request_bytes = 1 << 20;

  // 1000 B, a 500 B hole, then 2000 B: one request, and the hole is read and discarded.
  std::vector<sirius::hpln_extent> extents;
  extents.push_back(extent_into(100, a));
  extents.push_back(extent_into(100 + 1000 + 500, b));
  auto const plan = sirius::plan_hpln_reads(std::move(extents), policy);

  REQUIRE(plan.requests.size() == 1);
  CHECK(plan.requests.front().offset == 100);
  CHECK(plan.requests.front().bytes == 3500);
  CHECK(plan.bridged_bytes == 500);
  CHECK(plan.wanted_bytes == 3000);

  serve(plan, file);
  CHECK(a == std::vector<std::uint8_t>(file.begin() + 100, file.begin() + 1100));
  CHECK(b == std::vector<std::uint8_t>(file.begin() + 1600, file.begin() + 3600));
}

TEST_CASE("hpln io - a gap larger than the policy's costs a second request",
          "[compression][hpln_io]")
{
  auto const file = random_bytes(1 << 16, 12);
  std::vector<std::uint8_t> a(1000), b(2000);

  sirius::hpln_io_policy policy;
  policy.max_gap_bytes        = 400;  // smaller than the 500 B hole
  policy.target_request_bytes = 1 << 20;

  std::vector<sirius::hpln_extent> extents;
  extents.push_back(extent_into(100, a));
  extents.push_back(extent_into(1600, b));
  auto const plan = sirius::plan_hpln_reads(std::move(extents), policy);

  REQUIRE(plan.requests.size() == 2);
  CHECK(plan.bridged_bytes == 0);
  CHECK(plan.requests[0].offset == 100);
  CHECK(plan.requests[1].offset == 1600);

  serve(plan, file);
  CHECK(a == std::vector<std::uint8_t>(file.begin() + 100, file.begin() + 1100));
  CHECK(b == std::vector<std::uint8_t>(file.begin() + 1600, file.begin() + 3600));
}

TEST_CASE("hpln io - a run longer than the target is cut into requests of it",
          "[compression][hpln_io]")
{
  auto const file = random_bytes(1 << 16, 13);
  // One destination far larger than the target: the cut has to happen inside a single buffer, and
  // the pieces have to reassemble into the original bytes in order.
  std::vector<std::uint8_t> big(20000);

  sirius::hpln_io_policy policy;
  policy.max_gap_bytes        = 0;
  policy.target_request_bytes = 4096;

  std::vector<sirius::hpln_extent> extents;
  extents.push_back(extent_into(7, big));
  auto const plan = sirius::plan_hpln_reads(std::move(extents), policy);

  REQUIRE(plan.requests.size() == 5);  // ceil(20000 / 4096)
  std::uint64_t at = 7;
  for (auto const& r : plan.requests) {
    CHECK(r.offset == at);
    CHECK(r.bytes <= policy.target_request_bytes);
    at += r.bytes;
  }
  CHECK(at == 7 + big.size());

  serve(plan, file);
  CHECK(big == std::vector<std::uint8_t>(file.begin() + 7, file.begin() + 7 + 20000));
}

TEST_CASE("hpln io - scattered destinations for one file range stay in file order",
          "[compression][hpln_io]")
{
  auto const file = random_bytes(1 << 16, 14);
  // The pinned-block case: one contiguous file range landing in several separate buffers.
  std::vector<std::uint8_t> b0(300), b1(300), b2(150);
  sirius::hpln_extent e;
  e.offset = 4096;
  e.dst.push_back({b0.data(), b0.size()});
  e.dst.push_back({b1.data(), b1.size()});
  e.dst.push_back({b2.data(), b2.size()});

  std::vector<sirius::hpln_extent> extents;
  extents.push_back(std::move(e));
  auto const plan = sirius::plan_hpln_reads(std::move(extents), sirius::hpln_io_policy{});

  REQUIRE(plan.requests.size() == 1);
  CHECK(plan.requests.front().bytes == 750);

  serve(plan, file);
  CHECK(b0 == std::vector<std::uint8_t>(file.begin() + 4096, file.begin() + 4396));
  CHECK(b1 == std::vector<std::uint8_t>(file.begin() + 4396, file.begin() + 4696));
  CHECK(b2 == std::vector<std::uint8_t>(file.begin() + 4696, file.begin() + 4846));
}

TEST_CASE("hpln io - overlapping extents are refused, not half-served", "[compression][hpln_io]")
{
  std::vector<std::uint8_t> a(1000), b(1000);
  std::vector<sirius::hpln_extent> extents;
  extents.push_back(extent_into(0, a));
  extents.push_back(extent_into(500, b));
  CHECK_THROWS(sirius::plan_hpln_reads(std::move(extents), sirius::hpln_io_policy{}));
}

TEST_CASE("hpln io - the verbatim policy issues one request per extent", "[compression][hpln_io]")
{
  auto const file = random_bytes(1 << 16, 15);
  std::vector<std::uint8_t> a(1000), b(1000), c(1000);
  std::vector<sirius::hpln_extent> extents;
  extents.push_back(extent_into(0, a));
  extents.push_back(extent_into(2000, b));
  extents.push_back(extent_into(9000, c));
  auto const plan = sirius::plan_hpln_reads(std::move(extents), sirius::hpln_io_policy::verbatim());

  REQUIRE(plan.requests.size() == 3);
  CHECK(plan.bridged_bytes == 0);
  serve(plan, file);
  CHECK(a == std::vector<std::uint8_t>(file.begin(), file.begin() + 1000));
  CHECK(b == std::vector<std::uint8_t>(file.begin() + 2000, file.begin() + 3000));
  CHECK(c == std::vector<std::uint8_t>(file.begin() + 9000, file.begin() + 10000));
}

//===----------------------------------------------------------------------===//
// transports
//===----------------------------------------------------------------------===//

TEST_CASE("hpln io - a scheme path with no io_context is refused, not read locally",
          "[compression][hpln_io]")
{
  // The failure this guards is silent: a `s3://bucket/x.hpln` handed to std::ifstream would look
  // for a local file of that name, and finding one would return the wrong object's data.
  CHECK_THROWS(sirius::open_hpln_source("s3://bucket/x.hpln", nullptr, "test"));
  CHECK_THROWS(sirius::read_hpln_schema("s3://bucket/x.hpln"));
}

TEST_CASE("hpln io - the same ranges read identically through the filesystem and the io_context",
          "[compression][hpln_io]")
{
  auto const dir  = scratch_dir("ranges");
  auto const path = dir / "bytes.bin";
  auto const file = random_bytes(3 << 20, 21);
  write_file(path, file);

  auto io_ctx = make_uring_ioctx();
  if (!io_ctx) {
    WARN("no uring io_context available — skipping");
    fs::remove_all(dir);
    return;
  }

  // Scattered ranges: two near each other (bridgeable) and one far away.
  auto const read_all = [&](std::shared_ptr<sirius::io::sirius_ioctx> ctx) {
    auto src = sirius::open_hpln_source(path.string(), std::move(ctx), "test");
    std::vector<std::uint8_t> a(4000), b(8000), c(1 << 20);
    std::vector<sirius::hpln_extent> extents;
    extents.push_back(extent_into(1234, a));
    extents.push_back(extent_into(1234 + 4000 + 777, b));
    extents.push_back(extent_into(2 << 20, c));
    src->read_extents(std::move(extents), sirius::hpln_io_policy{}, "test ranges");
    return std::tuple{std::move(a), std::move(b), std::move(c), src->stats()};
  };

  auto const [la, lb, lc, lstats] = read_all(nullptr);
  auto const [ra, rb, rc, rstats] = read_all(io_ctx);

  CHECK(lstats.transport == "ifstream");
  CHECK(rstats.transport == "io_context:uring");
  // The bridged pair plus the far range: two requests, not three.
  CHECK(rstats.requests == 2);
  CHECK(la == std::vector<std::uint8_t>(file.begin() + 1234, file.begin() + 5234));
  CHECK(lb == std::vector<std::uint8_t>(file.begin() + 6011, file.begin() + 14011));
  CHECK(lc ==
        std::vector<std::uint8_t>(file.begin() + (2 << 20), file.begin() + (2 << 20) + (1 << 20)));
  CHECK(ra == la);
  CHECK(rb == lb);
  CHECK(rc == lc);

  io_ctx->shutdown();
  fs::remove_all(dir);
}

TEST_CASE("hpln io - a .hpln binds and stages through the local io_context",
          "[compression][hpln_io][hpln_ingest]")
{
  if (no_gpu()) { return; }
  auto const dir      = scratch_dir("uring");
  auto const path     = (dir / "t.hpln").string();
  auto const expected = write_fixture(path);

  auto io_ctx = make_uring_ioctx();
  if (!io_ctx) {
    WARN("no uring io_context available — skipping");
    fs::remove_all(dir);
    return;
  }

  sirius::hpln_open_options options;
  options.io_ctx = io_ctx;
  sirius::hpln_io_stats bind_stats;
  options.stats = &bind_stats;

  auto const schema = sirius::read_hpln_schema(path, options);
  CHECK(bind_stats.transport == "io_context:uring");
  CHECK(schema.num_rows == kChunks * kRowsPerChunk);
  CHECK(schema.chunk_rows.size() == kChunks);
  CHECK(schema.names == std::vector<std::string>{"k", "v"});
  // The trailer, the directory, every chunk header, the zone maps and the declared types --
  // and the headers of all four chunks come back in ONE request because they are contiguous.
  CHECK(bind_stats.requests <= 5);

  sirius::hpln_io_stats read_stats;
  options.stats = &read_stats;
  std::vector<std::size_t> const want{1, 2};
  auto const ingested =
    sirius::read_hpln_chunks_into_pinned(path, *env().host_space, want, options);
  CHECK(read_stats.transport == "io_context:uring");
  require_chunks_decode_to(ingested, want, expected);

  io_ctx->shutdown();
  fs::remove_all(dir);
}

TEST_CASE("hpln io - a .hpln is read over HTTP ranges through the REST io_context",
          "[compression][hpln_io][hpln_ingest]")
{
  if (no_gpu()) { return; }
  auto const dir      = scratch_dir("rest");
  auto const path     = (dir / "t.hpln").string();
  auto const expected = write_fixture(path);
  auto const object   = read_file(path);

  range_server server(object);
  auto io_ctx =
    make_rest_ioctx(server.endpoint(), "test-access-key", "test-secret-key", "us-east-1");
  if (!io_ctx) {
    WARN("no REST io_context available — skipping");
    fs::remove_all(dir);
    return;
  }

  auto const uri = std::string("s3://bucket/t.hpln");
  sirius::hpln_open_options options;
  options.io_ctx = io_ctx;
  sirius::hpln_io_stats bind_stats;
  options.stats = &bind_stats;

  auto const schema = sirius::read_hpln_schema(uri, options);
  CHECK(bind_stats.transport == "io_context:rest");
  CHECK(schema.num_rows == kChunks * kRowsPerChunk);
  CHECK(schema.names == std::vector<std::string>{"k", "v"});
  CHECK(schema.chunk_rows.size() == kChunks);

  sirius::hpln_io_stats read_stats;
  options.stats = &read_stats;
  std::vector<std::size_t> const want{0, 3};
  auto const ingested = sirius::read_hpln_chunks_into_pinned(uri, *env().host_space, want, options);
  CHECK(read_stats.transport == "io_context:rest");
  // The whole read: one tail probe, the chunk directory, every chunk header in one, and the
  // payloads of the two named chunks. Anything much larger means the walk went per-block.
  CHECK(read_stats.requests <= 6);
  require_chunks_decode_to(ingested, want, expected);

  // Half the file's chunks were asked for, and appreciably less than the whole object crossed the
  // wire -- which is the entire point of putting the format in front of an object store.
  CHECK(server.get_count() > 0);
  CHECK(server.body_bytes() < object.size());

  io_ctx->shutdown();
  fs::remove_all(dir);
}

TEST_CASE("hpln io - a .hpln in an object store is read through the REST io_context",
          "[s3][integration][hpln_io]")
{
  if (no_gpu()) { return; }
  if (!sirius::test::ensure_s3_container_env()) { return; }
  auto const* bucket   = std::getenv("SIRIUS_TEST_S3_BUCKET");
  auto const* endpoint = std::getenv("SIRIUS_TEST_S3_ENDPOINT");
  auto const* access   = std::getenv("SIRIUS_TEST_S3_ACCESS_KEY");
  auto const* secret   = std::getenv("SIRIUS_TEST_S3_SECRET_KEY");
  if (bucket == nullptr || endpoint == nullptr || access == nullptr || secret == nullptr) {
    WARN("SIRIUS_TEST_S3_* incomplete — skipping");
    return;
  }
  auto const* region = std::getenv("SIRIUS_TEST_S3_REGION");

  auto const dir      = scratch_dir("s3");
  auto const path     = (dir / "t.hpln").string();
  auto const expected = write_fixture(path);
  auto const object   = read_file(path);
  auto const key      = "hpln_io_" + std::to_string(::getpid()) + ".hpln";
  if (!sirius::test::put_s3_container_object(key, object)) {
    WARN("S3 environment is externally managed — skipping");
    fs::remove_all(dir);
    return;
  }

  auto io_ctx = make_rest_ioctx(endpoint, access, secret, region == nullptr ? "us-east-1" : region);
  if (!io_ctx) {
    WARN("no REST io_context available — skipping");
    fs::remove_all(dir);
    return;
  }

  auto const uri = "s3://" + std::string(bucket) + "/" + key;
  sirius::hpln_open_options options;
  options.io_ctx = io_ctx;
  sirius::hpln_io_stats stats;
  options.stats = &stats;

  auto const schema = sirius::read_hpln_schema(uri, options);
  CHECK(stats.transport == "io_context:rest");
  CHECK(schema.num_rows == kChunks * kRowsPerChunk);

  std::vector<std::size_t> const want{2};
  auto const ingested = sirius::read_hpln_chunks_into_pinned(uri, *env().host_space, want, options);
  require_chunks_decode_to(ingested, want, expected);

  io_ctx->shutdown();
  fs::remove_all(dir);
}
