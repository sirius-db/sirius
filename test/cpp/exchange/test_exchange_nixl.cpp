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

// The exchange transport over NIXL: two ranks in one process, and (hidden, launched by
// scripts/distributed_test_nixl.sh) one rank per process and GPU.

#include "catch.hpp"
#include "exchange/exchange_runtime.hpp"
#include "exchange/exchange_service.hpp"
#include "exchange/nixl_fabric.hpp"
#include "exchange/transport_test_utils.hpp"
#include "sirius/exception.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

using namespace sirius::test::exchange_utils;
using sirius::exchange::frame_ring;
using sirius::exchange::nixl_fabric;

/// One rank: its rings, NIXL fabric, transport and what it received.
struct nixl_rank {
  frame_ring send;
  frame_ring recv;
  std::unique_ptr<nixl_fabric> fab;
  received got;
  std::unique_ptr<transport> t;

  nixl_rank(nixl_fabric::config cfg, std::size_t frame_bytes)
    : send(4, frame_bytes), recv(4 * static_cast<std::uint32_t>(cfg.peers.size() - 1), frame_bytes)
  {
    fab = std::make_unique<nixl_fabric>(std::move(cfg), send, recv);
  }

  void start()
  {
    t = std::make_unique<transport>(*fab, send, recv, *env().gpu, env().host, recording(got));
  }

  ~nixl_rank() { t.reset(); }
};

/// Closes stream @p barrier to every rank and waits for every rank's: past it, no rank still
/// needs anything from another.
void barrier(nixl_rank& r, stream_id barrier)
{
  for (rank_t to = 0; to < r.t->world_size(); ++to) {
    r.t->close(barrier, to);
  }
  REQUIRE(r.got.wait_for([&] {
    if (r.got.error) { return true; }
    for (rank_t from = 0; from < r.t->world_size(); ++from) {
      if (!r.got.ended.contains({from, barrier})) { return false; }
    }
    return true;
  }));
  REQUIRE_FALSE(r.got.error);
}

/// A free port below the ephemeral range: an ephemeral one could be taken as the source port of
/// a peer's outgoing connection before NIXL's listener binds it.
int free_port()
{
  static std::mt19937 rng{std::random_device{}()};
  for (int attempt = 0; attempt < 100; ++attempt) {
    int const port = 20000 + static_cast<int>(rng() % 10000);
    int const fd   = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_in at{};
    at.sin_family      = AF_INET;
    at.sin_addr.s_addr = htonl(INADDR_ANY);
    at.sin_port        = htons(static_cast<std::uint16_t>(port));
    bool const bound   = ::bind(fd, reinterpret_cast<sockaddr*>(&at), sizeof(at)) == 0;
    ::close(fd);
    if (bound) { return port; }
  }
  FAIL("no free port in [20000, 30000)");
  return 0;
}

/// Two distinct free_port()s as 127.0.0.1 addresses: the second is redrawn while it matches.
std::vector<std::string> two_free_peers()
{
  int const first = free_port();
  int second      = free_port();
  while (second == first) {
    second = free_port();
  }
  return {"127.0.0.1:" + std::to_string(first), "127.0.0.1:" + std::to_string(second)};
}

/// SIRIUS_EXCHANGE_HOST_STAGING for the hidden one-rank-per-process tests (default auto).
nixl_fabric::host_staging staging_from_env()
{
  auto const* value = std::getenv("SIRIUS_EXCHANGE_HOST_STAGING");
  return sirius::exchange::parse_host_staging(value != nullptr ? value : "auto");
}

/// The count in environment variable @p name for the hidden tests, or @p fallback when unset.
std::uint32_t count_from_env(char const* name, std::uint32_t fallback)
{
  auto const* value = std::getenv(name);
  return value != nullptr ? static_cast<std::uint32_t>(std::stoul(value)) : fallback;
}

/// Records what one stream's receiver is handed into @p got.
struct recording_inbound : sirius::exchange::inbound {
  recording_inbound(received& got, stream_id stream) : got(got), stream(stream) {}
  void on_batch(rank_t from, std::uint64_t, std::shared_ptr<cucascade::data_batch> data) override
  {
    auto d = digest(data);
    std::lock_guard lock{got.mutex};
    got.batches[{from, stream}].push_back(std::move(d));
    got.changed.notify_all();
  }
  void on_end(rank_t from) override
  {
    std::lock_guard lock{got.mutex};
    got.ended.insert({from, stream});
    got.changed.notify_all();
  }
  void on_error(std::exception_ptr error) override
  {
    std::lock_guard lock{got.mutex};
    got.error = error;
    got.changed.notify_all();
  }
  received& got;
  stream_id stream;
};

/// Two ranks in this process, both on its GPU, connected with @p staging and @p lanes.
std::pair<std::unique_ptr<nixl_rank>, std::unique_ptr<nixl_rank>> two_ranks(
  nixl_fabric::host_staging staging, std::uint32_t lanes = 0)
{
  auto const peers = two_free_peers();
  auto config      = [&](rank_t rank) {
    nixl_fabric::config cfg{rank, peers};
    cfg.staging = staging;
    cfg.lanes   = lanes;
    return cfg;
  };
  // Each constructor waits for the other's hello.
  std::unique_ptr<nixl_rank> r0;
  std::exception_ptr r0_error;
  std::thread other([&] {
    try {
      r0 = std::make_unique<nixl_rank>(config(0), 64 * KiB);
    } catch (...) {
      r0_error = std::current_exception();
    }
  });
  std::unique_ptr<nixl_rank> r1;
  try {
    r1 = std::make_unique<nixl_rank>(config(1), 64 * KiB);
  } catch (...) {
    other.join();
    throw;
  }
  other.join();
  if (r0_error) { std::rethrow_exception(r0_error); }
  return {std::move(r0), std::move(r1)};
}

}  // namespace

TEST_CASE("two ranks in one process exchange batches over NIXL or through host staging",
          "[exchange][nixl]")
{
  // Host staging: frames go device to host into the peer's shared-memory mirror, then host to
  // device.
  auto const staging =
    GENERATE(nixl_fabric::host_staging::never, nixl_fabric::host_staging::always);
  auto [r0, r1]     = two_ranks(staging);
  bool const staged = staging == nixl_fabric::host_staging::always;
  REQUIRE(r0->fab->host_staged(1) == staged);
  REQUIRE(r1->fab->host_staged(0) == staged);
  r0->start();
  r1->start();

  send_all(*r0->t, {1, 2}, 2, 20'000);
  send_all(*r1->t, {1, 2}, 2, 20'000);
  check_all(r0->got, 0, 2, {1, 2}, 2, 20'000);
  check_all(r1->got, 1, 2, {1, 2}, 2, 20'000);
}

TEST_CASE("two ranks in one process exchange batches over TCP lanes", "[exchange][nixl]")
{
  // Frames go device to host into a lane's send slot, host to host over the lane into the
  // peer's mirror, then host to device: the path to a rank on another host.
  auto [r0, r1] = two_ranks(nixl_fabric::host_staging::remote, 3);
  REQUIRE(r0->fab->path_to(1) == sirius::exchange::path::tcp_lanes);
  REQUIRE(r1->fab->path_to(0) == sirius::exchange::path::tcp_lanes);
  REQUIRE(r0->fab->lanes_to(1) == 3);
  r0->start();
  r1->start();

  send_all(*r0->t, {1, 2}, 2, 20'000);
  send_all(*r1->t, {1, 2}, 2, 20'000);
  check_all(r0->got, 0, 2, {1, 2}, 2, 20'000);
  check_all(r1->got, 1, 2, {1, 2}, 2, 20'000);
}

TEST_CASE("a fabric that cannot reach its peer leaves nothing in /dev/shm", "[exchange][nixl]")
{
  nixl_fabric::config cfg{0, two_free_peers()};
  cfg.connect_timeout = std::chrono::milliseconds{500};
  cfg.staging         = nixl_fabric::host_staging::always;
  frame_ring send(4, 64 * KiB);
  frame_ring recv(4, 64 * KiB);
  auto const mirror = "/dev/shm/sirius-exchange-" + std::to_string(::getpid()) + "-0";
  CHECK_THROWS_AS(nixl_fabric(cfg, send, recv), sirius::internal_exception);
  CHECK_FALSE(std::filesystem::exists(mirror));
}

TEST_CASE(
  "a NIXL fabric refuses an address that is not host:port, a rank out of range, and a "
  "listen port in use",
  "[exchange][nixl]")
{
  // Each is refused before any connection, so no peer is needed; the timeout bounds a miss.
  frame_ring send(4, 64 * KiB);
  frame_ring recv(4, 64 * KiB);
  auto const mine   = "127.0.0.1:" + std::to_string(free_port());
  auto const config = [](rank_t rank, std::vector<std::string> peers) {
    nixl_fabric::config cfg{rank, std::move(peers)};
    cfg.connect_timeout = std::chrono::milliseconds{500};
    return cfg;
  };
  for (std::string const bad :
       {"127.0.0.1", "127.0.0.1:", "127.0.0.1:27000x", "127.0.0.1:0", "127.0.0.1:70000"}) {
    CHECK_THROWS_AS(nixl_fabric(config(0, {mine, bad}), send, recv),
                    sirius::invalid_input_exception);
  }
  CHECK_THROWS_AS(nixl_fabric(config(2, {mine, mine}), send, recv),
                  sirius::invalid_input_exception);
  CHECK_THROWS_AS(nixl_fabric(config(0, {}), send, recv), sirius::invalid_input_exception);

  int const port = free_port();
  int const fd   = ::socket(AF_INET, SOCK_STREAM, 0);
  REQUIRE(fd >= 0);
  sockaddr_in at{};
  at.sin_family      = AF_INET;
  at.sin_addr.s_addr = htonl(INADDR_ANY);
  at.sin_port        = htons(static_cast<std::uint16_t>(port));
  REQUIRE(::bind(fd, reinterpret_cast<sockaddr*>(&at), sizeof(at)) == 0);
  REQUIRE(::listen(fd, 1) == 0);
  auto const taken = config(0, {"127.0.0.1:" + std::to_string(port), mine});
  CHECK_THROWS_AS(nixl_fabric(taken, send, recv), sirius::invalid_input_exception);
  CHECK_THROWS_WITH(nixl_fabric(taken, send, recv), Catch::Matchers::ContainsSubstring("in use"));
  ::close(fd);
}

TEST_CASE("two ranks connect through make_nixl_exchange and exchange a batch", "[exchange][nixl]")
{
  using sirius::exchange::exchange_service;
  auto const peers = two_free_peers();
  sirius::exchange::nixl_exchange_options const opts{
    .frame_bytes = 64 * KiB, .send_frames = 4, .recv_frames_per_peer = 4};
  // Each waits for the other's hello.
  std::shared_ptr<exchange_service> s0;
  std::exception_ptr s0_error;
  std::thread other([&] {
    try {
      s0 = sirius::exchange::make_nixl_exchange(0, peers, *env().gpu, env().host, opts);
    } catch (...) {
      s0_error = std::current_exception();
    }
  });
  std::shared_ptr<exchange_service> s1;
  try {
    s1 = sirius::exchange::make_nixl_exchange(1, peers, *env().gpu, env().host, opts);
  } catch (...) {
    other.join();
    throw;
  }
  other.join();
  if (s0_error) { std::rethrow_exception(s0_error); }

  stream_id const stream = sirius::exchange::make_stream_id(1, 1);
  std::vector<std::shared_ptr<exchange_service>> services{s0, s1};
  std::vector<received> got(2);
  for (rank_t r = 0; r < 2; ++r) {
    services[r]->attach(stream, std::make_shared<recording_inbound>(got[r], stream));
  }
  for (rank_t r = 0; r < 2; ++r) {
    services[r]->send(stream, 1 - r, exchange_batch(r, 1 - r, stream, 0, 20'000));
    services[r]->close(stream, 1 - r);
  }
  for (rank_t r = 0; r < 2; ++r) {
    auto& g = got[r];
    REQUIRE(g.wait_for([&] { return g.error != nullptr || !g.ended.empty(); }));
    std::lock_guard lock{g.mutex};
    REQUIRE_FALSE(g.error);
    CHECK(g.batches[{1 - r, stream}] ==
          std::vector<table_digest>{digest(exchange_batch(1 - r, r, stream, 0, 20'000))});
  }
  // The last reference tears down the service, then the fabric and the rings.
  services.clear();
  s0.reset();
  s1.reset();
}

TEST_CASE("one rank per process exchanges batches over NIXL", "[.][nixl_multiprocess]")
{
  auto const* rank  = std::getenv("SIRIUS_EXCHANGE_RANK");
  auto const* peers = std::getenv("SIRIUS_EXCHANGE_PEERS");
  REQUIRE(rank != nullptr);
  REQUIRE(peers != nullptr);
  nixl_fabric::config cfg;
  cfg.rank = static_cast<rank_t>(std::stoul(rank));
  std::stringstream list{peers};
  for (std::string peer; std::getline(list, peer, ',');) {
    cfg.peers.push_back(peer);
  }
  auto const world = static_cast<rank_t>(cfg.peers.size());
  cfg.staging      = staging_from_env();
  cfg.lanes        = count_from_env("SIRIUS_EXCHANGE_TCP_LANES", 0);

  // 1 MiB frames; batches of about 7 MiB take several.
  nixl_rank r(cfg, MiB);
  r.start();
  send_all(*r.t, {1, 2}, 3, 200'000);
  check_all(r.got, cfg.rank, world, {1, 2}, 3, 200'000);
  barrier(r, 99);
}

TEST_CASE("one rank per process measures exchange bandwidth", "[.][nixl_bandwidth]")
{
  // Every rank sends the same number of large batches to every other rank at once, as a shuffle
  // does, through the transport with the runtime's ring sizes, and reports what it received per
  // second. SIRIUS_EXCHANGE_HOST_STAGING and SIRIUS_EXCHANGE_MNNVL (default auto) pick the path
  // (as sirius_exchange_host_staging and sirius_exchange_mnnvl do), SIRIUS_EXCHANGE_TCP_LANES and
  // SIRIUS_EXCHANGE_RECV_FRAMES_PER_PEER size it (as the settings sirius_exchange_tcp_lanes and
  // sirius_exchange_recv_frames_per_peer do), and SIRIUS_EXCHANGE_BW_BATCHES (default 96) sets
  // the batches to each peer.
  auto const* rank  = std::getenv("SIRIUS_EXCHANGE_RANK");
  auto const* peers = std::getenv("SIRIUS_EXCHANGE_PEERS");
  REQUIRE(rank != nullptr);
  REQUIRE(peers != nullptr);
  nixl_fabric::config cfg;
  cfg.rank = static_cast<rank_t>(std::stoul(rank));
  std::stringstream list{peers};
  for (std::string peer; std::getline(list, peer, ',');) {
    cfg.peers.push_back(peer);
  }
  cfg.staging       = staging_from_env();
  cfg.lanes         = count_from_env("SIRIUS_EXCHANGE_TCP_LANES", 0);
  auto const* mnnvl = std::getenv("SIRIUS_EXCHANGE_MNNVL");
  cfg.mnnvl         = sirius::exchange::parse_mnnvl(mnnvl != nullptr ? mnnvl : "auto");
  auto const world  = static_cast<rank_t>(cfg.peers.size());
  sirius::exchange::nixl_exchange_options const ring_opts{};
  auto const memory = sirius::exchange::ring_memory_for(cfg.mnnvl);
  auto const recv_frames =
    count_from_env("SIRIUS_EXCHANGE_RECV_FRAMES_PER_PEER", ring_opts.recv_frames_per_peer);
  frame_ring send(ring_opts.send_frames, ring_opts.frame_bytes, memory);
  frame_ring recv(recv_frames * (world - 1), ring_opts.frame_bytes, memory);
  nixl_fabric fab(cfg, send, recv);

  std::mutex mutex;
  std::condition_variable changed;
  std::size_t batches = 0;
  // Ends per stream: a fast peer's barrier end (stream 99) can arrive before a slow peer's data
  // end.
  std::map<stream_id, std::size_t> ended;
  std::exception_ptr error;
  transport t(fab,
              send,
              recv,
              *env().gpu,
              env().host,
              {[&](stream_id, rank_t, std::uint64_t, std::shared_ptr<cucascade::data_batch>) {
                 std::lock_guard lock{mutex};
                 ++batches;
                 changed.notify_all();
               },
               [&](stream_id stream, rank_t) {
                 std::lock_guard lock{mutex};
                 ++ended[stream];
                 changed.notify_all();
               },
               [&](std::exception_ptr e) {
                 std::lock_guard lock{mutex};
                 error = e;
                 changed.notify_all();
               },
               [&](std::uint64_t, rank_t, std::string what) {
                 std::lock_guard lock{mutex};
                 error = std::make_exception_ptr(std::runtime_error(what));
                 changed.notify_all();
               }},
              // The test GPU space holds 1 GiB: keep queued batches well inside it.
              transport::options{.max_queued_bytes = 256 * MiB});

  // About 38 MB per batch (the test GPU space is small); 96 batches to each peer.
  auto table          = sample_table(1'000'000);
  std::size_t batch_b = 0;
  for (cudf::size_type c = 0; c < table->num_columns(); ++c) {
    batch_b += table->get_column(c).alloc_size();
  }
  // Sent to every peer, every time: the transport only reads it, and the timing then leaves out
  // building batches.
  auto const batch    = gpu_batch(std::move(table));
  auto const per_peer = static_cast<int>(count_from_env("SIRIUS_EXCHANGE_BW_BATCHES", 96));
  // Start the clocks together: a rank that starts first would count its wait for the others.
  for (rank_t to = 0; to < world; ++to) {
    t.close(98, to);
  }
  {
    std::unique_lock lock{mutex};
    REQUIRE(changed.wait_for(
      lock, std::chrono::minutes{2}, [&] { return error != nullptr || ended[98] == world; }));
    REQUIRE_FALSE(error);
  }
  auto const start = std::chrono::steady_clock::now();
  for (int i = 0; i < per_peer; ++i) {
    for (rank_t to = 0; to < world; ++to) {
      if (to == cfg.rank) { continue; }
      t.send(1, to, batch);
    }
  }
  for (rank_t to = 0; to < world; ++to) {
    if (to != cfg.rank) { t.close(1, to); }
  }
  {
    std::unique_lock lock{mutex};
    REQUIRE(changed.wait_for(
      lock, std::chrono::minutes{10}, [&] { return error != nullptr || ended[1] == world - 1; }));
    REQUIRE_FALSE(error);
    REQUIRE(batches == static_cast<std::size_t>(per_peer) * (world - 1));
  }
  auto const seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  double const gb       = static_cast<double>(batch_b) * per_peer * (world - 1) / 1e9;
  auto const next       = (cfg.rank + 1) % world;
  auto const route      = fab.path_to(next);
  std::string const how = route == sirius::exchange::path::tcp_lanes
                            ? std::to_string(fab.lanes_to(next)) + " TCP lanes"
                            : sirius::exchange::describe(route);
  std::printf("BANDWIDTH rank %u: received %.2f GB from %u peers in %.2f s: %.2f GB/s (%s)\n",
              cfg.rank,
              gb,
              world - 1,
              seconds,
              gb / seconds,
              how.c_str());
  // Every rank must have taken its batches before any tears down its fabric.
  for (rank_t to = 0; to < world; ++to) {
    t.close(99, to);
  }
  std::unique_lock lock{mutex};
  REQUIRE(changed.wait_for(
    lock, std::chrono::minutes{2}, [&] { return error != nullptr || ended[99] == world; }));
  REQUIRE_FALSE(error);
  // The last end this rank sent may still be queued in UCX: tearing down now could drop it and
  // leave a slower peer waiting.
  lock.unlock();
  std::this_thread::sleep_for(std::chrono::seconds{2});
}

namespace {

/// A setting for the hidden multi-process tests: the environment variable @p name, else
/// @p fallback.
std::string env_or(char const* name, std::string fallback)
{
  auto const* value = std::getenv(name);
  return value != nullptr && *value != '\0' ? std::string{value} : std::move(fallback);
}

/// One GPU memory space of @p gpu_bytes, for batches far larger than env()'s.
struct big_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu  = nullptr;
  cucascade::memory::memory_space* host = nullptr;

  explicit big_env(std::size_t gpu_bytes)
  {
    sirius::converter_registry::reset_for_testing();
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(gpu_bytes)
      .set_reservation_fraction_per_gpu(0.9)
      .set_per_numa_region_capacity(std::size_t{4} << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.9);
    mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    gpu  = mgr->get_memory_space(Tier::GPU, 0);
    host = mgr->get_memory_space(Tier::HOST, 0);
  }
};

/// A batch of one INT64 column of @p bytes / 8 values: @p first, @p first + 1, ...
std::shared_ptr<cucascade::data_batch> sequence_batch(big_env& e,
                                                      std::size_t bytes,
                                                      std::int64_t first)
{
  auto stream = utils::default_stream();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::sequence(static_cast<cudf::size_type>(bytes / sizeof(std::int64_t)),
                                   cudf::numeric_scalar<std::int64_t>(first, true, stream),
                                   cudf::numeric_scalar<std::int64_t>(1, true, stream),
                                   stream));
  stream.sync();
  return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                 *e.gpu,
                                 stream,
                                 sirius::telemetry::batch_telemetry_info{});
}

/// How many values of @p batch differ from @p first, @p first + 1, ...
std::int64_t bad_values(std::shared_ptr<cucascade::data_batch> const& batch, std::int64_t first)
{
  auto ro = batch->to_read_only();
  REQUIRE(ro.get_current_tier() == Tier::GPU);
  if (cudaEvent_t writer = ro.get_writer_event()) { cudaEventSynchronize(writer); }
  auto const view     = ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
  auto stream         = utils::default_stream();
  auto const expected = cudf::sequence(view.num_rows(),
                                       cudf::numeric_scalar<std::int64_t>(first, true, stream),
                                       cudf::numeric_scalar<std::int64_t>(1, true, stream),
                                       stream);
  auto const differ   = cudf::binary_operation(view.column(0),
                                             expected->view(),
                                             cudf::binary_operator::NOT_EQUAL,
                                             cudf::data_type{cudf::type_id::BOOL8},
                                             stream);
  auto const sum      = cudf::reduce(differ->view(),
                                *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
                                cudf::data_type{cudf::type_id::INT64},
                                stream);
  return static_cast<cudf::numeric_scalar<std::int64_t> const&>(*sum).value(stream);
}

}  // namespace

TEST_CASE("pairs of ranks on two nodes move batches at NVLink speed", "[.][nixl_pairs]")
{
  // A raw NIXL bandwidth benchmark's phases, through the exchange: world / 2 ranks on each of two
  // nodes, rank r paired with rank (r + world / 2) % world, one rank per GPU. Each phase
  // sends one batch size for SIRIUS_EXCHANGE_PAIRS_SECONDS from the ranks that take part, and
  // each receiver reports what it received per second. Last, every rank sends its peer distinct
  // batches, and the peer checks every value. Ring sizes and sirius_exchange_mnnvl come from
  // SIRIUS_EXCHANGE_{FRAME_BYTES, SEND_FRAMES, RECV_FRAMES_PER_PEER, MNNVL} (default require).
  auto const* rank  = std::getenv("SIRIUS_EXCHANGE_RANK");
  auto const* peers = std::getenv("SIRIUS_EXCHANGE_PEERS");
  REQUIRE(rank != nullptr);
  REQUIRE(peers != nullptr);
  nixl_fabric::config cfg;
  cfg.rank = static_cast<rank_t>(std::stoul(rank));
  std::stringstream list{peers};
  for (std::string peer; std::getline(list, peer, ',');) {
    cfg.peers.push_back(peer);
  }
  auto const world = static_cast<rank_t>(cfg.peers.size());
  REQUIRE((world >= 2 && world % 2 == 0));
  rank_t const per_node = world / 2;
  rank_t const me       = cfg.rank;
  rank_t const peer     = (me + per_node) % world;
  rank_t const node     = me / per_node;

  sirius::exchange::nixl_exchange_options opts;
  opts.mnnvl       = sirius::exchange::parse_mnnvl(env_or("SIRIUS_EXCHANGE_MNNVL", "require"));
  opts.frame_bytes = std::stoull(env_or("SIRIUS_EXCHANGE_FRAME_BYTES", "16777216"));
  opts.send_frames =
    static_cast<std::uint32_t>(std::stoul(env_or("SIRIUS_EXCHANGE_SEND_FRAMES", "32")));
  opts.recv_frames_per_peer =
    static_cast<std::uint32_t>(std::stoul(env_or("SIRIUS_EXCHANGE_RECV_FRAMES_PER_PEER", "16")));
  cfg.staging        = staging_from_env();
  cfg.mnnvl          = opts.mnnvl;
  auto const seconds = std::stod(env_or("SIRIUS_EXCHANGE_PAIRS_SECONDS", "1.5"));
  big_env e(std::stoull(env_or("SIRIUS_EXCHANGE_PAIRS_GPU_GIB", "40")) << 30);

  auto const memory = sirius::exchange::ring_memory_for(opts.mnnvl);
  frame_ring send(opts.send_frames, opts.frame_bytes, memory);
  frame_ring recv(opts.recv_frames_per_peer * (world - 1), opts.frame_bytes, memory);
  nixl_fabric fab(cfg, send, recv);
  std::printf("PAIRS rank %u peer %u: %s, rings %s, frame %zu MiB x %u send, %u per peer\n",
              me,
              peer,
              sirius::exchange::describe(fab.path_to(peer)),
              recv.in_fabric_memory() ? "in fabric memory" : "from cudaMalloc",
              opts.frame_bytes >> 20,
              opts.send_frames,
              opts.recv_frames_per_peer);
  if (opts.mnnvl == sirius::exchange::mnnvl_mode::require) {
    REQUIRE(fab.path_to(peer) == sirius::exchange::path::nvlink_fabric);
  }

  std::mutex mutex;
  std::condition_variable changed;
  std::map<stream_id, std::size_t> received_bytes;
  std::map<stream_id, std::chrono::steady_clock::time_point> ended_at;
  std::map<stream_id, std::size_t> ends;
  std::vector<std::tuple<rank_t, std::uint64_t, std::shared_ptr<cucascade::data_batch>>> kept;
  stream_id constexpr verify_stream = 900;
  std::exception_ptr error;
  transport t(fab,
              send,
              recv,
              *e.gpu,
              e.host,
              {[&](stream_id stream,
                   rank_t from,
                   std::uint64_t index,
                   std::shared_ptr<cucascade::data_batch> batch) {
                 auto const bytes = batch->to_read_only().get_data()->get_size_in_bytes();
                 std::lock_guard lock{mutex};
                 received_bytes[stream] += bytes;
                 if (stream == verify_stream) { kept.emplace_back(from, index, std::move(batch)); }
                 changed.notify_all();
               },
               [&](stream_id stream, rank_t) {
                 std::lock_guard lock{mutex};
                 ++ends[stream];
                 ended_at[stream] = std::chrono::steady_clock::now();
                 changed.notify_all();
               },
               [&](std::exception_ptr err) {
                 std::lock_guard lock{mutex};
                 error = err;
                 changed.notify_all();
               },
               [&](std::uint64_t, rank_t, std::string what) {
                 std::lock_guard lock{mutex};
                 error = std::make_exception_ptr(std::runtime_error(what));
                 changed.notify_all();
               }},
              transport::options{.max_queued_bytes = std::size_t{2} << 30});

  // Every rank closes @p stream to every rank and waits for every rank's close.
  auto barrier = [&](stream_id stream) {
    for (rank_t to = 0; to < world; ++to) {
      t.close(stream, to);
    }
    std::unique_lock lock{mutex};
    REQUIRE(changed.wait_for(
      lock, std::chrono::minutes{5}, [&] { return error != nullptr || ends[stream] == world; }));
    REQUIRE_FALSE(error);
  };

  struct phase {
    std::string tag;
    std::size_t batch_bytes;
    bool both_ways;
    bool all_pairs;
  };
  constexpr std::size_t KB = 1024, MB = KB * KB, GB = MB * KB;
  std::vector<phase> phases;
  for (auto const size : {GB, 128 * MB}) {
    auto const s = size == GB ? std::string{"1G"} : std::string{"128M"};
    phases.push_back({"uni_1pair_" + s, size, false, false});
    phases.push_back({"bidi_1pair_" + s, size, true, false});
    phases.push_back({"uni_4pair_" + s, size, false, true});
    phases.push_back({"bidi_4pair_" + s, size, true, true});
  }
  for (auto const size : {64 * KB, 256 * KB, 1 * MB, 4 * MB, 16 * MB, 64 * MB}) {
    auto const s = size >= MB ? std::to_string(size / MB) + "M" : std::to_string(size / KB) + "K";
    phases.push_back({"batches_" + s, size, false, false});
  }
  phases.push_back({"bidi_4pair_1M", 1 * MB, true, true});

  barrier(1);
  for (std::size_t i = 0; i < phases.size(); ++i) {
    auto const& p       = phases[i];
    stream_id const s   = 100 + i;
    bool const sender   = (p.all_pairs || me % per_node == 0) && (p.both_ways || node == 0);
    bool const receiver = (p.all_pairs || peer % per_node == 0) && (p.both_ways || node == 1);
    auto const batch    = sender ? sequence_batch(e, p.batch_bytes, me) : nullptr;
    barrier(1000 + i);
    auto const start = std::chrono::steady_clock::now();
    std::size_t sent = 0;
    if (sender) {
      while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() <
             seconds) {
        t.send(s, peer, batch);
        ++sent;
      }
    }
    for (rank_t to = 0; to < world; ++to) {
      t.close(s, to);
    }
    {
      std::unique_lock lock{mutex};
      REQUIRE(changed.wait_for(
        lock, std::chrono::minutes{5}, [&] { return error != nullptr || ends[s] == world; }));
      REQUIRE_FALSE(error);
      if (receiver) {
        auto const secs = std::chrono::duration<double>(ended_at[s] - start).count();
        std::printf("PAIRS %-16s rank %u <- %u: %zu bytes in %.3f s: %.1f GB/s\n",
                    p.tag.c_str(),
                    me,
                    peer,
                    received_bytes[s],
                    secs,
                    static_cast<double>(received_bytes[s]) / secs / 1e9);
      }
    }
    std::fflush(stdout);
    static_cast<void>(sent);
  }

  // Every rank sends its peer distinct batches; the peer checks every value.
  auto const verify_batches = 8;
  auto const verify_bytes   = 256 * MB;
  barrier(2);
  for (int i = 0; i < verify_batches; ++i) {
    auto const first = (static_cast<std::int64_t>(me) << 40) + (static_cast<std::int64_t>(i) << 32);
    t.send(verify_stream, peer, sequence_batch(e, verify_bytes, first));
  }
  for (rank_t to = 0; to < world; ++to) {
    t.close(verify_stream, to);
  }
  {
    std::unique_lock lock{mutex};
    REQUIRE(changed.wait_for(lock, std::chrono::minutes{5}, [&] {
      return error != nullptr || ends[verify_stream] == world;
    }));
    REQUIRE_FALSE(error);
  }
  REQUIRE(kept.size() == static_cast<std::size_t>(verify_batches));
  std::int64_t bad = 0;
  for (auto const& [from, index, batch] : kept) {
    REQUIRE(from == peer);
    auto const first =
      (static_cast<std::int64_t>(from) << 40) + (static_cast<std::int64_t>(index) << 32);
    bad += bad_values(batch, first);
  }
  kept.clear();
  std::printf("PAIRS verify rank %u <- %u: %d batches of %zu MiB, %lld bad values\n",
              me,
              peer,
              verify_batches,
              verify_bytes >> 20,
              static_cast<long long>(bad));
  CHECK(bad == 0);
  barrier(3);
}
