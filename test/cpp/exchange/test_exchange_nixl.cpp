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
#include "exchange/nixl_fabric.hpp"
#include "exchange/transport_test_utils.hpp"

#include <cudf/copying.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
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
    for (rank_t from = 0; from < r.t->world_size(); ++from) {
      if (!r.got.ended.contains({from, barrier})) { return false; }
    }
    return true;
  }));
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

/// Two ranks in this process, both on its GPU, connected with @p staging.
std::pair<std::unique_ptr<nixl_rank>, std::unique_ptr<nixl_rank>> two_ranks(
  nixl_fabric::host_staging staging)
{
  std::vector<std::string> const peers{"127.0.0.1:" + std::to_string(free_port()),
                                       "127.0.0.1:" + std::to_string(free_port())};
  auto config = [&](rank_t rank) {
    nixl_fabric::config cfg{rank, peers};
    cfg.staging = staging;
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

TEST_CASE("two ranks in one process exchange batches over NIXL", "[exchange][nixl]")
{
  auto [r0, r1] = two_ranks(nixl_fabric::host_staging::never);
  REQUIRE_FALSE(r0->fab->host_staged(1));
  r0->start();
  r1->start();

  send_all(*r0->t, {1, 2}, 2, 20'000);
  send_all(*r1->t, {1, 2}, 2, 20'000);
  check_all(r0->got, 0, 2, {1, 2}, 2, 20'000);
  check_all(r1->got, 1, 2, {1, 2}, 2, 20'000);
}

TEST_CASE("two ranks in one process exchange batches through host staging", "[exchange][nixl]")
{
  // Frames go device to host into the peer's shared-memory mirror, then host to device.
  auto [r0, r1] = two_ranks(nixl_fabric::host_staging::always);
  REQUIRE(r0->fab->host_staged(1));
  REQUIRE(r1->fab->host_staged(0));
  r0->start();
  r1->start();

  send_all(*r0->t, {1, 2}, 2, 20'000);
  send_all(*r1->t, {1, 2}, 2, 20'000);
  check_all(r0->got, 0, 2, {1, 2}, 2, 20'000);
  check_all(r1->got, 1, 2, {1, 2}, 2, 20'000);
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
  cfg.staging      = nixl_fabric::staging_from_env();

  // 1 MiB frames; batches of about 7 MiB take several.
  nixl_rank r(cfg, KiB * KiB);
  r.start();
  send_all(*r.t, {1, 2}, 3, 200'000);
  check_all(r.got, cfg.rank, world, {1, 2}, 3, 200'000);
  barrier(r, 99);
}

TEST_CASE("one rank per process measures exchange bandwidth", "[.][nixl_bandwidth]")
{
  // Every rank sends the same number of large batches to every other rank at once, as a shuffle
  // does, through the transport with the runtime's ring sizes, and reports what it received per
  // second. SIRIUS_EXCHANGE_HOST_STAGING picks the path.
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
  cfg.staging      = nixl_fabric::staging_from_env();
  auto const world = static_cast<rank_t>(cfg.peers.size());
  frame_ring send(16, 8 * KiB * KiB);
  frame_ring recv(8 * (world - 1), 8 * KiB * KiB);
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
               }},
              // The test GPU space holds 1 GiB: keep queued batches well inside it.
              transport::options{.max_queued_bytes = std::size_t{256} << 20});

  // About 38 MB per batch (the test GPU space is small); 96 batches to each peer.
  auto const table    = sample_table(1'000'000);
  std::size_t batch_b = 0;
  for (cudf::size_type c = 0; c < table->num_columns(); ++c) {
    batch_b += table->get_column(c).alloc_size();
  }
  int const per_peer = 96;
  auto const start   = std::chrono::steady_clock::now();
  for (int i = 0; i < per_peer; ++i) {
    for (rank_t to = 0; to < world; ++to) {
      if (to == cfg.rank) { continue; }
      t.send(1, to, gpu_batch(std::make_unique<cudf::table>(*table)));
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
  double const gb = static_cast<double>(batch_b) * per_peer * (world - 1) / 1e9;
  std::printf("BANDWIDTH rank %u: received %.2f GB from %u peers in %.2f s: %.2f GB/s (%s)\n",
              cfg.rank,
              gb,
              world - 1,
              seconds,
              gb / seconds,
              fab.host_staged((cfg.rank + 1) % world) ? "host-staged" : "NIXL");
  // Every rank must have taken its batches before any tears down its fabric.
  for (rank_t to = 0; to < world; ++to) {
    t.close(99, to);
  }
  std::unique_lock lock{mutex};
  REQUIRE(changed.wait_for(lock, std::chrono::minutes{2}, [&] { return ended[99] == world; }));
}
