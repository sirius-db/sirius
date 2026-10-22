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
// scripts/exchange_nixl_ranks.sh) one rank per process and GPU.

#include "catch.hpp"
#include "exchange/nixl_fabric.hpp"
#include "exchange/transport_test_utils.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <exception>
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

}  // namespace

TEST_CASE("two ranks in one process exchange batches over NIXL", "[exchange][nixl]")
{
  std::vector<std::string> const peers{"127.0.0.1:" + std::to_string(free_port()),
                                       "127.0.0.1:" + std::to_string(free_port())};
  // Each constructor waits for the other's hello.
  std::unique_ptr<nixl_rank> r0;
  std::exception_ptr r0_error;
  std::thread other([&] {
    try {
      r0 = std::make_unique<nixl_rank>(nixl_fabric::config{0, peers}, 64 * KiB);
    } catch (...) {
      r0_error = std::current_exception();
    }
  });
  std::unique_ptr<nixl_rank> r1;
  try {
    r1 = std::make_unique<nixl_rank>(nixl_fabric::config{1, peers}, 64 * KiB);
  } catch (...) {
    other.join();
    throw;
  }
  other.join();
  if (r0_error) { std::rethrow_exception(r0_error); }
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

  // 1 MiB frames; batches of about 7 MiB take several.
  nixl_rank r(cfg, KiB * KiB);
  r.start();
  send_all(*r.t, {1, 2}, 3, 200'000);
  check_all(r.got, cfg.rank, world, {1, 2}, 3, 200'000);
  barrier(r, 99);
}
