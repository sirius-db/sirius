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

// How the exchange reaches a peer, and the fabric memory that lets a GPU on another node write
// its rings over NVLink.

#include "catch.hpp"
#include "exchange/fabric.hpp"
#include "exchange/path_choice.hpp"
#include "sirius/exception.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace sirius::exchange;

nvlink_domain domain(std::uint8_t cluster, std::uint32_t clique)
{
  nvlink_domain d;
  d.cluster_uuid.fill(cluster);
  d.clique_id = clique;
  return d;
}

endpoint at(rank_t rank, bool fabric_ring, nvlink_domain d)
{
  return endpoint{rank, "0008:01:00.0", fabric_ring, d};
}

}  // namespace

TEST_CASE("a peer in the same NVLink domain, both rings in fabric memory, gets the fabric",
          "[exchange][path]")
{
  auto const me   = at(0, true, domain(7, 32766));
  auto const peer = at(1, true, domain(7, 32766));
  for (auto const staging : {host_staging::automatic, host_staging::never}) {
    for (auto const mnnvl : {mnnvl_mode::automatic, mnnvl_mode::require}) {
      CHECK(choose_path(me, peer, staging, mnnvl, true) == path::nvlink_fabric);
      // Peer access matters only off the fabric.
      CHECK(choose_path(me, peer, staging, mnnvl, false) == path::nvlink_fabric);
    }
  }
  // always still stages every peer through shared memory, as before.
  CHECK(choose_path(me, peer, host_staging::always, mnnvl_mode::automatic, true) ==
        path::shared_memory);
}

TEST_CASE("off the NVLink fabric, the path is what it was", "[exchange][path]")
{
  auto const me = at(0, true, domain(7, 32766));
  std::vector<endpoint> const elsewhere{at(1, true, domain(7, 1)),        // another clique
                                        at(1, true, domain(8, 32766)),    // another cluster
                                        at(1, true, nvlink_domain{}),     // no domain
                                        at(1, false, domain(7, 32766))};  // a cudaMalloc ring
  for (auto const& peer : elsewhere) {
    CHECK(choose_path(me, peer, host_staging::automatic, mnnvl_mode::automatic, true) ==
          path::nixl);
    CHECK(choose_path(me, peer, host_staging::automatic, mnnvl_mode::automatic, false) ==
          path::shared_memory);
    CHECK(choose_path(me, peer, host_staging::never, mnnvl_mode::automatic, false) == path::nixl);
    CHECK_THROWS_WITH(choose_path(me, peer, host_staging::automatic, mnnvl_mode::require, true),
                      Catch::Matchers::ContainsSubstring("cannot write rank 1"));
  }
  // Two invalid domains are not one domain.
  CHECK(choose_path(
          at(0, true, {}), at(1, true, {}), host_staging::never, mnnvl_mode::automatic, true) ==
        path::nixl);
  // off never takes the fabric, even inside one domain.
  CHECK(
    choose_path(me, at(1, true, domain(7, 32766)), host_staging::never, mnnvl_mode::off, true) ==
    path::nixl);
}

TEST_CASE("a peer on another host gets TCP lanes unless the NVLink fabric reaches it",
          "[exchange][path]")
{
  auto on = [](endpoint e, char const* boot_id) {
    e.boot_id = boot_id;
    return e;
  };
  auto const me      = on(at(0, true, domain(7, 32766)), "boot-a");
  auto const near    = on(at(1, false, {}), "boot-a");
  auto const far     = on(at(1, false, {}), "boot-b");
  auto const far_fab = on(at(1, true, domain(7, 32766)), "boot-b");
  for (bool const peer_access : {true, false}) {
    // Another host: lanes, whatever NVML says of a GPU it cannot see.
    CHECK(choose_path(me, far, host_staging::automatic, mnnvl_mode::automatic, peer_access) ==
          path::tcp_lanes);
    CHECK(choose_path(me, far, host_staging::automatic, mnnvl_mode::off, peer_access) ==
          path::tcp_lanes);
    CHECK(choose_path(me, far_fab, host_staging::automatic, mnnvl_mode::off, peer_access) ==
          path::tcp_lanes);
    CHECK(choose_path(me, far, host_staging::always, mnnvl_mode::automatic, peer_access) ==
          path::tcp_lanes);
    CHECK(choose_path(me, far, host_staging::never, mnnvl_mode::automatic, peer_access) ==
          path::nixl);
    // The NVLink fabric reaches other hosts, and wins there.
    CHECK(choose_path(me, far_fab, host_staging::automatic, mnnvl_mode::automatic, peer_access) ==
          path::nvlink_fabric);
    CHECK(choose_path(me, far_fab, host_staging::never, mnnvl_mode::require, peer_access) ==
          path::nvlink_fabric);
    CHECK(choose_path(me, far_fab, host_staging::always, mnnvl_mode::automatic, peer_access) ==
          path::tcp_lanes);
    // remote: lanes to every peer, even on this host or on the fabric.
    for (auto const& peer : {near, far, far_fab}) {
      CHECK(choose_path(me, peer, host_staging::remote, mnnvl_mode::automatic, peer_access) ==
            path::tcp_lanes);
    }
    CHECK(choose_path(me, near, host_staging::always, mnnvl_mode::automatic, peer_access) ==
          path::shared_memory);
  }
  // This host: as before.
  CHECK(choose_path(me, near, host_staging::automatic, mnnvl_mode::automatic, true) == path::nixl);
  CHECK(choose_path(me, near, host_staging::automatic, mnnvl_mode::automatic, false) ==
        path::shared_memory);
  CHECK_THROWS_WITH(choose_path(me, far, host_staging::automatic, mnnvl_mode::require, true),
                    Catch::Matchers::ContainsSubstring("cannot write rank 1"));
  CHECK(std::string{describe(path::tcp_lanes)} == "host-staged over TCP lanes");
}

TEST_CASE("sirius_exchange_mnnvl takes auto, require or off", "[exchange][path]")
{
  CHECK(parse_mnnvl("auto") == mnnvl_mode::automatic);
  CHECK(parse_mnnvl("require") == mnnvl_mode::require);
  CHECK(parse_mnnvl("off") == mnnvl_mode::off);
  CHECK_THROWS_AS(parse_mnnvl(""), sirius::invalid_input_exception);
  CHECK_THROWS_AS(parse_mnnvl("on"), sirius::invalid_input_exception);
  CHECK(ring_memory_for(mnnvl_mode::automatic) == ring_memory::fabric_or_plain);
  CHECK(ring_memory_for(mnnvl_mode::require) == ring_memory::fabric);
  CHECK(ring_memory_for(mnnvl_mode::off) == ring_memory::plain);
  CHECK(domain(7, 32766).to_string() == "07070707-0707-0707-0707-070707070707/32766");
  CHECK(nvlink_domain{}.to_string() == "none");
}

TEST_CASE("a frame ring in fabric memory holds what is written to it", "[exchange][path]")
{
  frame_ring plain(2, 4096, ring_memory::plain);
  CHECK_FALSE(plain.in_fabric_memory());
  frame_ring ring(3, 1 << 20, ring_memory::fabric_or_plain);
  if (!ring.in_fabric_memory()) {
    // Not every GPU and driver have fabric handles; then fabric_or_plain falls back.
    CHECK_FALSE(ring.why_not_fabric().empty());
    CHECK_THROWS_AS(frame_ring(1, 4096, ring_memory::fabric), sirius::internal_exception);
    SKIP("no fabric memory here: " << ring.why_not_fabric());
  }
  CHECK(ring.why_not_fabric().empty());
  CHECK(ring.bytes() == 3u << 20);
  std::vector<std::uint32_t> in(ring.bytes() / 4);
  for (std::size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<std::uint32_t>(i * 2654435761u);
  }
  REQUIRE(cudaMemcpy(ring.base(), in.data(), ring.bytes(), cudaMemcpyHostToDevice) == cudaSuccess);
  // A device-to-device copy between frames, as the transport stages and lands them.
  REQUIRE(cudaMemcpy(ring.frame(0), ring.frame(2), ring.frame_bytes(), cudaMemcpyDeviceToDevice) ==
          cudaSuccess);
  std::vector<std::uint32_t> out(in.size());
  REQUIRE(cudaMemcpy(out.data(), ring.base(), ring.bytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
  auto const per_frame = ring.frame_bytes() / 4;
  for (std::size_t i = 0; i < per_frame; ++i) {
    REQUIRE(out[i] == in[2 * per_frame + i]);
  }
  for (std::size_t i = per_frame; i < out.size(); ++i) {
    REQUIRE(out[i] == in[i]);
  }
}
