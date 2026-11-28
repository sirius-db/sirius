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
  // always stages a same-host peer through shared memory, even on the fabric.
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
  // This host, off the fabric: NIXL with peer access, else shared memory.
  CHECK(choose_path(me, near, host_staging::automatic, mnnvl_mode::automatic, true) == path::nixl);
  CHECK(choose_path(me, near, host_staging::automatic, mnnvl_mode::automatic, false) ==
        path::shared_memory);
  CHECK_THROWS_WITH(choose_path(me, far, host_staging::automatic, mnnvl_mode::require, true),
                    Catch::Matchers::ContainsSubstring("cannot write rank 1"));
  // require cannot hold where every peer is staged, even for a peer the fabric reaches.
  for (auto const staging : {host_staging::always, host_staging::remote}) {
    CHECK_THROWS_WITH(choose_path(me, far_fab, staging, mnnvl_mode::require, true),
                      Catch::Matchers::ContainsSubstring("stages every peer"));
  }
  CHECK(std::string{describe(path::tcp_lanes)} == "host-staged over TCP lanes");
}

TEST_CASE("on a host with more than two ranks, GPUs joined only by PCIe stage each other",
          "[exchange][path]")
{
  auto gpu = [](rank_t rank, char const* bus_id, nvlink_domain d, bool mirror) {
    return endpoint{rank, bus_id, false, d, "boot-a", mirror};
  };
  auto const me        = gpu(0, "0000:01:00.0", {}, true);
  auto const near      = gpu(1, "0000:02:00.0", {}, true);
  auto const bare      = gpu(1, "0000:02:00.0", {}, false);
  auto const automatic = [](endpoint const& a, endpoint const& b, host_view host) {
    return choose_path(a, b, host_staging::automatic, mnnvl_mode::automatic, true, host);
  };
  CHECK(automatic(me, near, {.ranks = 2}) == path::nixl);
  for (auto const ranks : {std::size_t{3}, std::size_t{8}}) {
    CHECK(automatic(me, near, {.ranks = ranks}) == path::shared_memory);
    // Staging needs the peer's mirror, whatever this rank has.
    CHECK(automatic(me, bare, {.ranks = ranks}) == path::nixl);
    CHECK(automatic(gpu(0, "0000:01:00.0", {}, false), near, {.ranks = ranks}) ==
          path::shared_memory);
    // Ranks on one GPU, and a GPU with NVLink, share no PCIe link.
    CHECK(automatic(me, gpu(1, "0000:01:00.0", {}, true), {.ranks = ranks}) == path::nixl);
    CHECK(automatic(me, near, {.ranks = ranks, .nvlink = true}) == path::nixl);
    auto far    = near;
    far.boot_id = "boot-b";
    CHECK(automatic(me, far, {.ranks = ranks}) == path::tcp_lanes);
    // The other modes, and a peer without peer access, are as on any host.
    auto const with = [&](host_staging staging, endpoint const& peer, bool peer_access) {
      return choose_path(me, peer, staging, mnnvl_mode::automatic, peer_access, {.ranks = ranks});
    };
    CHECK(with(host_staging::never, near, true) == path::nixl);
    CHECK(with(host_staging::always, bare, true) == path::shared_memory);
    CHECK(with(host_staging::remote, near, true) == path::tcp_lanes);
    CHECK(with(host_staging::automatic, bare, false) == path::shared_memory);
    CHECK(choose_path(me,
                      bare,
                      host_staging::automatic,
                      mnnvl_mode::automatic,
                      false,
                      {.ranks = ranks, .nvlink = true}) == path::shared_memory);
    // The NVLink fabric comes first on a crowded host too.
    auto const fab = [](rank_t rank, char const* bus_id) {
      return endpoint{rank, bus_id, true, domain(7, 32766), "boot-a", true};
    };
    CHECK(automatic(fab(0, "0000:01:00.0"), fab(1, "0000:02:00.0"), {.ranks = ranks}) ==
          path::nvlink_fabric);
  }
}

TEST_CASE("a rank keeps a mirror where a same-host peer stages to it", "[exchange][path]")
{
  // always: every same-host peer stages. never and remote: no peer maps a named mirror.
  CHECK(keeps_mirror(host_staging::always, 2, {.ranks = 2, .nvlink = true}, false));
  CHECK_FALSE(keeps_mirror(host_staging::always, 1, {}, false));
  for (auto const staging : {host_staging::never, host_staging::remote}) {
    CHECK_FALSE(keeps_mirror(staging, 8, {.ranks = 8}, true));
  }
  // auto: a GPU without peer access to this one, or a crowded host of GPUs without NVLink.
  CHECK_FALSE(keeps_mirror(host_staging::automatic, 2, {.ranks = 2}, false));
  CHECK(keeps_mirror(host_staging::automatic, 2, {.ranks = 2}, true));
  CHECK(keeps_mirror(host_staging::automatic, 3, {.ranks = 3}, false));
  CHECK_FALSE(keeps_mirror(host_staging::automatic, 8, {.ranks = 8, .nvlink = true}, false));
  CHECK(keeps_mirror(host_staging::automatic, 8, {.ranks = 8, .nvlink = true}, true));
  // The crowd is this host's ranks, not the cluster's.
  CHECK_FALSE(keeps_mirror(host_staging::automatic, 8, {.ranks = 2}, false));
}

TEST_CASE("ranks that count their host differently never stage to a rank without a mirror",
          "[exchange][path]")
{
  // One machine, four GPUs with peer access: three ranks listed under one host name, the fourth
  // under another. Each rank's hello says keeps_mirror() of its own count.
  struct rank_at {
    endpoint self;
    host_view view;
  };
  auto const rank_on = [](rank_t rank, char const* bus_id, host_view view) {
    return rank_at{
      endpoint{
        rank, bus_id, false, {}, "boot-a", keeps_mirror(host_staging::automatic, 4, view, false)},
      view};
  };
  auto const route = [](rank_at const& from, rank_at const& to) {
    return choose_path(
      from.self, to.self, host_staging::automatic, mnnvl_mode::automatic, true, from.view);
  };
  auto const a    = rank_on(0, "0000:01:00.0", {.ranks = 3});
  auto const b    = rank_on(1, "0000:02:00.0", {.ranks = 3});
  auto const lone = rank_on(3, "0000:04:00.0", {.ranks = 1});
  CHECK(route(a, b) == path::shared_memory);
  CHECK(route(b, a) == path::shared_memory);
  CHECK(route(a, lone) == path::nixl);
  CHECK(route(lone, a) == path::nixl);
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
