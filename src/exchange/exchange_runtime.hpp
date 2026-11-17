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

#pragma once

#include "exchange/path_choice.hpp"
#include "exchange/transport.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::exchange {

class exchange_service;

/// The most send frames, and receive frames per peer, a rank may have.
constexpr std::uint32_t max_ring_frames = 1024;
/// The largest frame: 1 GiB.
constexpr std::size_t max_frame_bytes = std::size_t{1} << 30;
/// The most TCP lanes a rank may have.
constexpr std::uint32_t max_tcp_lanes = 64;

/// Ring sizing: @p send_frames staging frames shared by all peers, and @p recv_frames_per_peer
/// frames (credits) for each peer, each @p frame_bytes, a positive multiple of 256. A write covers
/// up to half a peer's credits of consecutive frames: NVLink between nodes needs writes of 16 MiB
/// or more (about 450 GB/s at 16 MiB, 700 at 128 MiB, 770 at 1 GiB), and 16 credits of 16 MiB
/// keep two 128 MiB writes to a peer in flight. The sizing is fixed until credits are granted on
/// demand (#2371).
struct nixl_exchange_options {
  std::size_t frame_bytes   = std::size_t{16} * 1024 * 1024;  // 16 MiB
  std::uint32_t send_frames = 32;
  /// The frames a peer may have in flight to this rank. Over TCP lanes a write is one frame and a
  /// peer keeps one frame per lane in flight, so fewer frames than lanes leave lanes idle.
  std::uint32_t recv_frames_per_peer = 16;
  host_staging staging               = host_staging::automatic;
  mnnvl_mode mnnvl                   = mnnvl_mode::automatic;
  /// TCP lanes per rank (at most max_tcp_lanes); 0 shares 16 among the ranks on this host.
  std::uint32_t tcp_lanes = 0;
  transport::options transport_options;
};

/**
 * @brief This process's exchange over NIXL: its rings, its fabric, connected to every peer, and
 * the service on top.
 *
 * Blocks until every peer is reached. The service shares ownership of the rings and the fabric,
 * so they outlive every plan that still holds it.
 *
 * @param peers "host:port" of every rank's listener, indexed by rank.
 * @throws sirius::internal_exception when NIXL fails or a peer is not reached within two minutes.
 */
[[nodiscard]] std::shared_ptr<exchange_service> make_nixl_exchange(
  rank_t rank,
  std::vector<std::string> peers,
  cucascade::memory::memory_space& gpu,
  cucascade::memory::memory_space* host,
  nixl_exchange_options opts = {});

}  // namespace sirius::exchange
