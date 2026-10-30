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

#include "exchange/transport.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::exchange {

class exchange_service;

/// Ring sizing: @p send_frames staging frames shared by all peers, and @p recv_frames_per_peer
/// frames (credits) for each peer, each @p frame_bytes, a positive multiple of 256. 8 MiB frames
/// keep NIXL writes large enough to reach link bandwidth. The sizing is fixed until credits are
/// granted on demand (#2371).
struct nixl_exchange_options {
  std::size_t frame_bytes            = std::size_t{8} * 1024 * 1024;  // 8 MiB
  std::uint32_t send_frames          = 16;
  std::uint32_t recv_frames_per_peer = 8;
};

/**
 * @brief This process's exchange over NIXL: its rings, its fabric, connected to every peer, and
 * the service on top.
 *
 * Blocks until every peer is reached. The service shares ownership of the rings and the fabric,
 * so they outlive every plan that still holds it.
 *
 * @param peers "host:port" of every rank's listener, indexed by rank.
 * @throws sirius::invalid_input_exception when Sirius was built without SIRIUS_WITH_NIXL, or
 *         SIRIUS_EXCHANGE_HOST_STAGING has an unknown value.
 * @throws sirius::internal_exception when NIXL fails or a peer is not reached within two minutes.
 */
[[nodiscard]] std::shared_ptr<exchange_service> make_nixl_exchange(
  rank_t rank,
  std::vector<std::string> peers,
  cucascade::memory::memory_space& gpu,
  cucascade::memory::memory_space* host,
  nixl_exchange_options opts = {});

}  // namespace sirius::exchange
