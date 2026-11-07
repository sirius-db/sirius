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

#include "exchange/fabric.hpp"
#include "exchange/fabric_memory.hpp"
#include "exchange/host_staging.hpp"

#include <string>
#include <string_view>

namespace sirius::exchange {

/// "auto", "always", "never" or "remote", as the sirius_exchange_host_staging setting takes it.
/// @throws sirius::invalid_input_exception on any other value.
[[nodiscard]] host_staging parse_host_staging(std::string_view value);

/// Whether the rings go in fabric memory, so that peers on other nodes of this NVLink domain
/// write them over NVLink (MNNVL). automatic: where the GPU and driver allow it; require: every
/// peer must be reached that way; off: cudaMalloc rings, as before.
enum class mnnvl_mode { automatic, require, off };

/// "auto", "require" or "off", as the sirius_exchange_mnnvl setting takes it.
/// @throws sirius::invalid_input_exception on any other value.
[[nodiscard]] mnnvl_mode parse_mnnvl(std::string_view value);

/// The ring memory @p mode asks for.
[[nodiscard]] ring_memory ring_memory_for(mnnvl_mode mode) noexcept;

/// How writes reach a peer. nixl and nvlink_fabric are the same NIXL write; nvlink_fabric goes
/// into fabric memory, which also reaches GPUs on other nodes of the NVLink domain. shared_memory
/// and tcp_lanes go through the peer's pinned host mirror: mapped here, or written host to host
/// over TCP lanes.
enum class path { nixl, nvlink_fabric, shared_memory, tcp_lanes };

[[nodiscard]] char const* describe(path p) noexcept;

/// What a rank tells its peers about itself in its hello, as far as the path depends on it.
struct endpoint {
  rank_t rank{0};
  std::string pci_bus_id;
  bool fabric_ring{false};
  nvlink_domain domain;
  /// The host's kernel boot id: two ranks are on one host when theirs are equal.
  std::string boot_id;
};

/**
 * @brief The path from @p me to @p peer. Both ranks see the same two endpoints, so they agree.
 *
 * remote → tcp_lanes. always → shared_memory on this host, tcp_lanes to another. Otherwise a peer
 * whose GPU is in this GPU's NVLink domain, with both receive rings in fabric memory, gets
 * nvlink_fabric, on this host or another. Otherwise: never → nixl; another host → tcp_lanes; no
 * peer access (@p peer_access, from NVML, which knows this host's GPUs only) → shared_memory;
 * else nixl.
 *
 * @throws sirius::invalid_input_exception when @p mnnvl is require and the peer cannot be
 *         reached over the NVLink fabric.
 */
[[nodiscard]] path choose_path(endpoint const& me,
                               endpoint const& peer,
                               host_staging staging,
                               mnnvl_mode mnnvl,
                               bool peer_access);

}  // namespace sirius::exchange
