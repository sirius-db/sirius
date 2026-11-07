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

#include "exchange/path_choice.hpp"

#include "sirius/exception.hpp"

namespace sirius::exchange {

host_staging parse_host_staging(std::string_view value)
{
  if (value == "auto") { return host_staging::automatic; }
  if (value == "always") { return host_staging::always; }
  if (value == "never") { return host_staging::never; }
  if (value == "remote") { return host_staging::remote; }
  throw sirius::invalid_input_exception(
    "sirius_exchange_host_staging must be auto, always, never or remote, not '{}'", value);
}

mnnvl_mode parse_mnnvl(std::string_view value)
{
  if (value == "auto") { return mnnvl_mode::automatic; }
  if (value == "require") { return mnnvl_mode::require; }
  if (value == "off") { return mnnvl_mode::off; }
  throw sirius::invalid_input_exception(
    "sirius_exchange_mnnvl must be auto, require or off, not '{}'", value);
}

ring_memory ring_memory_for(mnnvl_mode mode) noexcept
{
  switch (mode) {
    case mnnvl_mode::automatic: return ring_memory::fabric_or_plain;
    case mnnvl_mode::require: return ring_memory::fabric;
    case mnnvl_mode::off: return ring_memory::plain;
  }
  return ring_memory::plain;
}

char const* describe(path p) noexcept
{
  switch (p) {
    case path::nixl: return "NIXL";
    case path::nvlink_fabric: return "NIXL over the NVLink fabric";
    case path::shared_memory: return "host-staged through pinned shared memory";
    case path::tcp_lanes: return "host-staged over TCP lanes";
  }
  return "?";
}

path choose_path(endpoint const& me,
                 endpoint const& peer,
                 host_staging staging,
                 mnnvl_mode mnnvl,
                 bool peer_access)
{
  bool const fabric = mnnvl != mnnvl_mode::off && me.fabric_ring && peer.fabric_ring &&
                      me.domain.same_as(peer.domain);
  if (mnnvl == mnnvl_mode::require && !fabric) {
    throw sirius::invalid_input_exception(
      "exchange: sirius_exchange_mnnvl is require, but rank {} cannot write rank {} over the "
      "NVLink fabric (fabric rings: {} and {}; NVLink domains: {} and {})",
      me.rank,
      peer.rank,
      me.fabric_ring,
      peer.fabric_ring,
      me.domain.to_string(),
      peer.domain.to_string());
  }
  bool const same_host = me.boot_id == peer.boot_id;
  if (staging == host_staging::remote) { return path::tcp_lanes; }
  // Shared memory reaches only this host.
  if (staging == host_staging::always) { return same_host ? path::shared_memory : path::tcp_lanes; }
  if (fabric) { return path::nvlink_fabric; }
  if (staging == host_staging::never) { return path::nixl; }
  if (!same_host) { return path::tcp_lanes; }
  return peer_access ? path::nixl : path::shared_memory;
}

}  // namespace sirius::exchange
