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

#include "exchange/exchange_runtime.hpp"

#include "exchange/exchange_service.hpp"
#include "sirius/exception.hpp"

#ifdef SIRIUS_WITH_NIXL
#include "exchange/nixl_fabric.hpp"
#endif

#include <utility>

namespace sirius::exchange {

std::shared_ptr<exchange_service> make_nixl_exchange(rank_t rank,
                                                     std::vector<std::string> peers,
                                                     cucascade::memory::memory_space& gpu,
                                                     cucascade::memory::memory_space* host,
                                                     nixl_exchange_options opts)
{
#ifdef SIRIUS_WITH_NIXL
  // Torn down service first: its transport's engine uses the fabric and both rings.
  struct bundle {
    std::unique_ptr<frame_ring> send;
    std::unique_ptr<frame_ring> recv;
    std::unique_ptr<nixl_fabric> fabric;
    std::unique_ptr<exchange_service> service;
    ~bundle()
    {
      service.reset();
      fabric.reset();
    }
  };
  auto const peer_count = static_cast<std::uint32_t>(peers.size() > 1 ? peers.size() - 1 : 1);
  auto b                = std::make_shared<bundle>();
  b->send               = std::make_unique<frame_ring>(opts.send_frames, opts.frame_bytes);
  b->recv = std::make_unique<frame_ring>(opts.recv_frames_per_peer * peer_count, opts.frame_bytes);
  nixl_fabric::config cfg{rank, std::move(peers)};
  cfg.staging = nixl_fabric::staging_from_env();
  b->fabric   = std::make_unique<nixl_fabric>(std::move(cfg), *b->send, *b->recv);
  b->service = std::make_unique<exchange_service>(*b->fabric, *b->send, *b->recv, gpu, host);
  return std::shared_ptr<exchange_service>(b, b->service.get());
#else
  static_cast<void>(rank);
  static_cast<void>(peers);
  static_cast<void>(gpu);
  static_cast<void>(host);
  static_cast<void>(opts);
  throw sirius::invalid_input_exception(
    "distributed execution needs a Sirius built with -DSIRIUS_WITH_NIXL=ON");
#endif
}

}  // namespace sirius::exchange
