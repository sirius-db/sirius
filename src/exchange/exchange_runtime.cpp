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
#include "exchange/nixl_fabric.hpp"
#include "log/logging.hpp"
#include "sirius/exception.hpp"

#include <rmm/cuda_device.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <utility>

namespace sirius::exchange {

namespace {

/// The probe's bar: writes over the network rails top out near 100 GB/s.
constexpr double nvlink_probe_min_gbps = 150;

/**
 * @brief Times one write, as large as the transport's largest, to each peer on the NVLink
 * fabric path. UCX takes another transport without a word when `cuda_ipc` cannot map the peer's
 * memory; this says so at connect instead of in a slow query.
 *
 * Runs before the transport starts: each write lands in this rank's own credit frames on the
 * peer, which no other rank writes, and carries no notification.
 * @throws sirius::internal_exception on a slow peer when opts.mnnvl is require.
 */
void probe_nvlink_peers(nixl_fabric& fab, nixl_exchange_options const& opts)
{
  auto const frames =
    std::max<std::uint32_t>(1, std::min(opts.recv_frames_per_peer, opts.send_frames) / 2);
  auto const bytes = std::size_t{frames} * opts.frame_bytes;
  for (rank_t peer = 0; peer < fab.world_size(); ++peer) {
    if (peer == fab.rank() || fab.path_to(peer) != path::nvlink_fabric) { continue; }
    auto const slot = fab.rank() < peer ? fab.rank() : fab.rank() - 1;
    auto const dst  = std::size_t{slot} * opts.recv_frames_per_peer * opts.frame_bytes;
    // The first write also maps the peer's memory.
    static_cast<void>(fab.timed_write(peer, dst, bytes));
    auto const seconds = fab.timed_write(peer, dst, bytes);
    auto const gbps    = static_cast<double>(bytes) / seconds / 1e9;
    if (gbps >= nvlink_probe_min_gbps) {
      SIRIUS_LOG_INFO("exchange: probe to rank {}: {} MiB in {:.1f} us, {:.0f} GB/s",
                      peer,
                      bytes >> 20,
                      seconds * 1e6,
                      gbps);
      continue;
    }
    if (opts.mnnvl == mnnvl_mode::require) {
      throw sirius::internal_exception(
        "exchange: a write to rank {} over the NVLink fabric ran at {:.0f} GB/s; UCX did not "
        "take cuda_ipc (check UCX_TLS and UCX_CUDA_IPC_ENABLE_MNNVL)",
        peer,
        gbps);
    }
    SIRIUS_LOG_WARN(
      "exchange: a write to rank {} over the NVLink fabric ran at {:.0f} GB/s; UCX did not take "
      "cuda_ipc (check UCX_TLS and UCX_CUDA_IPC_ENABLE_MNNVL)",
      peer,
      gbps);
  }
}

}  // namespace

std::shared_ptr<exchange_service> make_nixl_exchange(rank_t rank,
                                                     std::vector<std::string> peers,
                                                     cucascade::memory::memory_space& gpu,
                                                     cucascade::memory::memory_space* host,
                                                     nixl_exchange_options opts)
{
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
  // The rings, the fabric's registrations and streams, and the transport's streams and engine
  // thread all take the current device.
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{gpu.get_device_id()}};
  // A one-rank world has no peers; still give the receive ring frames, since frame_ring needs one.
  auto const peer_count = static_cast<std::uint32_t>(peers.size() > 1 ? peers.size() - 1 : 1);
  auto b                = std::make_shared<bundle>();
  auto const memory     = ring_memory_for(opts.mnnvl);
  b->send               = std::make_unique<frame_ring>(opts.send_frames, opts.frame_bytes, memory);
  b->recv =
    std::make_unique<frame_ring>(opts.recv_frames_per_peer * peer_count, opts.frame_bytes, memory);
  if (opts.mnnvl != mnnvl_mode::off && !b->recv->in_fabric_memory()) {
    SIRIUS_LOG_WARN(
      "exchange: rank {} has no fabric memory ({}); its rings use cudaMalloc, and peers on other "
      "nodes reach it over the network",
      rank,
      b->recv->why_not_fabric());
  }
  nixl_fabric::config cfg{rank, std::move(peers)};
  cfg.staging = opts.staging;
  cfg.mnnvl   = opts.mnnvl;
  cfg.lanes   = opts.tcp_lanes;
  b->fabric   = std::make_unique<nixl_fabric>(std::move(cfg), *b->send, *b->recv);
  probe_nvlink_peers(*b->fabric, opts);
  b->service = std::make_unique<exchange_service>(
    *b->fabric, *b->send, *b->recv, gpu, host, std::move(opts.transport_options));
  return std::shared_ptr<exchange_service>(b, b->service.get());
}

}  // namespace sirius::exchange
