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

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class nixlAgent;
class nixlBackendH;
class nixlXferReqH;
struct CUstream_st;
struct CUevent_st;

namespace sirius::exchange {

/**
 * @brief The fabric over NIXL's UCX backend.
 *
 * Each rank runs a NIXL agent whose listener thread serves its metadata on its own port. The
 * constructor registers both rings, fetches every peer's metadata from that peer's listener, and
 * trades a hello notification with each peer giving its receive ring's address, size and device.
 * It returns once every peer has answered, so the transport starts with every peer reachable.
 *
 * There is no other control channel: everything after the hello is a NIXL write or notification.
 *
 * **Host staging.** UCX moves GPU memory between processes with `cuda_ipc`, which needs peer
 * access between the two GPUs. Without it (GPUs under different PCIe switches on many hosts) UCX
 * falls back to a far slower path. Every rank therefore also keeps a pinned host mirror of its
 * receive ring in POSIX shared memory. A write to a peer whose GPU this GPU cannot reach directly
 * is a device-to-host copy into that peer's mirror, followed by a notification naming the bytes;
 * the peer copies them host-to-device into its receive ring and only then hands the write's own
 * notification on. One PCIe crossing each way, no copy through the CPU.
 */
class nixl_fabric : public fabric {
 public:
  enum class host_staging { automatic, always, never };

  /// SIRIUS_EXCHANGE_HOST_STAGING: auto (or unset), always or never.
  /// @throws sirius::invalid_input_exception on any other value.
  [[nodiscard]] static host_staging staging_from_env();

  struct config {
    rank_t rank{0};
    /// "host:port" of every rank's listener, indexed by rank; its size is the world size.
    std::vector<std::string> peers;
    std::chrono::milliseconds connect_timeout{std::chrono::minutes{2}};
    /// automatic: host-stage writes to peers whose GPU lacks peer access to this one (NVML).
    host_staging staging{host_staging::automatic};
  };

  /// @throws sirius::invalid_input_exception on a bad config.
  /// @throws sirius::internal_exception when NIXL fails, or a peer is not reached in time.
  nixl_fabric(config cfg, frame_ring& send, frame_ring& recv);
  ~nixl_fabric() override;

  nixl_fabric(nixl_fabric const&)            = delete;
  nixl_fabric& operator=(nixl_fabric const&) = delete;

  [[nodiscard]] rank_t rank() const noexcept override { return _rank; }
  [[nodiscard]] rank_t world_size() const noexcept override
  {
    return static_cast<rank_t>(_peers.size());
  }

  [[nodiscard]] std::uint64_t write(rank_t peer,
                                    std::size_t src,
                                    std::size_t dst,
                                    std::size_t bytes,
                                    std::string notification) override;
  [[nodiscard]] write_status status(std::uint64_t id) override;
  void notify(rank_t peer, std::string notification) override;
  void poll(std::vector<fabric_notification>& out) override;

  /// Whether writes to @p peer go through its host mirror rather than NIXL.
  [[nodiscard]] bool host_staged(rank_t peer) const { return _peers.at(peer).staged; }

 private:
  struct peer {
    std::string host;
    int port{0};
    std::string agent;
    std::uintptr_t recv_base{0};
    std::size_t recv_bytes{0};
    std::uint32_t device{0};
    std::uint32_t pid{0};
    std::string pci_bus_id;
    bool hello{false};
    /// Writes go through the peer's receive mirror, mapped here at mirror.
    bool staged{false};
    std::byte* mirror{nullptr};
  };

  /// A host-staged write: its device-to-host copy, then the notification that says it landed.
  struct staged_write {
    CUevent_st* copied;
    rank_t peer;
    std::string notification;
  };

  /// A peer's host-staged write being copied into the receive ring; its notification waits.
  struct landing {
    CUevent_st* copied;
    rank_t from;
    std::string notification;
  };

  /// Maps rank @p r's receive mirror and registers it with CUDA.
  void map_mirror(rank_t r);
  CUevent_st* take_event();

  /// Receives notifications into _backlog, taking hellos out.
  void receive();

  rank_t _rank;
  frame_ring& _send;
  frame_ring& _recv;
  int _device{0};
  std::vector<peer> _peers;
  std::unordered_map<std::string, rank_t> _rank_of;
  std::unique_ptr<nixlAgent> _agent;
  nixlBackendH* _backend{nullptr};
  std::map<std::uint64_t, nixlXferReqH*> _writes;
  std::uint64_t _next_write{1};
  std::vector<fabric_notification> _backlog;

  /// This rank's receive mirror (null with host_staging::never) and its shared-memory name.
  std::byte* _mirror{nullptr};
  std::string _mirror_name;
  CUstream_st* _d2h{nullptr};
  CUstream_st* _h2d{nullptr};
  std::vector<CUevent_st*> _free_events;
  std::map<std::uint64_t, staged_write> _staged;
  std::vector<landing> _landings;
};

}  // namespace sirius::exchange
