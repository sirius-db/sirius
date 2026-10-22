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
 */
class nixl_fabric : public fabric {
 public:
  struct config {
    rank_t rank{0};
    /// "host:port" of every rank's listener, indexed by rank; its size is the world size.
    std::vector<std::string> peers;
    std::chrono::milliseconds connect_timeout{std::chrono::minutes{2}};
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

 private:
  struct peer {
    std::string host;
    int port{0};
    std::string agent;
    std::uintptr_t recv_base{0};
    std::size_t recv_bytes{0};
    std::uint32_t device{0};
    bool hello{false};
  };

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
};

}  // namespace sirius::exchange
