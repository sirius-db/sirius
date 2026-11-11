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

#include "exchange/exchange_runtime.hpp"
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
 * Writes come from a second agent, the writer, which has no listener: each rank sends its main
 * agent's metadata to every peer after its hello, and the peer's writer loads it.
 *
 * **Host staging.** UCX moves GPU memory between processes with `cuda_ipc`, which needs peer
 * access between the two GPUs. Without it (GPUs under different PCIe switches on many hosts) UCX
 * falls back to a far slower path. A rank therefore also keeps a pinned host mirror of its
 * receive ring in POSIX shared memory, when NVML reports a GPU on its host without peer access to
 * its own (or staging is always). A write to a peer whose GPU this GPU cannot reach directly
 * is a device-to-host copy into that peer's mirror, followed by a notification naming the bytes;
 * the peer copies them host-to-device into its receive ring and only then hands the write's own
 * notification on. One PCIe crossing each way, no copy through the CPU.
 *
 * **NVLink fabric (MNNVL).** `cuda_ipc` also reaches a GPU on another node of the same NVLink
 * domain, but only for memory with a fabric handle. The hello therefore carries whether the
 * receive ring is in fabric memory and the GPU's NVLink domain (NVML fabric info); a peer in this
 * GPU's domain, both rings in fabric memory, is written over NVLink (choose_path()).
 *
 * **TCP lanes.** UCX's TCP transport moves host memory directly but emulates a put of GPU
 * memory, about a hundred times slower, and one TCP connection is bound by the core that copies
 * its socket. Writes to a peer on another host (another kernel boot id, in the hello) that the
 * NVLink fabric does not reach therefore go host to host over lanes: extra NIXL agents, each with
 * its own UCX worker, connection and thread pinned to this GPU's CPUs, whose metadata travels in
 * notifications of the main agent. A lane initiates its own writes, into the peer's lane of the
 * same number, and not through the writer. Each lane owns pinned send slots of a frame each, so a
 * write to such a peer is at most a frame (max_write_bytes()). A write is a device-to-host copy
 * into a free slot, after which the send-ring bytes are free again; the lane then writes the slot
 * into the peer's receive mirror, and the peer lands it as a host-staged write. A rank with such
 * a peer makes its receive mirror when it first chooses that path.
 */
class nixl_fabric : public fabric {
 public:
  using host_staging = exchange::host_staging;

  struct config {
    rank_t rank{0};
    /// "host:port" of every rank's listener, indexed by rank; its size is the world size.
    std::vector<std::string> peers;
    std::chrono::milliseconds connect_timeout{std::chrono::minutes{2}};
    /// automatic: TCP lanes to peers on other hosts; on this host, host-stage writes to peers
    /// whose GPU lacks peer access to this one (NVML). always: host-stage on this host too.
    host_staging staging{host_staging::automatic};
    /// require: fail unless every peer is reached over the NVLink fabric.
    mnnvl_mode mnnvl{mnnvl_mode::automatic};
    /// TCP lanes per rank, at most max_tcp_lanes; 0 shares 16 among the ranks whose listener host
    /// is this one's (at least 2 each).
    std::uint32_t lanes{0};
    /// Pinned send slots of a frame each, per lane.
    std::uint32_t slots_per_lane{2};
  };

  /// Blocks until every peer is reached or @p cfg.connect_timeout passes. @p send and @p recv must
  /// be allocated on the current device and outlive the fabric.
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
  /// A frame to a peer over TCP lanes, whose send slots hold one; unbounded otherwise.
  [[nodiscard]] std::size_t max_write_bytes(rank_t peer) const override;
  [[nodiscard]] write_status status(std::uint64_t id) override;
  void notify(rank_t peer, std::string notification) override;
  void poll(std::vector<fabric_notification>& out) override;

  /// Whether writes to @p peer go through its host mirror (mapped here, or over TCP lanes)
  /// rather than straight into its receive ring.
  [[nodiscard]] bool host_staged(rank_t peer) const
  {
    auto const route = _peers.at(peer).route;
    return route == path::shared_memory || route == path::tcp_lanes;
  }
  /// How writes reach @p peer.
  [[nodiscard]] path path_to(rank_t peer) const { return _peers.at(peer).route; }
  /// The TCP lanes to @p peer (0 unless its path is tcp_lanes).
  [[nodiscard]] std::uint32_t lanes_to(rank_t peer) const { return _peers.at(peer).lanes; }

  /// Writes @p bytes from the start of the send ring to offset @p dst of @p peer's receive ring,
  /// with no notification, and waits until it is done. Not for a host-staged peer, nor while a
  /// transport uses the fabric.
  /// @return the seconds the write took.
  /// @throws sirius::internal_exception when the write fails or takes more than ten seconds.
  [[nodiscard]] double timed_write(rank_t peer, std::size_t dst, std::size_t bytes);

 private:
  struct lane;

  struct peer {
    std::string host;
    std::uint16_t port{0};
    std::string agent;
    std::uintptr_t recv_base{0};
    std::size_t recv_bytes{0};
    std::uint32_t device{0};
    std::uint32_t pid{0};
    std::string pci_bus_id;
    /// The writer has loaded the peer's main agent's metadata.
    bool writer_ready{false};
    bool fabric_ring{false};
    nvlink_domain domain;
    std::string boot_id;
    bool hello{false};
    path route{path::nixl};
    /// shared_memory: the peer's receive mirror, mapped here.
    std::byte* mirror{nullptr};
    /// tcp_lanes: the lanes both ranks have, the peer's lanes whose metadata is loaded, and its
    /// receive mirror's address in its process.
    std::uint32_t lanes{0};
    std::uint32_t lanes_loaded{0};
    std::uintptr_t mirror_base{0};
    std::uint32_t next_lane{0};
  };

  /// A write over TCP lanes: waits for a free send slot, then for its device-to-host copy.
  struct lane_write {
    rank_t peer;
    std::size_t src;
    std::size_t dst;
    std::size_t bytes;
    std::string notification;
    std::uint32_t lane{0};
    std::uint32_t slot{0};
    CUevent_st* copied{nullptr};
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
  /// Frees what the constructor acquired so far: for the destructor, and a constructor that
  /// throws.
  void release() noexcept;

  /// Creates this rank's receive mirror, near this GPU.
  void make_mirror();
  /// Picks @p r's path once its hello is in (choose_path()); for TCP lanes, creates this rank's
  /// mirror and lanes.
  void choose_route(rank_t r);
  /// Creates this rank's lanes (their agents; threads start with start_lanes).
  void make_lanes();
  void start_lanes();
  void stop_lanes() noexcept;
  /// Starts @p w's device-to-host copy if one of its peer's lanes has a free slot.
  bool start_lane_write(lane_write& w);
  /// Throws if a lane failed.
  void check_lanes() const;

  /// Receives the main agent's notifications; see take().
  void receive();
  /// Handles one notification from rank @p from's agent @p sender (lane notifications come here
  /// too): takes hellos and agent metadata out, starts each host-staged write's copy into the
  /// receive ring (its notification waits in _landings), and queues the rest in _backlog.
  void take(rank_t from, std::string const& sender, std::string body);

  rank_t _rank;
  frame_ring& _send;
  frame_ring& _recv;
  int _device{0};
  std::vector<peer> _peers;
  std::unordered_map<std::string, rank_t> _rank_of;
  /// Registers the receive ring, serves its metadata on the listen port, and carries every
  /// notification; never initiates a write.
  std::unique_ptr<nixlAgent> _agent;
  nixlBackendH* _backend{nullptr};
  /// Registers the send ring and initiates every write, into the peers' main agents. UCX 1.21
  /// crashes when a worker puts to an endpoint a peer's puts already wired up
  /// (ucp_worker_add_rkey_config reads a NULL lanes_distance with RC lanes); with writes only
  /// from this agent, every endpoint carries puts one way.
  std::unique_ptr<nixlAgent> _writer;
  nixlBackendH* _writer_backend{nullptr};
  std::map<std::uint64_t, nixlXferReqH*> _writes;
  std::uint64_t _next_write{1};
  std::vector<fabric_notification> _backlog;

  /// This rank's receive mirror (null with host_staging::never, or when no peer could stage or
  /// write over lanes to it) and its shared-memory name.
  std::byte* _mirror{nullptr};
  std::string _mirror_name;
  CUstream_st* _d2h{nullptr};
  CUstream_st* _h2d{nullptr};
  std::vector<CUevent_st*> _free_events;
  std::map<std::uint64_t, staged_write> _staged;
  std::vector<landing> _landings;

  host_staging _staging{host_staging::automatic};
  mnnvl_mode _mnnvl{mnnvl_mode::automatic};
  /// CPUs near this GPU (empty when NVML cannot tell): lane threads and pinned memory use them.
  std::vector<int> _near_cpus;
  std::uint32_t _lane_count{0};
  std::uint32_t _slots_per_lane{2};
  std::vector<std::unique_ptr<lane>> _lanes;
  std::map<std::uint64_t, lane_write> _lane_writes;
};

}  // namespace sirius::exchange
