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

#include "exchange/nixl_fabric.hpp"

#include "log/logging.hpp"
#include "sirius/exception.hpp"

#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <nixl.h>
#include <nvml.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <bit>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string_view>
#include <thread>

namespace sirius::exchange {
namespace {

static_assert(std::endian::native == std::endian::little, "the hello is little-endian");

// A notification's first byte names its kind; the transport's kinds are 1 to 4.
constexpr char hello_kind = '\x7f';
// A host-staged write landed in the receiver's mirror: kind | u64 offset | u64 bytes | the
// write's own notification.
constexpr char staged_kind = '\x7e';

// kind | u32 rank | u32 device | u64 receive ring base | u64 receive ring bytes | u32 pid |
// PCI bus id of the GPU, NUL-padded (names the receive mirror and decides host staging)
struct hello {
  std::uint32_t rank;
  std::uint32_t device;
  std::uint64_t base;
  std::uint64_t bytes;
  std::uint32_t pid;
  char pci_bus_id[20];
};

std::string mirror_name(std::uint32_t pid, rank_t rank)
{
  return "/sirius-exchange-" + std::to_string(pid) + "-" + std::to_string(rank);
}

/// Whether the GPUs at PCI bus ids @p a and @p b can reach each other's memory directly, by
/// NVML's peer-to-peer read and write status (NVML sees every GPU, whatever CUDA_VISIBLE_DEVICES
/// hides). True when NVML cannot tell, which keeps the NIXL path.
bool peer_access(std::string const& a, std::string const& b)
{
  if (a == b) { return true; }
  static bool const nvml_up = nvmlInit_v2() == NVML_SUCCESS;
  nvmlDevice_t da{};
  nvmlDevice_t db{};
  if (!nvml_up || nvmlDeviceGetHandleByPciBusId_v2(a.c_str(), &da) != NVML_SUCCESS ||
      nvmlDeviceGetHandleByPciBusId_v2(b.c_str(), &db) != NVML_SUCCESS) {
    SIRIUS_LOG_WARN("exchange: NVML cannot compare GPUs {} and {}; assuming peer access", a, b);
    return true;
  }
  for (auto const index : {NVML_P2P_CAPS_INDEX_READ, NVML_P2P_CAPS_INDEX_WRITE}) {
    nvmlGpuP2PStatus_t status{};
    if (nvmlDeviceGetP2PStatus(da, db, index, &status) != NVML_SUCCESS ||
        status != NVML_P2P_STATUS_OK) {
      return false;
    }
  }
  return true;
}

/// Maps @p bytes of the shared-memory object @p name, creating it when @p create.
std::byte* map_shared(std::string const& name, std::size_t bytes, bool create)
{
  int const fd = ::shm_open(name.c_str(), create ? (O_CREAT | O_RDWR | O_TRUNC) : O_RDWR, 0600);
  if (fd < 0) {
    throw sirius::internal_exception(
      "exchange: shm_open({}) failed: {}", name, std::strerror(errno));
  }
  if (create && ::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    ::close(fd);
    throw sirius::internal_exception(
      "exchange: sizing {} to {} bytes failed: {}", name, bytes, std::strerror(errno));
  }
  void* at = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (at == MAP_FAILED) {
    throw sirius::internal_exception(
      "exchange: mapping {} ({} bytes) failed: {}", name, bytes, std::strerror(errno));
  }
  RMM_CUDA_TRY(cudaHostRegister(at, bytes, cudaHostRegisterPortable));
  return static_cast<std::byte*>(at);
}

void unmap_shared(std::byte* at, std::size_t bytes)
{
  if (at == nullptr) { return; }
  static_cast<void>(cudaHostUnregister(at));
  ::munmap(at, bytes);
}

std::string encode_hello(hello const& h)
{
  std::string out(1, hello_kind);
  out.append(reinterpret_cast<char const*>(&h), sizeof(h));
  return out;
}

void check(nixl_status_t status, char const* what)
{
  if (status != NIXL_SUCCESS) {
    throw sirius::internal_exception(
      "exchange NIXL: {} failed: {}", what, nixlEnumStrings::statusStr(status));
  }
}

/// NIXL's listener does not report a port it failed to bind: peers would then wait out the
/// connect timeout. Bind it first to fail now, naming the port.
void require_free_port(int port)
{
  int const fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) { throw sirius::internal_exception("exchange NIXL: socket() failed"); }
  int const on = 1;
  static_cast<void>(::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)));
  sockaddr_in at{};
  at.sin_family      = AF_INET;
  at.sin_addr.s_addr = htonl(INADDR_ANY);
  at.sin_port        = htons(static_cast<std::uint16_t>(port));
  bool const bound   = ::bind(fd, reinterpret_cast<sockaddr*>(&at), sizeof(at)) == 0;
  ::close(fd);
  if (!bound) {
    throw sirius::invalid_input_exception("exchange NIXL: listen port {} is in use", port);
  }
}

std::string agent_name(rank_t rank) { return "sirius-exchange-" + std::to_string(rank); }

nixl_reg_dlist_t ring_descs(frame_ring const& ring, int device)
{
  nixl_reg_dlist_t descs(VRAM_SEG);
  descs.addDesc(nixlBlobDesc(reinterpret_cast<std::uintptr_t>(ring.base()),
                             ring.bytes(),
                             static_cast<std::uint64_t>(device),
                             ""));
  return descs;
}

}  // namespace

nixl_fabric::nixl_fabric(config cfg, frame_ring& send, frame_ring& recv)
  : _rank(cfg.rank), _send(send), _recv(recv)
{
  if (cfg.peers.empty() || cfg.rank >= cfg.peers.size()) {
    throw sirius::invalid_input_exception(
      "exchange NIXL: rank {} of {} peers", cfg.rank, cfg.peers.size());
  }
  RMM_CUDA_TRY(cudaGetDevice(&_device));
  _peers.resize(cfg.peers.size());
  for (rank_t r = 0; r < _peers.size(); ++r) {
    auto const& address = cfg.peers[r];
    auto const colon    = address.rfind(':');
    auto& p             = _peers[r];
    if (colon == std::string::npos ||
        std::from_chars(address.data() + colon + 1, address.data() + address.size(), p.port).ec !=
          std::errc{}) {
      throw sirius::invalid_input_exception("exchange NIXL: peer address '{}' is not host:port",
                                            address);
    }
    p.host            = address.substr(0, colon);
    p.agent           = agent_name(r);
    _rank_of[p.agent] = r;
  }

  RMM_CUDA_TRY(cudaStreamCreateWithFlags(&_d2h, cudaStreamNonBlocking));
  RMM_CUDA_TRY(cudaStreamCreateWithFlags(&_h2d, cudaStreamNonBlocking));
  char bus_id[20] = {};
  RMM_CUDA_TRY(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), _device));
  _peers[_rank].pci_bus_id = bus_id;
  _peers[_rank].pid        = static_cast<std::uint32_t>(::getpid());
  // The mirror exists before the hello names it, so a peer that hears the hello can map it.
  if (cfg.staging != host_staging::never && _peers.size() > 1) {
    _mirror_name = mirror_name(_peers[_rank].pid, _rank);
    _mirror      = map_shared(_mirror_name, _recv.bytes(), true);
  }

  require_free_port(_peers[_rank].port);
  // UCX parses its global configuration unsafely: two backends created at once in one process
  // can read a garbled setting and fail.
  static std::mutex creating;
  std::unique_lock create_lock{creating};
  nixlAgentConfig agent_cfg(/*use_prog_thread=*/false,
                            /*use_listen_thread=*/true,
                            static_cast<std::uint16_t>(_peers[_rank].port),
                            nixl_thread_sync_t::NIXL_THREAD_SYNC_RW);
  _agent = std::make_unique<nixlAgent>(agent_name(_rank), agent_cfg);
  nixl_mem_list_t mems;
  nixl_b_params_t params;
  check(_agent->getPluginParams("UCX", mems, params), "getPluginParams(UCX)");
  check(_agent->createBackend("UCX", params, _backend), "createBackend(UCX)");
  create_lock.unlock();
  check(_agent->registerMem(ring_descs(_send, _device)), "registerMem(send ring)");
  check(_agent->registerMem(ring_descs(_recv, _device)), "registerMem(receive ring)");

  // Fetch every peer's metadata from its listener and send it a hello. A peer that is not up
  // yet refuses the fetch, so fetches repeat until the deadline. The listener serves metadata
  // from the agent's start, before its rings are registered, so a peer is ready only once the
  // metadata loaded here covers the receive ring its hello names; until then it is fetched again.
  auto const deadline = std::chrono::steady_clock::now() + cfg.connect_timeout;
  hello mine{_rank,
             static_cast<std::uint32_t>(_device),
             reinterpret_cast<std::uint64_t>(_recv.base()),
             _recv.bytes(),
             _peers[_rank].pid,
             {}};
  std::memcpy(mine.pci_bus_id, bus_id, sizeof(mine.pci_bus_id));
  auto const me = encode_hello(mine);
  std::vector<bool> greeted(_peers.size(), false);
  std::vector<bool> ready(_peers.size(), false);
  ready[_rank]    = true;
  auto last_fetch = std::chrono::steady_clock::time_point{};
  while (true) {
    auto const now   = std::chrono::steady_clock::now();
    bool const fetch = now - last_fetch > std::chrono::milliseconds{200};
    if (fetch) { last_fetch = now; }
    receive();
    std::size_t waiting = 0;
    for (rank_t r = 0; r < _peers.size(); ++r) {
      if (ready[r]) { continue; }
      auto& p = _peers[r];
      bool const known =
        _agent->checkRemoteMD(p.agent, nixl_xfer_dlist_t(VRAM_SEG)) == NIXL_SUCCESS;
      if (known && !greeted[r]) {
        check(_agent->makeConnection(p.agent), "makeConnection");
        check(_agent->genNotif(p.agent, me), "genNotif(hello)");
        greeted[r] = true;
      }
      if (known && p.hello) {
        nixl_xfer_dlist_t ring(VRAM_SEG);
        ring.addDesc(nixlBasicDesc(p.recv_base, p.recv_bytes, p.device));
        if (_agent->checkRemoteMD(p.agent, ring) == NIXL_SUCCESS) {
          ready[r] = true;
          continue;
        }
        if (fetch) { static_cast<void>(_agent->invalidateRemoteMD(p.agent)); }
      }
      if (fetch && (!known || p.hello)) {
        nixl_opt_args_t at;
        at.ipAddr = p.host;
        at.port   = p.port;
        static_cast<void>(_agent->fetchRemoteMD(p.agent, &at));
      }
      ++waiting;
    }
    if (waiting == 0) { break; }
    if (now > deadline) {
      throw sirius::internal_exception("exchange NIXL: rank {} reached {} of {} peers in {} ms",
                                       _rank,
                                       _peers.size() - 1 - waiting,
                                       _peers.size() - 1,
                                       cfg.connect_timeout.count());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }

  // Every peer's hello is in: choose each peer's path.
  for (rank_t r = 0; r < _peers.size(); ++r) {
    if (r == _rank) { continue; }
    auto& p = _peers[r];
    p.staged =
      cfg.staging == host_staging::always || (cfg.staging == host_staging::automatic &&
                                              !peer_access(_peers[_rank].pci_bus_id, p.pci_bus_id));
    if (p.staged) { map_mirror(r); }
    SIRIUS_LOG_INFO("exchange: rank {} (GPU {}) -> rank {} (GPU {}): {}",
                    _rank,
                    _peers[_rank].pci_bus_id,
                    r,
                    p.pci_bus_id,
                    p.staged ? "host-staged through pinned shared memory" : "NIXL");
  }
}

nixl_fabric::host_staging nixl_fabric::staging_from_env()
{
  char const* value = std::getenv("SIRIUS_EXCHANGE_HOST_STAGING");
  std::string_view const v{value != nullptr ? value : ""};
  if (v.empty() || v == "auto") { return host_staging::automatic; }
  if (v == "always") { return host_staging::always; }
  if (v == "never") { return host_staging::never; }
  throw sirius::invalid_input_exception(
    "SIRIUS_EXCHANGE_HOST_STAGING must be auto, always or never, not '{}'", v);
}

void nixl_fabric::map_mirror(rank_t r)
{
  auto& p = _peers[r];
  if (p.pid == 0) {
    throw sirius::internal_exception("exchange: rank {} has no receive mirror to stage into", r);
  }
  p.mirror = map_shared(mirror_name(p.pid, r), p.recv_bytes, false);
}

CUevent_st* nixl_fabric::take_event()
{
  if (!_free_events.empty()) {
    auto* e = _free_events.back();
    _free_events.pop_back();
    return e;
  }
  cudaEvent_t e{};
  RMM_CUDA_TRY(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  return e;
}

nixl_fabric::~nixl_fabric()
{
  for (auto& [id, req] : _writes) {
    static_cast<void>(_agent->releaseXferReq(req));
  }
  static_cast<void>(_agent->deregisterMem(ring_descs(_send, _device)));
  static_cast<void>(_agent->deregisterMem(ring_descs(_recv, _device)));
  // Copies still in flight read or write the mirrors: let them finish first.
  if (_d2h != nullptr) { static_cast<void>(cudaStreamSynchronize(_d2h)); }
  if (_h2d != nullptr) { static_cast<void>(cudaStreamSynchronize(_h2d)); }
  for (auto& [id, w] : _staged) {
    _free_events.push_back(w.copied);
  }
  for (auto& l : _landings) {
    _free_events.push_back(l.copied);
  }
  for (auto* e : _free_events) {
    static_cast<void>(cudaEventDestroy(e));
  }
  for (auto& p : _peers) {
    unmap_shared(p.mirror, p.recv_bytes);
  }
  unmap_shared(_mirror, _recv.bytes());
  if (!_mirror_name.empty()) { ::shm_unlink(_mirror_name.c_str()); }
  if (_d2h != nullptr) { static_cast<void>(cudaStreamDestroy(_d2h)); }
  if (_h2d != nullptr) { static_cast<void>(cudaStreamDestroy(_h2d)); }
}

void nixl_fabric::receive()
{
  nixl_notifs_t notifs;
  check(_agent->getNotifs(notifs), "getNotifs");
  for (auto& [agent, bodies] : notifs) {
    auto const it = _rank_of.find(agent);
    if (it == _rank_of.end()) {
      throw sirius::internal_exception("exchange NIXL: a notification from unknown agent '{}'",
                                       agent);
    }
    for (auto& body : bodies) {
      if (!body.empty() && body.front() == hello_kind) {
        hello h{};
        if (body.size() != 1 + sizeof(h)) {
          throw sirius::invalid_input_exception("exchange NIXL: a malformed hello from {}", agent);
        }
        std::memcpy(&h, body.data() + 1, sizeof(h));
        auto& p = _peers[it->second];
        if (h.rank != it->second || h.bytes != _recv.bytes()) {
          throw sirius::invalid_input_exception(
            "exchange NIXL: {} says it is rank {} with a {}-byte receive ring; expected rank {} "
            "with {} bytes",
            agent,
            h.rank,
            h.bytes,
            it->second,
            _recv.bytes());
        }
        p.recv_base  = h.base;
        p.recv_bytes = h.bytes;
        p.device     = h.device;
        p.pid        = h.pid;
        p.pci_bus_id.assign(h.pci_bus_id, strnlen(h.pci_bus_id, sizeof(h.pci_bus_id)));
        p.hello = true;
        continue;
      }
      if (!body.empty() && body.front() == staged_kind) {
        std::uint64_t at    = 0;
        std::uint64_t bytes = 0;
        if (_mirror == nullptr || body.size() < 1 + 2 * sizeof(std::uint64_t)) {
          throw sirius::invalid_input_exception(
            "exchange NIXL: a host-staged write from {} that this rank cannot take", agent);
        }
        std::memcpy(&at, body.data() + 1, sizeof(at));
        std::memcpy(&bytes, body.data() + 1 + sizeof(at), sizeof(bytes));
        if (at + bytes > _recv.bytes()) {
          throw sirius::invalid_input_exception(
            "exchange NIXL: a host-staged {}-byte write at {} from {} is outside the ring",
            bytes,
            at,
            agent);
        }
        RMM_CUDA_TRY(
          cudaMemcpyAsync(_recv.base() + at, _mirror + at, bytes, cudaMemcpyHostToDevice, _h2d));
        auto* copied = take_event();
        RMM_CUDA_TRY(cudaEventRecord(copied, _h2d));
        _landings.push_back({copied, it->second, body.substr(1 + 2 * sizeof(std::uint64_t))});
        continue;
      }
      _backlog.push_back({it->second, std::move(body)});
    }
  }
}

std::uint64_t nixl_fabric::write(
  rank_t peer, std::size_t src, std::size_t dst, std::size_t bytes, std::string notification)
{
  auto const& p = _peers.at(peer);
  if (src + bytes > _send.bytes() || dst + bytes > p.recv_bytes) {
    throw sirius::invalid_input_exception(
      "exchange NIXL: a {}-byte write from {} to {} is outside the rings", bytes, src, dst);
  }
  if (p.staged) {
    RMM_CUDA_TRY(
      cudaMemcpyAsync(p.mirror + dst, _send.base() + src, bytes, cudaMemcpyDeviceToHost, _d2h));
    auto* copied = take_event();
    RMM_CUDA_TRY(cudaEventRecord(copied, _d2h));
    std::string landed(1, staged_kind);
    auto const at = static_cast<std::uint64_t>(dst);
    auto const n  = static_cast<std::uint64_t>(bytes);
    landed.append(reinterpret_cast<char const*>(&at), sizeof(at));
    landed.append(reinterpret_cast<char const*>(&n), sizeof(n));
    landed += notification;
    _staged.emplace(_next_write, staged_write{copied, peer, std::move(landed)});
    return _next_write++;
  }
  nixl_xfer_dlist_t local(VRAM_SEG);
  local.addDesc(nixlBasicDesc(reinterpret_cast<std::uintptr_t>(_send.base()) + src,
                              bytes,
                              static_cast<std::uint64_t>(_device)));
  nixl_xfer_dlist_t remote(VRAM_SEG);
  remote.addDesc(nixlBasicDesc(p.recv_base + dst, bytes, p.device));
  nixl_opt_args_t extra;
  extra.backends.push_back(_backend);
  extra.notif       = std::move(notification);
  nixlXferReqH* req = nullptr;
  check(_agent->createXferReq(NIXL_WRITE, local, remote, p.agent, req, &extra), "createXferReq");
  auto const posted = _agent->postXferReq(req);
  if (posted != NIXL_SUCCESS && posted != NIXL_IN_PROG) {
    static_cast<void>(_agent->releaseXferReq(req));
    check(posted, "postXferReq");
  }
  _writes.emplace(_next_write, req);
  return _next_write++;
}

fabric::write_status nixl_fabric::status(std::uint64_t id)
{
  if (auto const staged = _staged.find(id); staged != _staged.end()) {
    auto const copied = cudaEventQuery(staged->second.copied);
    if (copied == cudaErrorNotReady) { return write_status::pending; }
    auto w = std::move(staged->second);
    _staged.erase(staged);
    _free_events.push_back(w.copied);
    if (copied != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      return write_status::failed;
    }
    // The bytes are in the peer's mirror: tell it, so it copies them into its receive ring.
    notify(w.peer, std::move(w.notification));
    return write_status::done;
  }
  auto const it = _writes.find(id);
  if (it == _writes.end()) {
    throw sirius::invalid_input_exception("exchange NIXL: unknown write {}", id);
  }
  auto const s = _agent->getXferStatus(it->second);
  if (s == NIXL_IN_PROG) { return write_status::pending; }
  static_cast<void>(_agent->releaseXferReq(it->second));
  _writes.erase(it);
  return s == NIXL_SUCCESS ? write_status::done : write_status::failed;
}

void nixl_fabric::notify(rank_t peer, std::string notification)
{
  nixl_opt_args_t extra;
  extra.backends.push_back(_backend);
  check(_agent->genNotif(_peers.at(peer).agent, notification, &extra), "genNotif");
}

void nixl_fabric::poll(std::vector<fabric_notification>& out)
{
  receive();
  // A host-staged write's notification goes on only once its bytes are in the receive ring.
  std::erase_if(_landings, [&](landing& l) {
    auto const copied = cudaEventQuery(l.copied);
    if (copied == cudaErrorNotReady) { return false; }
    if (copied != cudaSuccess) {
      throw sirius::internal_exception(
        "exchange: copying a host-staged write from rank {} failed: {}",
        l.from,
        cudaGetErrorString(copied));
    }
    _free_events.push_back(l.copied);
    out.push_back({l.from, std::move(l.notification)});
    return true;
  });
  std::move(_backlog.begin(), _backlog.end(), std::back_inserter(out));
  _backlog.clear();
}

}  // namespace sirius::exchange
