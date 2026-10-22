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

#include "sirius/exception.hpp"

#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <nixl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <bit>
#include <charconv>
#include <cstring>
#include <mutex>
#include <thread>

namespace sirius::exchange {
namespace {

static_assert(std::endian::native == std::endian::little, "the hello is little-endian");

// A notification's first byte names its kind; the transport's kinds are 1 to 4.
constexpr char hello_kind = '\x7f';

// kind | u32 rank | u32 device | u64 receive ring base | u64 receive ring bytes
struct hello {
  std::uint32_t rank;
  std::uint32_t device;
  std::uint64_t base;
  std::uint64_t bytes;
};

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
  auto const me       = encode_hello({_rank,
                                      static_cast<std::uint32_t>(_device),
                                      reinterpret_cast<std::uint64_t>(_recv.base()),
                                      _recv.bytes()});
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
}

nixl_fabric::~nixl_fabric()
{
  for (auto& [id, req] : _writes) {
    static_cast<void>(_agent->releaseXferReq(req));
  }
  static_cast<void>(_agent->deregisterMem(ring_descs(_send, _device)));
  static_cast<void>(_agent->deregisterMem(ring_descs(_recv, _device)));
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
        p.hello      = true;
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
  std::move(_backlog.begin(), _backlog.end(), std::back_inserter(out));
  _backlog.clear();
}

}  // namespace sirius::exchange
