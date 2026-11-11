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

#include <absl/cleanup/cleanup.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <nixl.h>
#include <nvml.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>

namespace sirius::exchange {
namespace {

static_assert(std::endian::native == std::endian::little, "the hello is little-endian");

// A notification's first byte names its kind; the transport's kinds are 1 to 5 (messages.cpp:
// enum class kind); these stay clear of them.
constexpr char hello_kind = '\x7f';
// A host-staged write landed in the receiver's mirror: kind | u64 offset | u64 bytes | the
// write's own notification.
constexpr char staged_kind                = '\x7e';
constexpr std::size_t staged_header_bytes = 1 + 2 * sizeof(std::uint64_t);
constexpr std::size_t pci_bus_id_bytes    = NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE;

// One of the sender's TCP lanes: kind | u32 lane | u32 lanes | u64 receive mirror base |
// u64 receive mirror bytes | the lane agent's metadata.
constexpr char lane_kind = '\x7d';
// Lanes a host's ranks share by default: enough connections to fill a 200 Gb/s port.
constexpr std::uint32_t host_lanes = 16;

// The main agent's metadata, for the peer's writer: kind | the metadata.
constexpr char metadata_kind = '\x7c';

// kind | u32 rank | u32 device | u64 receive ring base | u64 receive ring bytes | u32 pid |
// PCI bus id of the GPU, NUL-padded (names the receive mirror and decides host staging) |
// u32 receive ring in fabric memory | u32 NVLink clique id | 16-byte NVLink cluster UUID (all
// zero: no NVLink domain) | the host's boot id, NUL-padded (tells ranks on other hosts apart)
struct hello {
  rank_t rank;
  std::uint32_t device;
  std::uint64_t base;
  std::uint64_t bytes;
  std::uint32_t pid;
  char pci_bus_id[pci_bus_id_bytes];
  std::uint32_t fabric_ring;
  std::uint32_t clique_id;
  std::uint8_t cluster_uuid[16];
  char boot_id[40];
};

struct lane_header {
  std::uint32_t lane;
  std::uint32_t lanes;
  std::uint64_t mirror_base;
  std::uint64_t mirror_bytes;
};

std::string mirror_name(std::uint32_t pid, rank_t rank)
{
  return "/sirius-exchange-" + std::to_string(pid) + "-" + std::to_string(rank);
}

/// Whether the GPUs at PCI bus ids @p a and @p b can reach each other's memory directly, by
/// NVML's peer-to-peer read and write status (NVML sees every GPU, whatever CUDA_VISIBLE_DEVICES
/// hides). True when NVML cannot tell, which keeps the NIXL path.
bool nvml_up()
{
  static bool const up = nvmlInit_v2() == NVML_SUCCESS;
  return up;
}

bool peer_access(std::string const& a, std::string const& b)
{
  if (a == b) { return true; }
  nvmlDevice_t da{};
  nvmlDevice_t db{};
  if (!nvml_up() || nvmlDeviceGetHandleByPciBusId_v2(a.c_str(), &da) != NVML_SUCCESS ||
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

/// Whether some GPU on this host lacks peer access to the one at @p bus_id, either way, so that a
/// rank on it would stage its writes to this rank through the host (see peer_access). True when
/// NVML cannot tell.
bool some_gpu_lacks_peer_access(std::string const& bus_id)
{
  nvmlDevice_t mine{};
  unsigned count = 0;
  if (!nvml_up() || nvmlDeviceGetHandleByPciBusId_v2(bus_id.c_str(), &mine) != NVML_SUCCESS ||
      nvmlDeviceGetCount_v2(&count) != NVML_SUCCESS) {
    return true;
  }
  for (unsigned i = 0; i < count; ++i) {
    nvmlDevice_t other{};
    if (nvmlDeviceGetHandleByIndex_v2(i, &other) != NVML_SUCCESS) { return true; }
    if (other == mine) { continue; }
    for (auto const index : {NVML_P2P_CAPS_INDEX_READ, NVML_P2P_CAPS_INDEX_WRITE}) {
      for (auto const [from, to] : {std::pair{mine, other}, std::pair{other, mine}}) {
        nvmlGpuP2PStatus_t status{};
        if (nvmlDeviceGetP2PStatus(from, to, index, &status) != NVML_SUCCESS ||
            status != NVML_P2P_STATUS_OK) {
          return true;
        }
      }
    }
  }
  return false;
}

/// The NVLink domain of the GPU at @p bus_id, by NVML's fabric info; invalid when NVML cannot
/// tell or the GPU's fabric registration has not completed.
nvlink_domain nvlink_domain_of(std::string const& bus_id)
{
  nvlink_domain out;
  nvmlDevice_t device{};
  if (!nvml_up() || nvmlDeviceGetHandleByPciBusId_v2(bus_id.c_str(), &device) != NVML_SUCCESS) {
    return out;
  }
  nvmlGpuFabricInfoV_t info{};
  info.version = nvmlGpuFabricInfo_v3;
  if (nvmlDeviceGetGpuFabricInfoV(device, &info) != NVML_SUCCESS ||
      info.state != NVML_GPU_FABRIC_STATE_COMPLETED || info.status != NVML_SUCCESS) {
    return out;
  }
  static_assert(sizeof(info.clusterUuid) == sizeof(out.cluster_uuid));
  std::memcpy(out.cluster_uuid.data(), info.clusterUuid, out.cluster_uuid.size());
  out.clique_id = info.cliqueId;
  return out;
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
  if (auto const status = cudaHostRegister(at, bytes, cudaHostRegisterPortable);
      status != cudaSuccess) {
    ::munmap(at, bytes);
    RMM_CUDA_TRY(status);
  }
  return static_cast<std::byte*>(at);
}

void unmap_shared(std::byte* at, std::size_t bytes)
{
  if (at == nullptr) { return; }
  static_cast<void>(cudaHostUnregister(at));
  ::munmap(at, bytes);
}

/// This boot of this host; empty when unreadable, which treats every peer as local.
std::string boot_id()
{
  std::ifstream in{"/proc/sys/kernel/random/boot_id"};
  std::string id;
  std::getline(in, id);
  return id;
}

std::string lane_name(rank_t rank, std::uint32_t lane)
{
  return "sirius-exchange-" + std::to_string(rank) + "-lane-" + std::to_string(lane);
}

/// The CPUs NVML places next to the GPU at @p bus_id; empty when it cannot tell.
std::vector<int> cpus_near(std::string const& bus_id)
{
  nvmlDevice_t device{};
  if (!nvml_up() || nvmlDeviceGetHandleByPciBusId_v2(bus_id.c_str(), &device) != NVML_SUCCESS) {
    return {};
  }
  constexpr unsigned words  = CPU_SETSIZE / (8 * sizeof(unsigned long));
  unsigned long mask[words] = {};
  if (nvmlDeviceGetCpuAffinity(device, words, mask) != NVML_SUCCESS) { return {}; }
  std::vector<int> cpus;
  for (unsigned w = 0; w < words; ++w) {
    for (unsigned b = 0; b < 8 * sizeof(unsigned long); ++b) {
      if ((mask[w] >> b) & 1UL) {
        cpus.push_back(static_cast<int>(w * 8 * sizeof(unsigned long) + b));
      }
    }
  }
  return cpus;
}

void pin_to(std::vector<int> const& cpus)
{
  if (cpus.empty()) { return; }
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int const c : cpus) {
    CPU_SET(c, &set);
  }
  static_cast<void>(pthread_setaffinity_np(pthread_self(), sizeof(set), &set));
}

/// Runs this thread on @p cpus while in scope, so memory it first touches is near them.
class pinned_scope {
 public:
  explicit pinned_scope(std::vector<int> const& cpus) : _active{!cpus.empty()}
  {
    if (!_active) { return; }
    _active = pthread_getaffinity_np(pthread_self(), sizeof(_saved), &_saved) == 0;
    if (_active) { pin_to(cpus); }
  }
  ~pinned_scope()
  {
    if (_active) {
      static_cast<void>(pthread_setaffinity_np(pthread_self(), sizeof(_saved), &_saved));
    }
  }
  pinned_scope(pinned_scope const&)            = delete;
  pinned_scope& operator=(pinned_scope const&) = delete;

 private:
  bool _active;
  cpu_set_t _saved{};
};

/// @p bytes of page-locked host memory, faulted in by this thread.
std::byte* pinned_host(std::size_t bytes)
{
  void* at = ::mmap(
    nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  if (at == MAP_FAILED) {
    throw sirius::internal_exception(
      "exchange: mapping {} bytes of send slots failed: {}", bytes, std::strerror(errno));
  }
  if (auto const e = cudaHostRegister(at, bytes, cudaHostRegisterPortable); e != cudaSuccess) {
    ::munmap(at, bytes);
    throw sirius::internal_exception(
      "exchange: pinning {} bytes of send slots failed: {}", bytes, cudaGetErrorString(e));
  }
  return static_cast<std::byte*>(at);
}

/// UCX parses its global configuration unsafely: two backends created at once in one process
/// can read a garbled setting and fail.
std::mutex& creating_backends()
{
  static std::mutex m;
  return m;
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
void require_free_port(std::uint16_t port)
{
  int const fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) { throw sirius::internal_exception("exchange NIXL: socket() failed"); }
  int const on = 1;
  static_cast<void>(::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)));
  sockaddr_in at{};
  at.sin_family      = AF_INET;
  at.sin_addr.s_addr = htonl(INADDR_ANY);
  at.sin_port        = htons(port);
  bool const bound   = ::bind(fd, reinterpret_cast<sockaddr*>(&at), sizeof(at)) == 0;
  ::close(fd);
  if (!bound) {
    throw sirius::invalid_input_exception("exchange NIXL: listen port {} is in use", port);
  }
}

std::string agent_name(rank_t rank) { return "sirius-exchange-" + std::to_string(rank); }
std::string writer_name(rank_t rank) { return agent_name(rank) + "-writer"; }

template <typename Peer>
endpoint endpoint_of(Peer const& p, rank_t rank)
{
  return endpoint{rank, p.pci_bus_id, p.fabric_ring, p.domain, p.boot_id};
}

nixl_reg_dlist_t ring_descs(frame_ring const& ring, int device)
{
  nixl_reg_dlist_t descs(VRAM_SEG);
  descs.addDesc(nixlBlobDesc(reinterpret_cast<std::uintptr_t>(ring.base()),
                             ring.bytes(),
                             static_cast<std::uint64_t>(device),
                             ""));
  return descs;
}

nixl_reg_dlist_t host_descs(std::byte const* at, std::size_t bytes)
{
  nixl_reg_dlist_t descs(DRAM_SEG);
  descs.addDesc(nixlBlobDesc(reinterpret_cast<std::uintptr_t>(at), bytes, 0, ""));
  return descs;
}

}  // namespace

/// A TCP lane: its agent, pinned send slots and thread. The engine thread hands it writes whose
/// bytes are in a slot; the lane thread posts them, gives slots back when they are written, and
/// collects the notifications its peers' lanes send.
struct nixl_fabric::lane {
  struct job {
    std::string agent;
    std::uintptr_t remote;
    std::uint32_t slot;
    std::size_t bytes;
    std::string notification;
  };

  std::uint32_t index{0};
  std::unique_ptr<nixlAgent> agent;
  nixlBackendH* backend{nullptr};
  std::byte* slots{nullptr};
  std::size_t slot_bytes{0};
  std::uint32_t slot_count{0};
  std::thread thread;
  std::atomic<bool> stop{false};
  std::atomic<bool> failed{false};

  std::mutex mutex;
  std::vector<std::uint32_t> free_slots;
  std::deque<job> jobs;
  std::vector<std::pair<std::string, std::string>> inbox;
  std::string error;

  [[nodiscard]] std::size_t slots_bytes() const { return slot_bytes * slot_count; }

  void run(std::vector<int> const& cpus)
  {
    pin_to(cpus);
    struct flying {
      nixlXferReqH* req;
      std::uint32_t slot;
    };
    std::vector<flying> in_flight;
    std::deque<job> taken;
    std::vector<std::pair<std::string, std::string>> heard;
    std::uint32_t idle = 0;
    std::chrono::steady_clock::time_point stopping{};
    try {
      while (true) {
        bool progress = false;
        {
          std::lock_guard lock{mutex};
          taken.swap(jobs);
        }
        for (auto& j : taken) {
          nixl_xfer_dlist_t local(DRAM_SEG);
          local.addDesc(nixlBasicDesc(
            reinterpret_cast<std::uintptr_t>(slots) + std::size_t{j.slot} * slot_bytes,
            j.bytes,
            0));
          nixl_xfer_dlist_t remote(DRAM_SEG);
          remote.addDesc(nixlBasicDesc(j.remote, j.bytes, 0));
          nixl_opt_args_t extra;
          extra.backends.push_back(backend);
          extra.notif       = std::move(j.notification);
          nixlXferReqH* req = nullptr;
          check(agent->createXferReq(NIXL_WRITE, local, remote, j.agent, req, &extra),
                "createXferReq(lane)");
          auto const posted = agent->postXferReq(req);
          if (posted != NIXL_SUCCESS && posted != NIXL_IN_PROG) {
            static_cast<void>(agent->releaseXferReq(req));
            check(posted, "postXferReq(lane)");
          }
          in_flight.push_back({req, j.slot});
          progress = true;
        }
        taken.clear();
        for (auto it = in_flight.begin(); it != in_flight.end();) {
          auto const s = agent->getXferStatus(it->req);
          if (s == NIXL_IN_PROG) {
            ++it;
            continue;
          }
          static_cast<void>(agent->releaseXferReq(it->req));
          check(s, "a TCP lane write");
          {
            std::lock_guard lock{mutex};
            free_slots.push_back(it->slot);
          }
          it       = in_flight.erase(it);
          progress = true;
        }
        nixl_notifs_t notifs;
        check(agent->getNotifs(notifs), "getNotifs(lane)");
        for (auto& [from, bodies] : notifs) {
          for (auto& body : bodies) {
            heard.emplace_back(from, std::move(body));
          }
        }
        if (!heard.empty()) {
          std::lock_guard lock{mutex};
          std::move(heard.begin(), heard.end(), std::back_inserter(inbox));
          heard.clear();
          progress = true;
        }
        if (stop.load()) {
          // A peer that is gone never completes its writes: give up on them after a while.
          if (stopping == std::chrono::steady_clock::time_point{}) {
            stopping = std::chrono::steady_clock::now();
          }
          std::lock_guard lock{mutex};
          if ((in_flight.empty() && jobs.empty()) ||
              std::chrono::steady_clock::now() - stopping > std::chrono::seconds{10}) {
            break;
          }
        }
        // Spin while busy; once idle, poll the sockets every 50 us instead of burning a core.
        if (progress) {
          idle = 0;
        } else if (++idle > 2000) {
          std::this_thread::sleep_for(std::chrono::microseconds{50});
        }
      }
      for (auto const& f : in_flight) {
        static_cast<void>(agent->releaseXferReq(f.req));
      }
    } catch (std::exception const& e) {
      std::lock_guard lock{mutex};
      error = e.what();
      failed.store(true);
    }
  }
};

nixl_fabric::nixl_fabric(config cfg, frame_ring& send, frame_ring& recv)
  : _rank(cfg.rank), _send(send), _recv(recv), _staging(cfg.staging), _mnnvl(cfg.mnnvl)
{
  if (cfg.peers.empty() || cfg.rank >= cfg.peers.size()) {
    throw sirius::invalid_input_exception(
      "exchange NIXL: rank {} of {} peers", cfg.rank, cfg.peers.size());
  }
  if (cfg.lanes > max_tcp_lanes || cfg.slots_per_lane == 0) {
    throw sirius::invalid_input_exception(
      "exchange NIXL: {} TCP lanes of {} send slots each", cfg.lanes, cfg.slots_per_lane);
  }
  RMM_CUDA_TRY(cudaGetDevice(&_device));
  _peers.resize(cfg.peers.size());
  for (rank_t r = 0; r < _peers.size(); ++r) {
    auto const& address   = cfg.peers[r];
    auto const colon      = address.rfind(':');
    auto& p               = _peers[r];
    auto const* const end = address.data() + address.size();
    auto const parsed     = colon == std::string::npos
                              ? std::from_chars_result{nullptr, std::errc::invalid_argument}
                              : std::from_chars(address.data() + colon + 1, end, p.port);
    if (parsed.ec != std::errc{} || parsed.ptr != end || p.port == 0) {
      throw sirius::invalid_input_exception("exchange NIXL: peer address '{}' is not host:port",
                                            address);
    }
    p.host                   = address.substr(0, colon);
    p.agent                  = agent_name(r);
    _rank_of[p.agent]        = r;
    _rank_of[writer_name(r)] = r;
    for (std::uint32_t l = 0; l < max_tcp_lanes; ++l) {
      _rank_of[lane_name(r, l)] = r;
    }
  }
  auto const& my_host = _peers[_rank].host;
  auto const on_my_host =
    std::count_if(_peers.begin(), _peers.end(), [&](peer const& p) { return p.host == my_host; });
  _lane_count     = cfg.lanes != 0
                      ? cfg.lanes
                      : std::max<std::uint32_t>(2, host_lanes / static_cast<std::uint32_t>(on_my_host));
  _slots_per_lane = cfg.slots_per_lane;

  require_free_port(_peers[_rank].port);
  // A destructor does not run for a constructor that throws, and a peer starting late is enough
  // to throw: the mirror would stay in /dev/shm.
  absl::Cleanup release_on_throw = [this] { release(); };
  RMM_CUDA_TRY(cudaStreamCreateWithFlags(&_d2h, cudaStreamNonBlocking));
  RMM_CUDA_TRY(cudaStreamCreateWithFlags(&_h2d, cudaStreamNonBlocking));
  char bus_id[pci_bus_id_bytes] = {};
  RMM_CUDA_TRY(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), _device));
  _peers[_rank].pci_bus_id  = bus_id;
  _peers[_rank].pid         = static_cast<std::uint32_t>(::getpid());
  _peers[_rank].fabric_ring = recv.in_fabric_memory();
  _peers[_rank].domain      = nvlink_domain_of(bus_id);
  _peers[_rank].boot_id     = boot_id();
  _near_cpus                = cpus_near(bus_id);
  // The mirror exists before the hello names it, so a peer that hears the hello can map it. In
  // automatic mode only a same-host GPU without peer access to this one would stage to it. A
  // peer that writes to it over TCP lanes learns its address with the lanes, so choose_route()
  // makes it then: boot ids and NVLink domains, which decide that path, come with the hellos.
  bool const mirror = _peers.size() > 1 && (cfg.staging == host_staging::always ||
                                            (cfg.staging == host_staging::automatic &&
                                             some_gpu_lacks_peer_access(_peers[_rank].pci_bus_id)));
  if (mirror) {
    try {
      make_mirror();
    } catch (std::exception const& e) {
      if (cfg.staging == host_staging::always) { throw; }
      SIRIUS_LOG_WARN("exchange: no receive mirror ({}); a peer that must stage to rank {} fails",
                      e.what(),
                      _rank);
    }
  }

  std::unique_lock create_lock{creating_backends()};
  nixlAgentConfig agent_cfg(/*use_prog_thread=*/false,
                            /*use_listen_thread=*/true,
                            _peers[_rank].port,
                            nixl_thread_sync_t::NIXL_THREAD_SYNC_RW);
  _agent = std::make_unique<nixlAgent>(agent_name(_rank), agent_cfg);
  nixl_mem_list_t mems;
  nixl_b_params_t params;
  check(_agent->getPluginParams("UCX", mems, params), "getPluginParams(UCX)");
  check(_agent->createBackend("UCX", params, _backend), "createBackend(UCX)");
  nixlAgentConfig writer_cfg(/*use_prog_thread=*/false,
                             /*use_listen_thread=*/false,
                             0,
                             nixl_thread_sync_t::NIXL_THREAD_SYNC_RW);
  _writer = std::make_unique<nixlAgent>(writer_name(_rank), writer_cfg);
  nixl_b_params_t writer_params;
  check(_writer->getPluginParams("UCX", mems, writer_params), "getPluginParams(UCX)");
  check(_writer->createBackend("UCX", writer_params, _writer_backend), "createBackend(UCX)");
  create_lock.unlock();
  check(_writer->registerMem(ring_descs(_send, _device)), "registerMem(send ring)");
  check(_agent->registerMem(ring_descs(_recv, _device)), "registerMem(receive ring)");
  std::string metadata(1, metadata_kind);
  {
    nixl_blob_t md;
    check(_agent->getLocalMD(md), "getLocalMD");
    metadata += md;
  }

  // Fetch every peer's metadata from its listener and send it a hello. A peer that is not up
  // yet refuses the fetch, so fetches repeat until the deadline. The listener serves metadata
  // from the agent's start, before its rings are registered, so a peer is ready only once the
  // metadata loaded here covers the receive ring its hello names; until then it is fetched again.
  auto const deadline = std::chrono::steady_clock::now() + cfg.connect_timeout;
  hello mine{_rank,
             static_cast<std::uint32_t>(_device),
             reinterpret_cast<std::uint64_t>(_recv.base()),
             _recv.bytes(),
             // 0: no mirror, which map_mirror() on a peer reports.
             _mirror != nullptr ? _peers[_rank].pid : 0u,
             {},
             _peers[_rank].fabric_ring ? 1u : 0u,
             _peers[_rank].domain.clique_id,
             {},
             {}};
  std::memcpy(mine.pci_bus_id, bus_id, sizeof(mine.pci_bus_id));
  std::memcpy(mine.cluster_uuid, _peers[_rank].domain.cluster_uuid.data(), 16);
  std::strncpy(mine.boot_id, _peers[_rank].boot_id.c_str(), sizeof(mine.boot_id) - 1);
  auto const me = encode_hello(mine);
  if (cfg.mnnvl != mnnvl_mode::off) {
    SIRIUS_LOG_INFO("exchange: rank {} rings in {} memory, NVLink domain {}",
                    _rank,
                    _peers[_rank].fabric_ring ? "fabric" : "cudaMalloc",
                    _peers[_rank].domain.to_string());
  }
  std::vector<bool> greeted(_peers.size(), false);
  std::vector<bool> lanes_sent(_peers.size(), false);
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
        check(_agent->genNotif(p.agent, metadata), "genNotif(metadata)");
        greeted[r] = true;
      }
      if (known && p.hello && p.writer_ready) {
        nixl_xfer_dlist_t ring(VRAM_SEG);
        ring.addDesc(nixlBasicDesc(p.recv_base, p.recv_bytes, p.device));
        if (_agent->checkRemoteMD(p.agent, ring) == NIXL_SUCCESS) {
          if (p.route != path::tcp_lanes) {
            ready[r] = true;
            continue;
          }
          // Over TCP lanes the peer is ready once both have every lane's metadata.
          if (!lanes_sent[r]) {
            for (auto const& l : _lanes) {
              nixl_blob_t lane_md;
              check(l->agent->getLocalMD(lane_md), "getLocalMD(lane)");
              lane_header const h{
                l->index, _lane_count, reinterpret_cast<std::uint64_t>(_mirror), _recv.bytes()};
              std::string body(1, lane_kind);
              body.append(reinterpret_cast<char const*>(&h), sizeof(h));
              body += lane_md;
              check(_agent->genNotif(p.agent, body), "genNotif(lane)");
            }
            lanes_sent[r] = true;
          }
          if (p.lanes > 0 && p.lanes_loaded == p.lanes) {
            ready[r] = true;
            continue;
          }
          ++waiting;
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

  // Every peer's path was chosen as its hello came in (take()).
  start_lanes();
  std::move(release_on_throw).Cancel();
}

void nixl_fabric::make_mirror()
{
  _mirror_name = mirror_name(_peers[_rank].pid, _rank);
  // Fault the mirror in near this GPU.
  pinned_scope near{_near_cpus};
  try {
    _mirror = map_shared(_mirror_name, _recv.bytes(), true);
  } catch (...) {
    ::shm_unlink(_mirror_name.c_str());
    _mirror_name.clear();
    throw;
  }
}

void nixl_fabric::choose_route(rank_t r)
{
  auto& p              = _peers[r];
  auto const& self     = _peers[_rank];
  bool const same_host = p.boot_id == self.boot_id;
  // NVML knows this host's GPUs only.
  p.route = choose_path(endpoint_of(self, _rank),
                        endpoint_of(p, r),
                        _staging,
                        _mnnvl,
                        _staging != host_staging::automatic || !same_host ||
                          peer_access(self.pci_bus_id, p.pci_bus_id));
  if (p.route == path::shared_memory) { map_mirror(r); }
  if (p.route == path::tcp_lanes) {
    // The peer's lanes write into the mirror, and this rank's lanes register it.
    if (_mirror == nullptr) { make_mirror(); }
    make_lanes();
  }
  SIRIUS_LOG_INFO("exchange: rank {} (GPU {}) -> rank {} (GPU {}{}, NVLink domain {}): {}",
                  _rank,
                  self.pci_bus_id,
                  r,
                  p.pci_bus_id,
                  same_host ? "" : ", another host",
                  p.domain.to_string(),
                  describe(p.route));
}

void nixl_fabric::make_lanes()
{
  if (!_lanes.empty()) { return; }
  // Slots are faulted in near this GPU; lane threads run there too.
  pinned_scope near{_near_cpus};
  for (std::uint32_t i = 0; i < _lane_count; ++i) {
    auto l   = std::make_unique<lane>();
    l->index = i;
    nixlAgentConfig agent_cfg(/*use_prog_thread=*/false,
                              /*use_listen_thread=*/false,
                              0,
                              nixl_thread_sync_t::NIXL_THREAD_SYNC_RW);
    l->agent = std::make_unique<nixlAgent>(lane_name(_rank, i), agent_cfg);
    {
      std::lock_guard create_lock{creating_backends()};
      nixl_mem_list_t mems;
      nixl_b_params_t params;
      check(l->agent->getPluginParams("UCX", mems, params), "getPluginParams(UCX lane)");
      check(l->agent->createBackend("UCX", params, l->backend), "createBackend(UCX lane)");
    }
    l->slot_bytes = _send.frame_bytes();
    l->slot_count = _slots_per_lane;
    // In _lanes before anything can throw, so release() frees what it holds.
    auto& lane = *_lanes.emplace_back(std::move(l));
    lane.slots = pinned_host(lane.slots_bytes());
    for (std::uint32_t s = lane.slot_count; s > 0; --s) {
      lane.free_slots.push_back(s - 1);
    }
    check(lane.agent->registerMem(host_descs(lane.slots, lane.slots_bytes())),
          "registerMem(send slots)");
    check(lane.agent->registerMem(host_descs(_mirror, _recv.bytes())),
          "registerMem(receive mirror)");
  }
  SIRIUS_LOG_INFO("exchange: rank {} has {} TCP lanes of {} {}-byte send slots each",
                  _rank,
                  _lane_count,
                  _slots_per_lane,
                  _send.frame_bytes());
}

void nixl_fabric::start_lanes()
{
  for (auto& l : _lanes) {
    l->thread = std::thread([this, raw = l.get()] { raw->run(_near_cpus); });
  }
}

void nixl_fabric::stop_lanes() noexcept
{
  for (auto& l : _lanes) {
    l->stop.store(true);
  }
  for (auto& l : _lanes) {
    if (l->thread.joinable()) { l->thread.join(); }
  }
}

void nixl_fabric::check_lanes() const
{
  for (auto const& l : _lanes) {
    if (l->failed.load()) {
      std::lock_guard lock{l->mutex};
      throw sirius::internal_exception("exchange: TCP lane {} failed: {}", l->index, l->error);
    }
  }
}

bool nixl_fabric::start_lane_write(lane_write& w)
{
  auto& p = _peers[w.peer];
  for (std::uint32_t k = 0; k < p.lanes; ++k) {
    auto const i = (p.next_lane + k) % p.lanes;
    auto& l      = *_lanes[i];
    std::uint32_t slot{};
    {
      std::lock_guard lock{l.mutex};
      if (l.free_slots.empty()) { continue; }
      slot = l.free_slots.back();
      l.free_slots.pop_back();
    }
    p.next_lane = i + 1;
    w.lane      = i;
    w.slot      = slot;
    RMM_CUDA_TRY(cudaMemcpyAsync(l.slots + std::size_t{slot} * l.slot_bytes,
                                 _send.base() + w.src,
                                 w.bytes,
                                 cudaMemcpyDeviceToHost,
                                 _d2h));
    w.copied = take_event();
    RMM_CUDA_TRY(cudaEventRecord(w.copied, _d2h));
    return true;
  }
  return false;
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

nixl_fabric::~nixl_fabric() { release(); }

void nixl_fabric::release() noexcept
{
  // Lanes finish the writes they hold (or give up on a peer that is gone) before they end.
  stop_lanes();
  if (_writer) {
    // A write still in flight reads the send ring: give it a moment before deregistering.
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    for (auto& [id, req] : _writes) {
      while (_writer->getXferStatus(req) == NIXL_IN_PROG &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds{100});
      }
      if (auto const s = _writer->releaseXferReq(req); s != NIXL_SUCCESS) {
        SIRIUS_LOG_WARN(
          "exchange NIXL: releasing write {} failed: {}", id, nixlEnumStrings::statusStr(s));
      }
    }
    static_cast<void>(_writer->deregisterMem(ring_descs(_send, _device)));
  }
  if (_agent) { static_cast<void>(_agent->deregisterMem(ring_descs(_recv, _device))); }
  // Copies still in flight read or write the mirrors and slots: let them finish first.
  if (_d2h != nullptr) { static_cast<void>(cudaStreamSynchronize(_d2h)); }
  if (_h2d != nullptr) { static_cast<void>(cudaStreamSynchronize(_h2d)); }
  for (auto& l : _lanes) {
    if (l->slots != nullptr) {
      static_cast<void>(l->agent->deregisterMem(host_descs(l->slots, l->slots_bytes())));
      static_cast<void>(l->agent->deregisterMem(host_descs(_mirror, _recv.bytes())));
    }
    l->agent.reset();
    unmap_shared(l->slots, l->slots_bytes());
  }
  _lanes.clear();
  for (auto& [id, w] : _lane_writes) {
    if (w.copied != nullptr) { _free_events.push_back(w.copied); }
  }
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
      take(it->second, agent, std::move(body));
    }
  }
}

void nixl_fabric::take(rank_t from, std::string const& agent, std::string body)
{
  if (!body.empty() && body.front() == hello_kind) {
    hello h{};
    if (body.size() != 1 + sizeof(h)) {
      throw sirius::invalid_input_exception("exchange NIXL: a malformed hello from {}", agent);
    }
    std::memcpy(&h, body.data() + 1, sizeof(h));
    auto& p = _peers[from];
    if (h.rank != from || h.bytes != _recv.bytes()) {
      throw sirius::invalid_input_exception(
        "exchange NIXL: {} says it is rank {} with a {}-byte receive ring; expected rank {} "
        "with {} bytes",
        agent,
        h.rank,
        h.bytes,
        from,
        _recv.bytes());
    }
    p.recv_base  = h.base;
    p.recv_bytes = h.bytes;
    p.device     = h.device;
    p.pid        = h.pid;
    p.pci_bus_id.assign(h.pci_bus_id, strnlen(h.pci_bus_id, sizeof(h.pci_bus_id)));
    p.fabric_ring = h.fabric_ring != 0;
    std::memcpy(p.domain.cluster_uuid.data(), h.cluster_uuid, p.domain.cluster_uuid.size());
    p.domain.clique_id = h.clique_id;
    p.boot_id.assign(h.boot_id, strnlen(h.boot_id, sizeof(h.boot_id)));
    if (!p.hello) { choose_route(from); }
    p.hello = true;
    return;
  }
  if (!body.empty() && body.front() == metadata_kind) {
    auto& p = _peers[from];
    if (agent != p.agent) {
      throw sirius::invalid_input_exception(
        "exchange NIXL: metadata for {} came from {}", p.agent, agent);
    }
    std::string loaded;
    check(_writer->loadRemoteMD(body.substr(1), loaded), "loadRemoteMD");
    if (loaded != p.agent) {
      throw sirius::invalid_input_exception(
        "exchange NIXL: {} sent the metadata of {}", agent, loaded);
    }
    check(_writer->makeConnection(p.agent), "makeConnection(writer)");
    p.writer_ready = true;
    return;
  }
  if (!body.empty() && body.front() == lane_kind) {
    lane_header h{};
    auto& p = _peers[from];
    if (body.size() <= 1 + sizeof(h)) {
      throw sirius::invalid_input_exception("exchange NIXL: a malformed lane from {}", agent);
    }
    std::memcpy(&h, body.data() + 1, sizeof(h));
    // The hello came first on the same connection, so this rank has chosen its path.
    if (p.route != path::tcp_lanes || h.mirror_base == 0 || h.mirror_bytes != _recv.bytes()) {
      throw sirius::invalid_input_exception(
        "exchange NIXL: {} offers TCP lanes, but this rank reaches it another way (set "
        "sirius_exchange_host_staging and sirius_exchange_mnnvl alike on every rank)",
        agent);
    }
    p.mirror_base = h.mirror_base;
    p.lanes       = std::min(_lane_count, h.lanes);
    if (h.lane < p.lanes) {
      auto& l = *_lanes[h.lane];
      std::string loaded;
      check(l.agent->loadRemoteMD(body.substr(1 + sizeof(h)), loaded), "loadRemoteMD(lane)");
      if (loaded != lane_name(from, h.lane)) {
        throw sirius::invalid_input_exception(
          "exchange NIXL: {} sent lane metadata of agent '{}'", agent, loaded);
      }
      ++p.lanes_loaded;
    }
    return;
  }
  if (!body.empty() && body.front() == staged_kind) {
    std::uint64_t at    = 0;
    std::uint64_t bytes = 0;
    if (_mirror == nullptr || body.size() < staged_header_bytes) {
      throw sirius::invalid_input_exception(
        "exchange NIXL: a host-staged write from {} that this rank cannot take", agent);
    }
    std::memcpy(&at, body.data() + 1, sizeof(at));
    std::memcpy(&bytes, body.data() + 1 + sizeof(at), sizeof(bytes));
    if (bytes > _recv.bytes() || at > _recv.bytes() - bytes) {
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
    _landings.push_back({copied, from, body.substr(staged_header_bytes)});
    return;
  }
  _backlog.push_back({from, std::move(body)});
}

std::size_t nixl_fabric::max_write_bytes(rank_t peer) const
{
  return _peers.at(peer).route == path::tcp_lanes ? _send.frame_bytes()
                                                  : fabric::max_write_bytes(peer);
}

std::uint64_t nixl_fabric::write(
  rank_t peer, std::size_t src, std::size_t dst, std::size_t bytes, std::string notification)
{
  auto const& p = _peers.at(peer);
  if (src + bytes > _send.bytes() || dst + bytes > p.recv_bytes) {
    throw sirius::invalid_input_exception(
      "exchange NIXL: a {}-byte write from {} to {} is outside the rings", bytes, src, dst);
  }
  if (p.route == path::shared_memory || p.route == path::tcp_lanes) {
    std::string landed(1, staged_kind);
    auto const at = static_cast<std::uint64_t>(dst);
    auto const n  = static_cast<std::uint64_t>(bytes);
    landed.append(reinterpret_cast<char const*>(&at), sizeof(at));
    landed.append(reinterpret_cast<char const*>(&n), sizeof(n));
    landed += notification;
    if (p.route == path::tcp_lanes) {
      if (bytes > _send.frame_bytes()) {
        throw sirius::invalid_input_exception(
          "exchange NIXL: a {}-byte write over TCP lanes, whose send slots hold {} bytes",
          bytes,
          _send.frame_bytes());
      }
      check_lanes();
      auto [it, _] =
        _lane_writes.emplace(_next_write, lane_write{peer, src, dst, bytes, std::move(landed)});
      static_cast<void>(start_lane_write(it->second));
      return _next_write++;
    }
    RMM_CUDA_TRY(
      cudaMemcpyAsync(p.mirror + dst, _send.base() + src, bytes, cudaMemcpyDeviceToHost, _d2h));
    auto* copied = take_event();
    RMM_CUDA_TRY(cudaEventRecord(copied, _d2h));
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
  extra.backends.push_back(_writer_backend);
  extra.notif       = std::move(notification);
  nixlXferReqH* req = nullptr;
  check(_writer->createXferReq(NIXL_WRITE, local, remote, p.agent, req, &extra), "createXferReq");
  auto const posted = _writer->postXferReq(req);
  if (posted != NIXL_SUCCESS && posted != NIXL_IN_PROG) {
    static_cast<void>(_writer->releaseXferReq(req));
    check(posted, "postXferReq");
  }
  _writes.emplace(_next_write, req);
  return _next_write++;
}

double nixl_fabric::timed_write(rank_t peer, std::size_t dst, std::size_t bytes)
{
  auto const& p = _peers.at(peer);
  if (host_staged(peer) || bytes > _send.bytes() || dst + bytes > p.recv_bytes) {
    throw sirius::invalid_input_exception(
      "exchange NIXL: a timed {}-byte write to {} of rank {} is not possible", bytes, dst, peer);
  }
  nixl_xfer_dlist_t local(VRAM_SEG);
  local.addDesc(nixlBasicDesc(
    reinterpret_cast<std::uintptr_t>(_send.base()), bytes, static_cast<std::uint64_t>(_device)));
  nixl_xfer_dlist_t remote(VRAM_SEG);
  remote.addDesc(nixlBasicDesc(p.recv_base + dst, bytes, p.device));
  nixl_opt_args_t extra;
  extra.backends.push_back(_writer_backend);
  nixlXferReqH* req = nullptr;
  check(_writer->createXferReq(NIXL_WRITE, local, remote, p.agent, req, &extra), "createXferReq");
  absl::Cleanup release_req = [&] { static_cast<void>(_writer->releaseXferReq(req)); };
  auto const start          = std::chrono::steady_clock::now();
  auto const deadline       = start + std::chrono::seconds{10};
  auto s                    = _writer->postXferReq(req);
  while (s == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
    // A peer probing this rank at the same time writes into this rank's main agent, whose
    // worker must make progress for that write's endpoint to wire up.
    receive();
    s = _writer->getXferStatus(req);
  }
  auto const seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (s == NIXL_IN_PROG) {
    throw sirius::internal_exception(
      "exchange NIXL: a timed {}-byte write to rank {} took over ten seconds", bytes, peer);
  }
  check(s, "timed write");
  return seconds;
}

fabric::write_status nixl_fabric::status(std::uint64_t id)
{
  if (auto const it = _lane_writes.find(id); it != _lane_writes.end()) {
    check_lanes();
    auto& w = it->second;
    if (w.copied == nullptr && !start_lane_write(w)) { return write_status::pending; }
    auto const copied = cudaEventQuery(w.copied);
    if (copied == cudaErrorNotReady) { return write_status::pending; }
    _free_events.push_back(w.copied);
    auto& l = *_lanes[w.lane];
    if (copied != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      SIRIUS_LOG_ERROR(
        "exchange NIXL: TCP lane write {} failed: {}", id, cudaGetErrorString(copied));
      {
        std::lock_guard lock{l.mutex};
        l.free_slots.push_back(w.slot);
      }
      _lane_writes.erase(it);
      return write_status::failed;
    }
    // The bytes are in the slot, so the send ring may take others; the lane writes the slot.
    auto const& p = _peers[w.peer];
    {
      std::lock_guard lock{l.mutex};
      l.jobs.push_back({lane_name(w.peer, w.lane),
                        p.mirror_base + w.dst,
                        w.slot,
                        w.bytes,
                        std::move(w.notification)});
    }
    _lane_writes.erase(it);
    return write_status::done;
  }
  if (auto const staged = _staged.find(id); staged != _staged.end()) {
    auto const copied = cudaEventQuery(staged->second.copied);
    if (copied == cudaErrorNotReady) { return write_status::pending; }
    auto w = std::move(staged->second);
    _staged.erase(staged);
    _free_events.push_back(w.copied);
    if (copied != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      SIRIUS_LOG_ERROR(
        "exchange NIXL: host-staged write {} failed: {}", id, cudaGetErrorString(copied));
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
  auto const s = _writer->getXferStatus(it->second);
  if (s == NIXL_IN_PROG) { return write_status::pending; }
  static_cast<void>(_writer->releaseXferReq(it->second));
  _writes.erase(it);
  if (s != NIXL_SUCCESS) {
    SIRIUS_LOG_ERROR("exchange NIXL: write {} failed: {}", id, nixlEnumStrings::statusStr(s));
    return write_status::failed;
  }
  return write_status::done;
}

void nixl_fabric::notify(rank_t peer, std::string notification)
{
  nixl_opt_args_t extra;
  extra.backends.push_back(_backend);
  check(_agent->genNotif(_peers.at(peer).agent, notification, &extra), "genNotif");
}

void nixl_fabric::poll(std::vector<fabric_notification>& out)
{
  check_lanes();
  receive();
  std::vector<std::pair<std::string, std::string>> heard;
  for (auto& l : _lanes) {
    {
      std::lock_guard lock{l->mutex};
      heard.swap(l->inbox);
    }
    for (auto& [agent, body] : heard) {
      auto const it = _rank_of.find(agent);
      if (it == _rank_of.end()) {
        throw sirius::internal_exception(
          "exchange NIXL: lane {} heard from unknown agent '{}'", l->index, agent);
      }
      take(it->second, agent, std::move(body));
    }
    heard.clear();
  }
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
