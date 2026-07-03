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

#include <rmm/cuda_device.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <op/dynamic_filter_replica_transfer.hpp>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace sirius::op::detail {

namespace {

[[noreturn]] void throw_cuda_error(char const* operation, cudaError_t status)
{
  throw std::runtime_error(std::string{operation} + ": " + cudaGetErrorString(status));
}

void check_cuda(cudaError_t status, char const* operation)
{
  if (status != cudaSuccess) { throw_cuda_error(operation, status); }
}

class current_device_restore final {
 public:
  current_device_restore() noexcept : _valid{cudaGetDevice(&_device) == cudaSuccess}
  {
    if (!_valid) { (void)cudaGetLastError(); }
  }

  ~current_device_restore() noexcept
  {
    if (_valid) {
      (void)cudaSetDevice(_device);
      (void)cudaGetLastError();
    }
  }

  [[nodiscard]] bool valid() const noexcept { return _valid; }

 private:
  int _device{-1};
  bool _valid{false};
};

class portable_pinned_buffer final {
 public:
  explicit portable_pinned_buffer(std::size_t bytes)
  {
    check_cuda(cudaHostAlloc(&_data, bytes, cudaHostAllocPortable),
               "cudaHostAlloc(portable dynamic-filter staging)");
  }

  ~portable_pinned_buffer() noexcept
  {
    if (_data != nullptr) {
      (void)cudaFreeHost(_data);
      (void)cudaGetLastError();
    }
  }

  portable_pinned_buffer(portable_pinned_buffer const&)            = delete;
  portable_pinned_buffer& operator=(portable_pinned_buffer const&) = delete;

  [[nodiscard]] void* get() const noexcept { return _data; }
  [[nodiscard]] void* release() noexcept { return std::exchange(_data, nullptr); }

 private:
  void* _data{nullptr};
};

bool pool_has_read_write_access(cudaMemPool_t pool, rmm::cuda_device_id accessor) noexcept
{
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeDevice;
  location.id   = accessor.value();
  cudaMemAccessFlags flags{};
  auto const status = cudaMemPoolGetAccess(&flags, pool, &location);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return flags == cudaMemAccessFlagsProtReadWrite;
}

// cudaMallocAsync allocations require access on their actual pool. Resolving the allocation, rather
// than the device's current/default pool, also handles RMM resources backed by recreated/custom
// pools. A null pool is a legacy cudaMalloc allocation governed by cudaDeviceEnablePeerAccess.
bool grant_allocation_peer_access(void const* allocation,
                                  rmm::cuda_device_id owner,
                                  rmm::cuda_device_id accessor) noexcept
{
  current_device_restore restore;
  if (!restore.valid() || cudaSetDevice(owner.value()) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }

  CUmemoryPool driver_pool = nullptr;
  auto const result        = cuPointerGetAttribute(
    &driver_pool, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, reinterpret_cast<CUdeviceptr>(allocation));
  if (result != CUDA_SUCCESS) { return false; }
  if (driver_pool == nullptr) { return true; }

  auto const pool = reinterpret_cast<cudaMemPool_t>(driver_pool);
  if (pool_has_read_write_access(pool, accessor)) { return true; }

  cudaMemAccessDesc access{};
  access.location.type = cudaMemLocationTypeDevice;
  access.location.id   = accessor.value();
  access.flags         = cudaMemAccessFlagsProtReadWrite;
  auto const status    = cudaMemPoolSetAccess(pool, &access, 1);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return pool_has_read_write_access(pool, accessor);
}

enum class peer_probe_state { supported, unsupported };

std::mutex& peer_probe_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::uint64_t, peer_probe_state>& peer_probe_cache()
{
  static std::unordered_map<std::uint64_t, peer_probe_state> cache;
  return cache;
}

std::optional<peer_probe_state> cached_peer_probe(std::uint64_t key) noexcept
{
  try {
    std::lock_guard<std::mutex> lock{peer_probe_mutex()};
    auto const& cache = peer_probe_cache();
    auto const it     = cache.find(key);
    return it == cache.end() ? std::nullopt : std::optional{it->second};
  } catch (...) {
    // The cache is only an optimization. Host allocation/locking failures must not terminate an
    // optional filter publication or prevent the correctness-first staging route.
    return std::nullopt;
  }
}

void cache_peer_probe(std::uint64_t key, peer_probe_state state) noexcept
{
  try {
    std::lock_guard<std::mutex> lock{peer_probe_mutex()};
    peer_probe_cache().insert_or_assign(key, state);
  } catch (...) {
    // A later publication may repeat the small empirical probe.
  }
}

std::uint64_t peer_probe_key(rmm::cuda_device_id source, rmm::cuda_device_id destination) noexcept
{
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(source.value())) << 32U) |
         static_cast<std::uint32_t>(destination.value());
}

bool enable_peer_access(rmm::cuda_device_id owner, rmm::cuda_device_id peer) noexcept
{
  if (cudaSetDevice(owner.value()) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  auto const status = cudaDeviceEnablePeerAccess(peer.value(), 0);
  if (status == cudaSuccess || status == cudaErrorPeerAccessAlreadyEnabled) {
    (void)cudaGetLastError();
    return true;
  }
  (void)cudaGetLastError();
  return false;
}

class peer_probe_resources final {
 public:
  peer_probe_resources(rmm::cuda_device_id source_device,
                       rmm::cuda_device_id destination_device) noexcept
    : _source_device{source_device}, _destination_device{destination_device}
  {
  }

  ~peer_probe_resources() noexcept
  {
    auto const destination_complete =
      release_device(_destination_device, destination, destination_stream);
    auto const source_complete = release_device(_source_device, source, source_stream);
    if (host != nullptr && destination_complete && source_complete) {
      (void)cudaFreeHost(host);
      (void)cudaGetLastError();
    }
    // If either stream could not be synchronized, deliberately leak the tiny pinned probe buffer.
    // Freeing it while an H2D/D2H operation may still reference it would be a use-after-free.
  }

  peer_probe_resources(peer_probe_resources const&)            = delete;
  peer_probe_resources& operator=(peer_probe_resources const&) = delete;

  void* source{nullptr};
  void* destination{nullptr};
  void* host{nullptr};
  cudaStream_t source_stream{nullptr};
  cudaStream_t destination_stream{nullptr};

 private:
  [[nodiscard]] static bool release_device(rmm::cuda_device_id device,
                                           void* allocation,
                                           cudaStream_t stream) noexcept
  {
    if (stream == nullptr) { return true; }
    if (cudaSetDevice(device.value()) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    if (allocation != nullptr) { (void)cudaFreeAsync(allocation, stream); }
    auto const synchronized = cudaStreamSynchronize(stream) == cudaSuccess;
    (void)cudaStreamDestroy(stream);
    (void)cudaGetLastError();
    return synchronized;
  }

  rmm::cuda_device_id _source_device;
  rmm::cuda_device_id _destination_device;
};

// Probe only the direction this filter needs. Private nonblocking streams and stream-ordered
// allocations avoid introducing a device-wide barrier into an opportunistic publication. The
// global mutex protects only cache lookup/commit; different pairs may probe concurrently.
// Transient CUDA/allocation/readback errors are not cached, so a later publication may retry.
bool peer_dma_moves_bytes(rmm::cuda_device_id source_device,
                          rmm::cuda_device_id destination_device) noexcept
{
  auto const key = peer_probe_key(source_device, destination_device);
  if (auto const cached = cached_peer_probe(key); cached.has_value()) {
    return *cached == peer_probe_state::supported;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || source_device.value() < 0 ||
      destination_device.value() < 0 || source_device.value() >= device_count ||
      destination_device.value() >= device_count) {
    (void)cudaGetLastError();
    return false;
  }

  int can_access = 0;
  auto const capability =
    cudaDeviceCanAccessPeer(&can_access, destination_device.value(), source_device.value());
  if (capability != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  if (can_access == 0) {
    cache_peer_probe(key, peer_probe_state::unsupported);
    return false;
  }

  current_device_restore restore;
  if (!restore.valid() || !enable_peer_access(destination_device, source_device)) { return false; }

  constexpr std::size_t probe_bytes = 64;
  constexpr std::size_t host_bytes  = 3 * probe_bytes;
  peer_probe_resources resources{source_device, destination_device};
  if (cudaHostAlloc(&resources.host, host_bytes, cudaHostAllocPortable) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  auto* const expected = static_cast<std::uint8_t*>(resources.host);
  auto* const sentinel = expected + probe_bytes;
  auto* const actual   = sentinel + probe_bytes;
  for (std::size_t i = 0; i < probe_bytes; ++i) {
    expected[i] = static_cast<std::uint8_t>(0x40U + (i & 0x3FU));
    sentinel[i] = 0xAAU;
    actual[i]   = 0U;
  }

  if (cudaSetDevice(source_device.value()) != cudaSuccess ||
      cudaStreamCreateWithFlags(&resources.source_stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaMallocAsync(&resources.source, probe_bytes, resources.source_stream) != cudaSuccess ||
      cudaMemcpyAsync(
        resources.source, expected, probe_bytes, cudaMemcpyHostToDevice, resources.source_stream) !=
        cudaSuccess ||
      cudaStreamSynchronize(resources.source_stream) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }

  if (cudaSetDevice(destination_device.value()) != cudaSuccess ||
      cudaStreamCreateWithFlags(&resources.destination_stream, cudaStreamNonBlocking) !=
        cudaSuccess ||
      cudaMallocAsync(&resources.destination, probe_bytes, resources.destination_stream) !=
        cudaSuccess ||
      cudaMemcpyAsync(resources.destination,
                      sentinel,
                      probe_bytes,
                      cudaMemcpyHostToDevice,
                      resources.destination_stream) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }

  // The copy executes on the destination stream, so only destination access to the source pool is
  // required. Destination storage is local to that stream's device.
  if (!grant_allocation_peer_access(resources.source, source_device, destination_device)) {
    return false;
  }

  if (cudaSetDevice(destination_device.value()) != cudaSuccess ||
      cudaMemcpyPeerAsync(resources.destination,
                          destination_device.value(),
                          resources.source,
                          source_device.value(),
                          probe_bytes,
                          resources.destination_stream) != cudaSuccess ||
      cudaMemcpyAsync(actual,
                      resources.destination,
                      probe_bytes,
                      cudaMemcpyDeviceToHost,
                      resources.destination_stream) != cudaSuccess ||
      cudaStreamSynchronize(resources.destination_stream) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }

  bool const copy_ok = std::memcmp(actual, expected, probe_bytes) == 0;
  // Reaching the comparison proves readback completed. A mismatch is therefore conclusive; all API
  // failures returned above remain uncached and retryable.
  cache_peer_probe(key, copy_ok ? peer_probe_state::supported : peer_probe_state::unsupported);
  return copy_ok;
}

}  // namespace

replica_transfer::replica_transfer(replica_transfer_route route,
                                   rmm::cuda_device_id destination_device,
                                   rmm::cuda_stream_view destination_stream,
                                   void* portable_staging) noexcept
  : _route{route},
    _destination_device{destination_device},
    _destination_stream{destination_stream},
    _portable_staging{portable_staging},
    _complete{route == replica_transfer_route::none}
{
}

replica_transfer::~replica_transfer() noexcept { wait_no_throw(); }

replica_transfer::replica_transfer(replica_transfer&& other) noexcept
  : _route{std::exchange(other._route, replica_transfer_route::none)},
    _destination_device{std::exchange(other._destination_device, rmm::cuda_device_id{-1})},
    _destination_stream{std::exchange(other._destination_stream, rmm::cuda_stream_view{})},
    _portable_staging{std::exchange(other._portable_staging, nullptr)},
    _complete{std::exchange(other._complete, true)}
{
}

replica_transfer& replica_transfer::operator=(replica_transfer&& other) noexcept
{
  if (this == &other) { return *this; }
  wait_no_throw();
  _route              = std::exchange(other._route, replica_transfer_route::none);
  _destination_device = std::exchange(other._destination_device, rmm::cuda_device_id{-1});
  _destination_stream = std::exchange(other._destination_stream, rmm::cuda_stream_view{});
  _portable_staging   = std::exchange(other._portable_staging, nullptr);
  _complete           = std::exchange(other._complete, true);
  return *this;
}

void replica_transfer::wait()
{
  if (!_complete) {
    rmm::cuda_set_device_raii guard{_destination_device};
    check_cuda(cudaStreamSynchronize(_destination_stream.value()),
               "cudaStreamSynchronize(dynamic-filter replica)");
    _complete = true;
  }
  if (_portable_staging != nullptr) {
    auto const status = cudaFreeHost(_portable_staging);
    if (status != cudaSuccess) { throw_cuda_error("cudaFreeHost(dynamic-filter staging)", status); }
    _portable_staging = nullptr;
  }
}

void replica_transfer::wait_no_throw() noexcept
{
  try {
    wait();
  } catch (...) {
    // Never free staging storage while a failed synchronization may have left an H2D copy in
    // flight. Leaking on an unrecoverable CUDA-context failure is safer than a host use-after-free.
  }
}

replica_transfer enqueue_replica_transfer(void* destination,
                                          rmm::cuda_device_id destination_device,
                                          void const* source,
                                          rmm::cuda_device_id source_device,
                                          std::size_t bytes,
                                          rmm::cuda_stream_view destination_stream,
                                          replica_transfer_policy policy)
{
  if (bytes == 0) { return {}; }
  if (destination == nullptr || source == nullptr) {
    throw std::invalid_argument(
      "enqueue_replica_transfer requires non-null pointers for a non-empty copy");
  }
  if (destination_device == source_device && policy == replica_transfer_policy::automatic) {
    rmm::cuda_set_device_raii guard{destination_device};
    check_cuda(cudaMemcpyAsync(
                 destination, source, bytes, cudaMemcpyDeviceToDevice, destination_stream.value()),
               "cudaMemcpyAsync(local dynamic-filter replica)");
    return {replica_transfer_route::local, destination_device, destination_stream, nullptr};
  }

  bool const peer_dma = policy == replica_transfer_policy::automatic &&
                        peer_dma_moves_bytes(source_device, destination_device) &&
                        grant_allocation_peer_access(source, source_device, destination_device);
  if (peer_dma) {
    rmm::cuda_set_device_raii guard{destination_device};
    auto const status = cudaMemcpyPeerAsync(destination,
                                            destination_device.value(),
                                            source,
                                            source_device.value(),
                                            bytes,
                                            destination_stream.value());
    if (status == cudaSuccess) {
      return {replica_transfer_route::peer_dma, destination_device, destination_stream, nullptr};
    }
    // An empirical probe and pool grant cannot make an individual enqueue infallible. Clear the
    // runtime's sticky error and continue through the explicit portable-host route instead of
    // dropping this optional replica.
    (void)cudaGetLastError();
  }

  // Source readiness is an explicit caller precondition (the publisher synchronizes the build
  // stream before replication). A blocking D2H leg then provides cross-device ordering without
  // depending on peer events.
  portable_pinned_buffer staging{bytes};
  {
    rmm::cuda_set_device_raii guard{source_device};
    check_cuda(cudaMemcpy(staging.get(), source, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(D2H dynamic-filter staging)");
  }
  {
    rmm::cuda_set_device_raii guard{destination_device};
    check_cuda(
      cudaMemcpyAsync(
        destination, staging.get(), bytes, cudaMemcpyHostToDevice, destination_stream.value()),
      "cudaMemcpyAsync(H2D dynamic-filter staging)");
  }
  return {replica_transfer_route::portable_host,
          destination_device,
          destination_stream,
          staging.release()};
}

}  // namespace sirius::op::detail
