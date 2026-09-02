/*
 * Copyright 2025, Sirius Contributors.
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

#include "exec/exchange_staging_arena.hpp"

#include "log/logging.hpp"
#include "sirius/exception.hpp"
#include "yaml_reader.hpp"  // sirius::yaml::parse_bytes

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>

namespace sirius::exec {

namespace {
constexpr std::uint64_t align_up(std::uint64_t len)
{
  return (len + exchange_staging_arena::kAlignment - 1) & ~(exchange_staging_arena::kAlignment - 1);
}

//! Driver-API error text, or a bare code when the driver cannot name it.
std::string driver_error(CUresult status)
{
  const char* name = nullptr;
  cuGetErrorString(status, &name);
  return name != nullptr ? std::string(name) : ("CUresult " + std::to_string(status));
}

//! True when the operator asked for the fabric-handle arena.
bool want_fabric_arena()
{
  const char* kind = std::getenv(exchange_staging_arena::kArenaKindEnvVar);
  if (kind == nullptr) { return false; }
  if (std::strcmp(kind, "fabric") == 0) { return true; }
  if (std::strcmp(kind, "cudamalloc") == 0) { return false; }
  throw sirius::invalid_input_exception(std::string(exchange_staging_arena::kArenaKindEnvVar) +
                                        ": expected \"fabric\" or \"cudamalloc\", got \"" + kind +
                                        "\"");
}
}  // namespace

exchange_staging_arena::exchange_staging_arena(std::uint64_t capacity_bytes)
  : capacity_(capacity_bytes)
{
  if (capacity_bytes == 0) {
    throw sirius::invalid_input_exception("exchange staging arena: capacity must be nonzero");
  }

  if (!want_fabric_arena()) {
    // Plain cudaMalloc, by contract (see the class comment): pool memory silently loses the
    // transport's GPU-to-GPU fast path. Correct for a single host; see kArenaKindEnvVar for why
    // it is not enough across hosts.
    if (auto err = cudaMalloc(&base_, capacity_bytes); err != cudaSuccess) {
      throw sirius::internal_exception("exchange staging arena: cudaMalloc of {} bytes failed: {}",
                                       capacity_bytes,
                                       cudaGetErrorString(err));
    }
    // The size the operator actually got, on the operator's own terms. This slab sits OUTSIDE
    // the RMM pool, so it appears in no config dump and in no pool accounting -- without this
    // line the only way to learn it is to read the launcher's environment.
    SIRIUS_LOG_INFO("exchange staging arena: {} bytes (cudaMalloc)", capacity_);
    // Seeded at the end of the constructor, after BOTH paths have settled `capacity_`.
    free_.emplace(0, capacity_);
    return;
  }

  // Fabric path (opt-in via SIRIUS_EXCHANGE_STAGING_ARENA=fabric; only needed when peers live on
  // another host; not exercised by the unit tests): reserve, back and map device memory whose
  // handle a peer on another host can import. This is the cuMemCreate / cuMemAddressReserve /
  // cuMemMap / cuMemSetAccess sequence a standalone two-host MNNVL harness validated at 765 GB/s.
  //
  // The device ordinal is 0 because a cross-host embedder pins its process to one GPU through
  // CUDA_VISIBLE_DEVICES before engine bring-up, so the process only ever sees one device, at
  // ordinal 0. A multi-device process would need the ordinal plumbed through instead.
  CUmemAllocationProp prop{};
  prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id          = 0;

  std::size_t granularity = 0;
  if (auto status =
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
      status != CUDA_SUCCESS) {
    throw sirius::internal_exception(
      "exchange staging arena: cuMemGetAllocationGranularity failed: {}", driver_error(status));
  }
  if (granularity == 0) {
    throw sirius::internal_exception("exchange staging arena: zero allocation granularity");
  }
  // cuMemCreate requires a granularity multiple. Rounding up grows the arena, never shrinks it,
  // so a lease that fit before still fits.
  const std::uint64_t size = ((capacity_bytes + granularity - 1) / granularity) * granularity;

  CUmemGenericAllocationHandle handle = 0;
  if (auto status = cuMemCreate(&handle, size, &prop, 0); status != CUDA_SUCCESS) {
    throw sirius::internal_exception(
      "exchange staging arena: cuMemCreate of {} bytes with CU_MEM_HANDLE_TYPE_FABRIC failed: {} "
      "-- this host cannot export fabric handles (is nvidia-imex running, and are the "
      "/dev/nvidia-caps-imex-channels devices accessible to this user?)",
      size,
      driver_error(status));
  }

  CUdeviceptr ptr = 0;
  if (auto status = cuMemAddressReserve(&ptr, size, granularity, 0, 0); status != CUDA_SUCCESS) {
    cuMemRelease(handle);
    throw sirius::internal_exception("exchange staging arena: cuMemAddressReserve failed: {}",
                                     driver_error(status));
  }
  if (auto status = cuMemMap(ptr, size, 0, handle, 0); status != CUDA_SUCCESS) {
    cuMemAddressFree(ptr, size);
    cuMemRelease(handle);
    throw sirius::internal_exception("exchange staging arena: cuMemMap failed: {}",
                                     driver_error(status));
  }

  CUmemAccessDesc access{};
  access.location = prop.location;
  access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (auto status = cuMemSetAccess(ptr, size, &access, 1); status != CUDA_SUCCESS) {
    cuMemUnmap(ptr, size);
    cuMemAddressFree(ptr, size);
    cuMemRelease(handle);
    throw sirius::internal_exception("exchange staging arena: cuMemSetAccess failed: {}",
                                     driver_error(status));
  }

  base_          = reinterpret_cast<void*>(ptr);
  capacity_      = size;
  mapped_bytes_  = size;
  fabric_handle_ = handle;
  // Both the requested and the granted size: cuMemCreate rounds up to a granularity multiple, so
  // these differ, and the granted one is what a lease is actually checked against.
  SIRIUS_LOG_INFO(
    "exchange staging arena: {} bytes (fabric handle, requested {})", capacity_, capacity_bytes);
  // Seeded only now: the fabric path OVERWRITES `capacity_` with the granularity-rounded size
  // (above), and the free list must describe the region actually mapped, not the one requested.
  free_.emplace(0, capacity_);
}

exchange_staging_arena::~exchange_staging_arena()
{
  // The one number that says how much arena a workload ACTUALLY needed. Sizing this slab is the
  // hardest knob to set (it depends on how many exports are in flight at once, not on data
  // volume) and the arena fails hard rather than degrading, so without this line the only
  // feedback an operator gets is "exhausted" or silence -- there is no way to learn a passing
  // run had 90% headroom. Nonzero `outstanding` at teardown means a leaked lease.
  {
    std::lock_guard lock(mutex_);
    SIRIUS_LOG_INFO(
      "exchange staging arena: peak live {} of {} bytes ({} leases outstanding, {} free blocks, "
      "largest {})",
      peak_live_bytes_,
      capacity_,
      leases_.size(),
      free_.size(),
      largest_free_locked());
  }

  if (fabric_handle_ != 0) {
    // Unmap before freeing the reservation, and release the physical handle last -- cudaFree
    // cannot release a VMM mapping and would leak the entire arena.
    auto ptr = reinterpret_cast<CUdeviceptr>(base_);
    cuMemUnmap(ptr, mapped_bytes_);
    cuMemAddressFree(ptr, mapped_bytes_);
    cuMemRelease(fabric_handle_);
    return;
  }
  cudaFree(base_);
}

std::unique_ptr<exchange_staging_arena> exchange_staging_arena::from_env()
{
  const char* value = std::getenv(kCapacityEnvVar);
  if (value == nullptr) { return nullptr; }
  std::uint64_t bytes = 0;
  try {
    bytes = sirius::yaml::parse_bytes(value);
  } catch (const std::exception& e) {
    throw sirius::invalid_input_exception(std::string(kCapacityEnvVar) + ": " + e.what());
  }
  return std::make_unique<exchange_staging_arena>(bytes);
}

exchange_staging_arena& exchange_staging_arena::require(exchange_staging_arena* arena)
{
  if (arena == nullptr) {
    throw sirius::invalid_input_exception(
      "exchange staging arena not configured (set SIRIUS_EXCHANGE_STAGING_BYTES)");
  }
  return *arena;
}

std::uint64_t exchange_staging_arena::lease(std::uint64_t len)
{
  if (len == 0) {
    // A zero-length lease would alias the next lease's offset and break release-by-offset.
    throw sirius::invalid_input_exception("exchange staging arena: zero-length lease");
  }
  std::lock_guard lock(mutex_);
  const auto aligned = align_up(len);
  // align_up wraps for len within kAlignment-1 of UINT64_MAX; a wrapped 0 would slip past the
  // fit scan and register a zero-length lease aliasing a live one. `len` will be wire-supplied
  // by a peer's lease request once a transport sits on top, so this guard is load-bearing, not
  // theoretical.
  if (aligned < len || aligned > capacity_) {
    throw sirius::invalid_input_exception(
      "exchange staging arena: lease of {} bytes exceeds the {} byte capacity", len, capacity_);
  }

  // Address-ordered first fit: keeps low addresses dense and needs no second index. At the tens
  // of blocks this arena holds, the linear scan is cheaper than maintaining a size index.
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    if (it->second < aligned) { continue; }
    const auto offset    = it->first;
    const auto block_len = it->second;
    free_.erase(it);
    if (block_len > aligned) { free_.emplace(offset + aligned, block_len - aligned); }
    leases_.emplace(offset, aligned);
    live_bytes_ += aligned;
    peak_live_bytes_ = std::max(peak_live_bytes_, live_bytes_);
    return offset;
  }

  // Both numbers, because they mean different things: total free short of the request means
  // raise capacity (or fix retention); total free ample but largest block short means external
  // fragmentation, which a bigger arena does not necessarily fix.
  throw sirius::invalid_input_exception(
    "exchange staging arena exhausted: requested {} bytes ({} aligned), {} free of {} capacity "
    "in {} blocks (largest {}), {} leases outstanding holding {} bytes "
    "(raise SIRIUS_EXCHANGE_STAGING_BYTES)",
    len,
    aligned,
    total_free_locked(),
    capacity_,
    free_.size(),
    largest_free_locked(),
    leases_.size(),
    live_bytes_);
}

void exchange_staging_arena::release(std::uint64_t offset)
{
  std::lock_guard lock(mutex_);
  auto it = leases_.find(offset);
  if (it == leases_.end()) {
    throw sirius::invalid_input_exception(
      "exchange staging arena: release of offset {} which is not an outstanding lease "
      "(double release?)",
      offset);
  }
  const auto len = it->second;
  leases_.erase(it);
  live_bytes_ -= len;

  // Insert and coalesce with both neighbours, so the free list never holds two adjacent blocks
  // and released space is reusable regardless of the order releases arrive in. Merge forward
  // first (this block absorbs its successor), then backward (the predecessor absorbs the
  // result) -- doing it in the other order would leave `ins` dangling before the forward merge.
  auto [ins, ok] = free_.emplace(offset, len);
  (void)ok;  // offset came out of leases_, so it cannot already be in free_

  auto next = std::next(ins);
  if (next != free_.end() && ins->first + ins->second == next->first) {
    ins->second += next->second;
    free_.erase(next);
  }
  if (ins != free_.begin()) {
    auto prev = std::prev(ins);
    if (prev->first + prev->second == ins->first) {
      prev->second += ins->second;
      free_.erase(ins);
    }
  }
}

std::uint64_t exchange_staging_arena::total_free_locked() const
{
  std::uint64_t sum = 0;
  for (const auto& [offset, len] : free_) {
    sum += len;
  }
  return sum;
}

std::uint64_t exchange_staging_arena::largest_free_locked() const
{
  std::uint64_t best = 0;
  for (const auto& [offset, len] : free_) {
    best = std::max(best, len);
  }
  return best;
}

std::size_t exchange_staging_arena::outstanding() const
{
  std::lock_guard lock(mutex_);
  return leases_.size();
}

std::uint64_t exchange_staging_arena::total_free() const
{
  std::lock_guard lock(mutex_);
  return total_free_locked();
}

std::uint64_t exchange_staging_arena::largest_free() const
{
  std::lock_guard lock(mutex_);
  return largest_free_locked();
}

std::uint64_t exchange_staging_arena::live_bytes() const
{
  std::lock_guard lock(mutex_);
  return live_bytes_;
}

std::uint64_t exchange_staging_arena::peak_live_bytes() const
{
  std::lock_guard lock(mutex_);
  return peak_live_bytes_;
}

}  // namespace sirius::exec
