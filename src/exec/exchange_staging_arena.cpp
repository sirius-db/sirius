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

#include "sirius/exception.hpp"
#include "yaml_reader.hpp"  // sirius::yaml::parse_bytes

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

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
  throw sirius::invalid_input_exception(
    std::string(exchange_staging_arena::kArenaKindEnvVar) +
    ": expected \"fabric\" or \"cudamalloc\", got \"" + kind + "\"");
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
    return;
  }

  // Fabric path: reserve, back and map device memory whose handle a peer on another host can
  // import. Mirrors the sequence proven by the two-node harness
  // (experimental/starrocks/src/nixl_transport/two_node_harness.rs, cuda_vmm::alloc_fabric).
  //
  // The device ordinal is 0 because the CN pins itself to one GPU with --gpu-device, which sets
  // CUDA_VISIBLE_DEVICES before engine bring-up -- so the process only ever sees one device, at
  // ordinal 0. That is the same assumption the harness and NixlDescriptor::device_id() make.
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
}

exchange_staging_arena::~exchange_staging_arena()
{
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
  const auto free    = capacity_ - head_;
  if (aligned > free) {
    throw sirius::invalid_input_exception(
      "exchange staging arena exhausted: requested {} bytes ({} aligned), {} free of {} capacity "
      "with {} leases outstanding (raise SIRIUS_EXCHANGE_STAGING_BYTES)",
      len,
      aligned,
      free,
      capacity_,
      leases_.size());
  }
  const auto offset = head_;
  head_ += aligned;
  high_water_ = std::max(high_water_, head_);
  leases_.emplace(offset, aligned);
  return offset;
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
  leases_.erase(it);
  // Trailing reclamation, no free list: the bump head drops back to the end of the highest
  // lease still outstanding (to the base when none remain). Gaps below the head are only
  // reusable once everything above them goes back — but a lease that outlives its neighbours
  // now pins at most the region up to its own end, never the whole arena, so steady-state
  // traffic keeps reusing the same space above it instead of burning arena for the process
  // lifetime.
  if (leases_.empty()) {
    head_ = 0;
  } else {
    const auto& highest = *leases_.rbegin();
    head_               = highest.first + highest.second;
  }
}

std::size_t exchange_staging_arena::outstanding() const
{
  std::lock_guard lock(mutex_);
  return leases_.size();
}

std::uint64_t exchange_staging_arena::high_water() const
{
  std::lock_guard lock(mutex_);
  return high_water_;
}

}  // namespace sirius::exec
