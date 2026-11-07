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

#include "exchange/fabric_memory.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <utility>

namespace sirius::exchange {
namespace {

/// The driver functions fabric memory needs, looked up once through the runtime.
struct driver {
  decltype(&cuDeviceGetAttribute) device_attribute{nullptr};
  decltype(&cuMemGetAllocationGranularity) granularity{nullptr};
  decltype(&cuMemCreate) create{nullptr};
  decltype(&cuMemRelease) release{nullptr};
  decltype(&cuMemAddressReserve) reserve{nullptr};
  decltype(&cuMemAddressFree) address_free{nullptr};
  decltype(&cuMemMap) map{nullptr};
  decltype(&cuMemUnmap) unmap{nullptr};
  decltype(&cuMemSetAccess) set_access{nullptr};
  decltype(&cuGetErrorName) error_name{nullptr};
  bool ok{false};
};

template <typename F>
bool lookup(char const* symbol, F& out)
{
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult found{};
  if (cudaGetDriverEntryPointByVersion(symbol, &fn, 12000, cudaEnableDefault, &found) !=
        cudaSuccess ||
      found != cudaDriverEntryPointSuccess || fn == nullptr) {
    return false;
  }
  out = reinterpret_cast<F>(fn);
  return true;
}

driver const& drv()
{
  static driver const d = [] {
    driver d;
    d.ok = lookup("cuDeviceGetAttribute", d.device_attribute) &&
           lookup("cuMemGetAllocationGranularity", d.granularity) &&
           lookup("cuMemCreate", d.create) && lookup("cuMemRelease", d.release) &&
           lookup("cuMemAddressReserve", d.reserve) && lookup("cuMemAddressFree", d.address_free) &&
           lookup("cuMemMap", d.map) && lookup("cuMemUnmap", d.unmap) &&
           lookup("cuMemSetAccess", d.set_access) && lookup("cuGetErrorName", d.error_name);
    return d;
  }();
  return d;
}

std::string name_of(CUresult r)
{
  char const* name = nullptr;
  if (drv().error_name == nullptr || drv().error_name(r, &name) != CUDA_SUCCESS ||
      name == nullptr) {
    return "CUresult " + std::to_string(static_cast<int>(r));
  }
  return name;
}

}  // namespace

std::optional<fabric_memory> fabric_memory::try_allocate(int device,
                                                         std::size_t bytes,
                                                         std::string& why)
{
  auto const& d = drv();
  if (!d.ok) {
    why = "the CUDA driver lacks the virtual memory API";
    return std::nullopt;
  }
  int supported = 0;
  if (auto const r =
        d.device_attribute(&supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
      r != CUDA_SUCCESS || supported == 0) {
    why = "GPU " + std::to_string(device) + " has no fabric handles";
    return std::nullopt;
  }
  CUmemAllocationProp prop{};
  prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id          = device;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  std::size_t granularity   = 0;
  if (auto const r = d.granularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
      r != CUDA_SUCCESS || granularity == 0) {
    why = "cuMemGetAllocationGranularity: " + name_of(r);
    return std::nullopt;
  }
  bytes = (bytes + granularity - 1) / granularity * granularity;
  CUmemGenericAllocationHandle handle{};
  // CUDA_ERROR_NOT_PERMITTED or NOT_SUPPORTED here: this process has no IMEX channel.
  if (auto const r = d.create(&handle, bytes, &prop, 0); r != CUDA_SUCCESS) {
    why = "cuMemCreate: " + name_of(r) + " (is an IMEX channel open?)";
    return std::nullopt;
  }
  CUdeviceptr va{};
  if (auto const r = d.reserve(&va, bytes, granularity, 0, 0); r != CUDA_SUCCESS) {
    d.release(handle);
    why = "cuMemAddressReserve: " + name_of(r);
    return std::nullopt;
  }
  if (auto const r = d.map(va, bytes, 0, handle, 0); r != CUDA_SUCCESS) {
    d.address_free(va, bytes);
    d.release(handle);
    why = "cuMemMap: " + name_of(r);
    return std::nullopt;
  }
  CUmemAccessDesc access{};
  access.location = prop.location;
  access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (auto const r = d.set_access(va, bytes, &access, 1); r != CUDA_SUCCESS) {
    d.unmap(va, bytes);
    d.address_free(va, bytes);
    d.release(handle);
    why = "cuMemSetAccess: " + name_of(r);
    return std::nullopt;
  }
  return fabric_memory{reinterpret_cast<std::byte*>(va), bytes, static_cast<std::uint64_t>(handle)};
}

fabric_memory::fabric_memory(fabric_memory&& other) noexcept
  : _base(std::exchange(other._base, nullptr)),
    _bytes(std::exchange(other._bytes, 0)),
    _handle(std::exchange(other._handle, 0))
{
}

fabric_memory& fabric_memory::operator=(fabric_memory&& other) noexcept
{
  if (this != &other) {
    release();
    _base   = std::exchange(other._base, nullptr);
    _bytes  = std::exchange(other._bytes, 0);
    _handle = std::exchange(other._handle, 0);
  }
  return *this;
}

fabric_memory::~fabric_memory() { release(); }

void fabric_memory::release() noexcept
{
  if (_base == nullptr) { return; }
  auto const& d = drv();
  auto const va = reinterpret_cast<CUdeviceptr>(_base);
  static_cast<void>(d.unmap(va, _bytes));
  static_cast<void>(d.address_free(va, _bytes));
  static_cast<void>(d.release(static_cast<CUmemGenericAllocationHandle>(_handle)));
  _base = nullptr;
}

std::string nvlink_domain::to_string() const
{
  if (!valid()) { return "none"; }
  std::string out;
  char hex[3];
  for (std::size_t i = 0; i < cluster_uuid.size(); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) { out += '-'; }
    std::snprintf(hex, sizeof(hex), "%02x", cluster_uuid[i]);
    out += hex;
  }
  return out + "/" + std::to_string(clique_id);
}

}  // namespace sirius::exchange
