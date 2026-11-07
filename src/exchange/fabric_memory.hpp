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

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace sirius::exchange {

/**
 * @brief Device memory with a fabric handle (`CU_MEM_HANDLE_TYPE_FABRIC`), which a process on
 * another node of the same NVLink domain can map.
 *
 * UCX's `cuda_ipc` reaches another node only through a fabric handle: plain `cudaMalloc` memory
 * has only a legacy IPC handle, and UCX then moves it over the network instead, without a word.
 * The import on the other node goes through the IMEX daemon.
 *
 * The driver API is looked up through the runtime, so Sirius does not link libcuda.
 */
class fabric_memory {
 public:
  /// @p bytes rounded up to the allocation granularity, on @p device; nullopt when the device
  /// has no fabric handles or the driver refuses one (typically: no IMEX channel), with the
  /// reason in @p why.
  [[nodiscard]] static std::optional<fabric_memory> try_allocate(int device,
                                                                 std::size_t bytes,
                                                                 std::string& why);

  fabric_memory(fabric_memory&& other) noexcept;
  fabric_memory& operator=(fabric_memory&& other) noexcept;
  fabric_memory(fabric_memory const&)            = delete;
  fabric_memory& operator=(fabric_memory const&) = delete;
  ~fabric_memory();

  [[nodiscard]] std::byte* base() const noexcept { return _base; }
  /// The bytes mapped: at least what was asked for.
  [[nodiscard]] std::size_t bytes() const noexcept { return _bytes; }

 private:
  fabric_memory(std::byte* base, std::size_t bytes, std::uint64_t handle) noexcept
    : _base(base), _bytes(bytes), _handle(handle)
  {
  }
  void release() noexcept;

  std::byte* _base{nullptr};
  std::size_t _bytes{0};
  std::uint64_t _handle{0};
};

/// The NVLink domain of a GPU, from NVML's fabric info: GPUs with equal domains reach each
/// other's fabric memory over NVLink, across nodes.
struct nvlink_domain {
  std::array<std::uint8_t, 16> cluster_uuid{};
  std::uint32_t clique_id{0};

  /// Whether the GPU's fabric registration completed; an invalid domain matches none.
  [[nodiscard]] bool valid() const noexcept
  {
    for (auto const b : cluster_uuid) {
      if (b != 0) { return true; }
    }
    return false;
  }
  [[nodiscard]] bool same_as(nvlink_domain const& other) const noexcept
  {
    return valid() && cluster_uuid == other.cluster_uuid && clique_id == other.clique_id;
  }
  /// "<cluster uuid>/<clique id>", or "none".
  [[nodiscard]] std::string to_string() const;
};

}  // namespace sirius::exchange
