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

#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/nvtx/ranges.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <numa.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace sirius {
namespace memory {

class numa_region_pinned_host_memory_resource final : public rmm::mr::device_memory_resource {
 public:
  explicit numa_region_pinned_host_memory_resource(int numa_node) : numa_node_(numa_node) {}
  ~numa_region_pinned_host_memory_resource()                                              = default;
  numa_region_pinned_host_memory_resource(numa_region_pinned_host_memory_resource const&) = default;
  numa_region_pinned_host_memory_resource(numa_region_pinned_host_memory_resource&&)      = default;
  numa_region_pinned_host_memory_resource& operator=(
    numa_region_pinned_host_memory_resource const&) = default;
  numa_region_pinned_host_memory_resource& operator=(numa_region_pinned_host_memory_resource&&) =
    default;

 private:
  /**
   * @brief Allocates pinned host memory of size at least \p bytes bytes.
   *
   * @throws rmm::out_of_memory if the requested allocation could not be fulfilled due to a
   * CUDA out of memory error.
   * @throws rmm::bad_alloc if the requested allocation could not be fulfilled due to any other
   * reason.
   *
   * The stream argument is ignored.
   *
   * @param bytes The size, in bytes, of the allocation.
   * @param stream CUDA stream on which to perform the allocation (ignored).
   *
   * @return Pointer to the newly allocated memory.
   */
  void* do_allocate(std::size_t bytes, [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    RMM_FUNC_RANGE();
    // don't allocate anything if the user requested zero bytes
    if (0 == bytes) { return nullptr; }

    if (numa_node_ == -1) {
      void* ptr{nullptr};
      RMM_CUDA_TRY_ALLOC(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), bytes);
      return ptr;
    } else {
      void* ptr = numa_alloc_onnode(bytes, numa_node_);
      if (ptr == nullptr) { throw rmm::bad_alloc(std::strerror(errno)); }
      RMM_CUDA_TRY_ALLOC(cudaHostRegister(ptr, bytes, cudaHostRegisterMapped), bytes);
      return ptr;
    }
  }

  /**
   * @brief Deallocate memory pointed to by \p ptr.
   *
   * The stream argument is ignored.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `ptr`.
   * @param stream This argument is ignored.
   */
  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     [[maybe_unused]] rmm::cuda_stream_view stream) noexcept override
  {
    RMM_FUNC_RANGE();
    if (numa_node_ == -1) {
      RMM_ASSERT_CUDA_SUCCESS(cudaFreeHost(ptr));
    } else {
      cudaHostUnregister(ptr);
      numa_free(ptr, bytes);
    }
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two pinned_host_memory_resources always compare equal, because they can each
   * deallocate memory allocated by the other.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(
    const rmm::mr::device_memory_resource& other) const noexcept override
  {
    auto* mr_ptr = dynamic_cast<numa_region_pinned_host_memory_resource const*>(&other);
    return mr_ptr == this && mr_ptr->numa_node_ == this->numa_node_;
  }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `pinned_host_memory_resource` provides device accessible memory
   */
  friend void get_property(numa_region_pinned_host_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   *
   * This property declares that a `pinned_host_memory_resource` provides host accessible memory
   */
  friend void get_property(numa_region_pinned_host_memory_resource const&,
                           cuda::mr::host_accessible) noexcept
  {
  }

  int numa_node_{-1};
};

}  // namespace memory
}  // namespace sirius
