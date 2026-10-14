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

#include <cuda/memory_resource>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <rmm/error.hpp>

#include <sys/mman.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numa.h>
#include <string>
#include <thread>
#include <vector>

namespace sirius::memory {

/// @brief Upstream for the host tier's pools: pages the process touches itself and then
/// registers with CUDA, instead of cudaHostAlloc.
///
/// cudaHostAlloc faults and pins a pool one 4 KiB page at a time on the calling thread, and
/// concurrent calls do not overlap, so a statement that fills the host tier for the first time
/// waits about 0.66 s per GiB for memory alone (measured on an RTX A5000 host). Touching the
/// pages on several threads and registering the range afterwards yields the same pinned,
/// device-mapped memory in about 0.14 s per GiB, and 0.04 s per GiB where the kernel grants
/// transparent huge pages.
///
/// The memory is not counted here: cucascade's fixed_size_host_memory_resource, which draws its
/// pools from this resource, owns capacity and reservations.
class registered_host_memory_resource final {
 public:
  static constexpr std::size_t huge_page_bytes = std::size_t{2} << 20;

  /// @param numa_node      node the pages are bound to, or -1 to leave placement to the kernel.
  /// @param huge_pages     ask for transparent huge pages; a kernel that declines falls back to
  ///                       4 KiB pages without an error.
  /// @param touch_threads  threads that fault the pages in; 0 is treated as 1.
  registered_host_memory_resource(int numa_node, bool huge_pages, unsigned touch_threads) noexcept
    : _numa_node(numa_node), _huge_pages(huge_pages), _touch_threads(std::max(1U, touch_threads))
  {
  }

  void* allocate([[maybe_unused]] ::cuda::stream_ref stream,
                 std::size_t bytes,
                 [[maybe_unused]] std::size_t alignment = alignof(std::max_align_t))
  {
    if (bytes == 0) { return nullptr; }
    auto const size = mapped_size(bytes);
    // A huge page can only back a range that starts on a huge-page boundary, and mmap aligns to
    // 4 KiB, so map one huge page more and give the slack on both sides back.
    auto const reserved = size + huge_page_bytes;
    void* raw = ::mmap(nullptr, reserved, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
      throw rmm::out_of_memory("registered_host_memory_resource: mmap of " +
                               std::to_string(reserved) + " bytes failed");
    }
    auto const raw_addr = reinterpret_cast<std::uintptr_t>(raw);
    auto const addr     = (raw_addr + huge_page_bytes - 1) & ~(huge_page_bytes - 1);
    auto* base          = reinterpret_cast<std::byte*>(addr);
    if (addr != raw_addr) { ::munmap(raw, addr - raw_addr); }
    if (auto const tail = raw_addr + reserved - (addr + size); tail != 0) {
      ::munmap(base + size, tail);
    }

    if (_numa_node >= 0 && numa_available() >= 0) { numa_tonode_memory(base, size, _numa_node); }
    if (_huge_pages) { ::madvise(base, size, MADV_HUGEPAGE); }
    touch(base, size);

    // Portable as well as mapped: a pool may be read by tasks bound to any of the engine's GPUs.
    auto const status =
      cudaHostRegister(base, size, cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (status != cudaSuccess) {
      ::munmap(base, size);
      throw rmm::out_of_memory(std::string{"registered_host_memory_resource: cudaHostRegister: "} +
                               cudaGetErrorString(status));
    }
    return base;
  }

  void deallocate([[maybe_unused]] ::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  [[maybe_unused]] std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    if (ptr == nullptr) { return; }
    cudaHostUnregister(ptr);
    ::munmap(ptr, mapped_size(bytes));
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  [[nodiscard]] bool operator==(registered_host_memory_resource const& other) const noexcept
  {
    return _numa_node == other._numa_node && _huge_pages == other._huge_pages &&
           _touch_threads == other._touch_threads;
  }

  friend void get_property(registered_host_memory_resource const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }
  friend void get_property(registered_host_memory_resource const&,
                           ::cuda::mr::host_accessible) noexcept
  {
  }

 private:
  [[nodiscard]] static std::size_t mapped_size(std::size_t bytes) noexcept
  {
    return (bytes + huge_page_bytes - 1) & ~(huge_page_bytes - 1);
  }

  /// One write per 4 KiB page, so the range is resident whether or not huge pages were granted.
  void touch(std::byte* base, std::size_t size) const
  {
    constexpr std::size_t page = 4096;
    // A thread is not worth starting for less than a few huge pages of work.
    auto const workers = static_cast<unsigned>(
      std::min<std::size_t>(_touch_threads, std::max<std::size_t>(1, size / (16 * huge_page_bytes))));
    auto const span = mapped_size(size / workers);
    auto fault      = [base, size, span](unsigned worker) {
      auto const begin = std::min(size, static_cast<std::size_t>(worker) * span);
      auto const end   = std::min(size, begin + span);
      for (auto offset = begin; offset < end; offset += page) {
        base[offset] = std::byte{0};
      }
    };
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    for (unsigned worker = 1; worker < workers; ++worker) {
      threads.emplace_back(fault, worker);
    }
    fault(0);
    for (auto& thread : threads) {
      thread.join();
    }
  }

  int _numa_node;
  bool _huge_pages;
  unsigned _touch_threads;
};

}  // namespace sirius::memory
