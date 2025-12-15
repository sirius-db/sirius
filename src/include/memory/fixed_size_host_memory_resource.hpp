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

#include "memory/common.hpp"
#include "memory/memory_reservation.hpp"
#include "memory/notification_channel.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/nvtx/ranges.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius {
namespace memory {

// TODO: this doesn't handle multiple numa domains yet. We need to make our own
// pinned allocator that allocates numa local memory and then registers it with cuda

/**
 * @brief A host memory resource that allocates fixed-size blocks using pinned host memory as
 * upstream.
 *
 * This memory resource pre-allocates a pool of fixed-size blocks from the pinned host memory
 * resource and manages them in a free list. Allocations are limited to the configured block size.
 *
 * The pool is allocated as a single large allocation from the upstream resource and then split
 * into individual blocks for efficient memory management and reduced allocation overhead.
 *
 * When the pool is exhausted, it automatically expands by allocating additional blocks from
 * the upstream resource, making it suitable for workloads with varying memory requirements.
 *
 * Based on the implementation from:
 * https://github.com/felipeblazing/memory_spilling/blob/main/include/fixed_size_host_memory_resource.hpp
 * Modified to derive from device_memory_resource instead of host_memory_resource for RMM
 * compatibility.
 */
class fixed_size_host_memory_resource : public rmm::mr::device_memory_resource {
 public:
  static constexpr std::size_t default_block_size = 1 << 20;  ///< Default block size (1MB)
  static constexpr std::size_t default_pool_size  = 128;      ///< Default number of blocks in pool
  static constexpr std::size_t default_initial_number_pools =
    4;  ///< Default number of pools to pre-allocate

  struct chunked_reserved_area : public reserved_arena {
    explicit chunked_reserved_area(fixed_size_host_memory_resource& mr,
                                   std::size_t bytes,
                                   std::unique_ptr<event_notifier> notify_on_exit)
      : reserved_arena(bytes, std::move(notify_on_exit)), mr_(&mr), uuid_(create_uid())
    {
    }

    ~chunked_reserved_area() noexcept { mr_->release_reservation(this); }

    std::size_t uuid() const noexcept { return uuid_; }

    bool grow_by(std::size_t additional_bytes) final
    {
      return mr_->grow_reservation_by(*this, additional_bytes);
    }

    void shrink_to_fit() final { mr_->shrink_reservation_to_fit(*this); }

   private:
    static std::size_t create_uid()
    {
      static std::atomic<std::size_t> uid_counter{0};
      return uid_counter.fetch_add(1) + 1;
    }

    fixed_size_host_memory_resource* mr_;
    const std::size_t uuid_;
  };

  /**
   * @brief Simple RAII wrapper for multiple block allocations.
   */
  struct multiple_blocks_allocation {
    const std::size_t block_size;

    static std::unique_ptr<multiple_blocks_allocation> empty()
    {
      return std::unique_ptr<multiple_blocks_allocation>(
        new multiple_blocks_allocation({}, nullptr, nullptr));
    }

    static std::unique_ptr<multiple_blocks_allocation> create(std::vector<std::byte*> b,
                                                              fixed_size_host_memory_resource& m,
                                                              reservation* res = nullptr)
    {
      return std::unique_ptr<multiple_blocks_allocation>(
        new multiple_blocks_allocation(std::move(b), std::addressof(m), res));
    }

    ~multiple_blocks_allocation()
    {
      if (mr_ && !blocks_.empty()) {
        mr_->return_allocated_chunks(std::move(blocks_), reseved_memory_);
      }
    }

    // Disable copy to prevent double deallocation
    multiple_blocks_allocation(const multiple_blocks_allocation&)            = delete;
    multiple_blocks_allocation& operator=(const multiple_blocks_allocation&) = delete;
    multiple_blocks_allocation(multiple_blocks_allocation&&)                 = delete;
    multiple_blocks_allocation& operator=(multiple_blocks_allocation&&)      = delete;

    std::size_t size_bytes() const noexcept { return blocks_.size() * block_size; }

    std::size_t size() const noexcept { return blocks_.size(); }

    std::span<std::byte*> get_blocks() noexcept { return blocks_; }

    std::span<std::byte> operator[](std::size_t i) const
    {
      return std::span<std::byte>{blocks_.at(i), block_size};
    }

    std::span<std::byte> at(std::size_t i) const
    {
      return std::span<std::byte>{blocks_.at(i), block_size};
    }

   private:
    explicit multiple_blocks_allocation(std::vector<std::byte*> buffers,
                                        fixed_size_host_memory_resource* m,
                                        reservation* res)
      : blocks_(std::move(buffers)), mr_(m), block_size(m->get_block_size())
    {
      if (res) {
        auto* h_res = dynamic_cast<chunked_reserved_area*>(res->arena_.get());
        if (!h_res)
          throw std::invalid_argument("need host reservation for allocation multiple blocks");
        reseved_memory_ = h_res;
      }
    }

    std::vector<std::byte*> blocks_;
    fixed_size_host_memory_resource* mr_;
    chunked_reserved_area* reseved_memory_{nullptr};
  };

  using fixed_multiple_blocks_allocation = std::unique_ptr<multiple_blocks_allocation>;

  /**
   * @brief Construct with custom upstream resource.
   *
   * @param upstream_mr Upstream memory resource to use
   * @param block_size Size of each block in bytes
   * @param pool_size Number of blocks to pre-allocate
   * @param initial_pools Number of pools to pre-allocate
   */
  explicit fixed_size_host_memory_resource(
    int device_id,
    rmm::mr::device_memory_resource& upstream_mr,
    std::size_t mem_limit,
    std::size_t capacity,
    std::size_t block_size    = default_block_size,
    std::size_t pool_size     = default_pool_size,
    std::size_t initial_pools = default_initial_number_pools);

  // Disable copy and move
  fixed_size_host_memory_resource(const fixed_size_host_memory_resource&)            = delete;
  fixed_size_host_memory_resource& operator=(const fixed_size_host_memory_resource&) = delete;
  fixed_size_host_memory_resource(fixed_size_host_memory_resource&&)                 = delete;
  fixed_size_host_memory_resource& operator=(fixed_size_host_memory_resource&&)      = delete;

  /**
   * @brief Destructor - frees all allocated blocks.
   */
  ~fixed_size_host_memory_resource() override;

  [[nodiscard]] int get_total_allocated_bytes() const noexcept { return allocated_bytes_.load(); }

  /**
   * @brief return the total available memory in the memory resource
   */
  [[nodiscard]] std::size_t get_available_memory() const noexcept;

  /**
   * @brief Get the block size.
   *
   * @return std::size_t The size of each block in bytes
   */
  [[nodiscard]] std::size_t get_block_size() const noexcept;

  /**
   * @brief Get the number of free blocks.
   *
   * @return std::size_t Number of available blocks
   */
  [[nodiscard]] std::size_t get_free_blocks() const noexcept;

  /**
   * @brief Get the total number of blocks in the pool.
   *
   * @return std::size_t Total number of blocks
   */
  [[nodiscard]] std::size_t get_total_blocks() const noexcept;

  /**
   * @brief Get the upstream memory resource.
   *
   * @return rmm::mr::host_memory_resource* Pointer to upstream resource (nullptr if using pinned
   * host)
   */
  [[nodiscard]] rmm::mr::device_memory_resource* get_upstream_resource() const noexcept;

  /**
   * @brief Get total reserved bytes.
   * @return return total reserved bytes in the fixed memory resource
   */
  [[nodiscard]] std::size_t get_total_reserved_bytes() const noexcept;

  /**
   * @brief makes reservations
   * @param bytes the size of reservation
   * @param on_release used to hook callbacks for when the reservation is released
   */
  std::unique_ptr<reserved_arena> reserve(std::size_t bytes,
                                          std::unique_ptr<event_notifier> notifer = nullptr);

  /**
   * @brief makes reservations upto the given size
   * @param bytes the size of reservation
   * @param on_release used to hook callbacks for when the reservation is released
   */
  std::unique_ptr<reserved_arena> reserve_upto(std::size_t bytes,
                                               std::unique_ptr<event_notifier> notifer = nullptr);

  /**
   * @brief the number of active reservation
   */
  std::size_t get_active_reservation_count() const noexcept;

  /**
   * @brief Allocate multiple blocks to satisfy a large allocation request.
   *
   * This method allocates the minimum number of blocks needed to satisfy the requested size.
   * The blocks are returned as a RAII wrapper that automatically deallocates all blocks
   * when it goes out of scope, preventing memory leaks.
   *
   * @param total_bytes Total size in bytes to allocate across multiple blocks
   * @return multiple_blocks_allocation RAII wrapper for the allocated blocks
   * @throws rmm::out_of_memory if insufficient blocks are available or upstream allocation fails
   */
  [[nodiscard]] fixed_multiple_blocks_allocation allocate_multiple_blocks(
    std::size_t total_bytes, reservation* res = nullptr);

  /**
   * @brief Gets the peak total allocated bytes across all streams.
   * @return The peak total allocated bytes
   */
  std::size_t get_peak_total_allocated_bytes() const;

 protected:
  /**
   * @brief grows reservation by a `bytes` size
   * @param res current_reservation
   * @param bytes the size of reservation
   */
  bool grow_reservation_by(reserved_arena& res, std::size_t bytes);

  /**
   * @brief grows reservation by a `bytes` size
   * @param res current_reservation
   */
  void shrink_reservation_to_fit(reserved_arena& res);

  bool do_reserve(std::size_t bytes, std::size_t mem_limit);

  std::size_t do_reserve_upto(std::size_t bytes, std::size_t mem_limit);

  /**
   * @brief Allocate memory of the specified size.
   *
   * @param bytes Size in bytes (must be <= block_size_)
   * @param stream CUDA stream (ignored for host memory)
   * @return void* Pointer to allocated memory
   * @throws rmm::logic_error if allocation size exceeds block size
   * @throws rmm::out_of_memory if no free blocks are available and upstream allocation fails
   */
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;

  /**
   * @brief Deallocate memory.
   *
   * @param ptr Pointer to deallocate
   * @param bytes Size in bytes (must be <= block_size_)
   * @param stream CUDA stream (ignored for host memory)
   */
  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override;

  /**
   * @brief Check if this resource is equal to another.
   *
   * @param other Other resource to compare
   * @return bool True if equal
   */
  [[nodiscard]] bool do_is_equal(
    const rmm::mr::device_memory_resource& other) const noexcept override;

 private:
  /**
   * @brief Expand the pool by allocating more blocks from upstream.
   *
   * Allocates a new chunk of blocks and adds them to the free list.
   */
  void expand_pool();

  /**
   * @brief registers reservation with the memory resource
   * @param reservation reserved bytes that is registered with the memory resource
   */
  void register_reservation(chunked_reserved_area* res);

  /**
   * @brief release reservation and returns the unsed bytes to back to the memory resource
   * @param reservation reserved bytes that is registered with the memory resource
   */
  void release_reservation(chunked_reserved_area* res);

  /**
   * @brief release reservation and returns the unsed bytes to back to the memory resource
   * @param chunks reserved bytes that is registered with the memory resource
   * @param res is the reserved memory bytes use to allocate the chunks from
   */
  void return_allocated_chunks(std::vector<std::byte*> chunks, chunked_reserved_area* res);

  memory_space_id space_id_;
  std::size_t memory_limit_;
  std::size_t memory_capacity_;
  std::size_t block_size_;                        ///< Size of each block
  std::size_t pool_size_;                         ///< Number of blocks in pool
  rmm::mr::device_memory_resource* upstream_mr_;  ///< Upstream memory resource (optional)
  std::vector<void*> allocated_blocks_;           ///< All allocated blocks
  std::vector<void*> free_blocks_;                ///< Currently free blocks
  mutable std::mutex mutex_;
  atomic_bounded_counter<size_t> allocated_bytes_{0};
  atomic_peak_tracker<size_t> peak_allocated_bytes_{0};

  struct allocation_tracker {
    explicit allocation_tracker(std::size_t uid) : uuid(uid) {}

    const std::size_t uuid;
    std::atomic<int64_t> allocated_bytes{0};
  };
  std::unordered_map<chunked_reserved_area*, allocation_tracker> active_reservations_;
};

using fixed_multiple_blocks_allocation =
  fixed_size_host_memory_resource::fixed_multiple_blocks_allocation;

}  // namespace memory
}  // namespace sirius