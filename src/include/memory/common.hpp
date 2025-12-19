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

#include <rmm/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <atomic>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <system_error>
#include <utility>

namespace cucascade {

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

namespace memory {
/**
 * Memory tier enumeration representing different types of memory storage.
 * Ordered roughly by performance (fastest to slowest access).
 */
enum class Tier : int32_t {
  GPU,   // GPU device memory (fastest but limited)
  HOST,  // Host system memory (fast, larger capacity)
  DISK,  // Disk/storage memory (slowest but largest capacity)
  SIZE   // Value = size of the enum, allows code to be more dynamic
};

/**
 * Memory space id, comprised of device id, and tier
 *
 */
class memory_space_id {
 public:
  Tier tier;
  int32_t device_id;

  explicit memory_space_id(Tier t, int32_t d_id) : tier(t), device_id(d_id) {}

  auto operator<=>(const memory_space_id&) const noexcept = default;

  std::size_t uuid() const noexcept
  {
    std::size_t key = 0;
    std::memcpy(&key, this, sizeof(key));
    return key;
  }
};

enum class MemoryError { SUCCESS, ALLOCATION_FAILED, LIMIT_EXCEEDED, POOL_EXHAUSTED, SIZE };

struct memory_error_category : std::error_category {
  const char* name() const noexcept final;

  std::string message(int ev) const final;
};

const memory_error_category& memory_category();

inline std::error_code make_error_code(MemoryError e);

struct cucascade_out_of_memory : public rmm::out_of_memory {
  explicit cucascade_out_of_memory(std::string_view message,
                                   std::size_t requested_bytes,
                                   std::size_t global_usage);

  const std::size_t requested_bytes;
  const std::size_t global_usage;
};

template <std::integral T>
struct atomic_peak_tracker {
  explicit atomic_peak_tracker(T initial_peak = 0) noexcept : peak_(initial_peak) {}

  void reset(T new_peak = 0) noexcept { peak_.store(new_peak); }

  [[nodiscard]] T peak() const noexcept { return peak_.load(); }

  void update_peak(T current_value)
  {
    auto peak_value = peak_.load();
    while (current_value > peak_value && !peak_.compare_exchange_weak(peak_value, current_value)) {}
  }

 private:
  std::atomic<T> peak_{0};
};

template <std::integral T>
struct atomic_bounded_counter {
  explicit atomic_bounded_counter(T initial = 0) : value_(initial) {}

  std::atomic<T>& native_handle() noexcept { return value_; }

  std::atomic<T> const& native_handle() const noexcept { return value_; }

  std::atomic<T>& operator*() noexcept { return native_handle(); }

  std::atomic<T> const& operator*() const noexcept { return native_handle(); }

  std::atomic<T>* operator->() noexcept { return &value_; }

  std::atomic<T> const* operator->() const noexcept { return &value_; }

  [[nodiscard]] T value(std::memory_order order = std::memory_order_seq_cst) const noexcept
  {
    return value_.load(order);
  }

  [[nodiscard]] T load(std::memory_order order = std::memory_order_seq_cst) const noexcept
  {
    return value_.load(order);
  }

  /// \@brief return updated value
  T add(T diff, std::memory_order order = std::memory_order_seq_cst) noexcept
  {
    return value_.fetch_add(diff, order) + diff;
  }

  [[nodiscard]] std::pair<bool, T> try_add(T diff, T upper_bound)
  {
    auto current = value_.load();
    while (true) {
      if (upper_bound < current || (upper_bound - current) < diff) { return {false, current}; }
      if (value_.compare_exchange_weak(current, current + diff)) { return {true, current + diff}; }
    }
  }

  [[nodiscard]] T add_bounded(T& diff, T upper_bound)
  {
    auto current = value_.load();
    while (current < upper_bound) {
      T space_left = upper_bound - current;
      diff         = std::min(diff, space_left);
      if (value_.compare_exchange_weak(current, current + diff)) { return current + diff; }
    }
    diff = 0;
    return current;
  }

  T sub(T diff, std::memory_order order = std::memory_order_seq_cst) noexcept
  {
    return value_.fetch_sub(diff, order) - diff;
  }

  [[nodiscard]] std::pair<bool, T> try_sub(T diff, T lower_bound)
  {
    auto current = value_.load();
    while (true) {
      if (current < lower_bound || (current - lower_bound) < diff) { return {false, current}; }
      if (value_.compare_exchange_weak(current, current - diff)) { return {true, current - diff}; }
    }
  }

  [[nodiscard]] T sub_bounded(T& diff, T lower_bound)
  {
    auto current = value_.load();
    while (current > lower_bound) {
      T space_above = current - lower_bound;
      diff          = std::min(diff, space_above);

      if (value_.compare_exchange_weak(current, current - diff)) { return current - diff; }
    }
    diff = 0;
    return current;
  }

 private:
  std::atomic<T> value_{0};
};

using DeviceMemoryResourceFactoryFn =
  std::function<std::unique_ptr<rmm::mr::device_memory_resource>(
    int device_id, std::size_t limit, std::size_t capacity)>;

std::unique_ptr<rmm::mr::device_memory_resource> make_default_gpu_memory_resource(
  int device_id, std::size_t limit, std::size_t capacity);

std::unique_ptr<rmm::mr::device_memory_resource> make_default_host_memory_resource(
  int device_id, std::size_t limit, std::size_t capacity);

DeviceMemoryResourceFactoryFn make_default_allocator_for_tier(Tier tier);

}  // namespace memory
}  // namespace cucascade

// Specialization for std::hash to enable use of std::pair<Tier, size_t> as key
namespace std {
template <>
struct hash<cucascade::memory::memory_space_id> {
  size_t operator()(const cucascade::memory::memory_space_id& p) const;
};

template <>
struct is_error_code_enum<cucascade::memory::MemoryError> : true_type {};

}  // namespace std
