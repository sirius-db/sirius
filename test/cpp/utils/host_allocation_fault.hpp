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

#include <dlfcn.h>

#include <cstddef>

namespace sirius::test {

/**
 * @brief Arms the `operator new` fault helper
 * (`test/cpp/utils/dynamic_filter_host_allocation_fault.cpp`) on the calling thread until
 * destruction.
 *
 * Only `scripts/run_dynamic_filter_host_allocation_fault.py` preloads the helper, for the hidden
 * `[host_allocation_fault]` tests, which must first require `available()`.
 */
class scoped_host_allocation_fault {
 public:
  /** @brief Matches allocations made inside `accumulated_bloom_builder::make_filters`. */
  static constexpr int filter_shell_scope = 1;
  /** @brief Matches every allocation. */
  static constexpr int every_allocation_scope = 2;

  /**
   * @brief Counts allocations in @p scope and fails the @p ordinal-th; zero only counts.
   */
  scoped_host_allocation_fault(int scope, std::size_t ordinal) { api().arm(scope, ordinal); }
  ~scoped_host_allocation_fault() { (void)api().disarm(); }
  scoped_host_allocation_fault(scoped_host_allocation_fault const&)            = delete;
  scoped_host_allocation_fault& operator=(scoped_host_allocation_fault const&) = delete;

  /**
   * @brief Whether the helper is loaded into this process.
   */
  [[nodiscard]] static bool available() noexcept
  {
    return api().arm != nullptr && api().disarm != nullptr && api().fired != nullptr;
  }

  /**
   * @brief Disarms early and returns the number of matching allocations.
   */
  [[nodiscard]] std::size_t stop() noexcept { return api().disarm(); }

  /**
   * @brief Whether an allocation was failed.
   */
  [[nodiscard]] bool fired() const noexcept { return api().fired(); }

 private:
  struct entry_points {
    void (*arm)(int, std::size_t) noexcept;
    std::size_t (*disarm)() noexcept;
    bool (*fired)() noexcept;
  };

  static entry_points const& api() noexcept
  {
    static entry_points const loaded{reinterpret_cast<decltype(entry_points::arm)>(
                                       dlsym(RTLD_DEFAULT, "sirius_test_host_fault_arm")),
                                     reinterpret_cast<decltype(entry_points::disarm)>(
                                       dlsym(RTLD_DEFAULT, "sirius_test_host_fault_disarm")),
                                     reinterpret_cast<decltype(entry_points::fired)>(
                                       dlsym(RTLD_DEFAULT, "sirius_test_host_fault_fired"))};
    return loaded;
  }
};

}  // namespace sirius::test
