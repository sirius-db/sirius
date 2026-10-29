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
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace sirius::scan_manager {
// Shared by the units of one bound scan. File indices and inline batch indices
// remain separate semantic coordinates; preparation credits need unique keys.
class preparation_identity {
 public:
  uint64_t next()
  {
    auto value = next_.load(std::memory_order_relaxed);
    for (;;) {
      if (value == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("Scan preparation identity exhausted");
      if (next_.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
    }
  }

 private:
  std::atomic<uint64_t> next_{0};
};
}  // namespace sirius::scan_manager
