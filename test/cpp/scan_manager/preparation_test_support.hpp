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
#include "scan_manager/preparation.hpp"
#include "scan_manager/preparation_ledger.hpp"

#include <atomic>
#include <chrono>
#include <limits>
#include <thread>

namespace sirius::scan_manager::test {
struct cleanup_observation {
  explicit cleanup_observation(std::shared_ptr<preparation_gate> gate) : gate(std::move(gate)) {}
  std::shared_ptr<preparation_gate> gate;
  std::atomic<size_t> releases{0}, outside{0}, errors{0}, callbacks_seen{0};
  std::function<void()> after_probe;
  void release() noexcept
  {
    try {
      // Never attempt recursive locking on the destruction thread.
      std::thread probe([&] {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{200};
        do {
          std::unique_lock lock(gate->mutex, std::try_to_lock);
          if (lock.owns_lock()) {
            ++outside;
            callbacks_seen = gate->callbacks;
            return;
          }
          std::this_thread::yield();
        } while (std::chrono::steady_clock::now() < deadline);
      });
      probe.join();
      if (after_probe) after_probe();
    } catch (...) {
      ++errors;
    }
    ++releases;
  }
};
struct reservation_observation {
  std::atomic<uint64_t> outstanding{0}, requests{0}, allocations{0}, allocated_bytes{0};
  bool fail_allocation = false;
};
class fake_backing final : public reservation_backing {
 public:
  fake_backing(uint64_t capacity, uint64_t quantum, std::shared_ptr<reservation_observation> seen)
    : capacity_(capacity), quantum_(quantum), seen_(std::move(seen))
  {
    ++seen_->outstanding;
  }
  ~fake_backing() override { --seen_->outstanding; }
  uint64_t charge_size(uint64_t n) const override
  {
    if (n > std::numeric_limits<uint64_t>::max() - (quantum_ - 1))
      throw std::overflow_error("rounding");
    return ((n + quantum_ - 1) / quantum_) * quantum_;
  }
  backing_allocation allocate(uint64_t bytes) override
  {
    ++seen_->allocations;
    if (seen_->fail_allocation) throw std::bad_alloc();
    auto charge = charge_size(bytes);
    if (seen_->allocated_bytes + charge > capacity_) throw std::bad_alloc();
    auto* p = new std::byte[charge];
    seen_->allocated_bytes += charge;
    return {p, std::shared_ptr<void>(p, [seen = seen_, charge](void* ptr) {
              delete[] static_cast<std::byte*>(ptr);
              seen->allocated_bytes -= charge;
            })};
  }

 private:
  uint64_t capacity_, quantum_;
  std::shared_ptr<reservation_observation> seen_;
};
struct test_reservation_provider final : reservation_provider {
  uint64_t grant_bytes                          = 64;
  bool return_null                              = false;
  uint64_t quantum                              = 1;
  std::shared_ptr<reservation_observation> seen = std::make_shared<reservation_observation>();
  uint64_t allocation_granularity(memory_space_id) const override { return quantum; }
  uint64_t outstanding() const { return seen->outstanding; }
  std::optional<reservation_grant> request(memory_space_id space, uint64_t) override
  {
    ++seen->requests;
    if (return_null) return std::nullopt;
    return reservation_grant{
      grant_bytes, space, std::make_shared<fake_backing>(grant_bytes, quantum, seen)};
  }
};
}  // namespace sirius::scan_manager::test
