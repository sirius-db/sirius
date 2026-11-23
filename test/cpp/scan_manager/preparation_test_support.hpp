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
#include "scan_manager/preparation_ledger.hpp"

#include <atomic>
#include <functional>
#include <limits>

namespace sirius::scan_manager::test {
struct reservation_observation {
  std::atomic<uint64_t> outstanding{0}, requests{0}, allocations{0}, allocated_bytes{0};
  bool fail_allocation = false;
  // Hand out allocate_blocks() requests as quantum-sized blocks with gaps between them.
  bool scatter_blocks = false;
  std::function<void()> before_allocate;
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
  backing_allocation allocate(uint64_t bytes) override { return buffer(bytes, 1); }
  backing_blocks allocate_blocks(uint64_t bytes) override
  {
    if (!seen_->scatter_blocks) return reservation_backing::allocate_blocks(bytes);
    // Every other quantum of one buffer, last first: no block is next to its predecessor. The
    // spare third keeps a caller that wrongly reads them as one range inside the buffer.
    auto spread = buffer(bytes, 3);
    backing_blocks result{{}, quantum_, std::move(spread.owner)};
    for (auto i = charge_size(bytes) / quantum_; i-- > 0;)
      result.blocks.push_back(spread.data + 2 * i * quantum_);
    return result;
  }

 private:
  // One buffer of `spread` times the charge, accounted as the charge alone.
  backing_allocation buffer(uint64_t bytes, uint64_t spread)
  {
    if (seen_->before_allocate) seen_->before_allocate();
    ++seen_->allocations;
    if (seen_->fail_allocation) throw std::bad_alloc();
    auto charge = charge_size(bytes);
    if (seen_->allocated_bytes + charge > capacity_) throw std::bad_alloc();
    auto* p = new std::byte[charge * spread];
    seen_->allocated_bytes += charge;
    return {p, std::shared_ptr<void>(p, [seen = seen_, charge](void* ptr) {
              delete[] static_cast<std::byte*>(ptr);
              seen->allocated_bytes -= charge;
            })};
  }
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
