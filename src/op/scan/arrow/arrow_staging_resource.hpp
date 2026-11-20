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
#include <cstddef>
#include <memory_resource>
#include <mutex>
namespace sirius::op::scan {
// Fixed, admitted storage. Reuse starts only after every allocation from the
// previous import is retired; release may run on a different host thread.
class arrow_staging_resource final : public std::pmr::memory_resource {
 public:
  arrow_staging_resource(void* buffer, size_t bytes)
    : storage_(buffer, bytes, std::pmr::null_memory_resource())
  {
  }

 private:
  void* do_allocate(size_t bytes, size_t alignment) override
  {
    std::lock_guard lock(mutex_);
    try {
      auto* result = storage_.allocate(bytes, alignment);
      ++outstanding_;
      return result;
    } catch (...) {
      if (!outstanding_) storage_.release();
      throw;
    }
  }
  void do_deallocate(void*, size_t, size_t) override
  {
    std::lock_guard lock(mutex_);
    if (!--outstanding_) storage_.release();
  }
  bool do_is_equal(std::pmr::memory_resource const& other) const noexcept override
  {
    return this == &other;
  }
  std::mutex mutex_;
  std::pmr::monotonic_buffer_resource storage_;
  size_t outstanding_ = 0;
};
}  // namespace sirius::op::scan
