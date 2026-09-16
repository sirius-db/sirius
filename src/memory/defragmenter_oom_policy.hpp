
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

#include <cucascade/memory/oom_handling_policy.hpp>

namespace sirius {
namespace memory {

/**
 * @brief OOM policy that attempts to recover from fragmentation before giving up.
 *
 * When an allocation fails, this policy checks whether the CUDA memory pool is
 * fragmented (i.e., reserved memory significantly exceeds used memory). If so,
 * it trims the pool to release fragmented free blocks back to the driver, then
 * retries the allocation. If the retry also fails, the original exception is
 * rethrown.
 */
struct defragmenter_oom_policy final : public cucascade::memory::oom_handling_policy {
  std::string get_policy_name() const noexcept override;

 protected:
  void* do_handle_oom(std::size_t bytes,
                      rmm::cuda_stream_view stream,
                      std::exception_ptr eptr,
                      RetryFunc retry_function) override;
};

}  // namespace memory
}  // namespace sirius
