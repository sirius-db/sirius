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

#include <pthread.h>
#include <sched.h>

#include <cerrno>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace sirius::exec {

/**
 * A validated CPU eligibility mask shared by every worker in a thread pool.
 *
 * Validation is intentionally performed on the constructing thread before any
 * workers are created. Linux workers inherit that thread's allowed mask, so a
 * requested CPU outside it could otherwise be silently discarded by the
 * kernel (for example under a cgroup/cpuset constraint).
 */
class validated_cpu_affinity {
 public:
  explicit validated_cpu_affinity(const std::vector<int>& cpu_ids) : enabled_(!cpu_ids.empty())
  {
    CPU_ZERO(&requested_);
    if (!enabled_) { return; }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &allowed) != 0) {
      throw std::system_error(errno, std::generic_category(), "sched_getaffinity failed");
    }

    for (int cpu_id : cpu_ids) {
      if (cpu_id < 0) {
        throw std::invalid_argument("CPU affinity ID " + std::to_string(cpu_id) +
                                    " must be non-negative");
      }
      if (cpu_id >= CPU_SETSIZE) {
        throw std::invalid_argument("CPU affinity ID " + std::to_string(cpu_id) +
                                    " must be less than CPU_SETSIZE (" +
                                    std::to_string(CPU_SETSIZE) + ")");
      }
      if (!CPU_ISSET(cpu_id, &allowed)) {
        throw std::invalid_argument("CPU affinity ID " + std::to_string(cpu_id) +
                                    " is not in the current process allowed CPU mask");
      }
      CPU_SET(cpu_id, &requested_);
    }
  }

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

  /** Apply the validated mask to the calling worker and verify the kernel readback. */
  void apply_to_current_thread() const
  {
    if (!enabled_) { return; }

    int const set_result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &requested_);
    if (set_result != 0) {
      throw std::system_error(set_result, std::generic_category(), "pthread_setaffinity_np failed");
    }

    cpu_set_t actual;
    CPU_ZERO(&actual);
    int const get_result = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &actual);
    if (get_result != 0) {
      throw std::system_error(get_result, std::generic_category(), "pthread_getaffinity_np failed");
    }

    for (int cpu_id = 0; cpu_id < CPU_SETSIZE; ++cpu_id) {
      if (CPU_ISSET(cpu_id, &requested_) != CPU_ISSET(cpu_id, &actual)) {
        throw std::runtime_error("CPU affinity readback does not match the requested mask");
      }
    }
  }

 private:
  cpu_set_t requested_{};
  bool enabled_{false};
};

/** YAML-reader validator backed by the same checks used by the pool constructors. */
struct valid_cpu_affinity {
  bool operator()(const std::vector<int>& cpu_ids) const
  {
    (void)validated_cpu_affinity(cpu_ids);
    return true;
  }
};

}  // namespace sirius::exec
