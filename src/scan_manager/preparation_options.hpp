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
#include <chrono>
#include <cstddef>
#include <optional>
#include <stdexcept>

namespace sirius::scan_manager {
inline constexpr std::chrono::milliseconds k_interrupt_check_interval{10};
struct partial_batch_options {
  size_t min_fill_percent = 50;
  std::chrono::milliseconds min_residence{50}, max_residence{2000};
  void validate() const
  {
    if (!min_fill_percent || min_fill_percent > 100 || min_residence.count() <= 0 ||
        max_residence < min_residence)
      throw std::invalid_argument(
        "partial_batch needs fill in [1, 100] and 0 < min_residence <= max_residence");
  }
};
struct preparation_options {
  size_t max_inflight_metadata_jobs, lookahead_batches, max_control_work, drain_quantum;
  std::chrono::milliseconds interrupt_check_interval = k_interrupt_check_interval;
  size_t max_inflight_dv_jobs                        = 0;
  std::optional<partial_batch_options> partial_batch;
};
// YAML and C++ overrides; zero job limits are unlimited.
struct preparation_config {
  std::optional<size_t> max_inflight_metadata_jobs, max_inflight_dv_jobs, lookahead_batches,
    max_control_work, drain_quantum;
  std::chrono::milliseconds interrupt_check_interval = k_interrupt_check_interval;
  std::optional<partial_batch_options> partial_batch = partial_batch_options{};
  preparation_options resolve(int workers) const
  {
    if (workers <= 0) throw std::invalid_argument("preparation needs positive scan worker count");
    auto jobs = static_cast<size_t>(workers);
    preparation_options options{max_inflight_metadata_jobs.value_or(4096),
                                lookahead_batches.value_or(2),
                                max_control_work.value_or(jobs),
                                drain_quantum.value_or(jobs),
                                interrupt_check_interval,
                                max_inflight_dv_jobs.value_or(0),
                                partial_batch};
    if (interrupt_check_interval.count() <= 0)
      throw std::invalid_argument("interrupt check interval must be positive");
    if (!options.lookahead_batches || !options.max_control_work || !options.drain_quantum)
      throw std::invalid_argument("preparation limits must be positive");
    if (partial_batch) partial_batch->validate();
    return options;
  }
};
}  // namespace sirius::scan_manager
