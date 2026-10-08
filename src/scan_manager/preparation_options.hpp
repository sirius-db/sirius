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
#include <limits>
#include <optional>
#include <stdexcept>

namespace sirius::scan_manager {
// Intentionally unset until performance measurements establish a production wait limit.
inline constexpr std::optional<std::chrono::milliseconds> k_underfilled_batch_residence =
  std::nullopt;
inline constexpr std::chrono::milliseconds k_interrupt_check_interval{10};
struct preparation_options {
  size_t max_inflight_jobs, max_active_units, max_pending_results, max_control_work, drain_quantum;
  std::optional<std::chrono::milliseconds> underfilled_batch_residence;
  std::chrono::milliseconds interrupt_check_interval = k_interrupt_check_interval;
};
// C++ construction seam only. No SQL, YAML or environment configuration keys.
struct preparation_config {
  std::optional<size_t> max_inflight_jobs, max_active_units, max_pending_results, max_control_work,
    drain_quantum;
  std::optional<std::chrono::milliseconds> underfilled_batch_residence =
    k_underfilled_batch_residence;
  std::chrono::milliseconds interrupt_check_interval = k_interrupt_check_interval;
  preparation_options resolve(int workers) const
  {
    if (workers <= 0) throw std::invalid_argument("preparation needs positive scan worker count");
    auto jobs = static_cast<size_t>(workers);
    if (jobs > std::numeric_limits<size_t>::max() / 2)
      throw std::invalid_argument("preparation worker count overflow");
    preparation_options options{max_inflight_jobs.value_or(jobs),
                                max_active_units.value_or(2 * jobs),
                                max_pending_results.value_or(2 * jobs),
                                max_control_work.value_or(jobs),
                                drain_quantum.value_or(jobs),
                                underfilled_batch_residence,
                                interrupt_check_interval};
    if (interrupt_check_interval.count() <= 0)
      throw std::invalid_argument("interrupt check interval must be positive");
    if (!options.max_inflight_jobs || !options.max_active_units || !options.max_pending_results ||
        !options.max_control_work || !options.drain_quantum ||
        (options.underfilled_batch_residence && options.underfilled_batch_residence->count() <= 0))
      throw std::invalid_argument("preparation limits and configured residence must be positive");
    return options;
  }
};
}  // namespace sirius::scan_manager
