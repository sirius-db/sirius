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

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace sirius::exec {

/// Per-pool defaults; independently capped, not a total CPU budget.
struct hardware_thread_defaults {
  int pipeline;
  int task_creator;
  int downgrade;
  int scan_manager;
  std::size_t rest_reactors;
  std::size_t uring_reactors;
};

/// T is logical CPU hardware threads; G counts configured GPU memory spaces.
/// GPU pools split T/(2G) as 75% pipeline, 25% downgrade, then round down and
/// clamp independently. Unknown CPU count / no configured GPUs use one.
/// Task creator and scan manager use fixed defaults, independent of T and G.
/// uring reactors are 2 below 32 hardware threads and 4 otherwise. REST reactors
/// are clamp(T/6, 16, 32), except when @p rest_from_nic: then 0, the request for
/// cuCascade to size the pool from the egress NIC (one reactor per IRQ CPU).
/// The caller sets it when kernel TLS is loaded and the NIC's queue IRQs are
/// steered to a subset of its NUMA node's cores (see @c rest_reactors_sized_from_nic).
[[nodiscard]] constexpr hardware_thread_defaults derive_thread_defaults(
  unsigned int cpu_threads, std::size_t gpu_count, bool rest_from_nic = false) noexcept
{
  std::uint64_t const t = std::max(1U, cpu_threads);
  // More GPUs than CPU threads already puts both GPU pools below their minima.
  // Bounding G by T also keeps 8*G and 3*T safe in 64-bit arithmetic.
  auto const g       = std::clamp<std::uint64_t>(gpu_count, 1, t);
  auto const bounded = [](std::uint64_t n, std::uint64_t lo, std::uint64_t hi) {
    return static_cast<int>(std::clamp(n, lo, hi));
  };
  return {.pipeline       = bounded(3 * t / (8 * g), 4, 8),
          .task_creator   = 4,
          .downgrade      = bounded(t / (8 * g), 1, 4),
          .scan_manager   = 18,
          .rest_reactors  = rest_from_nic
                              ? std::size_t{0}
                              : static_cast<std::size_t>(std::clamp<std::uint64_t>(t / 6, 16, 32)),
          .uring_reactors = t >= 32 ? 4U : 2U};
}

}  // namespace sirius::exec
