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

/// Upper bound of the default REST reactor count, and the IRQ CPU count from which the egress
/// NIC no longer sizes it.
inline constexpr std::size_t rest_reactors_max = 16;
/// Lower bound of the default REST reactor count with kernel TLS loaded (the kernel encrypts and
/// decrypts, so each reactor thread moves more bytes).
inline constexpr std::size_t rest_reactors_min_kernel_tls = 4;
/// Lower bound of the default REST reactor count without kernel TLS (user-space TLS costs the
/// reactor threads CPU per byte, so more of them are needed).
inline constexpr std::size_t rest_reactors_min_user_tls = 8;

/// T is logical CPU hardware threads; G counts configured GPU memory spaces.
/// GPU pools split T/(2G) as 75% pipeline, 25% downgrade, then round down and
/// clamp independently. Unknown CPU count / no configured GPUs use one.
/// Task creator and scan manager use fixed defaults, independent of T and G.
/// uring reactors are 2 below 32 hardware threads and 4 otherwise.
///
/// REST reactors are clamp(N, lo, 16), lo = 4 with @p kernel_tls and 8 without.  N is
/// @p nic_irq_cpus when that is in (0, 16), otherwise T/6.  The caller passes the egress NIC's
/// IRQ CPU count only when the pool is sized from the NIC (kernel TLS loaded and the queue IRQs
/// steered to a subset of the NIC's NUMA node cores, see @c rest_reactors_sized_from_nic), and 0
/// otherwise.
[[nodiscard]] constexpr hardware_thread_defaults derive_thread_defaults(
  unsigned int cpu_threads,
  std::size_t gpu_count,
  bool kernel_tls          = false,
  std::size_t nic_irq_cpus = 0) noexcept
{
  std::uint64_t const t = std::max(1U, cpu_threads);
  // More GPUs than CPU threads already puts both GPU pools below their minima.
  // Bounding G by T also keeps 8*G and 3*T safe in 64-bit arithmetic.
  auto const g       = std::clamp<std::uint64_t>(gpu_count, 1, t);
  auto const bounded = [](std::uint64_t n, std::uint64_t lo, std::uint64_t hi) {
    return static_cast<int>(std::clamp(n, lo, hi));
  };
  auto const rest_base =
    nic_irq_cpus > 0 && nic_irq_cpus < rest_reactors_max ? static_cast<std::uint64_t>(nic_irq_cpus) : t / 6;
  auto const rest_min = kernel_tls ? rest_reactors_min_kernel_tls : rest_reactors_min_user_tls;
  return {.pipeline       = bounded(3 * t / (8 * g), 4, 8),
          .task_creator   = 4,
          .downgrade      = bounded(t / (8 * g), 1, 4),
          .scan_manager   = 18,
          .rest_reactors  = static_cast<std::size_t>(
            std::clamp<std::uint64_t>(rest_base, rest_min, rest_reactors_max)),
          .uring_reactors = t >= 32 ? 4U : 2U};
}

}  // namespace sirius::exec
