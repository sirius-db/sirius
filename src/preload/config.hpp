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

namespace sirius::preload {

/**
 * @brief Configuration of the preload executor (`sirius.executor.preload`).
 *
 * The preload executor moves queued tasks' inputs that are off the GPU (spilled while queued, or
 * queued already spilled) back to a GPU (HOST->GPU, DISK->GPU) or, failing that, from DISK to
 * HOST, before the task runs. It is
 * advisory: a refused or failed promotion costs performance only.
 *
 * Every value is a fraction of the target memory space's capacity. Each ceiling is clamped per
 * space to that space's @c downgrade_stop_fraction, so preload alone never starts a downgrade.
 * The defaults (both budgets 0) leave the executor off.
 */
struct preload_executor_config {
  /// Promoted bytes per GPU whose task has not been popped yet. 0 = no promotion to a GPU.
  double device_budget{0.0};
  /// Promote to a GPU only while its consumed memory plus the promotion stays at or below this.
  double device_ceiling{0.5};
  /// Staged bytes per HOST space whose task has not been popped yet. 0 = no DISK->HOST staging.
  double host_budget{0.0};
  /// Stage into a HOST space only while its consumed memory plus the staging stays at or below
  /// this.
  double host_ceiling{0.7};
};

/// Whether the preload executor has anything to do: a device or a host budget.
[[nodiscard]] constexpr bool preload_enabled(preload_executor_config const& c) noexcept
{
  return c.device_budget > 0.0 || c.host_budget > 0.0;
}

}  // namespace sirius::preload
