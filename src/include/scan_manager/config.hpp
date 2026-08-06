
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

#include "creator/config.hpp"
#include "exec/config.hpp"

#include <cucascade/io/config.hpp>

#include <algorithm>
#include <thread>

namespace sirius::scan_manager {

/// Default uring reactor count; counted in the scan-manager sizing budget below.
inline constexpr std::size_t default_uring_n_reactors = 1;

/// Default scan-manager pool size: every core left after the other default pools
/// (downgrade, task_creator, pipeline, uring reactor), never below 4.
[[nodiscard]] inline int default_scan_manager_num_threads()
{
  constexpr int reserved =
    exec::default_downgrade_num_threads + creator::default_task_creator_num_threads +
    exec::default_gpu_pipeline_num_threads + static_cast<int>(default_uring_n_reactors);
  return std::max(4, static_cast<int>(std::thread::hardware_concurrency()) - reserved);
}

/**
 * @brief Configuration for the scan_manager.
 *
 * Derives from @c cucascade::io::io_config, which owns every knob the IO stack
 * itself consumes (@c uring_n_reactors, @c rest_n_reactors,
 * @c enable_prefetch_cache and the @c local / @c rest / @c kvikio / @c cache /
 * @c object_store sub-configs).  Inheriting rather than duplicating them means
 * a field added or removed upstream shows up here automatically, and the
 * registry — whose @c config_type is @c io_config — takes this by base
 * reference with no conversion step.
 *
 * The two members below are sirius-only: cuCascade's IO layer has no notion of
 * either.
 *
 * @c use_sirius_datasource selects the backend for local paths: the uring
 * reactor when true, kvikIO when false.  It is implemented by re-registering
 * the uring backend with a checker that claims nothing, so local files fall
 * through to the kvikio catch-all (see @c sirius_scan_manager's registry
 * setup).  Multi-GPU forces this to true.
 */
struct scan_manager_config : cucascade::io::io_config {
  /// Metadata / dispatch pool owned by the scan manager itself — unrelated to
  /// the reactor threads configured by the base's *_n_reactors.  Sizing comes from
  /// dev's #1337 budget helper, not a hard-coded count.
  exec::thread_pool_config thread_pool{.num_threads        = default_scan_manager_num_threads(),
                                       .thread_name_prefix = "scan_manager"};

  bool use_sirius_datasource{true};
};

}  // namespace sirius::scan_manager
