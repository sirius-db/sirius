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

namespace sirius::io::cache {

struct config {
  size_t inflight_io_chunk_budget = 2048;
  double min_prefetching_budget_fraction{0.05};
  double eviction_threshold_fraction{0.6};
  bool dispose_on_idle = false;
};

}  // namespace sirius::io::cache
