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
#include <cstdlib>
#include <exception>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace sirius::op::scan {

/// @brief Worker cap for the per-row-group statistics passes of the metadata
/// prepare walk. Env `SIRIUS_METADATA_WALK_THREADS` overrides (>=1); 1 forces
/// the serial path. Read per call so tests can flip it between walks.
inline std::size_t metadata_walk_threads()
{
  constexpr std::size_t kDefaultWalkThreads = 8;
  if (const char* env = std::getenv("SIRIUS_METADATA_WALK_THREADS")) {
    try {
      if (const auto v = std::stoull(env); v > 0) { return static_cast<std::size_t>(v); }
    } catch (...) { /* fall through to default */
    }
  }
  auto const hw = static_cast<std::size_t>(std::thread::hardware_concurrency());
  return std::max<std::size_t>(1, std::min(kDefaultWalkThreads, hw == 0 ? 1 : hw));
}

/// @brief Run @p body over [0, n) split into contiguous per-worker ranges.
///
/// Determinism contract: workers write only to disjoint, index-addressed slots
/// the caller preallocated, so the output is identical for every worker count
/// (including 1). Chunk boundaries depend only on (n, worker count).
///
/// Small inputs run inline: parallelism only pays off when each worker gets a
/// meaningful slice (row-group statistics reads are ~1-3 us each), so ranges
/// below @p min_per_worker rows-groups-per-worker shed workers first. An
/// explicit `SIRIUS_METADATA_WALK_THREADS` wins over the shedding so tests can
/// force multi-worker runs on tiny tables.
///
/// The first worker exception (by worker index, for determinism) is rethrown
/// on the calling thread after every worker joined.
inline void parallel_over_row_groups(std::size_t n,
                                     const std::function<void(std::size_t, std::size_t)>& body)
{
  if (n == 0) { return; }

  constexpr std::size_t kMinPerWorker = 512;
  bool const env_forced               = std::getenv("SIRIUS_METADATA_WALK_THREADS") != nullptr;
  std::size_t workers                 = metadata_walk_threads();
  if (!env_forced) { workers = std::min(workers, std::max<std::size_t>(1, n / kMinPerWorker)); }
  workers = std::min(workers, n);

  if (workers <= 1) {
    body(0, n);
    return;
  }

  auto const chunk = (n + workers - 1) / workers;
  std::vector<std::exception_ptr> errors(workers);
  std::vector<std::thread> threads;
  threads.reserve(workers - 1);
  auto run_range = [&](std::size_t w) noexcept {
    auto const begin = w * chunk;
    auto const end   = std::min(begin + chunk, n);
    if (begin >= end) { return; }
    try {
      body(begin, end);
    } catch (...) {
      errors[w] = std::current_exception();
    }
  };
  for (std::size_t w = 1; w < workers; ++w) {
    threads.emplace_back(run_range, w);
  }
  run_range(0);
  for (auto& t : threads) {
    t.join();
  }
  for (auto& e : errors) {
    if (e) { std::rethrow_exception(e); }
  }
}

}  // namespace sirius::op::scan
