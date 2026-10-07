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

#include <cucascade/data/data_repository.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <vector>

namespace sirius::vss {

/**
 * @brief One join side, materialized as an ordered list of batches that every stage agrees on.
 *
 * A neighbour id is a position in the corpus row order, and two operators in two different
 * pipelines resolve it: the fold writes ids, materialize turns them back into rows. Against a
 * pinned table both read the same immutable @c pinned_entry, so they cannot disagree about
 * what position 4,700,013 means. A build phase has no such pre-existing list — each stage
 * would otherwise ask the repository separately, and any difference in the two answers is a
 * silently wrong join rather than a failure.
 *
 * So the plan generator mints one of these per join and hands the same handle to every stage.
 * The first caller to need it takes the snapshot; the order recorded there *is* the corpus row
 * order for the rest of the query. This is the same move the plan generator already makes for
 * dynamic filters, where one channel object is handed to the producing join and the consuming
 * scan rather than letting each rediscover it.
 */
class materialized_side_buffer {
 public:
  /// Take the snapshot if it has not been taken, and return it. The first caller wins; later
  /// callers get that same list even if the repository has changed underneath.
  const std::vector<std::uint64_t>& ensure_snapshot(::cucascade::shared_data_repository& repo)
  {
    std::lock_guard<std::mutex> lg(_mutex);
    if (!_taken) {
      _repo      = &repo;
      _batch_ids = repo.get_batch_ids(/*partition_idx=*/0);
      // Any order is a correct corpus order -- what matters is that every stage uses the same
      // one -- so reversing it must not change a single output row. That makes this switch a
      // deterministic test for the property: whether the batches arrive in the table's order
      // is a race, so a test that waits for them to disagree passes for the wrong reason on
      // most runs. Reversing forces the disagreement that the race only sometimes produces.
      const char* reverse = std::getenv("SIRIUS_VECTOR_JOIN_REVERSE_BUILD_ORDER");
      if (reverse != nullptr && std::string_view{reverse} == "1") {
        std::reverse(_batch_ids.begin(), _batch_ids.end());
      }
      _taken = true;
    }
    return _batch_ids;
  }

  [[nodiscard]] bool has_snapshot() const
  {
    std::lock_guard<std::mutex> lg(_mutex);
    return _taken;
  }

  /// The snapshot. Empty until some stage has called @ref ensure_snapshot.
  [[nodiscard]] std::vector<std::uint64_t> batch_ids() const
  {
    std::lock_guard<std::mutex> lg(_mutex);
    return _batch_ids;
  }

  /// The repository the snapshot was taken from; null before the first snapshot.
  [[nodiscard]] ::cucascade::shared_data_repository* repo() const
  {
    std::lock_guard<std::mutex> lg(_mutex);
    return _repo;
  }

 private:
  mutable std::mutex _mutex;
  ::cucascade::shared_data_repository* _repo{nullptr};
  std::vector<std::uint64_t> _batch_ids;
  bool _taken{false};
};

}  // namespace sirius::vss
