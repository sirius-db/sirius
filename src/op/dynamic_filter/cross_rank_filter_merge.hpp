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

#include <cuda/stream>

#include <memory>
#include <span>

namespace sirius::op {

class sirius_dynamic_filter;

/**
 * @brief Merges one join's membership filters with the same join's filters in other processes,
 * whose builds hold the rest of its keys.
 *
 * A publication session calls exactly one of merge() or leave(), once, and a cancelled one
 * neither. A merge() that throws has still taken part in every key, so no leave() follows it.
 */
class cross_rank_filter_merge {
 public:
  virtual ~cross_rank_filter_merge() = default;

  /**
   * @brief Replaces each of @p membership, one per admitted key (null: none built), with the OR
   * of every process's filter for that key, or with null where they cannot all be merged.
   *
   * May block until every process has taken part, or a timeout. Takes part in every key's merge
   * even when it throws.
   */
  virtual void merge(std::span<std::shared_ptr<sirius_dynamic_filter>> membership,
                     ::cuda::stream_ref stream) = 0;

  /**
   * @brief Takes this process's part in every key's merge with no filter, so the others drop
   * each key at once.
   *
   * Never waits: it can run while a pipeline finishes, or on a thread whose progress the other
   * processes' merges need.
   */
  virtual void leave() noexcept = 0;
};

}  // namespace sirius::op
